"""
Alertes Telegram pour 2 indicateurs TradingView portés en Python :
  1. COMBO : EMA 9/21 + VWAP + RSI + Volume (SL/TP en ATR)
  2. ZIG PRO v7 : EMA 50/200 + RSI + ADX/DI + supports/résistances (SL/TP en R)

Seules les bougies CLÔTURÉES sont analysées (comme alert.freq_once_per_bar_close).
Les dernières bougies sont revérifiées pour rattraper un retard de GitHub,
et last_alert.txt évite d'envoyer deux fois le même signal.
"""
import os
import json
import math
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ============================ CONFIG ============================
# (actif Yahoo, timeframe, nb de bougies clôturées récentes revérifiées)
COMBO_WATCH = [
    ("BTC-USD", "15m", 2),
    ("GC=F", "5m", 3),
    ("GC=F", "15m", 2),
]
ZIG_WATCH = [
    ("BTC-USD", "5m", 4),
    ("BTC-USD", "15m", 3),
    ("GC=F", "5m", 4),
    ("GC=F", "15m", 3),
]

PERIOD = "1mo"  # historique téléchargé (sert à chauffer EMA200 et l'état du trade ZIG)
MINTICK = 0.01

TG_TOKEN = os.environ.get("TG_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "")
STATE_FILE = "last_alert.txt"

INTERVAL_MIN = {"1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60, "90m": 90}

# ---- COMBO ----
EMA_FAST, EMA_SLOW = 9, 21
RSI_LEN, ATR_LEN, VOL_MA = 14, 14, 20
RSI_LONG = (50, 70)
RSI_SHORT = (30, 50)
MULTS_BTC = (1.5, 1.0, 2.0, 3.5)   # SL, TP1, TP2, TP3 (x ATR)
MULTS_GOLD = (1.0, 1.0, 2.0, 3.0)

# ---- ZIG PRO v7 (mêmes valeurs par défaut que l'indicateur) ----
Z_EMA_FAST, Z_EMA_SLOW = 50, 200
Z_RSI_LEN = 14
Z_RSI_BUY = (52, 70)       # >= 52 et < 70
Z_RSI_SELL = (30, 48)      # > 30 et <= 48
Z_USE_ADX, Z_ADX_LEN, Z_ADX_MIN = True, 14, 18
Z_USE_SLOPE, Z_USE_DI, Z_USE_CANDLE = True, True, False
Z_ATR_LEN, Z_SL_MULT = 14, 1.5
Z_SL_METHOD = "ATR"        # "ATR" ou "Swing"
Z_SWING_LEN = 5
Z_RR1, Z_RR2, Z_RR3 = 1.0, 2.0, 3.0
Z_COOLDOWN = 3
Z_PIVOT_LEN = 5
Z_ZONE_ATR, Z_NEAR_ATR = 0.20, 0.50


# ============================ OUTILS ============================
def rma(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def true_range(df):
    pc = df["Close"].shift()
    return pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"] - pc).abs(),
    ], axis=1).max(axis=1)


def is_btc(symbol):
    return "BTC" in symbol


def fmt_ts(ts):
    local = ts - pd.Timedelta(hours=4)  # Guadeloupe = UTC-4, sans heure d'été
    return f"{ts.strftime('%H:%M')} UTC ({local.strftime('%H:%M')} Guadeloupe)"


def get_data(symbol, interval, cache):
    """Télécharge une fois par (actif, timeframe), en UTC, bougies clôturées uniquement."""
    key = (symbol, interval)
    if key not in cache:
        df = yf.download(symbol, period=PERIOD, interval=interval,
                         progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if df.empty:
            raise ValueError(f"Yahoo n'a renvoyé aucune donnée pour {symbol} {interval}")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("UTC")
        delta = pd.Timedelta(minutes=INTERVAL_MIN[interval])
        df = df[df.index + delta <= pd.Timestamp.now(tz="UTC")]  # bougie en cours exclue
        cache[key] = df
    return cache[key].copy()


# ============================ INDICATEUR 1 : COMBO ============================
def session_day(index, symbol):
    """Jour de session du VWAP : 00:00 UTC pour le BTC, 17:00 New York pour l'or."""
    if is_btc(symbol):
        return index.date
    ny = index.tz_convert("America/New_York")
    return (ny - pd.Timedelta(hours=17)).date


def compute_combo(df, symbol):
    c = df["Close"]
    df["emaF"] = c.ewm(span=EMA_FAST, adjust=False).mean()
    df["emaS"] = c.ewm(span=EMA_SLOW, adjust=False).mean()

    day = session_day(df.index, symbol)
    pv = (c * df["Volume"]).groupby(day).cumsum()
    v = df["Volume"].groupby(day).cumsum()
    df["vwap"] = pv / v

    delta = c.diff()
    gain = rma(delta.clip(lower=0), RSI_LEN)
    loss = rma(-delta.clip(upper=0), RSI_LEN)
    df["rsi"] = 100 - 100 / (1 + gain / loss)
    df["atr"] = rma(true_range(df), ATR_LEN)
    df["volOk"] = df["Volume"] > df["Volume"].rolling(VOL_MA).mean()

    cross_up = (df["emaF"] > df["emaS"]) & (df["emaF"].shift() <= df["emaS"].shift())
    cross_dn = (df["emaF"] < df["emaS"]) & (df["emaF"].shift() >= df["emaS"].shift())
    long_ = cross_up & (c > df["vwap"]) & df["rsi"].between(*RSI_LONG) & df["volOk"]
    short_ = cross_dn & (c < df["vwap"]) & df["rsi"].between(*RSI_SHORT) & df["volOk"]
    df["dir"] = long_.astype(int) - short_.astype(int)
    return df


def msg_combo(symbol, interval, bar, d, ts):
    mults = MULTS_BTC if is_btc(symbol) else MULTS_GOLD
    name = "BTC" if is_btc(symbol) else "GOLD"
    p, a = bar["Close"], bar["atr"]
    sl, t1, t2, t3 = (p - d * a * mults[0], p + d * a * mults[1],
                      p + d * a * mults[2], p + d * a * mults[3])
    return (f"{'🟢 LONG' if d == 1 else '🔴 SHORT'} {name} ({interval}) [COMBO]\n"
            f"Entrée: {p:.2f}\nSL: {sl:.2f}\n"
            f"TP1: {t1:.2f}\nTP2: {t2:.2f}\nTP3: {t3:.2f}\n"
            f"RSI: {bar['rsi']:.1f}\nBougie: {fmt_ts(ts)}")


# ============================ INDICATEUR 2 : ZIG PRO v7 ============================
def pivots(high, low, n):
    """Équivalent de ta.pivothigh/pivotlow(n, n) : la valeur apparaît n bougies après le pivot."""
    hv, lv = high.to_numpy(float), low.to_numpy(float)
    size = len(hv)
    ph = np.full(size, np.nan)
    pl = np.full(size, np.nan)
    for i in range(2 * n, size):
        ctr = i - n
        if hv[ctr] > hv[ctr - n:ctr].max() and hv[ctr] >= hv[ctr + 1:i + 1].max():
            ph[i] = hv[ctr]
        if lv[ctr] < lv[ctr - n:ctr].min() and lv[ctr] <= lv[ctr + 1:i + 1].min():
            pl[i] = lv[ctr]
    return ph, pl


def compute_zig(df):
    cs = df["Close"]
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    n = len(df)

    ema_f = cs.ewm(span=Z_EMA_FAST, adjust=False).mean().to_numpy()
    ema_s = cs.ewm(span=Z_EMA_SLOW, adjust=False).mean().to_numpy()

    delta = cs.diff()
    gain = rma(delta.clip(lower=0), Z_RSI_LEN)
    loss = rma(-delta.clip(upper=0), Z_RSI_LEN)
    rsi = (100 - 100 / (1 + gain / loss)).to_numpy()

    tr = true_range(df)
    atr = rma(tr, Z_ATR_LEN).to_numpy()

    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr_s = rma(tr, Z_ADX_LEN)
    di_p = 100 * rma(plus_dm, Z_ADX_LEN) / tr_s
    di_m = 100 * rma(minus_dm, Z_ADX_LEN) / tr_s
    tot = (di_p + di_m).replace(0, 1)
    adx = (100 * rma((di_p - di_m).abs() / tot, Z_ADX_LEN)).to_numpy()
    di_p, di_m = di_p.to_numpy(), di_m.to_numpy()

    ph, pl = pivots(df["High"], df["Low"], Z_PIVOT_LEN)
    sh, sl_ = pivots(df["High"], df["Low"], Z_SWING_LEN)

    rr1 = Z_RR1
    rr2 = max(Z_RR2, rr1 + 0.1)
    rr3 = max(Z_RR3, rr2 + 0.1)

    nan = math.nan
    isn = math.isnan
    sig = np.zeros(n, dtype=int)
    E, SL, T1, T2, T3 = (np.full(n, np.nan) for _ in range(5))

    last_res = last_sup = sw_hi = sw_lo = nan
    prev_res = prev_sup = nan
    active, direction, last_exit = False, 0, None
    cur_sl = cur_tp3 = nan

    for i in range(n):
        px, a = c[i], atr[i]

        if not isn(ph[i]):
            last_res = ph[i]
        if not isn(pl[i]):
            last_sup = pl[i]
        if not isn(sh[i]):
            sw_hi = sh[i]
        if not isn(sl_[i]):
            sw_lo = sl_[i]

        # cassures (calculées avant l'invalidation des niveaux, comme dans le Pine)
        cross_up = (i > 0 and not isn(last_res) and not isn(prev_res)
                    and px > last_res and c[i - 1] <= prev_res)
        cross_dn = (i > 0 and not isn(last_sup) and not isn(prev_sup)
                    and px < last_sup and c[i - 1] >= prev_sup)
        break_res = isn(ph[i]) and cross_up
        break_sup = isn(pl[i]) and cross_dn

        if not isn(last_res) and px > last_res + a * Z_ZONE_ATR:
            last_res = nan
        if not isn(last_sup) and px < last_sup - a * Z_ZONE_ATR:
            last_sup = nan
        prev_res, prev_sup = last_res, last_sup

        near_sup = not isn(last_sup) and abs(px - last_sup) <= a * Z_NEAR_ATR
        near_res = not isn(last_res) and abs(px - last_res) <= a * Z_NEAR_ATR

        trend_up, trend_dn = ema_f[i] > ema_s[i], ema_f[i] < ema_s[i]
        slope_up = (not Z_USE_SLOPE) or (i > 0 and ema_f[i] > ema_f[i - 1])
        slope_dn = (not Z_USE_SLOPE) or (i > 0 and ema_f[i] < ema_f[i - 1])
        adx_ok = (not Z_USE_ADX) or adx[i] >= Z_ADX_MIN
        di_buy = (not Z_USE_DI) or di_p[i] > di_m[i]
        di_sell = (not Z_USE_DI) or di_m[i] > di_p[i]
        mom_up = Z_RSI_BUY[0] <= rsi[i] < Z_RSI_BUY[1]
        mom_dn = Z_RSI_SELL[0] < rsi[i] <= Z_RSI_SELL[1]
        can_buy = (not Z_USE_CANDLE) or px > o[i]
        can_sell = (not Z_USE_CANDLE) or px < o[i]

        buy_setup = (trend_up and slope_up and mom_up and adx_ok and di_buy and can_buy
                     and (near_sup or break_res))
        sell_setup = (trend_dn and slope_dn and mom_dn and adx_ok and di_sell and can_sell
                      and (near_res or break_sup))

        # trade simulé en cours : SL prioritaire, seul TP3 clôture le trade
        exited = False
        if active:
            hit_sl = l[i] <= cur_sl if direction == 1 else h[i] >= cur_sl
            hit_tp3 = h[i] >= cur_tp3 if direction == 1 else l[i] <= cur_tp3
            if hit_sl or hit_tp3:
                active, exited, last_exit = False, True, i

        can_open = last_exit is None or i - last_exit > Z_COOLDOWN
        if not active and not exited and can_open and not isn(a) and a > 0:
            d = 1 if buy_setup else -1 if sell_setup else 0
            if d != 0:
                atr_stop = px - d * a * Z_SL_MULT
                swing_stop = sw_lo if d == 1 else sw_hi
                valid_swing = (not isn(swing_stop)) and (swing_stop < px if d == 1 else swing_stop > px)
                new_sl = swing_stop if (Z_SL_METHOD == "Swing" and valid_swing) else atr_stop
                risk = abs(px - new_sl)
                if risk > MINTICK:
                    active, direction = True, d
                    cur_sl = new_sl
                    cur_tp3 = px + d * risk * rr3
                    sig[i] = d
                    E[i], SL[i] = px, new_sl
                    T1[i], T2[i], T3[i] = px + d * risk * rr1, px + d * risk * rr2, cur_tp3

    df["dir"] = sig
    df["entry"], df["sl"], df["tp1"], df["tp2"], df["tp3"] = E, SL, T1, T2, T3
    df["rsi"] = rsi
    return df


def msg_zig(symbol, interval, bar, d, ts):
    name = "BTC" if is_btc(symbol) else "GOLD"
    return (f"{'🟢 BUY' if d == 1 else '🔴 SELL'} {name} ({interval}) [ZIG PRO]\n"
            f"Entrée: {bar['entry']:.2f}\nSL: {bar['sl']:.2f}\n"
            f"TP1: {bar['tp1']:.2f}\nTP2: {bar['tp2']:.2f}\nTP3: {bar['tp3']:.2f}\n"
            f"RSI: {bar['rsi']:.1f}\nBougie: {fmt_ts(ts)}")


# ============================ TELEGRAM ============================
def send(msg):
    """Retourne True si le message est parti (ou si Telegram n'est pas configuré)."""
    print(msg)
    if not TG_TOKEN or not TG_CHAT_ID:
        return True
    for _ in range(3):
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                data={"chat_id": TG_CHAT_ID, "text": msg}, timeout=10,
            )
            if r.ok:
                return True
            print("Telegram a répondu:", r.status_code, r.text[:200])
        except Exception as e:
            print("Erreur Telegram:", e)
    return False


# ============================ ÉTAT / EXÉCUTION ============================
def load_state():
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def parse_ts(value):
    if not value:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def scan(df, lookback, state, key, make_msg):
    last = parse_ts(state.get(key))
    sent = False
    for ts, bar in df.iloc[-lookback:].iterrows():  # de la plus ancienne à la plus récente
        d = int(bar["dir"])
        if d == 0:
            continue
        if last is not None and ts <= last:
            continue  # déjà alerté
        if not send(make_msg(bar, d, ts)):
            break  # échec d'envoi : on réessaiera au prochain passage
        state[key] = str(ts)
        last = ts
        sent = True
    return sent


def main():
    if os.environ.get("TEST_MODE") == "1":
        names = [f"COMBO {s} {i}" for s, i, _ in COMBO_WATCH] + [f"ZIG {s} {i}" for s, i, _ in ZIG_WATCH]
        send("✅ Test OK : le bot fonctionne (" + ", ".join(names) + ")")

    state = load_state()
    cache = {}
    changed = False

    for symbol, interval, lookback in COMBO_WATCH:
        try:
            df = compute_combo(get_data(symbol, interval, cache), symbol)
            changed |= scan(df, lookback, state, f"{symbol}_{interval}",
                            lambda b, d, ts, s=symbol, iv=interval: msg_combo(s, iv, b, d, ts))
        except Exception as e:
            print(f"Erreur COMBO {symbol} {interval}:", e)

    for symbol, interval, lookback in ZIG_WATCH:
        try:
            df = compute_zig(get_data(symbol, interval, cache))
            changed |= scan(df, lookback, state, f"ZIG_{symbol}_{interval}",
                            lambda b, d, ts, s=symbol, iv=interval: msg_zig(s, iv, b, d, ts))
        except Exception as e:
            print(f"Erreur ZIG {symbol} {interval}:", e)

    if changed:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)


if __name__ == "__main__":
    main()
