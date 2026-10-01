"""
Combo Signal (EMA + VWAP + RSI + Volume) -> alertes Telegram
Installation : pip install yfinance pandas requests
Variables d'environnement : TG_TOKEN et TG_CHAT_ID
"""
import os
import requests
import pandas as pd
import yfinance as yf

# ============ CONFIG ============
SYMBOL   = "BTC-USD"      # "BTC-USD" pour Bitcoin, "GC=F" pour l'or
IS_BTC   = "BTC" in SYMBOL
INTERVAL = "15m"          # 1m, 5m, 15m, 30m, 1h...

EMA_FAST, EMA_SLOW = 9, 21
RSI_LEN, ATR_LEN, VOL_MA = 14, 14, 20
RSI_LONG = (50, 70)
RSI_SHORT = (30, 50)

# (SL, TP1, TP2, TP3) en multiples d'ATR
MULTS = (1.5, 1.0, 2.0, 3.5) if IS_BTC else (1.0, 1.0, 2.0, 3.0)

TG_TOKEN = os.environ.get("TG_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "")


# ============ INDICATEURS ============
def rma(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def compute(df):
    c = df["Close"]
    df["emaF"] = c.ewm(span=EMA_FAST, adjust=False).mean()
    df["emaS"] = c.ewm(span=EMA_SLOW, adjust=False).mean()

    # VWAP remis à zéro chaque jour (comme Pine)
    day = df.index.date
    pv = (c * df["Volume"]).groupby(day).cumsum()
    v = df["Volume"].groupby(day).cumsum()
    df["vwap"] = pv / v

    delta = c.diff()
    gain = rma(delta.clip(lower=0), RSI_LEN)
    loss = rma(-delta.clip(upper=0), RSI_LEN)
    df["rsi"] = 100 - 100 / (1 + gain / loss)

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - c.shift()).abs(),
        (df["Low"] - c.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = rma(tr, ATR_LEN)

    df["volOk"] = df["Volume"] > df["Volume"].rolling(VOL_MA).mean()
    df["crossUp"] = (df["emaF"] > df["emaS"]) & (df["emaF"].shift() <= df["emaS"].shift())
    df["crossDn"] = (df["emaF"] < df["emaS"]) & (df["emaF"].shift() >= df["emaS"].shift())

    df["long"] = (df.crossUp & (c > df.vwap) & df.rsi.between(*RSI_LONG) & df.volOk)
    df["short"] = (df.crossDn & (c < df.vwap) & df.rsi.between(*RSI_SHORT) & df.volOk)
    return df


# ============ TELEGRAM ============
def send(msg):
    print(msg)
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT_ID, "text": msg}, timeout=10,
        )
    except Exception as e:
        print("Erreur Telegram:", e)


def build_message(row, is_long):
    p, a = row.Close, row.atr
    sgn = 1 if is_long else -1
    sl, t1, t2, t3 = (p - sgn * a * MULTS[0], p + sgn * a * MULTS[1],
                      p + sgn * a * MULTS[2], p + sgn * a * MULTS[3])
    return (f"{'🟢 LONG' if is_long else '🔴 SHORT'} {SYMBOL} ({INTERVAL})\n"
            f"Entrée: {p:.2f}\nSL: {sl:.2f}\n"
            f"TP1: {t1:.2f}\nTP2: {t2:.2f}\nTP3: {t3:.2f}\n"
            f"RSI: {row.rsi:.1f}")


# ============ EXÉCUTION UNIQUE (lancée par GitHub Actions) ============
STATE_FILE = "last_alert.txt"


def main():
    if os.environ.get("TEST_MODE") == "1":
        send(f"✅ Test OK : le bot fonctionne ({SYMBOL}, {INTERVAL})")
    df = yf.download(SYMBOL, period="5d", interval=INTERVAL,
                     progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = compute(df.dropna())
    bar = df.iloc[-2]          # dernière bougie CLÔTURÉE
    ts = str(df.index[-2])

    last = open(STATE_FILE).read().strip() if os.path.exists(STATE_FILE) else ""
    if ts != last and (bar["long"] or bar["short"]):
        send(build_message(bar, bool(bar["long"])))
        open(STATE_FILE, "w").write(ts)
    else:
        print("Pas de nouveau signal", ts)


if __name__ == "__main__":
    main()
