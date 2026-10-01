name: alert
on:
  schedule:
    - cron: "*/5 * * * *"
  workflow_dispatch:
permissions:
  contents: write
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install yfinance pandas requests
      - run: python ./*.py
        env:
          TG_TOKEN: ${{ secrets.TG_TOKEN }}
          TG_CHAT_ID: ${{ secrets.TG_CHAT_ID }}
          TEST_MODE: ${{ github.event_name == 'workflow_dispatch' && '1' || '0' }}
      - name: Sauvegarder l'état
        run: |
          git config user.name "bot"
          git config user.email "bot@users.noreply.github.com"
          git add last_alert.txt || true
          git diff --cached --quiet || (git commit -m "state" && git push)
