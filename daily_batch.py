cd C:\Users\podor\minervini_bot
New-Item -ItemType Directory -Force -Path .github\workflows | Out-Null

@"
name: Minervini Daily Paper Trading

on:
  schedule:
    # 화~토 오전 06:30 KST (UTC 21:30 Mon-Fri, 미국 정규장 마감 30분 후)
    - cron: '30 21 * * 1-5'
  workflow_dispatch:

permissions:
  contents: write

jobs:
  run-daily-batch:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install numpy pandas yfinance finance-datareader beautifulsoup4 lxml requests tzdata google-genai

      - name: Run Minervini daily batch
        env:
          GEMINI_API_KEY: `${{ secrets.GEMINI_API_KEY }}
        run: python daily_batch.py

      - name: Commit and push updated paper logs
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add paper_logs/
          git diff --cached --quiet || git commit -m "chore: update paper trading logs [skip ci]"
          git push
"@ | Set-Content -Encoding UTF8 .github\workflows\minervini_daily.yml