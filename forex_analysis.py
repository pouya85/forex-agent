import os
import json
import requests
import yfinance as yf
import pandas as pd
from anthropic import Anthropic
from datetime import datetime, timezone

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOLS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "XAU/USD": "GC=F",
    "DXY":     "DX-Y.NYB"
}

def get_market_data(ticker_symbol: str) -> dict:
    ticker = yf.Ticker(ticker_symbol)
    df_1d  = ticker.history(period="60d",  interval="1d")
    df_1h  = ticker.history(period="10d",  interval="1h")
    df_15m = ticker.history(period="5d",   interval="15m")

    if df_1d.empty:
        return {}

    latest = df_1d.iloc[-1]
    prev   = df_1d.iloc[-2]
    daily_change_pct = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

    prev_week = df_1d.tail(7)
    prev_day  = df_1d.iloc[-2]

    daily_candles = df_1d.tail(20)[["Open", "High", "Low", "Close"]].round(5).values.tolist()

    hourly_candles = []
    if not df_1h.empty:
        hourly_candles = df_1h.tail(30)[["Open", "High", "Low", "Close"]].round(5).values.tolist()

    m15_candles = []
    if not df_15m.empty:
        m15_candles = df_15m.tail(30)[["Open", "High", "Low", "Close"]].round(5).values.tolist()

    return {
        "current_price": round(float(latest["Close"]), 5),
        "daily_change_pct": round(float(daily_change_pct), 3),
        "day_high": round(float(latest["High"]), 5),
        "day_low": round(float(latest["Low"]), 5),
        "prev_day_high": round(float(prev_day["High"]), 5),
        "prev_day_low": round(float(prev_day["Low"]), 5),
        "prev_day_close": round(float(prev_day["Close"]), 5),
        "week_high": round(float(prev_week["High"].max()), 5),
        "week_low": round(float(prev_week["Low"].min()), 5),
        "20d_high": round(float(df_1d["High"].tail(20).max()), 5),
        "20d_low": round(float(df_1d["Low"].tail(20).min()), 5),
        "daily_candles_last20":  daily_candles,
        "hourly_candles_last30": hourly_candles,
        "m15_candles_last30":    m15_candles,
    }


def analyze_with_claude(market_data: dict) -> str:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    today = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")

    prompt = f"""You are a professional ICT (Inner Circle Trader) and Smart Money Concepts (SMC) trader with deep expertise in institutional order flow, liquidity engineering, and multi-timeframe price delivery.

Today is: {today}

Real market data (includes Daily, 1H, and 15M candles):
{json.dumps(market_data, indent=2, ensure_ascii=False)}

Analyze each instrument using ONLY ICT/SMC methodology across 3 timeframes. Write entirely in PERSIAN (Farsi).

Use this EXACT HTML format for Telegram:

🌅 <b>تحلیل صبحگاهی فارکس — ICT/SMC</b>
📅 {today} | ⏰ ۸ صبح لندن (London Kill Zone)

━━━━━━━━━━━━━━━━━━━━

🇪🇺 <b>EUR/USD</b> — [قیمت فعلی] | تغییر: [درصد]%

📅 <b>روزانه (HTF Bias):</b>
  • ساختار: [HH/HL یا LH/LL | آخرین BOS/ChoCH با سطح دقیق]
  • لیکوییدیتی: [BSL بالای X | SSL زیر Y | PDH/PDL | PWH/PWL]
  • پرمیوم/دیسکانت: [قیمت در کدام ناحیه است نسبت به رنج هفتگی]

⏱ <b>یک‌ساعته (Execution TF):</b>
  • ساختار: [جهت ساختار 1H | BOS/ChoCH اخیر]
  • اوردر بلاک: [Bullish/Bearish OB با سطح دقیق | وضعیت قیمت نسبت به آن]
  • FVG: [سطح Fair Value Gap | پر شده یا نشده]

⏰ <b>۱۵ دقیقه (Entry TF):</b>
  • ساختار: [جهت ساختار 15M]
  • اوردر بلاک / FVG: [نزدیک‌ترین سطح برای ورود]
  • وضعیت فعلی: [قیمت در حال رسیدن به OB/FVG یا دور شدن]

⚡ <b>سناریوی ICT:</b> [خرید/فروش/صبر]
  • Entry Zone: [سطح دقیق]
  • Trigger: [BOS در 15M | ChoCH | Displacement + FVG]
  • Target: [نزدیک‌ترین لیکوییدیتی مقابل]
  • Invalidation: [سطح باطل‌شدن]

[همین ساختار برای GBP/USD با 🇬🇧، XAU/USD با 🥇، DXY با 💵]

━━━━━━━━━━━━━━━━━━━━

🧠 <b>جمع‌بندی SMC:</b>
[۲-۳ جمله: Bias کلی HTF، رابطه DXY با جفت‌ارزها، کدام لیکوییدیتی هدف اصلی Smart Money است امروز]

⚠️ <i>این تحلیل صرفاً جهت اطلاع است و توصیه مالی نمی‌باشد.</i>

Multi-Timeframe ICT Rules:
- daily_candles_last20: HTF bias — swing highs/lows, BOS, ChoCH, premium/discount
- hourly_candles_last30: execution — OB, FVG, liquidity sweeps
- m15_candles_last30: entry refinement — OB/FVG entry, BOS confirmation
- Top-down: Daily bias → 1H setup → 15M entry (never trade against Daily bias)
- PDH/PDL = prev_day_high/low (most important intraday liquidity)
- PWH/PWL = week_high/low (major weekly liquidity targets)
- 20d_high/low = institutional liquidity pools (swing targets)
- Premium = above 50% of daily/weekly range → look for sells
- Discount = below 50% → look for buys
- London Kill Zone = 7:00-10:00 AM London (current session — highest probability)
- DO NOT mention RSI, MACD, EMA, or any retail indicators
- Give exact price levels, not vague ranges"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    max_len = 4000

    def split_text(t, size):
        chunks = []
        while len(t.encode("utf-8")) > size:
            cut = size
            encoded = t.encode("utf-8")[:cut]
            chunk = encoded.decode("utf-8", errors="ignore")
            chunks.append(chunk)
            t = t[len(chunk):]
        if t:
            chunks.append(t)
        return chunks

    chunks = split_text(text, max_len) if len(text.encode("utf-8")) > max_len else [text]
    for chunk in chunks:
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "parse_mode": "HTML"
        })


def main():
    print("📊 Fetching market data...")
    all_data = {}
    for name, symbol in SYMBOLS.items():
        print(f"  → {name} ({symbol})")
        data = get_market_data(symbol)
        if data:
            all_data[name] = data
        else:
            print(f"  ⚠️ No data for {name}")

    if not all_data:
        send_telegram("⚠️ خطا در دریافت داده‌های بازار. لطفاً بعداً بررسی کنید.")
        return

    print("🤖 Analyzing with Claude (ICT/SMC)...")
    analysis = analyze_with_claude(all_data)

    print("📬 Sending to Telegram...")
    send_telegram(analysis)
    print("✅ Done!")


if __name__ == "__main__":
    main()
