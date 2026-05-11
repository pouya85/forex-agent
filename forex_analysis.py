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
    df = ticker.history(period="60d", interval="1d")
    df_1h = ticker.history(period="10d", interval="1h")

    if df.empty:
        return {}

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    daily_change_pct = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

    # Last 20 daily candles for structure analysis
    daily_candles = df.tail(20)[["Open", "High", "Low", "Close"]].round(5).values.tolist()

    # Last 20 hourly candles for intraday structure
    hourly_candles = []
    if not df_1h.empty:
        hourly_candles = df_1h.tail(20)[["Open", "High", "Low", "Close"]].round(5).values.tolist()

    prev_week = df.tail(7)
    prev_day = df.iloc[-2]

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
        "20d_high": round(float(df["High"].tail(20).max()), 5),
        "20d_low": round(float(df["Low"].tail(20).min()), 5),
        "daily_candles_last20": daily_candles,
        "hourly_candles_last20": hourly_candles,
    }


def analyze_with_claude(market_data: dict) -> str:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    today = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")

    prompt = f"""You are a professional ICT (Inner Circle Trader) and Smart Money Concepts (SMC) trader with deep expertise in institutional order flow, liquidity engineering, and price delivery.

Today is: {today}

Real market data:
{json.dumps(market_data, indent=2, ensure_ascii=False)}

Analyze each instrument using ONLY ICT/SMC methodology. Write entirely in PERSIAN (Farsi).

Use this EXACT HTML format for Telegram:

🌅 <b>تحلیل صبحگاهی فارکس — ICT/SMC</b>
📅 {today} | ⏰ ۸ صبح لندن (Kill Zone)

━━━━━━━━━━━━━━━━━━━━

🇪🇺 <b>EUR/USD</b> — [قیمت فعلی]
📈 تغییر روزانه: [درصد]%

📊 <b>ساختار بازار:</b> [HH/HL = Bullish | LH/LL = Bearish | آخرین BOS یا ChoCH در چه سطحی بوده]
💧 <b>لیکوییدیتی:</b> [Buy-side liquidity (BSL) بالای چه سطحی | Sell-side liquidity (SSL) زیر چه سطحی | Equal Highs/Lows | PDH/PDL/PWH/PWL]
🧱 <b>اوردر بلاک:</b> [آخرین Bullish OB یا Bearish OB با سطح دقیق | آیا قیمت در حال بازگشت به آن است]
⚖️ <b>FVG / Imbalance:</b> [وجود Fair Value Gap با سطح دقیق | پر شده یا نشده]
🎯 <b>سطوح کلیدی:</b> مقاومت: [سطح] | حمایت: [سطح]
⚡ <b>سناریوی ICT:</b> [خرید/فروش/صبر] — [entry zone دقیق | چه trigger لازم است (BOS، ChoCH، Displacement) | invalidation level]

[همین ساختار برای GBP/USD با 🇬🇧، XAU/USD با 🥇، DXY با 💵]

━━━━━━━━━━━━━━━━━━━━

🧠 <b>جمع‌بندی SMC:</b>
[۲-۳ جمله: جهت کلی Smart Money، رابطه DXY با جفت‌ارزها (Inverse/Direct correlation)، کدام لیکوییدیتی احتمالاً هدف بعدی است]

⚠️ <i>این تحلیل صرفاً جهت اطلاع است و توصیه مالی نمی‌باشد.</i>

ICT/SMC Analysis Rules:
- Use daily_candles_last20 to identify market structure (swing highs/lows, BOS, ChoCH)
- Use hourly_candles_last20 for intraday OB and FVG identification
- PDH = prev_day_high, PDL = prev_day_low (key liquidity levels)
- PWH = week_high, PWL = week_low (premium/discount weekly range)
- 20d_high / 20d_low = major buy-side / sell-side liquidity pools
- Bullish OB: last bearish candle before a strong bullish displacement
- Bearish OB: last bullish candle before a strong bearish displacement
- FVG: three-candle pattern where the wicks don't overlap (imbalance)
- Price in Premium (above 50% of range) = look for sells | Price in Discount (below 50%) = look for buys
- London Kill Zone = 7:00-10:00 AM London time (current session)
- DO NOT mention RSI, MACD, or any retail indicators
- Be specific with exact price levels, not ranges where possible"""

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
