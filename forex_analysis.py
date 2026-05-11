import os
import json
import requests
import yfinance as yf
import pandas as pd
import pandas_ta as ta
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
    df_4h = ticker.history(period="20d", interval="1h")

    if df.empty:
        return {}

    df.ta.rsi(length=14, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.macd(append=True)

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    daily_change_pct = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

    recent_highs = df["High"].tail(20)
    recent_lows = df["Low"].tail(20)
    prev_week = df.tail(7)

    return {
        "current_price": round(float(latest["Close"]), 5),
        "daily_change_pct": round(float(daily_change_pct), 3),
        "day_high": round(float(latest["High"]), 5),
        "day_low": round(float(latest["Low"]), 5),
        "rsi_14": round(float(latest.get("RSI_14", 0)), 2),
        "ema_20": round(float(latest.get("EMA_20", 0)), 5),
        "ema_50": round(float(latest.get("EMA_50", 0)), 5),
        "ema_200": round(float(latest.get("EMA_200", 0)), 5),
        "macd": round(float(latest.get("MACD_12_26_9", 0)), 6),
        "macd_signal": round(float(latest.get("MACDs_12_26_9", 0)), 6),
        "week_high": round(float(prev_week["High"].max()), 5),
        "week_low": round(float(prev_week["Low"].min()), 5),
        "20d_high": round(float(recent_highs.max()), 5),
        "20d_low": round(float(recent_lows.min()), 5),
        "last_5_closes": [round(float(x), 5) for x in df["Close"].tail(5).tolist()],
        "volume": int(latest.get("Volume", 0)),
    }


def analyze_with_claude(market_data: dict) -> str:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    today = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")

    prompt = f"""You are an expert Forex and Gold market analyst specializing in Smart Money Concepts (SMC), order flow analysis, and liquidity.

Today is: {today}

Here is the real market data collected right now:

{json.dumps(market_data, indent=2, ensure_ascii=False)}

Perform a comprehensive professional morning analysis in PERSIAN (Farsi) language.

Use this EXACT HTML format for Telegram:

🌅 <b>تحلیل صبحگاهی فارکس</b>
📅 {today} | ⏰ ۸ صبح لندن

━━━━━━━━━━━━━━━━━━━━

🇪🇺 <b>EUR/USD</b> — [قیمت فعلی]
📈 تغییر روزانه: [درصد تغییر]%

🔍 روند: [Bullish/Bearish/Ranging + توضیح کوتاه]
📊 ساختار: [HH/HL یا LH/LL + آخرین BOS/ChoCH]
💧 لیکوییدیتی: [Equal Highs/Lows، PDH/PDL، PWH/PWL، FVG]
🧱 اوردر بلاک: [سطح اوردر بلاک کلیدی]
🎯 سطوح کلیدی: R: [مقاومت] | S: [حمایت]
📉 RSI: [مقدار] | EMA20: [مقدار] | MACD: [تفسیر]
⚡ سیگنال: [خرید/فروش/صبر] — [دلیل مختصر با entry zone و invalidation]

[همین ساختار برای GBP/USD با 🇬🇧، XAU/USD با 🥇، DXY با 💵]

━━━━━━━━━━━━━━━━━━━━

🧠 <b>جمع‌بندی کلی:</b>
[۲-۳ جمله: sentiment کلی بازار، تأثیر DXY روی سایر جفت‌ها، رویدادهای مهم روز]

⚠️ <i>این تحلیل صرفاً جهت اطلاع است و توصیه مالی نمی‌باشد.</i>

Important rules:
- Write ALL text in Persian/Farsi except symbol names and numbers
- Use actual numbers from the market data provided
- Identify liquidity zones from the 20-day high/low and weekly high/low data
- If EMA20 > EMA50 > EMA200: bullish trend; reverse = bearish
- RSI > 70: overbought, RSI < 30: oversold
- MACD > signal line: bullish momentum
- Be specific and actionable, not generic"""

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

    print("🤖 Analyzing with Claude...")
    analysis = analyze_with_claude(all_data)

    print("📬 Sending to Telegram...")
    send_telegram(analysis)
    print("✅ Done!")


if __name__ == "__main__":
    main()
