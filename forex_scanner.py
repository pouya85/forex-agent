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
}


# ─── ICT Structure Detection ──────────────────────────────────────────────────

def find_swing_points(df, n=3):
    highs, lows = [], []
    for i in range(n, len(df) - n):
        if df["High"].iloc[i] == df["High"].iloc[i - n: i + n + 1].max():
            highs.append((i, round(float(df["High"].iloc[i]), 5)))
        if df["Low"].iloc[i] == df["Low"].iloc[i - n: i + n + 1].min():
            lows.append((i, round(float(df["Low"].iloc[i]), 5)))
    return highs, lows


def find_equal_levels(levels, tolerance=0.0004):
    """Return price levels that appear at least twice (EQH / EQL)."""
    if len(levels) < 2:
        return []
    result = []
    visited = [False] * len(levels)
    for i in range(len(levels)):
        if visited[i]:
            continue
        group = [levels[i]]
        for j in range(i + 1, len(levels)):
            if not visited[j] and abs(levels[j] - levels[i]) / levels[i] <= tolerance:
                group.append(levels[j])
                visited[j] = True
        if len(group) >= 2:
            result.append(round(sum(group) / len(group), 5))
        visited[i] = True
    return result


def detect_sweep(df_15m, key_levels, tolerance=0.00025):
    """
    Returns a list of sweeps detected in the last 4 candles.
    A sweep = wick through a level but close on the other side.
    """
    sweeps = []
    recent = df_15m.tail(4)
    for label, price in key_levels.items():
        for _, c in recent.iterrows():
            # Buy-side sweep: wick above level, close back below
            if c["High"] >= price * (1 - tolerance) and c["Close"] < price:
                sweeps.append({"type": "BSL_sweep", "level": label, "price": price})
            # Sell-side sweep: wick below level, close back above
            elif c["Low"] <= price * (1 + tolerance) and c["Close"] > price:
                sweeps.append({"type": "SSL_sweep", "level": label, "price": price})
    return sweeps


def detect_bos_choch(df, n=3):
    """
    Detect the most recent BOS or ChoCH on the given dataframe.
    Returns dict with type and level, or None.
    """
    highs, lows = find_swing_points(df, n=n)
    if not highs or not lows:
        return None

    current_close = round(float(df["Close"].iloc[-1]), 5)
    last_high = highs[-1][1]
    last_low = lows[-1][1]

    # Determine prior trend from last 2 swing highs/lows
    if len(highs) >= 2 and len(lows) >= 2:
        prior_bullish = highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1]
        prior_bearish = highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1]
    else:
        prior_bullish = prior_bearish = False

    if current_close > last_high:
        structure_type = "ChoCH_Bullish" if prior_bearish else "BOS_Bullish"
        return {"type": structure_type, "level": last_high}
    if current_close < last_low:
        structure_type = "ChoCH_Bearish" if prior_bullish else "BOS_Bearish"
        return {"type": structure_type, "level": last_low}
    return None


def detect_fvg(df, direction, lookback=30):
    """Return the most recent unfilled FVG in the given direction."""
    df = df.tail(lookback).reset_index(drop=True)
    current_price = float(df["Close"].iloc[-1])
    fvgs = []
    for i in range(1, len(df) - 1):
        c1, c3 = df.iloc[i - 1], df.iloc[i + 1]
        if direction == "bullish" and c1["Low"] > c3["High"]:
            top, bot = round(float(c1["Low"]), 5), round(float(c3["High"]), 5)
            # Unfilled = current price hasn't traded back into it
            if current_price > bot:
                fvgs.append({"type": "Bullish_FVG", "top": top, "bottom": bot})
        elif direction == "bearish" and c1["High"] < c3["Low"]:
            top, bot = round(float(c3["Low"]), 5), round(float(c1["High"]), 5)
            if current_price < top:
                fvgs.append({"type": "Bearish_FVG", "top": top, "bottom": bot})
    return fvgs[-1] if fvgs else None


def detect_order_block(df, direction, lookback=30, min_displacement=0.0003):
    """Return the most recent valid OB in the given direction."""
    df = df.tail(lookback).reset_index(drop=True)
    if direction == "bullish":
        for i in range(len(df) - 3, 1, -1):
            c = df.iloc[i]
            if c["Close"] < c["Open"]:  # bearish candle = potential bullish OB
                future_high = df["High"].iloc[i + 1: i + 4].max()
                if (future_high - c["High"]) / c["High"] >= min_displacement:
                    return {
                        "type": "Bullish_OB",
                        "top": round(float(c["High"]), 5),
                        "bottom": round(float(c["Low"]), 5),
                    }
    else:
        for i in range(len(df) - 3, 1, -1):
            c = df.iloc[i]
            if c["Close"] > c["Open"]:  # bullish candle = potential bearish OB
                future_low = df["Low"].iloc[i + 1: i + 4].min()
                if (c["Low"] - future_low) / c["Low"] >= min_displacement:
                    return {
                        "type": "Bearish_OB",
                        "top": round(float(c["High"]), 5),
                        "bottom": round(float(c["Low"]), 5),
                    }
    return None


# ─── Data & Scanning ──────────────────────────────────────────────────────────

def scan_symbol(name, ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    df_1d  = ticker.history(period="20d", interval="1d")
    df_1h  = ticker.history(period="5d",  interval="1h")
    df_15m = ticker.history(period="2d",  interval="15m")

    if df_15m.empty or df_1h.empty or df_1d.empty:
        return None

    current_price = round(float(df_15m["Close"].iloc[-1]), 5)
    pdh = round(float(df_1d.iloc[-2]["High"]), 5)
    pdl = round(float(df_1d.iloc[-2]["Low"]), 5)
    pdc = round(float(df_1d.iloc[-2]["Close"]), 5)
    pwh = round(float(df_1d.tail(5)["High"].max()), 5)
    pwl = round(float(df_1d.tail(5)["Low"].min()), 5)

    # Equal Highs / Lows on 1H
    swing_h_1h, swing_l_1h = find_swing_points(df_1h, n=3)
    eq_highs = find_equal_levels([h[1] for h in swing_h_1h])
    eq_lows  = find_equal_levels([l[1] for l in swing_l_1h])

    key_levels = {"PDH": pdh, "PDL": pdl, "PWH": pwh, "PWL": pwl}
    for i, lvl in enumerate(eq_highs[:2]):
        key_levels[f"EQH_{i+1}"] = round(lvl, 5)
    for i, lvl in enumerate(eq_lows[:2]):
        key_levels[f"EQL_{i+1}"] = round(lvl, 5)

    # ── Setup detection ──
    sweeps = detect_sweep(df_15m, key_levels)
    mss_15m = detect_bos_choch(df_15m, n=2)
    mss_1h  = detect_bos_choch(df_1h,  n=3)

    # Must have at least a sweep OR a structure shift
    if not sweeps and not mss_15m:
        return None

    # Determine probable direction
    bullish_signals = sum([
        any("SSL" in s["type"] for s in sweeps),          # SSL sweep → expect up
        mss_15m is not None and "Bullish" in mss_15m.get("type", ""),
        mss_1h  is not None and "Bullish" in mss_1h.get("type", ""),
    ])
    bearish_signals = sum([
        any("BSL" in s["type"] for s in sweeps),          # BSL sweep → expect down
        mss_15m is not None and "Bearish" in mss_15m.get("type", ""),
        mss_1h  is not None and "Bearish" in mss_1h.get("type", ""),
    ])

    if bullish_signals == bearish_signals:
        return None  # Conflicting signals — skip
    direction = "bullish" if bullish_signals > bearish_signals else "bearish"

    ob  = detect_order_block(df_15m, direction)
    fvg = detect_fvg(df_15m, direction)

    # Must have at least OB or FVG for a valid entry zone
    if not ob and not fvg:
        return None

    return {
        "symbol": name,
        "current_price": current_price,
        "direction": direction,
        "key_levels": key_levels,
        "sweeps": sweeps,
        "mss_15m": mss_15m,
        "mss_1h": mss_1h,
        "order_block": ob,
        "fvg": fvg,
        "1h_candles_last20":  df_1h.tail(20)[["Open","High","Low","Close"]].round(5).values.tolist(),
        "15m_candles_last20": df_15m.tail(20)[["Open","High","Low","Close"]].round(5).values.tolist(),
    }


# ─── Claude Alert ─────────────────────────────────────────────────────────────

def build_alert(setup: dict, session: str) -> str:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")

    prompt = f"""You are a professional ICT trader. A potential high-probability setup has been algorithmically detected. Evaluate it and write a concise Telegram alert IN PERSIAN (Farsi).

Session: {session} Kill Zone | Time: {now}

Setup data:
{json.dumps(setup, indent=2, ensure_ascii=False)}

If this is a genuine ICT/SMC setup (sweep + structure shift + OB/FVG confluence), write a short alert.
If the data looks weak or conflicting, reply with exactly: NO_ALERT

Alert format (HTML for Telegram):
⚡ <b>ستاپ ICT شناسایی شد — {setup['symbol']}</b>
🕐 {now} | {session} Kill Zone

📍 قیمت: [current_price]
🎯 جهت: [خرید/فروش]

🔍 <b>دلیل ستاپ:</b>
• [sweep شناسایی شده]
• [BOS یا ChoCH در 15M/1H]
• [OB یا FVG به عنوان ناحیه ورود]

📊 <b>پلن معامله:</b>
• ورود: [سطح دقیق OB یا FVG]
• هدف: [نزدیک‌ترین لیکوییدیتی مقابل با سطح دقیق]
• Invalidation: [سطح باطل‌شدن]

⚠️ <i>این تحلیل صرفاً جهت اطلاع است.</i>

Rules:
- Be specific with price levels
- Only alert on genuine confluence (sweep + structure + OB/FVG)
- Do NOT mention RSI, MACD, or retail indicators"""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text.strip()


# ─── Telegram ─────────────────────────────────────────────────────────────────

def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    })


# ─── Main ─────────────────────────────────────────────────────────────────────

def get_session():
    hour = datetime.now(timezone.utc).hour
    if 6 <= hour < 10:
        return "London"
    if 11 <= hour < 16:
        return "New York"
    return None


def main():
    session = get_session()
    if not session:
        print("⏸ Not in a Kill Zone — skipping.")
        return

    print(f"🔍 Scanning during {session} Kill Zone...")
    alerts_sent = 0

    for name, symbol in SYMBOLS.items():
        print(f"  → {name}")
        setup = scan_symbol(name, symbol)
        if not setup:
            print(f"     No setup found.")
            continue

        print(f"     ✅ Potential {setup['direction']} setup detected — asking Claude...")
        alert = build_alert(setup, session)

        if alert == "NO_ALERT":
            print(f"     Claude: not strong enough — skipped.")
            continue

        send_telegram(alert)
        print(f"     📬 Alert sent for {name}!")
        alerts_sent += 1

    if alerts_sent == 0:
        print("✅ Scan complete — no quality setups found.")
    else:
        print(f"✅ Done — {alerts_sent} alert(s) sent.")


if __name__ == "__main__":
    main()
