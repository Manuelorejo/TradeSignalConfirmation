# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 21:01:47 2025

@author: Oreoluwa
"""

import streamlit as st
import yfinance as yf
import requests
import datetime
import os

st.header("🌍 Market Context Dashboard")
st.caption("A live macroeconomic overview combining key indices, forex news, volatility, and upcoming events.")

# -----------------------------------------------------
# 1️⃣ MARKET OVERVIEW SECTION
# -----------------------------------------------------
st.subheader("🪙 Market Overview")

def fetch_market_overview():
    symbols = {
        "EUR Index": "EURUSD=X",
        "USD Index (DXY)": "DX-Y.NYB",
        "Gold (XAUUSD)": "GC=F",
        "VIX Volatility Index": "^VIX"
    }
    overview = []
    for name, ticker in symbols.items():
        try:
            data = yf.Ticker(ticker).history(period="5d")
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            change = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
            overview.append({
                "name": name,
                "price": round(latest["Close"], 2),
                "change": round(change, 2)
            })
        except Exception as e:
            overview.append({"name": name, "price": "N/A", "change": 0})
            st.warning(f"⚠️ Could not fetch {name}: {e}")
    return overview

overview_data = fetch_market_overview()
cols = st.columns(len(overview_data))
for i, item in enumerate(overview_data):
    color = "#4CAF50" if item["change"] > 0 else "#F44336"
    cols[i].markdown(
        f"""
        <div style="background-color:#1E1E1E; padding:12px; border-radius:10px; border:1px solid #333;">
            <strong style="color:white;">{item['name']}</strong><br>
            <span style="color:#AAAAAA;">{item['price']}</span><br>
            <span style="color:{color};">{item['change']}%</span>
        </div>
        """, unsafe_allow_html=True
    )

st.markdown("<hr style='border: 0.5px solid #333;'>", unsafe_allow_html=True)

# -----------------------------------------------------
# 2️⃣ RELEVANT FOREX NEWS
# -----------------------------------------------------
st.subheader("📰 Most Relevant Forex News")

NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

def fetch_forex_news(max_results=6):
    query = "forex OR currency OR interest rates OR central bank OR inflation"
    base_url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": NEWSDATA_API_KEY,
        "q": query,
        "language": "en",
        "category": "business",
        "size": max_results
    }
    try:
        r = requests.get(base_url, params=params, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        return data.get("results", [])
    except:
        return []

forex_news = fetch_forex_news()
if forex_news:
    for n in forex_news:
        st.markdown(
            f"""
            <div style="background-color:#1E1E1E; padding:10px; border-radius:10px; margin-bottom:8px; border:1px solid #333;">
                <strong style="color:#6FA8DC;">{n.get('title', 'Untitled')}</strong><br>
                <span style="color:#CCCCCC; font-size:14px;">{n.get('description', '')}</span><br>
                <a href="{n.get('link', '#')}" target="_blank" style="color:#9AD0F5; text-decoration:none;">
                    {n.get('source_id', 'Unknown')} • {n.get('pubDate', '')[:10]}
                </a>
            </div>
            """, unsafe_allow_html=True
        )
else:
    st.caption("⚠️ No recent forex news available.")

st.markdown("<hr style='border: 0.5px solid #333;'>", unsafe_allow_html=True)

# -----------------------------------------------------
# 3️⃣ MARKET VOLATILITY
# -----------------------------------------------------
st.subheader("🌪 Market Volatility Gauge")

def assess_market_volatility(vix_value):
    if vix_value < 15:
        return 25, "Low"
    elif vix_value < 25:
        return 60, "Moderate"
    else:
        return 90, "High"

vix = [o for o in overview_data if "VIX" in o["name"]]
if vix:
    vix_value = vix[0]["price"] if isinstance(vix[0]["price"], (int, float)) else 20
    vol_score, vol_label = assess_market_volatility(vix_value)
    st.progress(vol_score / 100)
    st.caption(f"Current Market Volatility: **{vol_label} ({vix_value})**")
else:
    st.caption("VIX data unavailable.")

st.markdown("<hr style='border: 0.5px solid #333;'>", unsafe_allow_html=True)

# -----------------------------------------------------
# 4️⃣ UPCOMING ECONOMIC EVENTS (LIVE FROM FMP)
# -----------------------------------------------------
st.subheader("📅 Upcoming Economic Events")

FMP_API_KEY = 'XFvZF4eO6lJgCzu7gBtmHLPAAtUjN0SE'

# -----------------------------------------------------
# 4️⃣ UPCOMING ECONOMIC EVENTS (PUBLIC FMP SANDBOX)
# -----------------------------------------------------
st.subheader("📅 Upcoming Economic Events")

def fetch_economic_events_tradingeconomics(max_items=10):
    """Fetch upcoming economic events from TradingEconomics (guest access)."""
    try:
        url = "https://api.tradingeconomics.com/calendar?c=guest:guest"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            st.error(f"❌ TradingEconomics API error ({r.status_code}): {r.text[:200]}")
            return []
        data = r.json()
        events = []
        for e in data[:max_items]:
            events.append({
                "event": e.get("Event", "Unknown Event"),
                "country": e.get("Country", "Global"),
                "date": e.get("Date", ""),
                "importance": e.get("Importance", "N/A"),
                "actual": e.get("Actual", "N/A"),
                "forecast": e.get("Forecast", "N/A"),
                "previous": e.get("Previous", "N/A"),
            })
        return events
    except Exception as e:
        st.error(f"Failed to fetch events: {e}")
        return []

events = fetch_economic_events_tradingeconomics(max_items=10)
if events:
    for e in events:
        # Safely extract values with fallback defaults
        event_name = e.get("event", "Unknown Event")
        country = e.get("country", "Global")
        date = e.get("date", "N/A")
        impact = e.get("impact", "N/A")
        previous = e.get("previous", "N/A")
        estimate = e.get("estimate", e.get("forecast", "N/A"))
        actual = e.get("actual", "N/A")

        st.markdown(
            f"""
            <div style="background-color:#1E1E1E; padding:10px; border-radius:10px; margin-bottom:8px; border:1px solid #333;">
                <strong style="color:#FFC107;">{event_name}</strong> — {country}<br>
                <span style="color:#CCCCCC;">Date: {date}</span><br>
                <span style="color:#AAAAAA;">Previous: {previous} | Estimate: {estimate} | Actual: {actual}</span><br>
                <span style="color:#9AD0F5;">Impact: {impact}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.caption("⚠️ No upcoming economic events available at the moment.")
