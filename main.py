# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 20:59:52 2025

@author: Oreoluwa
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import altair as alt
from groq import Groq
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# -----------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------
st.set_page_config(page_title="📊 AI Market Sentiment Analyzer", layout="wide")
st.title("📈 AI-Powered Market Sentiment Analyzer")
st.caption("Analyze real-world market data, technical indicators, and sentiment — powered by Groq + FinBERT + NewsData.io")

# -----------------------------------------------------
# API KEYS
# -----------------------------------------------------
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)

# -----------------------------------------------------
# LOAD FINBERT PIPELINE
# -----------------------------------------------------
@st.cache_resource
def load_finbert():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

# -----------------------------------------------------
# FETCH MARKET DATA
# -----------------------------------------------------
def fetch_price_history(symbol: str, period="6mo", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    return df.dropna()

def add_technical_indicators(df):
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["RSI_14"] = 100 - (100 / (1 + rs))
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()
    return df

# -----------------------------------------------------
# FETCH LATEST NEWS (MODELED FUNCTION)
# -----------------------------------------------------
def fetch_newsdata(query, max_results=8):
    """
    Fetch recent business news articles for a given query using NewsData.io.
    Reliable, debuggable, and safe for Streamlit display.
    """
    base_url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": NEWSDATA_API_KEY,
        "q": query,
        "language": "en",
        "country": "us",
        "category": "business",
        "size": max_results,
    }

    try:
        st.write(f"🔎 Fetching latest market news for: **{query}**")
        r = requests.get(base_url, params=params, timeout=10)
        st.write(f"🌐 Response Code: {r.status_code}")

        if r.status_code != 200:
            st.error(f"❌ API error ({r.status_code}): {r.text[:200]}")
            return []

        data = r.json()
        articles = data.get("results", [])
        if not articles:
            st.warning(f"⚠️ No news headlines found for **{query}**.")
            return []

        cleaned_articles = []
        for a in articles:
            cleaned_articles.append({
                "title": a.get("title", "").strip(),
                "description": a.get("description", "").strip(),
                "source": a.get("source_id", "Unknown"),
                "date": a.get("pubDate", "")[:10],
                "link": a.get("link", "")
            })

        st.success(f"✅ Retrieved {len(cleaned_articles)} articles for '{query}'")
        return cleaned_articles

    except requests.exceptions.Timeout:
        st.error("⏳ Connection timed out while fetching news.")
        return []

    except Exception as e:
        st.error(f"❌ Failed to connect to NewsData API: {e}")
        return []

# -----------------------------------------------------
# GROQ SUMMARIZATION FOR FINBERT CONTEXT
# -----------------------------------------------------
def summarize_with_groq(title, description):
    if not client:
        return (title + " " + description).strip()
    prompt = f"Summarize this financial headline in one sentence with context:\n\nTitle: {title}\nDescription: {description}"
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=80
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"❌ Groq summarization failed: {e}")
        return (title + " " + description).strip()

# -----------------------------------------------------
# FINBERT SENTIMENT ANALYSIS
# -----------------------------------------------------
def finbert_sentiment_analysis(headlines, symbol=""):
    if not headlines:
        st.warning("⚠️ No headlines fetched.")
        return {"label": "Neutral", "avg_score": 0.0, "details": []}

    finbert = load_finbert()
    results = []
    st.write(f"🧠 Running FinBERT analysis on {len(headlines)} articles...")

    for h in headlines:
        summarized_text = summarize_with_groq(h["title"], h.get("description", ""))
        text = f"{symbol} stock: {summarized_text}"
        if not text.strip():
            continue
        try:
            res = finbert(text[:512])[0]
        except Exception as e:
            st.error(f"❌ FinBERT error: {e}")
            continue
        results.append({
            "headline": h["title"],
            "summary": summarized_text,
            "label": res["label"],
            "score": res["score"],
            "source": h["source"],
            "date": h["date"]
        })

    if not results:
        st.error("🚫 No valid FinBERT results.")
        return {"label": "Neutral", "avg_score": 0.0, "details": []}

    df = pd.DataFrame(results)
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["numeric"] = df["label"].str.lower().map(sentiment_map)
    weighted_sum = (df["numeric"] * df["score"]).sum()
    total_weight = df["score"].sum()
    avg_score = weighted_sum / total_weight if total_weight > 0 else 0
    label = "Bullish" if avg_score > 0.1 else "Bearish" if avg_score < -0.1 else "Neutral"

    return {"label": label, "avg_score": float(avg_score), "details": results}

# -----------------------------------------------------
# GROQ LLM MARKET SUMMARY (WITH FALLBACKS)
# -----------------------------------------------------
def generate_llm_summary(symbol, technicals, sentiment, headlines):
    if not client:
        return "Groq API key not configured."

    raw_details = sentiment.get("details", [])
    if not raw_details:
        st.warning("⚠️ No sentiment details found — using raw headlines instead.")
        raw_details = headlines

    summaries_list = []
    for h in raw_details[:5]:
        summary_text = h.get("summary", "").strip()
        title_text = h.get("title", "").strip()
        source = h.get("source", "Unknown")
        if not summary_text and title_text:
            summary_text = title_text
        if not summary_text:
            continue
        summaries_list.append(f"- {summary_text} ({source})")

    if not summaries_list:
        summaries_list = ["- No valid news summaries available."]

    summaries = "\n".join(summaries_list)

    prompt = f"""
You are a financial analyst AI.
Analyze **{symbol}** using the following context:

Technical Data:
- Close Price: {float(technicals['Close']):.2f}
- SMA50: {float(technicals['SMA_50']):.2f}
- SMA200: {float(technicals['SMA_200']):.2f}
- RSI(14): {float(technicals['RSI_14']):.2f}
- ATR(14): {float(technicals['ATR_14']):.2f}

FinBERT Sentiment:
- Label: {sentiment.get('label', 'Unknown')}
- Weighted Score: {sentiment.get('avg_score', 0.0):.2f}

News Context:
{summaries}

Return exactly two sections:
1. **Market Sentiment** (1 line)
2. **Market Context** (2–3 lines)
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"❌ Groq LLM request failed: {e}")
        return "⚠️ LLM summarization unavailable."

# -----------------------------------------------------
# VISUALIZATION: SENTIMENT BREAKDOWN + DRIVERS
# -----------------------------------------------------
def visualize_sentiment_distribution(sentiment_details):
    """Visualize bullish vs bearish sentiment (normalized to 100%, excluding neutral)."""
    if not sentiment_details:
        st.warning("⚠️ No sentiment details to visualize.")
        return

    df = pd.DataFrame(sentiment_details)
    df["label"] = df["label"].str.capitalize()

    # Count only Positive and Negative
    counts = df["label"].value_counts().reindex(["Positive", "Negative"], fill_value=0)
    total = counts.sum()

    if total == 0:
        st.warning("⚠️ No bullish or bearish sentiment found.")
        return

    # Normalize to 100%
    percentages = (counts / total * 100).round(2)

    chart_df = pd.DataFrame({
        "Sentiment": ["Bullish", "Bearish"],
        "Percentage": [percentages["Positive"], percentages["Negative"]],
        "Color": ["#4CAF50", "#F44336"]
    })

    st.markdown("### 📊 Market Sentiment Breakdown (Bullish vs Bearish)")
    st.write(f"**Bullish:** {percentages['Positive']:.2f}% | **Bearish:** {percentages['Negative']:.2f}%")

    chart = (
        alt.Chart(chart_df)
        .mark_bar(size=40)
        .encode(
            x=alt.X("Percentage:Q", title="Percentage (%)"),
            y=alt.Y("Sentiment:N", sort="-x"),
            color=alt.Color("Color:N", scale=None),
            tooltip=["Sentiment", "Percentage"]
        )
        .properties(height=150)
    )

    st.altair_chart(chart, use_container_width=True)


def show_key_sentiment_drivers(sentiment_details):
    """Display sentiment drivers in card-style boxes (white text, clickable links)."""
    if not sentiment_details:
        st.warning("⚠️ No sentiment details available for driver analysis.")
        return

    df = pd.DataFrame(sentiment_details)
    positives = df[df["label"].str.lower() == "positive"].head(3)
    negatives = df[df["label"].str.lower() == "negative"].head(3)

    st.markdown("### 🧩 Key Sentiment Drivers")

    # Two columns: Bullish | Bearish
    cols = st.columns(2)

    with cols[0]:
        if not positives.empty:
            st.markdown("#### 🟢 Bullish Drivers")
            for _, row in positives.iterrows():
                st.markdown(
                    f"""
                    <div style="
                        background-color:#1E1E1E;
                        padding:12px;
                        border-radius:12px;
                        margin-bottom:10px;
                        border:1px solid #333;">
                        <strong style="color:#4CAF50;">{row['headline']}</strong><br>
                        <span style="color:#CCCCCC; font-size:14px;">{row['summary']}</span><br>
                        <a href="{row.get('link', '#')}" target="_blank" style="color:#6FA8DC; text-decoration:none;">
                            {row['source']} • {row['date']}
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No strong bullish signals detected.")

    with cols[1]:
        if not negatives.empty:
            st.markdown("#### 🔴 Bearish Drivers")
            for _, row in negatives.iterrows():
                st.markdown(
                    f"""
                    <div style="
                        background-color:#1E1E1E;
                        padding:12px;
                        border-radius:12px;
                        margin-bottom:10px;
                        border:1px solid #333;">
                        <strong style="color:#F44336;">{row['headline']}</strong><br>
                        <span style="color:#CCCCCC; font-size:14px;">{row['summary']}</span><br>
                        <a href="{row.get('link', '#')}" target="_blank" style="color:#6FA8DC; text-decoration:none;">
                            {row['source']} • {row['date']}
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No strong bearish signals detected.")


# -----------------------------------------------------
# SYMBOL MAPPING
# -----------------------------------------------------
symbol_queries = {
    "AAPL": "Apple stock OR AAPL",
    "TSLA": "Tesla stock OR TSLA",
    "GOOGL": "Google stock OR Alphabet",
    "AMZN": "Amazon stock OR AMZN",
    "MSFT": "Microsoft stock OR MSFT",
    "BTC-USD": "Bitcoin OR cryptocurrency",
    "ETH-USD": "Ethereum OR crypto",
    "EURUSD=X": "EUR USD forex OR euro dollar",
    "GBPUSD=X": "GBP USD forex OR pound dollar",
}

# -----------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------
st.sidebar.header("🧭 Input Settings")
symbol = st.sidebar.selectbox("Select a market symbol:", list(symbol_queries.keys()), index=0)
period = st.sidebar.selectbox("Select historical period:", ["3mo", "6mo", "1y"], index=1)
analyze_btn = st.sidebar.button("🔍 Analyze Market")

# -----------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------
if analyze_btn:
    with st.spinner("Fetching and analyzing market data..."):
        query = symbol_queries[symbol]
        df = fetch_price_history(symbol, period)
        df = add_technical_indicators(df)
        latest = df.iloc[-1].astype(float, errors="ignore")
        headlines = fetch_newsdata(query)
        sentiment = finbert_sentiment_analysis(headlines, symbol)
        llm_summary = generate_llm_summary(symbol, latest, sentiment, headlines)

    st.subheader(f"Market Overview for **{symbol}**")
    col1, col2 = st.columns([2, 1])
    with col1:
        chart = alt.Chart(df.reset_index()).mark_line(color="#2196f3").encode(
            x="Date:T", y="Close:Q"
        ).properties(title="Price History").interactive()
        st.altair_chart(chart, use_container_width=True)
    with col2:
        st.metric("Current Price", f"${float(latest['Close']):.2f}")
        st.metric("RSI(14)", f"{float(latest['RSI_14']):.2f}")
        st.metric("ATR(14)", f"{float(latest['ATR_14']):.2f}")

    st.markdown("### 🧠 AI Market Analysis (Groq + FinBERT)")
    st.info(llm_summary)
    st.success(f"✅ Overall Sentiment: **{sentiment['label']}** | FinBERT Score: {sentiment['avg_score']:.2f}")

    visualize_sentiment_distribution(sentiment["details"])
    show_key_sentiment_drivers(sentiment["details"])

else:
    st.info("👆 Select a symbol and click *Analyze Market* to start.")
