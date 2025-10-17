# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 21:27:00 2025

@author: Oreoluwa
"""

import streamlit as st

st.set_page_config(page_title="TradionAI Risk Analysis", layout="wide")

st.title("⚖️ Risk Analysis Dashboard")
st.caption("Assess trade risk, reward potential, position sizing, and leverage impact for smarter trading decisions.")

# -----------------------------------------------------
# 1️⃣ USER INPUT SECTION
# -----------------------------------------------------
st.subheader("📋 Trade & Account Details")

col1, col2, col3 = st.columns(3)

with col1:
    account_balance = st.number_input("💰 Account Balance ($)", min_value=100.0, value=10000.0, step=100.0)
    risk_percent = st.number_input("📉 Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    leverage = st.number_input("🧮 Leverage (e.g. 10 = 1:10)", min_value=1, max_value=1000, value=100, step=10)

with col2:
    symbol = st.text_input("💹 Symbol / Pair", value="EUR/USD")
    direction = st.selectbox("Trade Direction", ["Buy", "Sell"])
    entry_price = st.number_input("🎯 Entry Price", min_value=0.0001, value=1.0850, step=0.0001, format="%.5f")

with col3:
    stop_loss = st.number_input("🛑 Stop Loss", min_value=0.0001, value=1.0800, step=0.0001, format="%.5f")
    take_profit = st.number_input("🏁 Take Profit", min_value=0.0001, value=1.0950, step=0.0001, format="%.5f")

st.markdown("<hr style='border: 0.5px solid #333;'>", unsafe_allow_html=True)

# -----------------------------------------------------
# 2️⃣ RISK CALCULATIONS
# -----------------------------------------------------
st.subheader("📊 Calculated Risk Metrics")

# Basic calculations
risk_amount = account_balance * (risk_percent / 100)
pip_risk = abs(entry_price - stop_loss)
pip_reward = abs(take_profit - entry_price)
rr_ratio = round(pip_reward / pip_risk, 2) if pip_risk != 0 else 0
position_size = round(risk_amount / pip_risk, 2) if pip_risk != 0 else 0

# Apply leverage adjustments
exposure = position_size * entry_price
margin_required = exposure / leverage if leverage > 0 else exposure
margin_percent = (margin_required / account_balance) * 100

# Determine risk level (now includes leverage impact)
effective_risk = risk_percent * (leverage / 100)
if rr_ratio >= 2 and effective_risk <= 2:
    risk_level = "🟢 Low Risk / High Reward"
    risk_score = 0.2
elif 1.5 <= rr_ratio < 2 or 2 < effective_risk <= 4:
    risk_level = "🟡 Moderate Risk"
    risk_score = 0.5
else:
    risk_level = "🔴 High Risk / Overleveraged"
    risk_score = 0.8

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Risk Amount ($)", f"{risk_amount:,.2f}")
with col2:
    st.metric("Reward–Risk Ratio", f"{rr_ratio}:1")
with col3:
    st.metric("Position Size (Units)", f"{position_size:,.2f}")

col4, col5, col6 = st.columns(3)

with col4:
    st.metric("Leverage", f"1:{leverage}")
with col5:
    st.metric("Margin Required ($)", f"{margin_required:,.2f}")
with col6:
    st.metric("Margin % of Account", f"{margin_percent:.2f}%")

# -----------------------------------------------------
# 3️⃣ RISK GAUGE BAR
# -----------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("📈 Risk Level")
st.progress(risk_score)
st.caption(f"Current Risk Assessment: **{risk_level}**")

st.markdown("<hr style='border: 0.5px solid #333;'>", unsafe_allow_html=True)

# -----------------------------------------------------
# 4️⃣ TEXTUAL SUMMARY (AI-LIKE EXPLANATION)
# -----------------------------------------------------
st.subheader("🧠 Trade Risk Summary")

summary = f"""
For the **{symbol}** trade setup:
- Account balance: **${account_balance:,.2f}** with **{risk_percent}% risk per trade** (≈ ${risk_amount:,.2f}).
- Entry: **{entry_price}**, Stop Loss: **{stop_loss}**, Take Profit: **{take_profit}**.
- Reward-to-risk ratio: **{rr_ratio}:1** → **{risk_level.replace('🟢 ','').replace('🟡 ','').replace('🔴 ','')}**.
- Position size: **{position_size:,.2f} units**, exposure ≈ **${exposure:,.2f}**.
- Using **1:{leverage} leverage**, margin required ≈ **${margin_required:,.2f}** ({margin_percent:.2f}% of account).
"""

st.markdown(
    f"""
    <div style="background-color:#1E1E1E; padding:15px; border-radius:10px; border:1px solid #333;">
        <span style="color:#CCCCCC; font-size:15px;">{summary}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
