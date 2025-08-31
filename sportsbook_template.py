# Sportsbook Simple (Tablet-Friendly, No CSS)
# Front-end demo only: header, odds buttons, sticky-ish betslip, decimal/fractional toggle.
# No custom CSS, so it won't go black on some Android tablets.
# Research-only. 18+ | BeGambleAware.org

import math
import streamlit as st

st.set_page_config(page_title="Sportsbook (Simple Demo)", layout="wide")

# -------- State --------
if "betslip" not in st.session_state:
    st.session_state.betslip = []  # each: {event_id, selection, price_dec, label}
if "odds_fmt" not in st.session_state:
    st.session_state.odds_fmt = "Decimal"

# -------- Helpers --------
def dec_to_frac(x):
    # quick decimal -> fractional (not perfect, but fine for demo)
    num = max(1, round((x - 1.0) * 100))
    den = 100
    from math import gcd
    g = gcd(num, den)
    num //= g; den //= g
    return f"{num}/{den}"

def show_price(p):
    return f"{p:.2f}" if st.session_state.odds_fmt == "Decimal" else dec_to_frac(p)

def toggle_pick(event_id, selection, price_dec, label):
    key = (event_id, selection)
    for i, leg in enumerate(st.session_state.betslip):
        if (leg["event_id"], leg["selection"]) == key:
            del st.session_state.betslip[i]  # remove if already there
            return
    st.session_state.betslip.append({
        "event_id": event_id,
        "selection": selection,
        "price_dec": float(price_dec),
        "label": label,
    })

def acca_price(legs):
    p = 1.0
    for leg in legs:
        p *= leg["price_dec"]
    return p

# -------- Demo data --------
events = [
    {"id":"E1", "kick":"Sat 31 Aug 17:30", "league":"Premier League", "home":"Arsenal",   "away":"Chelsea",   "prices":{"Home":1.95,"Draw":3.60,"Away":4.10}},
    {"id":"E2", "kick":"Sun 01 Sep 16:30", "league":"Premier League", "home":"Liverpool", "away":"Tottenham", "prices":{"Home":2.00,"Draw":3.65,"Away":3.90}},
    {"id":"E3", "kick":"Sun 01 Sep 14:00", "league":"La Liga",        "home":"Barcelona","away":"Valencia",  "prices":{"Home":1.60,"Draw":4.20,"Away":5.60}},
]

# -------- Top bar --------
top_left, top_right = st.columns([2, 1])
with top_left:
    st.title("Smart Sportsbook (Demo)")
    st.caption("Research only · 18+ | BeGambleAware.org")
with top_right:
    st.selectbox("Odds format", ["Decimal", "Fractional"], key="odds_fmt")

st.divider()

# -------- Main two-column layout --------
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Football · Match Result (1X2)")
    for ev in events:
        st.markdown(f"**{ev['home']} vs {ev['away']}**  \n{ev['league']} · {ev['kick']}")
        c1, c2, c3 = st.columns(3)
        for idx, sel in enumerate(["Home", "Draw", "Away"]):
            price = ev["prices"][sel]
            label = f"{sel} @ {show_price(price)}"
            active = any(b["event_id"] == ev["id"] and b["selection"] == sel
                         for b in st.session_state.betslip)
            # Use a checkbox to show "selected" state, and a button to toggle
            with (c1 if idx == 0 else c2 if idx == 1 else c3):
                st.checkbox(label, value=active, key=f"chk_{ev['id']}_{sel}", disabled=True)
                if st.button("Add/Remove", key=f"btn_{ev['id']}_{sel}"):
                    toggle_pick(ev["id"], sel, price, f"{ev['home']} vs {ev['away']} · {sel}")
        st.write("")

with right:
    st.subheader("Betslip")
    if not st.session_state.betslip:
        st.info("No selections yet. Tap Add/Remove on an odd.")
    else:
        # list legs
        for i, leg in enumerate(list(st.session_state.betslip)):
            row = st.container()
            with row:
                cA, cB, cC = st.columns([6, 3, 1])
                cA.write(leg["label"])
                cB.write(f"@ {show_price(leg['price_dec'])}")
                if cC.button("✕", key=f"rm_{i}"):
                    del st.session_state.betslip[i]
                    st.experimental_rerun()

        st.divider()
        stake = st.number_input("Stake (£)", min_value=0.0, value=5.0, step=0.5)
        if stake > 0 and st.session_state.betslip:
            mult = acca_price(st.session_state.betslip)
            returns = stake * mult
            st.write(f"Accumulator price: **{mult:.2f}** (decimal)")
            st.write(f"Potential returns: **£{returns:.2f}**")

st.caption("Prices are demo only. No real betting. 18+ | BeGambleAware.org")
