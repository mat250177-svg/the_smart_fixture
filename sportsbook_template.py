# Sportsbook-Style Streamlit Template (front-end demo)
# Looks/feels like a bookmaker UI: header, sport tabs, markets, odds buttons, betslip.
# Notes: UI-only (no accounts/betting). Add your own data sources (API-Football/Odds API/etc).

import math
import datetime as dt
import streamlit as st

st.set_page_config(page_title="Sportsbook Template (Demo)", layout="wide")

# ---------- Minimal theme + sticky betslip (uses CSS) ----------
st.markdown("""
<style>
:root{
  --bg:#0f1116; --card:#151823; --muted:#9aa3b2; --text:#e7ecf3; --accent:#00dc82; --accent-2:#39c0ff;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg);}
h1,h2,h3,h4,h5,h6,p,span,div,code{color:var(--text);}
.block-container{padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px;}
.sb-card{background:var(--card); border:1px solid #1f2433; border-radius:14px; padding:14px; box-shadow:0 1px 10px rgba(0,0,0,.24);}
.sb-chip{display:inline-block; padding:6px 10px; border-radius:999px; background:#1c2130; color:#e7ecf3; font-size:12px; margin-right:6px;}
.sb-badge{display:inline-block; padding:2px 8px; border-radius:6px; background:#121722; color:#9aa3b2; font-size:11px; margin-left:8px; border:1px solid #242a3a;}
.sb-odd{display:inline-block; min-width:72px; padding:10px 12px; border-radius:12px; background:#111624; border:1px solid #242a3a; text-align:center; cursor:pointer; user-select:none;}
.sb-odd:hover{border-color:#2f9fff; box-shadow:0 0 0 2px rgba(47,159,255,.15) inset;}
.sb-odd.active{border-color:var(--accent); box-shadow:0 0 0 2px rgba(0,220,130,.18) inset;}
.sb-row{display:flex; gap:10px; flex-wrap:wrap;}
.sb-team{display:flex; align-items:center; gap:8px; font-weight:600;}
.sb-kick{color:var(--muted); font-size:12px;}
.sb-nav{display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;}
.sb-tabs .tab{display:inline-block; padding:8px 14px; margin-right:6px; border-radius:10px; background:#121722; color:#cdd5e1; border:1px solid #242a3a; cursor:pointer;}
.sb-tabs .tab.active{background:#192234; border-color:#2f9fff; color:#e7f2ff;}
/* sticky betslip (right column) */
.sidebar-sticky{position: sticky; top: 12px;}
.small{font-size:12px; color:var(--muted);}
hr{border:0; border-top:1px solid #23293a; margin:12px 0;}
a, a:visited{color:var(--accent-2); text-decoration:none;}
</style>
""", unsafe_allow_html=True)

# ---------- State ----------
if "betslip" not in st.session_state:
    st.session_state.betslip = []   # list of dicts: {event_id, selection, price_dec, label}
if "odds_fmt" not in st.session_state:
    st.session_state.odds_fmt = "Decimal"

# ---------- Helpers ----------
def dec_to_frac(x):
    # Simple decimal -> fractional converter (e.g., 2.5 -> 6/4); not reduced perfectly for all cases
    num = max(1, round((x - 1.0)*100))
    den = 100
    # reduce
    from math import gcd
    g = gcd(num, den)
    num//=g
