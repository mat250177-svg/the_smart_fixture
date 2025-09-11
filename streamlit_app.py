# app.py — AI Football Predictor (Poisson + Dixon–Coles + Elo ensemble)
import pandas as pd, numpy as np
from math import factorial
from datetime import datetime
import streamlit as st

st.set_page_config(page_title="AI Football Predictor — Pro", page_icon="⚽", layout="wide")
st.title("⚽ AI Football Predictor — Pro")
st.caption("Upload historical results → Poisson + Dixon–Coles + Elo → single & batch predictions with Kelly")

# ---------- Utils ----------
def parse_date(s):
    try:
        return datetime.fromisoformat(str(s))
    except:
        for fmt in ("%Y-%m-%d","%d/%m/%Y","%d-%m-%Y","%m/%d/%Y"):
            try:
                return datetime.strptime(str(s), fmt)
            except:
                pass
    return None

def poisson_pmf(k, lam): 
    return np.exp(-lam) * (lam ** k) / factorial(k)

def score_matrix(lh, la, max_goals=6):
    ph = [poisson_pmf(i, lh) for i in range(max_goals+1)]
    pa = [poisson_pmf(j, la) for j in range(max_goals+1)]
    M = np.outer(ph, pa)
    return M / M.sum()

def outcome_probs(M):
    home_win = np.tril(M, -1).sum()
    draw = np.trace(M)
    away_win = np.triu(M, 1).sum()
    return float(home_win), float(draw), float(away_win)

def fair_odds(p): 
    return np.inf if p==0 else 1.0/p

def kelly_fraction(p, odds, b=None):
    if p <= 0 or p >= 1 or odds <= 1:
        return 0.0
    if b is None:
        b = odds - 1.0
    q = 1 - p
    f = (b*p - q) / b
    return max(0.0, f)

def clamp(x, lo, hi): 
    return max(lo, min(hi, x))

def dc_adjust_matrix(M, rho=0.05):
    # Dixon–Coles low-score correlation tweak
    M = M.copy()
    if M.shape[0] < 2 or M.shape[1] < 2:
        return M
    M[0,0] *= (1 + rho)
    M[0,1] *= (1 - rho)
    M[1,0] *= (1 - rho)
    M[1,1] *= (1 + rho)
    return M / M.sum()

def elo_probs(elo_h, elo_a, home_adv=50.0, scale=400.0, draw_width=0.22):
    # Convert Elo difference to H/D/A with a simple draw heuristic
    diff = (elo_h + home_adv) - elo_a
    p_home = 1.0 / (1.0 + 10 ** (-diff/scale))
    p_away = 1.0 - p_home
    mid = 4 * p_home * p_away      # max 1 at perfect balance
    p_draw = draw_width * mid
    rem = max(1e-9, 1 - p_draw)
    p_home, p_away = p_home*rem, p_away*rem
    s = p_home + p_draw + p_away
    return p_home/s, p_draw/s, p_away/s

def build_elos(frame, k=20.0, home_adv=50.0, scale=400.0, init=1500.0):
    elos = {}
    for _,row in frame.sort_values('date_parsed').iterrows():
        h, a = row['home'], row['away']
        gh, ga = row['home_goals'], row['away_goals']
        elos.setdefault(h, init)
        elos.setdefault(a, init)
        ph, pd, pa = elo_probs(elos[h], elos[a], home_adv=home_adv, scale=scale, draw_width=0.22)
        if   gh > ga: r_h, r_a = 1.0, 0.0
        elif gh == ga: r_h, r_a = 0.5, 0.5
        else:         r_h, r_a = 0.0, 1.0
        elos[h] += k * (r_h - ph)
        elos[a] += k * (r_a - pa)
    return elos

# ---------- Data upload ----------
st.subheader("1) Upload historical results")
hist = st.file_uploader("CSV columns (required): date,home,away,home_goals,away_goals  • (optional): competition", type=["csv"])

if hist is None:
    st.info("No file uploaded. Using a tiny demo dataset so you can try it now.")
    from io import StringIO
    demo = StringIO("""date,home,away,home_goals,away_goals,competition
2025-08-01,Alpha FC,Beta United,2,1,League
2025-08-08,Alpha FC,Gamma Town,3,1,League
2025-08-15,Beta United,Gamma Town,0,0,League
2025-08-22,Gamma Town,Alpha FC,1,2,League
2025-08-29,Beta United,Alpha FC,1,2,League
2025-09-05,Gamma Town,Beta United,2,2,League
""")
    df = pd.read_csv(demo)
else:
    df = pd.read_csv(hist)

df.columns = [c.strip().lower() for c in df.columns]
req = ['date','home','away','home_goals','away_goals']
missing = [c for c in req if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

for c in ['home','away']:
    df[c] = df[c].astype(str).str.strip()
df['date_parsed'] = df['date'].apply(parse_date)

with st.expander("Preview data"):
    st.dataframe(df.head())

# ---------- Filters & weighting ----------
left, right = st.columns(2)
competitions = sorted(df['competition'].dropna().unique()) if 'competition' in df.columns else []
comp = left.selectbox("Competition filter (optional)", options=["All"] + competitions, index=0)
last_n = left.number_input("Last N matches per team (0 = all)", min_value=0, value=0, step=1)
half_life = right.number_input("Time-decay half-life (days, 0 = off)", min_value=0, value=90, step=5)
dc_rho = right.slider("Dixon–Coles ρ (low-score tweak)", min_value=-0.2, max_value=0.2, value=0.05, step=0.01)

dff = df.copy() if comp=="All" else df[df.get('competition','').eq(comp)].copy()
dff = dff.sort_values('date_parsed')

# Keep last N appearances per team (combined home+away)
if last_n and last_n > 0:
    keep_idx = set()
    for team in sorted(set(dff['home']).union(set(dff['away']))):
        sub = dff[(dff['home']==team) | (dff['away']==team)]
        keep_idx |= set(sub.tail(last_n).index.tolist())
    dff = dff.loc[sorted(list(keep_idx))]

# Exponential time-decay weights
if half_life and half_life > 0:
    latest = dff['date_parsed'].max()
    ages = (latest - dff['date_parsed']).dt.days.clip(lower=0)
    weights = np.power(0.5, ages / half_life)
else:
    weights = np.ones(len(dff))
dff = dff.assign(weight=weights)

# ---------- Build strengths ----------
w = dff['weight'].values
home_goals = np.sum(dff['home_goals'] * w)
away_goals = np.sum(dff['away_goals'] * w)
n_matches = np.sum(w) if np.sum(w)>0 else 1.0
league_home_avg = home_goals / n_matches
league_away_avg = away_goals / n_matches
home_advantage = (league_home_avg / league_away_avg) if league_away_avg>0 else 1.0

home_gp = dff.groupby('home')['home'].count().rename('gp_home')
home_gf = dff.groupby('home').apply(lambda g: np.sum(g['home_goals']*g['weight'])).rename('gf_home')
home_ga = dff.groupby('home').apply(lambda g: np.sum(g['away_goals']*g['weight'])).rename('ga_home')

away_gp = dff.groupby('away')['away'].count().rename('gp_away')
away_gf = dff.groupby('away').apply(lambda g: np.sum(g['away_goals']*g['weight'])).rename('gf_away')
away_ga = dff.groupby('away').apply(lambda g: np.sum(g['home_goals']*g['weight'])).rename('ga_away')

teams = pd.DataFrame({'team': sorted(set(dff['home']).union(set(dff['away'])))}).set_index('team')
teams = teams.join([home_gp, home_gf, home_ga, away_gp, away_gf, away_ga]).fillna(0.0)

eps = 1e-6
teams['home_attack']   = ((teams['gf_home']+eps) / (teams['gp_home']+eps)) / (league_home_avg+eps)
teams['home_defence']  = ((teams['ga_home']+eps) / (teams['gp_home']+eps)) / (league_away_avg+eps)
teams['away_attack']   = ((teams['gf_away']+eps) / (teams['gp_away']+eps)) / (league_away_avg+eps)
teams['away_defence']  = ((teams['ga_away']+eps) / (teams['gp_away']+eps)) / (league_home_avg+eps)

with st.expander("Team strengths"):
    st.dataframe(teams[['home_attack','home_defence','away_attack','away_defence']].round(3))

# ---------- Elo ----------
st.subheader("Elo settings")
ec1, ec2, ec3 = st.columns(3)
elo_k = ec1.number_input("K-factor", min_value=1.0, value=20.0, step=1.0)
elo_home_adv = ec2.number_input("Elo home advantage (pts)", min_value=0.0, value=50.0, step=5.0)
elo_scale = ec3.number_input("Elo scale (denominator)", min_value=100.0, value=400.0, step=50.0)
alpha = st.slider("Ensemble weight α (Poisson/DC share)", 0.0, 1.0, 0.6, 0.05)
max_goals = st.slider("Max goals per side", 4, 10, 6)

elos = build_elos(dff.dropna(subset=['date_parsed']), k=elo_k, home_adv=elo_home_adv, scale=elo_scale)

# ---------- Single Fixture ----------
st.subheader("2) Single fixture prediction")
c1, c2 = st.columns(2)
home_team = c1.selectbox("Home team", options=teams.index.tolist())
away_team = c2.selectbox("Away team", options=[t for t in teams.index.tolist() if t != home_team])

# Optional manual adjustments (% multipliers)
st.caption("Optional: apply temporary attack/defence adjustments (e.g., injuries). 1.00 = no change.")
aj1, aj2, aj3, aj4 = st.columns(4)
home_att_adj = aj1.number_input("Home attack ×", min_value=0.5, max_value=1.5, value=1.00, step=0.01)
home_def_adj = aj2.number_input("Home defence ×", min_value=0.5, max_value=1.5, value=1.00, step=0.01)
away_att_adj = aj3.number_input("Away attack ×", min_value=0.5, max_value=1.5, value=1.00, step=0.01)
away_def_adj = aj4.number_input("Away defence ×", min_value=0.5, max_value=1.5, value=1.00, step=0.01)

ha = teams.loc[home_team,'home_attack'] * home_att_adj
hd = clamp(teams.loc[home_team,'home_defence'] * home_def_adj, 0.25, 4.0)
aa = teams.loc[away_team,'away_attack'] * away_att_adj
ad = clamp(teams.loc[away_team,'away_defence'] * away_def_adj, 0.25, 4.0)

lam_home = league_home_avg * ha * ad * home_advantage
lam_away = league_away_avg * aa * hd

M = score_matrix(lam_home, lam_away, max_goals=max_goals)
M = dc_adjust_matrix(M, rho=dc_rho)
pH_p, pD_p, pA_p = outcome_probs(M)

elo_h = elos.get(home_team,1500.0); elo_a = elos.get(away_team,1500.0)
pH_e, pD_e, pA_e = elo_probs(elo_h, elo_a, home_adv=elo_home_adv, scale=elo_scale, draw_width=0.22)

pH = alpha*pH_p + (1-alpha)*pH_e
pD = alpha*pD_p + (1-alpha)*pD_e
pA = alpha*pA_p + (1-alpha)*pA_e

st.write(f"**λ (xG proxy):** Home {lam_home:.3f} | Away {lam_away:.3f}")
st.write(f"Poisson/DC → H {pH_p:.3f} • D {pD_p:.3f} • A {pA_p:.3f}")
st.write(f"Elo → H {pH_e:.3f} • D {pD_e:.3f} • A {pA_e:.3f}")
st.write(f"**Ensemble → H {pH:.3f} • D {pD:.3f} • A {pA:.3f}**")
st.write(f"**Fair odds →** Home {1/pH:.2f} • Draw {1/pD:.2f} • Away {1/pA:.2f}")

st.subheader("Kelly staking (optional)")
bk1, bk2, bk3, bk4 = st.columns(4)
odds_home = bk1.number_input("Odds - Home", min_value=1.0, value=1.0, step=0.01)
odds_draw = bk2.number_input("Odds - Draw", min_value=1.0, value=1.0, step=0.01)
odds_away = bk3.number_input("Odds - Away", min_value=1.0, value=1.0, step=0.01)
bankroll = bk4.number_input("Bankroll", min_value=0.0, value=100.0, step=1.0)

fH = kelly_fraction(pH, odds_home) if odds_home>1 else 0.0
fD = kelly_fraction(pD, odds_draw) if odds_draw>1 else 0.0
fA = kelly_fraction(pA, odds_away) if odds_away>1 else 0.0
st.write(f"Stake (Full Kelly): Home £{fH*bankroll:.2f} • Draw £{fD*bankroll:.2f} • Away £{fA*bankroll:.2f}")
st.caption("Tip: use Half-Kelly (halve the stakes) to reduce volatility.")

with st.expander("Top 15 correct scores (Poisson/DC)"):
    flat = [((i,j), M[i,j]) for i in range(max_goals+1) for j in range(max_goals+1)]
    top = sorted(flat, key=lambda x: x[1], reverse=True)[:15]
    for (i,j),p in top:
        st.write(f"{i}-{j}: {p:.3f}")

# ---------- Batch predictions ----------
st.subheader("3) Batch predictions")
fix = st.file_uploader("Upcoming fixtures CSV (columns: date,home,away; optional: odds_home,odds_draw,odds_away)", type=["csv"], key="fixtures")

if fix is not None:
    fixtures = pd.read_csv(fix)
    fixtures.columns = [c.strip().lower() for c in fixtures.columns]
    for c in ['home','away']:
        fixtures[c] = fixtures[c].astype(str).str.strip()

    out = []
    for _,r in fixtures.iterrows():
        h, a = r['home'], r['away']
        if h not in teams.index or a not in teams.index:
            continue
        lam_h = league_home_avg * teams.loc[h,'home_attack'] * clamp(teams.loc[a,'away_defence'],0.25,4.0) * home_advantage
        lam_a = league_away_avg * teams.loc[a,'away_attack'] * clamp(teams.loc[h,'home_defence'],0.25,4.0)
        M2 = score_matrix(lam_h, lam_a, max_goals=max_goals)
        M2 = dc_adjust_matrix(M2, rho=dc_rho)
        ph_p, pd_p, pa_p = outcome_probs(M2)

        elo_h = elos.get(h,1500.0); elo_a = elos.get(a,1500.0)
        ph_e, pd_e, pa_e = elo_probs(elo_h, elo_a, home_adv=elo_home_adv, scale=elo_scale, draw_width=0.22)

        ph = alpha*ph_p + (1-alpha)*ph_e
        pdx = alpha*pd_p + (1-alpha)*pd_e
        pa = alpha*pa_p + (1-alpha)*pa_e

        row = {
            'date': r.get('date',''),
            'home': h, 'away': a,
            'p_home': ph, 'p_draw': pdx, 'p_away': pa,
            'fair_home': (1/ph) if ph>0 else np.inf,
            'fair_draw': (1/pdx) if pdx>0 else np.inf,
            'fair_away': (1/pa) if pa>0 else np.inf,
            'lam_home': lam_h, 'lam_away': lam_a
        }
        for k in ['odds_home','odds_draw','odds_away']:
            if k in fixtures.columns and pd.notna(r.get(k, np.nan)):
                row[k] = r[k]
                # crude value flag: model prob > implied prob
                implied = 1.0 / r[k] if r[k] and r[k] > 0 else np.inf
                prob = row['p_home'] if k=='odds_home' else (row['p_draw'] if k=='odds_draw' else row['p_away'])
                row[f'value_{k}'] = prob > (1.0 / r[k]) if r[k] and r[k] > 0 else False
        out.append(row)

    out_df = pd.DataFrame(out)
    st.dataframe(out_df)
    st.download_button("Download predictions.csv", out_df.to_csv(index=False).encode('utf-8'), file_name="predictions.csv", mime="text/csv")
else:
    st.info("Upload a fixtures CSV to run batch predictions.")