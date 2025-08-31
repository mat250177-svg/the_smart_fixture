# The Smart Fixture - live fixtures with season + date picker and season-long reel
# Uses API-Football if a key is present in Streamlit Secrets; otherwise demo data.
# ASCII-only, no CSS, no f-strings (tablet-safe).

import io
import json
import math
import datetime as dt
import pandas as pd
import streamlit as st

# Optional imports for live API and timezone
try:
    import requests
except Exception:
    requests = None

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # fallback if not available

st.set_page_config(page_title="The Smart Fixture", layout="wide")

APP_TITLE = "The Smart Fixture - Research Only"
DISCLAIMER = (
    "This is a research app, not a prediction or tipster service. "
    "We do not take bets. Prices change quickly. 18+ | BeGambleAware.org"
)

# Header
col1, col2 = st.columns([1, 9])
with col1:
    try:
        st.image("smart_fixture_shield_512_lime.png", width=64)
    except Exception:
        pass
with col2:
    st.title(APP_TITLE)
    st.info(DISCLAIMER)

# ---------------- helpers ----------------
def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0

def poisson_prob(lmbda, k):
    return math.exp(-lmbda) * (lmbda ** k) / math.factorial(k)

def match_probs(lambda_home, lambda_away, max_goals=10):
    # precompute PMFs for perf; normalize defensively
    ph = [poisson_prob(lambda_home, h) for h in range(max_goals + 1)]
    pa = [poisson_prob(lambda_away, a) for a in range(max_goals + 1)]
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = ph[h] * pa[a]
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p
    s = p_home + p_draw + p_away
    if s > 0:
        p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s
    return p_home, p_draw, p_away

def fair_odds(p):
    return (1.0 / p) if p > 0 else float("inf")

def to_london(dt_iso):
    # Convert API ISO string to Europe/London datetime and label
    try:
        t = dt.datetime.fromisoformat(str(dt_iso).replace("Z", "+00:00"))
        if t.tzinfo is None:
            # assume UTC if naive
            t = t.replace(tzinfo=dt.timezone.utc)
        if ZoneInfo is not None:
            t = t.astimezone(ZoneInfo("Europe/London"))
        return t, t.strftime("%Y-%m-%d"), t.strftime("%a %d %b %H:%M")
    except Exception:
        return None, "?", str(dt_iso)

def now_london():
    try:
        if ZoneInfo is not None:
            return dt.datetime.now(ZoneInfo("Europe/London"))
    except Exception:
        pass
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

def today_london_date():
    try:
        if ZoneInfo is not None:
            return dt.datetime.now(ZoneInfo("Europe/London")).date()
    except Exception:
        pass
    return dt.datetime.utcnow().date()

def week_bounds_london_monday(ref_date=None):
    if ref_date is None:
        ref_date = today_london_date()
    monday = ref_date - dt.timedelta(days=ref_date.weekday())
    sunday = monday + dt.timedelta(days=6)
    return monday, sunday

def parse_season(value):
    s = str(value).strip().replace(" ", "").replace("-", "/")
    start = s.split("/")[0] if "/" in s else s
    if not start.isdigit():
        today = dt.date.today()
        return today.year if today.month >= 7 else today.year - 1
    n = int(start)
    return n + 2000 if n < 100 else n

def season_week(season_start_year, ref_date=None):
    if ref_date is None:
        ref_date = dt.date.today()
    season_anchor = dt.date(season_start_year, 7, 1)
    offset = (7 - season_anchor.weekday()) % 7
    first_monday = season_anchor + dt.timedelta(days=offset)
    if ref_date < first_monday:
        return 1
    return ((ref_date - first_monday).days // 7) + 1

def form_symbols(results):
    # simple W D L chips as text for now
    return " ".join(results)

def wdl_from_rows(rows):
    out = []
    for r in rows[-5:]:
        if r.get("gf", 0) > r.get("ga", 0):
            out.append("W")
        elif r.get("gf", 0) < r.get("ga", 0):
            out.append("L")
        else:
            out.append("D")
    return out

def wdl_from_h2h(h2h_rows):
    home_res, away_res = [], []
    for r in h2h_rows[-5:]:
        hg = r.get("hg", 0)
        ag = r.get("ag", 0)
        if hg > ag:
            home_res.append("W"); away_res.append("L")
        elif hg < ag:
            home_res.append("L"); away_res.append("W")
        else:
            home_res.append("D"); away_res.append("D")
    return home_res, away_res

# ---------------- LIVE DATA (API-Football) ----------------
API_KEY = st.secrets.get("API_FOOTBALL_KEY", "")
DIRECT_HOST = st.secrets.get("API_FOOTBALL_HOST", "v3.football.api-sports.io")
RAPID_HOST = "api-football-v1.p.rapidapi.com"

def api_providers():
    # tuple of (base_url, headers)
    return [
        ("https://" + DIRECT_HOST, {"x-apisports-key": API_KEY}),
        ("https://" + RAPID_HOST + "/v3", {"x-rapidapi-key": API_KEY, "x-rapidapi-host": RAPID_HOST}),
    ]

@st.cache_data(ttl=3600, show_spinner=False)
def api_get(path, params=None):
    if (not API_KEY) or (requests is None):
        return None, "no_key_or_requests"
    params = params or {}
    last_err = None
    for base_url, hdrs in api_providers():
        try:
            r = requests.get(base_url + path, headers=hdrs, params=params, timeout=25)
            if r.status_code == 200:
                j = r.json()
                return j.get("response", []), None
            last_err = "{} {}".format(r.status_code, str(r.text)[:200])
        except Exception as e:
            last_err = str(e)
    return None, last_err or "unknown_error"

def apifb_countries():
    resp, err = api_get("/countries")
    if not resp:
        return []
    names = [r.get("name") for r in resp if r.get("name")]
    out, seen = [], set()
    for n in names:
        if n not in seen:
            out.append(n); seen.add(n)
    return sorted(out)

def apifb_leagues(country, type_filter):
    resp, err = api_get("/leagues", {"country": country})
    resp = resp or []
    rows = []
    for item in resp:
        league = item.get("league", {})
        if league.get("type") != type_filter:
            continue
        rows.append({
            "id": league.get("id"),
            "name": league.get("name"),
            "seasons": item.get("seasons", []),  # list of dicts with "year"
        })
    # unique by (id, name)
    uniq, seen = [], set()
    for r in rows:
        key = (r.get("id"), r.get("name"))
        if key not in seen:
            uniq.append(r); seen.add(key)
    return uniq

@st.cache_data(ttl=1800, show_spinner=False)
def apifb_fixtures_range(league_id, season_start_year, date_from, date_to):
    # Pull fixtures for a date range (YYYY-MM-DD). Ask API for Europe/London timezone.
    resp, err = api_get("/fixtures", {
        "league": league_id,
        "season": season_start_year,
        "from": date_from,
        "to": date_to,
        "timezone": "Europe/London",
    })
    return (resp or []), err

def season_bounds(season_start_year):
    # July 1 to June 30 of next year (covers most leagues; safe wide net)
    start = dt.date(season_start_year, 7, 1)
    end   = dt.date(season_start_year + 1, 6, 30)
    return start, end

def apifb_row_basic(r):
    teams = r.get("teams", {})
    fixture = r.get("fixture", {})
    iso = fixture.get("date", "")
    t_london, date_key, label = to_london(iso)
    return {
        "kickoff_iso": iso,
        "kickoff_label": label,
        "kickoff_dt_london": t_london,
        "date_key": date_key,  # YYYY-MM-DD in London
        "home": (teams.get("home") or {}).get("name", "?"),
        "away": (teams.get("away") or {}).get("name", "?"),
        "home_id": (teams.get("home") or {}).get("id"),
        "away_id": (teams.get("away") or {}).get("id"),
        "raw": r,
    }

# ---------------- DEMO data (fallback) ----------------
DEMO_COUNTRIES = ["England", "Spain"]
DEMO_LEAGUES = {
    "England": ["Premier League", "Championship"],
    "Spain": ["La Liga"]
}
DEMO_CUPS = {
    "England": ["FA Cup"],
    "Spain": ["Copa del Rey"]
}

def demo_row(days, home, away):
    # tz-aware UTC, then convert in to_london
    utc_dt = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=days)
    iso = utc_dt.isoformat()
    t_london, date_key, label = to_london(iso)
    return {
        "kickoff_iso": iso,
        "kickoff_label": label,
        "kickoff_dt_london": t_london,
        "date_key": date_key,
        "home": home,
        "away": away,
        "raw": {},
    }

DEMO_FIX_SEASON = {
    ("League","England","Premier League", 2025): [
        demo_row(1,"Arsenal","Chelsea"), demo_row(1,"Liverpool","Tottenham"), demo_row(2,"Man City","Newcastle")
    ],
    ("League","England","Championship", 2025): [
        demo_row(1,"Leeds","Leicester"), demo_row(2,"Southampton","Norwich"), demo_row(3,"Sunderland","Ipswich")
    ],
    ("League","Spain","La Liga", 2025): [
        demo_row(1,"Real Madrid","Sevilla"), demo_row(2,"Barcelona","Valencia")
    ],
    ("Cup","England","FA Cup", 2025): [
        demo_row(4,"Everton","West Ham")
    ],
    ("Cup","Spain","Copa del Rey", 2025): [
        demo_row(5,"Real Sociedad","Betis")
    ],
}

DEMO_FORM = {
    ("Arsenal","Chelsea"): {
        "home_home": [{"gf":2,"ga":0},{"gf":1,"ga":1},{"gf":3,"ga":2},{"gf":0,"ga":0},{"gf":2,"ga":1}],
        "away_away": [{"gf":1,"ga":2},{"gf":0,"ga":1},{"gf":2,"ga":2},{"gf":1,"ga":3},{"gf":0,"ga":0}],
        "h2h": [{"hg":2,"ag":1},{"hg":0,"ag":0},{"hg":3,"ag":0},{"hg":1,"ag":2},{"hg":1,"ag":1}],
    },
    ("Liverpool","Tottenham"): {
        "home_home": [{"gf":3,"ga":1},{"gf":2,"ga":0},{"gf":2,"ga":1},{"gf":1,"ga":0},{"gf":4,"ga":2}],
        "away_away": [{"gf":2,"ga":2},{"gf":1,"ga":1},{"gf":0,"ga":1},{"gf":2,"ga":3},{"gf":1,"ga":1}],
        "h2h": [{"hg":2,"ag":2},{"hg":3,"ag":1},{"hg":1,"ag":0},{"hg":2,"ag":1},{"hg":1,"ag":1}],
    },
    ("Man City","Newcastle"): {
        "home_home": [{"gf":4,"ga":1},{"gf":2,"ga":0},{"gf":5,"ga":2},{"gf":1,"ga":0},{"gf":3,"ga":1}],
        "away_away": [{"gf":1,"ga":1},{"gf":0,"ga":1},{"gf":2,"ga":2},{"gf":1,"ga":3},{"gf":0,"ga":0}],
        "h2h": [{"hg":3,"ag":1},{"hg":2,"ag":0},{"hg":4,"ag":1},{"hg":1,"ag":1},{"hg":2,"ag":2}],
    },
    ("Real Madrid","Sevilla"): {
        "home_home": [{"gf":2,"ga":0},{"gf":1,"ga":0},{"gf":2,"ga":1},{"gf":3,"ga":1},{"gf":2,"ga":0}],
        "away_away": [{"gf":1,"ga":2},{"gf":0,"ga":2},{"gf":2,"ga":2},{"gf":1,"ga":1},{"gf":0,"ga":1}],
        "h2h": [{"hg":2,"ag":0},{"hg":1,"ag":1},{"hg":2,"ag":1},{"hg":3,"ag":1},{"hg":1,"ag":0}],
    },
    ("Barcelona","Valencia"): {
        "home_home": [{"gf":3,"ga":0},{"gf":2,"ga":1},{"gf":2,"ga":0},{"gf":1,"ga":0},{"gf":4,"ga":2}],
        "away_away": [{"gf":1,"ga":1},{"gf":0,"ga":1},{"gf":1,"ga":2},{"gf":2,"ga":2},{"gf":1,"ga":0}],
        "h2h": [{"hg":2,"ag":1},{"hg":0,"ag":0},{"hg":3,"ag":1},{"hg":2,"ag":0},{"hg":1,"ag":1}],
    },
}

TEAM_NEWS = {
    "Arsenal": [{"name":"Player A","yc":2,"s":0},{"name":"Player B","yc":4,"s":1}],
    "Chelsea": [{"name":"Player C","yc":1,"s":0}],
    "Liverpool": [{"name":"Player D","yc":3,"s":0},{"name":"Player E","yc":5,"s":2}],
    "Tottenham": [{"name":"Player F","yc":2,"s":0}],
    "Man City": [{"name":"Player G","yc":1,"s":0}],
    "Newcastle": [{"name":"Player H","yc":4,"s":0}],
    "Real Madrid": [{"name":"Player I","yc":2,"s":0}],
    "Sevilla": [{"name":"Player J","yc":3,"s":1}],
    "Barcelona": [{"name":"Player K","yc":1,"s":0}],
    "Valencia": [{"name":"Player L","yc":2,"s":0}],
}

# ---------------- UI: competition, season, date, match ----------------
st.subheader("Choose competition")
c1, c2, c3, c4 = st.columns([1,1,1,1])

has_key = bool(API_KEY) and (requests is not None)
countries = apifb_countries() if has_key else DEMO_COUNTRIES
if not countries:
    countries = DEMO_COUNTRIES

country = c1.selectbox("Country", countries, index=0)

if has_key:
    league_rows = apifb_leagues(country, "League") or [{"id":None, "name":"-", "seasons":[]}]
    cup_rows    = apifb_leagues(country, "Cup")    or []
else:
    league_rows = [{"id":None, "name":n, "seasons":[{"year":2025}]} for n in DEMO_LEAGUES.get(country, [])]
    cup_rows    = [{"id":None, "name":n, "seasons":[{"year":2025}]} for n in DEMO_CUPS.get(country, [])]

league_names = [r.get("name", "-") for r in league_rows] or ["-"]
cup_names    = ["- None -"] + [r.get("name", "-") for r in cup_rows]

league_name = c2.selectbox("League", league_names, index=0)
cup_name    = c3.selectbox("Cup", cup_names, index=0)

# seasons list comes from chosen competition row
def get_season_options(rows, name):
    yrs = []
    for r in rows:
        if r.get("name") == name:
            for s in r.get("seasons", []):
                y = s.get("year")
                if isinstance(y, int):
                    yrs.append(y)
    yrs = sorted(list(set(yrs)), reverse=True)
    if not yrs:
        yrs = [parse_season("2025/26")]
    return yrs

if cup_name != "- None -":
    season_options = get_season_options(cup_rows, cup_name)
else:
    season_options = get_season_options(league_rows, league_name)

season = c4.selectbox("Season (start year)", season_options, index=0)

def find_row(rows, name):
    for r in rows:
        if r.get("name") == name:
            return r
    return None

comp_row = find_row(cup_rows, cup_name) if cup_name != "- None -" else find_row(league_rows, league_name)
league_id = (comp_row or {}).get("id")

# Fetch fixtures for the entire season (live or demo)
st.subheader("Pick a match")

using_demo = False
api_err = None
season_rows = []

if has_key and league_id:
    season_start, season_end = season_bounds(season)
    api_rows, api_err = apifb_fixtures_range(league_id, season, season_start.isoformat(), season_end.isoformat())
    if api_rows:
        season_rows = [apifb_row_basic(r) for r in api_rows]
    else:
        using_demo = True
else:
    using_demo = True

if using_demo:
    key = ("Cup", country, cup_name, season) if (cup_name and cup_name != "- None -") else ("League", country, league_name, season)
    season_rows = DEMO_FIX_SEASON.get(key, [])

# Demo note
if using_demo:
    st.caption("Using demo data. Live API not available or returned no fixtures.")
elif api_err:
    st.warning("Live API error: {}".format(api_err))

if not season_rows:
    st.warning("No fixtures found for the chosen competition and season.")
else:
    # Build date dropdown options from the season fixtures
    by_date = {}
    for r in season_rows:
        by_date.setdefault(r["date_key"], []).append(r)
    date_list = sorted(by_date.keys())

    # default date = next upcoming or first
    today_key = today_london_date().strftime("%Y-%m-%d")
    default_date = date_list[0]
    for dkey in date_list:
        if dkey >= today_key:
            default_date = dkey
            break

    tab_by_date, tab_reel = st.tabs(["By date", "Season reel"])

    # --- By date tab ---
    with tab_by_date:
        date_choice = st.selectbox("Date (Europe/London)", date_list, index=date_list.index(default_date) if default_date in date_list else 0)
        rows_for_date = sorted(by_date.get(date_choice, []), key=lambda x: x["kickoff_iso"])

        # optional filter: only upcoming
        only_upcoming = st.checkbox("Show upcoming only", value=True)
        if only_upcoming:
            now_local = now_london()
            rows_for_date = [r for r in rows_for_date if (r.get("kickoff_dt_london") and r["kickoff_dt_london"] >= now_local)]

        def label_for(ix):
            r = rows_for_date[ix]
            return "{} - {} vs {}".format(r["kickoff_label"], r["home"], r["away"])

        if rows_for_date:
            sel_ix = st.selectbox("Match on that date", list(range(len(rows_for_date))), format_func=label_for)
            match = rows_for_date[sel_ix]
        else:
            st.info("No matches on this date with the current filter.")
            match = None

    # --- Season-long reel tab ---
    with tab_reel:
        all_sorted = sorted(season_rows, key=lambda x: x["kickoff_iso"])
        reel_labels = ["{} - {} vs {}".format(r["kickoff_label"], r["home"], r["away"]) for r in all_sorted]

        # default reel position = next upcoming
        start_ix = 0
        for i, r in enumerate(all_sorted):
            if r.get("date_key", "") >= today_key:
                start_ix = i
                break

        # safe-guard against empty list
        if reel_labels:
            default_label = reel_labels[start_ix]
            reel_choice = st.select_slider("Scroll all season fixtures", options=reel_labels, value=default_label)
            match = all_sorted[reel_labels.index(reel_choice)]
        else:
            st.info("No fixtures to show in reel.")
            # do not override match picked in the other tab

    if match is not None:
        # Unpack the chosen match
        home, away = match["home"], match["away"]

        # --- Demo form/H2H placeholders until live stats are wired ---
        pair = DEMO_FORM.get((home, away), {"home_home":[], "away_away":[], "h2h":[]})
        home_form_wdl = wdl_from_rows(pair.get("home_home", []))
        away_form_wdl = wdl_from_rows(pair.get("away_away", []))
        h2h_home, h2h_away = wdl_from_h2h(pair.get("h2h", []))

        # Tiny model (demo blend)
        def estimate_expected_goals(pair_key, home_adv=0.15, blend=0.5):
            d = DEMO_FORM.get(pair_key, {})
            hh = d.get("home_home", []); aa = d.get("away_away", []); h2h = d.get("h2h", [])
            home_for = avg([r.get("gf", 0) for r in hh]) if hh else 1.2
            away_against = avg([r.get("ga", 0) for r in aa]) if aa else 1.0
            away_for = avg([r.get("gf", 0) for r in aa]) if aa else 1.0
            home_against = avg([r.get("ga", 0) for r in hh]) if hh else 1.0
            form_exp_home = (home_for + away_against) / 2.0
            form_exp_away = (away_for + home_against) / 2.0
            h2h_home_mean = avg([r.get("hg", 0) for r in h2h]) if h2h else 1.2
            h2h_away_mean = avg([r.get("ag", 0) for r in h2h]) if h2h else 1.0
            exp_home = blend * h2h_home_mean + (1 - blend) * form_exp_home + home_adv
            exp_away = blend * h2h_away_mean + (1 - blend) * form_exp_away
            return max(0.05, min(3.5, exp_home)), max(0.05, min(3.5, exp_away))

        expH, expA = estimate_expected_goals((home, away))
        pH, pD, pA = match_probs(expH, expA)
        result = {
            "expected_goals_home": round(expH, 3),
            "expected_goals_away": round(expA, 3),
            "prob_home": round(pH, 4),
            "prob_draw": round(pD, 4),
            "prob_away": round(pA, 4),
            "fair_home_odds": round(fair_odds(pH), 3),
            "fair_draw_odds": round(fair_odds(pD), 3),
            "fair_away_odds": round(fair_odds(pA), 3),
        }

        # --- Tabs with details ---
        tab_overview, tab_form, tab_news, tab_markets = st.tabs(["Overview", "Form & H2H", "Team news", "Markets"])

        with tab_overview:
            st.json(result, expanded=False)
            # export buttons
            j_bytes = json.dumps(result, indent=2).encode("utf-8")
            st.download_button("Download result JSON", data=j_bytes, file_name="smart_fixture_result.json", mime="application/json")

        with tab_form:
            st.write("{}".format(home))
            st.write(form_symbols(home_form_wdl) + "  (last 5)")
            st.write("{}".format(away))
            st.write(form_symbols(away_form_wdl) + "  (last 5)")
            st.write("Head-to-Head (last 5)")
            st.write("{} vs {}".format(home, away))
            st.write(form_symbols(h2h_home))
            st.write("{} vs {}".format(away, home))
            st.write(form_symbols(h2h_away))

        with tab_news:
            st.write("{} - team news".format(home))
       
