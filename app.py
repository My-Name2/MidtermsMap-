# ============================================
# STREAMLIT APP: US state map + state district map (Census shapes)
#   - Years: 2016 / 2020 / 2024
#   - State map colors: Pres margin OR Avg House margin
#   - State hover includes pres + avg house margin
#   - District hover includes:
#       * House Dem/Rep candidates + vote shares
#       * 2026 Cook / Sabato / Inside + toss-up agreement
#       * NEW: FEC campaign spending (receipts/disbursements/cash/debt) by party
# ============================================

import re, json
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

import plotly.express as px
import plotly.graph_objects as go

import geopandas as gpd
import streamlit as st


# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="US Elections Explorer", layout="wide")

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ----------------------------
# URLS (2026 ratings)
# ----------------------------
URL_CRYSTALBALL = "https://centerforpolitics.org/crystalball/2026-house/"
URL_INSIDE      = "https://insideelections.com/ratings/house"
URL_COOK_270    = "https://www.270towin.com/2026-house-election/index_show_table.php?map_title=cook-political-report-2026-house-ratings"
URL_SABATO_270  = "https://www.270towin.com/2026-house-election/table/crystal-ball-2026-house-forecast"
URL_INSIDE_270  = "https://www.270towin.com/2026-house-election/table/inside-elections-2026-house-ratings"

# ----------------------------
# DISTRICT SHAPES (Census cartographic boundary)
# ----------------------------
CD_ZIPS = {
    2016: "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_cd116_500k.zip",
    2020: "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_cd116_500k.zip",
    2024: "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_cd118_500k.zip",
}

STATE_FIPS = {
    "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11","FL":"12","GA":"13",
    "HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21","LA":"22","ME":"23","MD":"24","MA":"25",
    "MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36",
    "NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48",
    "UT":"49","VT":"50","VA":"51","WA":"53","WV":"54","WI":"55","WY":"56","PR":"72"
}

# ----------------------------
# HELPERS
# ----------------------------
DIST_RE = re.compile(r"\b([A-Z]{2}-(?:AL|\d{1,2}))\b", re.I)
RATING_KEYS = ["Likely Dem", "Leans Dem", "Tilt Dem", "Toss-up", "Tilt Rep", "Leans Rep", "Likely Rep"]

def safe_plot_col(series):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return s.apply(lambda x: None if (x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x)) else float(x))

def fmt_int(x):
    try:
        if pd.isna(x): return ""
        return f"{int(float(x)):,}"
    except Exception:
        return ""

def fmt_money(x):
    try:
        if pd.isna(x): return ""
        return f"${int(float(x)):,}"
    except Exception:
        return ""

def fmt_pct(x):
    try:
        if pd.isna(x): return ""
        return f"{float(x):.2%}"
    except Exception:
        return ""

def norm_dist_id(st, dist):
    st = (st or "").strip().upper()
    if pd.isna(dist):
        return f"{st}-AL"
    try:
        d = int(float(dist))
    except Exception:
        d = None
    if d is None or d == 0:
        return f"{st}-AL"
    return f"{st}-{d}"

def cand_join(names):
    names = [n for n in names if n and str(n).strip()]
    names = [str(n).strip() for n in names]
    names = [n for n in names if n.upper() != "NAN"]
    names = sorted(set(names))
    return " / ".join(names[:3]) if names else ""

def normalize_rating_label(s):
    s = str(s).strip().lower().replace("toss up", "toss-up")
    s = re.sub(r"\s+", " ", s)
    return s.title().replace("Toss Up", "Toss-up")

def is_tossup(x):
    return normalize_rating_label(x) == "Toss-up"

def party_to_simple(party_str: str) -> str:
    p = (party_str or "").strip().lower()
    # catch "Democratic-Farmer-Labor" variants too
    if "rep" in p:
        return "REPUBLICAN"
    if "dem" in p or "dfl" in p:
        return "DEMOCRAT"
    return "OTHER"

def fec_district_to_district_id(district_code: str) -> str:
    # examples: "AL-01", "AK-00", "PR-00"
    dc = (district_code or "").strip().upper()
    if not dc or "-" not in dc:
        return ""
    st, d = dc.split("-", 1)
    d = d.strip()
    if d == "00":
        return f"{st}-AL"
    try:
        return f"{st}-{int(d)}"
    except Exception:
        return f"{st}-{d}"


# ----------------------------
# HTML FETCH
# ----------------------------
@st.cache_data(show_spinner=False, ttl=6*60*60)
def fetch_html(url, timeout=30):
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        if r.status_code in (401, 403):
            return ""
        r.raise_for_status()
        return r.text
    except Exception:
        return ""


# ----------------------------
# RATINGS PARSERS
# ----------------------------
@st.cache_data(show_spinner=False, ttl=6*60*60)
def parse_270toWin_table_like(url):
    html = fetch_html(url)
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    tokens = [t.strip() for t in soup.get_text("\n").split("\n")]
    tokens = [t for t in tokens if t]

    current = None
    out = {}
    for t in tokens:
        t_norm = normalize_rating_label(t)
        for rk in RATING_KEYS:
            if t_norm.startswith(rk):
                current = rk
                break
        if not current:
            continue
        m = DIST_RE.search(t.upper())
        if m:
            out[m.group(1).upper()] = current
    return out

@st.cache_data(show_spinner=False, ttl=6*60*60)
def parse_centerforpolitics_crystalball(url):
    _ = fetch_html(url)
    return {}  # often image-only; use 270toWin fallback

@st.cache_data(show_spinner=False, ttl=6*60*60)
def parse_insideelections_house(url):
    html = fetch_html(url)
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    if not DIST_RE.search(text.upper()):
        return {}
    out = {}
    for line in text.split("\n"):
        m = DIST_RE.search(line.upper())
        if not m:
            continue
        did = m.group(1).upper()
        lnorm = normalize_rating_label(line)
        found = None
        for rk in RATING_KEYS:
            if rk.lower() in lnorm.lower():
                found = rk
                break
        if found:
            out[did] = found
    return out

@st.cache_data(show_spinner=False, ttl=6*60*60)
def get_2026_ratings_maps():
    cook_map = parse_270toWin_table_like(URL_COOK_270)

    sabato_map = parse_centerforpolitics_crystalball(URL_CRYSTALBALL)
    if not sabato_map:
        sabato_map = parse_270toWin_table_like(URL_SABATO_270)

    inside_map = parse_insideelections_house(URL_INSIDE)  # may 403
    if not inside_map:
        inside_map = parse_270toWin_table_like(URL_INSIDE_270)

    return cook_map, sabato_map, inside_map


# ----------------------------
# HOUSE LOADER (wrapped quote lines)
# ----------------------------
def load_house_wrapped_quotes_csv(path):
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        header_line = f.readline().strip().lstrip("\ufeff")
        header = [h.strip() for h in header_line.split(",")]
        cand_idx = header.index("candidate") if "candidate" in header else None

        rows = []
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith('"') and ln.endswith('"'):
                ln = ln[1:-1]
            parts = ln.split(",")

            if cand_idx is not None and len(parts) > len(header):
                extra = len(parts) - len(header)
                candidate_merged = ",".join(parts[cand_idx : cand_idx + extra + 1])
                parts = parts[:cand_idx] + [candidate_merged] + parts[cand_idx + extra + 1 :]

            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            if len(parts) > len(header):
                parts = parts[:len(header)]

            rows.append(parts)

    df = pd.DataFrame(rows, columns=header)
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
    return df


# ----------------------------
# LOAD DATA (from user-provided paths)
# ----------------------------
@st.cache_data(show_spinner=True)
def load_inputs(pres_path, house_path):
    pres_df = pd.read_csv(pres_path, low_memory=False)
    pres_df.columns = pres_df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

    try:
        house_df_try = pd.read_csv(house_path, low_memory=False)
        house_df_try.columns = house_df_try.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
        y = pd.to_numeric(house_df_try.get("year", pd.Series(dtype="object")), errors="coerce")
        if y.notna().sum() == 0:
            raise ValueError("House year didn't parse with normal read.")
        house_df = house_df_try
    except Exception:
        house_df = load_house_wrapped_quotes_csv(house_path)

    # Normalize president
    pres_df["year"] = pd.to_numeric(pres_df.get("year", pd.Series(dtype="object")), errors="coerce")
    pres_df["party_simplified"] = pres_df.get("party_simplified", "").astype(str).str.strip().str.upper()
    pres_df["state_po"] = pres_df.get("state_po", "").astype(str).str.strip().str.upper()
    pres_df["candidatevotes"] = pd.to_numeric(pres_df.get("candidatevotes", pd.Series(dtype="object")), errors="coerce")

    pres_cand_col = "candidate" if "candidate" in pres_df.columns else None
    if pres_cand_col:
        pres_df[pres_cand_col] = pres_df[pres_cand_col].fillna("").astype(str).str.strip()

    # Normalize house
    house_df["year"] = pd.to_numeric(house_df.get("year", pd.Series(dtype="object")), errors="coerce")
    for c in ["office", "stage", "party", "state_po", "candidate"]:
        if c in house_df.columns:
            house_df[c] = house_df[c].fillna("").astype(str).str.strip().str.upper()
    house_df["candidatevotes"] = pd.to_numeric(house_df.get("candidatevotes", pd.Series(dtype="object")), errors="coerce")

    return pres_df, pres_cand_col, house_df


# ----------------------------
# LOAD FEC SPENDING (xlsx) + AGGREGATE
# ----------------------------
@st.cache_data(show_spinner=True)
def load_fec_spending(spending_path: str) -> pd.DataFrame:
    p = (spending_path or "").strip()
    if not p:
        return pd.DataFrame()

    # Expect your uploaded workbook format
    try:
        df = pd.read_excel(p, sheet_name="House_Candidate_Spending", engine="openpyxl")
    except Exception:
        # Try first sheet as fallback
        df = pd.read_excel(p, sheet_name=0, engine="openpyxl")

    # normalize
    for c in ["cycle_year", "state_abbrev", "district_code", "candidate", "party"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    if "cycle_year" in df.columns:
        df["cycle_year"] = pd.to_numeric(df["cycle_year"], errors="coerce")

    for c in ["receipts", "disbursements", "cash_on_hand", "debts", "indiv_contrib", "pac_contrib", "cand_loans_contrib"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # build keys consistent with your app
    df["state_po"] = df.get("state_abbrev", "").astype(str).str.strip().str.upper()
    df["district_id"] = df.get("district_code", "").apply(fec_district_to_district_id)

    df["party_simple"] = df.get("party", "").apply(party_to_simple)

    return df


def compute_fec_spending_district(fec_df: pd.DataFrame, year: int) -> pd.DataFrame:
    if fec_df.empty:
        return pd.DataFrame()

    d = fec_df[fec_df["cycle_year"] == year].copy()
    if d.empty:
        return pd.DataFrame()

    d = d[(d["state_po"].str.len() == 2) & (d["district_id"].str.len() >= 4)]
    dmaj = d[d["party_simple"].isin(["DEMOCRAT", "REPUBLICAN"])].copy()
    if dmaj.empty:
        return pd.DataFrame()

    metric_cols = ["receipts", "disbursements", "cash_on_hand", "debts", "indiv_contrib", "pac_contrib", "cand_loans_contrib"]

    # sum by party within district
    agg = (
        dmaj.groupby(["state_po", "district_id", "party_simple"], dropna=False)[metric_cols]
        .sum()
        .reset_index()
    )

    piv = agg.pivot_table(index=["state_po", "district_id"], columns="party_simple", values=metric_cols, aggfunc="sum", fill_value=0)

    # flatten columns -> dem_receipts / rep_receipts etc
    out = piv.copy()
    out.columns = [f"{('dem' if party=='DEMOCRAT' else 'rep')}_{col}" for col, party in out.columns]
    out = out.reset_index()

    # pick “top” candidate per party (by receipts) for labeling
    top = (
        dmaj.sort_values("receipts", ascending=False)
        .groupby(["state_po", "district_id", "party_simple"], dropna=False)
        .head(1)[["state_po", "district_id", "party_simple", "candidate"]]
    )
    top_piv = top.pivot_table(index=["state_po","district_id"], columns="party_simple", values="candidate", aggfunc="first", fill_value="")
    top_piv = top_piv.reset_index().rename(columns={"DEMOCRAT":"fec_dem_candidate", "REPUBLICAN":"fec_rep_candidate"})
    out = out.merge(top_piv, on=["state_po","district_id"], how="left")

    # margins (Rep - Dem) / (Rep + Dem)
    def margin(rep, dem):
        denom = (rep + dem)
        return np.where(denom == 0, np.nan, (rep - dem) / denom)

    out["fec_receipts_margin"] = margin(out.get("rep_receipts", 0), out.get("dem_receipts", 0))
    out["fec_disb_margin"]     = margin(out.get("rep_disbursements", 0), out.get("dem_disbursements", 0))

    # formatted strings
    for c in ["dem_receipts","rep_receipts","dem_disbursements","rep_disbursements","dem_cash_on_hand","rep_cash_on_hand","dem_debts","rep_debts"]:
        if c in out.columns:
            out[c + "_str"] = out[c].map(fmt_money)

    out["fec_receipts_margin_str"] = pd.Series(out["fec_receipts_margin"]).map(fmt_pct)
    out["fec_disb_margin_str"]     = pd.Series(out["fec_disb_margin"]).map(fmt_pct)

    for c in ["fec_dem_candidate","fec_rep_candidate","fec_receipts_margin_str","fec_disb_margin_str"]:
        if c in out.columns:
            out[c] = out[c].fillna("").astype(str)

    return out


def compute_fec_spending_state(fec_dist: pd.DataFrame) -> pd.DataFrame:
    if fec_dist.empty:
        return pd.DataFrame()

    # sum across districts
    cols_sum = []
    for c in ["dem_receipts","rep_receipts","dem_disbursements","rep_disbursements","dem_cash_on_hand","rep_cash_on_hand","dem_debts","rep_debts"]:
        if c in fec_dist.columns:
            cols_sum.append(c)

    st_agg = fec_dist.groupby("state_po", dropna=False)[cols_sum].sum().reset_index()

    # margins at state level
    def margin(rep, dem):
        denom = (rep + dem)
        return np.where(denom == 0, np.nan, (rep - dem) / denom)

    st_agg["fec_state_receipts_margin"] = margin(st_agg.get("rep_receipts",0), st_agg.get("dem_receipts",0))
    st_agg["fec_state_disb_margin"]     = margin(st_agg.get("rep_disbursements",0), st_agg.get("dem_disbursements",0))

    # formatted
    for c in cols_sum:
        st_agg[c + "_str"] = st_agg[c].map(fmt_money)
    st_agg["fec_state_receipts_margin_str"] = pd.Series(st_agg["fec_state_receipts_margin"]).map(fmt_pct)
    st_agg["fec_state_disb_margin_str"]     = pd.Series(st_agg["fec_state_disb_margin"]).map(fmt_pct)

    return st_agg


# ----------------------------
# COMPUTATIONS (pres/house)
# ----------------------------
def compute_pres_state_results(pres_df, pres_cand_col, year):
    df = pres_df[
        (pres_df["year"] == year) &
        (pres_df["state_po"].notna()) &
        (pres_df["candidatevotes"].notna())
    ].copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "state_po","pres_margin","pres_dem_candidate","pres_rep_candidate",
            "pres_dem_votes","pres_rep_votes","pres_total_votes_all",
            "pres_dem_pct_all","pres_rep_pct_all"
        ])

    tot_all = df.groupby("state_po")["candidatevotes"].sum().rename("pres_total_votes_all").reset_index()
    maj = df[df["party_simplified"].isin(["DEMOCRAT","REPUBLICAN"])].copy()

    pv = maj.groupby(["state_po","party_simplified"])["candidatevotes"].sum().unstack(fill_value=0)
    if "DEMOCRAT" not in pv.columns: pv["DEMOCRAT"] = 0
    if "REPUBLICAN" not in pv.columns: pv["REPUBLICAN"] = 0
    pv = pv.reset_index().rename(columns={"DEMOCRAT":"pres_dem_votes","REPUBLICAN":"pres_rep_votes"})

    if pres_cand_col:
        dem_names = maj[maj["party_simplified"]=="DEMOCRAT"].groupby("state_po")[pres_cand_col].apply(lambda s: cand_join(s.tolist())).rename("pres_dem_candidate").reset_index()
        rep_names = maj[maj["party_simplified"]=="REPUBLICAN"].groupby("state_po")[pres_cand_col].apply(lambda s: cand_join(s.tolist())).rename("pres_rep_candidate").reset_index()
    else:
        dem_names = pd.DataFrame({"state_po": pv["state_po"], "pres_dem_candidate": ""})
        rep_names = pd.DataFrame({"state_po": pv["state_po"], "pres_rep_candidate": ""})

    out = pv.merge(tot_all, on="state_po", how="left").merge(dem_names, on="state_po", how="left").merge(rep_names, on="state_po", how="left")

    out["pres_dem_pct_all"] = out["pres_dem_votes"] / out["pres_total_votes_all"].replace(0, np.nan)
    out["pres_rep_pct_all"] = out["pres_rep_votes"] / out["pres_total_votes_all"].replace(0, np.nan)

    major_total = (out["pres_dem_votes"] + out["pres_rep_votes"]).replace(0, np.nan)
    out["pres_margin"] = (out["pres_rep_votes"] - out["pres_dem_votes"]) / major_total

    for c in ["pres_dem_pct_all","pres_rep_pct_all","pres_margin"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return out[[
        "state_po","pres_margin",
        "pres_dem_candidate","pres_rep_candidate",
        "pres_dem_votes","pres_rep_votes","pres_total_votes_all",
        "pres_dem_pct_all","pres_rep_pct_all"
    ]]

def compute_house_district_results(house_df, year):
    df = house_df[
        (house_df["year"] == year) &
        (house_df["office"] == "US HOUSE") &
        (house_df["stage"] == "GEN") &
        (house_df["candidatevotes"].notna())
    ].copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "state_po","district","district_id",
            "dem_candidate","rep_candidate",
            "dem_votes","rep_votes","total_votes_all",
            "dem_pct_all","rep_pct_all","house_margin"
        ])

    totals_all = df.groupby(["state_po","district"], dropna=False)["candidatevotes"].sum().rename("total_votes_all").reset_index()

    dmaj = df[df["party"].isin(["DEMOCRAT","REPUBLICAN"])].copy()
    if dmaj.empty:
        out = totals_all.copy()
        out["district_id"] = out.apply(lambda r: norm_dist_id(r["state_po"], r["district"]), axis=1)
        for c in ["dem_candidate","rep_candidate","dem_votes","rep_votes","dem_pct_all","rep_pct_all","house_margin"]:
            out[c] = np.nan
        return out

    pv = dmaj.groupby(["state_po","district","party"], dropna=False)["candidatevotes"].sum().unstack(fill_value=0)
    if "DEMOCRAT" not in pv.columns: pv["DEMOCRAT"] = 0
    if "REPUBLICAN" not in pv.columns: pv["REPUBLICAN"] = 0
    pv = pv.reset_index().rename(columns={"DEMOCRAT":"dem_votes","REPUBLICAN":"rep_votes"})

    dem_names = dmaj[dmaj["party"]=="DEMOCRAT"].groupby(["state_po","district"], dropna=False)["candidate"].apply(lambda s: cand_join(s.tolist())).rename("dem_candidate").reset_index()
    rep_names = dmaj[dmaj["party"]=="REPUBLICAN"].groupby(["state_po","district"], dropna=False)["candidate"].apply(lambda s: cand_join(s.tolist())).rename("rep_candidate").reset_index()

    out = pv.merge(totals_all, on=["state_po","district"], how="left")
    out = out.merge(dem_names, on=["state_po","district"], how="left").merge(rep_names, on=["state_po","district"], how="left")

    out["district_id"] = out.apply(lambda r: norm_dist_id(r["state_po"], r["district"]), axis=1)

    out["dem_pct_all"] = out["dem_votes"] / out["total_votes_all"].replace(0, np.nan)
    out["rep_pct_all"] = out["rep_votes"] / out["total_votes_all"].replace(0, np.nan)

    major_total = (out["dem_votes"] + out["rep_votes"]).replace(0, np.nan)
    out["house_margin"] = (out["rep_votes"] - out["dem_votes"]) / major_total

    for c in ["dem_pct_all","rep_pct_all","house_margin"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return out[[
        "state_po","district","district_id",
        "dem_candidate","rep_candidate",
        "dem_votes","rep_votes","total_votes_all",
        "dem_pct_all","rep_pct_all","house_margin"
    ]]

def compute_house_state_avg(house_df, year):
    ddf = compute_house_district_results(house_df, year)
    if ddf.empty:
        return ddf, pd.DataFrame(columns=["state_po","avg_house_margin"])
    avg = ddf.groupby("state_po")["house_margin"].mean().rename("avg_house_margin").reset_index()
    avg["avg_house_margin"] = pd.to_numeric(avg["avg_house_margin"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return ddf, avg

def attach_ratings(ddf, cook_map, sabato_map, inside_map):
    if ddf.empty:
        return ddf
    d = ddf.copy()
    d["Cook_2026"]   = d["district_id"].map(cook_map).fillna("")
    d["Sabato_2026"] = d["district_id"].map(sabato_map).fillna("")
    d["Inside_2026"] = d["district_id"].map(inside_map).fillna("")
    d["tossup_agree_count"] = (
        d["Cook_2026"].apply(is_tossup).astype(int) +
        d["Sabato_2026"].apply(is_tossup).astype(int) +
        d["Inside_2026"].apply(is_tossup).astype(int)
    )
    return d


# ----------------------------
# SHAPES (cached download + read)
# ----------------------------
def _download_cached(url, cache_path: Path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    r = requests.get(url, headers=UA, timeout=120)
    r.raise_for_status()
    cache_path.write_bytes(r.content)
    return cache_path

@st.cache_data(show_spinner=True)
def load_state_cd_geojson(year, state_po, cache_dir="district_shapes_cache"):
    state_po = state_po.upper().strip()
    if state_po not in STATE_FIPS:
        raise ValueError(f"Unknown state_po: {state_po}")

    url = CD_ZIPS.get(year)
    if not url:
        raise ValueError(f"No Census district shapes configured for year={year}")

    cache_dir = Path(cache_dir)
    zip_path = cache_dir / f"cd_shapes_{year}.zip"
    _download_cached(url, zip_path)

    gdf = gpd.read_file(f"zip://{zip_path}")

    cd_cols = [c for c in gdf.columns if re.match(r"^CD\d+FP$", str(c))]
    if not cd_cols:
        raise ValueError(f"Could not find district FP column. Columns: {list(gdf.columns)}")
    cd_col = cd_cols[0]

    if "STATEFP" not in gdf.columns:
        raise ValueError(f"Could not find STATEFP. Columns: {list(gdf.columns)}")

    stfp = STATE_FIPS[state_po]
    gdf = gdf[gdf["STATEFP"].astype(str).str.zfill(2) == stfp].copy()
    if gdf.empty:
        raise ValueError("No geometries for this state in the Census shapefile.")

    gdf[cd_col] = gdf[cd_col].astype(str).str.zfill(2)

    def mk_id(fp):
        return f"{state_po}-AL" if fp == "00" else f"{state_po}-{int(fp)}"

    gdf["district_id"] = gdf[cd_col].map(mk_id)

    gdf = gdf[gdf.geometry.notna()].copy()
    try:
        gdf["geometry"] = gdf.geometry.buffer(0)
    except Exception:
        pass

    geojson = json.loads(gdf.to_json())
    return geojson, gdf


# ----------------------------
# BUILD ALL YEAR DATA ONCE
# ----------------------------
@st.cache_data(show_spinner=True)
def build_year_data(pres_path, house_path, spending_path):
    pres_df, pres_cand_col, house_df = load_inputs(pres_path, house_path)
    cook_map, sabato_map, inside_map = get_2026_ratings_maps()

    fec_df = load_fec_spending(spending_path)

    YEARS = [2016, 2020, 2024]
    year_data = {}

    for y in YEARS:
        pres_state = compute_pres_state_results(pres_df, pres_cand_col, y)
        dist_year, house_avg = compute_house_state_avg(house_df, y)
        dist_year = attach_ratings(dist_year, cook_map, sabato_map, inside_map)

        # ---- attach spending (district + state totals) ----
        fec_dist = compute_fec_spending_district(fec_df, y)
        fec_state = compute_fec_spending_state(fec_dist) if not fec_dist.empty else pd.DataFrame()

        if not fec_dist.empty:
            dist_year = dist_year.merge(
                fec_dist.drop(columns=["state_po"], errors="ignore"),
                on="district_id",
                how="left"
            )

        sdf = pres_state.merge(house_avg, on="state_po", how="outer").sort_values("state_po").reset_index(drop=True)

        if not fec_state.empty:
            sdf = sdf.merge(fec_state, on="state_po", how="left")

        # pretty strings
        if not sdf.empty:
            sdf["pres_dem_votes_str"] = sdf.get("pres_dem_votes", np.nan).map(fmt_int)
            sdf["pres_rep_votes_str"] = sdf.get("pres_rep_votes", np.nan).map(fmt_int)
            sdf["pres_total_votes_all_str"] = sdf.get("pres_total_votes_all", np.nan).map(fmt_int)
            sdf["pres_dem_pct_all_str"] = sdf.get("pres_dem_pct_all", np.nan).map(fmt_pct)
            sdf["pres_rep_pct_all_str"] = sdf.get("pres_rep_pct_all", np.nan).map(fmt_pct)
            sdf["pres_margin_str"] = sdf.get("pres_margin", np.nan).map(fmt_pct)
            sdf["avg_house_margin_str"] = sdf.get("avg_house_margin", np.nan).map(fmt_pct)

            for c in [
                "pres_dem_candidate","pres_rep_candidate",
                "pres_dem_votes_str","pres_rep_votes_str","pres_total_votes_all_str",
                "pres_dem_pct_all_str","pres_rep_pct_all_str",
                "pres_margin_str","avg_house_margin_str",
                # spending strings (state)
                "dem_receipts_str","rep_receipts_str","dem_disbursements_str","rep_disbursements_str",
                "dem_cash_on_hand_str","rep_cash_on_hand_str","dem_debts_str","rep_debts_str",
                "fec_state_receipts_margin_str","fec_state_disb_margin_str",
            ]:
                if c in sdf.columns:
                    sdf[c] = sdf[c].fillna("").astype(str)

        year_data[y] = {"state_df": sdf, "dist_df": dist_year}

    # Toss-up table (prefer 2024 then 2020 then 2016)
    pref_year = 2024 if not year_data[2024]["dist_df"].empty else (2020 if not year_data[2020]["dist_df"].empty else 2016)
    dist_for_toss = year_data[pref_year]["dist_df"]
    if not dist_for_toss.empty:
        base_cols = [
            "district_id",
            "dem_candidate","rep_candidate",
            "dem_votes","rep_votes","total_votes_all",
            "dem_pct_all","rep_pct_all",
            "Cook_2026","Sabato_2026","Inside_2026",
            "tossup_agree_count","house_margin"
        ]
        # add spending cols if present
        spend_cols = []
        for c in ["dem_receipts","rep_receipts","fec_receipts_margin","dem_disbursements","rep_disbursements","fec_disb_margin"]:
            if c in dist_for_toss.columns:
                spend_cols.append(c)

        tossup_table = (
            dist_for_toss.loc[dist_for_toss["tossup_agree_count"] > 0, base_cols + spend_cols]
            .sort_values(["tossup_agree_count","district_id"], ascending=[False, True])
            .reset_index(drop=True)
        )
    else:
        tossup_table = pd.DataFrame()

    return year_data, tossup_table


# ----------------------------
# PLOTTERS
# ----------------------------
def make_state_map_figure(sdf, year, metric_col):
    if sdf.empty:
        return None

    if metric_col == "pres_margin":
        if "pres_margin" not in sdf.columns or sdf["pres_margin"].notna().sum() == 0:
            metric_col = "avg_house_margin"

    sdf = sdf.copy()
    sdf["_plot_val"] = safe_plot_col(sdf.get(metric_col, pd.Series([None]*len(sdf))))

    arr = pd.to_numeric(sdf["_plot_val"], errors="coerce")
    zmax = float(np.nanmax(np.abs(arr.values))) if np.isfinite(arr).any() else 0.5
    if not np.isfinite(zmax) or zmax == 0:
        zmax = 0.5

    title = f"{year} Presidential Margin by State (Rep - Dem)" if metric_col == "pres_margin" else f"{year} Avg House Margin by State (Rep - Dem)"
    subtitle = "blue = Dem, red = Rep"

    def col_or_blank(c):
        return sdf[c].fillna("").astype(str) if c in sdf.columns else pd.Series([""]*len(sdf))

    st_ = col_or_blank("state_po")
    dname = col_or_blank("pres_dem_candidate")
    rname = col_or_blank("pres_rep_candidate")
    dv = col_or_blank("pres_dem_votes_str")
    rv = col_or_blank("pres_rep_votes_str")
    dt = col_or_blank("pres_dem_pct_all_str")
    rt = col_or_blank("pres_rep_pct_all_str")
    pm = col_or_blank("pres_margin_str")
    hm = col_or_blank("avg_house_margin_str")

    fig = go.Figure(
        data=[
            go.Choropleth(
                locations=sdf["state_po"],
                locationmode="USA-states",
                z=sdf["_plot_val"],
                zmin=-zmax, zmax=zmax,
                colorscale="RdBu_r",
                colorbar_title=("Pres margin" if metric_col=="pres_margin" else "Avg House margin"),
                customdata=np.stack([st_, dname, dv, dt, rname, rv, rt, pm, hm], axis=1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Pres (D): %{customdata[1]} — %{customdata[2]} (%{customdata[3]})<br>"
                    "Pres (R): %{customdata[4]} — %{customdata[5]} (%{customdata[6]})<br>"
                    "Pres margin (R-D): %{customdata[7]}<br>"
                    "Avg House margin (R-D): %{customdata[8]}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title_text=f"{title} — {subtitle}",
        geo=dict(scope="usa", projection_type="albers usa"),
        margin=dict(l=0, r=0, t=50, b=0),
        height=520
    )
    return fig

def make_district_map_figure(state_po, year, sub):
    geojson, gdf = load_state_cd_geojson(year, state_po)

    # merge geometry district ids with sub data
    m = gdf[["district_id"]].merge(sub, on="district_id", how="left")

    m["house_margin_plot"] = safe_plot_col(m["house_margin"])
    arr = pd.to_numeric(m["house_margin_plot"], errors="coerce")
    zmax = float(np.nanmax(np.abs(arr.values))) if np.isfinite(arr).any() else 0.5
    if not np.isfinite(zmax) or zmax == 0:
        zmax = 0.5

    hover_data = {
        "district_id": True,
        "house_margin_plot":":.2%",
        "dem_candidate": True,
        "dem_votes_str": True,
        "dem_pct_all_str": True,
        "rep_candidate": True,
        "rep_votes_str": True,
        "rep_pct_all_str": True,
        "total_votes_str": True,
        "Cook_2026": True,
        "Sabato_2026": True,
        "Inside_2026": True,
        "tossup_agree_count": True,
    }

    # add spending hover fields if present
    for c in [
        "dem_receipts_str","rep_receipts_str","fec_receipts_margin_str",
        "dem_disbursements_str","rep_disbursements_str","fec_disb_margin_str",
        "dem_cash_on_hand_str","rep_cash_on_hand_str",
        "dem_debts_str","rep_debts_str",
        "fec_dem_candidate","fec_rep_candidate",
    ]:
        if c in m.columns:
            hover_data[c] = True

    fig2 = px.choropleth(
        m,
        geojson=geojson,
        locations="district_id",
        featureidkey="properties.district_id",
        color="house_margin_plot",
        color_continuous_scale="RdBu_r",
        range_color=(-zmax, zmax),
        title=f"{state_po} — {year} House margin by district + 2026 ratings + FEC spending (hover)",
        hover_data=hover_data,
        scope="usa",
    )
    fig2.update_geos(fitbounds="locations", visible=False)
    fig2.update_layout(margin=dict(l=0, r=0, t=60, b=0), height=520)
    return fig2


# ----------------------------
# SIDEBAR: FILE PATHS + CONTROLS
# ----------------------------
st.sidebar.header("Inputs")
default_pres = "1976-2024-president-extended.csv"
default_house = "1976-2024-house (1).tab"
default_spending = "fec_house_campaign_spending_2016_2018_2020_2022_2024.xlsx"

pres_path = st.sidebar.text_input("Presidential CSV path", value=default_pres)
house_path = st.sidebar.text_input("House TAB/CSV path", value=default_house)
spending_path = st.sidebar.text_input("FEC spending XLSX path (optional)", value=default_spending)

st.sidebar.divider()

YEARS = [2016, 2020, 2024]
year = st.sidebar.radio("Year", YEARS, index=0)

metric_label = st.sidebar.radio("State map colors", ["Pres margin", "Avg House margin"], index=0)
metric_col = "pres_margin" if metric_label == "Pres margin" else "avg_house_margin"

# Load everything once paths are provided
try:
    year_data, tossup_table = build_year_data(pres_path, house_path, spending_path)
except Exception as e:
    st.error("Failed to load/parse your input files. Check the paths and file formats.")
    st.exception(e)
    st.stop()

sdf = year_data[year]["state_df"]
if sdf.empty:
    st.error("No state-level data for the selected year.")
    st.stop()

states = sorted([s for s in sdf["state_po"].dropna().unique().tolist() if isinstance(s, str) and len(s)==2])
state_po = st.sidebar.selectbox("State", states, index=0)

# ----------------------------
# MAIN UI
# ----------------------------
st.title("US Elections Explorer (2016 / 2020 / 2024)")

left, right = st.columns([1.15, 1.0], gap="large")

with left:
    st.subheader("State map")
    fig = make_state_map_figure(sdf, year, metric_col)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader(f"{state_po} summary ({year})")
    row = sdf[sdf["state_po"] == state_po]
    if row.empty:
        st.info("No row for this state/year.")
    else:
        r0 = row.iloc[0]

        # base summary
        st.markdown(
            f"""
**Pres (D):** {r0.get("pres_dem_candidate","")} — {r0.get("pres_dem_votes_str","")} ({r0.get("pres_dem_pct_all_str","")})  
**Pres (R):** {r0.get("pres_rep_candidate","")} — {r0.get("pres_rep_votes_str","")} ({r0.get("pres_rep_pct_all_str","")})  
**Pres margin (Rep − Dem):** {r0.get("pres_margin_str","N/A")}  
**Avg House margin (Rep − Dem):** {r0.get("avg_house_margin_str","N/A")}
            """.strip()
        )

        # spending summary (state totals) if available
        if "dem_receipts_str" in sdf.columns or "rep_receipts_str" in sdf.columns:
            st.markdown("**FEC House spending (state total, major parties):**")
            st.markdown(
                f"""
- **Receipts:** D {r0.get("dem_receipts_str","")} | R {r0.get("rep_receipts_str","")} | **Margin (R−D):** {r0.get("fec_state_receipts_margin_str","")}  
- **Disbursements:** D {r0.get("dem_disbursements_str","")} | R {r0.get("rep_disbursements_str","")} | **Margin (R−D):** {r0.get("fec_state_disb_margin_str","")}  
- **Cash on hand:** D {r0.get("dem_cash_on_hand_str","")} | R {r0.get("rep_cash_on_hand_str","")}  
- **Debts:** D {r0.get("dem_debts_str","")} | R {r0.get("rep_debts_str","")}
                """.strip()
            )
        else:
            st.caption("FEC spending not loaded (leave blank or point to the XLSX in your repo).")

st.divider()

st.subheader(f"{state_po} districts ({year})")

ddf = year_data[year]["dist_df"]
if ddf.empty:
    st.info("No district-level House results for this year.")
    st.stop()

sub = ddf[ddf["state_po"] == state_po].copy()
if sub.empty:
    st.info("No districts found for this state/year.")
    st.stop()

def sort_key(did):
    if str(did).endswith("-AL"):
        return (-1, 0)
    try:
        return (0, int(str(did).split("-")[1]))
    except Exception:
        return (0, 999)

sub["k"] = sub["district_id"].apply(sort_key)
sub = sub.sort_values("k").drop(columns=["k"]).reset_index(drop=True)

# format vote strings
sub["dem_votes_str"] = sub["dem_votes"].map(fmt_int)
sub["rep_votes_str"] = sub["rep_votes"].map(fmt_int)
sub["total_votes_str"] = sub["total_votes_all"].map(fmt_int)
sub["dem_pct_all_str"] = sub["dem_pct_all"].map(fmt_pct)
sub["rep_pct_all_str"] = sub["rep_pct_all"].map(fmt_pct)

# format spending strings (if present)
for c in ["dem_receipts","rep_receipts","dem_disbursements","rep_disbursements","dem_cash_on_hand","rep_cash_on_hand","dem_debts","rep_debts"]:
    if c in sub.columns:
        sub[c + "_str"] = sub[c].map(fmt_money)
if "fec_receipts_margin" in sub.columns:
    sub["fec_receipts_margin_str"] = sub["fec_receipts_margin"].map(fmt_pct)
if "fec_disb_margin" in sub.columns:
    sub["fec_disb_margin_str"] = sub["fec_disb_margin"].map(fmt_pct)

# District map
try:
    fig2 = make_district_map_figure(state_po, year, sub)
    st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.warning("District map unavailable (could not load Census district shapes or plot them).")
    st.exception(e)

# District table
show_cols = [
    "district_id",
    "dem_candidate","dem_votes","dem_pct_all",
    "rep_candidate","rep_votes","rep_pct_all",
    "total_votes_all",
    "house_margin",
    "Cook_2026","Sabato_2026","Inside_2026","tossup_agree_count"
]

# add spending columns to the table if present
for c in [
    "dem_receipts","rep_receipts","fec_receipts_margin",
    "dem_disbursements","rep_disbursements","fec_disb_margin",
    "dem_cash_on_hand","rep_cash_on_hand",
    "dem_debts","rep_debts",
]:
    if c in sub.columns:
        show_cols.append(c)

show = sub[show_cols].copy()

# format
for c in ["dem_votes","rep_votes","total_votes_all"]:
    if c in show.columns: show[c] = show[c].map(fmt_int)
for c in ["dem_pct_all","rep_pct_all","house_margin","fec_receipts_margin","fec_disb_margin"]:
    if c in show.columns: show[c] = show[c].map(fmt_pct)
for c in ["dem_receipts","rep_receipts","dem_disbursements","rep_disbursements","dem_cash_on_hand","rep_cash_on_hand","dem_debts","rep_debts"]:
    if c in show.columns: show[c] = show[c].map(fmt_money)

st.dataframe(show, use_container_width=True, height=460)

# Toss-up table filtered to state
st.subheader("Toss-ups (filtered to this state)")
if isinstance(tossup_table, pd.DataFrame) and not tossup_table.empty:
    st_toss = tossup_table[tossup_table["district_id"].str.startswith(state_po + "-", na=False)].copy()
    if st_toss.empty:
        st.info("No toss-up districts (by your 3 sources) found for this state in the scraped tables.")
    else:
        st_toss_disp = st_toss.copy()
        for c in ["dem_votes","rep_votes","total_votes_all"]:
            if c in st_toss_disp.columns: st_toss_disp[c] = st_toss_disp[c].map(fmt_int)
        for c in ["dem_pct_all","rep_pct_all","house_margin","fec_receipts_margin","fec_disb_margin"]:
            if c in st_toss_disp.columns: st_toss_disp[c] = st_toss_disp[c].map(fmt_pct)
        for c in ["dem_receipts","rep_receipts","dem_disbursements","rep_disbursements"]:
            if c in st_toss_disp.columns: st_toss_disp[c] = st_toss_disp[c].map(fmt_money)

        st.dataframe(st_toss_disp, use_container_width=True, height=280)
else:
    st.info("No toss-up table available (ratings scrape returned no districts).")

st.caption(
    "Notes: District spending is aggregated from the FEC candidate spending workbook by cycle year, district, and major party. "
    "Margins are (Rep−Dem)/(Rep+Dem). Census district shapes are cached locally."
)
