"""
FPL Weekly Assistant — single-file Streamlit app (Thai Updated Version)

What it does
- Pulls live data from FPL API (bootstrap-static, fixtures, entry picks)
- Engineers features (recent form, xGI proxy, minutes reliability, fixture difficulty, photo_url)
- Predicts next GW points with a hybrid approach
- Optimizes your Starting XI & bench order
- Suggests transfers based on selected strategy
- **NEW**: Home Dashboard v1.9.6 (Added image fallbacks for missing player photos)
- **NEW**: Visual Fixture Planner with Logos (v1.9.3)
- Displays Starting XI in a "Pitch View" or "List View"
- Includes a "Simulation Mode" to manually edit your 15-man squad

How to run
1) pip install streamlit pandas numpy scikit-learn pulp requests altair
2) streamlit run fpl.py

Notes
- This app reads public FPL endpoints. No login required.
"""
###############################
# V1.9.6 - Added Image Fallback
###############################

import os
import math
import json
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt # <-- NEW IMPORT
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, LpStatus, PULP_CBC_CMD

###############################
# Fav icon
###############################
st.set_page_config(
    page_title="FPL Weekly Assistant",
    page_icon="⚽️",  # สามารถใช้ emoji หรือ URL รูปภาพได้
    layout="wide"
)

###############################
# Slide Settings
###############################
st.markdown(
    """
    <style>
    /* CSS สำหรับหน้าจอขนาดใหญ่ (Desktop) */
    @media (min-width: 769px) {
        .mobile-only {
            display: none !important;
        }
    }
    
    /* CSS สำหรับหน้าจอขนาดเล็ก (Mobile) */
    @media (max-width: 768px) {
        /* ซ่อนปุ่ม << >> ของ Streamlit บนมือถือ */
        .st-emotion-cache-1l02wac {
            display: none !important;
        }
        /* ปรับ padding บน mobile เพื่อให้มีพื้นที่มากขึ้น */
        .st-emotion-cache-1629p26 {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="mobile-only" style="text-align: center; margin-bottom: 20px;">
        <h2 style="color: #4CAF50; font-size: 24px;">⚙️ การตั้งค่าอยู่ที่แถบด้านข้าง</h2>
        <p style="color: #607D8B; font-size: 18px;">(คลิก >> มุมซ้ายบนเพื่อเปิด)</p>
    </div>
    """,
    unsafe_allow_html=True
)
###############################
# API helpers
###############################
FPL_BASE = "https://fantasy.premierleague.com/api"

def _fetch(url: str) -> Optional[Dict]:
    """Helper function to fetch JSON data with robust error handling."""
    try:
        response = requests.get(url, timeout=10)
        # ตรวจสอบสถานะการตอบกลับ เช่น 404 Not Found, 500 Internal Server Error
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        # ดักจับข้อผิดพลาดทั้งหมดที่เกี่ยวข้องกับการเชื่อมต่อ
        st.error(f"Error fetching data from FPL API: {e}")
        return None
    except json.JSONDecodeError as e:
        # ดักจับข้อผิดพลาดเมื่อข้อมูลที่ได้รับไม่ใช่ JSON ที่ถูกต้อง
        st.error(f"Error decoding JSON data from FPL API: {e}")
        return None

@st.cache_data(ttl=300)
def get_bootstrap() -> Dict:
    """Fetches the main bootstrap data from FPL API."""
    return _fetch(f"{FPL_BASE}/bootstrap-static/") or {}

@st.cache_data(ttl=300)
def get_fixtures() -> List[Dict]:
    """Fetches the fixtures data from FPL API."""
    return _fetch(f"{FPL_BASE}/fixtures/") or []

@st.cache_data(ttl=300)
def get_entry(entry_id: int) -> Dict:
    """Fetches a user's entry (team) data."""
    return _fetch(f"{FPL_BASE}/entry/{entry_id}/") or {}

@st.cache_data(ttl=300)
def get_entry_picks(entry_id: int, event: int) -> Dict:
    """Fetches a user's picks for a specific gameweek."""
    return _fetch(f"{FPL_BASE}/entry/{entry_id}/event/{event}/picks/") or {}

###############################
# ปรับปรุง Table Headers ให้ User-Friendly
###############################

# 1. สร้าง Dictionary สำหรับแปลง Column Names
def create_column_mapping():
    """สร้าง mapping สำหรับแปลงชื่อ column ให้เป็นภาษาไทย/อังกฤษที่เข้าใจง่าย"""
    
    # Thai + English Headers
    thai_english_headers = {
        "web_name": "ชื่อนักเตะ (Name)",
        "team_short": "ทีม (Team)",
        "element_type": "ตำแหน่ง (Position)",
        "pos": "ตำแหน่ง (Pos)",
        "now_cost": "ราคา (Price)",
        "price": "ราคา (Price)",
        "form": "ฟอร์ม (Form)",
        "avg_fixture_ease": "ความยากของเกม (Fixture)",
        "fixture_ease": "ความยากของเกมถัดไป (Fixture)",
        "pred_points": "คะแนนคาดการณ์ (Pred Points)",
        "points_per_game": "คะแนน/เกม (PPG)",
        "total_points": "คะแนนรวม (Total Pts)",
        "selected_by_percent": "% เลือก (Selected %)",
        "ict_index": "ICT Index",
        "play_prob": "โอกาสลงเล่น (Play %)",
        "num_fixtures": "จำนวนแมตช์ (Fixtures)",
        "out_name": "ขายออก (Out)",
        "in_name": "ซื้อเข้า (In)",
        "delta_points": "ผลต่าง(Points)",
        "net_gain": "กำไรสุทธิ",
        "out_cost": "ราคาขาย (£)",
        "in_cost": "ราคาซื้อ (£)",
        "hit_cost": "ค่าแรงลบ (Hit Cost)",
        "photo_url": "รูป" # Added for image
    }
    
    # English Only Headers (สำหรับคนที่ต้องการแค่ภาษาอังกฤษ)
    english_headers = {
        "web_name": "Player Name",
        "team_short": "Team",
        "element_type": "Position",
        "pos": "Pos",
        "now_cost": "Price (£)",
        "price": "Price (£)",
        "form": "Form",
        "avg_fixture_ease": "Fixture Difficulty",
        "fixture_ease": "Fixture Difficulty",
        "pred_points": "Predicted Points",
        "points_per_game": "Points Per Game",
        "total_points": "Total Points",
        "selected_by_percent": "Selected %",
        "ict_index": "ICT Index",
        "play_prob": "Play Probability",
        "num_fixtures": "Fixtures",
        "out_name": "Player Out",
        "in_name": "Player In",
        "delta_points": "Points Difference",
        "net_gain": "Net Gain",
        "out_cost": "Selling Price",
        "in_cost": "Buying Price",
        "hit_cost": "Hit Cost",
        "photo_url": "Photo" # Added for image
    }
    
    return thai_english_headers, english_headers

# 2. ฟังก์ชันสำหรับจัดรูปแบบ DataFrame
def format_dataframe(df, language="thai_english"):
    """จัดรูปแบบ DataFrame ให้สวยงามและเข้าใจง่าย"""
    
    thai_english_headers, english_headers = create_column_mapping()
    
    # เลือก header mapping ตามภาษา
    if language == "thai_english":
        headers = thai_english_headers
    else:
        headers = english_headers
    
    # สำเนา DataFrame เพื่อไม่ให้กระทบต้นฉบับ
    formatted_df = df.copy()
    
    # เปลี่ยนชื่อ column
    formatted_df.columns = [headers.get(col, col) for col in formatted_df.columns]
    
    return formatted_df

# 3. ฟังก์ชันสำหรับจัดรูปแบบตัวเลข
def format_numbers_in_dataframe(df):
    """จัดรูปแบบตัวเลขในตารางให้อ่านง่าย"""
    
    formatted_df = df.copy()
    
    # จัดรูปแบบตัวเลขต่างๆ
    for col in formatted_df.columns:
        if formatted_df[col].dtype in ['float64', 'int64']:
            # ราคา (มี £ หรือ price ในชื่อ)
            if any(keyword in col.lower() for keyword in ['price', '£', 'cost', 'ราคา']):
                formatted_df[col] = formatted_df[col].apply(lambda x: f"£{x:.1f}m" if pd.notnull(x) else "")
            
            # เปอร์เซ็นต์
            elif any(keyword in col.lower() for keyword in ['%', 'percent', 'prob']):
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "")
            
            # คะแนน
            elif any(keyword in col.lower() for keyword in ['points', 'คะแนน', 'form', 'ฟอร์ม']):
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
            
            # ตัวเลขทั่วไป
            else:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
    
    return formatted_df

# 4. ฟังก์ชันสำหรับเพิ่ม Color Coding
def add_color_coding(df, score_columns=None):
    """เพิ่มสีให้กับตารางตามค่าของคะแนน"""

    if score_columns is None:
        score_columns = ['pred_points', 'form', 'delta_points', 'net_gain']

    def highlight_scores(row):
        colors = []
        for col in row.index:
            original_col_name = col.lower()
            # Check if any keyword from score_columns is in the current column name
            if any(score_col in original_col_name for score_col in score_columns):
                val = row[col]
                # Convert string value to float for comparison
                if isinstance(val, str):
                    try:
                        val = float(val.replace('£', '').replace('m', '').replace('%', ''))
                    except (ValueError, AttributeError):
                        val = 0
                
                # Apply color based on value, with the new 4-color logic.
                # The order of these conditions is crucial, starting from the highest value.
                if val >= 7:
                    colors.append('background-color: #d4edda')  # Light green for high scores
                elif val >= 5:
                    colors.append('background-color: #fff3cd')  # Light yellow for medium scores
                elif val >= 4:
                    colors.append('background-color: #fce4b3')  # Light orange for medium-low scores
                elif val < 4:
                    colors.append('background-color: #f8d7da')  # Light red for low scores
                else:
                    colors.append('') # No color for other values
            else:
                colors.append('') # No color for non-score columns
        return colors

    # Apply the styling function.
    return df.style.apply(highlight_scores, axis=1)

# 5. ฟังก์ชันหลักสำหรับแสดงตาราง (Legacy)
def display_user_friendly_table(df, title="", language="thai_english",
                               add_colors=True, height=400):
    """แสดงตารางที่ user-friendly (แบบดั้งเดิม)"""
    
    if title:
        st.subheader(title)
    
    # Make a copy to avoid modifying the original dataframe
    display_df = df.copy()
    
    # Format column headers and numbers
    formatted_df = format_dataframe(display_df, language)
    formatted_df = format_numbers_in_dataframe(formatted_df)
    
    # Apply color coding if requested
    if add_colors:
        # Pass the original unformatted df for value-based coloring logic
        styled_df = add_color_coding(formatted_df)
        st.dataframe(styled_df, use_container_width=True, height=height)
    else:
        st.dataframe(formatted_df, use_container_width=True, height=height)


# 6. ฟังก์ชันเสริมสำหรับแสดงผลข้อมูล (Legacy)
def display_table_section(df: pd.DataFrame, title: str, columns: list = None, height: int = 400):
    """แสดงตารางข้อมูลในรูปแบบที่กำหนด (แบบดั้งเดิม)"""
    if columns:
        df = df[columns]
    display_user_friendly_table(
        df=df,
        title=title,
        language="thai_english",
        add_colors=True,
        height=height
    )

# 7. ฟังก์ชันสำหรับ Custom CSS
def add_table_css():
    """เพิ่ม CSS สำหรับปรับแต่งตาราง"""
    
    st.markdown("""
    <style>
    /* ปรับแต่งตาราง */
    .dataframe {
        font-size: 14px !important;
    }
    
    .dataframe th {
        background-color: #f0f2f6 !important;
        color: #262730 !important;
        font-weight: bold !important;
        text-align: center !important;
        padding: 12px 8px !important;
        border-bottom: 2px solid #e6e9ef !important;
    }
    
    .dataframe td {
        /* เพิ่มโค้ดนี้เข้าไป */
        text-align: center !important;
        padding: 8px !important;
        border-bottom: 1px solid #e6e9ef !important;
    }
    
    /* สำหรับ mobile */
    @media (max-width: 768px) {
        .dataframe {
            font-size: 12px !important;
        }
        
        .dataframe th, .dataframe td {
            padding: 6px 4px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

###############################
# Data & features
###############################

def current_and_next_event(events: List[Dict]) -> Tuple[Optional[int], Optional[int]]:
    """Determines the current and next gameweek IDs."""
    cur = next_ev = None
    for ev in events:
        if ev.get("is_current"):
            cur = ev["id"]
        if ev.get("is_next"):
            next_ev = ev["id"]
    if cur is None and next_ev is not None:
        cur = next_ev - 1 if next_ev > 1 else 1
    return cur, next_ev

# --- BUGFIX v1.8.2: Add 'code' to the list of columns ---
TEAM_MAP_COLS = ["id", "code", "name", "short_name", "strength_overall_home", "strength_overall_away",
                 "strength_attack_home", "strength_attack_away", "strength_defence_home", "strength_defence_away"]

# --- BUGFIX v1.9.1: Define POSITIONS globally ---
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def build_master_tables(bootstrap: Dict, fixtures: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Constructs the main dataframes for players, teams, events, and fixtures."""
    elements = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])[TEAM_MAP_COLS]
    events = pd.DataFrame(bootstrap.get("events", []))
    
    # --- BUGFIX (v1.8.3): Use 70px logos, 280px is Access Denied ---
    teams['logo_url'] = 'https://resources.premierleague.com/premierleague/badges/70/t' + teams['code'].astype(str) + '.png'

    elements = elements.merge(teams[["id","short_name"]], left_on="team", right_on="id", suffixes=("","_team"))
    elements.rename(columns={"short_name":"team_short"}, inplace=True)

    fixtures_df = pd.DataFrame(fixtures)
    return elements, teams, events, fixtures_df

def next_fixture_features(fixtures_df: pd.DataFrame, teams_df: pd.DataFrame, event_id: int) -> pd.DataFrame:
    """Computes next-game fixture difficulty per team, accounting for DGWs and BGWs."""
    next_gw_fixtures = fixtures_df[fixtures_df["event"] == event_id].copy()

    rows = []
    # Use a dictionary to handle multiple fixtures for a single team (DGW)
    team_data = {team_id: {'home_fixtures': [], 'away_fixtures': []} for team_id in teams_df['id'].unique()}
    
    # --- NEW (v1.9.0): Pre-index teams for faster lookup ---
    teams_idx = teams_df.set_index('id')

    for _, row in next_gw_fixtures.iterrows():
        home_team_id, away_team_id = row['team_h'], row['team_a']
        team_data[home_team_id]['home_fixtures'].append(away_team_id)
        team_data[away_team_id]['away_fixtures'].append(home_team_id)

    for team_id, fixtures_info in team_data.items():
        home_opps = fixtures_info['home_fixtures']
        away_opps = fixtures_info['away_fixtures']
        
        num_fixtures = len(home_opps) + len(away_opps)
        
        # --- NEW (v1.9.0): Opponent String Logic ---
        opp_list = []
        for opp_id in home_opps:
            opp_list.append(f"{teams_idx.loc[opp_id, 'short_name']} (H)")
        for opp_id in away_opps:
            opp_list.append(f"{teams_idx.loc[opp_id, 'short_name']} (A)")
        
        # Blank Gameweek (BGW)
        if num_fixtures == 0:
            opponent_str = "BLANK"
            rows.append({
                'team': team_id,
                'num_fixtures': 0,
                'total_opp_def_str': 0,
                'avg_fixture_ease': 0,
                'opponent_str': opponent_str
            })
            continue

        # Double Gameweek (DGW) or single GW
        opponent_str = ", ".join(opp_list)
        total_opp_def_str = 0
        total_opp_att_str = 0
        for opp_id in home_opps:
            opp_team = teams_idx.loc[opp_id]
            total_opp_def_str += opp_team['strength_defence_away']
            total_opp_att_str += opp_team['strength_attack_away']
        for opp_id in away_opps:
            opp_team = teams_idx.loc[opp_id]
            total_opp_def_str += opp_team['strength_defence_home']
            total_opp_att_str += opp_team['strength_attack_home']

        rows.append({
            'team': team_id,
            'num_fixtures': num_fixtures,
            'total_opp_def_str': total_opp_def_str,
            'avg_fixture_ease': 1.0 - (total_opp_def_str / (num_fixtures * teams_idx['strength_defence_home'].max())),
            'opponent_str': opponent_str
        })

    df = pd.DataFrame(rows)
    return df

def engineer_features(elements: pd.DataFrame, teams: pd.DataFrame, nf: pd.DataFrame) -> pd.DataFrame:
    """Joins player table with next fixture info and creates predictive features."""
    elements["element_type"] = pd.to_numeric(elements["element_type"], errors='coerce').fillna(0).astype(int)

    # Merge fixture data
    elements = elements.merge(nf, on="team", how="left")
    elements['num_fixtures'] = elements['num_fixtures'].fillna(0).astype(int)
    elements['avg_fixture_ease'] = elements['avg_fixture_ease'].fillna(0)
    # --- NEW (v1.9.0): Carry over opponent string ---
    elements['opponent_str'] = elements['opponent_str'].fillna("Error")

    # --- NEW (v1.9.4): Ensure price/transfer columns are numeric ---
    cols_to_process = [
        "form", "points_per_game", "ict_index", "selected_by_percent", "now_cost", "starts", "code",
        "cost_change_event", "transfers_in_event", "transfers_out_event"
    ]
    for col in cols_to_process:
        if col in elements.columns:
            elements[col] = pd.to_numeric(elements[col], errors="coerce").fillna(0)
    
    # --- NEW: Add photo_url ---
    elements['photo_url'] = 'https://resources.premierleague.com/premierleague/photos/players/110x140/p' + elements['code'].astype(int).astype(str) + '.png'
    
    elements["chance_of_playing_next_round"] = pd.to_numeric(elements["chance_of_playing_next_round"], errors="coerce").fillna(100)
    elements["play_prob"] = elements["chance_of_playing_next_round"] / 100.0
    elements["xgi_proxy"] = 0.6 * elements["points_per_game"] + 0.4 * (elements["ict_index"] / 10.0)

    # Heuristic prediction model
    pos_mult = np.select(
        [elements["element_type"] == 1, elements["element_type"] == 2, elements["element_type"] == 3, elements["element_type"] == 4],
        [0.6, 0.8, 1.0, 1.1],
        default=1.0
    )
    elements["pred_points_heur"] = (
        (0.45 * elements["xgi_proxy"] + 0.35 * elements["form"] + 0.2 * elements["points_per_game"]) *
        (0.6 + 0.4 * elements["avg_fixture_ease"]) *
        (0.5 + 0.5 * elements["play_prob"]) * pos_mult * elements['num_fixtures']
    )
    elements["pred_points_heur"] = elements["pred_points_heur"].clip(lower=0, upper=30)
    
    # Ensure BGW players have 0 points
    elements.loc[elements['num_fixtures'] == 0, 'pred_points_heur'] = 0

    return elements

# --- START: NEW FIXTURE PLANNER FUNCTIONS (v1.9.0) ---

@st.cache_data(ttl=300)
def get_fixture_difficulty_matrix(fixtures_df: pd.DataFrame, teams_df: pd.DataFrame, current_event: int, lookahead: int = 5):
    """
    Creates a 5-gameweek difficulty matrix for all 20 teams.
    """
    team_names = teams_df.set_index('id')['short_name'].to_dict() # Use to_dict() for mapping
    team_strength = teams_df.set_index('id')

    # Get all fixtures for the next 'lookahead' gameweeks
    future_gws = list(range(current_event, min(current_event + lookahead, 39)))
    future_fixtures = fixtures_df[fixtures_df['event'].isin(future_gws)]
    
    # Create dictionaries to store opponent and difficulty data
    opp_data = {team_id: {} for team_id in teams_df['id']}
    diff_data = {team_id: {} for team_id in teams_df['id']}

    for gw in future_gws:
        gw_fixtures = future_fixtures[future_fixtures['event'] == gw]
        teams_with_fixtures = set(gw_fixtures['team_h']).union(set(gw_fixtures['team_a']))

        for team_id in teams_df['id']:
            if team_id not in teams_with_fixtures:
                # Blank Gameweek (BGW)
                opp_data[team_id][f'GW{gw}'] = "BLANK"
                diff_data[team_id][f'GW{gw}'] = 0 # Neutral difficulty for BGW
                continue

            # Get fixtures for this team
            home_games = gw_fixtures[gw_fixtures['team_h'] == team_id]
            away_games = gw_fixtures[gw_fixtures['team_a'] == team_id]

            opponents = []
            difficulties = []
            
            for _, game in home_games.iterrows():
                opp_id = game['team_a']
                opponents.append(f"{team_names.get(opp_id, '?')} (H)")
                # Difficulty: Opponent's away strength
                diff = team_strength.loc[opp_id, 'strength_overall_away']
                difficulties.append(diff)

            for _, game in away_games.iterrows():
                opp_id = game['team_h']
                opponents.append(f"{team_names.get(opp_id, '?')} (A)")
                # Difficulty: Opponent's home strength
                diff = team_strength.loc[opp_id, 'strength_overall_home']
                difficulties.append(diff)
            
            # Store data (handles DGW by joining strings/averaging difficulty)
            opp_data[team_id][f'GW{gw}'] = ", ".join(opponents)
            diff_data[team_id][f'GW{gw}'] = np.mean(difficulties) if difficulties else 0

    # Create DataFrames
    # --- BUGFIX v1.9.0: Use .applymap, not .map ---
    # We must use .map here because team_names is a dict. applymap is for functions.
    # Ah, the error is because we need to map the *index* not the *dataframe*
    opp_df = pd.DataFrame(opp_data).T
    diff_df = pd.DataFrame(diff_data).T.fillna(0) # Fill BGW NaNs
    
    # Add team names as index
    opp_df.index = opp_df.index.map(team_names)
    diff_df.index = diff_df.index.map(team_names)
    
    # Sort by total difficulty (easiest first)
    diff_df['Total'] = diff_df.sum(axis=1)
    diff_df = diff_df.sort_values('Total', ascending=True)
    opp_df = opp_df.loc[diff_df.index] # Match order

    return opp_df, diff_df

# --- DELETED (v1.9.3): Removed redundant get_difficulty_css_class from v1.9.0 ---

# --- DELETED (v1.9.3): Removed redundant display_visual_fixture_planner from v1.9.0 ---

@st.cache_data(ttl=300)
def find_rotation_pairs(difficulty_matrix: pd.DataFrame, teams_df: pd.DataFrame, all_players: pd.DataFrame, budget: float = 9.0):
    """
    Finds the best GK rotation pairs within a budget for the next 5 GWs.
    """
    gks = all_players[all_players['element_type'] == 1].copy()
    gks['price'] = gks['now_cost'] / 10.0
    gks['team_short'] = gks['team'].map(teams_df.set_index('id')['short_name'])

    cheap_gks = gks[gks['price'] <= (budget - 4.0)] # Must be <= (budget - cheapest GK)
    
    pairs = []
    checked_pairs = set()

    for i, gk1 in cheap_gks.iterrows():
        for j, gk2 in cheap_gks.iterrows():
            if i >= j or (gk2['team'], gk1['team']) in checked_pairs:
                continue
            
            # Check budget
            if (gk1['price'] + gk2['price']) > budget:
                continue
                
            checked_pairs.add((gk1['team'], gk2['team']))
            
            # Get difficulty rows for these two teams
            try:
                diff1 = difficulty_matrix.loc[gk1['team_short']]
                diff2 = difficulty_matrix.loc[gk2['team_short']]
            except KeyError:
                continue # Skip if team not in matrix

            # Calculate rotation score (minimum difficulty for each GW)
            rotation_score = 0
            for col in difficulty_matrix.columns:
                if col == 'Total': continue
                rotation_score += min(diff1[col], diff2[col])
            
            pairs.append({
                'GK1': f"{gk1['web_name']} ({gk1['price']:.1f}m)",
                'GK2': f"{gk2['web_name']} ({gk2['price']:.1f}m)",
                'Total Cost': gk1['price'] + gk2['price'],
                'Rotation Score': rotation_score
            })

    if not pairs:
        return pd.DataFrame(columns=['GK1', 'GK2', 'Total Cost', 'Rotation Score'])
        
    pairs_df = pd.DataFrame(pairs).sort_values('Rotation Score', ascending=True).head(5)
    pairs_df['Total Cost'] = pairs_df['Total Cost'].apply(lambda x: f"£{x:.1f}m")
    pairs_df['Rotation Score'] = pairs_df['Rotation Score'].apply(lambda x: f"{x:.1f}")
    return pairs_df.reset_index(drop=True)

# --- END: NEW FIXTURE PLANNER FUNCTIONS ---


###############################
# Squad & optimization
###############################

def optimize_starting_xi(squad_players_df: pd.DataFrame) -> Tuple[List[int], List[int]]:
    """Return (start_ids, bench_ids) maximizing predicted points subject to FPL formation."""
    ids = list(squad_players_df.index)
    pred_points = squad_players_df['pred_points']
    positions = squad_players_df['element_type']

    prob = LpProblem("XI_Optimization", LpMaximize)
    x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in ids}
    prob += lpSum([pred_points.get(i, 0) * x[i] for i in ids])

    # Constraints
    prob += lpSum([x[i] for i in ids]) == 11
    prob += lpSum([x[i] for i in ids if positions.get(i) == 1]) == 1
    prob += lpSum([x[i] for i in ids if positions.get(i) == 2]) >= 3
    prob += lpSum([x[i] for i in ids if positions.get(i) == 2]) <= 5
    prob += lpSum([x[i] for i in ids if positions.get(i) == 3]) >= 2
    prob += lpSum([x[i] for i in ids if positions.get(i) == 3]) <= 5
    prob += lpSum([x[i] for i in ids if positions.get(i) == 4]) >= 1
    prob += lpSum([x[i] for i in ids if positions.get(i) == 4]) <= 3

    prob.solve(PULP_CBC_CMD(msg=0))
    
    if LpStatus[prob.status] == 'Optimal':
        start_ids = [i for i in ids if x[i].value() == 1]
        bench_ids = [i for i in ids if i not in start_ids]
        return start_ids, bench_ids
    else:
        # Return empty lists if no optimal solution found
        return [], []

def suggest_transfers(current_squad_ids: List[int], bank: float, free_transfers: int,
                      all_players: pd.DataFrame,
                      strategy: str) -> List[Dict]:
    """Greedy search for transfers based on the selected strategy.
    Returns a list of move dicts with keys used by the rest of the app
    (out_id, in_id, in_name, out_cost, in_cost, delta_points, warning, ...).
    Enforces max 3 players per Premier League team.
    """
    
    # Ensure squad IDs are valid
    valid_squad_ids = [pid for pid in current_squad_ids if pid in all_players.index]
    if not valid_squad_ids:
        st.error("No valid players found in current squad for transfer suggestion.")
        return []

    current_squad_df = all_players.loc[valid_squad_ids]
    start_ids, _ = optimize_starting_xi(current_squad_df)
    if not start_ids:
        st.error("Could not optimize starting XI for transfer suggestions.")
        return []
    base_sum = float(current_squad_df.loc[start_ids]['pred_points'].sum())

    if strategy == "Free Transfer":
        max_transfers = free_transfers
        hit_cost = float('inf')
    elif strategy == "Allow Hit (AI Suggest)":
        max_transfers = 5
        hit_cost = 4
    else:  # Wildcard/Free Hit
        max_transfers = 15
        hit_cost = 0

    # Count current players per team
    current_team_count = {}
    for pid in valid_squad_ids:
        team_id = int(all_players.loc[pid, 'team'])
        current_team_count[team_id] = current_team_count.get(team_id, 0) + 1

    # Group squad by position (element_type): 1 GK, 2 DEF, 3 MID, 4 FWD
    position_groups = {1: [], 2: [], 3: [], 4: []}
    for pid in valid_squad_ids:
        pos = int(all_players.loc[pid, 'element_type'])
        position_groups.setdefault(pos, []).append(pid)

    remaining_bank = bank
    used_in_players = set()
    potential_moves = []

    # Iterate positions in typical order (DEF/MID/FWD/GK not too important)
    for pos in [1, 2, 3, 4]:
        out_ids = position_groups.get(pos, [])
        if not out_ids:
            continue

        # consider weaker squad players first (ascending predicted points)
        for out_id in sorted(out_ids, key=lambda x: all_players.loc[x, 'pred_points']):
            out_player = all_players.loc[out_id]
            out_team_id = int(out_player['team'])

            # budget: player's selling price plus whatever cash we have (bank is in £ *10 in dataset)
            budget_for_replacement = out_player['selling_price'] + (remaining_bank * 10)

            # candidate pool: same position, not already in squad, within budget and better predicted points
            all_replacements = all_players[
                (all_players['element_type'] == out_player['element_type']) &
                (~all_players.index.isin(valid_squad_ids)) & # Use valid_squad_ids
                (all_players['now_cost'] <= budget_for_replacement) &
                (all_players['pred_points'] > out_player['pred_points'])
            ].sort_values('pred_points', ascending=False)

            if all_replacements.empty:
                continue

            best_replacement = None

            # Try find a candidate respecting 3-per-team rule
            for cid, candidate in all_replacements.iterrows():
                candidate_team_id = int(candidate['team'])

                # simulate future team counts after swapping out_id -> cid
                future_team_count = current_team_count.copy()
                future_team_count[out_team_id] = future_team_count.get(out_team_id, 0) - 1
                if future_team_count[out_team_id] <= 0:
                    future_team_count.pop(out_team_id, None)

                future_count = future_team_count.get(candidate_team_id, 0) + 1
                if future_count > 3:
                    # would violate 3-per-team -> skip candidate
                    continue

                # also skip if we already planned to bring this candidate in
                if int(cid) in used_in_players:
                    continue

                # if passes checks, pick as best_replacement (first highest pred_points)
                best_replacement = candidate
                best_replacement_id = int(cid)
                break

            # Fallback: try same-team replacements (if any)
            if best_replacement is None:
                # --- BUGFIX v1.6.2: Filter all_replacements, not all_players ---
                same_team_replacements = all_replacements[
                    (all_replacements['team'] == out_team_id) &
                    (~all_replacements.index.isin(used_in_players))
                ]
                if not same_team_replacements.empty:
                    # swapping within same team will not increase count for that team,
                    # so it's safe even when currently at 3 players
                    best_replacement = same_team_replacements.iloc[0]
                    best_replacement_id = int(best_replacement.name)

            if best_replacement is None:
                continue

            # Final team-limit double-check (safety net)
            future_team_count = current_team_count.copy()
            future_team_count[out_team_id] = future_team_count.get(out_team_id, 0) - 1
            if future_team_count[out_team_id] <= 0:
                future_team_count.pop(out_team_id, None)
            if future_team_count.get(int(best_replacement['team']), 0) + 1 > 3:
                continue

            # cost change (in - out) in £ (dataset stores cost*10)
            cost_change = (best_replacement['now_cost'] - out_player['selling_price']) / 10.0
            if cost_change > remaining_bank:
                continue

            # Update team counts and remaining bank as if we accepted this move (greedy)
            if out_team_id != int(best_replacement['team']):
                current_team_count[out_team_id] = current_team_count.get(out_team_id, 0) - 1
                if current_team_count[out_team_id] <= 0:
                    current_team_count.pop(out_team_id, None)
                current_team_count[int(best_replacement['team'])] = current_team_count.get(int(best_replacement['team']), 0) + 1

            remaining_bank = round(max(0.0, remaining_bank - cost_change), 2)
            used_in_players.add(best_replacement_id)

            # build move dict (match keys used elsewhere)
            move = {
                "out_id": int(out_id),
                "in_id": best_replacement_id,
                "out_name": out_player.get("web_name", ""),
                "in_name": best_replacement.get("web_name", ""),
                "out_pos": POSITIONS.get(int(out_player["element_type"]), str(out_player["element_type"])),
                "in_pos": POSITIONS.get(int(best_replacement["element_type"]), str(best_replacement["element_type"])),
                "out_team": out_player.get("team_short", ""),
                "in_team": best_replacement.get("team_short", ""),
                "in_points": float(best_replacement.get("pred_points", 0.0)),
                "delta_points": float(best_replacement.get('pred_points', 0.0) - out_player.get('pred_points', 0.0)),
                "in_cost": float(best_replacement.get('now_cost', 0.0)) / 10.0,
                "out_cost": float(out_player.get('selling_price', 0.0)) / 10.0,
            }

            # warning if after this move you'll have 3 players from that team (informational)
            if current_team_count.get(int(best_replacement['team']), 0) == 3:
                move["warning"] = f"Already have 3 players from {best_replacement.get('team_short','')}"

            # ✅ Hard rule: absolutely forbid >3 per team
            future_team_count = current_team_count.copy()
            future_team_count[out_team_id] = future_team_count.get(out_team_id, 0) - 1
            if future_team_count[out_team_id] <= 0:
                future_team_count.pop(out_team_id, None)
            if future_team_count.get(int(best_replacement['team']), 0) + 1 > 3:
                continue  # skip this move, it would create 4th player from same team
        
            potential_moves.append(move)

    # Rank moves by expected points gain
    potential_moves.sort(key=lambda x: x.get("delta_points", 0.0), reverse=True)

    # --- START: v1.6.1 AGGRESSIVE AI LOGIC ---
    final_suggestions = []
    
    # Define thresholds
    GREEDY_THRESHOLD = -2.0  # Allow net loss of 2.0 pts (i.e., delta_points > 2.0 for a -4 hit)
    CONSERVATIVE_THRESHOLD = -0.1 # Must almost break even (for Free Transfer)

    for i, move in enumerate(potential_moves):
        if len(final_suggestions) >= max_transfers:
            break

        hit = 0 if len(final_suggestions) < free_transfers else hit_cost
        
        # Calculate pure net gain (profit)
        net_gain = move["delta_points"] - hit

        # Create the move object
        m = move.copy()
        m['net_gain'] = round(net_gain, 2)
        m['hit_cost'] = hit

        # Apply strategy-based filtering
        if strategy == "Free Transfer":
            # Conservative: Only accept moves that don't cost points.
            if net_gain >= CONSERVATIVE_THRESHOLD:
                final_suggestions.append(m)
        
        elif strategy == "Allow Hit (AI Suggest)":
            # Aggressive: Accept any move that doesn't lose us *too many* points.
            if net_gain >= GREEDY_THRESHOLD:
                final_suggestions.append(m)

        elif strategy == "Wildcard / Free Hit":
            # Wildcard: hit_cost is 0, so net_gain = delta_points.
            # Accept all positive moves up to the max transfer limit (15).
            if net_gain > 0.0:
                final_suggestions.append(m)
    
    # --- END: v1.6.1 AGGRESSIVE AI LOGIC ---

    return final_suggestions

def optimize_wildcard_team(all_players: pd.DataFrame, budget: float) -> Optional[List[int]]:
    """Optimizes a 15-man squad for a wildcard or free hit."""
    ids = list(all_players.index)
    teams = all_players['team'].unique()
    
    prob = LpProblem("Wildcard_Optimization", LpMaximize)
    x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in ids}
    
    prob += lpSum([all_players.loc[i, 'pred_points'] * x[i] for i in ids])

    prob += lpSum([all_players.loc[i, 'now_cost'] * x[i] for i in ids]) <= budget * 10
    prob += lpSum([x[i] for i in ids]) == 15
    prob += lpSum([x[i] for i in ids if all_players.loc[i, 'element_type'] == 1]) == 2
    prob += lpSum([x[i] for i in ids if all_players.loc[i, 'element_type'] == 2]) == 5
    prob += lpSum([x[i] for i in ids if all_players.loc[i, 'element_type'] == 3]) == 5
    prob += lpSum([x[i] for i in ids if all_players.loc[i, 'element_type'] == 4]) == 3
    
    for team_id in teams:
        prob += lpSum([x[i] for i in ids if all_players.loc[i, 'team'] == team_id]) <= 3

    prob.solve(PULP_CBC_CMD(msg=0))
    
    if LpStatus[prob.status] == 'Optimal':
        return [i for i in ids if x[i].value() == 1]
    return None

def suggest_transfers_enhanced(current_squad_ids: List[int], bank: float, free_transfers: int,
                              all_players: pd.DataFrame, strategy: str) -> Tuple[List[Dict], List[Dict]]:
    """แนะนำ transfers แบบ 2 มุมมอง: ปกติ vs ระมัดระวัง"""
    # การแนะนำแบบปกติ
    normal_moves = suggest_transfers(current_squad_ids, bank, free_transfers, all_players, strategy)
    
    # การแนะนำแบบระมัดระวัง
    conservative_all_players = all_players.copy()
    for player_id in current_squad_ids:
        if player_id not in all_players.index: continue
        current_price = all_players.loc[player_id, 'selling_price']
        # ลดราคาขายลง 0.2 เพื่อความปลอดภัย
        conservative_price = max(current_price - 2, current_price * 0.95)  # ลดขั้นต่ำ 0.2 หรือ 5%
        conservative_all_players.loc[player_id, 'selling_price'] = conservative_price
    
    # คำนวณงบที่มีอยู่จริงหลังจากลดราคาขาย
    conservative_bank = bank
    for move in normal_moves:
        if move['out_id'] not in all_players.index: # Safety check
            continue
        original_price = all_players.loc[move['out_id'], 'selling_price']
        conservative_price = conservative_all_players.loc[move['out_id'], 'selling_price']
        price_diff = (original_price - conservative_price) / 10.0
        conservative_bank = max(0, conservative_bank - price_diff)
    
    # หา transfers ที่เป็นไปได้โดยใช้ราคาที่ระมัดระวัง
    conservative_moves = suggest_transfers(
        current_squad_ids,
        conservative_bank,  # ใช้งบที่ปรับแล้ว
        free_transfers,
        conservative_all_players,
        strategy
    )
    
    # กรองเฉพาะ transfers ที่แน่ใจว่าทำได้
    filtered_conservative_moves = []
    remaining_bank = conservative_bank
    used_players = set()
    
    for move in conservative_moves:
        if move['in_id'] not in used_players:
            cost_change = move['in_cost'] - move['out_cost']
            if cost_change <= remaining_bank:
                if move['out_id'] in conservative_all_players.index: # Safety check
                    move['out_cost'] = round(conservative_all_players.loc[move['out_id'], 'selling_price'] / 10.0, 1)
                    filtered_conservative_moves.append(move)
                    remaining_bank -= cost_change
                    used_players.add(move['in_id'])
    
    return normal_moves, filtered_conservative_moves


###############################
# --- NEW: Pitch View Function ---
###############################

def display_pitch_view(team_df: pd.DataFrame, title: str):
    """
    Displays the 11-man squad on a football pitch using HTML/CSS.
    """
    st.subheader(title)

    # Define CSS for the pitch and players
    pitch_css = """
    <style>
    .pitch-container {
        position: relative;
        width: 100%;
        max-width: 600px; /* Max width for better layout */
        margin: 20px auto;
        background-image: url('https://raw.githubusercontent.com/kengjirayus/fpl/refs/heads/main/Pix/FPL-Wiz-Field.png');
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
        aspect-ratio: 7/10; /* Proportion of a pitch */
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        padding: 5% 0; /* Padding top/bottom to space out rows */
    }
    .pitch-row {
        display: flex;
        justify-content: space-around;
        align-items: center;
        width: 100%;
        margin-bottom: 10%; /* Space between rows */
    }
    .player-card {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        width: 80px; /* Fixed width for card */
    }
    .player-card img {
        width: 60px;
        height: 80px;
        margin-bottom: 4px;
        /* Fallback for broken images */
        background-color: #eee;
        border-radius: 4px;
        object-fit: cover;
    }
    .player-name {
        font-size: 11px;
        font-weight: bold;
        color: white;
        background-color: rgba(0, 0, 0, 0.7);
        padding: 2px 5px;
        border-radius: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        width: 100%;
        box-sizing: border-box; /* Include padding in width */
    }
    .player-info {
        font-size: 10px;
        color: #f0f0f0;
        background-color: rgba(50, 50, 50, 0.6);
        padding: 1px 4px;
        border-radius: 4px;
        margin-top: 2px;
    }
    </style>
    """
    
    # Separate players by position
    team_df['pos'] = team_df['element_type'].map(POSITIONS)
    gk = team_df[team_df['pos'] == 'GK']
    defs = team_df[team_df['pos'] == 'DEF'].sort_values('pred_points', ascending=False)
    mids = team_df[team_df['pos'] == 'MID'].sort_values('pred_points', ascending=False)
    fwds = team_df[team_df['pos'] == 'FWD'].sort_values('pred_points', ascending=False)
    
    # --- NEW (v1.9.6): Add placeholder URL ---
    DEFAULT_PHOTO_URL_PITCH = "https://resources.premierleague.com/premierleague/photos/players/110x140/p-blank.png"

    def generate_player_html(player_row):
        # Add captain/vice-captain logic
        name = player_row['web_name']
        if player_row.get('is_captain', False):
            name = f"{name} (C)"
        elif player_row.get('is_vice_captain', False):
            name = f"{name} (V)"
            
        # --- MODIFIED (v1.9.6): Added onerror attribute ---
        return f"<div class='player-card'><img src='{player_row['photo_url']}' alt='{player_row['web_name']}' onerror=\"this.onerror=null;this.src='{DEFAULT_PHOTO_URL_PITCH}';\"><div class='player-name'>{name}</div><div class='player-info'>{player_row['team_short']} | {player_row['pred_points']:.1f}pts</div></div>"

    # Build HTML string
    html = f"{pitch_css}<div class='pitch-container'>"
    
    # GK Row
    html += "<div class='pitch-row'>"
    for _, player in gk.iterrows():
        html += generate_player_html(player)
    html += "</div>"
    
    # DEF Row
    html += "<div class='pitch-row'>"
    for _, player in defs.iterrows():
        html += generate_player_html(player)
    html += "</div>"
    
    # MID Row
    html += "<div class='pitch-row'>"
    for _, player in mids.iterrows():
        html += generate_player_html(player)
    html += "</div>"
    
    # FWD Row
    html += "<div class='pitch-row'>"
    for _, player in fwds.iterrows():
        html += generate_player_html(player)
    html += "</div>"
    
    html += "</div>"
    
    st.markdown(html, unsafe_allow_html=True)

###############################
# --- NEW: Visual Fixture Planner (v1.9.3) ---
###############################

def get_difficulty_css_class(val, min_val, max_val):
    """Returns the CSS class based on the difficulty score."""
    if val == 0: # BGW
        return "bg-blank" # Dark grey
    
    # Normalize from 0 (easy) to 1 (hard)
    norm_val = (val - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else 0.5
    
    if norm_val < 0.33:
        # Easy (Green) - เขียวสด
        return "bg-easy"
    elif norm_val < 0.66:
        # Medium (Orange) - สีส้ม
        return "bg-medium"
    else:
        # Hard (Red) - แดง
        return "bg-hard"

def display_visual_fixture_planner(opp_matrix: pd.DataFrame, diff_matrix: pd.DataFrame, teams_df: pd.DataFrame):
    """
    Displays the Fixture Planner as a visual HTML table with logos and colors.
    """
    
    # Create lookup dict for team logos
    team_logo_lookup = teams_df.set_index('short_name')['logo_url'].to_dict()
    
    # Get GW columns
    gw_cols = [col for col in diff_matrix.columns if col.startswith('GW')]
    
    # Calculate min/max for coloring (excluding BGWs and Total)
    non_zero_vals = diff_matrix[gw_cols][diff_matrix[gw_cols] != 0].unstack().dropna()
    min_val = non_zero_vals.min()
    max_val = non_zero_vals.max()

    # --- Start building HTML string ---
    html = """
    <style>
        .fixture-planner {
            width: 100%;
            border-collapse: collapse;
            font-family: 'Inter', sans-serif;
            border-radius: 8px;
            overflow: hidden; /* For border-radius */
        }
        .fixture-planner th, .fixture-planner td {
            text-align: center;
            padding: 8px 4px;
            border: 1px solid #444;
            min-width: 65px;
        }
        .fixture-planner th {
            background-color: #333;
            color: white;
            font-size: 14px;
        }
        .team-cell {
            width: 70px;
            background-color: #f0f2f6;
            padding: 4px;
        }
        .team-cell img {
            width: 35px;
            height: 35px;
        }
        .team-cell span {
            display: block;
            font-size: 13px;
            font-weight: bold;
            color: #333;
            margin-top: 2px;
        }
        .fixture-cell {
            vertical-align: middle;
            font-size: 13px;
            font-weight: bold;
            width: 70px;
            height: 60px;
        }
        .fixture-cell img {
            width: 25px;
            height: 25px;
            vertical-align: middle;
        }
        /* Color classes from user request (UPDATED) */
        .bg-easy { background-color: #35F00A; color: black; }
        .bg-medium { background-color: #FFF100; color: black; }
        .bg-hard { background-color: #FF0000; color: white; }
        .bg-blank { background-color: #373737; color: white; }
        .dgw-cell { 
            font-size: 12px; 
            line-height: 1.4;
            text-align: left;
            padding-left: 8px;
        }
    </style>
    <table class="fixture-planner">
        <thead>
            <tr>
                <th>Team</th>
    """
    
    # Add GW Headers
    for gw in gw_cols:
        html += f"<th>{gw}</th>"
    html += "</tr></thead><tbody>"

    # Add Team Rows
    for team_short_name, diff_row in diff_matrix.drop(columns=['Total']).iterrows():
        team_logo_url = team_logo_lookup.get(team_short_name, '')
        html += "<tr>"
        # Column 1: Team Logo & Name
        html += f"<td class='team-cell'><img src='{team_logo_url}' alt='{team_short_name}'><br><span>{team_short_name}</span></td>"
        
        # Columns 2-6: Fixtures
        for gw in gw_cols:
            diff_score = diff_row[gw]
            opp_string = opp_matrix.loc[team_short_name, gw]
            css_class = get_difficulty_css_class(diff_score, min_val, max_val)
            
            cell_content = ""
            if opp_string == "BLANK":
                cell_content = "BLANK"
            elif "," in opp_string:
                # Double Gameweek
                cell_content = opp_string.replace(", ", "<br>")
                css_class = "dgw-cell " + css_class # Add DGW style
            else:
                # Single Gameweek
                opp_short_name = opp_string.split(" ")[0]
                home_away = opp_string.split(" ")[1]
                opp_logo_url = team_logo_lookup.get(opp_short_name, '')
                cell_content = f"<img src='{opp_logo_url}' alt='{opp_short_name}'><br>{home_away}"

            html += f"<td class='fixture-cell {css_class}'>{cell_content}</td>"
        
        html += "</tr>"

    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)


###############################
# --- NEW: Home Dashboard Function (v1.9.0) ---
###############################

def display_home_dashboard(feat_df: pd.DataFrame, nf_df: pd.DataFrame, teams_df: pd.DataFrame, 
                           opp_matrix: pd.DataFrame, diff_matrix: pd.DataFrame, rotation_pairs: pd.DataFrame):
    """
    Displays the full home page dashboard (DGW/BGW, Captains, Top 20, Value, Fixtures, Trends).
    """
    
    # --- NEW (v1.9.6): Add placeholder URL and helper function ---
    DEFAULT_PHOTO_URL = "https://resources.premierleague.com/premierleague/photos/players/110x140/p-blank.png"

    def get_player_image_html(photo_url, player_name, width=60):
        """Generates an HTML img tag with a fallback placeholder."""
        # Use HTML entities for quotes inside the onerror attribute
        alt_text = player_name.replace("'", "").replace('"', '')
        return f'<img src="{photo_url}" alt="{alt_text}" width="{width}" style="border-radius: 4px; min-height: {int(width*1.33)}px; background-color: #eee;" onerror="this.onerror=null;this.src=\'{DEFAULT_PHOTO_URL}\';">'

    
    # --- 1. DGW/BGW Tracker (Conditional) ---
    dgw_teams = nf_df[nf_df['num_fixtures'] == 2]
    bgw_teams = nf_df[nf_df['num_fixtures'] == 0]

    if not dgw_teams.empty or not bgw_teams.empty:
        st.subheader("🚨 สรุปทีม DGW / BGW")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🟩 Double Gameweek (น่าซื้อ)")
            if dgw_teams.empty:
                st.caption("ไม่มีทีม Double Gameweek")
            else:
                dgw_teams_merged = dgw_teams.merge(teams_df[['id', 'short_name', 'logo_url']], left_on='team', right_on='id')
                for _, row in dgw_teams_merged.iterrows():
                    c1, c2 = st.columns([1, 4])
                    with c1: st.image(row['logo_url'], width=40)
                    with c2: st.markdown(f"**{row['short_name']}**"); st.caption(f"{row['opponent_str']}")
        
        with col2:
            st.markdown("#### 🟥 Blank Gameweek (น่าขาย)")
            if bgw_teams.empty:
                st.caption("ไม่มีทีม Blank Gameweek")
            else:
                bgw_teams_merged = bgw_teams.merge(teams_df[['id', 'short_name', 'logo_url']], left_on='team', right_on='id')
                for _, row in bgw_teams_merged.iterrows():
                    c1, c2 = st.columns([1, 4])
                    with c1: st.image(row['logo_url'], width=40)
                    with c2: st.markdown(f"**{row['short_name']}**"); st.caption("ไม่มีนัดแข่ง")
        st.markdown("---")

    # --- 2. Captaincy Corner & Price Movement (v1.9.5 - 3-col layout) ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👑 5 สุดยอดกัปตัน")
        captains = feat_df.nlargest(5, 'pred_points')
        if captains.empty:
            st.caption("ไม่มีข้อมูลกัปตัน")
        else:
            for _, row in captains.iterrows():
                c1, c2 = st.columns([1, 4])
                with c1:
                    # --- MODIFIED (v1.9.6): Use HTML fallback ---
                    st.markdown(get_player_image_html(row['photo_url'], row['web_name'], 60), unsafe_allow_html=True)
                with c2:
                    st.markdown(f"**{row['web_name']}** ({row['team_short']})")
                    st.markdown(f"**คะแนน: {row['pred_points']:.1f}**")
                    st.caption(f"คู่แข่ง: {row['opponent_str']}")

    with col2:
        st.subheader("💹 ราคากำลังขึ้น 🔼")
        
        # --- Price Risers ---
        risers = feat_df[feat_df['cost_change_event'] > 0].sort_values('cost_change_event', ascending=False).head(5)
        if risers.empty:
            st.caption("ไม่มีนักเตะราคาขึ้นในสัปดาห์นี้")
        else:
            for _, row in risers.iterrows():
                c1, c2 = st.columns([1, 4])
                with c1: 
                    # --- MODIFIED (v1.9.6): Use HTML fallback ---
                    st.markdown(get_player_image_html(row['photo_url'], row['web_name'], 60), unsafe_allow_html=True)
                with c2: 
                    st.markdown(f"**{row['web_name']}** ({row['team_short']})")
                    st.caption(f"▲ ราคาขึ้น: £{row['cost_change_event']/10.0:.1f}m")

    with col3:
        st.subheader("🔻 ราคากำลังลง 📉")
        
        # --- Price Fallers ---
        fallers = feat_df[feat_df['cost_change_event'] < 0].sort_values('cost_change_event', ascending=True).head(5)
        if fallers.empty:
            st.caption("ไม่มีนักเตะราคาลงในสัปดาห์นี้")
        else:
            for _, row in fallers.iterrows():
                c1, c2 = st.columns([1, 4])
                with c1: 
                    # --- MODIFIED (v1.9.6): Use HTML fallback ---
                    st.markdown(get_player_image_html(row['photo_url'], row['web_name'], 60), unsafe_allow_html=True)
                with c2: 
                    st.markdown(f"**{row['web_name']}** ({row['team_short']})")
                    st.caption(f"▼ ราคาลง: £{row['cost_change_event']/10.0:.1f}m")

    st.markdown("---")

    # --- 3. Top 20 Players ---
    st.subheader("⭐ Top 20 นักเตะคะแนนคาดการณ์สูงสุด")
    st.caption("หมายเหตุ: ตารางนี้อาจยังแสดงไอคอนรูปเสีย 🖼️ หากไม่มีรูปใน API ครับ")
    top_tbl = feat_df[["photo_url", "web_name", "team_short", "element_type", "now_cost", "form", "avg_fixture_ease", "pred_points"]].copy()
    top_tbl.rename(columns={"element_type": "pos", "now_cost": "price", "avg_fixture_ease": "fixture_ease"}, inplace=True)
    top_tbl["pos"] = top_tbl["pos"].map(POSITIONS)
    top_tbl["price"] = (top_tbl["price"] / 10.0)
    
    top_players = top_tbl.sort_values("pred_points", ascending=False).head(20) # <-- Changed to 20
    
    top_players.reset_index(drop=True, inplace=True)
    top_players.index = np.arange(1, len(top_players) + 1)
    top_players.index.name = "ลำดับ"
    
    cols_to_show = ["photo_url", "web_name", "team_short", "pos", "price", "form", "fixture_ease", "pred_points"]
    
    st.data_editor(
        top_players[cols_to_show],
        column_config={
            "photo_url": st.column_config.ImageColumn(
                "รูป", help="รูปนักเตะ", width="small"
            ),
            "web_name": st.column_config.TextColumn(
                "ชื่อนักเตะ", width="medium"
            ),
            "team_short": st.column_config.TextColumn(
                "ทีม", width="small"
            ),
            "pos": st.column_config.TextColumn(
                "ตำแหน่ง", width="small"
            ),
            "price": st.column_config.NumberColumn(
                "ราคา (£)", format="£%.1f"
            ),
            "form": st.column_config.NumberColumn(
                "ฟอร์ม", format="%.1f"
            ),
            "fixture_ease": st.column_config.NumberColumn(
                "ความง่าย", help="ความง่ายของเกมถัดไป", format="%.2f"
            ),
            "pred_points": st.column_config.NumberColumn(
                "คะแนนคาดการณ์", format="%.1f"
            ),
        },
        column_order=("ลำดับ", "photo_url", "web_name", "team_short", "pos", "price", "form", "fixture_ease", "pred_points"),
        use_container_width=True,
        height=750, # <-- Changed height for 20 players
        disabled=True # Read-only
    )
    st.markdown("---")

    # --- 4. Value Scatter Plot ---
    st.subheader("💰 กราฟนักเตะคุ้มค่า (Value Finder)")
    st.markdown("🪄 เอาเมาส์ไปชี้เพื่อดูชื่อนักเตะได้เลย!แต่ละจุดบอกตำแหน่ง ส่วนจุดใกล้มุมซ้ายบนคือของดีราคาถูก 💰")
    value_df = feat_df[feat_df['pred_points'] > 2.0].copy() # Filter out duds
    value_df['price'] = value_df['now_cost'] / 10.0
    value_df['position'] = value_df['element_type'].map(POSITIONS)
    
    chart = alt.Chart(value_df).mark_circle().encode(
        x=alt.X('price', title='ราคา (£)'),
        y=alt.Y('pred_points', title='คะแนนคาดการณ์'),
        color='position',
        tooltip=['web_name', 'team_short', 'price', 'pred_points'] # Add name and team
    ).interactive() # Make it zoomable/pannable
    
    st.altair_chart(chart, use_container_width=True)
    st.caption("หมายเหตุ: การแสดงรูปนักเตะใน tooltip ของกราฟนี้ยังไม่รองรับครับ")
    st.markdown("---")

    # --- 5. Fixture Difficulty ---
    st.subheader("🗓️ ตารางแข่ง 5 นัดล่วงหน้า (Fixture Planner)")
    st.markdown("เรียงตามความยากง่ายของตารางแข่งขัน (สีเขียว = ง่าย, สีแดง = ยาก)")
    
    # --- NEW (v1.9.3): Display Visual HTML Heatmap ---
    display_visual_fixture_planner(opp_matrix, diff_matrix, teams_df)
    
    # Display Rotation Pairs
    st.markdown("#### 🔄 Top 5 คู่ผู้รักษาประตู (GK Rotation Pairs)")
    st.caption(f"ค้นหาคู่ GK ที่ตารางแข่งสลับกันดีที่สุด (งบรวมไม่เกิน £9.0m)")
    st.dataframe(rotation_pairs, use_container_width=True, hide_index=True)
    st.markdown("---")
    
    # --- 6. Player Trends (Now 3 columns) ---
    st.subheader("🔥 นักเตะน่าสนใจ (Player Trends)")
    col1, col2, col3 = st.columns(3) # <-- Changed to 3
    
    with col1:
        st.markdown("#### 🔥 Top 5 ฟอร์มแรง (Form)")
        on_fire = feat_df.nlargest(5, 'form')
        for _, row in on_fire.iterrows():
            c1, c2 = st.columns([1, 3])
            with c1: 
                # --- MODIFIED (v1.9.6): Use HTML fallback ---
                st.markdown(get_player_image_html(row['photo_url'], row['web_name'], 50), unsafe_allow_html=True)
            with c2: st.markdown(f"**{row['web_name']}**"); st.caption(f"ฟอร์ม: {row['form']:.1f}")
    
    with col2:
        st.markdown("#### 💎 Top 5 ตัวแรร์ (<10% Owned)")
        diffs = feat_df[feat_df['selected_by_percent'] < 10.0].nlargest(5, 'pred_points')
        for _, row in diffs.iterrows():
            c1, c2 = st.columns([1, 3])
            with c1: 
                # --- MODIFIED (v1.9.6): Use HTML fallback ---
                st.markdown(get_player_image_html(row['photo_url'], row['web_name'], 50), unsafe_allow_html=True)
            with c2: st.markdown(f"**{row['web_name']}**"); st.caption(f"คะแนน: {row['pred_points']:.1f} | คนมี: {row['selected_by_percent']:.1f}%")

    with col3:
        st.markdown("#### 👥 Top 5 ขวัญใจมหาชน (Most Owned)")
        most_owned = feat_df.nlargest(5, 'selected_by_percent')
        for _, row in most_owned.iterrows():
            c1, c2 = st.columns([1, 3])
            with c1: 
                # --- MODIFIED (v1.9.6): Use HTML fallback ---
                st.markdown(get_player_image_html(row['photo_url'], row['web_name'], 50), unsafe_allow_html=True)
            with c2: st.markdown(f"**{row['web_name']}**"); st.caption(f"คนมี: {row['selected_by_percent']:.1f}%")


###############################
# Streamlit UI
###############################

def main():
    # st.set_page_config(page_title="FPL WIZ จัดตัวนักเตะ", layout="wide") # Moved to top
    st.title("🏟️ FPL WIZ จัดตัวนักเตะด้วย AI | FPL WIZ AI-Powered 🤖")
    st.markdown("เครื่องมือช่วยวิเคราะห์และแนะนำนักเตะ FPL ในแต่ละสัปดาห์ 🧠")
    
    # Add CSS for table styling
    add_table_css()

    with st.sidebar:
        st.header("⚙️ Settings | ตั้งค่า")

        # Callback function to clear the text input
        def reset_team_id():
            st.session_state.team_id_input = ""
            # --- NEW: Clear simulation state on reset ---
            if 'simulated_squad_ids' in st.session_state:
                del st.session_state['simulated_squad_ids']
            if 'current_team_id' in st.session_state:
                del st.session_state['current_team_id']
            # --- BUGFIX: Clear submitted state ---
            if 'analysis_submitted' in st.session_state:
                st.session_state.analysis_submitted = False

        # Create a form to handle the main analysis submission
        with st.form("settings_form"):
            entry_id_str = st.text_input(
                "Your FPL Team ID (ระบุ ID ทีมของคุณ)",
                key="team_id_input",
                help="นำเลข ID ทีมของคุณจาก URL หลังจากเข้าสู่ระบบ FPL บนเว็บแล้ว Click ดู Points จะเห็น URL https://fantasy.premierleague.com/entry/xxxxxxx/event/2 ให้นำเลข xxxxxxx มาใส่"
            )
            
            transfer_strategy = st.radio(
                "Transfer Strategy (เลือกรูปแบบการเปลี่ยนตัว)",
                ("Free Transfer", "Allow Hit (AI Suggest)", "Wildcard / Free Hit")
            )

            free_transfers = 1
            if transfer_strategy == "Free Transfer":
                free_transfers = st.number_input(
                    "เลือกจำนวนย้ายตัวฟรีที่คุณมี (สูงสุด 5 ตัว)",
                    min_value=0,
                    max_value=5,
                    value=1,
                    help="Select how many free transfers you have available (0-5)"
                )
        
            elif transfer_strategy == "Allow Hit (AI Suggest)":
                free_transfers = 1
        
        # ปุ่ม Analyze Team
            
            submitted = st.form_submit_button(
                label="Analyze Team",
                help="คลิกเพื่อวิเคราะห์ทีมของคุณ",
                use_container_width=False
            )
            
            # --- BUGFIX: Set session state on submission ---
            if submitted:
                st.session_state.analysis_submitted = True
            
            st.markdown(
            """
            <style>
            div[data-testid="stFormSubmitButton"] button {
                background-color: #4CAF50;
                color: white;
            }
            div[data-testid="stFormSubmitButton"] button:hover {
                background-color: #FF9800; /* สีส้มเมื่อ hover */
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Create a reset button outside of the form with an on_click callback
        st.button("Reset", on_click=reset_team_id, help="ล้างค่า ID และรีเฟรชหน้าจอ", type="primary")

        st.markdown(
            """
            <hr style="border-top: 1px solid #bbb;">
            <a href="https://www.kengji.co/2025/08/30/fpl-wiz/" target="_blank">
                <button style="width: 100%; font-size: 16px; padding: 10px; border-radius: 8px;">
                    คู่มือการใช้งาน 📖
                </button>
            </a>
            <hr style="border-top: 1px solid #bbb;">
            """,
            unsafe_allow_html=True
        )
    
    bootstrap = get_bootstrap()
    fixtures = get_fixtures()
    elements, teams, events, fixtures_df = build_master_tables(bootstrap, fixtures)
    cur_event, next_event = current_and_next_event(bootstrap.get("events", []))
    
    if not next_event and not cur_event:
        st.error("Could not determine the current or next gameweek.")
        st.stop()
        
    target_event = next_event or (cur_event + 1 if cur_event else 1)
    
    # ดึงข้อมูล deadline ของ target_event
    target_event_info = next(
        (e for e in bootstrap.get("events", []) if e.get("id") == target_event),
        None
    )
    
    deadline_text = ""
    if target_event_info and target_event_info.get("deadline_time"):
        from datetime import datetime
        import pytz
        
        # แปลงเวลา UTC จาก API ให้เป็นเวลาท้องถิ่น
        utc_time = datetime.fromisoformat(target_event_info["deadline_time"].replace("Z", "+00:00"))
        local_tz = pytz.timezone('Asia/Bangkok') # หรือ 'UTC'
        local_time = utc_time.astimezone(local_tz)
        
        # จัดรูปแบบการแสดงผลที่อ่านง่าย
        deadline_text = f" | ⏳ Deadline: **{local_time.strftime('%a, %d %b %H:%M %Z')}**"

    st.info(f"📅 สัปดาห์ที่: **{cur_event}** | วิเคราะห์เกมสัปดาห์ที่: **{target_event}**{deadline_text}")

    # --- โค้ดที่เพิ่มใหม่เพื่อแสดงข้อมูล DGW/BGW (ย้ายมาตรงนี้เพื่อให้ข้อมูล 'nf' พร้อมใช้) ---
    nf = next_fixture_features(fixtures_df, teams, target_event)
    
    feat = engineer_features(elements, teams, nf)
    feat.set_index('id', inplace=True)
    feat["pred_points"] = feat["pred_points_heur"]

    # --- START: Create player search map for simulation ---
    # We need a stable list for selectbox options, sorted by name
    feat_sorted = feat.sort_values('web_name')
    player_search_map = {
        f"{row['web_name']} ({row['team_short']}) - £{row['now_cost']/10.0}m": idx
        for idx, row in feat_sorted.iterrows()
    }
    # Create a reverse map from ID back to the string name
    player_id_to_name_map = {v: k for k, v in player_search_map.items()}
    # This is the list of options for the selectbox
    all_player_name_options = list(player_search_map.keys())
    # --- END: Create player search map ---


    # --- BUGFIX: Change main logic to check session state ---
    if not st.session_state.get('analysis_submitted', False):
        
        try:
            # --- NEW: Display the full home dashboard (v1.9.0) ---
            st.header(f"ภาพรวมสัปดาห์ที่ {target_event} (GW{target_event} Overview)")
            
            # --- NEW: Generate Fixture Planner data ---
            opponent_matrix, difficulty_matrix = get_fixture_difficulty_matrix(fixtures_df, teams, target_event)
            rotation_pairs = find_rotation_pairs(difficulty_matrix, teams, feat)

            display_home_dashboard(feat, nf, teams, opponent_matrix, difficulty_matrix, rotation_pairs)
        
        except Exception as e:
            st.error(f"Error creating home dashboard: {e}")
            st.exception(e) # Show full error
        
        # Show landing page info only if not submitted
        st.markdown("---")
        st.error("❗กรุณากรอก FPL Team ID ของคุณในช่องด้านข้างเพื่อเริ่มการวิเคราะห์")
        st.info("💡 FPL Team ID จากเว็บไซต์ https://fantasy.premierleague.com/ คลิกที่ Points แล้วจะเห็น Team ID ตามตัวอย่างรูปด้านล่าง")
        st.markdown(
            """
            <style>
            .custom-image img {
                width: 100%;
                max-width: 800px;
                height: auto;
                display: block;
                margin: 0 auto;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="custom-image"><img src="https://mlkrw8gmc4ni.i.optimole.com/w:1920/h:1034/q:mauto/ig:avif/https://www.kengji.co/wp-content/uploads/2025/08/FPL-01-scaled.webp"></div>',
            unsafe_allow_html=True
        )

    # --- BUGFIX: Change main logic to check session state ---
    if st.session_state.get('analysis_submitted', False):
        # --- BUGFIX: Get entry_id_str from session_state ---
        entry_id_str = st.session_state.team_id_input
        
        if entry_id_str:
            # --- BUGFIX v1.5.1: Separate ID validation from main logic ---
            try:
                # 1. Validate ID first
                entry_id = int(entry_id_str)
            except (ValueError, TypeError):
                st.error("❗ กรุณากรอก FPL Team ID เป็นตัวเลขเท่านั้น (Invalid ID)")
                st.session_state.analysis_submitted = False 
                st.stop()

            # 2. If ID is valid, proceed with main data fetching and logic
            try:
                entry = get_entry(entry_id)
                ev_for_picks = cur_event or 1
                picks = get_entry_picks(entry_id, ev_for_picks)

                # ========== การจัดการข้อมูล selling_price ที่อาจหายไป ==========
                picks_data = picks.get("picks", [])
                
                if not picks_data:
                    st.error(f"ไม่สามารถดึงข้อมูลนักเตะสำหรับ Team ID {entry_id} (อาจเป็นเพราะยังไม่เริ่มฤดูกาล หรือ ID ผิด)")
                    st.session_state.analysis_submitted = False # Reset state
                    st.stop()

                # สร้าง selling_price_map สำหรับทุกผู้เล่นในทีม
                selling_price_map = {}
                for p in picks_data:
                    player_id = p['element']
                    
                    if 'selling_price' in p and p['selling_price'] is not None:
                        # ถ้ามี selling_price ใน API ใช้เลย
                        selling_price_map[player_id] = p['selling_price']
                    elif 'purchase_price' in p and p['purchase_price'] is not None:
                        # คำนวณ selling_price จาก purchase_price
                        purchase_price = p['purchase_price']
                        
                        # Safety check if player_id is not in feat (e.g., transferred out)
                        if player_id not in feat.index:
                            now_cost = purchase_price # Assume no change if not found
                        else:
                            now_cost = feat.loc[player_id, 'now_cost']
                            
                        profit = now_cost - purchase_price
                        selling_price = purchase_price + (profit // 2)  # ปัดเศษลง
                        selling_price_map[player_id] = selling_price
                    else:
                        # fallback สุดท้าย: ใช้ราคาปัจจุบัน
                        if player_id not in feat.index:
                            selling_price_map[player_id] = 0 # Or some default
                        else:
                            selling_price_map[player_id] = feat.loc[player_id, 'now_cost']


                # อัปเดต selling_price ใน DataFrame
                feat['selling_price'] = feat.index.map(selling_price_map)
                feat['selling_price'].fillna(feat['now_cost'], inplace=True)
                # ========== จบการจัดการข้อมูล selling_price ==========         

                st.header(f"🚀 Analysis for '{entry['name']}'")

                if transfer_strategy == "Wildcard / Free Hit":
                    st.subheader("🤖 AI Suggested Wildcard / Free Hit Team")
                    total_value = (entry.get('last_deadline_value', 1000) + entry.get('last_deadline_bank', 0)) / 10.0
                    st.info(f"Optimizing for a total budget of **£{total_value:.1f}m**")
                    
                    with st.spinner("Finding the optimal 15-man squad... this may take a moment."):
                        wildcard_ids = optimize_wildcard_team(feat, total_value)
                    
                    if wildcard_ids:
                        squad_df = feat.loc[wildcard_ids].copy()
                        xi_ids, bench_ids = optimize_starting_xi(squad_df)
                    else:
                        st.error("Could not find an optimal wildcard team. This might be due to budget constraints or player availability.")
                        st.stop()
                    
                    # --- NEW: Pitch View for Wildcard ---
                    xi_df = squad_df.loc[xi_ids].copy()
                    
                    # Add Captain/Vice
                    cap_id = xi_df.sort_values("pred_points", ascending=False).iloc[0].name
                    vc_id = xi_df.sort_values("pred_points", ascending=False).iloc[1].name
                    xi_df['is_captain'] = xi_df.index == cap_id
                    xi_df['is_vice_captain'] = xi_df.index == vc_id

                    # --- BUGFIX v1.5.2: Correct st.tabs syntax ---
                    tab_pitch_wc, tab_list_wc = st.tabs(["Pitch View ⚽", "List View 📋"])

                    with tab_pitch_wc:
                        with st.container(border=False):
                            display_pitch_view(xi_df, "✅ แนะนำนักเตะ 11 ตัวจริง (Suggested Starting XI)")
                    
                    with tab_list_wc:
                        with st.container(border=False):
                            st.subheader("✅ แนะนำนักเตะ 11 ตัวจริง (List View)")
                            xi_df_list = xi_df.copy()
                            xi_df_list['pos'] = xi_df_list['element_type'].map(POSITIONS)
                            position_order = ['GK', 'DEF', 'MID', 'FWD']
                            xi_df_list['pos'] = pd.Categorical(xi_df_list['pos'], categories=position_order, ordered=True)
                            xi_df_list = xi_df_list.sort_values('pos')
                            xi_display_df = xi_df_list[['web_name', 'team_short', 'pos', 'pred_points']]
                            display_user_friendly_table(
                                df=xi_display_df,
                                title="", # Title is handled by tab
                                height=420
                            )

                    st.success(f"👑 Captain: **{xi_df.loc[cap_id]['web_name']}** | Vice-Captain: **{xi_df.loc[vc_id]['web_name']}**")
                    
                    bench_df = squad_df.loc[bench_ids].copy()
                    bench_gk = bench_df[bench_df['element_type'] == 1]
                    bench_outfield = bench_df[bench_df['element_type'] != 1].sort_values('pred_points', ascending=False)
                    ordered_bench_df = pd.concat([bench_gk, bench_outfield])
                    ordered_bench_df['pos'] = ordered_bench_df['element_type'].map(POSITIONS)
                    
                    bench_display_df = ordered_bench_df[['web_name', 'team_short', 'pos', 'pred_points']]
                    display_user_friendly_table(
                        df=bench_display_df,
                        title="ตัวสำรอง (เรียงตามลำดับ)",
                        height=175
                    )
                    
                    total_points = squad_df['pred_points'].sum()
                    total_cost = squad_df['now_cost'].sum() / 10.0
                    st.success(f"Total Expected Points: **{total_points:.1f}** | Team Value: **£{total_cost:.1f}m**")

                
                else:  # Free Transfer / Allow Hit
                    bank = (entry.get('last_deadline_bank', 0)) / 10.0
                    free_transfers_from_api = entry.get('free_transfers', 1)

                    pick_ids = [p["element"] for p in picks.get("picks", [])]
                    
                    # --- NEW: Initialize session state for simulation ---
                    if 'simulated_squad_ids' not in st.session_state:
                        st.session_state.simulated_squad_ids = pick_ids
                    
                    # Check if the team ID has changed, if so, reset the simulated squad
                    if 'current_team_id' not in st.session_state or st.session_state.current_team_id != entry_id:
                        st.session_state.simulated_squad_ids = pick_ids
                        st.session_state.current_team_id = entry_id
                    
                    # Filter out any pick_ids that might not be in the 'feat' dataframe (e.g. transferred players)
                    valid_pick_ids = [pid for pid in pick_ids if pid in feat.index]
                    if len(valid_pick_ids) < len(pick_ids):
                        st.warning("Some players in your squad could not be found (e.g., transferred out of PL) and were removed.")
                    
                    squad_df = feat.loc[valid_pick_ids].copy() # This is the REAL squad from API
                    
                    overall_points = entry.get('summary_overall_points', 0)
                    gameweek_points = entry.get('summary_event_points', 0)
                    
                    st.info(f"🏦 Bank: **£{bank:.1f}m** | 🆓 Free Transfer: **{free_transfers_from_api}** | 🎯 Overall points: **{overall_points}** | Gameweek points: **{gameweek_points}**")

                    # --- Original XI/Bench Display (from API data) ---
                    st.subheader("📊 ทีมปัจจุบัน (ข้อมูลจาก FPL API)")
                    
                    if len(valid_pick_ids) < 15:
                        st.error(f"ทีมของคุณมีนักเตะไม่ครบ 15 คน (พบ {len(valid_pick_ids)} คน). ไม่สามารถจัดทีมได้.")
                    else:
                        xi_ids, bench_ids = optimize_starting_xi(squad_df)

                        if not xi_ids or len(xi_ids) != 11:
                            st.error("ไม่สามารถจัด 11 ตัวจริงจากทีมปัจจุบันของคุณได้ (API).")
                            squad_display_df = squad_df[['web_name', 'element_type']].rename(columns={'element_type':'pos'})
                            squad_display_df['pos'] = squad_display_df['pos'].map(POSITIONS)
                            st.dataframe(squad_display_df)
                        else:
                            # --- NEW: Pitch View for API Team ---
                            xi_df = squad_df.loc[xi_ids].copy()
                            
                            # Add Captain/Vice
                            cap_id = xi_df.sort_values("pred_points", ascending=False).iloc[0].name
                            vc_id = xi_df.sort_values("pred_points", ascending=False).iloc[1].name
                            xi_df['is_captain'] = xi_df.index == cap_id
                            xi_df['is_vice_captain'] = xi_df.index == vc_id

                            # --- BUGFIX v1.5.2: Correct st.tabs syntax ---
                            tab_pitch_api, tab_list_api = st.tabs(["Pitch View ⚽", "List View 📋"])

                            with tab_pitch_api:
                                with st.container(border=False):
                                    display_pitch_view(xi_df, "✅ แนะนำนักเตะ 11 ตัวจริง (Suggested Starting XI)")
                            
                            with tab_list_api:
                                with st.container(border=False):
                                    st.subheader("✅ แนะนำนักเตะ 11 ตัวจริง (List View)")
                                    xi_df_list = xi_df.copy()
                                    xi_df_list['pos'] = xi_df_list['element_type'].map(POSITIONS)
                                    position_order = ['GK', 'DEF', 'MID', 'FWD']
                                    xi_df_list['pos'] = pd.Categorical(xi_df_list['pos'], categories=position_order, ordered=True)
                                    xi_df_list = xi_df_list.sort_values('pos')
                                    xi_display_df = xi_df_list[['web_name', 'team_short', 'pos', 'pred_points']]
                                    display_user_friendly_table(
                                        df=xi_display_df,
                                        title="", # Title is handled by tab
                                        height=420
                                    )

                            st.success(f"👑 Captain: **{xi_df.loc[cap_id]['web_name']}** | Vice-Captain: **{xi_df.loc[vc_id]['web_name']}**")
                            
                            xi_dgw_teams = xi_df[xi_df['num_fixtures'] > 1]['team_short'].unique()
                            xi_bgw_teams = xi_df[xi_df['num_fixtures'] == 0]['team_short'].unique()

                            dgw_note = ""
                            bgw_note = ""

                            if len(xi_dgw_teams) > 0:
                                dgw_note = f"สัปดาห์นี้มี Double Gameweek ของทีม ({', '.join(xi_dgw_teams)})"
                            if len(xi_bgw_teams) > 0:
                                bgw_note = f"สัปดาห์นี้มี Blank Gameweek ของทีม ({', '.join(xi_bgw_teams)})"

                            if dgw_note or bgw_note:
                                full_note = ""
                                if dgw_note and bgw_note:
                                    full_note = f"{dgw_note}. {bgw_note}."
                                elif dgw_note:
                                    full_note = f"{dgw_note}."
                                elif bgw_note:
                                    full_note = f"{bgw_note}."
                                st.info(f"💡 {full_note}")
                            
                            bench_df = squad_df.loc[bench_ids].copy()
                            bench_gk = bench_df[bench_df['element_type'] == 1]
                            bench_outfield = bench_df[bench_df['element_type'] != 1].sort_values('pred_points', ascending=False)
                            ordered_bench_df = pd.concat([bench_gk, bench_outfield])
                            ordered_bench_df['pos'] = ordered_bench_df['element_type'].map(POSITIONS)
                            
                            bench_display_df = ordered_bench_df[['web_name', 'team_short', 'pos', 'pred_points']]
                            display_user_friendly_table(
                                df=bench_display_df,
                                title="ตัวสำรอง (เรียงตามลำดับ)",
                                height=175
                            )
                    
                    st.markdown("---")
                    

                    # --- Original Transfer Suggestion Section ---
                    st.subheader("🔄 Suggested Transfers (Based on API Team)")
                    st.markdown(f"💡 คำแนะนำซื้อขายนักเตะจากทีมคุณ (ข้อมูลจาก API) ⚠️ *เนื่องจากข้อจำกัดของ FPL API เราแสดง 2 มุมมองเพื่อให้คุณตัดสินใจ* 🔎")
                    with st.spinner("Analyzing potential transfers..."):
                        normal_moves, conservative_moves = suggest_transfers_enhanced(
                            valid_pick_ids, bank=bank, free_transfers=free_transfers,
                            all_players=feat, strategy=transfer_strategy
                        )

                    if not normal_moves and not conservative_moves:
                        st.write("⚠️ ไม่มีคำแนะนำการซื้อขายนักเตะ ลองเปลี่ยนกลยุทธ์หรือเพิ่ม Free Transfer")
                    
                    else:
                        col1, col2 = st.columns(2)
                        
                        # =========================
                        # ตารางข้อเสนอหลัก (normal)
                        # =========================
                        with col1:
                            st.markdown("#### 📊 ข้อเสนอหลัก (ราคาปัจจุบัน)")
                            if normal_moves:
                                normal_df = pd.DataFrame(normal_moves)
                                normal_df.index = np.arange(1, len(normal_df) + 1)
                                
                                total_in = normal_df['in_cost'].sum()
                                total_out = normal_df['out_cost'].sum()
                                st.info(f"💰 งบประมาณ: ขายออก **£{total_out:.1f}m** | ซื้อเข้า **£{total_in:.1f}m**")
                                
                                # คำนวณความสูงไดนามิก
                                dynamic_height = 45 + (len(normal_df) * 35) 
                                
                                display_user_friendly_table(
                                    df=normal_df.rename(columns={
                                        "out_name": "ขายออก (Out)",
                                        "out_cost": "ราคาขาย (£)",
                                        "in_name": "ซื้อเข้า (In)",
                                        "in_cost": "ราคาซื้อ (£)",
                                        "in_points": "คะแนนคาดการณ์ (Pred Points)"
                                    })[["ขายออก (Out)", "ราคาขาย (£)", "ซื้อเข้า (In)", "ราคาซื้อ (£)", "คะแนนคาดการณ์ (Pred Points)"]],
                                    title="",
                                    height=dynamic_height
                                )
                            else:
                                st.write("ไม่มีการแนะนำ")
                        
                        # =============================
                        # ตารางข้อเสนอสำรอง (conserve)
                        # =============================
                        with col2:
                            st.markdown("#### 🛡️ ข้อเสนอสำรอง (ปรับราคาขายลง)")
                            if conservative_moves:
                                conservative_df = pd.DataFrame(conservative_moves)
                                conservative_df.index = np.arange(1, len(conservative_df) + 1)
                                
                                total_in_c = conservative_df['in_cost'].sum()
                                total_out_c = conservative_df['out_cost'].sum()
                                st.info(f"💰 งบประมาณ: ขายออก **£{total_out_c:.1f}m** | ซื้อเข้า **£{total_in_c:.1f}m**")
                                
                                # คำนวณความสูงไดนามิก
                                dynamic_height_c = 45 + (len(conservative_df) * 35)
                                
                                display_user_friendly_table(
                                    df=conservative_df.rename(columns={
                                        "out_name": "ขายออก (Out)",
                                        "out_cost": "ราคาขาย (£)",
                                        "in_name": "ซื้อเข้า (In)",
                                        "in_cost": "ราคาซื้อ (£)",
                                        "in_points": "คะแนนคาดการณ์ (Pred Points)"
                                    })[["ขายออก (Out)", "ราคาขาย (£)", "ซื้อเข้า (In)", "ราคาซื้อ (£)", "คะแนนคาดการณ์ (Pred Points)"]],
                                    title="",
                                    height=dynamic_height_c
                                )
                                
                                st.caption("🔍 ราคาขายลดลง 0.1-0.2m เผื่อกรณีราคาเปลี่ยนแปลง")
                            else:
                                st.write("ไม่มีการแนะนำที่ปลอดภัยพอ")
                        
                        # เพิ่มคำเตือน
                        st.warning("⚠️ **สำคัญ**: ตรวจสอบราคาขายจริงในแอป FPL ก่อนทำ transfer")
                    
                    st.markdown("---")
                    
                    # --- START: NEW SIMULATION SECTION (MOVED) ---
                    st.subheader("🛠️ ทดลองจัดทีม (Simulation Mode)")
                    st.markdown("ใช้ส่วนนี้เพื่อจำลองการย้ายทีมของคุณ *หลังจาก* ที่คุณกดยืนยันใน FPL แล้ว แต่ API ยังไม่อัปเดต")
                    
                    if st.button("♻️ Reset to Current API Team"):
                        st.session_state.simulated_squad_ids = valid_pick_ids # Use valid_pick_ids
                        st.rerun()

                    st.markdown("#### แก้ไข 15 นักเตะในทีมของคุณ:")
                    
                    new_simulated_ids = []
                    
                    # Prepare display columns
                    cols = st.columns([3, 1, 4])
                    cols[0].markdown("**นักเตะคนปัจจุบัน (ในโหมดจำลอง)**")
                    cols[2].markdown("**เลือกนักเตะที่จะเปลี่ยน**")
                    
                    # Use the session_state list as the source of truth
                    current_sim_ids = st.session_state.get('simulated_squad_ids', valid_pick_ids) # Use valid_pick_ids
                    
                    # Ensure current_sim_ids has 15 players, if not, reset
                    if len(current_sim_ids) != 15:
                        st.warning("ทีมจำลองของคุณไม่ครบ 15 คน, กำลังรีเซ็ต...")
                        current_sim_ids = valid_pick_ids
                        st.session_state.simulated_squad_ids = valid_pick_ids


                    for i, player_id in enumerate(current_sim_ids):
                        # Safety check if player_id is valid
                        if player_id not in feat.index:
                            st.warning(f"Player ID {player_id} not found in data. Resetting to default.")
                            player_id = valid_pick_ids[i] # Reset to default
                            current_sim_ids[i] = player_id
                        
                        player = feat.loc[player_id]
                        current_player_name_str = player_id_to_name_map.get(player_id)
                        
                        # Fallback if player ID isn't in the map (e.g., player transferred out of PL)
                        if not current_player_name_str:
                             current_player_name_str = f"{player['web_name']} ({player['team_short']}) - £{player['now_cost']/10.0}m"
                             if current_player_name_str not in player_search_map:
                                 # Add them temporarily to the list to make selectbox work
                                all_player_name_options.append(current_player_name_str)
                                player_search_map[current_player_name_str] = player_id
                                player_id_to_name_map[player_id] = current_player_name_str

                        # Find the index in the options list
                        try:
                            # Check if the key already has a value in session_state (from a previous change)
                            key_name = f"sim_player_{i}"
                            if key_name in st.session_state:
                                selected_name_from_state = st.session_state[key_name]
                                current_index = all_player_name_options.index(selected_name_from_state)
                            else:
                                current_index = all_player_name_options.index(current_player_name_str)
                        except (ValueError, KeyError):
                            current_index = 0 # Default to first player if something goes wrong

                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 4])
                            # Show current player info
                            col1.text(f"{i+1}. {player['web_name']} ({POSITIONS[player['element_type']]})")
                            col2.text("➡️")
                            
                            # Show selectbox for replacement
                            selected_name = col3.selectbox(
                                f"Select player {i+1}",
                                options=all_player_name_options,
                                index=current_index,
                                key=key_name, # Use the key_name defined above
                                label_visibility="collapsed"
                            )
                            new_player_id = player_search_map[selected_name]
                            new_simulated_ids.append(new_player_id)
                    
                    # Update session state *if* there's a change
                    if new_simulated_ids != current_sim_ids:
                        st.session_state.simulated_squad_ids = new_simulated_ids
                        # Rerun to update the text display in col1
                        st.rerun()

                    st.markdown("---")
                    
                    # --- Simulation Analysis Button ---
                    if st.button("คำนวณ 11 ตัวจริง (Simulated Team)", type="primary"):
                        sim_ids = st.session_state.simulated_squad_ids
                        sim_squad_df = feat.loc[sim_ids]

                        # --- Validation ---
                        errors = []
                        counts = sim_squad_df['element_type'].value_counts().to_dict()
                        if counts.get(1, 0) != 2:
                            errors.append(f"❌ ผู้รักษาประตู: {counts.get(1, 0)} (ต้องมี 2)")
                        if counts.get(2, 0) != 5:
                            errors.append(f"❌ กองหลัง: {counts.get(2, 0)} (ต้องมี 5)")
                        if counts.get(3, 0) != 5:
                            errors.append(f"❌ กองกลาง: {counts.get(3, 0)} (ต้องมี 5)")
                        if counts.get(4, 0) != 3:
                            errors.append(f"❌ กองหน้า: {counts.get(4, 0)} (ต้องมี 3)")

                        team_counts = sim_squad_df['team_short'].value_counts()
                        for team, count in team_counts[team_counts > 3].items():
                            errors.append(f"❌ ทีม {team}: มี {count} คน (สูงสุด 3)")
                        
                        if errors:
                            st.error("ไม่สามารถจัดทีมได้! กรุณาแก้ไขทีมจำลองของคุณ:")
                            for error in errors:
                                st.write(error)
                        else:
                            # --- Run Optimization on Simulated Team ---
                            st.success("✅ ทีมจำลองถูกต้องตามกฎ FPL! กำลังคำนวณ...")
                            
                            # Calculate cost
                            original_budget = (entry.get('last_deadline_value', 1000) + entry.get('last_deadline_bank', 0)) / 10.0
                            total_cost = sim_squad_df['now_cost'].sum() / 10.0
                            diff = original_budget - total_cost
                            
                            if diff < 0:
                                st.warning(f"มูลค่าทีมจำลอง: **£{total_cost:.1f}m** | งบประมาณติดลบ: **£{diff:.1f}m**")
                            else:
                                st.info(f"มูลค่าทีมจำลอง: **£{total_cost:.1f}m** | งบประมาณคงเหลือ: **£{diff:.1f}m**")
                            
                            
                            xi_ids_sim, bench_ids_sim = optimize_starting_xi(sim_squad_df)
                            
                            if not xi_ids_sim or len(xi_ids_sim) != 11:
                                st.error("เกิดข้อผิดพลาดในการคำนวณ 11 ตัวจริง (Simulated)")
                            else:
                                # --- NEW: Pitch View for Simulated Team ---
                                xi_df_sim = sim_squad_df.loc[xi_ids_sim].copy()
                                
                                # Add Captain/Vice
                                cap_id_sim = xi_df_sim.sort_values("pred_points", ascending=False).iloc[0].name
                                vc_id_sim = xi_df_sim.sort_values("pred_points", ascending=False).iloc[1].name
                                xi_df_sim['is_captain'] = xi_df_sim.index == cap_id_sim
                                xi_df_sim['is_vice_captain'] = xi_df_sim.index == vc_id_sim

                                # --- BUGFIX v1.5.2: Correct st.tabs syntax ---
                                tab_pitch_sim, tab_list_sim = st.tabs(["Pitch View ⚽", "List View 📋"])

                                with tab_pitch_sim:
                                    with st.container(border=False):
                                        display_pitch_view(xi_df_sim, "✅ 11 ตัวจริง (Simulated Team)")
                                
                                with tab_list_sim:
                                    with st.container(border=False):
                                        st.subheader("✅ 11 ตัวจริง (Simulated List View)")
                                        xi_df_sim_list = xi_df_sim.copy()
                                        xi_df_sim_list['pos'] = xi_df_sim_list['element_type'].map(POSITIONS)
                                        position_order = ['GK', 'DEF', 'MID', 'FWD']
                                        xi_df_sim_list['pos'] = pd.Categorical(xi_df_sim_list['pos'], categories=position_order, ordered=True)
                                        xi_df_sim_list = xi_df_sim_list.sort_values('pos')
                                        xi_display_df_sim = xi_df_sim_list[['web_name', 'team_short', 'pos', 'pred_points']]
                                        display_user_friendly_table(
                                            df=xi_display_df_sim,
                                            title="", # Title is handled by tab
                                            height=420
                                        )
                                
                                st.success(f"👑 Captain (Simulated): **{xi_df_sim.loc[cap_id_sim]['web_name']}** | Vice: **{xi_df_sim.loc[vc_id_sim]['web_name']}**")
                                
                                # Display Bench
                                bench_df = sim_squad_df.loc[bench_ids_sim].copy()
                                bench_gk = bench_df[bench_df['element_type'] == 1]
                                bench_outfield = bench_df[bench_df['element_type'] != 1].sort_values('pred_points', ascending=False)
                                ordered_bench_df = pd.concat([bench_gk, bench_outfield])
                                ordered_bench_df['pos'] = ordered_bench_df['element_type'].map(POSITIONS)
                                
                                bench_display_df = ordered_bench_df[['web_name', 'team_short', 'pos', 'pred_points']]
                                display_user_friendly_table(
                                    df=bench_display_df,
                                    title="ตัวสำรอง (Simulated Team - เรียงตามลำดับ)",
                                    height=175
                                )
                    
                    st.markdown("---")
                    # --- END: NEW SIMULATION SECTION (MOVED) ---

            except requests.exceptions.HTTPError as e:
                st.error(f"Could not fetch data for Team ID {entry_id_str}. Please check if the ID is correct. (Error: {e.response.status_code})")
                st.session_state.analysis_submitted = False # Reset state
            except Exception as e:
                # --- BUGFIX v1.5.1: Catch other processing errors ---
                st.error(f"An unexpected error occurred while processing your team data: {e}")
                st.session_state.analysis_submitted = False 
                st.exception(e) # Print the full error for debugging
        else:
            # This handles the case where the button is 'submitted' but the text box is empty
            st.error("❗กรุณากรอก FPL Team ID ของคุณในช่องด้านข้างเพื่อเริ่มการวิเคราะห์")
            st.session_state.analysis_submitted = False # Reset state
            # (The landing page info will be shown by the `if not st.session_state.get...` block)


if __name__ == "__main__":
    main()