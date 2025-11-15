"""
FPL Weekly Assistant — single-file Streamlit app (Thai Updated Version v2.0 FULL)

What it does
- Pulls live data from FPL API (bootstrap-static, fixtures, entry picks)
- Engineers features (recent form, xGI proxy, minutes reliability, fixture difficulty, photo_url)
- Predicts next GW points with a hybrid approach (Enhanced with Understat xG/xA)
- Optimizes your Starting XI & bench order
- Suggests transfers based on selected strategy
- **NEW**: Transfer ROI Calculator (3-GW Projection)
- Home Dashboard v1.9.7 (Added Understat xG/xA/xPTS section)
- Visual Fixture Planner with Logos (v1.9.3)
- Displays Starting XI in a "Pitch View" or "List View"
- Includes a "Simulation Mode" to manually edit your 15-man squad

How to run
1) pip install streamlit pandas numpy scikit-learn pulp requests altair beautifulsoup4
2) streamlit run fpl.py

Notes
- This app reads public FPL endpoints. No login required.
"""
###############################
# V2.0 - Enhanced Features & ROI Calculator
###############################

import os
import math
import json
import time
import re # <-- NEW IMPORT for Understat
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, LpStatus, PULP_CBC_CMD
from bs4 import BeautifulSoup # <-- NEW IMPORT for Understat

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
# --- NEW: Understat Data Functions ---
###############################

# This hardcoded map links Understat's full team name to FPL's 'name'
UNDERSTAT_TEAM_TO_FPL_NAME = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich": "Ipswich",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Southampton": "Southampton",
    "Tottenham": "Spurs",
    "West Ham": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    # Add other teams as needed if they change
    "Leeds": "Leeds",
    "Burnley": "Burnley",
    "Sheffield United": "Sheffield Utd",
    "Luton": "Luton"
}

@st.cache_data(ttl=3600) # Cache for 1 hour
def get_understat_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetches player and team data from Understat.com."""
    try:
        url = "https://understat.com/league/EPL"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find script tags
        scripts = soup.find_all('script')
        
        players_data_str = None
        teams_data_str = None
        
        for script in scripts:
            if script.string and 'playersData' in script.string:
                players_data_str = script.string
            if script.string and 'teamsData' in script.string:
                teams_data_str = script.string

        # 1. Process Players Data
        players_df = pd.DataFrame()
        if players_data_str:
            match = re.search(r"var playersData\s*=\s*JSON\.parse\('(.+?)'\);", players_data_str)
            if match:
                json_data = match.group(1).encode('utf-8').decode('unicode_escape')
                players_data = json.loads(json_data)
                players_df = pd.DataFrame(players_data)
                # Convert necessary columns
                players_df['xG'] = pd.to_numeric(players_df['xG'], errors='coerce')
                players_df['xA'] = pd.to_numeric(players_df['xA'], errors='coerce')
                players_df = players_df[['id', 'player_name', 'team_title', 'xG', 'xA']]
            
        # 2. Process Teams Data
        teams_df = pd.DataFrame()
        if teams_data_str:
            match = re.search(r"var teamsData\s*=\s*JSON\.parse\('(.+?)'\);", teams_data_str)
            if match:
                json_data = match.group(1).encode('utf-8').decode('unicode_escape')
                teams_data = json.loads(json_data)
                
                team_list = []
                
                # --- START FIX ---
                # โค้ดเดิมใช้ 'team_name' (ซึ่งเป็น ID เช่น "80")
                # เราต้องใช้ 'team_data['title']' (ซึ่งเป็นชื่อเช่น "Arsenal")
                for team_id_key, team_data in teams_data.items():
                # --- END FIX ---
                
                    # Get the last entry in history for the most up-to-date xPTS
                    if team_data.get('history'):
                        latest_stats = team_data['history'][-1]
                        team_list.append({
                            # --- START FIX ---
                            'title': team_data.get('title'), # <-- แก้ไขตรงนี้: ดึงชื่อทีมจาก value
                            # --- END FIX ---
                            'xpts': latest_stats.get('xpts', 0)
                        })
                teams_df = pd.DataFrame(team_list)
                teams_df['xpts'] = pd.to_numeric(teams_df['xpts'], errors='coerce')

        return players_df, teams_df

    except Exception as e:
        # st.warning(f"Could not fetch Understat data: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
def check_name_match(fpl_name: str, understat_name: str) -> bool:
    """Checks if FPL web_name matches Understat player_name."""
    if not fpl_name or not understat_name:
        return False
        
    fpl_name_lower = str(fpl_name).lower()
    understat_name_lower = str(understat_name).lower()
    
    # Direct match (e.g., "Salah" in "mohamed salah")
    if fpl_name_lower in understat_name_lower:
        return True
        
    # Check last name
    try:
        fpl_last = fpl_name_lower.split(' ')[-1]
        understat_last = understat_name_lower.split(' ')[-1]
        if fpl_last == understat_last:
            return True
    except Exception:
        pass # Ignore splitting errors
        
    return False

@st.cache_data(ttl=3600)
def merge_understat_data(
    us_players_df: pd.DataFrame, 
    us_teams_df: pd.DataFrame, 
    fpl_players_df: pd.DataFrame, 
    fpl_teams_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merges Understat data with FPL data for photos and logos."""
    
    # 1. Merge Teams
    merged_teams = pd.DataFrame()
    if not us_teams_df.empty and not fpl_teams_df.empty:
        try:
            # Map Understat team name to FPL team name
            us_teams_df['fpl_name'] = us_teams_df['title'].map(UNDERSTAT_TEAM_TO_FPL_NAME)
            # Merge with FPL teams on name
            merged_teams = us_teams_df.merge(
                fpl_teams_df[['name', 'logo_url']], 
                left_on='fpl_name', 
                right_on='name',
                how='left'
            )
            
            # --- START EDIT ---
            # เราจะเก็บ 'name' (ชื่อ FPL) และ 'title' (ชื่อ Understat) ไว้ทั้งคู่
            merged_teams = merged_teams[['title', 'name', 'xpts', 'logo_url']].sort_values('xpts', ascending=False)
            # --- END EDIT ---

        except Exception as e:
            st.warning(f"Error merging Understat teams: {e}")

    # 2. Merge Players
    merged_players = pd.DataFrame()
    if not us_players_df.empty and not fpl_players_df.empty and not fpl_teams_df.empty:
        try:
            # Create a lookup for FPL team name -> FPL team id
            fpl_name_to_id_map = fpl_teams_df.set_index('name')['id'].to_dict()
            
            # Map Understat team title -> FPL team name -> FPL team id
            us_players_df['fpl_name'] = us_players_df['team_title'].map(UNDERSTAT_TEAM_TO_FPL_NAME)
            us_players_df['fpl_team_id'] = us_players_df['fpl_name'].map(fpl_name_to_id_map)
            
            # Get the columns we need from FPL data
            fpl_lookup = fpl_players_df[['team', 'web_name', 'photo_url', 'team_short', 'goals_scored', 'assists']].copy()
            
            # Merge Understat players with FPL lookup on team ID
            # This creates many potential matches for each team
            combined_df = us_players_df.merge(
                fpl_lookup, 
                left_on='fpl_team_id', 
                right_on='team',
                how='inner'
            )
            
            # Find the correct name match
            combined_df['name_match'] = combined_df.apply(
                lambda row: check_name_match(row['web_name'], row['player_name']), 
                axis=1
            )
            
            # Filter for correct matches
            final_players = combined_df[combined_df['name_match'] == True].copy()
            
            # Clean up and drop duplicates
            # A player might match multiple (e.g., "Jota" matches "Diogo Jota")
            # We sort by xG and take the best, this isn't perfect but good enough
            final_players = final_players.sort_values('xG', ascending=False)
            merged_players = final_players.drop_duplicates(subset=['id', 'player_name'])
            
            merged_players = merged_players[[
                'player_name', 'team_short', 'photo_url', 'xG', 'xA', 'goals_scored', 'assists'
            ]]
        except Exception as e:
            st.warning(f"Error merging Understat players: {e}")

    return merged_players, merged_teams

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

TEAM_MAP_COLS = ["id", "code", "name", "short_name", "strength_overall_home", "strength_overall_away",
                 "strength_attack_home", "strength_attack_away", "strength_defence_home", "strength_defence_away","position"]

# --- BUGFIX v1.9.1: Define POSITIONS globally ---
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

def build_master_tables(bootstrap: Dict, fixtures: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Constructs the main dataframes for players, teams, events, and fixtures."""
    elements = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"]) # ดึงมาทั้งหมดก่อน แล้วค่อยเลือกคอลัมน์

    # --- NEW: Calculate overall strengths ---
    # FPL API doesn't always provide overall strength directly, so we average home/away
    teams['strength_overall'] = (teams['strength_overall_home'] + teams['strength_overall_away']) / 2.0
    teams['strength_attack_overall'] = (teams['strength_attack_home'] + teams['strength_attack_away']) / 2.0
    teams['strength_defence_overall'] = (teams['strength_defence_home'] + teams['strength_defence_away']) / 2.0

    # Ensure we keep these new columns
    cols_to_keep = TEAM_MAP_COLS + ['strength_overall', 'strength_attack_overall', 'strength_defence_overall']
    # Filter only columns that actually exist to avoid errors if API changes
    cols_to_keep = [col for col in cols_to_keep if col in teams.columns]
    teams = teams[cols_to_keep].copy()
    
    events = pd.DataFrame(bootstrap.get("events", []))
    
    # --- BUGFIX (v1.8.3): Use 70px logos ---
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

# --- REPLACED: Old engineer_features with Enhanced Version (v2.0) ---
def engineer_features_enhanced(elements: pd.DataFrame, teams: pd.DataFrame, nf: pd.DataFrame, understat_players: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced feature engineering that incorporates Understat xG/xA into predictions.
    """
    elements = elements.copy()
    elements["element_type"] = pd.to_numeric(elements["element_type"], errors='coerce').fillna(0).astype(int)

    # 1. Merge basic fixture data
    elements = elements.merge(nf, on="team", how="left")
    elements['num_fixtures'] = elements['num_fixtures'].fillna(0).astype(int)
    elements['avg_fixture_ease'] = elements['avg_fixture_ease'].fillna(0)
    elements['opponent_str'] = elements['opponent_str'].fillna("-")

    # 2. Numeric conversion
    cols_to_process = [
        "form", "points_per_game", "ict_index", "selected_by_percent", "now_cost", 
        "minutes", "goals_scored", "assists", "clean_sheets",
        "cost_change_event","cost_change_start", "transfers_in_event", "transfers_out_event", "code"
    ]
    for col in cols_to_process:
        if col in elements.columns:
            elements[col] = pd.to_numeric(elements[col], errors="coerce").fillna(0)
            
    if 'cost_change_event' in elements.columns:
        elements['cost_change_event'] = elements['cost_change_event'].astype(int)

    elements['photo_url'] = 'https://resources.premierleague.com/premierleague/photos/players/110x140/p' + elements['code'].astype(int).astype(str) + '.png'
    elements["chance_of_playing_next_round"] = pd.to_numeric(elements["chance_of_playing_next_round"], errors="coerce").fillna(100)
    elements["play_prob"] = elements["chance_of_playing_next_round"] / 100.0

    # 3. Merge Understat Data
    # Using web_name approximation for simplicity. In production, ID mapping is preferred.
    if not understat_players.empty and 'xG' in understat_players.columns:
        # Temp lower case columns for better matching
        elements['web_name_lower'] = elements['web_name'].str.lower()
        understat_players['player_name_lower'] = understat_players['player_name'].str.lower()
        
        # Merge and deduplicate on player name
        us_dedup = understat_players[['player_name_lower', 'xG', 'xA']].drop_duplicates('player_name_lower')
        elements = elements.merge(us_dedup, 
                                  left_on='web_name_lower', right_on='player_name_lower', 
                                  how='left')
        elements['xG'] = elements['xG'].fillna(0)
        elements['xA'] = elements['xA'].fillna(0)
        
        # Clean up temp columns
        elements.drop(columns=['web_name_lower', 'player_name_lower'], inplace=True, errors='ignore')
    else:
        elements['xG'] = 0.0
        elements['xA'] = 0.0

    # 4. Enhanced Prediction Logic
    # Base attacking points (using Understat if available, else FPL stats)
    if elements['xG'].sum() > 0: 
        # Normalize season xG/xA to per-game (approximate)
        games_played_est = (elements['minutes'] / 90.0).replace(0, 1)
        xg_per_game = elements['xG'] / games_played_est
        xa_per_game = elements['xA'] / games_played_est
        
        # Apply position multipliers for goals (DEF=6, MID=5, FWD=4)
        goal_pts_mult = np.select(
            [elements["element_type"] == 2, elements["element_type"] == 3, elements["element_type"] == 4],
            [6.0, 5.0, 4.0], default=0.0
        )
        assist_pts_mult = 3.0
        att_base = (xg_per_game * goal_pts_mult) + (xa_per_game * assist_pts_mult)
    else:
        # Fallback to old xgi_proxy if no Understat
        att_base = 0.6 * elements["points_per_game"] + 0.4 * (elements["ict_index"] / 10.0)

    # Base defensive points (Clean Sheets proxy from form)
    def_base = elements['form'] * 0.1 

    # Combine into a new robust base Predict Score (before fixture adjustment)
    # Weighting: Attack 60%, Form 40% + Defensive base
    elements['base_xP'] = (att_base * 0.6) + (elements['form'] * 0.4) + def_base
    
    # Positional Multiplier for Heuristic (slightly boost FWDs/MIDs who play)
    pos_mult = np.select(
        [elements["element_type"] == 1, elements["element_type"] == 2, elements["element_type"] == 3, elements["element_type"] == 4],
        [0.9, 0.95, 1.0, 1.05], default=1.0
    )

    # Final Heuristic with Fixture & Play Probability
    elements["pred_points_enhanced"] = (
        elements['base_xP'] * (0.5 + 0.5 * elements["avg_fixture_ease"]) * # Fixture adjustment
        (0.4 + 0.6 * elements["play_prob"]) * # Playing time adjustment
        pos_mult * elements['num_fixtures']                      # DGW/BGW adjustment
    )
    
    # Clip reasonable values
    elements["pred_points_enhanced"] = elements["pred_points_enhanced"].clip(lower=0, upper=25)
    elements.loc[elements['num_fixtures'] == 0, 'pred_points_enhanced'] = 0

    # ===== เพิ่มบรรทัดนี้ (ตาม .docx) =====
    elements['selection_score'] = elements.apply(calculate_smart_selection_score, axis=1)
    # ===== จบ =====

    return elements

def calculate_smart_selection_score(player_row):
    """
    คำนวณคะแนนสำหรับการจัดตัว (Selection Score) 
    โดยพิจารณาหลายปัจจัย (ตามคำแนะนำใน .docx)
    """
    score = 0.0
    
    # 1. Base Points (40%)
    score += player_row.get('pred_points', 0) * 0.4
    
    # 2. Expected Goals/Assists from Understat (30%)
    # (เพิ่ม .get() เพื่อความปลอดภัย)
    if player_row.get('xG', 0) > 0 or player_row.get('xA', 0) > 0:
        # ใช้นาทีจาก 'minutes' ถ้ามี, ถ้าไม่มีให้หาร 1 (ปลอดภัยกว่า)
        minutes_played = player_row.get('minutes', 0)
        games_played_est = max(1, minutes_played / 90.0)
        xgi_bonus = (player_row.get('xG', 0) * 5 + player_row.get('xA', 0) * 3) / games_played_est
        score += xgi_bonus * 0.3
    
    # 3. Form (15%)
    score += player_row.get('form', 0) * 0.15
    
    # 4. Fixture Difficulty (10%)
    # (avg_fixture_ease 1.0 คือง่ายสุด, 0.0 คือยากสุด)
    score += player_row.get('avg_fixture_ease', 0) * 10 * 0.1
    
    # 5. Play Probability (5%) - ลดคะแนนถ้าโอกาสลงเล่นต่ำ
    score *= (0.5 + 0.5 * player_row.get('play_prob', 1.0))
    
    # 6. DGW Bonus (ถ้ามี Double Gameweek)
    if player_row.get('num_fixtures', 1) == 2:
        score *= 1.3
    
    # 7. BGW Penalty (ถ้าไม่มีเกม)
    if player_row.get('num_fixtures', 1) == 0:
        score = 0
    
    return score

def smart_bench_order(bench_df: pd.DataFrame) -> pd.DataFrame:
    """
    จัดเรียงตัวสำรองอย่างชาญฉลาด:
    1. GK ต้องอยู่แรกเสมอ
    2. Outfield: เรียงตาม "Auto-sub Value"
    """
    bench_gk = bench_df[bench_df['element_type'] == 1]
    bench_outfield = bench_df[bench_df['element_type'] != 1].copy()
    
    # คำนวณ Auto-sub Value (โอกาสที่จะถูกเปลี่ยนตัวอัตโนมัติ)
    bench_outfield['autosub_value'] = (
        bench_outfield['play_prob'] * 0.4 +  # โอกาสลงเล่น
        (bench_outfield.get('selection_score', bench_outfield['pred_points']) / 10) * 0.4 + # คะแนนคาด
        (bench_outfield['num_fixtures'] > 0).astype(int) * 0.2  # มีนัดแข่งหรือไม่ (ใช้ .astype(int) เพื่อแปลง True/False เป็น 1/0)
    )
    
    # เรียงจากมากไปน้อย (คนที่ควรถูก auto-sub มากที่สุดอยู่ลำดับแรก)
    bench_outfield = bench_outfield.sort_values('autosub_value', ascending=False)
    
    ordered_bench_df = pd.concat([bench_gk, bench_outfield])
    return ordered_bench_df

def select_captain_vice(xi_df: pd.DataFrame) -> Tuple[int, int]:
    """
    เลือกกัปตัน/รองกัปตันอย่างชาญฉลาด
    """
    xi_candidates = xi_df.copy()
    
    # คำนวณ Captain Score (ใช้ .get() เพื่อความปลอดภัย)
    xi_candidates['captain_score'] = (
        xi_candidates.get('selection_score', xi_candidates['pred_points']) * 0.5 +
        xi_candidates.get('form', 0) * 0.2 +
        xi_candidates.get('avg_fixture_ease', 0) * 10 * 0.2 + # avg_fixture_ease (1.0 = ง่าย)
        (xi_candidates.get('xG', 0) + xi_candidates.get('xA', 0)) * 0.1
    )
    
    # ถ้ามี DGW ให้โบนัส
    xi_candidates.loc[xi_candidates['num_fixtures'] == 2, 'captain_score'] *= 1.5
    
    # เรียงและเลือก
    sorted_candidates = xi_candidates.sort_values('captain_score', ascending=False)
    cap_id = sorted_candidates.iloc[0].name
    vc_id = sorted_candidates.iloc[1].name
    
    return cap_id, vc_id

def analyze_lineup_insights(xi_df: pd.DataFrame, bench_df: pd.DataFrame) -> List[str]:
    """
    วิเคราะห์และให้คำแนะนำเกี่ยวกับการจัดตัว
    """
    insights = []
    
    # 1. เช็ค DGW/BGW
    dgw_count = (xi_df['num_fixtures'] == 2).sum()
    bgw_count = (xi_df['num_fixtures'] == 0).sum()
    
    if dgw_count > 0:
        insights.append(f"✅ คุณมี {dgw_count} คนใน XI ที่มี Double Gameweek")
    if bgw_count > 0:
        insights.append(f"⚠️ คุณมี {bgw_count} คนใน XI ที่ไม่มีนัดแข่ง!")
    
    # 2. เช็คโอกาสลงเล่น
    low_prob_players = xi_df[xi_df['play_prob'] < 0.75]
    if len(low_prob_players) > 0:
        names = ", ".join(low_prob_players['web_name'].tolist())
        insights.append(f"⚠️ นักเตะที่อาจไม่ลงเล่น: {names} (โอกาส < 75%)")
    
    # 3. เช็ค Fixture Difficulty (avg_fixture_ease 0.3 คือยากมาก)
    hard_fixtures = xi_df[xi_df['avg_fixture_ease'] < 0.3]
    if len(hard_fixtures) > 2:
        insights.append(f"⚠️ คุณมี {len(hard_fixtures)} คนที่เจอคู่แข่งยาก (ความง่าย < 0.3)")
    
    # 4. เช็ค xG/xA Leaders
    if 'xG' in xi_df.columns and not xi_df.empty:
        try:
            top_xg_player = xi_df.nlargest(1, 'xG').iloc[0]
            insights.append(f"🎯 นักเตะที่มี xG สูงสุดใน XI: {top_xg_player['web_name']} ({top_xg_player['xG']:.2f})")
        except IndexError:
            pass # เกิดขึ้นได้ถ้า xi_df ว่าง
    
    # 5. เช็ค Bench Strength
    bench_score_col = 'selection_score' if 'selection_score' in bench_df.columns else 'pred_points'
    bench_total = bench_df.get(bench_score_col, 0).sum()
    if bench_total < 7.5:
        insights.append(f"⚠️ ตัวสำรองของคุณคะแนนคาดการณ์ต่ำ (คะแนนรวม: {bench_total:.1f})")
    
    return insights

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
                opp_rank = team_strength.loc[opp_id, 'position'] # <--- ดึงอันดับคู่แข่ง
                opponents.append(f"{team_names.get(opp_id, '?')} (H) (อันดับ {opp_rank})") # <--- (สำคัญ) เพิ่ม (อันดับ)
                # Difficulty: Opponent's league position
                diff = opp_rank # <--- ใช้ diff เป็นอันดับเหมือนเดิม
                difficulties.append(diff)

            for _, game in away_games.iterrows():
                opp_id = game['team_h']
                opp_rank = team_strength.loc[opp_id, 'position'] # <--- ดึงอันดับคู่แข่ง
                opponents.append(f"{team_names.get(opp_id, '?')} (A) (อันดับ {opp_rank})") # <--- (สำคัญ) เพิ่ม (อันดับ)
                # Difficulty: Opponent's league position
                diff = opp_rank # <--- ใช้ diff เป็นอันดับเหมือนเดิม
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
    diff_df = diff_df.sort_values('Total', ascending=False)
    opp_df = opp_df.loc[diff_df.index] # Match order

    return opp_df, diff_df

@st.cache_data(ttl=300)
def find_rotation_pairs(difficulty_matrix: pd.DataFrame, teams_df: pd.DataFrame, all_players: pd.DataFrame, budget: float = 9.0):
    """
    Finds the best GK rotation pairs within a budget for the next 5 GWs.
    REVISED: Returns a 1-5 star rating instead of the raw score.
    """
    # กรองเฉพาะ GK ที่มีโอกาสลงเล่นสูง (>75%) หรือมีคะแนนคาดการณ์ > 0.5 เพื่อป้องกันตัวสำรองที่ไม่เคยลง
    gks = all_players[
        (all_players['element_type'] == 1) &
        ((all_players['chance_of_playing_next_round'] > 75) | (all_players['pred_points'] > 0.5))
    ].copy()
    
    gks['price'] = gks['now_cost'] / 10.0
    gks['team_short'] = gks['team'].map(teams_df.set_index('id')['short_name'])

    cheap_gks = gks[gks['price'] <= (budget - 4.0)]
    
    pairs = []
    checked_pairs = set()

    for i, gk1 in cheap_gks.iterrows():
        for j, gk2 in cheap_gks.iterrows():
            if i >= j or (gk2['team'], gk1['team']) in checked_pairs:
                continue
            
            if (gk1['price'] + gk2['price']) > budget:
                continue
                
            checked_pairs.add((gk1['team'], gk2['team']))
            
            try:
                diff1 = difficulty_matrix.loc[gk1['team_short']]
                diff2 = difficulty_matrix.loc[gk2['team_short']]
            except KeyError:
                continue

            rotation_score = 0
            for col in difficulty_matrix.columns:
                if col == 'Total': continue
                rotation_score += min(diff1[col], diff2[col])
            
            pairs.append({
                'GK1': f"{gk1['web_name']} ({gk1['price']:.1f}m)",
                'GK2': f"{gk2['web_name']} ({gk2['price']:.1f}m)",
                'Total Cost': gk1['price'] + gk2['price'], # <-- บั๊กที่แก้ไปแล้ว
                'Rotation Score': rotation_score
            })

    if not pairs:
        return pd.DataFrame(columns=['GK1', 'GK2', 'Total Cost','Rating'])
        
    pairs_df = pd.DataFrame(pairs)
    
    # --- NEW RATING LOGIC (เริ่ม) ---
    if not pairs_df.empty:
        # 1. Sort by score first (best = lowest score)
        pairs_df = pairs_df.sort_values('Rotation Score', ascending=True)
        
        # 2. Get min/max score from the *entire list* found
        min_score = pairs_df['Rotation Score'].min()
        max_score = pairs_df['Rotation Score'].max()
        
        # 3. Define star rating function
        def get_star_rating(score, min_s, max_s):
            if max_s == min_s:
                return "⭐⭐⭐⭐⭐" # All are equally good
            
            # Normalize (0=best, 1=worst)
            norm_score = (score - min_s) / (max_s - min_s)
            
            if norm_score < 0.2:
                return "⭐⭐⭐⭐⭐"
            elif norm_score < 0.4:
                return "⭐⭐⭐⭐"
            elif norm_score < 0.6:
                return "⭐⭐⭐"
            elif norm_score < 0.8:
                return "⭐⭐"
            else:
                return "⭐"

        # 4. Apply star rating
        pairs_df['Rating'] = pairs_df['Rotation Score'].apply(lambda x: get_star_rating(x, min_score, max_score))
    # --- NEW RATING LOGIC (จบ) ---

    # 5. Format Total Cost
    pairs_df['Total Cost'] = pairs_df['Total Cost'].apply(lambda x: f"£{x:.1f}m")
    
    # 6. Select final columns and take top 10
    final_cols = ['GK1', 'GK2', 'Total Cost','Rating']
    pairs_df = pairs_df[final_cols].head(10)
    
    return pairs_df.reset_index(drop=True)
# --- END: NEW FIXTURE PLANNER FUNCTIONS ---


###############################
# Squad & optimization
###############################

# --- NEW (v2.0): 3-GW Projector for ROI ---
@st.cache_data(ttl=300)
def predict_next_n_gws(player_id: int, n_gws: int, current_gw: int, 
                       elements_df: pd.DataFrame, fixtures_df: pd.DataFrame, teams_df: pd.DataFrame) -> float:
    """
    Predicts total points for a single player over the next N gameweeks.
    Uses base_xP combined with specific future fixture difficulties.
    """
    if player_id not in elements_df.index:
        return 0.0
        
    player = elements_df.loc[player_id]
    team_id = player['team']
    # Use base_xP if available, else derive roughly from predicted points
    base_xp = player.get('base_xP', player.get('pred_points', 0) / max(0.5, player.get('avg_fixture_ease', 1)))
    play_prob = player.get('play_prob', 1.0)
    
    total_expected_points = 0.0
    
    # Pre-calculate team strengths for speed
    team_def_strength = teams_df.set_index('id')['strength_defence_overall'].to_dict()
    avg_def_strength = np.mean(list(team_def_strength.values()))

    for gw in range(current_gw, current_gw + n_gws):
        # Get fixtures for this team in this GW
        gw_fixtures = fixtures_df[(fixtures_df['event'] == gw) & 
                                  ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id))]
        
        if gw_fixtures.empty:
            continue # Blank GW adds 0 points

        for _, fixture in gw_fixtures.iterrows():
            is_home = fixture['team_h'] == team_id
            opponent_id = fixture['team_a'] if is_home else fixture['team_h']
            
            # Calculate Fixture Multiplier (Avg Strength / Opponent Strength)
            opp_strength = team_def_strength.get(opponent_id, avg_def_strength)
            fixture_multiplier = avg_def_strength / max(opp_strength, 1) # Avoid div by zero
            
            # Adjust for Home/Away (approx 10% boost for home)
            home_boost = 1.1 if is_home else 0.95
            
            # Calculate points for this fixture
            match_xp = base_xp * fixture_multiplier * home_boost * play_prob
            total_expected_points += match_xp

    return total_expected_points

# --- NEW (v2.0): Transfer ROI Calculator Function ---
def calculate_transfer_roi(player_out_id: int, player_in_id: int, current_gw: int,
                           elements_df: pd.DataFrame, fixtures_df: pd.DataFrame, teams_df: pd.DataFrame,
                           hit_cost: int = 0, lookahead: int = 3) -> Dict:
    """
    Calculates the expected Net Gain over the next N gameweeks for a specific transfer.
    """
    # Predict next N GWs for both players
    out_xp_3gw = predict_next_n_gws(player_out_id, lookahead, current_gw, elements_df, fixtures_df, teams_df)
    in_xp_3gw = predict_next_n_gws(player_in_id, lookahead, current_gw, elements_df, fixtures_df, teams_df)
    
    # Calculate ROI
    gross_gain = in_xp_3gw - out_xp_3gw
    net_gain = gross_gain - hit_cost
    
    return {
        "out_xp_3gw": out_xp_3gw,
        "in_xp_3gw": in_xp_3gw,
        "gross_gain": gross_gain,
        "net_gain": net_gain,
        "is_worth_it": net_gain > 0.5 # Threshold for "worth it"
    }

def optimize_starting_xi(squad_players_df: pd.DataFrame) -> Tuple[List[int], List[int]]:
    """Return (start_ids, bench_ids) maximizing predicted points subject to FPL formation."""
    ids = list(squad_players_df.index)
    positions = squad_players_df['element_type']

    prob = LpProblem("XI_Optimization", LpMaximize)
    x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in ids}

    # ===== เปลี่ยนตรงนี้ (ตาม .docx) =====
    # เช็คว่ามี column 'selection_score' หรือไม่
    if 'selection_score' in squad_players_df.columns:
        objective_scores = squad_players_df['selection_score']
    else:
        # Fallback ใช้วิธีเดิม (pred_points * play_prob)
        pred_points = squad_players_df['pred_points']
        play_probs = squad_players_df['play_prob']
        objective_scores = pred_points * play_probs
    # ===== จบการแก้ไข =====
    
    # ใช้ 'objective_scores' ใหม่นี้
    prob += lpSum([objective_scores.get(i, 0) * x[i] for i in ids])

    # Constraints (เหมือนเดิม)
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

# (Old calculate_3gw_roi preserved for backward compatibility if needed, though new one is better)
def calculate_3gw_roi(player, fixtures_df, teams_df, current_event):
    """Calculate expected points over next 3 GWs for ROI analysis (Legacy version)."""
    try:
        team_id = int(player['team'])
        next_3gw = list(range(current_event, current_event + 3))
        total_points = 0
        for gw in next_3gw:
            team_fixtures = fixtures_df[((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)) & (fixtures_df['event'] == gw)]
            if team_fixtures.empty: continue
            for _, fixture in team_fixtures.iterrows():
                is_home = fixture['team_h'] == team_id
                opp_team = fixture['team_a'] if is_home else fixture['team_h']
                opp_str = teams_df.loc[teams_df['id'] == opp_team, 'strength_overall_away' if is_home else 'strength_overall_home'].iloc[0]
                base_points = float(player.get('pred_points', 0)) / 2.0 # Rough est per game
                max_str = teams_df['strength_overall_home'].max()
                fixture_diff = 1.0 - (opp_str / max_str)
                points = base_points * (1.1 if is_home else 0.9) * (1.0 + fixture_diff)
                total_points += points
        return total_points
    except Exception:
        return float(player.get('pred_points', 0))

def suggest_transfers(current_squad_ids: List[int], bank: float, free_transfers: int,
                      all_players: pd.DataFrame, strategy: str,
                      fixtures_df: pd.DataFrame, teams_df: pd.DataFrame, 
                      current_event: int) -> List[Dict]:
    """Greedy search for transfers based on the selected strategy."""
    valid_squad_ids = [pid for pid in current_squad_ids if pid in all_players.index]
    if not valid_squad_ids: return []

    current_squad_df = all_players.loc[valid_squad_ids]
    start_ids, _ = optimize_starting_xi(current_squad_df)
    if not start_ids: return []

    if strategy == "Free Transfer": max_transfers, hit_cost = free_transfers, float('inf')
    elif strategy == "Allow Hit (AI Suggest)": max_transfers, hit_cost = 5, 4
    else: max_transfers, hit_cost = 15, 0

    current_team_count = {}
    for pid in valid_squad_ids:
        tid = int(all_players.loc[pid, 'team'])
        current_team_count[tid] = current_team_count.get(tid, 0) + 1

    position_groups = {1: [], 2: [], 3: [], 4: []}
    for pid in valid_squad_ids:
        position_groups.setdefault(int(all_players.loc[pid, 'element_type']), []).append(pid)

    remaining_bank, used_in_players, potential_moves = bank, set(), []

    for pos in [1, 2, 3, 4]:
        out_ids = position_groups.get(pos, [])
        if not out_ids: continue
        for out_id in sorted(out_ids, key=lambda x: all_players.loc[x, 'pred_points']):
            out_player = all_players.loc[out_id]
            out_team_id = int(out_player['team'])
            budget = out_player['selling_price'] + (remaining_bank * 10)
            all_replacements = all_players[(all_players['element_type'] == out_player['element_type']) & (~all_players.index.isin(valid_squad_ids)) & (all_players['now_cost'] <= budget) & (all_players['pred_points'] > out_player['pred_points'])].sort_values('pred_points', ascending=False)
            if all_replacements.empty: continue

            best_replacement, best_replacement_id = None, None
            for cid, candidate in all_replacements.iterrows():
                candidate_team_id = int(candidate['team'])
                ftc = current_team_count.copy()
                ftc[out_team_id] = ftc.get(out_team_id, 0) - 1
                if ftc[out_team_id] <= 0: ftc.pop(out_team_id, None)
                if ftc.get(candidate_team_id, 0) + 1 > 3: continue
                if int(cid) in used_in_players: continue
                best_replacement, best_replacement_id = candidate, int(cid)
                break

            if best_replacement is None: continue

            cost_change = (best_replacement['now_cost'] - out_player['selling_price']) / 10.0
            if cost_change > remaining_bank: continue

            if out_team_id != int(best_replacement['team']):
                current_team_count[out_team_id] = current_team_count.get(out_team_id, 0) - 1
                if current_team_count[out_team_id] <= 0: current_team_count.pop(out_team_id, None)
                current_team_count[int(best_replacement['team'])] = current_team_count.get(int(best_replacement['team']), 0) + 1

            remaining_bank = round(max(0.0, remaining_bank - cost_change), 2)
            used_in_players.add(best_replacement_id)

            # Using new ROI function here if possible, or keep legacy for speed in loop
            roi_in = calculate_3gw_roi(best_replacement, fixtures_df, teams_df, current_event)
            roi_out = calculate_3gw_roi(out_player, fixtures_df, teams_df, current_event)

            potential_moves.append({
                "out_id": int(out_id), "in_id": best_replacement_id,
                "out_name": out_player.get("web_name", ""), "in_name": best_replacement.get("web_name", ""),
                "out_pos": POSITIONS.get(int(out_player["element_type"]), str(out_player["element_type"])),
                "in_pos": POSITIONS.get(int(best_replacement["element_type"]), str(best_replacement["element_type"])),
                "out_team": out_player.get("team_short", ""), "in_team": best_replacement.get("team_short", ""),
                "in_points": float(best_replacement.get("pred_points", 0.0)),
                "delta_points": float(best_replacement.get('pred_points', 0.0) - out_player.get('pred_points', 0.0)),
                "roi_3gw": float(roi_in - roi_out),
                "in_cost": float(best_replacement.get('now_cost', 0.0)) / 10.0, "out_cost": float(out_player.get('selling_price', 0.0)) / 10.0,
            })

    potential_moves.sort(key=lambda x: x.get("delta_points", 0.0), reverse=True)
    final_suggestions = []
    GREEDY_THRESHOLD, CONSERVATIVE_THRESHOLD = -2.0, -0.1
    for i, move in enumerate(potential_moves):
        if len(final_suggestions) >= max_transfers: break
        hit = 0 if len(final_suggestions) < free_transfers else hit_cost
        net_gain = move["delta_points"] - hit
        m = move.copy(); m['net_gain'] = round(net_gain, 2); m['hit_cost'] = hit
        if strategy == "Free Transfer" and net_gain >= CONSERVATIVE_THRESHOLD: final_suggestions.append(m)
        elif strategy == "Allow Hit (AI Suggest)" and net_gain >= GREEDY_THRESHOLD: final_suggestions.append(m)
        elif strategy == "Wildcard / Free Hit" and net_gain > 0.0: final_suggestions.append(m)

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
                              all_players: pd.DataFrame, strategy: str,
                              fixtures_df: pd.DataFrame, teams_df: pd.DataFrame, 
                              current_event: int) -> Tuple[List[Dict], List[Dict]]:
    """แนะนำ transfers แบบ 2 มุมมอง: ปกติ vs ระมัดระวัง"""
    # การแนะนำแบบปกติ
    normal_moves = suggest_transfers(current_squad_ids, bank, free_transfers, 
                                   all_players, strategy,
                                   fixtures_df, teams_df, current_event)
    
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
        strategy,
        fixtures_df,      # Add these new arguments
        teams_df,         # Add these new arguments
        current_event     # Add these new arguments
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
# --- NEW: Visual Fixture Planner (v2.0) ---
###############################

def get_difficulty_css_class(val, min_val, max_val): # min_val/max_val ไม่ได้ใช้แล้ว
    """Returns the CSS class based on the opponent's league position."""
    if val == 0: # BGW
        return "bg-blank" # Dark grey
    
    # Logic ใหม่: อิงตามอันดับตารางคะแนน (1-20)
    if val >= 15: # อันดับ 15-20 (โซนท้ายตาราง)
        # Easy (Green) - เขียวสด
        return "bg-easy"
    elif val >= 8: # อันดับ 8-14 (กลางตาราง)
        # Medium (Orange) - สีส้ม
        return "bg-medium"
    else: # อันดับ 1-7 (ท็อป 7)
        # Hard (Red) - แดง
        return "bg-hard"

def display_visual_fixture_planner(opp_matrix: pd.DataFrame, diff_matrix: pd.DataFrame, teams_df: pd.DataFrame):
    """
    Displays the Fixture Planner as a visual HTML table with logos and colors.
    (Enhanced to show team ranks and opponent ranks)
    """
    
    # --- ENHANCE: สร้าง lookup สำหรับ "อันดับ" ของทีม ---
    team_logo_lookup = teams_df.set_index('short_name')['logo_url'].to_dict()
    team_rank_lookup = teams_df.set_index('short_name')['position'].to_dict()
    
    # Get GW columns
    gw_cols = [col for col in diff_matrix.columns if col.startswith('GW')]
    
    # (min/max ไม่ได้ใช้แล้ว แต่เก็บไว้เผื่อ)
    min_val = 1
    max_val = 20

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
            width: 85px; /* <--- ENHANCE: ขยายความกว้างคอลัมน์ทีม */
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
        /* --- ENHANCE: CSS สำหรับอันดับทีม --- */
        .team-rank {
            font-size: 11px;
            font-weight: normal;
            color: #555;
            margin-top: 0;
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
        /* --- ENHANCE: CSS สำหรับอันดับคู่แข่ง --- */
        .opponent-rank {
            display: block;
            font-size: 11px;
            font-weight: normal;
        }
        /* Color classes (ยังเหมือนเดิม) */
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
        
        # --- ENHANCE: ดึงอันดับของทีมเรา ---
        team_rank = team_rank_lookup.get(team_short_name, '?')
        
        html += "<tr>"
        # --- ENHANCE: แสดงชื่อทีม + อันดับ ---
        html += f"<td class='team-cell'>"
        html += f"<img src='{team_logo_url}' alt='{team_short_name}'><br>"
        html += f"<span>{team_short_name}</span>"
        html += f"<span class='team-rank'>(อันดับ {team_rank})</span>"
        html += "</td>"
        
        # Columns 2-6: Fixtures
        for gw in gw_cols:
            diff_score = diff_row[gw]
            opp_string = opp_matrix.loc[team_short_name, gw] # นี่คือ "CHE (H) (อันดับ 7)"
            css_class = get_difficulty_css_class(diff_score, min_val, max_val)
            
            cell_content = ""
            if opp_string == "BLANK":
                cell_content = "BLANK"
            elif "," in opp_string:
                # Double Gameweek: "CHE (H) (อันดับ 7), ARS (A) (อันดับ 2)"
                cell_content = opp_string.replace(", ", "<br>") # แสดงผล "CHE (H) (อันดับ 7)" [ขึ้นบรรทัดใหม่] "ARS (A) (อันดับ 2)"
                css_class = "dgw-cell " + css_class # Add DGW style
            else:
                # Single Gameweek: "CHE (H) (อันดับ 7)"
                
                # --- ENHANCE V2: แก้ไข Bug การแยกชื่อทีม (เช่น "Man City") ---
                try:
                    # "Man City (H) (อันดับ 1)"
                    
                    # 1. ค้นหาวงเล็บสุดท้าย (ของอันดับ)
                    last_paren_index = opp_string.rfind('(')
                    if last_paren_index == -1: raise Exception("No last parenthesis")
                    
                    opp_rank_str = opp_string[last_paren_index:].strip() # -> "(อันดับ 1)"
                    main_part = opp_string[:last_paren_index].strip()    # -> "Man City (H)"
                    
                    # 2. ค้นหาวงเล็บที่สอง (ของ H/A)
                    second_last_paren_index = main_part.rfind('(')
                    if second_last_paren_index == -1: raise Exception("No second parenthesis")
                    
                    home_away = main_part[second_last_paren_index:].strip()    # -> "(H)"
                    opp_short_name = main_part[:second_last_paren_index].strip() # -> "Man City"

                    # 3. ดึงโลโก้
                    opp_logo_url = team_logo_lookup.get(opp_short_name, '')
                    
                    # 4. สร้าง HTML
                    cell_content = f"<img src='{opp_logo_url}' alt='{opp_short_name}'>"
                    cell_content += f"<br>{home_away}"
                    cell_content += f"<span class='opponent-rank'>{opp_rank_str}</span>" # แสดงอันดับบรรทัดล่าง
                
                except Exception as e:
                    # st.error(f"Error parsing fixture string: {opp_string} | {e}") # สามารถเปิดเพื่อ debug
                    cell_content = opp_string # Fallback

            html += f"<td class='fixture-cell {css_class}'>{cell_content}</td>"
        
        html += "</tr>"

    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)

###############################
# --- NEW: Home Dashboard Function (v1.9.0) ---
###############################

# --- NEW (v1.9.7): Helper function for Understat section ---
def display_understat_section(merged_players: pd.DataFrame, merged_teams: pd.DataFrame):
    """Displays the Understat xG, xA, and xPTS section."""
    
    st.subheader("📊 สถิติจาก Understat (xG, xA, xPTS)")
    
    # --- NEW (v1.9.6): Add placeholder URL and helper function ---
    DEFAULT_PHOTO_URL = "https://resources.premierleague.com/premierleague/photos/players/110x140/p-blank.png"

    def get_player_image_html(photo_url, player_name, width=60):
        """Generates an HTML img tag with a fallback placeholder."""
        # Use HTML entities for quotes inside the onerror attribute
        alt_text = str(player_name).replace("'", "").replace('"', '')
        src_url = photo_url if pd.notna(photo_url) else DEFAULT_PHOTO_URL
        return f'<img src="{src_url}" alt="{alt_text}" width="{width}" style="border-radius: 4px; min-height: {int(width*1.33)}px; background-color: #eee;" onerror="this.onerror=null;this.src=\'{DEFAULT_PHOTO_URL}\';">'

    col1, col2, col3 = st.columns(3)

    # --- Column 1: Top 5 xG ---
    with col1:
        st.markdown("#### 🎯 Top 5 xG (โอกาสยิง)")
        if merged_players.empty or 'xG' not in merged_players.columns:
            st.caption("ไม่มีข้อมูล xG")
        else:
            top_xg = merged_players.nlargest(5, 'xG')
            for _, row in top_xg.iterrows():
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.markdown(get_player_image_html(row['photo_url'], row['player_name'], 60), unsafe_allow_html=True)
                with c2:
                    st.markdown(f"**{row['player_name']}** ({row['team_short']})")
                    st.markdown(f"**xG: {row['xG']:.2f} | ยิง: {row['goals_scored']:.0f}**")

    # --- Column 2: Top 5 xA ---
    with col2:
        st.markdown("#### 🅰️ Top 5 xA (โอกาสจ่าย)")
        if merged_players.empty or 'xA' not in merged_players.columns:
            st.caption("ไม่มีข้อมูล xA")
        else:
            top_xa = merged_players.nlargest(5, 'xA')
            for _, row in top_xa.iterrows():
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.markdown(get_player_image_html(row['photo_url'], row['player_name'], 60), unsafe_allow_html=True)
                with c2:
                    st.markdown(f"**{row['player_name']}** ({row['team_short']})")
                    st.markdown(f"**xA: {row['xA']:.2f} | จ่าย: {row['assists']:.0f}**")

    # --- Column 3: Top 5 xPTS ---
    with col3:
        st.markdown("#### 📈 Top 5 xPTS (คะแนนคาดหวัง)")
        if merged_teams.empty or 'xpts' not in merged_teams.columns:
            st.caption("ไม่มีข้อมูล xPTS")
        else:
            top_xpts = merged_teams.nlargest(5, 'xpts')
            for _, row in top_xpts.iterrows():
                c1, c2 = st.columns([1, 4])
                
                # --- START EDIT ---
                with c1:
                    # 1. ดึง logo_url (ถ้ามี)
                    logo_url = row['logo_url'] if pd.notna(row['logo_url']) else ""
                    # 2. หาชื่อที่จะใช้เป็น alt text (ชื่อ FPL ถ้ามี, ไม่มีก็ใช้ชื่อ Understat)
                    alt_text = row['name'] if pd.notna(row['name']) else row['title']
                    # 3. ใช้ st.markdown เพื่อแสดงผล (แก้ปัญหา st.image error)
                    st.markdown(f'<img src="{logo_url}" alt="{alt_text}" width="40" style="min-height: 40px; background-color: #eee; border-radius: 4px;">', unsafe_allow_html=True)
                
                with c2:
                    # 4. แสดงชื่อทีม FPL (row['name']) ถ้ามี, ถ้าไม่มี (merge ไม่เจอ)
                    #    ให้แสดงชื่อทีม Understat (row['title']) แทน
                    display_name = row['name'] if pd.notna(row['name']) else row['title']
                    st.markdown(f"**{display_name}**")
                    st.markdown(f"**xPTS: {row['xpts']:.2f}**")
                # --- END EDIT ---
                    
    st.markdown("---")

def display_home_dashboard(
    feat_df: pd.DataFrame, 
    nf_df: pd.DataFrame, 
    teams_df: pd.DataFrame, 
    opp_matrix: pd.DataFrame, 
    diff_matrix: pd.DataFrame, 
    rotation_pairs: pd.DataFrame,
    merged_understat_players: pd.DataFrame, # <-- NEW
    merged_understat_teams: pd.DataFrame  # <-- NEW
):
    """
    Displays the full home page dashboard (DGW/BGW, Captains, Top 20, Value, Fixtures, Trends).
    """
    
    # --- NEW (v1.9.6): Add placeholder URL and helper function ---
    DEFAULT_PHOTO_URL = "https://resources.premierleague.com/premierleague/photos/players/110x140/p-blank.png"

    def get_player_image_html(photo_url, player_name, width=60):
        """Generates an HTML img tag with a fallback placeholder."""
        # Use HTML entities for quotes inside the onerror attribute
        alt_text = str(player_name).replace("'", "").replace('"', '')
        src_url = photo_url if pd.notna(photo_url) else DEFAULT_PHOTO_URL
        return f'<img src="{src_url}" alt="{alt_text}" width="{width}" style="border-radius: 4px; min-height: {int(width*1.33)}px; background-color: #eee;" onerror="this.onerror=null;this.src=\'{DEFAULT_PHOTO_URL}\';">'

    
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

    # --- Captaincy Corner & Price Movement (v1.9.5 - 3-col layout) ---
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
                    st.caption(f"ราคาปัจจุบัน: £{row['now_cost']/10.0:.1f}m")

    with col2:
        st.subheader("💹 ราคากำลังขึ้น 🔼")
        
        # --- Price Risers ---
        risers = feat_df[feat_df['cost_change_start'] > 0].sort_values('cost_change_start', ascending=False).head(5)
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
                    # 1. แสดงราคาที่ขึ้นในสัปดาห์นี้ (ถ้ามี)
                    weekly_change = row['cost_change_event']
                    if weekly_change > 0:
                        st.caption(f"▲ ขึ้นสัปดาห์นี้: +£{weekly_change/10.0:.1f}m")
                    # ===== START USER EDIT =====
                    st.caption(f"▲ ขึ้นรวม: +£{row['cost_change_start']/10.0:.1f}m")
                    st.caption(f"ราคาปัจจุบัน: £{row['now_cost']/10.0:.1f}m")
                    # ===== END USER EDIT =====

    with col3:
        st.subheader("🔻 ราคากำลังลง 📉")
        
        # --- Price Fallers ---
        fallers = feat_df[feat_df['cost_change_start'] < 0].sort_values('cost_change_start', ascending=True).head(5)
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
                    # 1. แสดงราคาที่ลงในสัปดาห์นี้ (ถ้ามี)
                    weekly_change = row['cost_change_event']
                    if weekly_change < 0:
                        st.caption(f"▼ ลงสัปดาห์นี้: -£{abs(weekly_change/10.0):.1f}m")
                    # ===== START USER EDIT =====
                    st.caption(f"▼ ลงรวม: -£{abs(row['cost_change_start']/10.0):.1f}m")
                    st.caption(f"ราคาปัจจุบัน: £{row['now_cost']/10.0:.1f}m")
                    # ===== END USER EDIT =====


    st.markdown("---")

        # --- Top 20 Players ---
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
    
    # --- NEW (v1.9.7): Call Understat Section ---
    display_understat_section(merged_understat_players, merged_understat_teams)

        # --- Player Trends (Now 3 columns) ---
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
        st.markdown("#### 👥 Top 5 ขวัญใจมหาชน")
        most_owned = feat_df.nlargest(5, 'selected_by_percent')
        for _, row in most_owned.iterrows():
            c1, c2 = st.columns([1, 3])
            with c1: 
                # --- MODIFIED (v1.9.6): Use HTML fallback ---
                st.markdown(get_player_image_html(row['photo_url'], row['web_name'], 50), unsafe_allow_html=True)
            with c2: st.markdown(f"**{row['web_name']}**"); st.caption(f"คนมี: {row['selected_by_percent']:.1f}%")
    st.markdown("---")

    # --- Fixture Difficulty ---
    st.subheader("🗓️ ตารางแข่ง 5 นัดล่วงหน้า (Fixture Planner)")
    st.markdown("เรียงตามความง่าย ➡ ยาก **อันดับตารางคะแนน** ของคู่แข่ง (สีเขียว = ง่าย, สีเหลือง = ปานกลาง, สีแดง = ยาก)")
    display_visual_fixture_planner(opp_matrix, diff_matrix, teams_df)
    st.markdown("---")

    
    # --- Value Scatter Plot ---
    st.subheader("💰 กราฟนักเตะคุ้มค่า (Value Finder)")
    st.markdown("🪄 เอาเมาส์ไปชี้เพื่อดูชื่อนักเตะได้เลย!แต่ละสีบอกตำแหน่ง ส่วนจุดใกล้มุมซ้ายบนที่สุดคือของดีราคาถูกในตำแหน่งนั้นๆ 💰")
    value_df = feat_df[feat_df['pred_points'] > 1.2].copy() # Filter out duds
    value_df['price'] = value_df['now_cost'] / 10.0
    value_df['position'] = value_df['element_type'].map(POSITIONS)
    
        # --- START EDIT: ปรับแต่งสีและลำดับกราฟ ---
    
    # 1. กำหนดลำดับของตำแหน่ง (Domain)
    position_domain = ['GK', 'DEF', 'MID', 'FWD']
    
    # 2. กำหนดสีที่ต้องการ (Range)
    # (GK, DEF, MID, FWD)
    color_range = ['#EE7733', '#0077BB', '#CC3311', '#33BBEE'] 

    chart = alt.Chart(value_df).mark_circle(size=80, opacity=0.85, stroke='#CCCCCC',strokeWidth=0.8).encode(
        x=alt.X('price', title='ราคา (£)'),
        y=alt.Y('pred_points', title='คะแนนคาดการณ์'),
        
        # 3. ใช้ alt.Color() เพื่อกำหนดค่า domain (ลำดับ) และ range (สี)
        color=alt.Color('position', 
                        legend=alt.Legend(title="ตำแหน่ง"), # ตั้งชื่อ legend
                        scale=alt.Scale(domain=position_domain, range=color_range)),
                        
        tooltip=['web_name', 'team_short', 'position', 'price', 'pred_points'] # เพิ่ม position ใน tooltip
    ).interactive() # Make it zoomable/pannable
    
    # --- END EDIT ---
    
    st.altair_chart(chart, use_container_width=True)
    st.caption("หมายเหตุ: การแสดงรูปนักเตะใน tooltip ของกราฟนี้ยังไม่รองรับครับ")
    st.markdown("---")
    
    # Display Rotation Pairs
    st.markdown("#### 🥅 Top 10 คู่ผู้รักษาประตู (GK Rotation Pairs)")
    st.caption(f"ค้นหาคู่ GK ที่ตารางแข่งสลับกันดีที่สุด (งบรวมไม่เกิน £9.0m)")
    st.dataframe(rotation_pairs, use_container_width=True, hide_index=True)

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
    if not bootstrap or "elements" not in bootstrap:
        st.error("⚠️ FPL API กำลังปิดปรับปรุงชั่วคราว (Game is updating). กรุณารอสักครู่แล้วลองใหม่ครับ")
        st.stop()
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
        deadline_text = f" | ⏳ Deadline: <b>{local_time.strftime('%a, %d %b %H:%M %Z')}</b>"
    
    st.markdown(f"""
    <div style="
        background-color:#e8f4fd;
        padding:1rem;
        border-radius:0.5rem;
        border-left:5px solid #2b8ad7;
        font-size:28px;
        font-weight:600;
        color:#0a2540;
    ">
        📅 สัปดาห์ที่: <b>{cur_event}</b> | 
        วิเคราะห์เกมสัปดาห์ที่: <b>{target_event}</b>
        {deadline_text}
    </div>
""", unsafe_allow_html=True)

    # --- โค้ดที่เพิ่มใหม่เพื่อแสดงข้อมูล DGW/BGW (ย้ายมาตรงนี้เพื่อให้ข้อมูล 'nf' พร้อมใช้) ---
    nf = next_fixture_features(fixtures_df, teams, target_event)
    
    # --- NEW (v2.0): Get Understat data BEFORE engineering features ---
    us_players, us_teams = get_understat_data()

    # --- NEW (v2.0): Use Enhanced Feature Engineering ---
    feat = engineer_features_enhanced(elements, teams, nf, us_players)
    feat.set_index('id', inplace=True)
    # Map enhanced prediction to standard column for compatibility
    feat["pred_points"] = feat["pred_points_enhanced"]

    # --- START: Create player search map for simulation & ROI ---
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
            ##st.header(f"ภาพรวมสัปดาห์ที่ {target_event} (GW{target_event} Overview)")
            
            # --- NEW: Generate Fixture Planner data ---
            opponent_matrix, difficulty_matrix = get_fixture_difficulty_matrix(fixtures_df, teams, target_event)
            rotation_pairs = find_rotation_pairs(difficulty_matrix, teams, feat)

            # --- NEW (v1.9.7): Merge Understat Data for Dashboard ---
            merged_us_players, merged_us_teams = merge_understat_data(
                us_players, us_teams, feat, teams
            )

            display_home_dashboard(
                feat, nf, teams, 
                opponent_matrix, difficulty_matrix, rotation_pairs,
                merged_us_players, merged_us_teams # <-- Pass new data
            )
        
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
                if not entry or 'name' not in entry:
                     st.error(f"ไม่สามารถดึงข้อมูลทีม ID {entry_id} ได้ในขณะนี้ (API อาจกำลังอัปเดต หรือ ID ไม่ถูกต้อง)")
                     st.session_state.analysis_submitted = False
                     st.stop()
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
                    cap_id, vc_id = select_captain_vice(xi_df)
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
                    ordered_bench_df = smart_bench_order(bench_df)
                    ordered_bench_df['pos'] = ordered_bench_df['element_type'].map(POSITIONS)

                    insights = analyze_lineup_insights(xi_df, ordered_bench_df)
                    if insights:
                        st.info("💡 **คำแนะนำ:**\n\n" + "\n\n".join([f"- {i}" for i in insights]))
                    
                                # --- ปรับปรุง Bench Display (เริ่ม) ---
                    bench_display_df = ordered_bench_df[['web_name', 'team_short', 'pos', 'pred_points']].copy()
                    bench_display_df.reset_index(drop=True, inplace=True)
                    bench_display_df.index = bench_display_df.index + 1
                                    # --- ปรับปรุง Bench Display (จบ) ---

                    display_user_friendly_table(
                        df=bench_display_df,
                        title="ตัวสำรอง (Simulated Team - เรียงตามลำดับ)",
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
                            cap_id, vc_id = select_captain_vice(xi_df)
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
                            ordered_bench_df = smart_bench_order(bench_df)
                            ordered_bench_df['pos'] = ordered_bench_df['element_type'].map(POSITIONS)

                            insights = analyze_lineup_insights(xi_df, ordered_bench_df)
                            if insights:
                                st.info("💡 **คำแนะนำ:**\n\n" + "\n\n".join([f"- {i}" for i in insights]))
                            
                                # --- ปรับปรุง Bench Display (เริ่ม) ---
                            bench_display_df = ordered_bench_df[['web_name', 'team_short', 'pos', 'pred_points']].copy()
                            bench_display_df.reset_index(drop=True, inplace=True)
                            bench_display_df.index = bench_display_df.index + 1
                                # --- ปรับปรุง Bench Display (จบ) ---

                            display_user_friendly_table(
                                df=bench_display_df,
                                title="ตัวสำรอง (Simulated Team - เรียงตามลำดับ)",
                                height=175
                            )
                    
                    # --- NEW (v2.0): Transfer ROI Calculator (Improved Logic) ---
                    st.markdown("---")
                    st.subheader("🧮 คำนวณความคุ้มค่าการย้ายตัว (Transfer ROI Calculator)")
                    st.markdown("💡 เลือกนักเตะจากทีมของคุณเพื่อขายออก (OUT) ระบบจะกรองนักเตะตำแหน่งเดียวกันมาให้เลือกซื้อเข้า (IN) เพื่อเปรียบเทียบคะแนน 3 นัดล่วงหน้า")
                    
                    with st.expander("คลิกเพื่อเปิดเครื่องมือคำนวณ (3-GW Projection)", expanded=True):
                        # 1. เตรียมข้อมูลสำหรับ Dropdown Player OUT (เฉพาะ 15 คนในทีม)
                        # ใช้ list comprehension สร้างรายการชื่อจาก ID นักเตะในทีมปัจจุบัน
                        squad_options = [player_id_to_name_map[pid] for pid in valid_pick_ids if pid in player_id_to_name_map]
                        
                        col_out, col_in, col_hit = st.columns([2, 2, 1])
                        
                        # --- Player OUT Selection ---
                        with col_out:
                            p_out_name = st.selectbox(
                                "🔴 Player OUT (เลือกจากทีมปัจจุบัน)", 
                                options=squad_options, 
                                key="roi_out_restricted"
                            )
                            # หา ID จากชื่อที่เลือก
                            p_out_id = player_search_map[p_out_name]
                            # หาตำแหน่งของนักเตะที่เลือก (1=GK, 2=DEF, 3=MID, 4=FWD)
                            out_player_pos_id = feat.loc[p_out_id, 'element_type']
                            out_player_pos_name = POSITIONS.get(out_player_pos_id, "")

                        # --- Player IN Selection (Dynamic Filtering) ---
                        with col_in:
                            # กรองเฉพาะนักเตะที่ตำแหน่งตรงกับคนที่จะขายออก (และไม่ใช่คนเดิม)
                            filtered_in_df = feat[
                                (feat['element_type'] == out_player_pos_id) & 
                                (feat.index != p_out_id)
                            ].sort_values('web_name')
                            
                            # สร้าง options ใหม่สำหรับขาเข้าตามตำแหน่งที่กรองแล้ว
                            in_player_options = [
                                f"{row['web_name']} ({row['team_short']}) - £{row['now_cost']/10.0}m"
                                for idx, row in filtered_in_df.iterrows()
                            ]
                            
                            p_in_name = st.selectbox(
                                f"🟢 Player IN (เฉพาะตำแหน่ง {out_player_pos_name})", 
                                options=in_player_options,
                                key="roi_in_restricted"
                            )
                            p_in_id = player_search_map[p_in_name]

                        # --- Hit Selection ---
                        with col_hit:
                            # เปลี่ยน options เป็นข้อความที่เข้าใจง่าย
                            hit_option = st.radio(
                                "การย้ายตัวโดนหักคะแนนหรือไม่?",
                                ["ไม่เสีย", "โดนหัก (-4)"],
                                horizontal=True,
                                key="roi_hit_val",
                                help="เลือก 'โดนหัก (-4)' ถ้าการย้ายตัวนี้ทำให้คุณเสียแต้ม"
                            )
                            # แปลงข้อความที่เลือกกลับเป็นตัวเลขสำหรับคำนวณ
                            hit_val = 4 if hit_option == "โดนหัก (-4)" else 0

                        # --- Calculation Button & Result ---
                        if st.button("คำนวณความคุ้มค่า (Calculate ROI)", type="primary", use_container_width=True):
                            # ใช้ค่า hit_val ที่แปลงเป็นตัวเลขแล้วส่งเข้าฟังก์ชัน
                            roi_data = calculate_transfer_roi(p_out_id, p_in_id, target_event, feat, fixtures_df, teams, hit_cost=hit_val)
                            
                            # แสดงผลลัพธ์
                            st.markdown(f"##### ผลการวิเคราะห์: {out_player_pos_name} Transfer")
                            c1, c2, c3 = st.columns(3)
                            c1.metric(f"🔴 OUT: {feat.loc[p_out_id, 'web_name']}", f"{roi_data['out_xp_3gw']:.1f} pts", help="คะแนนคาดการณ์รวม 3 นัดถัดไป")
                            c2.metric(f"🟢 IN: {feat.loc[p_in_id, 'web_name']}", f"{roi_data['in_xp_3gw']:.1f} pts", help="คะแนนคาดการณ์รวม 3 นัดถัดไป")
                            
                            delta = roi_data['net_gain']
                            c3.metric("Net Gain (3 GWs)", f"{delta:+.1f} pts", delta=delta, help="ผลต่างคะแนนสุทธิหลังหักลบค่า Hit แล้ว")
                            
                            # คำแนะนำสรุป
                            if roi_data['is_worth_it']:
                                st.success(f"✅ **แนะนำให้เปลี่ยน!** {feat.loc[p_in_id, 'web_name']} น่าจะทำแต้มได้มากกว่าในระยะยาว (3 นัด)")
                            elif delta > 0:
                                st.warning(f"⚠️ **เปลี่ยนได้แต่มีความเสี่ยง** คุ้มทุนเพียงเล็กน้อย ({delta:+.1f} แต้ม)")
                            else:
                                st.error(f"❌ **ไม่แนะนำ** {feat.loc[p_out_id, 'web_name']} ยังน่าจะทำแต้มได้ดีกว่า หรือไม่คุ้มค่า Hit")

                    st.markdown("---")

                    # --- ENHANCED: Transfer Suggestions (Single View) ---
                    st.subheader("🔄 แนะนำการย้ายตัว (Suggested Transfers)")
                    st.markdown("💡 คำแนะนำนี้ใช้ **ราคาขายจริง (Selling Price)** จาก FPL API ของคุณ")
                    
                    with st.spinner("Analyzing potential transfers..."):
                        
                        # ใช้ฟังก์ชัน suggest_transfers ดั้งเดิม (เนื่องจากเรามี selling_price ที่แม่นยำแล้ว)
                        moves = suggest_transfers(
                            current_squad_ids=valid_pick_ids,
                            bank=bank,
                            free_transfers=free_transfers, # นี่คือค่าที่ User เลือกใน Sidebar
                            all_players=feat,
                            strategy=transfer_strategy,
                            fixtures_df=fixtures_df,
                            teams_df=teams,
                            current_event=target_event
                        )

                        if moves:
                            moves_df = pd.DataFrame(moves)
                            moves_df = moves_df.reset_index(drop=True)
                            moves_df.index = moves_df.index + 1
                            moves_df.index.name = "ลำดับ"
                            
                            total_out = moves_df['out_cost'].sum()
                            total_in = moves_df['in_cost'].sum()
                            total_hit_cost = moves_df['hit_cost'].sum()
                            
                            st.info(f"💰 งบประมาณ: ขายออก **£{total_out:.1f}m** | ซื้อเข้า **£{total_in:.1f}m** | เสียแต้ม: **-{total_hit_cost}**")
                            
                            # แปลงชื่อคอลัมน์ให้เข้าใจง่ายขึ้น
                            cols_to_ren = {
                                "out_name": "ขายออก (Out)",
                                "out_cost": "ราคาขาย (£)",
                                "in_name": "ซื้อเข้า (In)",
                                "in_cost": "ราคาซื้อ (£)",
                                "delta_points": "กำไร (GW นี้)",
                                "roi_3gw": "กำไร (3 GW)",
                                "net_gain": "กำไรสุทธิ (GW นี้)",
                                "hit_cost": "แต้มที่เสีย"
                            }
                            
                            moves_display = moves_df.rename(columns=cols_to_ren)
                            
                            # กำหนดคอลัมน์และลำดับที่จะแสดงผล
                            final_cols_order = [
                                "ขายออก (Out)", 
                                "ราคาขาย (£)", 
                                "ซื้อเข้า (In)", 
                                "ราคาซื้อ (£)",
                                "กำไร (GW นี้)",
                                "กำไร (3 GW)",
                                "แต้มที่เสีย",
                                "กำไรสุทธิ (GW นี้)"
                            ]
                            
                            # กรองเฉพาะคอลัมน์ที่มีอยู่จริง
                            final_cols = [c for c in final_cols_order if c in moves_display.columns]

                            dynamic_height = 45 + (len(moves_df) * 35)
                            display_user_friendly_table(
                                df=moves_display[final_cols],
                                title="",
                                height=dynamic_height
                            )
                        else:
                            st.success("✅ ทีมของคุณยอดเยี่ยมแล้ว! ไม่จำเป็นต้องย้ายตัวในสัปดาห์นี้")                   
                       
                        # Add warning
                        st.warning("⚠️ **สำคัญ**: ตรวจสอบราคาขายจริงในแอป FPL ก่อนทำ transfer")
                    
                    st.markdown("---")
                    
                    # --- START: NEW SIMULATION SECTION (MOVED) ---
                    st.subheader("🛠️ ทดลองจัดทีม (Simulation Mode)")
                    st.markdown("ใช้ส่วนนี้เพื่อจำลองการย้ายทีมของคุณ *หลังจาก* ที่คุณกดยืนยันใน FPL แล้ว แต่ในนี้ยังไม่อัปเดตตาม")
                    
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
                                cap_id_sim, vc_id_sim = select_captain_vice(xi_df_sim)
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
                                ordered_bench_df = smart_bench_order(bench_df)
                                ordered_bench_df['pos'] = ordered_bench_df['element_type'].map(POSITIONS)

                                insights = analyze_lineup_insights(xi_df_sim, ordered_bench_df)
                                if insights:
                                    st.info("💡 **คำแนะนำ:**\n\n" + "\n\n".join([f"- {i}" for i in insights]))
                                
                                                                # --- ปรับปรุง Bench Display (เริ่ม) ---
                                bench_display_df = ordered_bench_df[['web_name', 'team_short', 'pos', 'pred_points']].copy()
                                bench_display_df.reset_index(drop=True, inplace=True)
                                bench_display_df.index = bench_display_df.index + 1
                                # --- ปรับปรุง Bench Display (จบ) ---

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