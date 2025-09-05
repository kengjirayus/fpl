"""
FPL Weekly Assistant — single-file Streamlit app (Thai Updated Version)

What it does
- Pulls live data from FPL API (bootstrap-static, fixtures, entry picks)
- Engineers features (recent form, xGI proxy, minutes reliability, fixture difficulty)
- Predicts next GW points with a hybrid approach:
  • If you have a local historical CSV (optional), trains a RandomForestRegressor
  • Otherwise uses a robust heuristic model tailored to FPL signals
- Optimizes your Starting XI & bench order subject to FPL formation rules
- Suggests transfers based on selected strategy (Free, Hits, or Wildcard) to maximize net expected points

How to run
1) pip install streamlit pandas numpy scikit-learn pulp requests
2) streamlit run fpl_assistant.py

Notes
- This app reads public FPL endpoints. No login required.
- Transfer suggestions consider the upcoming GW only by default.
- If you provide a historical CSV (schema documented below), the ML model will be used.
"""

import os
import math
import json
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
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

TEAM_MAP_COLS = ["id","name","short_name","strength_overall_home","strength_overall_away",
                 "strength_attack_home","strength_attack_away","strength_defence_home","strength_defence_away"]

def build_master_tables(bootstrap: Dict, fixtures: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Constructs the main dataframes for players, teams, events, and fixtures."""
    elements = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])[TEAM_MAP_COLS]
    events = pd.DataFrame(bootstrap.get("events", []))

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

    for _, row in next_gw_fixtures.iterrows():
        home_team_id, away_team_id = row['team_h'], row['team_a']
        team_data[home_team_id]['home_fixtures'].append(away_team_id)
        team_data[away_team_id]['away_fixtures'].append(home_team_id)

    for team_id, fixtures_info in team_data.items():
        home_opps = fixtures_info['home_fixtures']
        away_opps = fixtures_info['away_fixtures']
        
        num_fixtures = len(home_opps) + len(away_opps)
        
        # Blank Gameweek (BGW)
        if num_fixtures == 0:
            rows.append({
                'team': team_id, 
                'num_fixtures': 0, 
                'total_opp_def_str': 0, 
                'avg_fixture_ease': 0
            })
            continue

        # Double Gameweek (DGW) or single GW
        total_opp_def_str = 0
        total_opp_att_str = 0
        for opp_id in home_opps:
            opp_team = teams_df.set_index('id').loc[opp_id]
            total_opp_def_str += opp_team['strength_defence_away']
            total_opp_att_str += opp_team['strength_attack_away']
        for opp_id in away_opps:
            opp_team = teams_df.set_index('id').loc[opp_id]
            total_opp_def_str += opp_team['strength_defence_home']
            total_opp_att_str += opp_team['strength_attack_home']

        rows.append({
            'team': team_id,
            'num_fixtures': num_fixtures,
            'total_opp_def_str': total_opp_def_str,
            'avg_fixture_ease': 1.0 - (total_opp_def_str / (num_fixtures * teams_df['strength_defence_home'].max()))
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

    for col in ["form", "points_per_game", "ict_index", "selected_by_percent", "now_cost", "starts"]:
        elements[col] = pd.to_numeric(elements[col], errors="coerce").fillna(0)
    
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

###############################
# Squad & optimization
###############################

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

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
    """Greedy search for transfers based on the selected strategy."""
    
    current_squad_df = all_players.loc[current_squad_ids]
    start_ids, _ = optimize_starting_xi(current_squad_df)
    if not start_ids: return []
    base_sum = float(current_squad_df.loc[start_ids]['pred_points'].sum())

    # Modify transfer limits and hit costs based on strategy
    if strategy == "Free Transfer":
        max_transfers = free_transfers
        hit_cost = float('inf')
    elif strategy == "Allow Hit (AI Suggest)":
        max_transfers = 5
        hit_cost = 4
    else:  # Wildcard/Free Hit
        max_transfers = 15
        hit_cost = 0

    position_groups = {}
    for out_id in current_squad_ids:
        out_player = all_players.loc[out_id]
        pos = out_player['element_type']
        if pos not in position_groups:
            position_groups[pos] = []
        position_groups[pos].append(out_id)

    potential_moves = []
    used_in_players = set()
    remaining_bank = bank

    for pos in sorted(position_groups.keys()):
        out_ids = position_groups[pos]
        
        max_budget_for_pos = all_players.loc[out_ids]['selling_price'].max() + (bank * 10)
        
        all_replacements = all_players[
            (all_players['element_type'] == pos) &
            (~all_players.index.isin(current_squad_ids)) &
            (all_players['now_cost'] <= max_budget_for_pos)
        ].sort_values('pred_points', ascending=False)

        for out_id in sorted(out_ids, key=lambda x: all_players.loc[x, 'pred_points']):
            out_player = all_players.loc[out_id]
            
            budget_for_replacement = out_player['selling_price'] + (remaining_bank * 10)
            
            valid_replacements = all_replacements[
                (~all_replacements.index.isin(used_in_players)) &
                (all_replacements['pred_points'] > out_player['pred_points']) &
                (all_replacements['now_cost'] <= budget_for_replacement)
            ]

            if valid_replacements.empty:
                continue

            in_player = valid_replacements.iloc[0]
            cost_change = (in_player['now_cost'] - out_player['selling_price']) / 10.0

            if cost_change <= remaining_bank:
                potential_moves.append({
                    "out_id": out_id,
                    "in_id": in_player.name,
                    "out_name": out_player["web_name"],
                    "in_name": in_player["web_name"],
                    "out_pos": POSITIONS[out_player["element_type"]],
                    "in_pos": POSITIONS[in_player["element_type"]],
                    "delta_points": in_player['pred_points'] - out_player['pred_points'],
                    "in_cost": in_player['now_cost'] / 10.0,
                    "out_cost": out_player['selling_price'] / 10.0,
                })
                used_in_players.add(in_player.name)
                remaining_bank -= cost_change

    potential_moves = sorted(potential_moves, key=lambda x: x["delta_points"], reverse=True)

    final_suggestions = []
    total_hit_cost = 0

    for move in potential_moves[:max_transfers]:
        hit = 0 if len(final_suggestions) < free_transfers else hit_cost
        total_hit_cost += hit
        
        net_gain = move["delta_points"]
        if hit > 0:
            net_gain = net_gain * 2.5 - total_hit_cost

        if strategy == "Free Transfer" or net_gain > -0.1:
            move['net_gain'] = round(net_gain, 2)
            move['hit_cost'] = hit
            final_suggestions.append(move)

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
        current_price = all_players.loc[player_id, 'selling_price']
        # ลดราคาขายลง 0.2 เพื่อความปลอดภัย
        conservative_price = max(current_price - 2, current_price * 0.95)  # ลดขั้นต่ำ 0.2 หรือ 5%
        conservative_all_players.loc[player_id, 'selling_price'] = conservative_price
    
    # คำนวณงบที่มีอยู่จริงหลังจากลดราคาขาย
    conservative_bank = bank
    for move in normal_moves:
        out_id = move['out_id']
        original_price = all_players.loc[out_id, 'selling_price']
        conservative_price = conservative_all_players.loc[out_id, 'selling_price']
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
                move['out_cost'] = round(conservative_all_players.loc[move['out_id'], 'selling_price'] / 10.0, 1)
                filtered_conservative_moves.append(move)
                remaining_bank -= cost_change
                used_players.add(move['in_id'])
    
    return normal_moves, filtered_conservative_moves

###############################
# Streamlit UI
###############################

def main():
    st.set_page_config(page_title="FPL WIZ จัดตัวนักเตะ", layout="wide")
    st.title("🏟️ FPL WIZ จัดตัวนักเตะด้วย AI | FPL WIZ AI-Powered 🤖")
    st.markdown("เครื่องมือช่วยวิเคราะห์และแนะนำนักเตะ FPL ในแต่ละสัปดาห์ 🧠")

    with st.sidebar:
        st.header("⚙️ Settings | ตั้งค่า")

        # Callback function to clear the text input
        def reset_team_id():
            st.session_state.team_id_input = ""

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

    # --- โค้ดที่เพิ่มใหม่เพื่อแสดงข้อมูล DGW/BGW ---
    nf = next_fixture_features(fixtures_df, teams, target_event)
    dgw_teams = nf[nf['num_fixtures'] == 2]['team'].map(teams.set_index('id')['short_name'])
    bgw_teams = nf[nf['num_fixtures'] == 0]['team'].map(teams.set_index('id')['short_name'])

    dgw_note = ""
    bgw_note = ""
    if not dgw_teams.empty:
        dgw_note = f"สัปดาห์นี้มี Double Gameweek: **{', '.join(dgw_teams)}**"
    if not bgw_teams.empty:
        bgw_note = f"สัปดาห์นี้มี Blank Gameweek: **{', '.join(bgw_teams)}**"
    
    if dgw_note and bgw_note:
        st.info(f"💡 {dgw_note}. {bgw_note}")
    elif dgw_note:
        st.info(f"💡 {dgw_note}")
    elif bgw_note:
        st.info(f"💡 {bgw_note}")
    # ----------------------------------------------------
    feat = engineer_features(elements, teams, nf)
    feat.set_index('id', inplace=True)
    feat["pred_points"] = feat["pred_points_heur"]


    if not submitted:
        st.header("⭐ Top Projected Players")
        st.markdown(f"ผู้เล่นที่คาดว่าจะทำคะแนนได้สูงสุดใน GW {target_event}")
        
        # แก้ไขชื่อคอลัมน์ 'fixture_ease' เป็น 'avg_fixture_ease'
        show_cols = ["web_name", "team_short", "element_type", "now_cost", "form", "avg_fixture_ease", "pred_points"]
        top_tbl = feat[show_cols].copy()
        
        # เพิ่มการเปลี่ยนชื่อคอลัมน์ avg_fixture_ease ให้เป็น fixture_ease เพื่อการแสดงผลที่สวยงาม
        top_tbl.rename(columns={"element_type": "pos", "now_cost": "price", "avg_fixture_ease": "fixture_ease"}, inplace=True)
        top_tbl["pos"] = top_tbl["pos"].map(POSITIONS)
        top_tbl["price"] = (top_tbl["price"] / 10.0).round(1)
        top_tbl["pred_points"] = top_tbl["pred_points"].round(2)
        st.dataframe(top_tbl.sort_values("pred_points", ascending=False).head(25), use_container_width=True)

    if submitted:
        if entry_id_str:
            if not entry_id_str.isdigit():
                st.error("❗ กรุณากรอก FPL Team ID เป็นตัวเลขเท่านั้น")
                st.stop()
            try:
                entry_id = int(entry_id_str)
                entry = get_entry(entry_id)
                ev_for_picks = cur_event or 1
                picks = get_entry_picks(entry_id, ev_for_picks)

                # ========== การจัดการข้อมูล selling_price ที่อาจหายไป ==========
                picks_data = picks.get("picks", [])

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
                        now_cost = feat.loc[player_id, 'now_cost']
                        profit = now_cost - purchase_price
                        selling_price = purchase_price + (profit // 2)  # ปัดเศษลง
                        selling_price_map[player_id] = selling_price
                    else:
                        # fallback สุดท้าย: ใช้ราคาปัจจุบัน
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
                
                else:  # Free Transfer / Allow Hit (Enhanced with AI Suggest)
                    bank = (entry.get('last_deadline_bank', 0)) / 10.0
                    free_transfers_from_api = entry.get('free_transfers', 1)

                    pick_ids = [p["element"] for p in picks.get("picks", [])]
                    squad_df = feat.loc[pick_ids].copy() 
                    
                    overall_points = entry.get('summary_overall_points', 0)
                    gameweek_points = entry.get('summary_event_points', 0)

                    xi_ids, bench_ids = optimize_starting_xi(squad_df)
                    
                    st.info(f"🏦 Bank: **£{bank:.1f}m** | 🆓 Free Transfer: **{free_transfers_from_api}** | 🎯 Overall points: **{overall_points}** | Gameweek points: **{gameweek_points}**")

                if not xi_ids or len(xi_ids) != 11:
                    st.error("Could not form a valid starting XI from your current squad. This can happen with unusual team structures (e.g., during pre-season).")
                    st.write("Current Squad Composition:")
                    st.dataframe(squad_df[['web_name', 'element_type']].rename(columns={'element_type':'pos'}).assign(pos=lambda df: df['pos'].map(POSITIONS)))
                else:
                    xi_df = squad_df.loc[xi_ids].copy()
                    xi_df['pos'] = xi_df['element_type'].map(POSITIONS)

                    position_order = ['GK', 'DEF', 'MID', 'FWD']
                    xi_df['pos'] = pd.Categorical(xi_df['pos'], categories=position_order, ordered=True)

                    xi_df['pred_points'] = xi_df['pred_points'].round(2)
                    st.markdown("✅ **แนะนำนักเตะ 11 ตัวจริง**")
                    st.dataframe(xi_df[['web_name', 'team_short', 'pos', 'pred_points']].sort_values('pos'), use_container_width=True, height=420)

                    cap_row = xi_df.sort_values("pred_points", ascending=False).iloc[0]
                    vc_row = xi_df.sort_values("pred_points", ascending=False).iloc[1]
                    st.success(f"👑 Captain: **{cap_row['web_name']}** ({cap_row['team_short']}) | Vice-Captain: **{vc_row['web_name']}** ({vc_row['team_short']})")
                    
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
                    ordered_bench_df['pred_points'] = ordered_bench_df['pred_points'].round(2)
                    st.markdown("**ตัวสำรอง (เรียงตามความสามารถ)**")
                    st.dataframe(ordered_bench_df[['web_name', 'team_short', 'pos', 'pred_points']], use_container_width=True)
                    
                    if transfer_strategy == "Wildcard / Free Hit":
                        total_points = squad_df['pred_points'].sum()
                        total_cost = squad_df['now_cost'].sum() / 10.0
                        st.success(f"Total Expected Points: **{total_points:.1f}** | Team Value: **£{total_cost:.1f}m**")
                    
                    else:
                        st.subheader("🔄 Suggested Transfers")
                        st.markdown(f"💡 คำแนะนำซื้อขายนักเตะจากทีมคุณ ⚠️เนื่องจากข้อจำกัดของ FPL API เราแสดง 2 มุมมองเพื่อให้คุณตัดสินใจ 🔎")
                        with st.spinner("Analyzing potential transfers..."):
                            normal_moves, conservative_moves = suggest_transfers_enhanced(
                                pick_ids, bank=bank, free_transfers=free_transfers,
                                all_players=feat, strategy=transfer_strategy
                            )

                        if not normal_moves and not conservative_moves:
                            st.write("⚠️ ไม่มีคำแนะนำการซื้อขายนักเตะ ลองเปลี่ยนกลยุทธ์หรือเพิ่ม Free Transfer")
                        else:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### 📊 ข้อเสนอหลัก (ราคาปัจจุบัน)")
                                if normal_moves:
                                    normal_df = pd.DataFrame(normal_moves)
                                    normal_df.index = np.arange(1, len(normal_df) + 1)
                                    total_in = normal_df['in_cost'].sum()
                                    total_out = normal_df['out_cost'].sum()
                                    st.info(f"💰 งบประมาณ: ซื้อเข้า **£{total_in:.1f}m** | ขายออก **£{total_out:.1f}m**")
                                    st.dataframe(normal_df[["out_name", "in_name", "delta_points", "net_gain", "out_cost", "in_cost"]], 
                                               use_container_width=True)
                                else:
                                    st.write("ไม่มีการแนะนำ")
                            
                            with col2:
                                st.markdown("#### 🛡️ ข้อเสนอสำรอง (ปรับราคาขายลง)")
                                if conservative_moves:
                                    conservative_df = pd.DataFrame(conservative_moves)
                                    conservative_df.index = np.arange(1, len(conservative_df) + 1)
                                    total_in_c = conservative_df['in_cost'].sum()
                                    total_out_c = conservative_df['out_cost'].sum()
                                    st.info(f"💰 งบประมาณ: ซื้อเข้า **£{total_in_c:.1f}m** | ขายออก **£{total_out_c:.1f}m**")
                                    st.dataframe(conservative_df[["out_name", "in_name", "delta_points", "net_gain", "out_cost", "in_cost"]], 
                                               use_container_width=True)
                                    st.caption("🔍 ราคาขายลดลง 0.1-0.2m เผื่อกรณีราคาเปลี่ยนแปลง")
                                else:
                                    st.write("ไม่มีการแนะนำที่ปลอดภัยพอ")
                            
                            # เพิ่มคำเตือน
                            st.warning("⚠️ **สำคัญ**: ตรวจสอบราคาขายจริงในแอป FPL ก่อนทำ transfer")

            except requests.exceptions.HTTPError as e:
                st.error(f"Could not fetch data for Team ID {entry_id_str}. Please check if the ID is correct. (Error: {e.response.status_code})")
            except (ValueError, TypeError):
                st.error("Invalid Team ID. Please enter a numeric ID.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.exception(e)
        else:
            st.error("❗กรุณากรอก FPL Team ID ของคุณในช่องด้านข้างเพื่อเริ่มการวิเคราะห์")
            st.info("💡 FPL Team ID จากเว็บไซต์ https://fantasy.premierleague.com/ คลิกที่ Points แล้วจะเห็น Team ID ตามตัวอย่างรูปด้านล่าง")

            st.markdown(
                """
                <style>
                .custom-image img {
                    width: 100%;
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

if __name__ == "__main__":
    main()