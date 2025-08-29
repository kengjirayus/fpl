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
# API helpers
###############################
FPL_BASE = "https://fantasy.premierleague.com/api"

@st.cache_data(ttl=300)
def get_bootstrap() -> Dict:
    """Fetches the main bootstrap data from FPL API."""
    r = requests.get(f"{FPL_BASE}/bootstrap-static/")
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300)
def get_fixtures() -> List[Dict]:
    """Fetches the fixtures data from FPL API."""
    r = requests.get(f"{FPL_BASE}/fixtures/")
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300)
def get_entry(entry_id: int) -> Dict:
    """Fetches a user's entry (team) data."""
    r = requests.get(f"{FPL_BASE}/entry/{entry_id}/")
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300)
def get_entry_picks(entry_id: int, event: int) -> Dict:
    """Fetches a user's picks for a specific gameweek."""
    r = requests.get(f"{FPL_BASE}/entry/{entry_id}/event/{event}/picks/")
    r.raise_for_status()
    return r.json()

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
    """Computes next-game fixture difficulty per team."""
    next_gw_fixtures = fixtures_df[fixtures_df["event"] == event_id].copy()
    if next_gw_fixtures.empty:
        return pd.DataFrame()

    rows = []
    for _, row in next_gw_fixtures.iterrows():
        home_team_id, away_team_id = row['team_h'], row['team_a']
        home_team = teams_df.set_index('id').loc[home_team_id]
        away_team = teams_df.set_index('id').loc[away_team_id]

        rows.append({
            'team': home_team_id, 'opp': away_team_id, 'is_home': 1,
            'opp_att_str': away_team['strength_attack_away'],
            'opp_def_str': away_team['strength_defence_away']
        })
        rows.append({
            'team': away_team_id, 'opp': home_team_id, 'is_home': 0,
            'opp_att_str': home_team['strength_attack_home'],
            'opp_def_str': home_team['strength_defence_home']
        })
    df = pd.DataFrame(rows)
    # Normalize difficulty
    max_def = df['opp_def_str'].max()
    df['fixture_diff'] = df['opp_def_str'] / max_def if max_def > 0 else 0.5
    return df

def engineer_features(elements: pd.DataFrame, teams: pd.DataFrame, nf: pd.DataFrame) -> pd.DataFrame:
    """Joins player table with next fixture info and creates predictive features."""
    # **FIX**: Sanitize element_type to prevent errors from bad API data
    elements["element_type"] = pd.to_numeric(elements["element_type"], errors='coerce').fillna(0).astype(int)

    if nf.empty:
        elements['fixture_ease'] = 0.5
    else:
        elements = elements.merge(nf, on="team", how="left")
        max_opp_def = elements["opp_def_str"].max() if "opp_def_str" in elements.columns and not elements["opp_def_str"].isnull().all() else 1
        elements["opp_def_norm"] = elements["opp_def_str"] / max_opp_def if max_opp_def > 0 else 0.5
        elements["fixture_ease"] = 1.0 - elements["opp_def_norm"].fillna(0.5)

    # Basic player signals
    for col in ["form", "points_per_game", "ict_index", "selected_by_percent", "now_cost", "starts"]:
        elements[col] = pd.to_numeric(elements[col], errors="coerce").fillna(0)

    # Play probability
    elements["chance_of_playing_next_round"] = pd.to_numeric(elements["chance_of_playing_next_round"], errors="coerce").fillna(100)
    elements["play_prob"] = elements["chance_of_playing_next_round"] / 100.0

    # Simple xGI proxy from ICT and PPG
    elements["xgi_proxy"] = 0.6 * elements["points_per_game"] + 0.4 * (elements["ict_index"] / 10.0)

    # Heuristic prediction model
    pos_mult = np.select(
        [elements["element_type"] == 1, elements["element_type"] == 2, elements["element_type"] == 3, elements["element_type"] == 4],
        [0.6, 0.8, 1.0, 1.1],
        default=1.0
    )
    elements["pred_points_heur"] = (
        (0.45 * elements["xgi_proxy"] + 0.35 * elements["form"] + 0.2 * elements["points_per_game"]) *
        (0.6 + 0.4 * elements["fixture_ease"]) *
        (0.5 + 0.5 * elements["play_prob"]) * pos_mult
    )
    elements["pred_points_heur"] = elements["pred_points_heur"].clip(lower=0, upper=15)

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
        max_transfers = free_transfers  # Exactly match user selection
        hit_cost = float('inf')  # Never take hits
    elif strategy == "Allow Hit (AI Suggest)":
        max_transfers = 5  # Allow up to 5 transfers total
        hit_cost = 4  # -4 points per extra transfer
    else:  # Wildcard/Free Hit
        max_transfers = 15
        hit_cost = 0

    # Group players by position for better replacement suggestions
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

    # Process each position separately
    for pos in sorted(position_groups.keys()):
        out_ids = position_groups[pos]
        
        # Get all possible replacements for this position
        all_replacements = all_players[
            (all_players['element_type'] == pos) &
            (~all_players.index.isin(current_squad_ids)) &
            (all_players['now_cost'] <= max(all_players.loc[out_ids]['now_cost'].max() + (bank * 10), all_players.loc[out_ids]['now_cost'].min()))
        ].sort_values('pred_points', ascending=False)

        # Find best unique replacements for each outgoing player
        for out_id in sorted(out_ids, key=lambda x: all_players.loc[x, 'pred_points']):
            out_player = all_players.loc[out_id]
            
            valid_replacements = all_replacements[
                (~all_replacements.index.isin(used_in_players)) &
                (all_replacements['pred_points'] > out_player['pred_points']) &
                (all_replacements['now_cost'] <= out_player['now_cost'] + (remaining_bank * 10))
            ]

            if valid_replacements.empty:
                continue

            # Take best available replacement
            in_player = valid_replacements.iloc[0]
            cost_change = (in_player['now_cost'] - out_player['now_cost']) / 10.0

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
                    "out_cost": out_player['now_cost'] / 10.0,
                })
                used_in_players.add(in_player.name)
                remaining_bank -= cost_change

    # Sort moves by predicted point improvement
    potential_moves = sorted(potential_moves, key=lambda x: x["delta_points"], reverse=True)

    # Calculate final suggestions with hit costs
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

###############################
# Streamlit UI
###############################

def main():
    st.set_page_config(page_title="FPL Weekly Assistant", layout="wide")
    st.title("🏟️ FPL Weekly Assistant — AI-Powered Suggestions")
    st.markdown("เครื่องมือช่วยวิเคราะห์และแนะนำผู้เล่น FPL ในแต่ละสัปดาห์")

    with st.sidebar:
        st.header("Settings")
        with st.form("settings_form"):
            entry_id_str = st.text_input(
                "Your FPL Team ID (ระบุ ID ทีมของคุณ)",
                help="นำเลข ID ทีมของคุณจาก URL หลังจากเข้าสู่ระบบ FPL บนเว็บแล้ว Click ดู Points จะเห็น URL https://fantasy.premierleague.com/entry/xxxxxxx/event/2 ให้นำเลข xxxxxxx มาใส่"
            )
            
            transfer_strategy = st.radio(
                "Transfer Strategy (เลือกรูปแบบการเปลี่ยนตัว)",
                ("Free Transfer", "Allow Hit (AI Suggest)", "Wildcard / Free Hit")
            )
            
            free_transfers = 1
            if transfer_strategy == "Free Transfer":
                free_transfers = st.number_input(
                    "Number of Free Transfers Available (จำนวน FT ที่มี)",
                    min_value=0,
                    max_value=5,
                    value=1,
                    help="Select how many free transfers you have available (0-5)"
                )
            elif transfer_strategy == "Allow Hit (AI Suggest)":
                free_transfers = 1
            
            # Button to submit the form
            submitted = st.form_submit_button("Analyze Team")

    bootstrap = get_bootstrap()
    fixtures = get_fixtures()
    elements, teams, events, fixtures_df = build_master_tables(bootstrap, fixtures)
    cur_event, next_event = current_and_next_event(bootstrap.get("events", []))
    
    if not next_event and not cur_event:
        st.error("Could not determine the current or next gameweek.")
        st.stop()
        
    target_event = next_event or (cur_event + 1 if cur_event else 1)
    
    st.info(f"Current GW: **{cur_event}** | Target GW for analysis: **{target_event}**")

    nf = next_fixture_features(fixtures_df, teams, target_event)
    feat = engineer_features(elements, teams, nf)
    feat.set_index('id', inplace=True)
    feat["pred_points"] = feat["pred_points_heur"]

    st.header("⭐ Top Projected Players")
    st.markdown(f"ผู้เล่นที่คาดว่าจะทำคะแนนได้สูงสุดใน GW {target_event}")
    
    show_cols = ["web_name", "team_short", "element_type", "now_cost", "form", "fixture_ease", "pred_points"]
    top_tbl = feat[show_cols].copy()
    top_tbl.rename(columns={"element_type": "pos", "now_cost": "price"}, inplace=True)
    top_tbl["pos"] = top_tbl["pos"].map(POSITIONS)
    top_tbl["price"] = (top_tbl["price"] / 10.0).round(1)
    top_tbl["pred_points"] = top_tbl["pred_points"].round(2)
    st.dataframe(top_tbl.sort_values("pred_points", ascending=False).head(25), use_container_width=True)

    if submitted:
        if entry_id_str:
            try:
                entry_id = int(entry_id_str)
                entry = get_entry(entry_id)
                ev_for_picks = cur_event or 1
                picks = get_entry_picks(entry_id, ev_for_picks)
                
                st.header(f"🚀 Analysis for '{entry['name']}'")

                if transfer_strategy == "Wildcard / Free Hit":
                    st.subheader("🤖 AI Suggested Wildcard / Free Hit Team")
                    total_value = (entry.get('last_deadline_value', 1000) + entry.get('last_deadline_bank', 0)) / 10.0
                    st.info(f"Optimizing for a total budget of **£{total_value:.1f}m**")
                    
                    with st.spinner("Finding the optimal 15-man squad... this may take a moment."):
                        wildcard_ids = optimize_wildcard_team(feat, total_value)
                    
                    if wildcard_ids:
                        wc_squad_df = feat.loc[wildcard_ids].copy()
                        wc_squad_df['pos'] = wc_squad_df['element_type'].map(POSITIONS)
                        display_df = (wc_squad_df[['web_name', 'team_short', 'pos', 'now_cost', 'pred_points']]
                                     .rename(columns={'now_cost': 'price'})
                                     .assign(price=lambda df: (df['price']/10.0).round(1))
                                     .sort_values('pos'))
                        st.dataframe(display_df, use_container_width=True)
                        
                        total_points = wc_squad_df['pred_points'].sum()
                        total_cost = wc_squad_df['now_cost'].sum() / 10.0
                        st.success(f"Total Expected Points: **{total_points:.1f}** | Team Value: **£{total_cost:.1f}m**")
                    else:
                        st.error("Could not find an optimal wildcard team. This might be due to budget constraints or player availability.")
                else:
                    bank = (picks.get("entry_history", {}).get("bank", 0)) / 10.0
                    st.info(f"Bank: **£{bank:.1f}m** | Free Transfers: **{free_transfers}**")

                    pick_ids = [p["element"] for p in picks.get("picks", [])]
                    squad_df = feat.loc[pick_ids]

                    st.subheader("✅ Recommended Starting XI & Bench Order")
                    xi_ids, bench_ids = optimize_starting_xi(squad_df)
                    
                    if not xi_ids or len(xi_ids) != 11:
                        st.error("Could not form a valid starting XI from your current squad. This can happen with unusual team structures (e.g., during pre-season).")
                        st.write("Current Squad Composition:")
                        st.dataframe(squad_df[['web_name', 'element_type']].rename(columns={'element_type':'pos'}).assign(pos=lambda df: df['pos'].map(POSITIONS)))
                    else:
                        xi_df = squad_df.loc[xi_ids].copy()
                        xi_df['pos'] = xi_df['element_type'].map(POSITIONS)
                        xi_df['pred_points'] = xi_df['pred_points'].round(2)
                        st.markdown("**Starting XI**")
                        st.dataframe(xi_df[['web_name', 'team_short', 'pos', 'pred_points']].sort_values('pos'), use_container_width=True)

                        cap_row = xi_df.sort_values("pred_points", ascending=False).iloc[0]
                        vc_row = xi_df.sort_values("pred_points", ascending=False).iloc[1]
                        st.success(f"Captain: **{cap_row['web_name']}** ({cap_row['team_short']}) | Vice-Captain: **{vc_row['web_name']}** ({vc_row['team_short']})")

                        bench_df = squad_df.loc[bench_ids].copy()
                        bench_gk = bench_df[bench_df['element_type'] == 1]
                        bench_outfield = bench_df[bench_df['element_type'] != 1].sort_values('pred_points', ascending=False)
                        ordered_bench_df = pd.concat([bench_gk, bench_outfield])
                        ordered_bench_df['pos'] = ordered_bench_df['element_type'].map(POSITIONS)
                        ordered_bench_df['pred_points'] = ordered_bench_df['pred_points'].round(2)
                        st.markdown("**Bench (in order)**")
                        st.dataframe(ordered_bench_df[['web_name', 'team_short', 'pos', 'pred_points']], use_container_width=True)

                        st.subheader("🔄 Suggested Transfers")
                        st.markdown(f"💡 คำแนะนำในการซื้อขายนักเตะจากทีมคุณ")
                        with st.spinner("Analyzing potential transfers..."):
                            moves = suggest_transfers(pick_ids, bank=bank, free_transfers=free_transfers,
                                                      all_players=feat, strategy=transfer_strategy)
                        
                        if not moves:
                            st.write("No clear positive-EV transfer found based on the selected strategy.")
                        else:
                            mv_df = pd.DataFrame(moves)
                            
                            total_in_cost = mv_df['in_cost'].sum()
                            total_out_cost = mv_df['out_cost'].sum()

                            st.success(f"💸 **Total In-Cost:** £{total_in_cost:.1f}m | **Total Out-Cost:** £{total_out_cost:.1f}m")
                            
                            st.dataframe(mv_df[["out_name", "in_name", "delta_points", "net_gain", "in_cost", "out_cost"]], use_container_width=True)

            except requests.exceptions.HTTPError as e:
                st.error(f"Could not fetch data for Team ID {entry_id_str}. Please check if the ID is correct. (Error: {e.response.status_code})")
            except (ValueError, TypeError):
                st.error("Invalid Team ID. Please enter a numeric ID.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.exception(e)
        else:
            st.error("❗กรุณากรอก FPL Team ID ของคุณในช่องด้านข้างเพื่อเริ่มการวิเคราะห์")
            st.info("💡 FPL Team ID ของคุณอยู่ในแถบ **Settings** ด้านซ้ายมือ")                    

if __name__ == "__main__":
    main()
