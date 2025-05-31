import json
import math
import itertools
from itertools import product
import joblib
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from apichecker import score_moveset_with_gpt
import argparse

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def evaluate_configuration_batch(meta, move_sets_batch,
                                 pokemon_info, numeric_cols, categorical_cols,
                                 scaler, encoder, model):
    ability, item, nature, tera, ev_spread, iv_config = meta

    # Build features for the whole batch
    feats = []
    for move_set in move_sets_batch:
        f = build_configuration_features_custom(
            pokemon_info, ev_spread, iv_config,
            nature, ability, item, tera, move_set
        )
        f["_moveset"] = move_set
        feats.append(f)
    df = pd.DataFrame(feats)

    # Vectorized transforms
    X_num = scaler.transform(df[numeric_cols])
    X_cat = encoder.transform(df[categorical_cols])
    X = np.hstack([X_num, X_cat])

    # Batch predict via PyTorch surrogate
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X.astype(np.float32))
        preds = model(X_t).cpu().numpy()   # shape = (n_movesets, 8)
    overall = preds[:, SCORE_KEYS.index("Overall")]
    idx     = int(np.argmax(overall))
    best_pred = float(overall[idx])
    best_row     = df.iloc[idx]
    best_moveset = best_row["_moveset"]

    best_config = {
        "ability":    ability,
        "item":       item,
        "nature":     nature,
        "teraType":   tera,
        "EV_spread":  ev_spread,
        "IV_config":  iv_config,
        "move_set":   best_moveset
    }
    best_features = best_row.drop("_moveset").to_dict()
    return best_pred, best_config, best_features

def generate_moveset(pokemon_name):
    """
    Run the search once for a single Pokémon, print & return its moveset text and GPT score.
    """
    res = find_optimal_configuration_custom(pokemon_name)
    if not res:
        print(f"No data for '{pokemon_name}'.")
        return None, None
    config, _, _, _ = res
    text = build_moveset_text(pokemon_name, config)
    score = score_moveset_with_gpt(text)
    print(text + "\n")
    print(f"GPT score: {score}")
    return text, score

# -------------------------------
# CONFIGURATION AND DATA LOADING
# -------------------------------

# Define file paths (adjust these paths as needed)
DATA_DIR = Path("data")
SMOGON_FILE    = DATA_DIR / "smogonMovesets.json"
POKEMON_FILE   = DATA_DIR / "pokemon.json"
MOVES_FILE     = DATA_DIR / "moves.json"
ITEMS_FILE     = DATA_DIR / "items.json"
ABILITIES_FILE = DATA_DIR / "abilities.json"
ALIAS_FILE     = DATA_DIR / "aliases.json"

# The keys your GPT scorer returns, in a fixed order:
SCORE_KEYS = ['Nature','Evs','Ivs','Moves','Ability','Item','Tera_Type','Overall']

# For candidate Tera types, we restrict to the Pokémon’s primary type.
# Nature modifiers as provided.
natureModifiers = {
    "Adamant": {"Attack": 1.1, "SpecialAttack": 0.9},
    "Bold": {"Defense": 1.1, "Attack": 0.9},
    "Brave": {"Attack": 1.1, "Speed": 0.9},
    "Calm": {"SpecialDefense": 1.1, "Attack": 0.9},
    "Careful": {"SpecialDefense": 1.1, "SpecialAttack": 0.9},
    "Gentle": {"SpecialDefense": 1.1, "Defense": 0.9},
    "Hasty": {"Speed": 1.1, "Defense": 0.9},
    "Impish": {"Defense": 1.1, "SpecialAttack": 0.9},
    "Jolly": {"Speed": 1.1, "SpecialAttack": 0.9},
    "Lax": {"Defense": 1.1, "SpecialDefense": 0.9},
    "Lonely": {"Attack": 1.1, "Defense": 0.9},
    "Mild": {"SpecialAttack": 1.1, "Defense": 0.9},
    "Modest": {"SpecialAttack": 1.1, "Attack": 0.9},
    "Naive": {"Speed": 1.1, "SpecialDefense": 0.9},
    "Naughty": {"Attack": 1.1, "SpecialDefense": 0.9},
    "Quiet": {"SpecialAttack": 1.1, "Speed": 0.9},
    "Rash": {"SpecialAttack": 1.1, "SpecialDefense": 0.9},
    "Relaxed": {"Defense": 1.1, "Speed": 0.9},
    "Sassy": {"SpecialDefense": 1.1, "Speed": 0.9},
    "Timid": {"Speed": 1.1, "Attack": 0.9}
}
neutral_natures = {"Bashful", "Docile", "Hardy", "Quirky", "Serious"}
candidate_natures = [n for n in natureModifiers.keys() if n not in neutral_natures] + ["Neutral"]

# Candidate IV options: For Attack, SpecialAttack, and Speed, allow either 0 or 31;
# For other stats, fixed at 31.
IV_OPTIONS = {
    "HP": [31],
    "Attack": [0, 31],
    "Defense": [31],
    "SpecialAttack": [0, 31],
    "SpecialDefense": [31],
    "Speed": [0, 31]
}

# EV and IV default for training
LEVEL = 100

# -------------------------------
# DATA LOADING AND LOOKUP BUILDERS
# -------------------------------

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Build lookup from list or dict keyed by lowercase name
def build_lookup(data, key="name"):
    lookup = {}
    if isinstance(data, list):
        for entry in data:
            lookup[entry[key].lower()] = entry
    elif isinstance(data, dict):
        for _, entry in data.items():
            if "name" in entry:
                lookup[entry["name"].lower()] = entry
    return lookup

# Load datasets
smogon_data    = load_json(SMOGON_FILE)
pokemon_data   = load_json(POKEMON_FILE)
moves_data     = load_json(MOVES_FILE)
items_data     = load_json(ITEMS_FILE)
abilities_data = load_json(ABILITIES_FILE)
# Load alias map (Smogon name -> canonical pokemon.json name)
try:
    alias_map = load_json(ALIAS_FILE)
except FileNotFoundError:
    alias_map = {}

def _slugify_move(name: str) -> str:
    # lower‑case, spaces → hyphens
    return name.lower().replace(" ", "-")

GLOBAL_SMOGON_MOVES = set()
for entry in smogon_data:
    for mv in entry.get("moves", []):
        GLOBAL_SMOGON_MOVES.add(_slugify_move(mv))

# Build base lookups
pokemon_lookup   = build_lookup(pokemon_data, key="name")
moves_lookup     = {m.get("name","").lower(): m for m in moves_data.values()}
abilities_lookup = build_lookup(abilities_data, key="name")

# Inject aliases into pokemon_lookup
def _inject_aliases():
    for smogon_name, canonical in alias_map.items():
        if not canonical:
            continue
        alias_key = smogon_name.lower()
        canon_key = canonical.lower()
        entry = pokemon_lookup.get(canon_key)
        if entry:
            pokemon_lookup[alias_key] = entry
        else:
            print(f"Warning: alias value '{canonical}' not found in pokemon dataset")
_inject_aliases()

# For candidate items from Smogon movesets
def get_candidate_items_for_pokemon(pokemon_name):
    items = set()
    lookup_key = pokemon_name.lower()
    for ms in smogon_data:
        ms_key = ms.get("pokemonName", "").lower()
        if ms_key == lookup_key:
            if item := ms.get("item"):
                items.add(item)
    return list(items)

# -------------------------------
# STAT CALCULATION FUNCTIONS (with variable IVs)
# -------------------------------

def compute_final_stats_variable_iv(base_stats, ev_spread, iv_config, nature, level=LEVEL):
    stats = {}
    for stat in ['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed']:
        base = int(base_stats.get(stat, 0))
        ev = ev_spread.get(stat, 0)
        iv = iv_config.get(stat, 31)
        if stat == 'HP':
            stats[stat] = math.floor(((2 * base + iv + math.floor(ev / 4)) * level) / 100) + level + 10
        else:
            modifier = 1.0 if nature == 'Neutral' else natureModifiers.get(nature, {}).get(stat, 1.0)
            stat_val = math.floor(((2 * base + iv + math.floor(ev / 4)) * level) / 100) + 5
            stats[stat] = math.floor(stat_val * modifier)
    return stats


def compute_BST(stats):
    return sum(stats.values())

# -------------------------------
# MOVES: DAMAGE SCORING (BALANCED APPROACH)
# -------------------------------

def compute_damage_score(move):
    move_obj = moves_lookup.get(move.lower())
    if move_obj is None:
        return 0
    base_power = move_obj.get("basePower", 0)
    acc = move_obj.get("accuracy", 100)
    if acc is True:
        acc = 100
    category = move_obj.get("category", "").lower()
    if category == "status":
        return 10
    return base_power * (acc / 100.0)

# -------------------------------
# FEATURE EXTRACTION FOR TRAINING
# -------------------------------

def extract_features_balanced(moveset, pokemon_info):
    features = {}
    # Base stats from Pokémon info.
    for stat in ['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed']:
        features[stat] = int(pokemon_info.get("stats", {}).get(stat, 0))
    features["BST"] = sum(features[s] for s in ['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed'])
    # EVs and IVs from moveset.
    for stat in ['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed']:
        ev = moveset.get("evs", {}).get(stat, 0)
        iv = moveset.get("ivs", {}).get(stat, 31)
        features[f"EV_{stat}"] = ev / 252.0
        features[f"IV_{stat}"] = iv / 31.0
    # Categorical features from moveset.
    features["nature"] = moveset.get("nature", "Neutral")
    features["teraType"] = moveset.get("teraType", "Unknown")
    features["item"] = moveset.get("item", "None")
    features["ability"] = moveset.get("ability", "Unknown")
    features["tier"] = moveset.get("pokemonTier", "Unrated")
    # Move features: average damage score and base power.
    damage_scores = [compute_damage_score(m) for m in moveset.get("moves", [])]
    features["avg_damage_score"] = np.mean(damage_scores) if damage_scores else 0.0
    base_powers = [moves_lookup.get(m.lower(), {}).get("basePower", 0) for m in moveset.get("moves", [])]
    features["avg_base_power"] = np.mean(base_powers) if base_powers else 0.0
    # Training target: sum of damage score and BST component.
    features["heuristic_score"] = features["avg_damage_score"] + (features["BST"] / 300.0)
    return features

# -------------------------------
# MODEL TRAINING WITH FULL DATA (NO SPLIT)
# -------------------------------

def train_surrogate_network(epochs=20, batch_size=64, lr=1e-3):
    """
    1) Build X/y from smogon_data → features & heuristic_score  
    2) One-hot & scale  
    3) Train an MLP whose output dim = len(SCORE_KEYS)  
       - Pre-train only supervising the 'Overall' head  
    Returns: model, scaler, encoder, numeric_cols, categorical_cols
    """
    # --- 1) build feature dicts & targets ---
    feature_dicts, targets, missing = [], [], set()
    for ms in smogon_data:
        name = ms.get("pokemonName","")
        pinfo = pokemon_lookup.get(name.lower())
        if pinfo is None:
            missing.add(name)
            continue
        feats = extract_features_balanced(ms, pinfo)
        feature_dicts.append(feats)
        targets.append(feats["heuristic_score"])
    if missing:
        with open(DATA_DIR/"missing_pokemon_names.txt","w") as f:
            f.write("\n".join(sorted(missing)))
        print(f"Exported {len(missing)} missing names for alias mapping")

    df = pd.DataFrame(feature_dicts)
    y = df.pop("heuristic_score")

    # --- 2) split numeric vs categorical & fit preprocessors ---
    numeric_cols = [
        'HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed',
        'BST','avg_damage_score','avg_base_power'
    ] + [f"EV_{s}" for s in ['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed']] \
      + [f"IV_{s}" for s in ['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed']]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(df[categorical_cols])
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[numeric_cols])

    X_full = np.hstack([X_num, X_cat]).astype(np.float32)
    y_full = np.array(targets, dtype=np.float32).reshape(-1,1)

    # --- 3) DataLoader ---
    ds = TensorDataset(torch.from_numpy(X_full), torch.from_numpy(y_full))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # --- 4) define MLP surrogate ---
    input_dim  = X_full.shape[1]
    output_dim = len(SCORE_KEYS)
    model = nn.Sequential(
        nn.Linear(input_dim, 256), nn.ReLU(),
        nn.Linear(256, 128),       nn.ReLU(),
        nn.Linear(128, output_dim)
    )

    overall_idx = SCORE_KEYS.index("Overall")
    optimiser   = optim.Adam(model.parameters(), lr=lr)
    loss_fn     = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        for xb, yb in loader:
            preds = model(xb)
            loss  = loss_fn(preds[:, overall_idx:overall_idx+1], yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * xb.size(0)
        #print(f"[Pre-train] Epoch {epoch}/{epochs} — loss: {epoch_loss/len(ds):.4f}")

    return model, scaler, encoder, numeric_cols, categorical_cols

# Execute training
model, scaler, encoder, numeric_cols, categorical_cols = train_surrogate_network()

# -------------------------------
# CANDIDATE SPACE EXTRACTION FROM SMOGON DATA
# -------------------------------

def get_candidates_from_smogon(pokemon_name):
    """
    For a given Pokémon name, extract the unique EV spreads, candidate items, and union of moves
    from the Smogon dataset.
    """
    ev_spreads = []
    items = set()
    moves = set()
    for ms in smogon_data:
        if ms.get("pokemonName", "").lower() == pokemon_name.lower():
            ev = ms.get("evs", {})
            ev_tuple = tuple(sorted(ev.items()))
            if ev_tuple not in [tuple(sorted(e.items())) for e in ev_spreads]:
                ev_spreads.append(ev)
            item = ms.get("item")
            if item:
                items.add(item)
            for m in ms.get("moves", []):
                moves.add(m)
    return ev_spreads, list(items), list(moves)

# -------------------------------
# FEATURE EXTRACTION FOR A CANDIDATE CONFIGURATION (with variable IVs)
# -------------------------------

def build_configuration_features_custom(pokemon_info, ev_spread, iv_config, nature, ability, item, tera, move_set):
    """
    Build a feature dictionary for a candidate configuration using:
      - Final stats (computed with variable IVs, EV spread and nature),
      - BST,
      - Average damage score and average base power of the moveset,
      - Normalized EV and IV features,
      - And categorical features (nature, tera type, item, ability, tier).
    """
    base_stats = pokemon_info.get("stats", {})
    final_stats = compute_final_stats_variable_iv(base_stats, ev_spread, iv_config, nature)
    bst = compute_BST(final_stats)
    damage_scores = [compute_damage_score(m) for m in move_set]
    avg_damage_score = np.mean(damage_scores) if damage_scores else 0.0
    base_powers = [moves_lookup.get(m.lower(), {}).get("basePower", 0) for m in move_set]
    avg_base_power = np.mean(base_powers) if base_powers else 0.0
    ev_features = {f"EV_{stat}": ev_spread.get(stat, 0) / 252.0 for stat in ['HP', 'Attack', 'Defense', 'SpecialAttack', 'SpecialDefense', 'Speed']}
    iv_features = {f"IV_{stat}": iv_config.get(stat, 31) / 31.0 for stat in ['HP', 'Attack', 'Defense', 'SpecialAttack', 'SpecialDefense', 'Speed']}
    base_stats_num = {stat: int(base_stats.get(stat, 0)) for stat in ['HP', 'Attack', 'Defense', 'SpecialAttack', 'SpecialDefense', 'Speed']}
    
    features = {}
    features.update(base_stats_num)
    features["BST"] = bst
    features["avg_damage_score"] = avg_damage_score
    features["avg_base_power"] = avg_base_power
    features.update(ev_features)
    features.update(iv_features)
    # Categorical features.
    features["tier"] = pokemon_info.get("tier", "Unrated")
    features["nature"] = nature
    features["teraType"] = tera
    features["item"] = item
    features["ability"] = ability
    return features

# -------------------------------
# CANDIDATE CONFIGURATION SEARCH
# -------------------------------

def find_optimal_configuration_custom(pokemon_name, batch_size=500):
    """
    For a given Pokémon, extract candidates and evaluate all configurations
    in parallel batches of `batch_size` move‑sets.
    """
    pokemon_info = pokemon_lookup.get(pokemon_name.lower())
    if not pokemon_info:
        print(f"Pokémon '{pokemon_name}' not found in the dataset.")
        return None

    candidate_abilities = pokemon_info.get("abilities", [])
    if not candidate_abilities:
        print(f"No abilities found for {pokemon_name}.")
        return None

    ev_spreads, candidate_items, smogon_specific = get_candidates_from_smogon(pokemon_name)

    # 1) Build the base pool of legal, globally‑Smogon moves
    legal_learnable = set(pokemon_info.get("learnable_moves", []))
    pool = [m for m in legal_learnable if m in GLOBAL_SMOGON_MOVES]

    # 2) Guarantee that Pokémon’s own Smogon moves are included (if legal)
    guaranteed = [m for m in smogon_specific if m in legal_learnable]

    # 3) The rest of our pool—exclude guaranteed to avoid duplicates
    fallback_pool = [m for m in pool if m not in guaranteed]

    # 4) Dynamic fallback logic:
    stats = pokemon_info.get("stats", {})
    atk, spa, spe = map(int, (stats.get("Attack",0), stats.get("SpecialAttack",0), stats.get("Speed",0)))
    role_cat = None
    if abs(atk - spa) >= 10:
        role_cat = "physical" if atk > spa else "special"

    def mo(m): return moves_lookup.get(m.lower(), {})
    def dps(m):
        o = mo(m)
        bp = o.get("basePower",0)
        acc = o.get("accuracy",100)
        if acc is True: acc = 100
        return bp*(acc/100.0) if o.get("category","").lower()!="status" else 0

    # a) Split fallback_pool
    status_moves = [m for m in fallback_pool if mo(m).get("category","").lower()=="status"]
    dmg_moves    = [m for m in fallback_pool if m not in status_moves]

    selected = set(status_moves)   # include all status utilities

    # b) Cover own types: best DPS + best accuracy
    types_ = [pokemon_info["types"]["primary"]]
    sec = pokemon_info["types"].get("secondary")
    if sec and sec!="N/A": types_.append(sec)
    for t in types_:
        cands = [m for m in fallback_pool if mo(m).get("type")==t]
        if role_cat:
            cands = [m for m in cands if mo(m).get("category","").lower()==role_cat]
        if not cands: continue
        selected.add(max(cands, key=dps))
        best_acc = max(
            cands,
            key=lambda m: (
                (mo(m).get("accuracy",100) if mo(m).get("accuracy") is not True else 100),
                mo(m).get("basePower",0)
            )
        )
        selected.add(best_acc)

    # c) Coverage: one top DPS for up to 4 other types
    other = {}
    for m in fallback_pool:
        t = mo(m).get("type")
        if t not in types_:
            other.setdefault(t, []).append(m)
    cov = []
    for t,mvs in other.items():
        if role_cat:
            mvs = [m for m in mvs if mo(m).get("category","").lower()==role_cat]
        if mvs:
            cov.append(max(mvs, key=dps))
    for m in sorted(cov, key=dps, reverse=True)[:4]:
        selected.add(m)

    # 5) Merge guaranteed + dynamic fallback, preserving order
    candidate_moves = guaranteed + [m for m in selected if m not in guaranteed]

    if not ev_spreads:
        ev_spreads = [{"HP":0, "Attack":0, "Defense":0,
                       "SpecialAttack":0, "SpecialDefense":0, "Speed":0}]
    if not candidate_items:
        candidate_items = ["None"]
    if not candidate_moves:
        print(f"No candidate moves found for {pokemon_name}.")
        return None

    candidate_movesets = list(itertools.combinations(candidate_moves, 4))
    if not candidate_movesets:
        print(f"Not enough moves to form a moveset for {pokemon_name}.")
        return None

    # Build IV candidates exactly as before
    iv_candidates = []
    for atk in IV_OPTIONS["Attack"]:
        for spa in IV_OPTIONS["SpecialAttack"]:
            for spe in IV_OPTIONS["Speed"]:
                iv_candidates.append({
                    "HP":31, "Attack":atk, "Defense":31,
                    "SpecialAttack":spa, "SpecialDefense":31, "Speed":spe
                })

    candidate_natures_local = candidate_natures
    primary_type = pokemon_info.get("types", {}).get("primary", "Unknown")
    candidate_tera_types = [primary_type]

    # Compute total for your original print
    total_candidates = (
        len(candidate_abilities)
      * len(candidate_items)
      * len(candidate_natures_local)
      * len(candidate_tera_types)
      * len(ev_spreads)
      * len(iv_candidates)
      * len(candidate_movesets)
    )
    print(f"Evaluating {total_candidates} candidate configurations for {pokemon_name}...")

    # Build all “meta” combos except movesets
    metas = list(product(
        candidate_abilities,
        candidate_items,
        candidate_natures_local,
        candidate_tera_types,
        ev_spreads,
        iv_candidates
    ))

    # Split move‑sets into batches
    batches = [
        candidate_movesets[i : i + batch_size]
        for i in range(0, len(candidate_movesets), batch_size)
    ]

    best_score    = float("-inf")
    best_config   = None
    best_features = None

    # Parallel over each batch for every meta
    for meta in metas:
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(evaluate_configuration_batch)(
                meta, batch,
                pokemon_info,
                numeric_cols, categorical_cols,
                scaler, encoder,
                model
            )
            for batch in batches
        )
        # each result is (score, config, features)
        score, config, features = max(results, key=lambda x: x[0])
        if score > best_score:
            best_score    = score
            best_config   = config
            best_features = features

    # Return exactly as your original main() expects
    return best_config, best_score, best_features, pokemon_info

# -------------------------------
# RL FINE-TUNING
# -------------------------------

def run_reinforcement_learning(num_rounds, model, scaler, encoder,
                                numeric_cols, categorical_cols):
    """
    For each round:
      - generate best moveset
      - score via GPT ⇒ dict of 8 floats
      - aggregate all (X,y_dict)
      - perform one gradient step on full batch
    """
    optimiser = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn   = nn.MSELoss()

    for r in range(1, num_rounds+1):
        print(f"=== RL Round {r}/{num_rounds} ===")
        feats_list, score_dicts = [], []

        for name, info in pokemon_lookup.items():
            res = find_optimal_configuration_custom(name)
            if not res:
                continue
            config, _, _, _ = res
            text = build_moveset_text(name, config)
            reward = score_moveset_with_gpt(text)

            feats_list.append(
                build_configuration_features_custom(
                    info,
                    config["EV_spread"],
                    config["IV_config"],
                    config["nature"],
                    config["ability"],
                    config["item"],
                    config["teraType"],
                    config["move_set"]
                )
            )
            score_dicts.append(reward)

        # assemble batch
        df_rl = pd.DataFrame(feats_list)
        y_rl  = pd.DataFrame(score_dicts)[SCORE_KEYS].values.astype(np.float32)
        X_num = scaler.transform(df_rl[numeric_cols])
        X_cat = encoder.transform(df_rl[categorical_cols])
        X_rl  = np.hstack([X_num, X_cat]).astype(np.float32)

        # single gradient step on all 8 targets
        model.train()
        xb = torch.from_numpy(X_rl)
        yb = torch.from_numpy(y_rl)
        preds = model(xb)
        loss  = loss_fn(preds, yb)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        print(f"[RL] Round {r} — batch loss: {loss.item():.4f}")

    # save final surrogate + preprocessors
    torch.save({
        'state_dict': model.state_dict(),
        'scaler': scaler,
        'encoder': encoder,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }, "moveset_rl_model.pt")
    print("Saved RL-trained PyTorch surrogate → moveset_rl_model.pt")

# -------------------------------
# MAIN EXECUTION
# -------------------------------

def build_moveset_text(pokemon_name, config):
    """Extracted from your old main(): format a config dict into a moveset string."""
    lines = [
        f"{pokemon_name} @ {config['item']}",
        f"Ability: {config['ability']}",
        f"Nature: {config['nature']} Nature"
    ]
    evs = [f"{ev} {stat}" for stat, ev in config["EV_spread"].items() if ev > 0]
    if evs:
        lines.append("EVs: " + " / ".join(evs))
    ivs = [f"{iv} {stat}" for stat, iv in config["IV_config"].items() if iv < 31]
    if ivs:
        lines.append("IVs: " + " / ".join(ivs))
    lines += [
        f"Tera Type: {config['teraType']}",
        "– Moves –"
    ] + [f"- {m}" for m in config["move_set"]]
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(
        description="Pokémon moveset RL trainer & generator"
    )
    parser.add_argument(
        "--rounds", "-r", type=int, default=0,
        help="Run N reinforcement‑learning rounds (uses GPT scores)"
    )
    parser.add_argument(
        "--pokemon", "-p", type=str, default="",
        help="Generate & score a moveset for a single Pokémon"
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="After RL, save model + preprocessors to moveset_rl_model.pkl"
    )
    parser.add_argument(
        "--load-model", "-l", type=str, default="",
        help="Path to an existing RL model pickle to resume training"
    )
    parser.add_argument("--export-json", action="store_true",
                        help="Generate movesets for all Pokémon and export to all_movesets.json")
    args = parser.parse_args()

    # 1) Pre-train surrogate on Smogon
    model, scaler, encoder, numeric_cols, categorical_cols = train_surrogate_network()

    # 2) Optionally load an existing PyTorch surrogate
    if args.load_model:
        ckpt = torch.load(args.load_model, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        scaler         = ckpt['scaler']
        encoder        = ckpt['encoder']
        numeric_cols   = ckpt['numeric_cols']
        categorical_cols = ckpt['categorical_cols']
        print(f"Loaded surrogate from '{args.load_model}'")

    if args.export_json:
        # load Paldea-dex ordering
        with open(POKEMON_FILE, "r", encoding="utf-8") as f:
            paldea_dex = [p["name"] for p in json.load(f)]

        all_results = []
        for nm in paldea_dex:
            res = find_optimal_configuration_custom(nm)
            if not res:
                continue
            config, _, _, pinfo = res
            all_results.append({
                "pokemonName": pinfo["name"],
                "evs":         config["EV_spread"],
                "ivs":         config["IV_config"],
                "nature":      config["nature"],
                "item":        config["item"],
                "ability":     config["ability"],
                "teraType":    config["teraType"],
                "moves":       config["move_set"]
            })

        # write out JSON
        with open("all_movesets.json", "w", encoding="utf-8") as out:
            json.dump(all_results, out, indent=4)
        print("✅ Generated movesets for all Pokémon and saved to all_movesets.json")
        return

    if args.rounds > 0:
        run_reinforcement_learning(
            args.rounds,
            model, scaler, encoder,
            numeric_cols, categorical_cols
        )
        if args.save_model:
            torch.save({
                'state_dict': model.state_dict(),
                'scaler': scaler,
                'encoder': encoder,
                'numeric_cols': numeric_cols,
                'categorical_cols': categorical_cols
            }, args.load_model or "moveset_rl_model.pt", _use_new_zipfile_serialization=False)
            print("Saved RL-trained surrogate to disk")

    if args.pokemon:
        # Load your PyTorch surrogate checkpoint
        ckpt_path = args.load_model or "moveset_rl_model.pt"
        ckpt = torch.load(ckpt_path, weights_only=False)
        # Restore model + preprocessors
        model.load_state_dict(ckpt['state_dict'])
        scaler           = ckpt['scaler']
        encoder          = ckpt['encoder']
        numeric_cols     = ckpt['numeric_cols']
        categorical_cols = ckpt['categorical_cols']
        # Now generate & score
        generate_moveset(args.pokemon)

    if args.rounds == 0 and not args.pokemon:
        # interactive fall‑back
        name = input("Enter a Pokémon name: ").strip()
        generate_moveset(name)

if __name__ == "__main__":
    main()