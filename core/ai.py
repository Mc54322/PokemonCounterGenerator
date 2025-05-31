import json
import math
from itertools import product, combinations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import argparse
from apichecker import score_moveset_with_gpt
from tqdm import tqdm

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

# -------------------------------
# MULTI-TASK SURROGATE NETWORK
# -------------------------------
class MultiTaskSurrogate(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        # heads sized by global vocab lengths
        self.head_nature   = nn.Linear(128, len(candidate_natures))
        self.head_evs      = nn.Linear(128, 6)
        self.head_ivs      = nn.Linear(128, 6)
        self.head_moves    = nn.Linear(128, len(GLOBAL_SMOGON_MOVES))
        self.head_ability  = nn.Linear(128, len(candidate_abilities))
        self.head_item     = nn.Linear(128, len(candidate_items))
        self.head_tera     = nn.Linear(128, len(candidate_tera_types))
        self.head_overall  = nn.Linear(128, 1)

    def forward(self, x):
        z = self.trunk(x)
        return {
            'nature':  self.head_nature(z),
            'evs':     self.head_evs(z),
            'ivs':     self.head_ivs(z),
            'moves':   self.head_moves(z),
            'ability': self.head_ability(z),
            'item':    self.head_item(z),
            'tera':    self.head_tera(z),
            'overall': self.head_overall(z)
        }

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
candidate_natures = [n for n in natureModifiers.keys() if n not in neutral_natures] + ["Quirky"]

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

candidate_abilities = sorted(
    entry.get('name')
    for entry in abilities_lookup.values()
    if isinstance(entry, dict) and 'name' in entry
)
# Items: all items seen in Smogon movesets
candidate_items = sorted({
    ms.get('item')
    for ms in smogon_data
    if ms.get('item')
})
# Tera types: all primary (and secondary, if present) types from your Pokémon data
candidate_tera_types = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison",
    "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark",
    "Steel", "Fairy", "Stellar"
]
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
# FEATURE BUILDER WITH INTERACTIONS
# -------------------------------
def build_configuration_features_interactions(pinfo, ev_spread, iv_config,
                                              nature, ability, item, tera,
                                              move_set,
                                              numeric_cols, categorical_cols,
                                              encoder, scaler):
    """
    Build a feature vector with numeric, categorical, and interaction terms,
    ALWAYS including a fixed-size interaction block so the output length
    never changes.
    """
    # 1) Build the base feature dict & DataFrame.
    feats = build_configuration_features_custom(
        pinfo, ev_spread, iv_config,
        nature, ability, item, tera, move_set
    )
    df = pd.DataFrame([feats])

    # 2) Split into numeric & categorical (reindexed to full training set).
    df_num = df[numeric_cols]
    df_cat = df.reindex(columns=categorical_cols, fill_value=None)

    # 3) Transform.
    X_num = scaler.transform(df_num)
    X_cat = encoder.transform(df_cat)
    X = np.hstack([X_num, X_cat])

    # 4) Build interaction terms for EVERY nature,
    #    padding with zeros if needed.
    #    We know every “boosting” nature has exactly 2 stats → 4 terms.
    #    So we’ll always make 4 slots.
    interactions = []
    # real interactions when available:
    for stat, factor in natureModifiers.get(nature, {}).items():
        ev_i  = numeric_cols.index(f"EV_{stat}")
        iv_i  = numeric_cols.index(f"IV_{stat}")
        cat_names = encoder.get_feature_names_out(categorical_cols)
        feat = f"nature_{nature}"
        if feat in cat_names:
            nat_i = list(cat_names).index(feat)
            interactions.append(X_num[0, ev_i] * X_cat[0, nat_i])
            interactions.append(X_num[0, iv_i] * X_cat[0, nat_i])
    # pad to 4 entries total:
    while len(interactions) < 4:
        interactions.append(0.0)

    X_cross = np.array(interactions).reshape(1, -1)
    X = np.hstack([X, X_cross])

    return X

# -------------------------------
# RULE-BASED PRUNING FUNCTION
# -------------------------------
def rule_based_filter(nature, ev_spread, iv_config, move_set, pinfo):
    # hard prune: nature boosts Attack requires Attack EV >= 100
    boosts = natureModifiers.get(nature, {})
    if 'Attack' in boosts and ev_spread.get('Attack',0) < 100:
        return False
    # soft prune: no mixed physical/special on pure roles
    atk, spa = pinfo['stats']['Attack'], pinfo['stats']['SpecialAttack']
    role = 'physical' if atk > spa else 'special'
    for m in move_set:
        cat = moves_lookup[m.lower()]['category'].lower()
        if cat =='status': continue
        if role=='physical' and cat=='special': return False
        if role=='special'  and cat=='physical': return False
    return True

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

def train_multitask_surrogate(smogon_data, pokemon_lookup, epochs=20, batch_size=64, lr=1e-3):
    # 1) extract feature dicts & raw labels
    feature_dicts, labels = [], []
    for ms in smogon_data:
        name = ms['pokemonName'].lower()
        pinfo = pokemon_lookup.get(name)
        if not pinfo:
            continue
        feats = extract_features_balanced(ms, pinfo)
        move_slugs = { _slugify_move(m) for m in ms.get('moves', []) }
        # build label dict
        y = {
            'nature':  candidate_natures.index(ms.get('nature', 'Neutral')),
            'evs':     np.array([ms['evs'].get(s,0)/252 for s in ['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed']]),
            'ivs':     np.array([ms['ivs'].get(s,31)/31  for s in ['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed']]),
            'moves':   np.array([1 if slug in move_slugs else 0 for slug in GLOBAL_SMOGON_MOVES]),
            'ability': candidate_abilities.index(ms.get('ability','Unknown')),
            'item':    candidate_items.index(ms.get('item','None')),
            'tera':    candidate_tera_types.index(ms.get('teraType') or pinfo['types']['primary']),
            'overall': feats['heuristic_score']
        }
        feature_dicts.append(feats)
        labels.append(y)

    # 2) dataframe + preprocessors
    df = pd.DataFrame(feature_dicts)
    numeric_cols = [
        'HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed',
        'BST','avg_damage_score','avg_base_power'
    ] + [f"EV_{s}" for s in ['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed']]
    numeric_cols += [f"IV_{s}" for s in ['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed']]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = encoder.fit_transform(df[categorical_cols])
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[numeric_cols])
    X_full = np.hstack([X_num, X_cat]).astype(np.float32)

    # 3) labels → tensors
    y_nature = torch.tensor([y['nature'] for y in labels], dtype=torch.long)
    y_evs    = torch.tensor(np.stack([y['evs'] for y in labels]), dtype=torch.float32)
    y_ivs    = torch.tensor(np.stack([y['ivs'] for y in labels]), dtype=torch.float32)
    y_moves  = torch.tensor(np.stack([y['moves'] for y in labels]), dtype=torch.float32)
    y_abil   = torch.tensor([y['ability'] for y in labels], dtype=torch.long)
    y_item   = torch.tensor([y['item'] for y in labels], dtype=torch.long)
    y_tera   = torch.tensor([y['tera'] for y in labels], dtype=torch.long)
    y_over   = torch.tensor([y['overall'] for y in labels], dtype=torch.float32).unsqueeze(1)

    # 4) model & optimizer
    model = MultiTaskSurrogate(input_dim=X_full.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5) losses
    loss_ce  = nn.CrossEntropyLoss()
    loss_bce = nn.BCEWithLogitsLoss()
    loss_mse = nn.MSELoss()

    # 6) dataloader
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_full), y_nature, y_evs, y_ivs,
        y_moves, y_abil, y_item, y_tera, y_over
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    # 7) training loop
    for epoch in range(epochs):
        total_loss = 0.
        model.train()
        for xb, yn, ye, yi, ym, ya, yi2, yt, yo in loader:
            preds = model(xb)
            loss = (
                loss_ce(preds['nature'], yn)
              + loss_mse(preds['evs'], ye)
              + loss_mse(preds['ivs'], yi)
              + loss_bce(preds['moves'], ym)
              + loss_ce(preds['ability'], ya)
              + loss_ce(preds['item'], yi2)
              + loss_ce(preds['tera'], yt)
              + loss_mse(preds['overall'], yo)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"[Multitask] Epoch {epoch+1}/{epochs} — loss: {total_loss/len(ds):.4f}")

    return model, scaler, encoder, numeric_cols, categorical_cols

# Execute training
model, scaler, encoder, numeric_cols, categorical_cols = train_multitask_surrogate(smogon_data, pokemon_lookup, epochs=20, batch_size=64, lr=1e-3)

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

def get_candidate_configs(pokemon_name):
    """
    Returns a list of all legal configuration dicts for `pokemon_name`,
    each containing keys:
      'ability', 'item', 'nature', 'teraType',
      'EV_spread', 'IV_config', 'move_set'
    following your Smogon‐style reductions.
    """
    pinfo = pokemon_lookup.get(pokemon_name.lower())
    if not pinfo:
        return []

    # 1) Abilities from the Pokémon database
    candidate_abilities = pinfo.get("abilities", [])
    if not candidate_abilities:
        return []

    # 2) Items & EV spreads from Smogon (fall back to zero‐EV if none)
    ev_spreads, candidate_items, _ = get_candidates_from_smogon(pokemon_name)
    if not ev_spreads:
        ev_spreads = [{
            "HP":0, "Attack":0, "Defense":0,
            "SpecialAttack":0, "SpecialDefense":0, "Speed":0
        }]
    if not candidate_items:
        candidate_items = ["None"]

    # 3) IV configs: HP/Def/SpD fixed at 31; Atk/SpA/Spe ∈ {0,31}
    iv_candidates = []
    for atk in IV_OPTIONS["Attack"]:
        for spa in IV_OPTIONS["SpecialAttack"]:
            for spe in IV_OPTIONS["Speed"]:
                iv_candidates.append({
                    "HP":31, "Attack":atk, "Defense":31,
                    "SpecialAttack":spa, "SpecialDefense":31, "Speed":spe
                })

    # 4) Natures & Tera Types
    candidate_natures_local = candidate_natures
    primary = pinfo.get("types", {}).get("primary", "Unknown")
    candidate_tera_types   = [primary]

    # 5) Build reduced move pool
    learnable = set(pinfo.get("learnable_moves", []))
    common_moves = {m for m in learnable if m in GLOBAL_SMOGON_MOVES}

    # helper to map slug → moves_lookup entry
    def lookup_move(slug):
        key = slug.replace("-", " ").lower()
        return moves_lookup.get(key)

    # 5a) All status utilities
    status_moves = [
        m for m in common_moves
        if (mv := lookup_move(m)) and mv.get("category","").lower() == "status"
    ]

    # 5b) Determine role(s)
    atk = int(pinfo["stats"].get("Attack",0))
    spa = int(pinfo["stats"].get("SpecialAttack",0))
    if abs(atk - spa) <= 5:
        roles = ["physical", "special"]
    elif atk > spa:
        roles = ["physical"]
    else:
        roles = ["special"]

    # 5c) Per‐type best DPS & best accuracy
    def move_dps(slug):
        mv = lookup_move(slug)
        if not mv: 
            return 0
        bp = mv.get("basePower",0)
        acc = mv.get("accuracy",100)
        acc = 100 if acc is True else acc
        return bp * (acc/100.0) if mv.get("category","").lower()!="status" else 0

    def move_acc_key(slug):
        mv = lookup_move(slug)
        if not mv:
            return (0, 0)
        acc = mv.get("accuracy",100)
        acc = 100 if acc is True else acc
        return (acc, mv.get("basePower",0))

    types_ = [primary]
    sec = pinfo["types"].get("secondary")
    if sec and sec!="N/A":
        types_.append(sec)

    type_specific = set()
    for t in types_:
        for role in roles:
            cand = [
                m for m in common_moves
                if (mv := lookup_move(m))
                and mv.get("type")==t
                and mv.get("category","").lower()==role
            ]
            if not cand:
                continue
            type_specific.add(max(cand, key=move_dps))
            type_specific.add(max(cand, key=move_acc_key))

    # 5d) From other types, top3 DPS + top3 Accuracy
    other = {
        m for m in common_moves
        if (mv := lookup_move(m))
        and mv.get("type") not in types_
        and mv.get("category","").lower() in roles
    }
    other_dps = sorted(other, key=move_dps, reverse=True)[:3]
    other_acc = sorted(other, key=move_acc_key, reverse=True)[:3]

    candidate_moves = set(status_moves) | type_specific | set(other_dps) | set(other_acc)
    if len(candidate_moves) < 4:
        return []

    # 6) All 4‐move combinations
    candidate_movesets = list(combinations(candidate_moves, 4))

    # 7) Assemble full config tuples
    configs = []
    for ability, item, nature, tera, ev_spread, iv_conf in product(
        candidate_abilities,
        candidate_items,
        candidate_natures_local,
        candidate_tera_types,
        ev_spreads,
        iv_candidates
    ):
        for moveset in candidate_movesets:
            configs.append({
                "ability":    ability,
                "item":       item,
                "nature":     nature,
                "teraType":   tera,
                "EV_spread":  ev_spread,
                "IV_config":  iv_conf,
                "move_set":   moveset
            })

    return configs

# -------------------------------
# HIERARCHICAL MODELS & TRAINING
# -------------------------------

class Stage1NatureAbility(nn.Module):
    """
    Predicts nature and ability from base stats.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.head_nature  = nn.Linear(64, len(candidate_natures))
        self.head_ability = nn.Linear(64, len(candidate_abilities))

    def forward(self, x):
        z = self.net(x)
        return {
            'nature':  self.head_nature(z),
            'ability': self.head_ability(z)
        }

def train_stage1(smogon_data, pokemon_lookup, epochs=10, lr=1e-3):
    # Features: base stats (ensuring floats)
    X_list, y_nat, y_abil = [], [], []
    stat_keys = ['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed']
    for ms in smogon_data:
        pinfo = pokemon_lookup.get(ms['pokemonName'].lower())
        if not pinfo:
            continue
        # Safely extract and cast stats to float
        stats_vals = [float(pinfo.get('stats', {}).get(key, 0)) for key in stat_keys]
        # Verify numeric
        if any(not isinstance(v, (int, float)) for v in stats_vals):
            print("Skipping invalid stats for", ms.get('pokemonName'))
            continue
        X_list.append(stats_vals)
        y_nat.append(candidate_natures.index(ms.get('nature','Neutral')))
        y_abil.append(candidate_abilities.index(ms.get('ability','Unknown')))
    if not X_list:
        raise ValueError("No valid training examples found for Stage1.")
    # Convert to tensor
    X = torch.tensor(X_list, dtype=torch.float32)
    yn = torch.tensor(y_nat, dtype=torch.long)
    ya = torch.tensor(y_abil, dtype=torch.long)

    model1 = Stage1NatureAbility(input_dim=X.shape[1])
    opt1 = optim.Adam(model1.parameters(), lr=lr)
    loss_ce = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
        model1.train()
        out1 = model1(X)
        loss = loss_ce(out1['nature'], yn) + loss_ce(out1['ability'], ya)
        opt1.zero_grad()
        loss.backward()
        opt1.step()
        print(f"[Stage1] Epoch {epoch}/{epochs} — loss: {loss.item():.4f}")
    return model1

class Stage2EVIV(nn.Module):
    """
    Predicts EV and IV spreads conditioned on base stats + chosen nature & ability.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.head_evs = nn.Linear(64, 6)
        self.head_ivs = nn.Linear(64, 6)

    def forward(self, x):
        z = self.net(x)
        return {
            'evs': self.head_evs(z),
            'ivs': self.head_ivs(z)
        }

def train_stage2(smogon_data, pokemon_lookup, model1, scaler, encoder,
                 numeric_cols, categorical_cols, epochs=10, lr=1e-4):
    X_feats, y_evs, y_ivs = [], [], []
    for ms in smogon_data:
        pinfo = pokemon_lookup.get(ms['pokemonName'].lower())
        if not pinfo: continue
        # Use same features as interactions builder
        feats = extract_features_balanced(ms, pinfo)
        df = pd.DataFrame([feats])
        X_num = scaler.transform(df[numeric_cols])
        X_cat = encoder.transform(df[categorical_cols])
        X_feats.append(np.hstack([X_num.flatten(), X_cat.flatten()]))
        y_evs.append([ms['evs'].get(s,0)/252 for s in ['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed']])
        y_ivs.append([ms['ivs'].get(s,31)/31 for s in ['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed']])
    X2 = torch.tensor(np.vstack(X_feats), dtype=torch.float32)
    ye = torch.tensor(np.vstack(y_evs), dtype=torch.float32)
    yi = torch.tensor(np.vstack(y_ivs), dtype=torch.float32)

    model2 = Stage2EVIV(input_dim=X2.shape[1])
    opt2 = optim.Adam(model2.parameters(), lr=lr)
    loss_mse = nn.MSELoss()

    for epoch in range(1, epochs+1):
        model2.train()
        out2 = model2(X2)
        loss = loss_mse(out2['evs'], ye) + loss_mse(out2['ivs'], yi)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
        print(f"[Stage2] Epoch {epoch}/{epochs} — loss: {loss.item():.4f}")
    return model2


class Stage3Moveset(nn.Module):
    """
    Predicts a 4-move combination (multi-label) from the full configuration.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.head_moves = nn.Linear(128, len(GLOBAL_SMOGON_MOVES))

    def forward(self, x):
        z = self.net(x)
        return {
            'moves': self.head_moves(z)
        }

def train_stage3(smogon_data, pokemon_lookup, model1, model2,
                 scaler, encoder, numeric_cols, categorical_cols,
                 epochs=10, lr=1e-4):
    X_feats, y_moves = [], []
    stat_keys = ['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed']
    for ms in smogon_data:
        pinfo = pokemon_lookup.get(ms['pokemonName'].lower())
        if not pinfo:
            continue
        # Stage1 prediction: safely parse stats
        stats_vals = [float(pinfo.get('stats', {}).get(k, 0)) for k in stat_keys]
        X1 = torch.tensor([stats_vals], dtype=torch.float32)
        out1 = model1(X1)
        nat_idx = out1['nature'].argmax(dim=1).item()
        abl_idx = out1['ability'].argmax(dim=1).item()
        # Stage2 prediction
        feats = extract_features_balanced(ms, pinfo)
        df = pd.DataFrame([feats])
        X_num = scaler.transform(df[numeric_cols])
        X_cat = encoder.transform(df[categorical_cols])
        # combine
        X2 = torch.tensor(np.hstack([X_num.flatten(), X_cat.flatten()]), dtype=torch.float32).unsqueeze(0)
        out2 = model2(X2)
        ev_pred = out2['evs'].detach().numpy()[0]
        iv_pred = out2['ivs'].detach().numpy()[0]
        # Full feature for Stage3
        X3 = build_configuration_features_interactions(
            pinfo,
            {s: int(val*252) for s,val in zip(['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed'], ev_pred)},
            {s: int(val*31)  for s,val in zip(['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed'], iv_pred)},
            candidate_natures[nat_idx], candidate_abilities[abl_idx],
            candidate_items[0], candidate_tera_types[0],
            ms['moves'], numeric_cols, categorical_cols, encoder, scaler
        )
        X_feats.append(X3.flatten())
        y_moves.append([1 if m in ms['moves'] else 0 for m in GLOBAL_SMOGON_MOVES])
    X3_t = torch.tensor(np.vstack(X_feats), dtype=torch.float32)
    ym = torch.tensor(np.vstack(y_moves), dtype=torch.float32)

    model3 = Stage3Moveset(input_dim=X3_t.shape[1])
    opt3 = optim.Adam(model3.parameters(), lr=lr)
    loss_bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs+1):
        model3.train()
        out3 = model3(X3_t)
        loss = loss_bce(out3['moves'], ym)
        opt3.zero_grad()
        loss.backward()
        opt3.step()
        print(f"[Stage3] Epoch {epoch}/{epochs} — loss: {loss.item():.4f}")
    return model3

# ----------------------------------------------------------------
# Hybrid Cascade Pipeline Function
# ----------------------------------------------------------------

def hybrid_pipeline(pokemon_name,
                    stage1, stage2, stage3,
                    multi_model,
                    scaler, encoder,
                    numeric_cols, categorical_cols):
    """
    Cascades Stage1→Stage2→Stage3 but only feeds the surrogate
    the numeric+categorical features it expects.
    """
    # 1) Stage1: predict nature & ability
    pinfo = pokemon_lookup.get(pokemon_name.lower())
    if not pinfo:
        raise ValueError(f"Pokémon '{pokemon_name}' not found.")
    stat_keys = ['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed']
    stats = [float(pinfo['stats'].get(s, 0)) for s in stat_keys]
    X1 = torch.tensor([stats], dtype=torch.float32)
    out1 = stage1(X1)
    nat_idx = out1['nature'].argmax(dim=1).item()
    abl_idx = out1['ability'].argmax(dim=1).item()

    # 2) Stage2: predict EVs & IVs
    placeholder = {'evs': {s:0 for s in stat_keys},
                   'ivs': {s:31 for s in stat_keys},
                   'moves': []}
    feats0 = extract_features_balanced(placeholder, pinfo)
    df0 = pd.DataFrame([feats0])
    X_num0 = scaler.transform(df0[numeric_cols])
    X_cat0 = encoder.transform(df0[categorical_cols])
    X2 = torch.from_numpy(np.hstack([X_num0, X_cat0]).astype(np.float32))
    out2 = stage2(X2)

    # 3) Stage3: refine features with interactions
    ev_vals = out2['evs'].detach().numpy().flatten()
    iv_vals = out2['ivs'].detach().numpy().flatten()
    ev_dict = {s: int(val*252) for s,val in zip(stat_keys, ev_vals)}
    iv_dict = {s: int(val*31)  for s,val in zip(stat_keys, iv_vals)}
    # build the full scaled+encoded+interaction vector
    X3_np = build_configuration_features_interactions(
        pinfo, ev_dict, iv_dict,
        candidate_natures[nat_idx], candidate_abilities[abl_idx],
        candidate_items[0], candidate_tera_types[0], [],
        numeric_cols, categorical_cols, encoder, scaler
    )

    # 4) Surrogate input: only the numeric+categorical part (drop interactions & any stage-3 logits)
    all_cat_feats = encoder.get_feature_names_out(categorical_cols)
    feat_dim = len(numeric_cols) + len(all_cat_feats)
    X_full = torch.from_numpy(X3_np[:, :feat_dim].astype(np.float32))

    # 5) Final multi-task prediction
    final_out = multi_model(X_full)
    return final_out

# -------------------------------
# RL FINE-TUNING
# -------------------------------

def run_reinforcement_learning(num_rounds,
                               multi_model, scaler, encoder,
                               numeric_cols, categorical_cols):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_model.to(device)
    optimiser = torch.optim.Adam(multi_model.parameters(), lr=1e-4)
    loss_fn   = torch.nn.MSELoss()

    global_moves = list(GLOBAL_SMOGON_MOVES)
    cat_names    = encoder.get_feature_names_out(categorical_cols)
    nature_map   = {
        nm.split("nature_")[1]: i
        for i, nm in enumerate(cat_names)
        if nm.startswith("nature_")
    }
    feat_dim     = len(numeric_cols) + len(cat_names)

    for r in range(1, num_rounds+1):
        print(f"\n=== RL Round {r}/{num_rounds} ===")

        raw_feats       = []
        nat_idx_list    = []
        abil_idx_list   = []
        item_idx_list   = []
        tera_idx_list   = []
        mv_idx_list     = []

        rewards = {k: [] for k in SCORE_KEYS}

        # choose Pokémon this round (uncomment to sample 300)
        all_names = list(pokemon_lookup.keys())
        # names_to_train = random.sample(all_names, min(300, len(all_names)))
        names_to_train = all_names

        for name in tqdm(names_to_train, desc=f"Processing Pokémon"):
            tqdm.write(f"→ {name}")
            pinfo = pokemon_lookup[name]
            configs = get_candidate_configs(name)
            configs = [c for c in configs if c['item'] in candidate_items]
            configs = [c for c in configs if c['ability'] in candidate_abilities]
            if not configs:
                tqdm.write(f" Skipping {name} (no valid configs)")
                continue
            
            def normalize(s: str) -> str:
                return s.replace("’", "'")

            # Heuristic pre‐filter to top K configs
            K = 1000
            if len(configs) > K:
                heur_scores = []
                for cfg in configs:
                    feats_h = build_configuration_features_custom(
                        pinfo,
                        cfg['EV_spread'], cfg['IV_config'],
                        cfg['nature'], cfg['ability'],
                        cfg['item'], cfg['teraType'],
                        cfg['move_set']
                    )
                    heur = feats_h["avg_damage_score"] + feats_h["BST"] / 300.0
                    heur_scores.append(heur)
                top_idxs = np.argsort(heur_scores)[-K:]
                configs = [configs[i] for i in top_idxs]

            # Build batch features for these configs
            raws = []
            head_info = []
            for cfg in configs:
                feats_c = build_configuration_features_custom(
                    pinfo,
                    cfg['EV_spread'], cfg['IV_config'],
                    cfg['nature'], cfg['ability'],
                    cfg['item'], cfg['teraType'],
                    cfg['move_set']
                )
                raws.append(feats_c)
                head_info.append({
                    'nat':  candidate_natures.index(normalize(cfg['nature'])),
                    'abil': candidate_abilities.index(normalize(cfg['ability'])),
                    'item': candidate_items.index(normalize(cfg['item'])),
                    'tera': candidate_tera_types.index(normalize(cfg['teraType'])),
                    'moves': [global_moves.index(m) for m in cfg['move_set']]
                })

            df     = pd.DataFrame(raws)
            X_num  = scaler.transform(df[numeric_cols])
            df_cat = df.reindex(columns=categorical_cols, fill_value=None)
            X_cat  = encoder.transform(df_cat)

            # interactions
            N_cfg = len(raws)
            X_int = np.zeros((N_cfg, 4), dtype=np.float32)
            for i, hi in enumerate(head_info):
                nat = candidate_natures[hi['nat']]
                boosts = natureModifiers.get(nat, {})
                terms = []
                nat_i = nature_map.get(nat)
                if nat_i is not None:
                    for stat in boosts:
                        ev_i = numeric_cols.index(f"EV_{stat}")
                        iv_i = numeric_cols.index(f"IV_{stat}")
                        terms.append(X_num[i, ev_i] * X_cat[i, nat_i])
                        terms.append(X_num[i, iv_i] * X_cat[i, nat_i])
                terms += [0.0] * (4 - len(terms))
                X_int[i] = terms

            X_all   = np.hstack([X_num, X_cat, X_int]).astype(np.float32)[:, :feat_dim]
            xb      = torch.from_numpy(X_all).to(device)

            # surrogate → pick best
            multi_model.eval()
            with torch.no_grad():
                scores = multi_model(xb)['overall'].cpu().numpy().flatten()
            best_i   = int(np.argmax(scores))
            best_cfg = configs[best_i]
            hi       = head_info[best_i]

            # record chosen config
            raw_feats.append(raws[best_i])
            nat_idx_list .append(hi['nat'])
            abil_idx_list.append(hi['abil'])
            item_idx_list.append(hi['item'])
            tera_idx_list.append(hi['tera'])
            mv_idx_list  .append(hi['moves'])

            # one GPT call
            txt = build_moveset_text(name, best_cfg)
            out = score_moveset_with_gpt(txt)
            if isinstance(out, dict):
                for k in SCORE_KEYS:
                    rewards[k].append(out[k])
            else:
                idx = {k:i for i,k in enumerate(SCORE_KEYS)}
                for k in SCORE_KEYS:
                    rewards[k].append(out[idx[k]])

        # if no data, skip
        N = len(raw_feats)
        if N == 0:
            print("No configs to train on this round.")
            continue

        # build training batch
        df     = pd.DataFrame(raw_feats)
        X_num  = scaler.transform(df[numeric_cols])
        df_cat = df.reindex(columns=categorical_cols, fill_value=None)
        X_cat  = encoder.transform(df_cat)

        # rebuild interactions for chosen best
        X_int = np.zeros((N, 4), dtype=np.float32)
        for i, nat_i in enumerate(nat_idx_list):
            nat = candidate_natures[nat_i]
            boosts = natureModifiers.get(nat, {})
            terms = []
            one_i = nature_map.get(nat)
            if one_i is not None:
                for stat in boosts:
                    ev_i = numeric_cols.index(f"EV_{stat}")
                    iv_i = numeric_cols.index(f"IV_{stat}")
                    terms.append(X_num[i, ev_i] * X_cat[i, one_i])
                    terms.append(X_num[i, iv_i] * X_cat[i, one_i])
            terms += [0.0] * (4 - len(terms))
            X_int[i] = terms

        X_all   = np.hstack([X_num, X_cat, X_int]).astype(np.float32)[:, :feat_dim]
        xb      = torch.from_numpy(X_all).to(device)

        # targets for each head
        idx_tensor    = torch.arange(N, device=device)
        mv_idx_tensor = torch.tensor(mv_idx_list, dtype=torch.long, device=device)

        y = {k: torch.tensor(rewards[k], dtype=torch.float32, device=device).unsqueeze(1)
             for k in SCORE_KEYS}

        # multi-head MSE
        multi_model.train()
        outs = multi_model(xb)
        loss = 0.0
        loss += loss_fn(outs['nature'] .gather(1, torch.tensor(nat_idx_list, device=device).unsqueeze(1)), y['Nature'])
        loss += loss_fn(outs['evs']    .mean(1, keepdim=True), y['Evs'])
        loss += loss_fn(outs['ivs']    .mean(1, keepdim=True), y['Ivs'])
        loss += loss_fn(outs['moves']  .gather(1, mv_idx_tensor).mean(1, keepdim=True), y['Moves'])
        loss += loss_fn(outs['ability'].gather(1, torch.tensor(abil_idx_list, device=device).unsqueeze(1)), y['Ability'])
        loss += loss_fn(outs['item']   .gather(1, torch.tensor(item_idx_list, device=device).unsqueeze(1)), y['Item'])
        loss += loss_fn(outs['tera']   .gather(1, torch.tensor(tera_idx_list, device=device).unsqueeze(1)), y['Tera_Type'])
        loss += loss_fn(outs['overall'], y['Overall'])

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        print(f"[RL] Round {r} — loss: {loss.item():.4f}")

    # save final surrogate
    torch.save({
        'state_dict':       multi_model.state_dict(),
        'scaler':           scaler,
        'encoder':          encoder,
        'numeric_cols':     numeric_cols,
        'categorical_cols': categorical_cols
    }, "moveset_rl_model.pt")
    print("Saved RL-trained surrogate → moveset_rl_model.pt")


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
    parser = argparse.ArgumentParser(description="Hybrid Pokémon Moveset Generator")
    parser.add_argument("--rl-rounds", "--rounds", type=int, default=0,
                        help="Number of RL fine-tuning rounds using GPT scores.")
    parser.add_argument("-p", "--pokemon", type=str, help="Name of the Pokémon to generate moveset for.")
    parser.add_argument("--pipeline", choices=["multitask","hierarchy","hybrid"], default="hybrid",
                        help="Which pipeline to run: 'multitask', 'hierarchy', or 'hybrid'.")
    parser.add_argument("--save-model", action="store_true",
                        help="Save the fine-tuned surrogate model to disk after training.")
    parser.add_argument("--load-model", "-l", type=str, default="",
                        help="Path to a saved surrogate model file to load before training or inference.")
    parser.add_argument("--export-json", action="store_true",
                        help="Generate movesets for all Pokémon and export to all_movesets.json")
    args = parser.parse_args()

    # Pre-train or load existing surrogate
    if args.load_model:
        ckpt = torch.load(args.load_model, weights_only=False)
        # infer input_dim from saved trunk weights
        weight = ckpt['state_dict']['trunk.0.weight']
        input_dim = weight.shape[1]
        mt_model = MultiTaskSurrogate(input_dim)
        mt_model.load_state_dict(ckpt['state_dict'])
        mt_scaler = ckpt['scaler']
        mt_encoder = ckpt['encoder']
        num_cols = ckpt['numeric_cols']
        cat_cols = ckpt['categorical_cols']
        print(f"Loaded surrogate from '{args.load_model}'")
    else:
        # train new surrogate
        mt_model, mt_scaler, mt_encoder, num_cols, cat_cols = train_multitask_surrogate(
            smogon_data, pokemon_lookup, epochs=20, batch_size=64, lr=1e-3
        )
        
    # Hierarchical stages
    st1 = train_stage1(smogon_data, pokemon_lookup, epochs=10, lr=1e-3)
    st2 = train_stage2(smogon_data, pokemon_lookup, st1,
                       mt_scaler, mt_encoder, num_cols, cat_cols,
                       epochs=10, lr=1e-4)
    st3 = train_stage3(smogon_data, pokemon_lookup, st1, st2,
                       mt_scaler, mt_encoder, num_cols, cat_cols,
                       epochs=10, lr=1e-4)

    # Optionally run RL fine-tuning
    if args.rl_rounds > 0:
        run_reinforcement_learning(
            args.rl_rounds,
            mt_model, mt_scaler, mt_encoder,
            num_cols, cat_cols
        )

    # If --export-json, run through every Pokémon and dump to JSON
    if args.export_json:
        all_results = []
        for name in tqdm(sorted(pokemon_lookup.keys()), desc="Generating movesets"):
            pinfo = pokemon_lookup.get(name.lower())
            if not pinfo:
                continue

            # 1) build candidate configs
            configs_all = get_candidate_configs(name)
            configs_all = [
                c for c in configs_all
                if c['item'] in candidate_items
                and c['ability'] in candidate_abilities
            ]
            if not configs_all:
                print(f"No legal configs for {name}.")
                continue
    
            # 1) Heuristic pre‐filter to top K
            K = 5000
            if len(configs_all) > K:
                scores_h = []
                for cfg in tqdm(configs_all, desc=f"Heuristic scoring for {name}", total=len(configs_all)):
                    f = build_configuration_features_custom(
                        pokemon_lookup[name],
                        cfg['EV_spread'], cfg['IV_config'],
                        cfg['nature'], cfg['ability'],
                        cfg['item'], cfg['teraType'],
                        cfg['move_set']
                    )
                    scores_h.append(f["avg_damage_score"] + f["BST"] / 300.0)
                idxs = np.argsort(scores_h)[-K:]
                configs = [configs_all[i] for i in idxs]
                print(f"→ Pruned {len(configs_all)} → {len(configs)} configs by heuristic")
            else:
                configs = configs_all
                print(f"→ Evaluating {len(configs)} configs for {name}")
    
            # 2) Batch‐build raw features
            raw_feats = []
            natures   = []
            for cfg in tqdm(configs, desc=f"Featurizing {name}", total=len(configs)):
                raw_feats.append(build_configuration_features_custom(
                    pokemon_lookup[name],
                    cfg['EV_spread'], cfg['IV_config'],
                    cfg['nature'], cfg['ability'],
                    cfg['item'], cfg['teraType'],
                    cfg['move_set']
                ))
                natures.append(cfg['nature'])
    
            # 3) One DataFrame + transforms
            df     = pd.DataFrame(raw_feats)
            X_num  = mt_scaler.transform(df[numeric_cols])
            df_cat = df.reindex(columns=categorical_cols, fill_value=None)
            X_cat  = mt_encoder.transform(df_cat)
    
            # 4) Interaction dims (vectorised, pads to 4)
            cat_names = mt_encoder.get_feature_names_out(categorical_cols)
            nature_map = {
                nm.split("nature_")[1]: i
                for i,nm in enumerate(cat_names) if nm.startswith("nature_")
            }
            N = len(configs)
            X_int = np.zeros((N,4), dtype=np.float32)
            for i, nat in enumerate(natures):
                boosts = natureModifiers.get(nat, {})
                terms  = []
                nat_i  = nature_map.get(nat)
                if nat_i is not None:
                    for stat in boosts:
                        ev_i = numeric_cols.index(f"EV_{stat}")
                        iv_i = numeric_cols.index(f"IV_{stat}")
                        terms.append(X_num[i, ev_i] * X_cat[i, nat_i])
                        terms.append(X_num[i, iv_i] * X_cat[i, nat_i])
                # pad/truncate so always length 4
                terms += [0.0] * (4 - len(terms))
                X_int[i] = terms
    
            # 5) Concatenate + trim to training dim
            feat_dim = len(numeric_cols) + len(cat_names)
            X_all    = np.hstack([X_num, X_cat, X_int]).astype(np.float32)[:, :feat_dim]
    
            # 6) Single surrogate forward & pick best
            device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_tensor = torch.from_numpy(X_all).to(device)
            mt_model.to(device).eval()
            with torch.no_grad():
                overall_scores = mt_model(X_tensor)['overall'].cpu().numpy().flatten()
            best_idx = int(np.argmax(overall_scores))
            best_cfg = configs[best_idx]

            # 4) assemble JSON entry
            all_results.append({
                "pokemonName":   pinfo["name"],
                "evs":           best_cfg["EV_spread"],
                "ivs":           best_cfg["IV_config"],
                "nature":        best_cfg["nature"],
                "item":          best_cfg["item"],
                "ability":       best_cfg["ability"],
                "teraType":      best_cfg["teraType"],
                "moves":         best_cfg["move_set"],
            })

        # write out
        with open("all_movesets.json", "w") as f:
            json.dump(all_results, f, indent=4)

        print("✅ Generated movesets for all Pokémon and saved to all_movesets.json")
        return

    # Generate for a single Pokémon or prompt interactively
    if args.pokemon:
        name = args.pokemon
    else:
        name = input("Enter a Pokémon name: ").strip()

    if args.pipeline == "multitask":
        # build balanced features then predict
        pinfo = pokemon_lookup.get(name.lower())
        feats = extract_features_balanced({'evs':{s:0 for s in ['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed']},
                                          'ivs':{s:31 for s in ['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed']},
                                          'moves':[]}, pinfo)
        df = pd.DataFrame([feats])
        X_num = mt_scaler.transform(df[num_cols])
        X_cat = mt_encoder.transform(df[cat_cols])
        X_full = np.hstack([X_num, X_cat]).astype(np.float32)
        out = mt_model(torch.from_numpy(X_full))
        print("Multi-task output:", out)

    elif args.pipeline == "hierarchy":
        # Stage1 -> Stage2 -> Stage3 sequential inference
        pinfo = pokemon_lookup.get(name.lower())
        if not pinfo:
            print(f"Pokémon '{name}' not found.")
            return
        # Stage1: predict nature & ability
        stats = [pinfo['stats'][s] for s in ['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed']]
        X1 = torch.tensor([stats], dtype=torch.float32)
        out1 = st1(X1)
        nat_idx = out1['nature'].argmax(dim=1).item()
        abl_idx = out1['ability'].argmax(dim=1).item()
        nature = candidate_natures[nat_idx]
        ability = candidate_abilities[abl_idx]
        # Stage2: predict EVs & IVs
        # build placeholder features for Stage2 input
        feats0 = extract_features_balanced({'evs':{s:0 for s in stats},
                                           'ivs':{s:31 for s in stats},
                                           'moves':[]}, pinfo)
        df0 = pd.DataFrame([feats0])
        X_num = mt_scaler.transform(df0[num_cols])
        X_cat = mt_encoder.transform(df0[cat_cols])
        X2 = torch.from_numpy(np.hstack([X_num, X_cat]).astype(np.float32))
        out2 = st2(X2)
        ev_vals = out2['evs'].detach().numpy().flatten()
        iv_vals = out2['ivs'].detach().numpy().flatten()
        ev_spread = {s: int(val*252) for s, val in zip(['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed'], ev_vals)}
        iv_config = {s: int(val*31) for s, val in zip(['HP','Attack','Defense','SpecialAttack','SpecialDefense','Speed'], iv_vals)}
        # Stage3: predict moveset
        X3 = build_configuration_features_interactions(
            pinfo, ev_spread, iv_config,
            nature, ability,
            candidate_items[0], candidate_tera_types[0], [],
            num_cols, cat_cols, mt_encoder, mt_scaler
        )
        out3 = st3(torch.from_numpy(X3.astype(np.float32)))
        mv_logits = out3['moves'].detach().numpy().flatten()
        top4 = np.argsort(mv_logits)[-4:][::-1]
        move_set = [GLOBAL_SMOGON_MOVES[i] for i in top4]
        # Print results
        print("Hierarchy pipeline output:")
        print(f" Nature: {nature} | Ability: {ability}")
        print(f" EVs: {ev_spread}")
        print(f" IVs: {iv_config}")
        print(f" Moves: {move_set}")

    else:
        # LEGAL‐POOL GENERATION WITH HEURISTIC PRUNING
        configs_all = get_candidate_configs(name)
        # filter out bad placeholders
        configs_all = [
            c for c in configs_all
            if c['item'] in candidate_items 
            and c['ability'] in candidate_abilities
        ]
        if not configs_all:
            print(f"No legal configs for {name}.")
            return

        # 1) Heuristic pre‐filter to top K
        K = 5000  # cap at 5k combos
        if len(configs_all) > K:
            scores_h = []
            for cfg in tqdm(configs_all, desc="Heuristic scoring", total=len(configs_all)):
                f = build_configuration_features_custom(
                    pokemon_lookup[name],
                    cfg['EV_spread'], cfg['IV_config'],
                    cfg['nature'], cfg['ability'],
                    cfg['item'], cfg['teraType'],
                    cfg['move_set']
                )
                # same as extract_features_balanced’s heuristic
                scores_h.append(f["avg_damage_score"] + f["BST"] / 300.0)
            idxs = np.argsort(scores_h)[-K:]
            configs = [configs_all[i] for i in idxs]
            print(f"→ Pruned {len(configs_all)} → {len(configs)} configs by heuristic")
        else:
            configs = configs_all
            print(f"→ Evaluating {len(configs)} configs for {name}")

        # 2) Batch‐build raw features
        raw_feats = []
        natures   = []
        for cfg in tqdm(configs, desc=f"Featurizing {name}", total=len(configs)):
            raw_feats.append(build_configuration_features_custom(
                pokemon_lookup[name],
                cfg['EV_spread'], cfg['IV_config'],
                cfg['nature'], cfg['ability'],
                cfg['item'], cfg['teraType'],
                cfg['move_set']
            ))
            natures.append(cfg['nature'])

        # 3) One DataFrame + transforms
        df     = pd.DataFrame(raw_feats)
        X_num  = mt_scaler.transform(df[numeric_cols])
        df_cat = df.reindex(columns=categorical_cols, fill_value=None)
        X_cat  = mt_encoder.transform(df_cat)

        # 4) Interaction dims (vectorised)
        cat_names = mt_encoder.get_feature_names_out(categorical_cols)
        nature_map = {
            nm.split("nature_")[1]: i
            for i,nm in enumerate(cat_names) if nm.startswith("nature_")
        }
        N = len(configs)
        X_int = np.zeros((N,4), dtype=np.float32)
        for i, nat in enumerate(natures):
            boosts = natureModifiers.get(nat, {})
            terms  = []
            nat_i  = nature_map.get(nat)
            if nat_i is not None:
                for stat in boosts:
                    ev_i = numeric_cols.index(f"EV_{stat}")
                    iv_i = numeric_cols.index(f"IV_{stat}")
                    terms.append(X_num[i, ev_i] * X_cat[i, nat_i])
                    terms.append(X_num[i, iv_i] * X_cat[i, nat_i])
            terms += [0.0]*(4 - len(terms))
            X_int[i] = terms

        # 5) Concatenate + trim to training dim
        feat_dim = len(numeric_cols) + len(cat_names)
        X_all    = np.hstack([X_num, X_cat, X_int]).astype(np.float32)[:, :feat_dim]

        # 6) Single surrogate forward
        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.from_numpy(X_all).to(device)
        mt_model.to(device).eval()
        with torch.no_grad():
            overall_scores = mt_model(X_tensor)['overall'].cpu().numpy().flatten()

        # 7) Pick best & print
        best_idx = int(np.argmax(overall_scores))
        best_cfg = configs[best_idx]
        print(build_moveset_text(name, best_cfg))

if __name__ == "__main__":
    main()