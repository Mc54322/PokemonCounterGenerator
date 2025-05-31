import json
import os
from parser import parseCompetitiveMoveset

filename = "smogonMovesets.json"

if not os.path.exists(filename):
    with open(filename, "w") as f:
        json.dump([], f)

def saveExtractedMoveset(moveset, tier):
    # Parse the moveset into its detailed components.
    pokemonName, evs, ivs, nature, item, ability, teraType, moves = parseCompetitiveMoveset(moveset)
    entry = {
        "pokemonName": pokemonName,
        "evs": evs,
        "ivs": ivs,
        "nature": nature,
        "item": item,
        "ability": ability,
        "teraType": teraType,
        "moves": moves,
        "pokemonTier": tier
    }
    with open(filename, "r") as f:
        data = json.load(f)
    data.append(entry)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
