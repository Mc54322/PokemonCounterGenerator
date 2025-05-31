import json
import os
import re

def cleanText(text: str) -> str:
    """Remove all non-alphanumeric characters and lowercase the text."""
    return re.sub(r'[^a-z0-9]', '', text.lower())

def getMoveByName(moves: dict, moveInput: str) -> dict:
    """Look up a move by input string using the dictionary key or the move's 'name' field."""
    cleanInput = cleanText(moveInput)
    if cleanInput in moves:
        return moves[cleanInput]
    for move in moves.values():
        if cleanText(move.get("name", "")) == cleanInput:
            return move
    return None

def getPokemonByName(pokemonData: list, pokeInput: str) -> dict:
    """Look up a Pokémon by input string by searching through the list's 'name' field."""
    cleanInput = cleanText(pokeInput)
    for poke in pokemonData:
        if cleanText(poke.get("name", "")) == cleanInput:
            return poke
    return None

def calculateMoveUtility(move: dict, pokemon: dict) -> float:
    """
    Compute the competitive utility value of a move given its JSON data and a Pokémon.
    
    Factors include:
      1. Damaging component (basePower, secondary chance, drain, STAB)
      2. Move-applied boosts (from move.boosts) – if the move is a Status move with accuracy True,
         or its target is "self" or "adjacentAllyOrSelf", treat it as self-targeted.
      3. Self-effects (from move.self.boosts)
      4. Recoil penalty (if a recoil array exists)
      5. Recharge penalty (if move.self.volatileStatus is "mustrecharge")
      6. Field effects (volatileStatus, secondary status, sideCondition, pseudoWeather, weather, terrain)
      7. Healing bonus (if move.flags.heal exists; increased for competitive play)
      8. Self-destruct penalty (if move.selfdestruct exists)
      9. PP factor (lower PP adds a slight bonus)
    """
    # Constants
    damageMultiplier = 2.0
    selfBoostMultiplier = 100
    enemyBoostMultiplier = 50
    recoilPenaltyFactor = 1.0
    rechargePenalty = 50
    healingBonus = 100        # Increased bonus for healing moves
    selfDestructPenalty = 350 # Penalty for moves with selfdestruct

    # Field effect mappings
    volatileStatusValues = {
        "flinch": 5, "aquaring": 20, "attract": 10, "confusion": 10,
        "banefulbunker": 80, "bide": 5, "partiallytrapped": 10, "burningbulwark": 70,
        "charge": 5, "curse": 30, "defensecurl": 0, "destinybond": 40,
        "protect": 100, "disable": 10, "dragoncheer": 5, "electrify": 10,
        "embargo": 10, "encore": 40, "endure": 10, "focusenergy": 5,
        "followme": 10, "foresight": 10, "gastroacid": 15, "grudge": 5,
        "healblock": 30, "helpinghand": 10, "imprison": 5, "ingrain": 30,
        "kingsshield": 100, "laserfocus": 5, "leechseed": 30, "magiccoat": 5,
        "magnetrise": 5, "maxguard": 100, "minimize": 5, "miracleeye": 5,
        "nightmare": 10, "noretreat": 5, "obstruct": 100, "octolock": 100,
        "powder": 5, "powershift": 5, "powertrick": 5, "ragepowder": 100,
        "saltcure": 5, "substitute": 20, "silktrap": 5, "smackdown": 5,
        "snatch": 10, "sparklingaria": 5, "spikyshield": 100, "spotlight": 5,
        "stockpile": 20, "syrupbomb": 5, "tarshot": 5, "taunt": 20,
        "telekinesis": 5, "torment": 5, "yawn": 5
    }
    statusValues = {
        "psn": 20, "brn": 20, "frz": 20, "par": 15, "tox": 30, "slp": 15
    }
    sideConditionValues = {
        "auroraveil": 60, "craftyshield": 50, "lightscreen": 50, "luckychant": 40,
        "matblock": 40, "mist": 30, "quickguard": 50, "reflect": 50,
        "safeguard": 40, "spikes": 40, "stealthrock": 60, "stickyweb": 40,
        "tailwind": 50, "toxicspikes": 50, "wideguard": 50
    }
    pseudoWeatherValues = {
        "fairylock": 40, "gravity": 30, "iondeluge": 40, "magicroom": 30,
        "mudsport": 20, "trickroom": 60, "watersport": 20, "wonderroom": 30
    }
    weatherValues = {
        "snow": 30, "hail": 30, "raindance": 40, "sandstorm": 40, "sunnyday": 40
    }
    terrainValues = {
        "electricterrain": 40, "grassyterrain": 40, "mistyterrain": 40, "psychicterrain": 40
    }
    
    utility = 0.0

    # 1. Damaging component
    if move.get("category") != "Status" and isinstance(move.get("basePower"), (int, float)) and move["basePower"] > 0:
        moveType = move.get("type", "").lower()
        pokeTypes = pokemon.get("types", {})
        primaryType = pokeTypes.get("primary", "").lower()
        secondaryType = ""
        if pokeTypes.get("secondary") and pokeTypes.get("secondary").lower() != "n/a":
            secondaryType = pokeTypes.get("secondary", "").lower()
        if moveType and (moveType == primaryType or moveType == secondaryType):
            stab = 1.5
        else:
            stab = 1
        utility += move["basePower"] * damageMultiplier * stab
        sec = move.get("secondary")
        if sec and isinstance(sec.get("chance"), (int, float)):
            utility += sec["chance"] * 0.5
        if "drain" in move:
            utility += 10
        

    # 2. Move-applied boosts (from move.boosts)
    if "boosts" in move and isinstance(move["boosts"], dict):
        targetValue = move.get("target", "")
        isSelf = (targetValue in ["self", "adjacentAllyOrSelf"] or (move.get("category") == "Status" and move.get("accuracy") is True))
        pos = sum(v for v in move["boosts"].values() if v > 0)
        neg = sum(abs(v) for v in move["boosts"].values() if v < 0)
        if isSelf:
            utility += pos * selfBoostMultiplier
            utility -= neg * selfBoostMultiplier
        else:
            utility -= pos * enemyBoostMultiplier
            utility += neg * enemyBoostMultiplier

    # 3. Self-effects: move.self.boosts
    if "self" in move and isinstance(move["self"], dict) and "boosts" in move["self"]:
        pos = sum(v for v in move["self"]["boosts"].values() if v > 0)
        neg = sum(abs(v) for v in move["self"]["boosts"].values() if v < 0)
        utility += pos * selfBoostMultiplier
        utility -= neg * selfBoostMultiplier

    # 4. Recoil penalty
    if "recoil" in move and isinstance(move["recoil"], (list, tuple)) and move.get("basePower"):
        num, den = move["recoil"]
        fraction = num / den
        utility -= move["basePower"] * fraction * recoilPenaltyFactor

    # 5. Recharge penalty
    if "self" in move and isinstance(move["self"], dict):
        if move["self"].get("volatileStatus", "").lower() == "mustrecharge":
            utility -= rechargePenalty

    # 6. Field effects
    if isinstance(move.get("volatileStatus"), str):
        utility += volatileStatusValues.get(move["volatileStatus"].lower(), 0)
    sec = move.get("secondary")
    if sec and isinstance(sec, dict):
        if isinstance(sec.get("volatileStatus"), str):
            utility += volatileStatusValues.get(sec["volatileStatus"].lower(), 0)
        if isinstance(sec.get("status"), str):
            utility += statusValues.get(sec["status"].lower(), 0)
    if isinstance(move.get("sideCondition"), str):
        utility += sideConditionValues.get(move["sideCondition"].lower(), 0)
    if isinstance(move.get("pseudoWeather"), str):
        utility += pseudoWeatherValues.get(move["pseudoWeather"].lower(), 0)
    if isinstance(move.get("weather"), str):
        utility += weatherValues.get(move["weather"].lower(), 0)
    if isinstance(move.get("terrain"), str):
        utility += terrainValues.get(move["terrain"].lower(), 0)

    # 7. Healing bonus (if move.flags.heal exists)
    if "flags" in move and isinstance(move["flags"], dict) and move["flags"].get("heal") and not "drain" in move:
        utility += healingBonus

    # 8. Self-destruct penalty (if move.selfdestruct exists and is truthy)
    if move.get("selfdestruct"):
        utility -= selfDestructPenalty

    # 9. PP factor
    if isinstance(move.get("pp"), (int, float)):
        utility += (100 - move["pp"]) * 0.05

    return utility

if __name__ == "__main__":
    baseDir = os.path.dirname(__file__)
    movesPath = os.path.join(baseDir, "../core/data/moves.json")
    pokemonPath = os.path.join(baseDir, "../core/data/pokemon.json")
    
    with open(movesPath, "r") as f:
        moves = json.load(f)
    with open(pokemonPath, "r") as f:
        pokemonData = json.load(f)  # List of Pokémon dictionaries
    
    moveInput = input("Enter move name: ")
    pokeInput = input("Enter Pokemon name: ")
    
    moveData = getMoveByName(moves, moveInput)
    pokeData = getPokemonByName(pokemonData, pokeInput)
    
    if moveData and pokeData:
        util = calculateMoveUtility(moveData, pokeData)
        print(f"Move: {moveData.get('name', moveInput)}")
        print(f"Pokemon: {pokeData.get('name', pokeInput)}")
        print(f"Final Utility (with STAB if applicable): {util:.2f}")
    else:
        if not moveData:
            print(f"Move '{moveInput}' not found.")
        if not pokeData:
            print(f"Pokemon '{pokeInput}' not found.")
