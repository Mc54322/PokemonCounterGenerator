from typing import Dict, List, Tuple

def parseCompetitiveMoveset(moveset: str) -> Tuple[
    str, Dict[str, int], Dict[str, int], str, str, str, str, List[str]
]:
    """
    Parses a competitive moveset input and extracts:
      - Pokémon name and item (if provided)
      - Ability (if provided)
      - EVs and IVs (if provided; defaults for IVs are maxed)
      - Tera Type (if provided)
      - Nature
      - Moves (list)
    
    Expected moveset format example:
    
      Annihilape @ Leftovers
      Ability: Defiant
      EVs: 240 HP / 252 SpD / 16 Spe
      Tera Type: Water
      Careful Nature
      - Bulk Up
      - Taunt
      - Drain Punch
      - Rage Fist
    """
    lines = moveset.strip().splitlines()

    # Defaults
    pokemonName = ""
    item = ""
    ability = ""
    tera_type = ""
    evs = {"HP": 0, "Attack": 0, "Defense": 0, "SpecialAttack": 0, "SpecialDefense": 0, "Speed": 0}
    ivs = {"HP": 31, "Attack": 31, "Defense": 31, "SpecialAttack": 31, "SpecialDefense": 31, "Speed": 31}
    nature = ""
    moves: List[str] = []

    # Mapping for stat abbreviations to full stat names
    stat_mapping = {
        "HP": "HP",
        "Atk": "Attack",
        "Def": "Defense",
        "SpA": "SpecialAttack",
        "Sp. Atk": "SpecialAttack",
        "SpD": "SpecialDefense",
        "Sp. Def": "SpecialDefense",
        "Spe": "Speed"
    }

    for idx, line in enumerate(lines):
        line = line.strip()
        if idx == 0:
            if "@" in line:
                parts = line.split("@")
                pokemonName = parts[0].strip()
                item = parts[1].strip()
            else:
                pokemonName = line
        elif line.startswith("Ability:"):
            ability = line.split("Ability:")[1].strip()
        elif line.startswith("EVs:"):
            ev_parts = line[4:].split("/")
            for part in ev_parts:
                part = part.strip()
                if part:
                    tokens = part.split()
                    if len(tokens) >= 2:
                        try:
                            value = int(tokens[0])
                            stat_abbr = tokens[1]
                            if stat_abbr in stat_mapping:
                                evs[stat_mapping[stat_abbr]] = value
                        except ValueError:
                            continue
        elif line.startswith("IVs:"):
            iv_parts = line[4:].split("/")
            for part in iv_parts:
                part = part.strip()
                if part:
                    tokens = part.split()
                    if len(tokens) >= 2:
                        try:
                            value = int(tokens[0])
                            stat_abbr = tokens[1]
                            if stat_abbr in stat_mapping:
                                ivs[stat_mapping[stat_abbr]] = value
                        except ValueError:
                            continue
        elif line.startswith("Tera Type:"):
            tera_type = line.split("Tera Type:")[1].strip()
        elif "Nature" in line:
            tokens = line.split()
            if tokens:
                nature = tokens[0].strip()
        elif line.startswith("-"):
            move = line.lstrip("-").strip()
            moves.append(move)
    if not nature:
        nature = "Bashful"
    return pokemonName, evs, ivs, nature, item, ability, tera_type, moves

def getCompetitiveMovesetInput() -> Tuple[
    str, Dict[str, int], Dict[str, int], str, str, str, str, List[str]
]:
    print("Enter your competitive moveset (end input with an empty line):")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    moveset_str = "\n".join(lines)
    return parseCompetitiveMoveset(moveset_str)
