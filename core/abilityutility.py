import json
import re
from typing import Dict, Tuple, List

def loadAbilitiesData(filePath: str) -> Dict[str, dict]:
    """
    Loads abilities data from a JSON file.
    """
    with open(filePath, "r") as f:
        return json.load(f)

def normalizeAbilityName(abilityName: str) -> str:
    """
    Normalizes an ability name by converting it to lowercase and removing any non-alphanumeric characters.
    """
    return re.sub(r'[^a-z0-9]', '', abilityName.lower())

def getHighestRatedAbility(pokemon_abilities: List[str], abilities_data: Dict[str, dict] = None, filePath: str = "data/abilities.json") -> Tuple[str, float]:
    """
    Given a list of Pokémon abilities, returns the name and rating of the ability with the highest rating.
    If abilities_data is not provided, it is loaded from filePath.
    """
    if abilities_data is None:
        abilities_data = loadAbilitiesData(filePath)
    best_ability = None
    best_rating = -float('inf')
    for ability in pokemon_abilities:
        norm = normalizeAbilityName(ability)
        if norm in abilities_data:
            rating = abilities_data[norm].get("rating", 0)
            if rating > best_rating:
                best_rating = rating
                best_ability = abilities_data[norm]["name"]
    return best_ability, best_rating