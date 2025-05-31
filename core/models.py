from typing import Dict, List
from dataclasses import dataclass

# Global nature modifiers (using camelCase)
natureModifiers: Dict[str, Dict[str, float]] = {
    "Adamant": {"Attack": 1.1, "SpecialAttack": 0.9},
    "Bashful": {"Attack": 1.0, "SpecialAttack": 1.0},  # Neutral nature
    "Bold": {"Defense": 1.1, "Attack": 0.9},
    "Brave": {"Attack": 1.1, "Speed": 0.9},
    "Calm": {"SpecialDefense": 1.1, "Attack": 0.9},
    "Careful": {"SpecialDefense": 1.1, "SpecialAttack": 0.9},
    "Docile": {"Attack": 1.0, "Defense": 1.0},  # Neutral nature
    "Gentle": {"SpecialDefense": 1.1, "Defense": 0.9},
    "Hardy": {"Attack": 1.0, "Defense": 1.0},  # Neutral nature
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
    "Quirky": {"Attack": 1.0, "SpecialAttack": 1.0},  # Neutral nature
    "Rash": {"SpecialAttack": 1.1, "SpecialDefense": 0.9},
    "Relaxed": {"Defense": 1.1, "Speed": 0.9},
    "Sassy": {"SpecialDefense": 1.1, "Speed": 0.9},
    "Serious": {"Speed": 1.0, "Defense": 1.0},  # Neutral nature
    "Timid": {"Speed": 1.1, "Attack": 0.9}
}

class Pokemon:
    """
    Represents a Pokémon with its base stats, types, learnable moves, and abilities.
    """
    def __init__(self, data: Dict):
        self.name: str = data["name"]
        self.types: Dict[str, str] = data["types"]
        self.baseStats: Dict[str, int] = {stat: int(value) for stat, value in data["stats"].items()}
        self.learnable_moves: List[str] = data.get("learnable_moves", [])
        self.abilities: List[str] = data.get("abilities", [])
    
    def getAdjustedStats(self, evs: Dict[str, int], ivs: Dict[str, int], nature: str) -> Dict[str, int]:
        """
        Calculate adjusted stats based on EVs, IVs, and nature.
        Assumes a simplified level 100 formula.
        """
        adjustedStats: Dict[str, int] = {}
        for stat, base in self.baseStats.items():
            ev = evs.get(stat, 0)
            iv = ivs.get(stat, 0)
            statTotal = (2 * base + iv + (ev // 4))
            adjusted = int(statTotal + 5)
            if nature in natureModifiers and stat in natureModifiers[nature]:
                adjusted = int(adjusted * natureModifiers[nature][stat])
            adjustedStats[stat] = adjusted
        return adjustedStats

@dataclass
class PokemonConfig:
    """
    Dataclass representing a candidate counter Pokémon configuration.
    """
    name: str
    weight: float
    evs: Dict[str, int]
    ivs: Dict[str, int]
    nature: str
    adjustedStats: Dict[str, int]
    moves: List[str]
    ability: str
    item: str
    teraType: str