import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from models import Pokemon

class PokemonRepository:
    """
    Loads and indexes Pokémon from a JSON file.
    """
    def __init__(self, filePath: Path):
        with open(filePath, "r") as file:
            data = json.load(file)
        self.pokemonList: List[Pokemon] = [Pokemon(p) for p in data]
        self.index: Dict[str, Pokemon] = {p.name.lower(): p for p in self.pokemonList}

    def getPokemonByName(self, name: str) -> Pokemon:
        pokemon = self.index.get(name.lower())
        if not pokemon:
            raise ValueError(f"Pokémon '{name}' not found.")
        return pokemon

class TypeChart:
    """
    Loads the type effectiveness chart and computes weaknesses and resistances.
    """
    def __init__(self, filePath: Path):
        self.df = pd.read_csv(filePath, index_col=0)
        self.df.columns = self.df.columns.str.strip().str.title()
        self.df.index = self.df.index.str.strip().str.title()

    def getOverallWeakness(self, primaryType: str, secondaryType: str) -> pd.Series:
        primaryWeakness = self.df.loc[:, primaryType]
        if secondaryType == "N/A" or primaryType == secondaryType:
            return primaryWeakness
        else:
            secondaryWeakness = self.df.loc[:, secondaryType]
            return primaryWeakness * secondaryWeakness

    def getBestCounters(self, primaryType: str, secondaryType: str) -> Tuple[List[str], float]:
        overallWeakness = self.getOverallWeakness(primaryType, secondaryType)
        maxWeakness = overallWeakness.max()
        counterTypes = overallWeakness[overallWeakness == maxWeakness].index.tolist()
        return counterTypes, maxWeakness

    def getDefensiveResistance(self, primaryType: str, secondaryType: str) -> Dict[str, List[str]]:
        primaryResistances = self.df.loc[primaryType, :]
        if secondaryType != "N/A" and primaryType != secondaryType:
            secondaryResistances = self.df.loc[secondaryType, :]
            combinedResistance = primaryResistances.combine(secondaryResistances, min)
        else:
            combinedResistance = primaryResistances
        immuneTypes = combinedResistance[combinedResistance == 0].index.tolist()
        resistantTypes = combinedResistance[combinedResistance < 1].index.tolist()
        return {"immune": immuneTypes, "resistant": resistantTypes}
