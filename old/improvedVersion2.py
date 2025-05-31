from pathlib import Path
import json
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    Represents a Pokémon with its base stats and types.
    """
    def __init__(self, data: Dict):
        self.name: str = data["name"]
        self.types: Dict[str, str] = data["types"]
        self.baseStats: Dict[str, int] = {stat: int(value) for stat, value in data["stats"].items()}

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
            adjusted = int(statTotal + 5)  # Constant term from the formula
            if nature in natureModifiers and stat in natureModifiers[nature]:
                adjusted = int(adjusted * natureModifiers[nature][stat])
            adjustedStats[stat] = adjusted
        return adjustedStats


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


class CounterFinder:
    """
    Finds the best counter Pokémon based on type effectiveness and stat matchups.
    """
    def __init__(self, repository: PokemonRepository, typeChart: TypeChart):
        self.repository = repository
        self.typeChart = typeChart

    def determineOpponentRole(self, inputStats: Dict[str, int]) -> str:
        """
        Determine the opponent's role by computing weighted scores for each role.
        Returns the role with the highest score, or "Balanced" if the top scores are too close.
        """
        physicalScore = inputStats["Attack"] + 0.5 * inputStats["Speed"]
        specialScore  = inputStats["SpecialAttack"] + 0.5 * inputStats["Speed"]
        wallScore     = inputStats["Defense"] + inputStats["SpecialDefense"]
        tankScore     = inputStats["HP"] + 0.5 * (inputStats["Defense"] + inputStats["SpecialDefense"])
        scores = {
            "Physical Sweeper": physicalScore,
            "Special Sweeper": specialScore,
            "Wall": wallScore,
            "Tank": tankScore,
        }
        maxRole = max(scores, key=scores.get)
        sortedScores = sorted(scores.values(), reverse=True)
        # If the top two scores are within 20 points, consider the role as Balanced.
        if len(sortedScores) > 1 and (sortedScores[0] - sortedScores[1]) < 20:
            return "Balanced"
        return maxRole

    def calculateCandidateAdvantageBonus(
        self, opponentRole: str, candidateStats: Dict[str, int], inputStats: Dict[str, int]
    ) -> int:
        """
        Calculate an additional bonus for a candidate based on how well its stats counter
        the opponent's role. Each condition met adds +2 to the bonus.
        """
        bonus = 0
        if opponentRole == "Physical Sweeper":
            if candidateStats["Defense"] > inputStats["Attack"]:
                bonus += 2
            if candidateStats["HP"] > inputStats["HP"] * 0.8:
                bonus += 2
        elif opponentRole == "Special Sweeper":
            if candidateStats["SpecialDefense"] > inputStats["SpecialAttack"]:
                bonus += 2
            if candidateStats["HP"] > inputStats["HP"] * 0.8:
                bonus += 2
        elif opponentRole == "Wall":
            if candidateStats["Attack"] > inputStats["Attack"]:
                bonus += 2
            if candidateStats["SpecialAttack"] > inputStats["SpecialAttack"]:
                bonus += 2
        elif opponentRole == "Tank":
            if candidateStats["HP"] > inputStats["HP"]:
                bonus += 2
            if candidateStats["Defense"] > inputStats["Defense"] or candidateStats["SpecialDefense"] > inputStats["SpecialDefense"]:
                bonus += 2
        elif opponentRole == "Balanced":
            if candidateStats["Speed"] > inputStats["Speed"]:
                bonus += 2
            if candidateStats["Attack"] > inputStats["Attack"] or candidateStats["SpecialAttack"] > inputStats["SpecialAttack"]:
                bonus += 2
        return bonus

    def findCounters(self, inputPokemon: Pokemon, inputStats: Dict[str, int]) -> List[PokemonConfig]:
        primaryType = inputPokemon.types["primary"]
        secondaryType = inputPokemon.types.get("secondary", "N/A")
        counterTypes, maxWeakness = self.typeChart.getBestCounters(primaryType, secondaryType)
        resistanceTypes = self.typeChart.getDefensiveResistance(primaryType, secondaryType)
        opponentRole = self.determineOpponentRole(inputStats)

        matchingCounters: List[PokemonConfig] = []
        for candidate in self.repository.pokemonList:
            candidateTypes = [candidate.types["primary"], candidate.types.get("secondary", "N/A")]
            candidateStats = candidate.baseStats
            weight = 0

            # Bonus for candidate being a counter type.
            # Use a bonus of +5 if maxWeakness is severe (>= 4), otherwise +4.
            if any(ct in candidateTypes for ct in counterTypes):
                bonus = 5 if maxWeakness >= 4 else 4
                weight += bonus

            # Bonus for defensive resistances:
            if any(rt in candidateTypes for rt in resistanceTypes["immune"]):
                weight += 4
            elif any(rt in candidateTypes for rt in resistanceTypes["resistant"]):
                weight += 3

            # Bonus based on candidate stat advantages against the opponent.
            bonus = self.calculateCandidateAdvantageBonus(opponentRole, candidateStats, inputStats)
            weight += bonus

            # Skip candidates with no added advantages.
            if weight <= 0:
                continue

            configEVs, configNature = self.getCounterConfig(opponentRole, candidateStats)
            configIVs = {stat: 31 for stat in candidateStats}  # Assign max IVs for all stats
            candidateAdjustedStats = candidate.getAdjustedStats(configEVs, configIVs, configNature)
            config = PokemonConfig(
                name=candidate.name,
                weight=weight,
                evs=configEVs,
                ivs=configIVs,
                nature=configNature,
                adjustedStats=candidateAdjustedStats,
            )
            matchingCounters.append(config)

        matchingCounters.sort(key=lambda x: x.weight, reverse=True)
        return matchingCounters[:5] if matchingCounters else []

    def getCounterConfig(self, opponentRole: str, candidateStats: Dict[str, int]) -> Tuple[Dict[str, int], str]:
        if opponentRole == "Physical Sweeper":
            return {"HP": 252, "Defense": 252, "SpecialDefense": 4}, "Bold"
        elif opponentRole == "Special Sweeper":
            return {"HP": 252, "SpecialDefense": 252, "Defense": 4}, "Calm"
        elif opponentRole == "Wall":
            nature = "Adamant" if candidateStats["Attack"] > candidateStats["SpecialAttack"] else "Modest"
            return {"Attack": 252, "Speed": 252, "HP": 4}, nature
        elif opponentRole == "Tank":
            nature = "Careful" if candidateStats["SpecialDefense"] > candidateStats["Defense"] else "Impish"
            return {"HP": 252, "Attack": 128, "SpecialDefense": 128}, nature
        else:  # Balanced
            nature = "Bold" if candidateStats["Defense"] > candidateStats["SpecialDefense"] else "Calm"
            return {"HP": 252, "Defense": 128, "SpecialDefense": 128}, nature


def getUserInput() -> Tuple[str, Dict[str, int], Dict[str, int], str]:
    """
    Gathers user input for Pokémon name, EVs, IVs, and nature.
    """
    pokemonName = input("Enter the name of the Pokémon: ")
    stats = ["HP", "Attack", "Defense", "SpecialAttack", "SpecialDefense", "Speed"]

    evs: Dict[str, int] = {}
    print("Enter EVs (0-252) for each stat. Total EVs must not exceed 510.")
    for stat in stats:
        while True:
            try:
                value = int(input(f"{stat} EV: "))
                evs[stat] = value
                break
            except ValueError:
                print(f"Invalid input for {stat}. Please enter an integer.")

    ivs: Dict[str, int] = {}
    print("Enter IVs (0-31) for each stat.")
    for stat in stats:
        while True:
            try:
                value = int(input(f"{stat} IV: "))
                ivs[stat] = value
                break
            except ValueError:
                print(f"Invalid input for {stat}. Please enter an integer.")

    nature = input("Enter the nature of the Pokémon: ")
    return pokemonName, evs, ivs, nature


def validateEvs(evs: Dict[str, int]) -> bool:
    return all(0 <= value <= 252 for value in evs.values()) and sum(evs.values()) <= 510


def validateIvs(ivs: Dict[str, int]) -> bool:
    return all(0 <= value <= 31 for value in ivs.values())


def validateNature(nature: str) -> bool:
    return nature in natureModifiers


def main() -> None:
    baseDir = Path(__file__).resolve().parent
    pokemonFilePath = baseDir.joinpath("../data/pokemon.json")
    typeChartFilePath = baseDir.joinpath("../data/typeChart.csv")

    try:
        repository = PokemonRepository(pokemonFilePath)
    except Exception as e:
        print("Error loading Pokémon data:", e)
        return

    try:
        typeChart = TypeChart(typeChartFilePath)
    except Exception as e:
        print("Error loading type chart:", e)
        return

    pokemonName, evs, ivs, nature = getUserInput()

    if not validateEvs(evs):
        print("Invalid EVs. Please ensure each EV is between 0 and 252, and total EVs do not exceed 510.")
        return
    if not validateIvs(ivs):
        print("Invalid IVs. Please ensure each IV is between 0 and 31.")
        return
    if not validateNature(nature):
        print("Invalid nature. Please enter a valid nature.")
        return

    try:
        inputPokemon = repository.getPokemonByName(pokemonName)
    except ValueError as ve:
        print(ve)
        return

    inputStats = inputPokemon.getAdjustedStats(evs, ivs, nature)
    print(f"\n{inputPokemon.name}'s types are: {inputPokemon.types['primary']} {inputPokemon.types.get('secondary', 'N/A')}")

    # Display type matchup information
    bestCounters, maxWeakness = typeChart.getBestCounters(inputPokemon.types["primary"], inputPokemon.types.get("secondary", "N/A"))
    resistanceTypes = typeChart.getDefensiveResistance(inputPokemon.types["primary"], inputPokemon.types.get("secondary", "N/A"))
    print(f"\nBest counters for {inputPokemon.name}:")
    print(f"Types that deal {maxWeakness}x damage: {', '.join(bestCounters)}")
    if resistanceTypes["immune"]:
        print(f"Types immune to {inputPokemon.name}'s types: {', '.join(resistanceTypes['immune'])}")
    else:
        print(f"Types resistant to {inputPokemon.name}'s types: {', '.join(resistanceTypes['resistant'])}")

    counterFinder = CounterFinder(repository, typeChart)
    counterPokemon = counterFinder.findCounters(inputPokemon, inputStats)
    print("\nPokémon with both offensive and defensive advantages:")
    if counterPokemon:
        for counter in counterPokemon:
            print(
                f"Name: {counter.name}, Nature: {counter.nature}, EVs: {counter.evs}, "
                f"IVs: {counter.ivs}, Adjusted Stats: {counter.adjustedStats}, Weight: {counter.weight}"
            )
    else:
        print("No Pokémon found with both offensive and defensive advantages.")


if __name__ == "__main__":
    main()
