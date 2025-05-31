from typing import Dict, List, Tuple
from models import PokemonConfig
from repository import PokemonRepository, TypeChart

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
        if len(sortedScores) > 1 and (sortedScores[0] - sortedScores[1]) < 20:
            return "Balanced"
        return maxRole

    def calculateCandidateAdvantageBonus(
        self, opponentRole: str, candidateStats: Dict[str, int], inputStats: Dict[str, int]
    ) -> int:
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

    def findCounters(self, inputPokemon, inputStats: Dict[str, int]) -> List[PokemonConfig]:
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
            if any(ct in candidateTypes for ct in counterTypes):
                bonus = 5 if maxWeakness >= 4 else 4
                weight += bonus
            if any(rt in candidateTypes for rt in resistanceTypes["immune"]):
                weight += 4
            elif any(rt in candidateTypes for rt in resistanceTypes["resistant"]):
                weight += 3
            bonus = self.calculateCandidateAdvantageBonus(opponentRole, candidateStats, inputStats)
            weight += bonus
            if weight <= 0:
                continue
            configEVs, configNature = self.getCounterConfig(opponentRole, candidateStats)
            configIVs = {stat: 31 for stat in candidateStats}
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
