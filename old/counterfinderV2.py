from typing import Dict, List, Tuple
from models import PokemonConfig
from repository import PokemonRepository, TypeChart
from moveutility import evaluateMoveUtilities, evaluateDefensiveUtilities, loadMovesData
from abilityutility import getHighestRatedAbility  # new import

class CounterFinder:
    """
    Finds the best counter Pokémon based on offensive and defensive utilities,
    including bonuses from base stat total (BST) and ability ratings.
    """
    def __init__(self, repository: PokemonRepository, typeChart: TypeChart):
        self.repository = repository
        self.typeChart = typeChart
        # Load moves data once from moves.json
        self.movesData = loadMovesData("data/moves.json")

    def determineOpponentRole(self, inputStats: Dict[str, int]) -> str:
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

    def findCounters(self, inputPokemon, inputStats: Dict[str, int], inputMoves: List[str]) -> List[PokemonConfig]:
        opponentRole = self.determineOpponentRole(inputStats)
        matchingCounters: List[PokemonConfig] = []
        for candidate in self.repository.pokemonList:
            if candidate.name.lower() == inputPokemon.name.lower():
                continue
            candidateStats = candidate.baseStats
            weight = 0

            # Offensive evaluation: candidate's moves utility against the input Pokémon.
            moveUtilities = evaluateMoveUtilities(inputPokemon, candidate, self.typeChart, self.movesData)
            ctMoves = []
            if moveUtilities:
                avgMoveUtility = sum(mv[1] for mv in moveUtilities) / len(moveUtilities)
                moveBonus = avgMoveUtility / 50
                weight += moveBonus
                for mv in moveUtilities:
                    ctMoves.append(mv[0])

            # Defensive evaluation: input moves' utility against the candidate.
            defensiveUtilities = evaluateDefensiveUtilities(inputMoves, candidate, self.typeChart, self.movesData)
            if defensiveUtilities:
                # Compute average multiplier from the defensive evaluation.
                avgMultiplier = sum(mv[2] for mv in defensiveUtilities) / len(defensiveUtilities)
                # If candidate is vulnerable (avgMultiplier > 1), this yields a negative bonus; if resistant, positive.
                defensiveBonus = (1 - avgMultiplier) * 5  # Adjust constant as needed.
                weight += defensiveBonus

            # Advantage bonus based on candidate's stats.
            bonus = self.calculateCandidateAdvantageBonus(opponentRole, candidateStats, inputStats)
            weight += bonus

            # BST bonus.
            bst = sum(candidateStats.values())
            bst_bonus = bst / 300  # Adjust this divisor as needed.
            weight += bst_bonus

            # Ability bonus: get the candidate's best ability and its rating.
            candidate_best_ability, candidate_ability_rating = getHighestRatedAbility(candidate.abilities)
            # Divide by 2 to scale the ability rating (range -1 to 5 becomes about -0.5 to 2.5).
            ability_bonus = candidate_ability_rating / 2
            weight += ability_bonus

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
                moves=ctMoves,
                ability=candidate_best_ability
            )
            matchingCounters.append(config)
        matchingCounters.sort(key=lambda x: x.weight, reverse=True)
        return matchingCounters[:5] if matchingCounters else []