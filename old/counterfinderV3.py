from typing import Dict, List, Tuple
import json
from collections import defaultdict
from models import PokemonConfig
from repository import PokemonRepository, TypeChart
from moveutility import evaluateMoveUtilities, evaluateDefensiveUtilities, loadMovesData
from abilityutility import getHighestRatedAbility

class CounterFinder:
    """
    Finds the best counter Pokémon based on offensive and defensive utilities,
    including bonuses from base stat total (BST), ability ratings, refined role classification,
    and integrates Smogon movesets via direct match or aliases mapping.
    """
    # Movepool-based role indicators
    supportMoves = {"Tailwind", "Trick Room", "Reflect", "Light Screen", "Follow Me",
                    "Rain Dance", "Sunny Day", "Taunt", "Safeguard", "Trick"}
    pivotMoves   = {"U-Turn", "Volt Switch", "Parting Shot", "Flip Turn", "Shed Tail",
                    "Teleport", "Baton Pass", "Chilly Reception"}
    setupMoves   = {"Swords Dance", "Calm Mind", "Dragon Dance", "Nasty Plot", "Bulk Up",
                    "Shift Gear", "Quiver Dance"}

    def __init__(self, repository: PokemonRepository, typeChart: TypeChart):
        self.repository = repository
        self.typeChart = typeChart
        # Load moves data
        self.movesData = loadMovesData("data/moves.json")
        # Load aliases and Smogon movesets
        with open("data/aliases.json") as f:
            raw_aliases = json.load(f)
        # normalize alias keys and values to lowercase
        self.aliases = {k.lower(): v.lower() for k, v in raw_aliases.items()}
        with open("data/smogonMovesets.json") as f:
            self.smogonMovesets = json.load(f)
        # Build mapping from canonical names to Smogon keys via aliases
        self.canonToAliases: Dict[str, List[str]] = defaultdict(list)
        for smKey, canonName in self.aliases.items():
            self.canonToAliases[canonName].append(smKey)
        # Build direct Smogon lookup by normalized Pokémon name
        self.smogonByName: Dict[str, dict] = {}
        for entry in self.smogonMovesets:
            name = entry.get("pokemonName", "").lower()
            key = name.replace(" ", "").replace("-", "")
            self.smogonByName[key] = entry

    def determineOpponentRole(self, inputStats: Dict[str, int], inputMoves: List[str]) -> str:
        # Movepool-based roles
        if any(move in self.supportMoves for move in inputMoves):
            return "Support"
        if any(move in self.pivotMoves for move in inputMoves):
            return "Pivot"
        if any(move in self.setupMoves for move in inputMoves):
            return "SetupSweeper"
        # Trick Room Sweeper criteria
        if inputStats["Speed"] < 60 and (inputStats["Attack"] >= 110 or inputStats["SpecialAttack"] >= 110):
            return "TrickRoomSweeper"
        # Fast sweepers by thresholds
        if inputStats["Attack"] >= 120 and inputStats["Speed"] >= 90:
            return "PhysicalSweeper"
        if inputStats["SpecialAttack"] >= 120 and inputStats["Speed"] >= 90:
            return "SpecialSweeper"
        # Fallback buckets
        pScore = inputStats["Attack"] + 0.5 * inputStats["Speed"]
        sScore = inputStats["SpecialAttack"] + 0.5 * inputStats["Speed"]
        wScore = inputStats["Defense"] + inputStats["SpecialDefense"]
        tScore = inputStats["HP"] + 0.5 * (inputStats["Defense"] + inputStats["SpecialDefense"])
        scores = {
            "Wall": wScore,
            "Tank": tScore,
            "PhysicalSweeper": pScore,
            "SpecialSweeper": sScore,
        }
        sorted_vals = sorted(scores.values(), reverse=True)
        if len(sorted_vals) > 1 and (sorted_vals[0] - sorted_vals[1]) < 20:
            return "Balanced"
        return max(scores, key=scores.get)

    def calculateCandidateAdvantageBonus(self, role: str, cand: Dict[str, int], inp: Dict[str, int]) -> int:
        bonus = 0
        if role == "PhysicalSweeper":
            if cand["Defense"] > inp["Attack"]:
                bonus += 2
            if cand["HP"] > 0.8 * inp["HP"]:
                bonus += 2
        elif role == "SpecialSweeper":
            if cand["SpecialDefense"] > inp["SpecialAttack"]:
                bonus += 2
            if cand["HP"] > 0.8 * inp["HP"]:
                bonus += 2
        elif role == "Wall":
            if cand["Attack"] > inp["Attack"]:
                bonus += 2
            if cand["SpecialAttack"] > inp["SpecialAttack"]:
                bonus += 2
        elif role == "Tank":
            if cand["HP"] > inp["HP"]:
                bonus += 2
            if cand["Defense"] > inp["Defense"] or cand["SpecialDefense"] > inp["SpecialDefense"]:
                bonus += 2
        elif role == "Balanced":
            if cand["Speed"] > inp["Speed"]:
                bonus += 2
            if cand["Attack"] > inp["Attack"] or cand["SpecialAttack"] > inp["SpecialAttack"]:
                bonus += 2
        elif role == "SetupSweeper":
            if cand["Attack"] > inp["Attack"] or cand["SpecialAttack"] > inp["SpecialAttack"]:
                bonus += 2
            if cand["Speed"] > inp["Speed"]:
                bonus += 2
        elif role == "Support":
            if (cand["Defense"] + cand["SpecialDefense"]) > (inp["Defense"] + inp["SpecialDefense"]):
                bonus += 2
            if cand["HP"] > inp["HP"]:
                bonus += 2
        elif role == "Pivot":
            if cand["HP"] > inp["HP"]:
                bonus += 2
            if cand["Attack"] > inp["Attack"] or cand["SpecialAttack"] > inp["SpecialAttack"]:
                bonus += 2
        elif role == "TrickRoomSweeper":
            if cand["Attack"] > inp["Attack"]:
                bonus += 2
            if cand["SpecialAttack"] > inp["SpecialAttack"]:
                bonus += 2
            if cand["HP"] > 0.8 * inp["HP"]:
                bonus += 2
        return bonus

    def getCounterConfig(self, role: str, candStats: Dict[str, int]) -> Tuple[Dict[str, int], str]:
        # Helper to pick the higher defensive stat
        def higher(a: str, b: str) -> str:
            return a if candStats[a] >= candStats[b] else b

        # Initialize EVs
        evs = {stat: 0 for stat in candStats}
        nature = ""
        # Sweepers
        if role in ("PhysicalSweeper", "SpecialSweeper", "SetupSweeper"):
            if role == "PhysicalSweeper" or (role == "SetupSweeper" and candStats["Attack"] >= candStats["SpecialAttack"]):
                evs["Attack"] = 252
                nature = "Jolly" if candStats["Speed"] >= candStats["Attack"] else "Adamant"
            else:
                evs["SpecialAttack"] = 252
                nature = "Timid" if candStats["Speed"] >= candStats["SpecialAttack"] else "Modest"
            evs["Speed"] = 252
            evs["HP"] = 4
            return evs, nature
        # Trick Room Sweeper
        if role == "TrickRoomSweeper":
            evs["HP"] = 252
            if candStats["Attack"] >= candStats["SpecialAttack"]:
                evs["Attack"] = 252
                nature = "Brave"
            else:
                evs["SpecialAttack"] = 252
                nature = "Quiet"
            evs[higher("Defense", "SpecialDefense")] = 4
            return evs, nature
        # Defensive roles
        if role in ("Wall", "Tank", "Balanced", "Support", "Pivot"):
            evs["HP"] = 252
            if role == "Wall":
                d = higher("Defense", "SpecialDefense")
                evs[d] = 252
                nature = "Impish" if d == "Defense" else "Careful"
            elif role == "Tank":
                d = higher("Defense", "SpecialDefense")
                evs[d] = 168
                off = "Attack" if d == "Defense" else "SpecialAttack"
                evs[off] = 84
                nature = "Impish" if d == "Defense" else "Careful"
            elif role == "Balanced":
                evs["Defense"] = 128
                evs["SpecialDefense"] = 128
                nature = "Bold" if candStats["Defense"] >= candStats["SpecialDefense"] else "Calm"
            elif role == "Support":
                if candStats["Speed"] >= 70:
                    evs["Speed"] = 252
                    evs["Defense"] = 4
                    evs["SpecialDefense"] = 252 - 4
                else:
                    d = higher("Defense", "SpecialDefense")
                    evs[d] = 252
                    other = "SpecialDefense" if d == "Defense" else "Defense"
                    evs[other] = 4
                nature = "Bold" if candStats["Defense"] >= candStats["SpecialDefense"] else "Calm"
            else:  # Pivot
                evs["Speed"] = 252
                evs["Defense"] = 4
                evs["HP"] = 508 - (evs["Speed"] + evs["Defense"])  # fill remaining
                nature = "Jolly"
            return evs, nature
        # Fallback Balanced
        evs["HP"] = 252
        evs["Defense"] = 128
        evs["SpecialDefense"] = 128
        nature = "Bold" if candStats["Defense"] >= candStats["SpecialDefense"] else "Calm"
        return evs, nature

    def findCounters(self, inputPokemon, inputStats: Dict[str, int], inputMoves: List[str]) -> List[PokemonConfig]:
        role = self.determineOpponentRole(inputStats, inputMoves)
        candidatesWithWeight: List[Tuple[PokemonRepository, float, List[str]]] = []

        # 1) Compute base weights and collect move candidates
        for candidate in self.repository.pokemonList:
            if candidate.name.lower() == inputPokemon.name.lower():
                continue

            candStats = candidate.baseStats
            weight = 0.0
            ctMoves: List[str] = []

            # Offensive utility
            moveUtils = evaluateMoveUtilities(inputPokemon, candidate, self.typeChart, self.movesData)
            if moveUtils:
                avgUtil = sum(m[1] for m in moveUtils) / len(moveUtils)
                weight += avgUtil / 50
                ctMoves = [m[0] for m in moveUtils]

            # Defensive utility
            defUtils = evaluateDefensiveUtilities(inputMoves, candidate, self.typeChart, self.movesData)
            if defUtils:
                avgMult = sum(d[2] for d in defUtils) / len(defUtils)
                weight += (1 - avgMult) * 5

            # Advantage bonus, BST bonus, ability bonus
            weight += self.calculateCandidateAdvantageBonus(role, candStats, inputStats)
            weight += sum(candStats.values()) / 300
            bestAbility, abilityRating = getHighestRatedAbility(candidate.abilities)
            weight += abilityRating / 2

            if weight > 0:
                candidatesWithWeight.append((candidate, weight, ctMoves))

        # 2) Sort by descending weight
        candidatesWithWeight.sort(key=lambda x: x[1], reverse=True)

        # 3) Build final list with Smogon moveset and EV/nature match bonus
        finalCounters: List[PokemonConfig] = []
        for candidate, weight, ctMoves in candidatesWithWeight:
            # Normalize name for direct Smogon lookup
            normName = candidate.name.lower().replace(" ", "").replace("-", "")
            movesetEntry = self.smogonByName.get(normName)

            # Fallback to aliases if no direct match
            if not movesetEntry:
                for aliasKey in self.canonToAliases.get(candidate.name.lower(), []):
                    movesetEntry = next(
                        (e for e in self.smogonMovesets if e.get("pokemonName", "").lower() == aliasKey),
                        None
                    )
                    if movesetEntry:
                        break

            # Skip if no Smogon moveset
            if not movesetEntry:
                continue

            # EV/nature match bonus
            configEvs, configNature = self.getCounterConfig(role, candidate.baseStats)
            entryEvs = movesetEntry.get("evs", {})
            entryNature = movesetEntry.get("nature", "")
            if entryEvs == configEvs and entryNature == configNature:
                weight += 1

            # Extract Smogon config (with fallbacks)
            evs = entryEvs
            ivs = movesetEntry.get("ivs", {stat: 31 for stat in candidate.baseStats})
            nature = entryNature
            moves = movesetEntry.get("moves", ctMoves)
            ability = movesetEntry.get("ability", bestAbility)
            item = movesetEntry.get("item", getattr(candidate, "item", None))
            teraType = movesetEntry.get("teraType", getattr(candidate, "teraType", None))

            adjustedStats = candidate.getAdjustedStats(evs, ivs, nature)
            finalCounters.append(PokemonConfig(
                name=candidate.name,
                weight=weight,
                evs=evs,
                ivs=ivs,
                nature=nature,
                adjustedStats=adjustedStats,
                moves=moves,
                ability=ability,
                item=item,
                teraType=teraType
            ))

            if len(finalCounters) >= 5:
                break

        return finalCounters