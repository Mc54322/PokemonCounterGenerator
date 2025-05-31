import json
import re
from typing import List, Tuple, Dict, Any
from repository import TypeChart
from models import Pokemon

def normalizeMoveName(moveName: str) -> str:
    """
    Normalizes a move name by converting it to lowercase and removing any non-alphanumeric characters.
    For example, "10,000,000 Volt Thunderbolt" becomes "10000000voltthunderbolt".
    """
    return re.sub(r'[^a-z0-9]', '', moveName.lower())

def loadMovesData(filePath: str) -> Dict[str, Any]:
    """
    Loads moves data from a JSON file.
    """
    with open(filePath, "r") as f:
        return json.load(f)

def evaluateMoveUtilities(inputPokemon: Pokemon, counterPokemon: Pokemon, typeChart: TypeChart, movesData: Dict[str, Any]) -> List[Tuple[str, float, float, float]]:
    """
    Evaluate the offensive move utilities for a candidate Pokémon's moves against the input Pokémon.
    Returns the top four moves sorted by effective utility in descending order.
    """
    primary = inputPokemon.types["primary"]
    secondary = inputPokemon.types.get("secondary", "N/A")
    overallWeakness = typeChart.getOverallWeakness(primary, secondary)
    moveSet = getattr(counterPokemon, "learnable_moves", [])
    moveUtilities = []
    counterTypes = [counterPokemon.types["primary"]]
    if counterPokemon.types.get("secondary", "N/A") != "N/A":
        counterTypes.append(counterPokemon.types["secondary"])
    for move in moveSet:
        normMove = normalizeMoveName(move)
        if normMove not in movesData:
            continue
        moveInfo = movesData[normMove]
        moveType = moveInfo.get("type", "")
        if moveType.lower() in [t.lower() for t in counterTypes]:
            baseUtility = moveInfo.get("stabutility", moveInfo.get("utility", 0))
        else:
            baseUtility = moveInfo.get("utility", 0)
        moveTypeTitle = moveType.title()
        multiplier = overallWeakness.get(moveTypeTitle, 1.0)
        effectiveUtility = baseUtility * multiplier if multiplier != 0 else 0
        moveUtilities.append((move, effectiveUtility, multiplier, baseUtility))
    moveUtilities.sort(key=lambda x: x[1], reverse=True)
    return moveUtilities[:1]

def evaluateDefensiveUtilities(inputMoves: List[str], candidate: Pokemon, typeChart: TypeChart, movesData: Dict[str, Any]) -> List[Tuple[str, float, float, float]]:
    """
    Evaluate how effective the input Pokémon's moves are against the candidate Pokémon.
    Returns a list of tuples (move name, effective utility, multiplier, base utility)
    for the top four input moves sorted by effective utility in ascending order (lower is better defensively).
    """
    primary = candidate.types["primary"]
    secondary = candidate.types.get("secondary", "N/A")
    overallWeakness = typeChart.getOverallWeakness(primary, secondary)
    moveUtilities = []
    for move in inputMoves:
        normMove = normalizeMoveName(move)
        if normMove not in movesData:
            continue
        moveInfo = movesData[normMove]
        baseUtility = moveInfo.get("utility", 0)
        moveType = moveInfo.get("type", "").title()
        multiplier = overallWeakness.get(moveType, 1.0)
        effectiveUtility = baseUtility * multiplier
        moveUtilities.append((move, effectiveUtility, multiplier, baseUtility))
    moveUtilities.sort(key=lambda x: x[1])
    return moveUtilities[:2]