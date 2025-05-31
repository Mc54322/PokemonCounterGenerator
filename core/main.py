from pathlib import Path
from repository import PokemonRepository, TypeChart
from counterfinder import CounterFinder
from parser import getCompetitiveMovesetInput
from utils import validateEvs, validateIvs, validateNature

# mapping from full stat names to Showdown abbreviations
statAbbrs = {
    "HP": "HP",
    "Attack": "Atk",
    "Defense": "Def",
    "SpecialAttack": "SpA",
    "SpecialDefense": "SpD",
    "Speed": "Spe",
}

def formatStatBlock(stats: dict, defaults: dict, abbrs: dict) -> str:
    """
    Given a stats dict (e.g. evs or ivs) and its default fill (0 or 31),
    return e.g. '252 Atk / 4 SpD' omitting any at the default value.
    """
    parts = []
    for fullName, short in abbrs.items():
        val = stats.get(fullName, defaults.get(fullName))
        if val is None:
            continue
        if val != defaults.get(fullName):
            parts.append(f"{val} {short}")
    return " / ".join(parts)

def main() -> None:
    baseDir = Path(__file__).resolve().parent
    pokemonFilePath = baseDir / "data" / "pokemon.json"
    typeChartFilePath = baseDir / "data" / "typeChart.csv"

    try:
        repository = PokemonRepository(pokemonFilePath)
        typeChart  = TypeChart(typeChartFilePath)
    except Exception as e:
        print("Error loading data:", e)
        return

    # get user input
    (pokemonName,
     evs,
     ivs,
     nature,
     item,
     ability,
     teraType,
     moves) = getCompetitiveMovesetInput()

    # validate
    if not validateEvs(evs):
        print("Invalid EVs …")
        return
    if not validateIvs(ivs):
        print("Invalid IVs …")
        return
    if not validateNature(nature):
        print("Invalid nature …")
        return

    try:
        inputPokemon = repository.getPokemonByName(pokemonName)
    except ValueError as ve:
        print(ve)
        return

    inputStats = inputPokemon.getAdjustedStats(evs, ivs, nature)

    counterFinder  = CounterFinder(repository, typeChart)
    counterPokemon = counterFinder.findCounters(inputPokemon, inputStats, moves)

    print("\nPokémon with both offensive and defensive advantages:\n")
    if not counterPokemon:
        print("No Pokémon found with both offensive and defensive advantages.")
        return

    for counter in counterPokemon:
        # build Showdown export string
        evBlock = formatStatBlock(
            counter.evs,
            defaults={k: 0 for k in statAbbrs},
            abbrs=statAbbrs
        )
        ivBlock = formatStatBlock(
            counter.ivs,
            defaults={k: 31 for k in statAbbrs},
            abbrs=statAbbrs
        )

        item = getattr(counter, "item", None)
        if item:
            header = f"{counter.name} @ {item}"
        else:
            header = counter.name

        lines = [
            header,
            f"Ability: {counter.ability}",
        ]
        if evBlock:
            lines.append(f"EVs: {evBlock}")
        if ivBlock:
            lines.append(f"IVs: {ivBlock}")
        lines.append(f"{counter.nature} Nature")
        teraType = getattr(counter, "teraType", None)
        if teraType:
            lines.append(f"Tera Type: {teraType}")
        else:
            lines.append(f"Tera Type: Stellar")
        for mv in counter.moves:
            lines.append(f"- {mv}")

        # join into one long string
        exportStr = "\n".join(lines)
        print(exportStr)
        print()  # blank line between exports

if __name__ == "__main__":
    main()
