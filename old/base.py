import os
import json
import pandas as pd

# Define base directory and file paths
baseDirectory = os.path.dirname(__file__)  # Directory of the script
pokemonFilePath = os.path.join(baseDirectory, "../data/pokemon.json")
typeChartFilePath = os.path.join(baseDirectory, "../data/typeChart.csv")

# Load Pokémon data and type chart
with open(pokemonFilePath, 'r') as file:
    pokemonData = json.load(file)

typeChart = pd.read_csv(typeChartFilePath, index_col=0)

# Clean up type chart names (strip whitespace, capitalize)
typeChart.columns = typeChart.columns.str.strip().str.title()
typeChart.index = typeChart.index.str.strip().str.title()

def validatePokemon(pokemonName, pokemonData):
    """
    Check if the given Pokémon name exists in the dataset.
    """
    return any(pokemon['name'].lower() == pokemonName.lower() for pokemon in pokemonData)

def getPokemonTypes(pokemonName, pokemonData):
    """
    Retrieve the types of a Pokémon by its name.
    """
    for pokemon in pokemonData:
        if pokemon['name'].lower() == pokemonName.lower():
            primaryType = pokemon['types']['primary']
            secondaryType = pokemon['types'].get('secondary')
            if secondaryType == "N/A":
                secondaryType = None
            return primaryType, secondaryType
    return None, None

def findPokemonByType(pokemonData, targetType):
    """
    Find all Pokémon of a specified type.
    """
    return [
        pokemon['name'] for pokemon in pokemonData
        if pokemon['types']['primary'] == targetType or pokemon['types'].get('secondary') == targetType
    ]

def findBestCounters(primaryType, secondaryType, typeChart):
    """
    Determine the best counters for the given Pokémon types.
    """
    primaryWeaknesses = typeChart.loc[:, primaryType]

    # Combine weaknesses if a secondary type exists
    if secondaryType is None or primaryType == secondaryType:
        overallWeakness = primaryWeaknesses
    else:
        secondaryWeaknesses = typeChart.loc[:, secondaryType]
        overallWeakness = primaryWeaknesses * secondaryWeaknesses

    maxWeakness = overallWeakness.max()
    counterTypes = overallWeakness[overallWeakness == maxWeakness].index.tolist()

    return counterTypes, maxWeakness

def findDefensiveResistance(primaryType, secondaryType, typeChart):
    """
    Find types that are immune or resistant to the given Pokémon's types.
    """
    primaryResistances = typeChart.loc[primaryType, :]

    if secondaryType is not None and primaryType != secondaryType:
        secondaryResistances = typeChart.loc[secondaryType, :]
        combinedResistance = primaryResistances.combine(secondaryResistances, min)
    else:
        combinedResistance = primaryResistances

    immuneTypes = combinedResistance[combinedResistance == 0].index.tolist()

    if immuneTypes:
        return immuneTypes, "immune"
    resistantTypes = combinedResistance[combinedResistance < 1].index.tolist()
    return resistantTypes, "resistant"

def findDualTypePokemon(pokemonData, offensiveTypes, defensiveTypes):
    """
    Find Pokémon that match both offensive and defensive criteria.
    """
    matchingPokemon = []
    for pokemon in pokemonData:
        pokemonTypes = [pokemon['types']['primary'], pokemon['types'].get('secondary')]
        if any(offense in pokemonTypes for offense in offensiveTypes) and \
           any(defense in pokemonTypes for defense in defensiveTypes):
            matchingPokemon.append(pokemon['name'])
    return matchingPokemon

def main():
    """
    Main function to interact with the user and analyze Pokémon matchups.
    """
    pokemonName = input("Enter the name of the Pokémon: ")

    if not validatePokemon(pokemonName, pokemonData):
        print(f"{pokemonName} is not a valid Pokémon.")
        return

    primaryType, secondaryType = getPokemonTypes(pokemonName, pokemonData)
    if primaryType is None:
        print(f"Could not find types for {pokemonName}.")
        return

    print(f"{pokemonName}'s types are: {primaryType} {secondaryType if secondaryType else ''}")

    counters, maxWeakness = findBestCounters(primaryType, secondaryType, typeChart)
    resistantTypes, resistanceType = findDefensiveResistance(primaryType, secondaryType, typeChart)

    print(f"\nBest counters for {pokemonName}:")
    print(f"Types that deal {maxWeakness}x damage: {', '.join(counters)}")

    if resistanceType == "immune":
        print(f"Types immune to {pokemonName}'s types: {', '.join(resistantTypes)}")
    else:
        print(f"Types resistant to {pokemonName}'s types: {', '.join(resistantTypes)}")

    print("\nPokémon with both offensive and defensive advantages:")
    dualTypePokemon = findDualTypePokemon(pokemonData, counters, resistantTypes)
    if dualTypePokemon:
        print(", ".join(dualTypePokemon))
    else:
        print("No Pokémon found with both offensive and defensive advantages.")

if __name__ == "__main__":
    main()
