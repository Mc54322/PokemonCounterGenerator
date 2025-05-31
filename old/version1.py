import os
import json
import pandas as pd

# Define base directory and file paths
baseDir = os.path.dirname(__file__)  # Directory of the script
pokemonFilePath = os.path.join(baseDir, "../data/pokemon.json")
typeChartFilePath = os.path.join(baseDir, "../data/typeChart.csv")

# Load Pokémon data and type chart
with open(pokemonFilePath, 'r') as file:
    pokemonData = json.load(file)

typeChart = pd.read_csv(typeChartFilePath, index_col=0)

# Clean up type chart names (strip whitespace, capitalize for consistency)
typeChart.columns = typeChart.columns.str.strip().str.title()
typeChart.index = typeChart.index.str.strip().str.title()

def validatePokemon(pokemonName, pokemonData):
    """
    Check if the given Pokémon name exists in the dataset.

    Args:
        pokemonName (str): Name of the Pokémon.
        pokemonData (list): List of Pokémon data.

    Returns:
        bool: True if the Pokémon exists, False otherwise.
    """
    return any(pokemon['name'].lower() == pokemonName.lower() for pokemon in pokemonData)

def getPokemonStats(pokemonName, pokemonData):
    """
    Retrieve the stats of a Pokémon by its name.

    Args:
        pokemonName (str): Name of the Pokémon.
        pokemonData (list): List of Pokémon data.

    Returns:
        dict: A dictionary of stats for the Pokémon.
    """
    for pokemon in pokemonData:
        if pokemon['name'].lower() == pokemonName.lower():
            return {stat: int(value) for stat, value in pokemon['stats'].items()}
    return {}

def getPokemonTypes(pokemonName, pokemonData):
    """
    Retrieve the types of a Pokémon by its name.

    Args:
        pokemonName (str): Name of the Pokémon.
        pokemonData (list): List of Pokémon data.

    Returns:
        tuple: (Primary type, Secondary type) or ('N/A', 'N/A') if not found.
    """
    for pokemon in pokemonData:
        if pokemon['name'].lower() == pokemonName.lower():
            primaryType = pokemon['types']['primary']
            secondaryType = pokemon['types'].get('secondary', 'N/A')
            return primaryType, secondaryType
    return None, None

def findPokemonByType(pokemonData, targetType):
    """
    Find all Pokémon of a specified type.

    Args:
        pokemonData (list): List of Pokémon data.
        targetType (str): Target type to search for.

    Returns:
        list: Names of matching Pokémon.
    """
    return [
        pokemon['name'] for pokemon in pokemonData
        if pokemon['types']['primary'] == targetType or pokemon['types'].get('secondary') == targetType
    ]

def findBestCounters(primaryType, secondaryType, typeChart):
    """
    Determine the best counters for the given Pokémon types.

    Args:
        primaryType (str): Primary type of the Pokémon.
        secondaryType (str): Secondary type of the Pokémon (or 'N/A').
        typeChart (pd.DataFrame): Type effectiveness chart.

    Returns:
        tuple: (List of counter types, maximum weakness multiplier).
    """
    primaryWeaknesses = typeChart.loc[:, primaryType]

    # Combine weaknesses if secondary type exists
    if secondaryType == "N/A" or primaryType == secondaryType:
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

    Args:
        primaryType (str): Primary type of the Pokémon.
        secondaryType (str): Secondary type of the Pokémon (or 'N/A').
        typeChart (pd.DataFrame): Type effectiveness chart.

    Returns:
        dict: A dictionary containing immune and resistant types with priority on immune types.
    """
    primaryResistances = typeChart.loc[primaryType, :]

    if secondaryType != "N/A" and primaryType != secondaryType:
        secondaryResistances = typeChart.loc[secondaryType, :]
        combinedResistance = primaryResistances.combine(secondaryResistances, min)
    else:
        combinedResistance = primaryResistances

    immuneTypes = combinedResistance[combinedResistance == 0].index.tolist()
    resistantTypes = combinedResistance[combinedResistance < 1].index.tolist()

    return {
        "immune": immuneTypes,
        "resistant": resistantTypes
    }

def findCounterPokemon(pokemonData, counterTypes, inputStats, resistanceTypes):
    """
    Find the best counter Pokémon based on type advantage, stats, and defensive resistance.

    Args:
        pokemonData (list): List of Pokémon data.
        counterTypes (list): List of effective types.
        inputStats (dict): Stats of the input Pokémon.
        resistanceTypes (dict): Dictionary of immune and resistant types.

    Returns:
        list: Names of the top counter Pokémon (up to 5).
    """
    matchingPokemon = []
    for pokemon in pokemonData:
        pokemonTypes = [pokemon['types']['primary'], pokemon['types'].get('secondary', 'N/A')]
        if any(counterType in pokemonTypes for counterType in counterTypes):
            pokemonStats = {stat: int(value) for stat, value in pokemon['stats'].items()}

            weight = 0  # Weight determines suitability as a counter

            # Prioritize Pokémon with immunity or resistance
            if any(resistanceType in pokemonTypes for resistanceType in resistanceTypes['immune']):
                weight += 2  # Higher priority for immunity
            elif any(resistanceType in pokemonTypes for resistanceType in resistanceTypes['resistant']):
                weight += 1  # Lower priority for resistance

            # Adjust weight based on offensive or defensive needs
            if inputStats['Speed'] > pokemonStats['Speed']:
                # Defensive priority
                if inputStats['Attack'] > inputStats['SpecialAttack']:
                    if pokemonStats['Defense'] > inputStats['Attack']:
                        weight += 1
                else:
                    if pokemonStats['SpecialDefense'] > inputStats['SpecialAttack']:
                        weight += 1
            else:
                # Offensive priority
                if inputStats['Defense'] > inputStats['SpecialDefense']:
                    if pokemonStats['Attack'] > inputStats['Defense']:
                        weight += 1
                else:
                    if pokemonStats['SpecialAttack'] > inputStats['SpecialDefense']:
                        weight += 1

            # Add Pokémon to matching list if it has positive weight
            if weight > 0:
                matchingPokemon.append((pokemon['name'], weight))

    # Sort by weight in descending order and return top 5
    matchingPokemon.sort(key=lambda x: x[1], reverse=True)
    return [pokemon[0] for pokemon in matchingPokemon[:5]] if matchingPokemon else []

def main():
    """
    Main function to interact with the user and analyze Pokémon matchups.
    """
    pokemonName = input("Enter the name of the Pokémon: ")

    if not validatePokemon(pokemonName, pokemonData):
        print(f"{pokemonName} is not a valid Pokémon.")
        return

    primaryType, secondaryType = getPokemonTypes(pokemonName, pokemonData)
    inputStats = getPokemonStats(pokemonName, pokemonData)

    print(f"{pokemonName}'s types are: {primaryType} {secondaryType if secondaryType != 'N/A' else ''}")

    counters, maxWeakness = findBestCounters(primaryType, secondaryType, typeChart)
    resistanceTypes = findDefensiveResistance(primaryType, secondaryType, typeChart)

    print(f"\nBest counters for {pokemonName}:")
    print(f"Types that deal {maxWeakness}x damage: {', '.join(counters)}")

    if resistanceTypes['immune']:
        print(f"Types immune to {pokemonName}'s types: {', '.join(resistanceTypes['immune'])}")
    else:
        print(f"Types resistant to {pokemonName}'s types: {', '.join(resistanceTypes['resistant'])}")

    print("\nPokémon with both offensive and defensive advantages:")
    counterPokemon = findCounterPokemon(pokemonData, counters, inputStats, resistanceTypes)
    if counterPokemon:
        print(", ".join(counterPokemon))
    else:
        print("No Pokémon found with both offensive and defensive advantages.")

if __name__ == "__main__":
    main()
