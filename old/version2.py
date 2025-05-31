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

# Nature modifiers
natureModifiers = {
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

def validateEvs(evs):
    """
    Validate EVs to ensure they are within acceptable ranges.

    Args:
        evs (dict): Dictionary of EVs to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    return all(0 <= value <= 252 for value in evs.values()) and sum(evs.values()) <= 510

def validateIvs(ivs):
    """
    Validate IVs to ensure they are within acceptable ranges.

    Args:
        evs (dict): Dictionary of EVs to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    return all(0 <= value <= 31 for value in ivs.values())

def validateNature(nature):
    """
    Validate if the given nature is valid.

    Args:
        nature (str): Nature of the Pokémon.

    Returns:
        bool: True if the nature is valid, False otherwise.
    """
    return nature in natureModifiers


def getPokemonStats(pokemonName, pokemonData, evs, ivs, nature):
    """
    Retrieve the stats of a Pokémon by its name and adjust them with EVs, IVs, and Nature.

    Args:
        pokemonName (str): Name of the Pokémon.
        pokemonData (list): List of Pokémon data.
        evs (dict): EVs for the Pokémon.
        ivs (dict): IVs for the Pokémon.
        nature (str): Nature of the Pokémon.

    Returns:
        dict: A dictionary of adjusted stats for the Pokémon.
    """
    for pokemon in pokemonData:
        if pokemon['name'].lower() == pokemonName.lower():
            baseStats = {stat: int(value) for stat, value in pokemon['stats'].items()}
            adjustedStats = {}
            
            # Calculate adjusted stats
            for stat in baseStats:
                ev = evs.get(stat, 0)
                iv = ivs.get(stat, 0)
                base = baseStats[stat]
                adjustedStats[stat] = int(((2 * base + iv + (ev // 4)) * 100) // 100 + 5)
                
                # Apply Nature modifier if applicable
                if nature in natureModifiers and stat in natureModifiers[nature]:
                    adjustedStats[stat] = int(adjustedStats[stat] * natureModifiers[nature][stat])
            
            return adjustedStats
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
        list: List of dictionaries containing the names of the top counter Pokémon along with their EVs, IVs, and Nature (up to 5).
    """
    matchingPokemon = []
    for pokemon in pokemonData:
        pokemonTypes = [pokemon['types']['primary'], pokemon['types'].get('secondary', 'N/A')]
        if any(counterType in pokemonTypes for counterType in counterTypes):
            pokemonStats = {stat: int(value) for stat, value in pokemon['stats'].items()}

            weight = 0  # Weight determines suitability as a counter

            # Prioritize Pokémon with immunity or resistance
            if any(resistanceType in pokemonTypes for resistanceType in resistanceTypes['immune']):
                weight += 3  # Higher priority for immunity
            elif any(resistanceType in pokemonTypes for resistanceType in resistanceTypes['resistant']):
                weight += 2  # Lower priority for resistance

            # Categorize opponent's role based on stats
            if inputStats['Attack'] > inputStats['SpecialAttack'] and inputStats['Speed'] > 300:
                opponentRole = 'Physical Sweeper'
            elif inputStats['SpecialAttack'] > inputStats['Attack'] and inputStats['Speed'] > 300:
                opponentRole = 'Special Sweeper'
            elif inputStats['Defense'] > 200 or inputStats['SpecialDefense'] > 200:
                opponentRole = 'Wall'
            elif inputStats['HP'] > 300 and (inputStats['Attack'] > 150 or inputStats['SpecialAttack'] > 150):
                opponentRole = 'Tank'
            else:
                opponentRole = 'Balanced'

            # Adjust weight based on countering the opponent's role
            if opponentRole == 'Physical Sweeper':
                if pokemonStats['Defense'] > inputStats['Attack'] and pokemonStats['HP'] > 150:
                    weight += 3  # Effective physical wall
            elif opponentRole == 'Special Sweeper':
                if pokemonStats['SpecialDefense'] > inputStats['SpecialAttack'] and pokemonStats['HP'] > 150:
                    weight += 3  # Effective special wall
            elif opponentRole == 'Wall':
                if pokemonStats['Attack'] > 150 or pokemonStats['SpecialAttack'] > 150:
                    weight += 3  # Strong attacker to break the wall
            elif opponentRole == 'Tank':
                if pokemonStats['HP'] > 300 and (pokemonStats['Defense'] > 200 or pokemonStats['SpecialDefense'] > 200):
                    weight += 2  # Bulky enough to handle the tank
            elif opponentRole == 'Balanced':
                if pokemonStats['Speed'] > inputStats['Speed'] and (pokemonStats['Attack'] > 150 or pokemonStats['SpecialAttack'] > 150):
                    weight += 2  # Outspeed and hit hard

            # Calculate EVs, IVs, and Nature based on counter strategy
            if opponentRole == 'Physical Sweeper':
                # Defensive counter
                counterEVs = {"HP": 252, "Defense": 252, "SpecialDefense": 4}
                counterNature = "Bold"
            elif opponentRole == 'Special Sweeper':
                # Special defensive counter
                counterEVs = {"HP": 252, "SpecialDefense": 252, "Defense": 4}
                counterNature = "Calm"
            elif opponentRole == 'Wall':
                # Offensive counter
                counterEVs = {"Attack": 252, "Speed": 252, "HP": 4}
                counterNature = "Adamant" if pokemonStats['Attack'] > pokemonStats['SpecialAttack'] else "Modest"
            elif opponentRole == 'Tank':
                # Balanced counter with offensive capabilities
                counterEVs = {"HP": 252, "Attack": 128, "SpecialDefense": 128}
                counterNature = "Careful" if pokemonStats['SpecialDefense'] > pokemonStats['Defense'] else "Impish"
            else:
                # General balanced approach
                counterEVs = {"HP": 252, "Defense": 128, "SpecialDefense": 128}
                counterNature = "Bold" if pokemonStats['Defense'] > pokemonStats['SpecialDefense'] else "Calm"

            counterIVs = {stat: 31 for stat in pokemonStats}  # Assign max IVs for all stats

            # Calculate adjusted stats with EVs, IVs, and Nature
            counterStats = getPokemonStats(pokemon['name'], pokemonData, counterEVs, counterIVs, counterNature)

            # Add Pokémon to matching list if it has positive weight
            if weight > 0:
                matchingPokemon.append({
                    'name': pokemon['name'],
                    'weight': weight,
                    'evs': counterEVs,
                    'ivs': counterIVs,
                    'nature': counterNature,
                    'adjustedStats': counterStats
                })

    # Sort by weight in descending order and return top 5
    matchingPokemon.sort(key=lambda x: x['weight'], reverse=True)
    return matchingPokemon[:5] if matchingPokemon else []


def main():
    """
    Main function to interact with the user and analyze Pokémon matchups.
    """
    pokemonName = input("Enter the name of the Pokémon: ")

    if not validatePokemon(pokemonName, pokemonData):
        print(f"{pokemonName} is not a valid Pokémon.")
        return

    # Get EVs, IVs, and Nature from the user
    evs = {}
    ivs = {}
    stats = ["HP", "Attack", "Defense", "SpecialAttack", "SpecialDefense", "Speed"]

    print("Enter EVs (0-252) for each stat. Total EVs must not exceed 510.")
    for stat in stats:
        evs[stat] = int(input(f"{stat} EV: "))
    if not validateEvs(evs):
        print("Invalid EVs. Please ensure each EV is between 0 and 252, and total EVs do not exceed 510.")
        return

    print("Enter IVs (0-31) for each stat.")
    for stat in stats:
        ivs[stat] = int(input(f"{stat} IV: "))
    if not validateIvs(ivs):
        print("Invalid IVs. Please ensure each IV is between 0 and 31.")
        return

    nature = input("Enter the nature of the Pokémon: ")
    if not validateNature(nature):
        print("Invalid nature. Please enter a valid nature.")
        return

    primaryType, secondaryType = getPokemonTypes(pokemonName, pokemonData)
    inputStats = getPokemonStats(pokemonName, pokemonData, evs, ivs, nature)

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
        for counter in counterPokemon:
            print(f"Name: {counter['name']}, Nature: {counter['nature']}, EVs: {counter['evs']}, IVs: {counter['ivs']}, Adjusted Stats: {counter['adjustedStats']}")
    else:
        print("No Pokémon found with both offensive and defensive advantages.")

if __name__ == "__main__":
    main()
