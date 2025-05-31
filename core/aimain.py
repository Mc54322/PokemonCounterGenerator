from pathlib import Path
import json
import argparse
from repository import PokemonRepository, TypeChart
from aicounter import AICounterFinder
from parser import getCompetitiveMovesetInput, parseCompetitiveMoveset
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

def buildMoveset(pokemon, evs, ivs, nature, item, ability, teraType, moves):
    """
    Build a complete moveset text string in Showdown format.
    """
    lines = []
    if item:
        lines.append(f"{pokemon} @ {item}")
    else:
        lines.append(pokemon)
        
    if ability:
        lines.append(f"Ability: {ability}")
        
    # Add EVs if any
    ev_parts = []
    for stat_name, value in evs.items():
        if value > 0:
            abbr = {"HP": "HP", "Attack": "Atk", "Defense": "Def", 
                    "SpecialAttack": "SpA", "SpecialDefense": "SpD", "Speed": "Spe"}
            ev_parts.append(f"{value} {abbr.get(stat_name, stat_name)}")
    if ev_parts:
        lines.append(f"EVs: {' / '.join(ev_parts)}")
        
    # Add IVs if not all 31
    iv_parts = []
    for stat_name, value in ivs.items():
        if value < 31:  # Only include non-perfect IVs
            abbr = {"HP": "HP", "Attack": "Atk", "Defense": "Def", 
                    "SpecialAttack": "SpA", "SpecialDefense": "SpD", "Speed": "Spe"}
            iv_parts.append(f"{value} {abbr.get(stat_name, stat_name)}")
    if iv_parts:
        lines.append(f"IVs: {' / '.join(iv_parts)}")
        
    if nature:
        lines.append(f"{nature} Nature")
        
    if teraType:
        lines.append(f"Tera Type: {teraType}")
        
    # Add moves
    for move in moves:
        lines.append(f"- {move}")
        
    return "\n".join(lines)

def processSmogonMoveset(counterFinder, movesetData, repository):
    """
    Process a single Smogon moveset entry and find counters.
    Returns the best counters and the input moveset string.
    """
    pokemonName = movesetData["pokemonName"]
    evs = movesetData.get("evs", {})
    ivs = movesetData.get("ivs", {stat: 31 for stat in ["HP", "Attack", "Defense", "SpecialAttack", "SpecialDefense", "Speed"]})
    nature = movesetData.get("nature", "Neutral")
    item = movesetData.get("item", "")
    ability = movesetData.get("ability", "")
    teraType = movesetData.get("teraType", "")
    moves = movesetData.get("moves", [])
    
    try:
        inputPokemon = repository.getPokemonByName(pokemonName)
    except ValueError:
        print(f"Pokémon '{pokemonName}' not found in repository. Skipping.")
        return [], ""
        
    # Calculate adjusted stats using EVs, IVs, and nature
    inputStats = inputPokemon.getAdjustedStats(evs, ivs, nature)
    
    # Build complete moveset string
    moveset_str = buildMoveset(
        pokemonName, evs, ivs, nature, item, ability, teraType, moves
    )
    
    # Find counters using the AI counter finder
    counterPokemon = counterFinder.findCounters(inputPokemon, inputStats, moves, moveset_str)
    
    return counterPokemon, moveset_str

def main() -> None:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="AI-Enhanced Pokémon Counter Finder")
    parser.add_argument("--train", action="store_true", help="Train the AI on Smogon movesets")
    parser.add_argument("--episodes", type=int, default=3, help="Number of RL episodes per matchup")
    parser.add_argument("--battles", type=int, default=5, help="Number of battles per episode")
    parser.add_argument("--load-weights", type=str, default="ai_counter_weights.json", help="Load weights from file")
    parser.add_argument("--save-weights", type=str, default="ai_counter_weights.json", help="Save weights to file")
    parser.add_argument("--input-file", type=str, help="Input file with moveset in Showdown format")
    parser.add_argument("--test-smogon", action="store_true", help="Test counter finding on all Smogon movesets")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of movesets to process")
    args = parser.parse_args()

    baseDir = Path(__file__).resolve().parent
    pokemonFilePath = baseDir / "data" / "pokemon.json"
    typeChartFilePath = baseDir / "data" / "typeChart.csv"
    smogonFilePath = baseDir / "data" / "smogonMovesets.json"

    try:
        repository = PokemonRepository(pokemonFilePath)
        typeChart = TypeChart(typeChartFilePath)
        counterFinder = AICounterFinder(repository, typeChart)
        
        # Try to load existing weights
        try:
            counterFinder.loadWeights(args.load_weights)
        except Exception as e:
            print(f"Note: Could not load weights ({e}). Using default weights.")
    except Exception as e:
        print("Error loading data:", e)
        return

    if args.train:
        # Training mode: process Smogon movesets and learn
        print("Training AI counter finder on Smogon movesets...")
        
        try:
            with open(smogonFilePath, "r") as f:
                smogonData = json.load(f)
        except Exception as e:
            print(f"Error loading Smogon data: {e}")
            return
            
        # Limit number of movesets for training if specified
        if args.limit > 0:
            movesets_to_process = smogonData[:args.limit]
        else:
            movesets_to_process = smogonData
            
        for i, moveset in enumerate(movesets_to_process):
            print(f"\nProcessing {i+1}/{len(movesets_to_process)}: {moveset['pokemonName']}")
            counterPokemon, moveset_str = processSmogonMoveset(counterFinder, moveset, repository)
            
            if counterPokemon:
                print(f"Found {len(counterPokemon)} potential counters.")
                counterFinder.runReinforcementLearning(
                    moveset_str, counterPokemon, episodes=args.episodes, battles_per_episode=args.battles
                )
            else:
                print("No counters found. Skipping RL.")
        
        # Save final weights
        counterFinder.saveWeights(args.save_weights)
        print(f"Training complete. Weights saved to {args.save_weights}")
        
    elif args.test_smogon:
        # Test mode: evaluate counter finding on Smogon movesets without learning
        print("Testing AI counter finder on Smogon movesets...")
        
        try:
            with open(smogonFilePath, "r") as f:
                smogonData = json.load(f)
        except Exception as e:
            print(f"Error loading Smogon data: {e}")
            return
            
        # Limit number of movesets for testing if specified
        if args.limit > 0:
            movesets_to_process = smogonData[:args.limit]
        else:
            movesets_to_process = smogonData
            
        for i, moveset in enumerate(movesets_to_process):
            print(f"\nProcessing {i+1}/{len(movesets_to_process)}: {moveset['pokemonName']}")
            counterPokemon, moveset_str = processSmogonMoveset(counterFinder, moveset, repository)
            
            if counterPokemon:
                print(f"Found {len(counterPokemon)} potential counters:")
                for j, counter in enumerate(counterPokemon):
                    print(f"\nCounter {j+1}: {counter.name} (Score: {counter.weight:.2f})")
                    counter_str = counterFinder.buildShowdownFormat(counter)
                    print(counter_str)
            else:
                print("No counters found.")
                
    elif args.input_file:
        # Input file mode: read moveset from file
        try:
            with open(args.input_file, "r") as f:
                moveset_str = f.read()
                
            (pokemonName,
             evs,
             ivs,
             nature,
             item,
             ability,
             teraType,
             moves) = parseCompetitiveMoveset(moveset_str)
            
            # validate
            if not validateEvs(evs):
                print("Invalid EVs. Total must not exceed 510, and individual values must be ≤ 252.")
                return
            if not validateIvs(ivs):
                print("Invalid IVs. Values must be between 0 and 31.")
                return
            if not validateNature(nature):
                print(f"Invalid nature: {nature}")
                return

            try:
                inputPokemon = repository.getPokemonByName(pokemonName)
            except ValueError as ve:
                print(ve)
                return

            inputStats = inputPokemon.getAdjustedStats(evs, ivs, nature)
            counterPokemon = counterFinder.findCounters(inputPokemon, inputStats, moves, moveset_str)
            
            print("\nPokémon with both offensive and defensive advantages:\n")
            if not counterPokemon:
                print("No Pokémon found with both offensive and defensive advantages.")
                return

            for counter in counterPokemon:
                counter_str = counterFinder.buildShowdownFormat(counter)
                print(f"{counter_str}\n")
                
            # Option to run simulations on found counters
            run_sims = input("\nRun battle simulations with these counters? (y/n): ").lower().strip()
            if run_sims == 'y':
                for i, counter in enumerate(counterPokemon):
                    print(f"\nTesting counter {i+1}: {counter.name}")
                    win_rate = counterFinder.evaluateCounter(
                        moveset_str, counter, battles=10
                    )
                    print(f"Win rate: {win_rate:.2f}")
                    
        except Exception as e:
            print(f"Error processing input file: {e}")
            return
    else:
        # Interactive mode: get user input
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
            print("Invalid EVs. Total must not exceed 510, and individual values must be ≤ 252.")
            return
        if not validateIvs(ivs):
            print("Invalid IVs. Values must be between 0 and 31.")
            return
        if not validateNature(nature):
            print(f"Invalid nature: {nature}")
            return

        try:
            inputPokemon = repository.getPokemonByName(pokemonName)
        except ValueError as ve:
            print(ve)
            return

        inputStats = inputPokemon.getAdjustedStats(evs, ivs, nature)
        
        # Build complete moveset string for simulations
        moveset_str = buildMoveset(
            pokemonName, evs, ivs, nature, item, ability, teraType, moves
        )
        
        counterPokemon = counterFinder.findCounters(inputPokemon, inputStats, moves, moveset_str)

        print("\nPokémon with both offensive and defensive advantages:\n")
        if not counterPokemon:
            print("No Pokémon found with both offensive and defensive advantages.")
            return

        for i, counter in enumerate(counterPokemon):
            print(f"Counter {i+1}: {counter.name} (Score: {counter.weight:.2f})")
            counter_str = counterFinder.buildShowdownFormat(counter)
            print(counter_str)
            print()  # blank line between counters
            
        # Option to run simulations on found counters
        run_sims = input("\nRun battle simulations with these counters? (y/n): ").lower().strip()
        if run_sims == 'y':
            for i, counter in enumerate(counterPokemon):
                print(f"\nTesting counter {i+1}: {counter.name}")
                win_rate = counterFinder.evaluateCounter(
                    moveset_str, counter, battles=10
                )
                print(f"Win rate: {win_rate:.2f}")
                
        # Option to learn from results
        learn = input("\nUpdate AI with these results? (y/n): ").lower().strip()
        if learn == 'y':
            counterFinder.runReinforcementLearning(
                moveset_str, counterPokemon, episodes=args.episodes, battles_per_episode=args.battles
            )
            counterFinder.saveWeights(args.save_weights)
            print(f"AI updated. Weights saved to {args.save_weights}")

if __name__ == "__main__":
    main()