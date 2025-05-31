#!/usr/bin/env python3
import argparse
import json
import random
import time
import asyncio
from pathlib import Path
from repository import PokemonRepository, TypeChart
from aicounter import AICounterFinder
from models import PokemonConfig
from showdownsim import simulate_battle, sanitize_team

def generateShortNickname(pokemon_name: str) -> str:
    """
    Generate a short nickname (≤18 characters) for a Pokemon to avoid Showdown errors.
    """
    # Remove common suffixes and prefixes that make names too long
    name = pokemon_name.replace("-", "").replace(" ", "")
    
    # Special handling for common long names
    nickname_map = {
        "urshifurapidstrike": "UrshifuRS",
        "urshifusinglestrike": "UrshifuSS",
        "necrozmadawnwings": "NecrozmaD",
        "necrozmaduskmane": "NecrozmaM",
        "calyrexicerider": "CalyrexI",
        "calyrexshadowrider": "CalyrexS",
        "basculegionmale": "BascuM",
        "basculegionmale": "BascuF",
        "ogerponwellspring": "OgerponW",
        "ogerponhearthflame": "OgerponH",
        "ogerponcornerstone": "OgerponC",
        "indeedeemale": "IndeedeeM",
        "indeedeefemale": "IndeedeeF",
        "toxtricitylowkey": "ToxtricityL",
        "toxtricity": "ToxtricityA",
        "lycanrocdusk": "LycanrocD",
        "lycanroc": "LycanrocM",
        "meloettaaria": "Meloetta",
        "shayminsky": "ShayminS",
        "shaymin": "ShayminL",
        "hoopaUnbound": "HoopaU",
        "giratinaorigin": "GiratinaO",
        "giratina": "GiratinaA",
        "deoxysattack": "DeoxysA",
        "deoxysdefense": "DeoxysD",
        "deoxysspeed": "DeoxysS",
        "deoxys": "DeoxysN",
        "kyuremblack": "KyuremB",
        "kyuremwhite": "KyuremW",
        "landorus": "LandorusI",
        "landorustherian": "LandorusT",
        "thundurus": "ThundurusI",
        "thundurustherian": "ThundurusT",
        "tornadus": "TornadusI",
        "tornadustherian": "TornadusT",
        "enamorusincarmate": "EnamorusI",
        "enamorustherian": "EnamorusT"
    }
    
    # Check if we have a predefined short name
    name_lower = name.lower()
    for long_name, short_name in nickname_map.items():
        if long_name in name_lower:
            return short_name
    
    # Generic shortening: take first 15 characters
    if len(name) <= 18:
        return name
    else:
        return name[:15]

def buildShowdownMoveset(pokemon_name, evs, ivs, nature, item, ability, teraType, moves, aliases):
    """
    Build a complete moveset text string in Showdown format using proper aliases.
    Includes EV validation to prevent exceeding 510 total.
    Uses short nicknames to avoid Showdown name length errors.
    """
    # Validate EV spread first
    total_evs = sum(evs.values())
    if total_evs > 510:
        print(f"Warning: {pokemon_name} has {total_evs} total EVs, this will cause errors in Showdown")
        return None
    
    # Convert pokemon name to Showdown format
    showdown_name = pokemon_name
    for showdown_key, pokemon_json_name in aliases.items():
        if pokemon_json_name.lower() == pokemon_name.lower():
            showdown_name = showdown_key
            break
    
    # Generate short nickname to avoid length issues
    nickname = generateShortNickname(showdown_name)
    
    lines = []
    if item:
        lines.append(f"{nickname} ({showdown_name}) @ {item}")
    else:
        lines.append(f"{nickname} ({showdown_name})")
        
    if ability:
        lines.append(f"Ability: {ability}")
        
    # Format EVs
    ev_parts = []
    for stat, value in evs.items():
        if value > 0:
            abbr = {"HP": "HP", "Attack": "Atk", "Defense": "Def", 
                    "SpecialAttack": "SpA", "SpecialDefense": "SpD", "Speed": "Spe"}
            ev_parts.append(f"{value} {abbr.get(stat, stat)}")
    if ev_parts:
        lines.append(f"EVs: {' / '.join(ev_parts)}")
        
    # Format IVs
    iv_parts = []
    for stat, value in ivs.items():
        if value < 31:
            abbr = {"HP": "HP", "Attack": "Atk", "Defense": "Def", 
                    "SpecialAttack": "SpA", "SpecialDefense": "SpD", "Speed": "Spe"}
            iv_parts.append(f"{value} {abbr.get(stat, stat)}")
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

def findPokemonWithAliases(repository, pokemon_name, aliases):
    """
    Find a Pokemon by name, trying aliases if direct lookup fails.
    """
    try:
        return repository.getPokemonByName(pokemon_name)
    except ValueError:
        # Try looking up through aliases
        for alias_key, alias_value in aliases.items():
            if alias_key == pokemon_name.lower():
                try:
                    return repository.getPokemonByName(alias_value)
                except ValueError:
                    continue
        return None

def processMoveset(counterFinder, repository, movesetData, aliases):
    """
    Process a single Smogon moveset entry and find counters.
    """
    pokemon_name = movesetData["pokemonName"]
    evs = movesetData.get("evs", {})
    ivs = movesetData.get("ivs", {stat: 31 for stat in ["HP", "Attack", "Defense", "SpecialAttack", "SpecialDefense", "Speed"]})
    nature = movesetData.get("nature", "Neutral")
    item = movesetData.get("item", "")
    ability = movesetData.get("ability", "")
    teraType = movesetData.get("teraType", "")
    moves = movesetData.get("moves", [])
    
    # Validate input EV spread
    total_evs = sum(evs.values())
    if total_evs > 510:
        print(f"Input Pokemon {pokemon_name} has invalid EV spread ({total_evs} total), skipping.")
        return [], ""
    
    # Find input Pokemon
    inputPokemon = findPokemonWithAliases(repository, pokemon_name, aliases)
    if not inputPokemon:
        return [], ""
        
    # Calculate adjusted stats
    inputStats = inputPokemon.getAdjustedStats(evs, ivs, nature)
    
    # Build complete moveset string using Showdown names
    moveset_str = buildShowdownMoveset(
        pokemon_name, evs, ivs, nature, item, ability, teraType, moves, aliases
    )
    
    if moveset_str is None:
        print(f"Failed to build valid moveset for {pokemon_name}")
        return [], ""
    
    # Find counters - this will now only use Pokemon with valid Smogon movesets
    counterPokemon = counterFinder.findCounters(inputPokemon, inputStats, moves, moveset_str)
    
    return counterPokemon, moveset_str

def trainBatch(counterFinder, repository, smogonData, aliases, batch_size=20, episodes=3, battles_per_episode=5):
    """
    Train on a batch of random Smogon movesets.
    Uses memory management to prevent Showdown crashes.
    """
    import gc
    import time
    
    # Randomly sample movesets for this batch
    if len(smogonData) <= batch_size:
        selected_movesets = smogonData
    else:
        selected_movesets = random.sample(smogonData, batch_size)
        
    print(f"Training on {len(selected_movesets)} movesets...")
    
    successful_trainings = 0
    
    for i, moveset in enumerate(selected_movesets):
        pokemon_name = moveset.get("pokemonName", "Unknown")
        print(f"\n[{i+1}/{len(selected_movesets)}] Processing {pokemon_name}")
        
        try:
            counterPokemon, moveset_str = processMoveset(counterFinder, repository, moveset, aliases)
            
            if not counterPokemon:
                print("No counters found. Skipping.")
                continue
            
            # Run reinforcement learning with reduced battle count to prevent memory issues
            print(f"Found {len(counterPokemon)} potential counters.")
            
            counterFinder.runReinforcementLearning(
                moveset_str, counterPokemon, 
                episodes=episodes, battles_per_episode=battles_per_episode
            )
            
            successful_trainings += 1
            
            # Memory management: force garbage collection every few Pokemon
            if i % 5 == 0:
                gc.collect()
                time.sleep(0.5)  # Brief pause to let memory settle
            
        except Exception as e:
            print(f"Error processing {pokemon_name}: {e}")
            # Force cleanup on error
            gc.collect()
            time.sleep(0.2)
            continue
    
    # Final cleanup
    gc.collect()
    
    print(f"\nTraining complete. Successfully processed {successful_trainings}/{len(selected_movesets)} movesets.")
    return successful_trainings

def main():
    parser = argparse.ArgumentParser(description="Mass train AI CounterFinder")
    parser.add_argument("--batches", type=int, default=10, help="Number of training batches")
    parser.add_argument("--batch-size", type=int, default=20, help="Movesets per batch")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per moveset")
    parser.add_argument("--battles", type=int, default=5, help="Battles per episode")
    parser.add_argument("--weights-file", type=str, default="ai_counter_weights.json", 
                        help="File to save/load weights")
    parser.add_argument("--results-dir", type=str, default="results", 
                        help="Directory for result data")
    args = parser.parse_args()

    # Data paths
    baseDir = Path(__file__).resolve().parent
    pokemonFilePath = baseDir / "data" / "pokemon.json"
    typeChartFilePath = baseDir / "data" / "typeChart.csv"
    smogonFilePath = baseDir / "data" / "smogonMovesets.json"
    aliasesFilePath = baseDir / "data" / "aliases.json"

    # Initialize repository and counter finder
    try:
        repository = PokemonRepository(pokemonFilePath)
        typeChart = TypeChart(typeChartFilePath)
        counterFinder = AICounterFinder(repository, typeChart)
        
        # Load aliases
        with open(aliasesFilePath, "r") as f:
            aliases = json.load(f)
        # Convert to lowercase for case-insensitive matching
        aliases = {k.lower(): v for k, v in aliases.items()}
        
        # Load Smogon data and validate it
        with open(smogonFilePath, "r") as f:
            smogonData = json.load(f)
        
        print(f"Loaded {len(smogonData)} Smogon movesets")
        
        # Validate and filter Smogon data
        validSmogonData = []
        invalidCount = 0
        for entry in smogonData:
            evs = entry.get("evs", {})
            total_evs = sum(evs.values())
            if total_evs <= 510:
                validSmogonData.append(entry)
            else:
                invalidCount += 1
                
        print(f"Found {invalidCount} invalid movesets with >510 EVs, using {len(validSmogonData)} valid movesets")
        smogonData = validSmogonData
        
        # Try to load existing weights
        try:
            counterFinder.loadWeights(args.weights_file)
            print(f"Loaded existing weights from {args.weights_file}")
        except Exception as e:
            print(f"Could not load weights file: {e}")
            print("Starting with default weights")
            
        print(f"Initialized trainer with weights from {args.weights_file}")
        print(f"Results will be saved to {args.results_dir}")
        
        # Ensure results directory exists
        resultsDir = Path(args.results_dir)
        resultsDir.mkdir(exist_ok=True)
        
    except Exception as e:
        print(f"Error initializing: {e}")
        return

    # Run massive training
    print("\n=== Starting Massive Training ===")
    print(f"Configuration:")
    print(f"- {args.batches} batches of {args.batch_size} movesets each")
    print(f"- {args.episodes} episodes per moveset with {args.battles} battles per episode")
    print(f"- Total battles: approximately {args.batches * args.batch_size * args.episodes * args.battles}")
    
    start_time = time.time()
    total_successful = 0
    
    try:
        for batch in range(args.batches):
            print(f"\n=== Batch {batch+1}/{args.batches} ===")
            
            successful = trainBatch(
                counterFinder, repository, smogonData, aliases,
                batch_size=args.batch_size,
                episodes=args.episodes,
                battles_per_episode=args.battles
            )
            
            total_successful += successful
            
            # Save weights periodically
            if (batch + 1) % 5 == 0:
                counterFinder.saveWeights(args.weights_file)
                print(f"Progress saved. Total successful trainings: {total_successful}")
            
            # Track progress
            elapsed = time.time() - start_time
            remaining = (elapsed / (batch + 1)) * (args.batches - (batch + 1))
            print(f"Progress: {batch+1}/{args.batches} batches")
            print(f"Time elapsed: {elapsed/60:.1f} minutes")
            print(f"Estimated time remaining: {remaining/60:.1f} minutes")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving progress...")
        counterFinder.saveWeights(args.weights_file)
    except Exception as e:
        print(f"\nError during training: {e}")
        counterFinder.saveWeights(args.weights_file)
        
    # Save final weights
    counterFinder.saveWeights(args.weights_file)
    
    print("\nMassive training complete!")
    print(f"Total successful trainings: {total_successful}")
    print(f"Time taken: {(time.time() - start_time)/60:.1f} minutes")
    
    # Print final weights
    print("\nFinal feature weights:")
    for feature, weight in counterFinder.weights.items():
        print(f"  {feature}: {weight:.2f}")
    
    print("\nProcess complete!")

if __name__ == "__main__":
    main()