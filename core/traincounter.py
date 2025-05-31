import json
import random
import time
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from pathlib import Path
from aicounter import AICounterFinder
from repository import PokemonRepository, TypeChart
from showdownsim import has_high_winrate, sanitize_team
from models import PokemonConfig

class AICounterTrainer:
    """
    Manages the training process for the AI CounterFinder.
    Handles batch processing of movesets, simulation coordination, and evaluation.
    """
    
    def __init__(self, repository: PokemonRepository, typeChart: TypeChart, 
                 smogon_file: str = "data/smogonMovesets.json",
                 weights_file: str = "ai_counter_weights.json",
                 results_dir: str = "results"):
        self.repository = repository
        self.typeChart = typeChart
        self.counterFinder = AICounterFinder(repository, typeChart)
        
        # Try to load existing weights
        try:
            self.counterFinder.loadWeights(weights_file)
        except Exception as e:
            print(f"Starting with default weights: {e}")
        
        # Load Smogon movesets for training
        try:
            with open(smogon_file, "r") as f:
                self.smogonData = json.load(f)
        except Exception as e:
            print(f"Error loading Smogon data: {e}")
            self.smogonData = []
        
        # Ensure results directory exists
        self.resultsDir = Path(results_dir)
        self.resultsDir.mkdir(exist_ok=True)
        
        # Training metrics
        self.trainingStats = {
            "total_battles": 0,
            "win_rate_history": [],
            "feature_weights_history": [],
            "best_counters": {}
        }
        
        self.weightsFile = weights_file
    
    def trainBatch(self, batch_size: int = 10, episodes: int = 3, 
                   battles_per_episode: int = 5, save_interval: int = 5):
        """
        Train on a batch of random Smogon movesets.
        
        Args:
            batch_size: Number of different Pokémon to train on
            episodes: Number of learning episodes per moveset
            battles_per_episode: Number of simulated battles per episode
            save_interval: Save weights every N movesets
        """
        # Randomly sample movesets for this batch
        if len(self.smogonData) <= batch_size:
            selected_movesets = self.smogonData
        else:
            selected_movesets = random.sample(self.smogonData, batch_size)
            
        print(f"Training on {len(selected_movesets)} movesets...")
        
        for i, moveset in enumerate(selected_movesets):
            pokemon_name = moveset.get("pokemonName", "Unknown")
            print(f"\n[{i+1}/{len(selected_movesets)}] Processing {pokemon_name}")
            
            try:
                # Find suitable input Pokémon - try direct lookup first, then aliases
                try:
                    inputPokemon = self.repository.getPokemonByName(pokemon_name)
                except ValueError:
                    # Try looking up through aliases
                    canonical_name = None
                    for alias_key, alias_value in self.counterFinder.aliases.items():
                        if alias_key == pokemon_name.lower():
                            canonical_name = alias_value
                            break
                    
                    if canonical_name:
                        try:
                            inputPokemon = self.repository.getPokemonByName(canonical_name)
                        except ValueError:
                            print(f"Pokémon '{pokemon_name}' (canonical: '{canonical_name}') not found. Skipping.")
                            continue
                    else:
                        print(f"Pokémon '{pokemon_name}' not found in repository or aliases. Skipping.")
                        continue
                
                # Extract moveset details
                evs = moveset.get("evs", {})
                ivs = moveset.get("ivs", {stat: 31 for stat in ["HP", "Attack", "Defense", 
                                                             "SpecialAttack", "SpecialDefense", "Speed"]})
                nature = moveset.get("nature", "Neutral")
                item = moveset.get("item", "")
                ability = moveset.get("ability", "")
                teraType = moveset.get("teraType", "")
                moves = moveset.get("moves", [])
                
                # Calculate adjusted stats
                inputStats = inputPokemon.getAdjustedStats(evs, ivs, nature)
                
                # Build moveset text for simulations using Showdown-compatible name
                moveset_lines = []
                
                # Convert pokemon name to Showdown format
                showdown_name = pokemon_name
                canonical_name = pokemon_name.lower()
                
                # Look for this pokemon in aliases (where key = showdown name, value = pokemon.json name)
                for showdown_key, pokemon_json_name in self.counterFinder.aliases.items():
                    if pokemon_json_name.lower() == inputPokemon.name.lower():
                        showdown_name = showdown_key
                        break
                
                if item:
                    moveset_lines.append(f"{showdown_name} @ {item}")
                else:
                    moveset_lines.append(showdown_name)
                    
                if ability:
                    moveset_lines.append(f"Ability: {ability}")
                    
                # Format EVs
                ev_parts = []
                for stat, value in evs.items():
                    if value > 0:
                        abbr = {"HP": "HP", "Attack": "Atk", "Defense": "Def", 
                                "SpecialAttack": "SpA", "SpecialDefense": "SpD", "Speed": "Spe"}
                        ev_parts.append(f"{value} {abbr.get(stat, stat)}")
                if ev_parts:
                    moveset_lines.append(f"EVs: {' / '.join(ev_parts)}")
                    
                # Format IVs
                iv_parts = []
                for stat, value in ivs.items():
                    if value < 31:
                        abbr = {"HP": "HP", "Attack": "Atk", "Defense": "Def", 
                                "SpecialAttack": "SpA", "SpecialDefense": "SpD", "Speed": "Spe"}
                        iv_parts.append(f"{value} {abbr.get(stat, stat)}")
                if iv_parts:
                    moveset_lines.append(f"IVs: {' / '.join(iv_parts)}")
                
                if nature:
                    moveset_lines.append(f"{nature} Nature")
                
                if teraType:
                    moveset_lines.append(f"Tera Type: {teraType}")
                
                # Add moves
                for move in moves:
                    moveset_lines.append(f"- {move}")
                
                moveset_str = "\n".join(moveset_lines)
                
                # Find counters
                counterPokemon = self.counterFinder.findCounters(
                    inputPokemon, inputStats, moves, moveset_str
                )
                
                if not counterPokemon:
                    print("No counters found. Skipping.")
                    continue
                
                # Run reinforcement learning
                print(f"Found {len(counterPokemon)} potential counters.")
                
                self.counterFinder.runReinforcementLearning(
                    moveset_str, counterPokemon, 
                    episodes=episodes, battles_per_episode=battles_per_episode
                )
                
                # Save top counters for this Pokémon
                self.trainingStats["best_counters"][pokemon_name] = [
                    {
                        "name": counter.name,
                        "weight": counter.weight,
                        "moveset": self.counterFinder.buildShowdownFormat(counter)
                    }
                    for counter in counterPokemon[:3]  # Save top 3
                ]
                
                # Track weights history
                self.trainingStats["feature_weights_history"].append({
                    "iteration": len(self.trainingStats["feature_weights_history"]) + 1,
                    "weights": self.counterFinder.weights.copy()
                })
                
                # Track win rates
                for key, rate in self.counterFinder.winRates.items():
                    if key not in [entry["matchup"] for entry in self.trainingStats["win_rate_history"]]:
                        self.trainingStats["win_rate_history"].append({
                            "matchup": key,
                            "win_rate": rate
                        })
                
                # Update total battles
                self.trainingStats["total_battles"] += episodes * battles_per_episode
                
                # Save weights periodically
                if (i + 1) % save_interval == 0:
                    self.saveProgress()
                    
            except Exception as e:
                print(f"Error processing {pokemon_name}: {e}")
                continue
        
        # Save final weights
        self.saveProgress()
        print(f"\nTraining complete. Processed {len(selected_movesets)} movesets.")
    
    def saveProgress(self):
        """Save weights and training statistics."""
        # Save weights
        self.counterFinder.saveWeights(self.weightsFile)
        
        # Save training statistics
        stats_file = self.resultsDir / "training_stats.json"
        with open(stats_file, "w") as f:
            json.dump(self.trainingStats, f, indent=2)
            
        print(f"Progress saved. Total battles: {self.trainingStats['total_battles']}")
    
    def evaluateCounter(self, input_moveset: str, counter_config: PokemonConfig, battles: int = 50) -> float:
        """
        Evaluate a counter against an input moveset using Showdown simulations.
        Returns the actual win rate (0.0 to 1.0).
        """
        from showdownsim import simulate_battle, sanitize_team
        import asyncio
        
        counter_text = self.counterFinder.buildShowdownFormat(counter_config)
        
        # Sanitize movesets for simulation
        input_clean = sanitize_team(input_moveset)
        counter_clean = sanitize_team(counter_text)
        
        print(f"Simulating {battles} battles: {counter_config.name} vs {input_moveset.split()[0]}")
        
        try:
            # Run the actual battle simulation
            wins_by_counter, wins_by_challenger = asyncio.run(
                simulate_battle(input_clean, counter_clean, battles)
            )
            
            total_battles = wins_by_counter + wins_by_challenger
            
            if total_battles == 0:
                print(f"No battles completed.")
                return 0.0
            
            # Calculate actual win rate
            actual_win_rate = wins_by_counter / total_battles
            
            # Store result
            key = f"{counter_config.name}_vs_{input_moveset.split()[0]}"
            self.counterFinder.winRates[key] = actual_win_rate
            
            print(f"Result: {wins_by_counter}/{total_battles} wins = {actual_win_rate:.3f} win rate")
            
            return actual_win_rate
            
        except Exception as e:
            print(f"Error in battle simulation: {e}")
            return 0.0
            
    def runMassiveTraining(self, total_batches: int = 10, batch_size: int = 10, 
                           episodes: int = 3, battles_per_episode: int = 5):
        """
        Run a large-scale training operation over multiple batches.
        
        Args:
            total_batches: Number of batches to train on
            batch_size: Number of movesets per batch
            episodes: Number of episodes per moveset
            battles_per_episode: Number of battles per episode
        """
        print(f"Starting massive training: {total_batches} batches of {batch_size} movesets each")
        print(f"- {episodes} episodes per moveset")
        print(f"- {battles_per_episode} battles per episode")
        print(f"- Total battles: approximately {total_batches * batch_size * episodes * battles_per_episode}")
        
        start_time = time.time()
        
        for batch in range(total_batches):
            print(f"\n=== Batch {batch+1}/{total_batches} ===")
            self.trainBatch(
                batch_size=batch_size,
                episodes=episodes,
                battles_per_episode=battles_per_episode
            )
            
            # Track progress
            elapsed = time.time() - start_time
            remaining = (elapsed / (batch + 1)) * (total_batches - (batch + 1))
            print(f"Progress: {batch+1}/{total_batches} batches")
            print(f"Time elapsed: {elapsed/60:.1f} minutes")
            print(f"Estimated time remaining: {remaining/60:.1f} minutes")
            
        print("\nMassive training complete!")
        print(f"Total battles: {self.trainingStats['total_battles']}")
        print(f"Time taken: {(time.time() - start_time)/60:.1f} minutes")
        
        # Print final weights
        print("\nFinal feature weights:")
        for feature, weight in self.counterFinder.weights.items():
            print(f"  {feature}: {weight:.2f}")
    
    def evaluateOnTestSet(self, test_file: str = None, num_samples: int = 50):
        """
        Evaluate current model on a test set of movesets.
        
        Args:
            test_file: JSON file with test movesets (uses random Smogon if None)
            num_samples: Number of random samples to use if no test file
        """
        if test_file:
            # Load test set
            try:
                with open(test_file, "r") as f:
                    test_movesets = json.load(f)
            except Exception as e:
                print(f"Error loading test file: {e}")
                return
        else:
            # Use random samples from Smogon data
            if len(self.smogonData) <= num_samples:
                test_movesets = self.smogonData
            else:
                test_movesets = random.sample(self.smogonData, num_samples)
        
        results = {
            "total_movesets": len(test_movesets),
            "successful_counter_searches": 0,
            "average_win_rate": 0.0,
            "win_rates": [],
            "top_counters": {},
            "feature_weights": self.counterFinder.weights.copy()
        }
        
        print(f"Evaluating on {len(test_movesets)} test movesets...")
        total_win_rate = 0.0
        win_count = 0
        
        for i, moveset in enumerate(tqdm(test_movesets)):
            pokemon_name = moveset.get("pokemonName", "Unknown")
            
            try:
                # Find suitable input Pokémon - try direct lookup first, then aliases
                try:
                    inputPokemon = self.repository.getPokemonByName(pokemon_name)
                except ValueError:
                    # Try looking up through aliases
                    canonical_name = None
                    for alias_key, alias_value in self.counterFinder.aliases.items():
                        if alias_key == pokemon_name.lower():
                            canonical_name = alias_value
                            break
                    
                    if canonical_name:
                        try:
                            inputPokemon = self.repository.getPokemonByName(canonical_name)
                        except ValueError:
                            print(f"Pokémon '{pokemon_name}' (canonical: '{canonical_name}') not found. Skipping.")
                            continue
                    else:
                        print(f"Pokémon '{pokemon_name}' not found in repository or aliases. Skipping.")
                        continue
                
                # Extract moveset details
                evs = moveset.get("evs", {})
                ivs = moveset.get("ivs", {stat: 31 for stat in ["HP", "Attack", "Defense", 
                                                             "SpecialAttack", "SpecialDefense", "Speed"]})
                nature = moveset.get("nature", "Neutral")
                item = moveset.get("item", "")
                ability = moveset.get("ability", "")
                teraType = moveset.get("teraType", "")
                moves = moveset.get("moves", [])
                
                # Calculate adjusted stats
                inputStats = inputPokemon.getAdjustedStats(evs, ivs, nature)
                
                # Build moveset text for simulations using Showdown-compatible name
                moveset_lines = []
                
                # Convert pokemon name to Showdown format
                showdown_name = pokemon_name
                
                # Look for this pokemon in aliases (where key = showdown name, value = pokemon.json name)
                for showdown_key, pokemon_json_name in self.counterFinder.aliases.items():
                    if pokemon_json_name.lower() == inputPokemon.name.lower():
                        showdown_name = showdown_key
                        break
                
                if item:
                    moveset_lines.append(f"{showdown_name} @ {item}")
                else:
                    moveset_lines.append(showdown_name)
                    
                if ability:
                    moveset_lines.append(f"Ability: {ability}")
                    
                # Format EVs
                ev_parts = []
                for stat, value in evs.items():
                    if value > 0:
                        abbr = {"HP": "HP", "Attack": "Atk", "Defense": "Def", 
                                "SpecialAttack": "SpA", "SpecialDefense": "SpD", "Speed": "Spe"}
                        ev_parts.append(f"{value} {abbr.get(stat, stat)}")
                if ev_parts:
                    moveset_lines.append(f"EVs: {' / '.join(ev_parts)}")
                    
                # Format IVs
                iv_parts = []
                for stat, value in ivs.items():
                    if value < 31:
                        abbr = {"HP": "HP", "Attack": "Atk", "Defense": "Def", 
                                "SpecialAttack": "SpA", "SpecialDefense": "SpD", "Speed": "Spe"}
                        iv_parts.append(f"{value} {abbr.get(stat, stat)}")
                if iv_parts:
                    moveset_lines.append(f"IVs: {' / '.join(iv_parts)}")
                
                if nature:
                    moveset_lines.append(f"{nature} Nature")
                
                if teraType:
                    moveset_lines.append(f"Tera Type: {teraType}")
                
                # Add moves
                for move in moves:
                    moveset_lines.append(f"- {move}")
                
                moveset_str = "\n".join(moveset_lines)
                
                # Find counters
                counterPokemon = self.counterFinder.findCounters(
                    inputPokemon, inputStats, moves, moveset_str
                )
                
                if not counterPokemon:
                    print(f"No counters found for {pokemon_name}. Skipping.")
                    continue
                
                results["successful_counter_searches"] += 1
                
                # Test top counter with battle simulations
                best_counter = counterPokemon[0]
                win_rate = self.evaluateCounter(moveset_str, best_counter, battles=20)
                
                # Track results
                results["win_rates"].append({
                    "pokemon": pokemon_name,
                    "counter": best_counter.name,
                    "win_rate": win_rate
                })
                
                # Save top counters for this Pokémon
                results["top_counters"][pokemon_name] = [
                    {
                        "name": counter.name,
                        "weight": counter.weight,
                        "win_rate": win_rate if counter is best_counter else None
                    }
                    for counter in counterPokemon[:3]  # Save top 3
                ]
                
                # Update totals
                if win_rate >= 0.5:  # Consider a successful counter if win rate >= 50%
                    win_count += 1
                    total_win_rate += win_rate
                
            except Exception as e:
                print(f"Error evaluating {pokemon_name}: {e}")
                continue
        
        # Calculate final metrics
        if win_count > 0:
            results["average_win_rate"] = total_win_rate / win_count
        
        results["success_rate"] = win_count / results["total_movesets"]
        
        # Save evaluation results
        eval_file = self.resultsDir / "evaluation_results.json"
        with open(eval_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\nEvaluation complete!")
        print(f"Tested {results['total_movesets']} movesets")
        print(f"Successfully found counters for {results['successful_counter_searches']} movesets")
        print(f"Success rate: {results['success_rate']:.2f}")
        print(f"Average win rate: {results['average_win_rate']:.2f}")
        
        return results
        
    def getTopCountersForPokemon(self, pokemon_name: str, num_counters: int = 5):
        """
        Get the best counters for a specific Pokémon.
        
        Args:
            pokemon_name: Name of the Pokémon to find counters for
            num_counters: Number of counters to return
            
        Returns:
            List of counter Pokémon configurations
        """
        # Find moveset for the Pokémon
        moveset = None
        for entry in self.smogonData:
            if entry.get("pokemonName", "").lower() == pokemon_name.lower():
                moveset = entry
                break
        
        if not moveset:
            print(f"No moveset found for {pokemon_name}")
            return []
        
        try:
            # Find suitable input Pokémon - try direct lookup first, then aliases
            try:
                inputPokemon = self.repository.getPokemonByName(pokemon_name)
            except ValueError:
                # Try looking up through aliases
                canonical_name = None
                for alias_key, alias_value in self.counterFinder.aliases.items():
                    if alias_key == pokemon_name.lower():
                        canonical_name = alias_value
                        break
                
                if canonical_name:
                    try:
                        inputPokemon = self.repository.getPokemonByName(canonical_name)
                    except ValueError:
                        print(f"Pokémon '{pokemon_name}' (canonical: '{canonical_name}') not found.")
                        return []
                else:
                    print(f"Pokémon '{pokemon_name}' not found in repository or aliases.")
                    return []
            
            # Extract moveset details
            evs = moveset.get("evs", {})
            ivs = moveset.get("ivs", {stat: 31 for stat in ["HP", "Attack", "Defense", 
                                                         "SpecialAttack", "SpecialDefense", "Speed"]})
            nature = moveset.get("nature", "Neutral")
            item = moveset.get("item", "")
            ability = moveset.get("ability", "")
            teraType = moveset.get("teraType", "")
            moves = moveset.get("moves", [])
            
            # Calculate adjusted stats
            inputStats = inputPokemon.getAdjustedStats(evs, ivs, nature)
            
            # Build moveset text using Showdown-compatible name
            showdown_name = pokemon_name
            
            # Look for this pokemon in aliases (where key = showdown name, value = pokemon.json name)
            for showdown_key, pokemon_json_name in self.counterFinder.aliases.items():
                if pokemon_json_name.lower() == inputPokemon.name.lower():
                    showdown_name = showdown_key
                    break
                    
            moveset_str = self._buildMoveset(
                showdown_name, evs, ivs, nature, item, ability, teraType, moves
            )
            
            # Find counters
            counterPokemon = self.counterFinder.findCounters(
                inputPokemon, inputStats, moves, moveset_str
            )
            
            return counterPokemon[:num_counters]
            
        except Exception as e:
            print(f"Error finding counters for {pokemon_name}: {e}")
            return []
    
    def _buildMoveset(self, pokemon_name, evs, ivs, nature, item, ability, teraType, moves):
        """
        Helper method to build a complete moveset text string in Showdown format.
        """
        lines = []
        if item:
            lines.append(f"{pokemon_name} @ {item}")
        else:
            lines.append(pokemon_name)
            
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