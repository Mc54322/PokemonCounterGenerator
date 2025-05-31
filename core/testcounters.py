#!/usr/bin/env python3
"""
Efficient Counter Evaluation System

Optimized for rapid evaluation with controlled testing parameters:
- Limited Pokemon subset for fair comparison
- Minimal battles per counter (10-20) for speed
- Aggressive resource management
- Statistical sampling approach
- Progress tracking and early termination options
"""

import json
import argparse
import gc
import time
import psutil
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
from tqdm import tqdm
import asyncio
import numpy as np

# Import both counter systems
from aicounter import AICounterFinder
from counterfinder import CounterFinder
from repository import PokemonRepository, TypeChart
from showdownsim import simulate_battle, sanitize_team

@dataclass
class QuickBattleResult:
    """Simplified structure for quick battle results"""
    pokemon_name: str
    counter_name: str
    counter_weight: float
    battles_completed: int
    win_rate: float
    evaluation_time: float

@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters"""
    max_pokemon: int = 50  # Limit total Pokemon tested
    battles_per_counter: int = 10  # Minimal battles for speed
    max_counters_per_pokemon: int = 2  # Reduce counter testing
    timeout_per_battle_set: int = 30  # Strict timeout
    memory_limit_mb: int = 2000  # Lower memory limit

class EfficientCounterEvaluator:
    """Streamlined evaluator focused on speed and reliability"""
    
    def __init__(self, pokemon_repo: PokemonRepository, type_chart: TypeChart, 
                 use_ai_counter: bool = True, weights_file: Optional[str] = None, 
                 config: EvaluationConfig = None):
        self.pokemon_repo = pokemon_repo
        self.type_chart = type_chart
        self.use_ai_counter = use_ai_counter
        self.config = config or EvaluationConfig()
        
        # Load aliases mapping for Smogon compatibility
        self.aliases = {}
        try:
            aliases_file = Path(__file__).resolve().parent / "data" / "aliases.json"
            with open(aliases_file, 'r', encoding='utf-8') as f:
                raw_aliases = json.load(f)
            # Store as showdown_name -> pokemon_json_name (lowercase for consistency)
            self.aliases = {k.lower(): v.lower() for k, v in raw_aliases.items()}
            print(f"✓ Loaded {len(self.aliases)} aliases for Smogon compatibility")
        except Exception as e:
            print(f"⚠ Could not load aliases.json: {e}")
            self.aliases = {}
        
        # Initialize counter finder
        if use_ai_counter:
            self.counter_finder = AICounterFinder(pokemon_repo, type_chart)
            if weights_file and Path(weights_file).exists():
                try:
                    self.counter_finder.loadWeights(weights_file)
                    print(f"✓ Loaded AI weights from {weights_file}")
                except Exception as e:
                    print(f"⚠ Failed to load weights: {e}")
        else:
            self.counter_finder = CounterFinder(pokemon_repo, type_chart)
        
        # Results storage
        self.results: List[QuickBattleResult] = []
        self.failed_evaluations = []
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return psutil.Process().memory_info().rss / 1024 / 1024
    
    def aggressive_cleanup(self):
        """Force cleanup and brief pause"""
        gc.collect()
        gc.collect()
        time.sleep(0.1)
    
    def get_showdown_name(self, pokemon_name: str) -> str:
        """Convert Pokemon name to Smogon/Showdown compatible format using aliases"""
        pokemon_name_lower = pokemon_name.lower()
        
        # Check if any alias value matches this Pokemon name
        for showdown_key, pokemon_json_name in self.aliases.items():
            if pokemon_json_name == pokemon_name_lower:
                return showdown_key
        
        # If no alias found, return original name
        return pokemon_name
    
    def generate_short_nickname(self, pokemon_name: str) -> str:
        """Generate a short nickname (≤18 characters) for Pokemon to avoid Showdown errors"""
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
    
    def quick_battle_test(self, opponent_team: str, counter_team: str) -> Tuple[int, int, float]:
        """Minimal battle test with strict resource management"""
        start_time = time.time()
        
        try:
            # Sanitize teams
            opponent_clean = sanitize_team(opponent_team)
            counter_clean = sanitize_team(counter_team)
            
            # Create isolated event loop for this test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Single battle set with strict timeout
                future = asyncio.wait_for(
                    simulate_battle(opponent_clean, counter_clean, self.config.battles_per_counter),
                    timeout=self.config.timeout_per_battle_set
                )
                counter_wins, opponent_wins = loop.run_until_complete(future)
                
                return counter_wins, opponent_wins, time.time() - start_time
                
            except (asyncio.TimeoutError, asyncio.CancelledError):
                return 0, 0, time.time() - start_time
            except Exception:
                return 0, 0, time.time() - start_time
            finally:
                loop.close()
                
        except Exception:
            return 0, 0, time.time() - start_time
    
    def validate_ev_spread(self, evs: Dict[str, int]) -> bool:
        """Validate that EV spread doesn't exceed 510 total and individual stats don't exceed 252"""
        if not evs:
            return True
            
        total_evs = sum(evs.values())
        if total_evs > 510:
            return False
            
        for stat, value in evs.items():
            if value > 252 or value < 0:
                return False
                
        return True
    
    def fix_ev_spread(self, evs: Dict[str, int]) -> Dict[str, int]:
        """Fix invalid EV spreads by proportionally scaling them down to 510"""
        if not evs:
            return evs
            
        # Cap individual stats at 252
        fixed_evs = {stat: min(252, max(0, value)) for stat, value in evs.items()}
        
        total_evs = sum(fixed_evs.values())
        
        # If still over 510, scale proportionally
        if total_evs > 510:
            scale_factor = 510 / total_evs
            scaled_evs = {}
            remaining = 510
            
            # Scale down each stat, ensuring we don't go over 510 total
            for stat, value in fixed_evs.items():
                if value > 0:
                    scaled_value = int(value * scale_factor)
                    scaled_evs[stat] = min(scaled_value, remaining)
                    remaining -= scaled_evs[stat]
                else:
                    scaled_evs[stat] = 0
            
            return scaled_evs
            
        return fixed_evs

    def build_basic_moveset_text(self, moveset_data: Dict) -> str:
        """Build minimal moveset text for simulation with Smogon-compatible names"""
        pokemon_name = moveset_data['pokemonName']
        
        # Convert to Smogon-compatible name using aliases
        showdown_name = self.get_showdown_name(pokemon_name)
        
        # Generate short nickname to avoid Showdown length issues
        nickname = self.generate_short_nickname(showdown_name)
        
        lines = []
        
        # Add Pokemon name with item if available
        item = moveset_data.get('item', '')
        if item:
            lines.append(f"{nickname} ({showdown_name}) @ {item}")
        else:
            lines.append(f"{nickname} ({showdown_name})")
        
        # Add essential components only
        if ability := moveset_data.get('ability'):
            lines.append(f"Ability: {ability}")
        
        # Add EVs if present and non-zero (with validation and fixing)
        evs = moveset_data.get('evs', {})
        if evs and any(v > 0 for v in evs.values()):
            # Validate and fix EV spread
            if not self.validate_ev_spread(evs):
                evs = self.fix_ev_spread(evs)
                print(f"⚠ Fixed invalid EV spread for {pokemon_name}: {moveset_data.get('evs', {})} -> {evs}")
            
            ev_parts = []
            stat_abbrev = {
                "HP": "HP", "Attack": "Atk", "Defense": "Def",
                "SpecialAttack": "SpA", "SpecialDefense": "SpD", "Speed": "Spe"
            }
            for stat, value in evs.items():
                if value > 0:
                    ev_parts.append(f"{value} {stat_abbrev.get(stat, stat)}")
            if ev_parts:
                lines.append(f"EVs: {' / '.join(ev_parts)}")
        
        # Add IVs if present and not all 31
        ivs = moveset_data.get('ivs', {})
        if ivs and any(v < 31 for v in ivs.values()):
            iv_parts = []
            stat_abbrev = {
                "HP": "HP", "Attack": "Atk", "Defense": "Def", 
                "SpecialAttack": "SpA", "SpecialDefense": "SpD", "Speed": "Spe"
            }
            for stat, value in ivs.items():
                if value < 31:
                    iv_parts.append(f"{value} {stat_abbrev.get(stat, stat)}")
            if iv_parts:
                lines.append(f"IVs: {' / '.join(iv_parts)}")
        
        # Add nature
        if nature := moveset_data.get('nature'):
            if nature != 'Neutral':
                lines.append(f"{nature} Nature")
        
        # Add Tera Type
        if tera_type := moveset_data.get('teraType'):
            lines.append(f"Tera Type: {tera_type}")
        
        # Add moves
        for move in moveset_data.get('moves', []):
            lines.append(f"- {move}")
        
        return "\n".join(lines)
    
    def build_basic_counter_text(self, counter_config) -> str:
        """Build minimal counter text for simulation with Smogon-compatible names"""
        if self.use_ai_counter:
            # Use the AI counter's built-in format method if available
            if hasattr(self.counter_finder, 'buildShowdownFormat'):
                return self.counter_finder.buildShowdownFormat(counter_config)
        
        # Build format manually with Smogon compatibility
        counter_name = counter_config.name
        showdown_name = self.get_showdown_name(counter_name)
        nickname = self.generate_short_nickname(showdown_name)
        
        lines = []
        
        # Add Pokemon name with item
        if hasattr(counter_config, 'item') and counter_config.item:
            lines.append(f"{nickname} ({showdown_name}) @ {counter_config.item}")
        else:
            lines.append(f"{nickname} ({showdown_name})")
        
        if hasattr(counter_config, 'ability') and counter_config.ability:
            lines.append(f"Ability: {counter_config.ability}")
        
        # Add EVs (with validation and fixing)
        if hasattr(counter_config, 'evs') and counter_config.evs:
            evs = counter_config.evs
            
            # Validate and fix EV spread
            if not self.validate_ev_spread(evs):
                evs = self.fix_ev_spread(evs)
                print(f"⚠ Fixed invalid EV spread for counter {counter_config.name}: {counter_config.evs} -> {evs}")
            
            ev_parts = []
            stat_abbrev = {
                "HP": "HP", "Attack": "Atk", "Defense": "Def",
                "SpecialAttack": "SpA", "SpecialDefense": "SpD", "Speed": "Spe"
            }
            for stat, value in evs.items():
                if value > 0:
                    ev_parts.append(f"{value} {stat_abbrev.get(stat, stat)}")
            if ev_parts:
                lines.append(f"EVs: {' / '.join(ev_parts)}")
        
        # Add IVs if not all 31
        if hasattr(counter_config, 'ivs') and counter_config.ivs:
            iv_parts = []
            stat_abbrev = {
                "HP": "HP", "Attack": "Atk", "Defense": "Def",
                "SpecialAttack": "SpA", "SpecialDefense": "SpD", "Speed": "Spe"
            }
            for stat, value in counter_config.ivs.items():
                if value < 31:
                    iv_parts.append(f"{value} {stat_abbrev.get(stat, stat)}")
            if iv_parts:
                lines.append(f"IVs: {' / '.join(iv_parts)}")
        
        if hasattr(counter_config, 'nature') and counter_config.nature:
            lines.append(f"{counter_config.nature} Nature")
        
        if hasattr(counter_config, 'teraType') and counter_config.teraType:
            lines.append(f"Tera Type: {counter_config.teraType}")
        
        if hasattr(counter_config, 'moves') and counter_config.moves:
            for move in counter_config.moves:
                lines.append(f"- {move}")
        
        return "\n".join(lines)
    
    def evaluate_single_pokemon(self, moveset_data: Dict) -> List[QuickBattleResult]:
        """Evaluate counters for a single Pokemon with minimal overhead"""
        pokemon_name = moveset_data['pokemonName']
        start_time = time.time()
        
        try:
            # Check for invalid EV spread early and skip if unfixable
            evs = moveset_data.get('evs', {stat: 0 for stat in ['HP', 'Attack', 'Defense', 'SpecialAttack', 'SpecialDefense', 'Speed']})
            if evs and not self.validate_ev_spread(evs):
                total_evs = sum(evs.values())
                if total_evs > 510:
                    print(f"⚠ Skipping {pokemon_name} due to invalid EV spread (total: {total_evs})")
                    self.failed_evaluations.append({
                        'pokemon': pokemon_name,
                        'error': f'Invalid EV spread: total {total_evs} > 510',
                        'time': time.time() - start_time
                    })
                    return []
            
            # Get Pokemon from repository
            try:
                input_pokemon = self.pokemon_repo.getPokemonByName(pokemon_name)
            except ValueError:
                # Quick alias lookup - search for canonical name in aliases
                canonical_name = None
                for alias_key, alias_value in self.aliases.items():
                    if alias_key == pokemon_name.lower():
                        canonical_name = alias_value
                        break
                
                if canonical_name:
                    try:
                        input_pokemon = self.pokemon_repo.getPokemonByName(canonical_name)
                    except ValueError:
                        return []
                else:
                    return []
            
            # Calculate basic stats - use fixed EVs if needed
            if not self.validate_ev_spread(evs):
                evs = self.fix_ev_spread(evs)
                
            ivs = moveset_data.get('ivs', {stat: 31 for stat in ['HP', 'Attack', 'Defense', 'SpecialAttack', 'SpecialDefense', 'Speed']})
            nature = moveset_data.get('nature', 'Neutral')
            moves = moveset_data.get('moves', [])
            
            input_stats = input_pokemon.getAdjustedStats(evs, ivs, nature)
            
            # Build minimal moveset text with Smogon compatibility (this will handle EV fixing internally)
            moveset_text = self.build_basic_moveset_text(moveset_data)
            
            # Find counters
            if self.use_ai_counter:
                counter_pokemon = self.counter_finder.findCounters(
                    input_pokemon, input_stats, moves, moveset_text
                )
            else:
                counter_pokemon = self.counter_finder.findCounters(
                    input_pokemon, input_stats, moves
                )
            
            if not counter_pokemon:
                return []
            
            # Test only top N counters
            top_counters = counter_pokemon[:self.config.max_counters_per_pokemon]
            results = []
            
            for counter in top_counters:
                # Quick memory check
                if self.get_memory_usage() > self.config.memory_limit_mb:
                    self.aggressive_cleanup()
                
                # Build counter text with Smogon compatibility (this will handle EV fixing internally)
                counter_text = self.build_basic_counter_text(counter)
                
                # Run quick battle test
                counter_wins, opponent_wins, duration = self.quick_battle_test(
                    moveset_text, counter_text
                )
                
                total_battles = counter_wins + opponent_wins
                win_rate = counter_wins / total_battles if total_battles > 0 else 0.0
                
                result = QuickBattleResult(
                    pokemon_name=pokemon_name,
                    counter_name=counter.name,
                    counter_weight=float(counter.weight),
                    battles_completed=total_battles,
                    win_rate=float(win_rate),
                    evaluation_time=duration
                )
                
                results.append(result)
                
                # Quick cleanup between counters
                self.aggressive_cleanup()
            
            return results
            
        except Exception as e:
            self.failed_evaluations.append({
                'pokemon': pokemon_name,
                'error': str(e),
                'time': time.time() - start_time
            })
            return []
    
    def calculate_pokemon_strength(self, moveset_data: Dict) -> float:
        """Calculate a rough strength score for stratified sampling"""
        try:
            pokemon_name = moveset_data['pokemonName']
            
            # Get Pokemon base stats
            try:
                pokemon = self.pokemon_repo.getPokemonByName(pokemon_name)
            except ValueError:
                # Try alias lookup
                canonical_name = None
                for alias_key, alias_value in self.aliases.items():
                    if alias_key == pokemon_name.lower():
                        canonical_name = alias_value
                        break
                
                if canonical_name:
                    try:
                        pokemon = self.pokemon_repo.getPokemonByName(canonical_name)
                    except ValueError:
                        return 400  # Default medium strength
                else:
                    return 400
            
            # Calculate base stat total
            base_stats = pokemon.baseStats
            bst = sum(base_stats.values())
            
            # Get competitive tier if available (higher tier = stronger)
            tier_weights = {
                'Uber': 100,
                'OU': 80, 
                'UU': 60,
                'RU': 40,
                'NU': 20,
                'PU': 10,
                'Unrated': 30
            }
            tier_bonus = tier_weights.get(moveset_data.get('pokemonTier', 'Unrated'), 30)
            
            # Combine BST and tier for strength score
            strength_score = bst + tier_bonus
            
            return strength_score
            
        except Exception:
            return 400  # Default medium strength if calculation fails
    
    def stratified_sample_pokemon(self, full_dataset: List[Dict], sample_size: int) -> List[Dict]:
        """Sample Pokemon ensuring balanced strength distribution"""
        
        print("📊 Calculating Pokemon strength distribution...")
        
        # Calculate strength for all Pokemon
        pokemon_with_strength = []
        for moveset in full_dataset:
            strength = self.calculate_pokemon_strength(moveset)
            pokemon_with_strength.append((moveset, strength))
        
        # Sort by strength
        pokemon_with_strength.sort(key=lambda x: x[1])
        
        # Define strength tiers
        total_pokemon = len(pokemon_with_strength)
        tier_size = total_pokemon // 4
        
        weak_tier = pokemon_with_strength[:tier_size]
        low_tier = pokemon_with_strength[tier_size:tier_size*2]  
        mid_tier = pokemon_with_strength[tier_size*2:tier_size*3]
        strong_tier = pokemon_with_strength[tier_size*3:]
        
        print(f"📈 Strength distribution:")
        print(f"   Weak tier: {len(weak_tier)} Pokemon (BST: {weak_tier[0][1]:.0f}-{weak_tier[-1][1]:.0f})")
        print(f"   Low tier: {len(low_tier)} Pokemon (BST: {low_tier[0][1]:.0f}-{low_tier[-1][1]:.0f})")
        print(f"   Mid tier: {len(mid_tier)} Pokemon (BST: {mid_tier[0][1]:.0f}-{mid_tier[-1][1]:.0f})")
        print(f"   Strong tier: {len(strong_tier)} Pokemon (BST: {strong_tier[0][1]:.0f}-{strong_tier[-1][1]:.0f})")
        
        # Stratified sampling - proportional from each tier
        samples_per_tier = sample_size // 4
        remaining_samples = sample_size % 4
        
        sampled_pokemon = []
        
        # Sample from each tier
        for tier_name, tier_data in [("weak", weak_tier), ("low", low_tier), 
                                   ("mid", mid_tier), ("strong", strong_tier)]:
            tier_sample_size = samples_per_tier
            if remaining_samples > 0:
                tier_sample_size += 1
                remaining_samples -= 1
            
            if len(tier_data) >= tier_sample_size:
                tier_sample = random.sample(tier_data, tier_sample_size)
            else:
                tier_sample = tier_data
            
            sampled_pokemon.extend([pokemon for pokemon, strength in tier_sample])
            print(f"   Sampled {len(tier_sample)} from {tier_name} tier")
        
        return sampled_pokemon
    
    def tier_based_sample_pokemon(self, full_dataset: List[Dict], sample_size: int) -> List[Dict]:
        """Sample Pokemon based on competitive tiers for maximum fairness"""
        
        print("🏆 Sampling by competitive tiers...")
        
        # Group by tiers
        tier_groups = {
            'Uber': [],
            'OU': [],
            'UU': [], 
            'RU': [],
            'NU': [],
            'PU': [],
            'Unrated': []
        }
        
        for moveset in full_dataset:
            tier = moveset.get('pokemonTier', 'Unrated')
            if tier in tier_groups:
                tier_groups[tier].append(moveset)
            else:
                tier_groups['Unrated'].append(moveset)
        
        # Print tier distribution
        print("🏆 Tier distribution:")
        for tier, pokemon_list in tier_groups.items():
            if pokemon_list:
                print(f"   {tier}: {len(pokemon_list)} Pokemon")
        
        # Sample proportionally from viable tiers (skip very weak tiers for fair comparison)
        viable_tiers = ['Uber', 'OU', 'UU', 'RU']
        viable_pokemon = []
        for tier in viable_tiers:
            viable_pokemon.extend(tier_groups[tier])
        
        # If not enough viable Pokemon, add some from lower tiers
        if len(viable_pokemon) < sample_size:
            viable_pokemon.extend(tier_groups['NU'][:sample_size//4])
            viable_pokemon.extend(tier_groups['Unrated'][:sample_size//4])
        
        # Sample from viable Pokemon
        if len(viable_pokemon) >= sample_size:
            sampled = random.sample(viable_pokemon, sample_size)
        else:
            # Use all viable and fill with random from others
            sampled = viable_pokemon
            remaining_needed = sample_size - len(viable_pokemon)
            others = tier_groups['NU'] + tier_groups['PU'] + tier_groups['Unrated']
            if others and remaining_needed > 0:
                additional = random.sample(others, min(remaining_needed, len(others)))
                sampled.extend(additional)
        
        print(f"📦 Tier-based sample: {len(sampled)} competitive Pokemon selected")
        return sampled
            
    def run_efficient_evaluation(self, dataset_path: str, sample_size: Optional[int] = None, 
                               sampling_strategy: str = 'random', seed: Optional[int] = None,
                               output_name: Optional[str] = None) -> str:
        """Run efficient evaluation with controlled parameters and consistent sampling"""
        
        # Set random seed for reproducible results
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            print(f"🎲 Random seed set to: {seed}")
        
        print(f"🚀 Starting EFFICIENT counter evaluation")
        print(f"📊 Configuration:")
        print(f"   Max Pokemon: {self.config.max_pokemon}")
        print(f"   Battles per counter: {self.config.battles_per_counter}")
        print(f"   Max counters per Pokemon: {self.config.max_counters_per_pokemon}")
        print(f"   Timeout per battle set: {self.config.timeout_per_battle_set}s")
        print(f"   Memory limit: {self.config.memory_limit_mb}MB")
        print(f"   Sampling strategy: {sampling_strategy}")
        print(f"   Seed: {seed if seed is not None else 'None (random)'}")
        print(f"   Aliases loaded: {len(self.aliases)}")
        print(f"   Output file: {output_name}.json" if output_name else "Auto-generated name")
        
        # Load and sample dataset
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                full_dataset = json.load(f)
            
            # Determine sample size
            sample_size = sample_size or min(self.config.max_pokemon, len(full_dataset))
            
            if len(full_dataset) > sample_size:
                if sampling_strategy == 'stratified':
                    movesets = self.stratified_sample_pokemon(full_dataset, sample_size)
                elif sampling_strategy == 'tier':
                    movesets = self.tier_based_sample_pokemon(full_dataset, sample_size)
                else:  # random
                    movesets = random.sample(full_dataset, sample_size)
                    print(f"📦 Random sample: {sample_size} movesets from {len(full_dataset)} total")
            else:
                movesets = full_dataset
                print(f"📦 Using all {len(movesets)} movesets")
            
            # Log sampled Pokemon names for verification
            pokemon_names = [m.get('pokemonName', 'Unknown') for m in movesets]
            print(f"📋 Selected Pokemon: {', '.join(pokemon_names[:10])}" + 
                  (f" ... and {len(pokemon_names)-10} more" if len(pokemon_names) > 10 else ""))
                
        except Exception as e:
            print(f"✗ Failed to load dataset: {e}")
            return ""
        
        # Process movesets with progress tracking
        start_time = time.time()
        successful_evaluations = 0
        
        for i, moveset in enumerate(tqdm(movesets, desc="Evaluating Pokemon")):
            pokemon_name = moveset.get('pokemonName', f'Unknown_{i}')
            
            # Memory management
            if i % 5 == 0:
                current_memory = self.get_memory_usage()
                if current_memory > self.config.memory_limit_mb * 0.8:
                    print(f"🧹 Memory cleanup at {current_memory:.0f}MB")
                    self.aggressive_cleanup()
            
            # Evaluate this Pokemon
            results = self.evaluate_single_pokemon(moveset)
            
            if results:
                self.results.extend(results)
                successful_evaluations += 1
            
            # Progress update every 10 Pokemon
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(movesets) - i - 1) / rate if rate > 0 else 0
                current_memory = self.get_memory_usage()
                
                print(f"📈 Progress: {i+1}/{len(movesets)} | "
                      f"Rate: {rate:.1f} Pokemon/sec | "
                      f"ETA: {eta/60:.1f}min | "
                      f"Memory: {current_memory:.0f}MB | "
                      f"Success: {successful_evaluations}")
        
        # Save results
        results_file = self.save_results(output_name)
        
        # Print summary
        self.print_summary(time.time() - start_time, successful_evaluations, len(movesets))
        
        return results_file
    
    def save_results(self, output_name: Optional[str] = None) -> str:
        """Save evaluation results"""
        timestamp = int(time.time())
        counter_type = "ai" if self.use_ai_counter else "rule"
        
        # Use custom output name or generate default
        if output_name:
            results_file = f"{output_name}.json"
        else:
            results_file = f"efficient_eval_{counter_type}_{timestamp}.json"
        
        summary_data = {
            'config': asdict(self.config),
            'counter_type': 'AI-Enhanced' if self.use_ai_counter else 'Rule-Based',
            'aliases_loaded': len(self.aliases),
            'total_evaluations': len(self.results),
            'failed_evaluations': len(self.failed_evaluations),
            'results': [asdict(result) for result in self.results],
            'failures': self.failed_evaluations,
            'summary_stats': self.generate_summary_stats()
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"💾 Results saved to {results_file}")
        return results_file
    
    def generate_summary_stats(self) -> Dict:
        """Generate summary statistics"""
        if not self.results:
            return {}
        
        df = pd.DataFrame([asdict(result) for result in self.results])
        
        return {
            'total_battles': int(df['battles_completed'].sum()),
            'average_win_rate': float(df['win_rate'].mean()),
            'median_win_rate': float(df['win_rate'].median()),
            'win_rate_std': float(df['win_rate'].std()),
            'average_evaluation_time': float(df['evaluation_time'].mean()),
            'total_evaluation_time': float(df['evaluation_time'].sum()),
            'unique_pokemon': int(df['pokemon_name'].nunique()),
            'unique_counters': int(df['counter_name'].nunique()),
            'success_rate': len(self.results) / (len(self.results) + len(self.failed_evaluations)) if self.results else 0
        }
    
    def print_summary(self, total_time: float, successful: int, attempted: int):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("📋 EFFICIENT EVALUATION SUMMARY")
        print("="*60)
        
        stats = self.generate_summary_stats()
        
        print(f"Pokemon Evaluated: {successful}/{attempted} ({successful/attempted*100:.1f}%)")
        print(f"Total Evaluations: {len(self.results)}")
        print(f"Failed Evaluations: {len(self.failed_evaluations)}")
        print(f"Total Time: {total_time/60:.1f} minutes")
        print(f"Rate: {attempted/total_time:.1f} Pokemon/sec")
        print(f"Aliases Used: {len(self.aliases)}")
        print()
        print(f"Battle Statistics:")
        print(f"  Total Battles: {stats.get('total_battles', 0)}")
        print(f"  Average Win Rate: {stats.get('average_win_rate', 0):.3f}")
        print(f"  Median Win Rate: {stats.get('median_win_rate', 0):.3f}")
        print(f"  Win Rate Std Dev: {stats.get('win_rate_std', 0):.3f}")
        print()
        print(f"Performance:")
        print(f"  Avg Evaluation Time: {stats.get('average_evaluation_time', 0):.2f}s per Pokemon")
        print(f"  Unique Pokemon: {stats.get('unique_pokemon', 0)}")
        print(f"  Unique Counters: {stats.get('unique_counters', 0)}")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Efficient Counter Evaluation System")
    
    parser.add_argument("--dataset", "-d", type=str, required=True,
                       help="Path to moveset dataset JSON file")
    parser.add_argument("--counter-type", "-c", type=str, choices=["ai", "rule"], default="ai",
                       help="Counter finder type (default: ai)")
    parser.add_argument("--weights", "-w", type=str, default=None,
                       help="Path to AI weights JSON file")
    parser.add_argument("--max-pokemon", "-n", type=int, default=50,
                       help="Maximum Pokemon to evaluate (default: 50)")
    parser.add_argument("--battles", "-b", type=int, default=10,
                       help="Battles per counter (default: 10)")
    parser.add_argument("--counters", "-k", type=int, default=2,
                       help="Max counters per Pokemon (default: 2)")
    parser.add_argument("--timeout", "-t", type=int, default=30,
                       help="Timeout per battle set in seconds (default: 30)")
    parser.add_argument("--memory-limit", "-m", type=int, default=2000,
                       help="Memory limit in MB (default: 2000)")
    parser.add_argument("--sampling", "-s", type=str, 
                       choices=["stratified", "tier", "random"], default="random",
                       help="Sampling strategy: stratified (by BST), tier (by competitive tier), or random (default: random)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Custom output filename (without extension)")
    
    args = parser.parse_args()
    
    # Validate files
    if not Path(args.dataset).exists():
        print(f"✗ Dataset file not found: {args.dataset}")
        return
    
    if args.weights and not Path(args.weights).exists():
        print(f"✗ Weights file not found: {args.weights}")
        return
    
    # Initialize repositories
    try:
        base_dir = Path(__file__).resolve().parent
        pokemon_file = base_dir / "data" / "pokemon.json"
        type_chart_file = base_dir / "data" / "typeChart.csv"
        
        pokemon_repo = PokemonRepository(pokemon_file)
        type_chart = TypeChart(type_chart_file)
        
        print("✓ Data repositories initialized")
        
    except Exception as e:
        print(f"✗ Failed to initialize repositories: {e}")
        return
    
    # Create evaluation configuration
    config = EvaluationConfig(
        max_pokemon=args.max_pokemon,
        battles_per_counter=args.battles,
        max_counters_per_pokemon=args.counters,
        timeout_per_battle_set=args.timeout,
        memory_limit_mb=args.memory_limit
    )
    
    # Initialize evaluator
    try:
        evaluator = EfficientCounterEvaluator(
            pokemon_repo=pokemon_repo,
            type_chart=type_chart,
            use_ai_counter=args.counter_type == "ai",
            weights_file=args.weights,
            config=config
        )
        
        print("✓ Efficient evaluator initialized")
        
    except Exception as e:
        print(f"✗ Failed to initialize evaluator: {e}")
        return
    
    # Run evaluation
    try:
        results_file = evaluator.run_efficient_evaluation(
            args.dataset, 
            sample_size=args.max_pokemon,
            sampling_strategy=args.sampling,
            seed=args.seed,
            output_name=args.output
        )
        print(f"\n🎉 Evaluation completed!")
        print(f"📄 Results: {results_file}")
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")

if __name__ == "__main__":
    main()