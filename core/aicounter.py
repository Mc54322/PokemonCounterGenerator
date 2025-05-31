from typing import Dict, List, Tuple, Optional, Set
import json
import numpy as np
import random
from collections import defaultdict
from models import PokemonConfig
from repository import PokemonRepository, TypeChart
from moveutility import evaluateMoveUtilities, evaluateDefensiveUtilities, loadMovesData
from abilityutility import getHighestRatedAbility
from showdownsim import has_high_winrate, sanitize_team

def evaluate_configuration_batch(meta, move_sets_batch,
                                 pokemon_info, numeric_cols, categorical_cols,
                                 scaler, encoder, model):
    """
    Simplified batch evaluation function for compatibility.
    Returns dummy values since this is mainly used by the original AI training code.
    """
    ability, item, nature, tera, ev_spread, iv_config = meta
    
    # For now, return a simple heuristic score
    best_config = {
        "ability": ability,
        "item": item,
        "nature": nature,
        "teraType": tera,
        "EV_spread": ev_spread,
        "IV_config": iv_config,
        "move_set": move_sets_batch[0] if move_sets_batch else []
    }
    
    # Simple heuristic based on stat totals
    bst = sum(pokemon_info.get("stats", {}).values())
    best_score = bst / 600.0  # Normalize BST
    best_features = {"bst_bonus": best_score}
    
    return best_score, best_config, best_features


class AICounterFinder:
    """
    AI-enhanced version of CounterFinder that learns from battle simulations
    to improve counter selection using reinforcement learning.
    """
    # Movepool-based role indicators
    supportMoves = {"Tailwind", "Trick Room", "Reflect", "Light Screen", "Follow Me",
                    "Rain Dance", "Sunny Day", "Taunt", "Safeguard", "Trick"}
    pivotMoves   = {"U-Turn", "Volt Switch", "Parting Shot", "Flip Turn", "Shed Tail",
                    "Teleport", "Baton Pass", "Chilly Reception"}
    setupMoves   = {"Swords Dance", "Calm Mind", "Dragon Dance", "Nasty Plot", "Bulk Up",
                    "Shift Gear", "Quiver Dance"}

    def __init__(self, repository: PokemonRepository, typeChart: TypeChart, 
                 learning_rate: float = 0.1, discount_factor: float = 0.95):
        self.repository = repository
        self.typeChart = typeChart
        # Load moves data
        self.movesData = loadMovesData("data/moves.json")
        # Learning parameters
        self.learningRate = learning_rate
        self.discountFactor = discount_factor
        # Initialize weights
        self.weights = {
            "offensive_util": 0.5,
            "defensive_util": 0.5,
            "advantage_bonus": 0.5,
            "bst_bonus": 0.5,
            "ability_bonus": 0.5,
            "type_resistance": 0.5,
            "role_counter": 0.5,
            "speed_advantage": 0.5,
            "move_coverage": 0.5
        }
        # Load aliases and Smogon movesets
        with open("data/aliases.json") as f:
            raw_aliases = json.load(f)
        # Store aliases as showdown_name -> pokemon_json_name
        self.aliases = {k.lower(): v.lower() for k, v in raw_aliases.items()}
        
        #with open("data/smogonMovesets.json") as f:
        with open("data/models/all_movesets_r6.json") as f:
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
        
        # Performance tracking for RL
        self.winRates = {}  # track win rates of counter selections
        self.counterHistory = {}  # track which counters were selected
        self.rlSessions = 0  # count of RL sessions

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
        pScore = inputStats["Attack"] + self.weights["speed_advantage"] * inputStats["Speed"]
        sScore = inputStats["SpecialAttack"] + self.weights["speed_advantage"] * inputStats["Speed"]
        wScore = inputStats["Defense"] + inputStats["SpecialDefense"]
        tScore = inputStats["HP"] + self.weights["type_resistance"] * (inputStats["Defense"] + inputStats["SpecialDefense"])
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

    def calculateCandidateAdvantageBonus(self, role: str, cand: Dict[str, int], inp: Dict[str, int]) -> float:
        bonus = 0.0
        if role == "PhysicalSweeper":
            if cand["Defense"] > inp["Attack"]:
                bonus += 2 * self.weights["advantage_bonus"]
            if cand["HP"] > 0.8 * inp["HP"]:
                bonus += 2 * self.weights["advantage_bonus"]
            if cand["Speed"] > inp["Speed"]:
                bonus += self.weights["speed_advantage"]
        elif role == "SpecialSweeper":
            if cand["SpecialDefense"] > inp["SpecialAttack"]:
                bonus += 2 * self.weights["advantage_bonus"]
            if cand["HP"] > 0.8 * inp["HP"]:
                bonus += 2 * self.weights["advantage_bonus"]
            if cand["Speed"] > inp["Speed"]:
                bonus += self.weights["speed_advantage"]
        elif role == "Wall":
            if cand["Attack"] > inp["Attack"]:
                bonus += 2 * self.weights["advantage_bonus"]
            if cand["SpecialAttack"] > inp["SpecialAttack"]:
                bonus += 2 * self.weights["advantage_bonus"]
        elif role == "Tank":
            if cand["HP"] > inp["HP"]:
                bonus += 2 * self.weights["advantage_bonus"]
            if cand["Defense"] > inp["Defense"] or cand["SpecialDefense"] > inp["SpecialDefense"]:
                bonus += 2 * self.weights["advantage_bonus"]
        elif role == "Balanced":
            if cand["Speed"] > inp["Speed"]:
                bonus += 2 * self.weights["speed_advantage"]
            if cand["Attack"] > inp["Attack"] or cand["SpecialAttack"] > inp["SpecialAttack"]:
                bonus += 2 * self.weights["advantage_bonus"]
        elif role == "SetupSweeper":
            if cand["Attack"] > inp["Attack"] or cand["SpecialAttack"] > inp["SpecialAttack"]:
                bonus += 2 * self.weights["advantage_bonus"]
            if cand["Speed"] > inp["Speed"]:
                bonus += 2 * self.weights["speed_advantage"]
        elif role == "Support":
            if (cand["Defense"] + cand["SpecialDefense"]) > (inp["Defense"] + inp["SpecialDefense"]):
                bonus += 2 * self.weights["advantage_bonus"]
            if cand["HP"] > inp["HP"]:
                bonus += 2 * self.weights["advantage_bonus"]
        elif role == "Pivot":
            if cand["HP"] > inp["HP"]:
                bonus += 2 * self.weights["advantage_bonus"]
            if cand["Attack"] > inp["Attack"] or cand["SpecialAttack"] > inp["SpecialAttack"]:
                bonus += 2 * self.weights["advantage_bonus"]
        elif role == "TrickRoomSweeper":
            if cand["Attack"] > inp["Attack"]:
                bonus += 2 * self.weights["advantage_bonus"]
            if cand["SpecialAttack"] > inp["SpecialAttack"]:
                bonus += 2 * self.weights["advantage_bonus"]
            if cand["HP"] > 0.8 * inp["HP"]:
                bonus += 2 * self.weights["advantage_bonus"]
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
                off = "Attack" if candStats["Attack"] >= candStats["SpecialAttack"] else "SpecialAttack"
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
                    evs["SpecialDefense"] = 0
                else:
                    d = higher("Defense", "SpecialDefense")
                    evs[d] = 252
                    other = "SpecialDefense" if d == "Defense" else "Defense"
                    evs[other] = 4
                nature = "Bold" if candStats["Defense"] >= candStats["SpecialDefense"] else "Calm"
            else:  # Pivot
                evs["Speed"] = 252
                evs["Defense"] = 4
                nature = "Jolly"
            return evs, nature
        # Fallback Balanced
        evs["HP"] = 252
        evs["Defense"] = 128
        evs["SpecialDefense"] = 128
        nature = "Bold" if candStats["Defense"] >= candStats["SpecialDefense"] else "Calm"
        return evs, nature

    def getSmogonLookupName(self, pokemon_name: str) -> str:
        """
        Converts a pokemon.json name to its corresponding Smogon name if available,
        using the reverse aliases dictionary.
        """
        lookup_key = pokemon_name.lower()
        if lookup_key in self.reverseAliases:
            return self.reverseAliases[lookup_key]
        return pokemon_name

    def findCounters(self, inputPokemon, inputStats: Dict[str, int], inputMoves: List[str],
                     moveset_text: Optional[str] = None) -> List[PokemonConfig]:
        """
        Find Pokémon that counter the input Pokémon based on typing, stats, and moves.
        Only uses Pokemon that exist in smogonMovesets.json with their exact configurations.
        """
        role = self.determineOpponentRole(inputStats, inputMoves)
        candidatesWithWeight: List[Tuple[object, float, List[str], Dict[str, float]]] = []

        # 1) First, get all unique Pokemon names from Smogon data
        smogon_pokemon_names = set()
        for entry in self.smogonMovesets:
            pokemon_name = entry.get("pokemonName", "")
            if pokemon_name:
                smogon_pokemon_names.add(pokemon_name.lower())

        # 2) For each Pokemon in our repository, check if it exists in Smogon data
        for candidate in self.repository.pokemonList:
            if candidate.name.lower() == inputPokemon.name.lower():
                continue
                
            # Check if this Pokemon exists in Smogon movesets
            candidate_name_lower = candidate.name.lower()
            
            # Also check aliases - if candidate has an alias that matches Smogon data
            has_smogon_data = candidate_name_lower in smogon_pokemon_names
            if not has_smogon_data:
                # Check if any alias of this Pokemon exists in Smogon data
                for smogon_key, canonical_name in self.aliases.items():
                    if canonical_name.lower() == candidate_name_lower and smogon_key in smogon_pokemon_names:
                        has_smogon_data = True
                        break
            
            if not has_smogon_data:
                continue  # Skip Pokemon not in Smogon data

            candStats = candidate.baseStats
            features = {
                "offensive_util": 0.0,
                "defensive_util": 0.0,
                "advantage_bonus": 0.0,
                "bst_bonus": 0.0,
                "ability_bonus": 0.0,
                "role_counter": 0.0,
                "type_resistance": 0.0,
                "speed_advantage": 0.0,
                "move_coverage": 0.0
            }
            ctMoves: List[str] = []

            # Offensive utility
            moveUtils = evaluateMoveUtilities(inputPokemon, candidate, self.typeChart, self.movesData)
            if moveUtils:
                avgUtil = sum(m[1] for m in moveUtils) / len(moveUtils)
                features["offensive_util"] = avgUtil / 50
                ctMoves = [m[0] for m in moveUtils]
                features["move_coverage"] = min(1.0, len(set(ctMoves)) / 4.0) * 0.5

            # Defensive utility
            defUtils = evaluateDefensiveUtilities(inputMoves, candidate, self.typeChart, self.movesData)
            if defUtils:
                avgMult = sum(d[2] for d in defUtils) / len(defUtils)
                features["defensive_util"] = (1 - avgMult) * 5
                features["type_resistance"] = (1 - avgMult) * 2

            # Advantage bonus based on role
            features["advantage_bonus"] = self.calculateCandidateAdvantageBonus(role, candStats, inputStats)
            
            # Speed advantage
            features["speed_advantage"] = 0.5 if candStats["Speed"] > inputStats["Speed"] else 0.0
            
            # BST bonus
            features["bst_bonus"] = sum(candStats.values()) / 600
            
            # Ability bonus
            bestAbility, abilityRating = getHighestRatedAbility(candidate.abilities)
            features["ability_bonus"] = abilityRating / 2
            
            # Role counter bonus
            roleCounters = {
                "PhysicalSweeper": ["Wall", "Tank"],
                "SpecialSweeper": ["Wall", "Tank"],
                "Tank": ["SetupSweeper", "Support"],
                "Wall": ["Balanced", "Support"],
                "Balanced": ["Wall", "Tank"],
                "Support": ["PhysicalSweeper", "SpecialSweeper"],
                "SetupSweeper": ["PhysicalSweeper", "SpecialSweeper", "Pivot"],
                "TrickRoomSweeper": ["PhysicalSweeper", "SpecialSweeper"],
                "Pivot": ["Wall", "Support"]
            }
            candRole = self.determineOpponentRole(candStats, ctMoves)
            if role in roleCounters and candRole in roleCounters[role]:
                features["role_counter"] = 1.0
            
            # Calculate weighted sum
            weight = sum(self.weights[k] * v for k, v in features.items())

            if weight > 0:
                candidatesWithWeight.append((candidate, weight, ctMoves, features))

        # 3) Sort by descending weight
        candidatesWithWeight.sort(key=lambda x: x[1], reverse=True)

        # 4) Build final list using ONLY Smogon movesets
        finalCounters: List[PokemonConfig] = []
        
        for candidate, weight, ctMoves, features in candidatesWithWeight:
            # Find the Smogon moveset for this Pokemon
            movesetEntry = self.getSmogonMovesetForPokemon(candidate.name)
            
            if not movesetEntry:
                continue  # Skip if no Smogon moveset found

            # Extract Smogon config EXACTLY as provided
            evs = movesetEntry.get("evs", {})
            ivs = movesetEntry.get("ivs", {stat: 31 for stat in candidate.baseStats})
            nature = movesetEntry.get("nature", "Neutral")
            moves = movesetEntry.get("moves", [])
            ability = movesetEntry.get("ability", "")
            item = movesetEntry.get("item", "")
            teraType = movesetEntry.get("teraType", "")

            # Validate EV spread
            if not self.validateEvSpread(evs):
                print(f"Warning: {candidate.name} has invalid EV spread: {evs} (total: {sum(evs.values())}), skipping.")
                continue

            adjustedStats = candidate.getAdjustedStats(evs, ivs, nature)
            
            # Create PokemonConfig without features parameter
            config = PokemonConfig(
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
            )
            
            # Add features as an attribute after creation
            config.features = features
            finalCounters.append(config)

            if len(finalCounters) >= 5:
                break

        return finalCounters

    def validateEvSpread(self, evs: Dict[str, int]) -> bool:
        """
        Validate that an EV spread doesn't exceed the 510 total limit
        and individual stats don't exceed 252.
        """
        total = sum(evs.values())
        if total > 510:
            return False
        
        for stat, value in evs.items():
            if value > 252 or value < 0:
                return False
                
        return True

    def getSmogonMovesetForPokemon(self, pokemon_name: str) -> Optional[dict]:
        """
        Get the first Smogon moveset entry for a given Pokemon name.
        Handles both direct names and aliases.
        """
        # Try direct lookup first
        for entry in self.smogonMovesets:
            if entry.get("pokemonName", "").lower() == pokemon_name.lower():
                return entry
        
        # Try alias lookup - check if this Pokemon has an alias in Smogon data
        for smogon_key, canonical_name in self.aliases.items():
            if canonical_name.lower() == pokemon_name.lower():
                # Found an alias, now look for the Smogon key in movesets
                for entry in self.smogonMovesets:
                    if entry.get("pokemonName", "").lower() == smogon_key:
                        return entry
        
        return None

    def buildShowdownFormat(self, pokemon_config: PokemonConfig) -> str:
        """
        Convert a PokemonConfig to Showdown format for simulation.
        Uses proper Pokemon names and short nicknames to avoid Showdown errors.
        """
        # Use Smogon format name if available (search through aliases for the key)
        smogon_name = pokemon_config.name
        canonical_name = pokemon_config.name.lower()
        
        # Look for this pokemon in aliases (where key = showdown name, value = pokemon.json name)
        for showdown_name, pokemon_json_name in self.aliases.items():
            if pokemon_json_name.lower() == canonical_name:
                smogon_name = showdown_name
                break
        
        # Generate a short nickname to avoid length issues
        nickname = self.generateShortNickname(smogon_name)
        
        lines = []
        if pokemon_config.item:
            lines.append(f"{nickname} ({smogon_name}) @ {pokemon_config.item}")
        else:
            lines.append(f"{nickname} ({smogon_name})")
            
        if pokemon_config.ability:
            lines.append(f"Ability: {pokemon_config.ability}")
            
        # Add EVs if any
        ev_parts = []
        for stat_name, value in pokemon_config.evs.items():
            if value > 0:
                # Convert to showdown stat abbreviations
                abbr = {"HP": "HP", "Attack": "Atk", "Defense": "Def", 
                        "SpecialAttack": "SpA", "SpecialDefense": "SpD", "Speed": "Spe"}
                ev_parts.append(f"{value} {abbr.get(stat_name, stat_name)}")
        if ev_parts:
            lines.append(f"EVs: {' / '.join(ev_parts)}")
            
        # Add IVs if not all 31
        iv_parts = []
        for stat_name, value in pokemon_config.ivs.items():
            if value < 31:  # Only include non-perfect IVs
                abbr = {"HP": "HP", "Attack": "Atk", "Defense": "Def", 
                        "SpecialAttack": "SpA", "SpecialDefense": "SpD", "Speed": "Spe"}
                iv_parts.append(f"{value} {abbr.get(stat_name, stat_name)}")
        if iv_parts:
            lines.append(f"IVs: {' / '.join(iv_parts)}")
            
        if pokemon_config.nature:
            lines.append(f"{pokemon_config.nature} Nature")
            
        if pokemon_config.teraType:
            lines.append(f"Tera Type: {pokemon_config.teraType}")
            
        # Add moves
        for move in pokemon_config.moves:
            lines.append(f"- {move}")
            
        return "\n".join(lines)

    def generateShortNickname(self, pokemon_name: str) -> str:
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
            "ennamorusincarmate": "EnnamorusI",
            "enamorustherian": "EnnamorusT"
        }
        
        # Check if we have a predefined short name
        name_lower = name.lower()
        for long_name, short_name in nickname_map.items():
            if long_name in name_lower:
                return short_name
        
        # Generic shortening: take first 8 characters and add suffix if needed
        if len(name) <= 18:
            return name
        elif len(name) <= 15:
            return name
        else:
            # Take first part of name
            return name[:15]

    def evaluateCounter(self, input_moveset: str, counter_config: PokemonConfig, 
                        battles: int = 50) -> float:
        """
        Evaluate a counter against an input moveset using Showdown simulations.
        Returns the actual win rate (0.0 to 1.0) of the counter.
        """
        from showdownsim import simulate_battle, sanitize_team
        import asyncio
        import gc
        import time
        
        counter_text = self.buildShowdownFormat(counter_config)
        
        # Sanitize movesets for simulation
        input_clean = sanitize_team(input_moveset)
        counter_clean = sanitize_team(counter_text)
        
        try:
            # Add small delay to prevent overwhelming the simulator
            time.sleep(0.1)
            
            # Run battles in smaller chunks to manage memory
            total_wins_counter = 0
            total_wins_challenger = 0
            chunk_size = min(10, battles)  # Run max 10 battles at a time
            
            for chunk_start in range(0, battles, chunk_size):
                chunk_battles = min(chunk_size, battles - chunk_start)
                
                # Run the battle chunk
                wins_by_counter, wins_by_challenger = asyncio.run(
                    simulate_battle(input_clean, counter_clean, chunk_battles)
                )
                
                total_wins_counter += wins_by_counter
                total_wins_challenger += wins_by_challenger
                
                # Force garbage collection between chunks
                gc.collect()
                time.sleep(0.05)  # Brief pause between chunks
            
            total_battles = total_wins_counter + total_wins_challenger
            
            if total_battles == 0:
                print(f"No battles completed for {counter_config.name}")
                return 0.0
            
            # Calculate actual win rate
            actual_win_rate = total_wins_counter / total_battles
            
            # Store result for learning
            key = f"{counter_config.name}_vs_{input_moveset.split()[0]}"
            self.winRates[key] = actual_win_rate
            
            return actual_win_rate
            
        except Exception as e:
            print(f"Error in battle simulation: {e}")
            return 0.0

    def updateWeights(self, counter_config: PokemonConfig, reward: float):
        """
        Update feature weights based on battle simulation results.
        Uses a simplified policy gradient approach.
        """
        if not hasattr(counter_config, 'features'):
            return
            
        features = counter_config.features
        
        # Calculate weight updates using policy gradient
        for feature_name, feature_value in features.items():
            if feature_name in self.weights:
                # Gradient update: increase weights for features that led to wins
                # decrease for features that led to losses
                adjustment = self.learningRate * (reward - 0.5) * feature_value
                self.weights[feature_name] += adjustment
                
                # Ensure weights stay in reasonable range
                self.weights[feature_name] = max(0.1, min(1.0, self.weights[feature_name]))
                
        self.rlSessions += 1
        
        # Save updated weights periodically
        if self.rlSessions % 10 == 0:
            self.saveWeights("ai_counter_weights.json")

    def runReinforcementLearning(self, input_moveset_str: str, top_counters: List[PokemonConfig], 
                                 episodes: int = 5, battles_per_episode: int = 10):
        """
        Run reinforcement learning to improve counter selection.
        Tests each counter, evaluates actual win rate, and updates weights.
        """
        print(f"\nRunning reinforcement learning ({episodes} episodes)...")
        
        for episode in range(episodes):
            print(f"Episode {episode+1}/{episodes}")
            
            for i, counter in enumerate(top_counters[:3]):  # Train on top 3 counters
                print(f"  Testing counter {i+1}: {counter.name}")
                
                # Evaluate counter with simulations - get actual win rate
                actual_win_rate = self.evaluateCounter(
                    input_moveset_str, counter, battles=battles_per_episode
                )
                
                # Convert win rate to reward signal
                # Win rates > 0.5 are positive rewards, < 0.5 are negative
                # Scale reward to be more sensitive to high win rates
                if actual_win_rate >= 0.8:
                    reward = 1.0  # Excellent counter
                elif actual_win_rate >= 0.7:
                    reward = 0.8  # Very good counter  
                elif actual_win_rate >= 0.6:
                    reward = 0.6  # Good counter
                elif actual_win_rate >= 0.5:
                    reward = 0.4  # Decent counter
                else:
                    reward = 0.2  # Poor counter
                
                # Update weights based on actual performance
                self.updateWeights(counter, reward)
                
                print(f"    Win rate: {actual_win_rate:.3f} (reward: {reward:.1f})")
                
        print("\nReinforcement learning complete. Updated weights:")
        for feature, weight in self.weights.items():
            print(f"  {feature}: {weight:.3f}")

    def saveWeights(self, filename: str):
        """Save current weight values to a JSON file."""
        with open(filename, 'w') as f:
            json.dump({
                'weights': self.weights,
                'sessions': self.rlSessions,
                'win_rates': self.winRates
            }, f, indent=2)
            
    def loadWeights(self, filename: str):
        """Load weight values from a JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.weights = data.get('weights', self.weights)
                self.rlSessions = data.get('sessions', 0)
                self.winRates = data.get('win_rates', {})
            print(f"Loaded AI counter weights from {filename}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Could not load weights file: {e}")