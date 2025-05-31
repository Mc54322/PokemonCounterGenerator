#!/usr/bin/env python3
import json
import time
from pathlib import Path
from typing import Dict, List
from apichecker import score_moveset_with_gpt

def format_moveset_for_gpt(pokemon_data: Dict) -> str:
    """
    Convert a Pokemon data entry to Showdown format for GPT evaluation.
    """
    lines = []
    
    # Pokemon name and item
    if pokemon_data.get("item") and pokemon_data["item"] != "None":
        lines.append(f"{pokemon_data['pokemonName']} @ {pokemon_data['item']}")
    else:
        lines.append(pokemon_data['pokemonName'])
    
    # Ability
    if pokemon_data.get("ability"):
        lines.append(f"Ability: {pokemon_data['ability']}")
    
    # Format EVs (only show non-zero values)
    evs = pokemon_data.get("evs", {})
    ev_parts = []
    stat_abbrs = {
        "HP": "HP",
        "Attack": "Atk", 
        "Defense": "Def",
        "SpecialAttack": "SpA",
        "SpecialDefense": "SpD",
        "Speed": "Spe"
    }
    
    for stat, value in evs.items():
        if value > 0:
            abbr = stat_abbrs.get(stat, stat)
            ev_parts.append(f"{value} {abbr}")
    
    if ev_parts:
        lines.append(f"EVs: {' / '.join(ev_parts)}")
    
    # Format IVs (only show non-31 values)
    ivs = pokemon_data.get("ivs", {})
    iv_parts = []
    
    for stat, value in ivs.items():
        if value < 31:
            abbr = stat_abbrs.get(stat, stat)
            iv_parts.append(f"{value} {abbr}")
    
    if iv_parts:
        lines.append(f"IVs: {' / '.join(iv_parts)}")
    
    # Nature
    if pokemon_data.get("nature"):
        lines.append(f"{pokemon_data['nature']} Nature")
    
    # Tera Type
    if pokemon_data.get("teraType"):
        lines.append(f"Tera Type: {pokemon_data['teraType']}")
    
    # Moves
    moves = pokemon_data.get("moves", [])
    for move in moves:
        # Capitalize first letter and replace hyphens with spaces for better readability
        formatted_move = move.replace("-", " ").title()
        lines.append(f"- {formatted_move}")
    
    return "\n".join(lines)

def score_all_movesets(input_file: str = "all_movesets.json", 
                      output_file: str = "scored_movesets.json",
                      delay: float = 1.0,
                      start_from: int = 0) -> None:
    """
    Score all movesets in the input file using GPT and save results.
    
    Args:
        input_file: Path to the all_movesets.json file
        output_file: Path to save the scored results
        delay: Delay between API calls to avoid rate limiting
        start_from: Index to start from (for resuming interrupted runs)
    """
    # Load existing results if output file exists
    output_path = Path(output_file)
    if output_path.exists():
        try:
            with open(output_file, "r") as f:
                scored_data = json.load(f)
            print(f"Loaded {len(scored_data)} existing scores from {output_file}")
        except json.JSONDecodeError:
            print(f"Could not read {output_file}, starting fresh")
            scored_data = []
    else:
        scored_data = []
    
    # Load moveset data
    try:
        with open(input_file, "r") as f:
            movesets = json.load(f)
        print(f"Loaded {len(movesets)} movesets from {input_file}")
    except FileNotFoundError:
        print(f"Error: {input_file} not found!")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not parse {input_file}")
        return
    
    # Create a set of already scored Pokemon names for quick lookup
    scored_names = {entry["pokemonName"] for entry in scored_data if "pokemonName" in entry}
    
    # Start processing from the specified index
    total_processed = len(scored_data)
    successful_scores = 0
    failed_scores = 0
    
    for i, pokemon_data in enumerate(movesets):
        if i < start_from:
            continue
            
        pokemon_name = pokemon_data.get("pokemonName", f"Unknown_{i}")
        
        # Skip if already scored
        if pokemon_name in scored_names:
            print(f"[{i+1}/{len(movesets)}] Skipping {pokemon_name} (already scored)")
            continue
        
        print(f"[{i+1}/{len(movesets)}] Scoring {pokemon_name}...")
        
        try:
            # Format moveset for GPT
            moveset_text = format_moveset_for_gpt(pokemon_data)
            print(f"Formatted moveset:\n{moveset_text}\n")
            
            # Get GPT score
            gpt_scores = score_moveset_with_gpt(moveset_text)
            
            # Create scored entry
            scored_entry = pokemon_data.copy()
            scored_entry["gpt_scores"] = gpt_scores
            scored_entry["moveset_text"] = moveset_text
            
            # Add to results
            scored_data.append(scored_entry)
            scored_names.add(pokemon_name)
            
            successful_scores += 1
            total_processed += 1
            
            print(f"✅ Successfully scored {pokemon_name}")
            print(f"Overall Score: {gpt_scores.get('Overall', 'N/A')}")
            print(f"Scores: {gpt_scores}")
            
        except Exception as e:
            print(f"❌ Error scoring {pokemon_name}: {e}")
            failed_scores += 1
            
            # Still add the entry without scores to track progress
            failed_entry = pokemon_data.copy()
            failed_entry["gpt_scores"] = None
            failed_entry["error"] = str(e)
            failed_entry["moveset_text"] = format_moveset_for_gpt(pokemon_data)
            scored_data.append(failed_entry)
            total_processed += 1
        
        # Save progress periodically (every 10 Pokemon)
        if total_processed % 10 == 0:
            try:
                with open(output_file, "w") as f:
                    json.dump(scored_data, f, indent=2)
                print(f"💾 Progress saved: {total_processed} total, {successful_scores} successful, {failed_scores} failed")
            except Exception as e:
                print(f"Error saving progress: {e}")
        
        # Rate limiting delay
        if delay > 0:
            print(f"Waiting {delay}s...")
            time.sleep(delay)
    
    # Final save
    try:
        with open(output_file, "w") as f:
            json.dump(scored_data, f, indent=2)
        print(f"✅ Final results saved to {output_file}")
    except Exception as e:
        print(f"❌ Error saving final results: {e}")
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"Total processed: {total_processed}")
    print(f"Successful scores: {successful_scores}")
    print(f"Failed scores: {failed_scores}")
    print(f"Success rate: {successful_scores/(successful_scores+failed_scores)*100:.1f}%" if (successful_scores+failed_scores) > 0 else "N/A")

def analyze_scores(scored_file: str = "scored_movesets.json") -> None:
    """
    Analyze the distribution of GPT scores.
    """
    try:
        with open(scored_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File {scored_file} not found!")
        return
    
    # Filter successful scores
    successful_scores = [entry for entry in data if entry.get("gpt_scores") is not None]
    failed_scores = [entry for entry in data if entry.get("gpt_scores") is None]
    
    print(f"📈 SCORE ANALYSIS:")
    print(f"Total entries: {len(data)}")
    print(f"Successfully scored: {len(successful_scores)}")
    print(f"Failed to score: {len(failed_scores)}")
    
    if not successful_scores:
        print("No successful scores to analyze!")
        return
    
    # Analyze overall scores
    overall_scores = [entry["gpt_scores"]["Overall"] for entry in successful_scores]
    
    print(f"\n🎯 OVERALL SCORES:")
    print(f"Average: {sum(overall_scores)/len(overall_scores):.2f}")
    print(f"Min: {min(overall_scores):.1f}")
    print(f"Max: {max(overall_scores):.1f}")
    
    # Score distribution
    score_buckets = {
        "9-10 (Excellent)": len([s for s in overall_scores if s >= 9]),
        "7-8.9 (Good)": len([s for s in overall_scores if 7 <= s < 9]),
        "5-6.9 (Average)": len([s for s in overall_scores if 5 <= s < 7]),
        "3-4.9 (Poor)": len([s for s in overall_scores if 3 <= s < 5]),
        "1-2.9 (Terrible)": len([s for s in overall_scores if s < 3])
    }
    
    print(f"\n📊 SCORE DISTRIBUTION:")
    for bucket, count in score_buckets.items():
        percentage = (count / len(overall_scores)) * 100
        print(f"{bucket}: {count} ({percentage:.1f}%)")
    
    # Best and worst performers
    sorted_scores = sorted(successful_scores, key=lambda x: x["gpt_scores"]["Overall"], reverse=True)
    
    print(f"\n🏆 TOP 5 HIGHEST SCORING POKEMON:")
    for i, entry in enumerate(sorted_scores[:5]):
        score = entry["gpt_scores"]["Overall"]
        name = entry["pokemonName"]
        print(f"{i+1}. {name}: {score}/10")
    
    print(f"\n💥 TOP 5 LOWEST SCORING POKEMON:")
    for i, entry in enumerate(sorted_scores[-5:]):
        score = entry["gpt_scores"]["Overall"]
        name = entry["pokemonName"]
        print(f"{i+1}. {name}: {score}/10")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Score Pokemon movesets using GPT")
    parser.add_argument("--input", "-i", default="all_movesets.json", 
                       help="Input file with movesets (default: all_movesets.json)")
    parser.add_argument("--output", "-o", default="scored_movesets.json",
                       help="Output file for scored movesets (default: scored_movesets.json)")
    parser.add_argument("--delay", "-d", type=float, default=1.0,
                       help="Delay between API calls in seconds (default: 1.0)")
    parser.add_argument("--start-from", type=int, default=0,
                       help="Index to start from (for resuming, default: 0)")
    parser.add_argument("--analyze", "-a", action="store_true",
                       help="Analyze existing scored results instead of scoring")
    parser.add_argument("--test", "-t", action="store_true",
                       help="Test with first 5 Pokemon only")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_scores(args.output)
    else:
        if args.test:
            print("🧪 TEST MODE: Processing first 5 Pokemon only")
            # Load and limit to first 5
            with open(args.input, "r") as f:
                movesets = json.load(f)
            test_movesets = movesets[:5]
            test_file = "test_movesets.json"
            with open(test_file, "w") as f:
                json.dump(test_movesets, f, indent=2)
            score_all_movesets(test_file, "test_scored_movesets.json", args.delay, args.start_from)
        else:
            score_all_movesets(args.input, args.output, args.delay, args.start_from)

if __name__ == "__main__":
    main()