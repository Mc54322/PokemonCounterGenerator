#!/usr/bin/env python3
"""
Counter Statistics Analyzer

This script extracts summary_stats sections from JSON files and compiles
them into a counterresults.json file with filename:{stats} format.
"""

import json
import argparse
import os
from pathlib import Path


def load_json_file(filepath):
    """Load and parse a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{filepath}': {e}")
        return None


def extract_summary_stats(data):
    """Extract summary_stats from JSON data."""
    if isinstance(data, dict) and 'summary_stats' in data:
        return data['summary_stats']
    elif isinstance(data, list):
        # If it's a list, look for summary_stats in the first element or any element
        for item in data:
            if isinstance(item, dict) and 'summary_stats' in item:
                return item['summary_stats']
    
    return None


def load_existing_results(results_file):
    """Load existing results from the results file."""
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as file:
                return json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not load existing results from '{results_file}'. Starting fresh.")
    
    return {}


def save_results(results, results_file):
    """Save results to the results file."""
    try:
        with open(results_file, 'w', encoding='utf-8') as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
        print(f"Results saved to '{results_file}'")
    except Exception as e:
        print(f"Error saving results to '{results_file}': {e}")


def format_stats_output(stats, verbose=False):
    """Format stats for display."""
    if not stats:
        return "No stats found"
    
    output = []
    if verbose:
        for key, value in stats.items():
            if isinstance(value, float):
                output.append(f"  {key}: {value:.6f}")
            else:
                output.append(f"  {key}: {value}")
    else:
        # Brief summary
        battles = stats.get('total_battles', 'N/A')
        win_rate = stats.get('average_win_rate', 'N/A')
        if isinstance(win_rate, float):
            win_rate = f"{win_rate:.3f}"
        pokemon_count = stats.get('unique_pokemon', 'N/A')
        output.append(f"  Battles: {battles}, Avg Win Rate: {win_rate}, Pokemon: {pokemon_count}")
    
    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(
        description='Extract summary_stats from JSON files and compile into counterresults.json'
    )
    parser.add_argument(
        'input_files',
        nargs='+',
        help='Path(s) to JSON file(s) containing summary_stats data'
    )
    parser.add_argument(
        '--output',
        '-o',
        default='counterresults.json',
        help='Output file for results (default: counterresults.json)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing results instead of appending'
    )
    
    args = parser.parse_args()
    
    # Load existing results (unless overwrite is specified)
    if args.overwrite:
        results = {}
        if args.verbose:
            print("Starting with fresh results (overwrite mode)")
    else:
        results = load_existing_results(args.output)
        if args.verbose and results:
            print(f"Loaded {len(results)} existing result(s)")
    
    # Process each input file
    processed_count = 0
    for filepath in args.input_files:
        filename = Path(filepath).name
        
        if args.verbose:
            print(f"\nProcessing file: {filepath}")
        
        # Load JSON data
        data = load_json_file(filepath)
        if data is None:
            continue
        
        # Extract summary stats
        summary_stats = extract_summary_stats(data)
        
        if summary_stats is None:
            print(f"Warning: No summary_stats found in '{filepath}'")
            continue
        
        # Store results
        results[filename] = summary_stats
        processed_count += 1
        
        if args.verbose:
            print(f"  Extracted summary_stats:")
            print(format_stats_output(summary_stats, verbose=True))
        else:
            print(f"✓ {filename}")
            print(format_stats_output(summary_stats, verbose=False))
    
    # Save updated results
    if processed_count > 0:
        save_results(results, args.output)
        print(f"\nProcessed {processed_count} file(s) successfully")
        print(f"Total entries in results: {len(results)}")
    else:
        print("No files were processed successfully")


def validate_summary_stats_structure(stats):
    """Validate that the summary_stats has expected structure."""
    expected_fields = [
        'total_battles', 'average_win_rate', 'median_win_rate', 
        'win_rate_std', 'average_evaluation_time', 'total_evaluation_time',
        'unique_pokemon', 'unique_counters', 'success_rate'
    ]
    
    if not isinstance(stats, dict):
        return False, "summary_stats is not a dictionary"
    
    missing_fields = [field for field in expected_fields if field not in stats]
    if missing_fields:
        return False, f"Missing fields: {', '.join(missing_fields)}"
    
    return True, "Valid structure"


if __name__ == '__main__':
    main()