#!/usr/bin/env python3
"""
Pokemon JSON Statistics Analyzer

This script analyzes JSON files containing Pokemon data and calculates
mean, standard deviation, and median for the overall scores.
Results are stored in a results.json file.
"""

import json
import argparse
import statistics
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


def extract_overall_scores(data):
    """Extract overall scores from Pokemon data."""
    overall_scores = []
    
    # Handle both single objects and arrays
    if isinstance(data, dict):
        # Single Pokemon entry
        if ('gpt_scores' in data and 
            data['gpt_scores'] is not None and 
            isinstance(data['gpt_scores'], dict) and 
            'Overall' in data['gpt_scores']):
            overall_scores.append(data['gpt_scores']['Overall'])
    elif isinstance(data, list):
        # Array of Pokemon entries
        for entry in data:
            if (isinstance(entry, dict) and 
                'gpt_scores' in entry and 
                entry['gpt_scores'] is not None and 
                isinstance(entry['gpt_scores'], dict) and 
                'Overall' in entry['gpt_scores']):
                overall_scores.append(entry['gpt_scores']['Overall'])
    
    return overall_scores


def calculate_statistics(scores):
    """Calculate mean, standard deviation, and median for a list of scores."""
    if not scores:
        return None
    
    try:
        stats = {
            'mean': statistics.mean(scores),
            'standard_deviation': statistics.stdev(scores) if len(scores) > 1 else 0.0,
            'median': statistics.median(scores),
            'count': len(scores),
            'min': min(scores),
            'max': max(scores)
        }
        return stats
    except Exception as e:
        print(f"Error calculating statistics: {e}")
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


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Pokemon JSON files and calculate statistics for overall scores'
    )
    parser.add_argument(
        'input_files',
        nargs='+',
        help='Path(s) to JSON file(s) containing Pokemon data'
    )
    parser.add_argument(
        '--output',
        '-o',
        default='results.json',
        help='Output file for results (default: results.json)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Load existing results
    results = load_existing_results(args.output)
    
    # Process each input file
    for filepath in args.input_files:
        filename = Path(filepath).name
        
        if args.verbose:
            print(f"Processing file: {filepath}")
        
        # Load JSON data
        data = load_json_file(filepath)
        if data is None:
            continue
        
        # Extract overall scores
        overall_scores = extract_overall_scores(data)
        
        if not overall_scores:
            print(f"Warning: No overall scores found in '{filepath}'")
            continue
        
        # Calculate statistics
        stats = calculate_statistics(overall_scores)
        if stats is None:
            continue
        
        # Store results
        results[filename] = stats
        
        if args.verbose:
            print(f"  Found {stats['count']} scores")
            print(f"  Mean: {stats['mean']:.3f}")
            print(f"  Standard Deviation: {stats['standard_deviation']:.3f}")
            print(f"  Median: {stats['median']:.3f}")
            print(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")
            print()
    
    # Save updated results
    save_results(results, args.output)
    
    # Summary
    print(f"Processed {len(args.input_files)} file(s)")
    print(f"Results stored in '{args.output}'")


if __name__ == '__main__':
    main()