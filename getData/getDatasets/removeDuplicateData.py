import json

def remove_duplicate_pokemon(input_path: str, output_path: str):
    """
    Remove duplicate Pokémon entries from a JSON file.
    
    Args:
        input_path: Path to the input JSON file
        output_path: Path where the deduplicated JSON will be saved
    """
    try:
        # Read the JSON file
        with open(input_path, 'r') as file:
            data = json.load(file)
        
        # Convert to list if it's not already
        if not isinstance(data, list):
            data = [data]
        
        # Create a dictionary to store unique Pokémon based on their name
        unique_pokemon = {}
        
        # Keep only the first occurrence of each Pokémon
        for pokemon in data:
            if pokemon['name'] not in unique_pokemon:
                unique_pokemon[pokemon['name']] = pokemon
        
        # Convert back to list
        deduplicated_data = list(unique_pokemon.values())
        
        # Write the deduplicated data to a new file
        with open(output_path, 'w') as file:
            json.dump(deduplicated_data, file, indent=2)
            
        print(f"Successfully removed duplicates. Found {len(data) - len(deduplicated_data)} duplicate entries.")
        print(f"Result saved to {output_path}")
            
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except FileNotFoundError:
        print(f"Could not find file: {input_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    input_file = "pokemonData.json"
    output_file = "pokemon.json"
    remove_duplicate_pokemon(input_file, output_file)