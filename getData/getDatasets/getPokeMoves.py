import json
import requests

# Load the Pokémon data (assuming it's stored in a JSON file named "pokemon.json")
with open('expanded_pokemon_data.json', 'r') as file:
    pokemon_data = json.load(file)

# Function to get Pokémon moves from the PokéAPI
def get_pokemon_moves_from_api(pokemon_name):
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        moves = [move["move"]["name"] for move in data["moves"]]
        return moves
    else:
        return []

# Expand the original JSON to include learnable moves from the API
def expand_pokemon_with_moves(pokemon_data):
    for pokemon in pokemon_data:
        name = pokemon["name"]
        pokemon["learnable_moves"] = get_pokemon_moves_from_api(name)
        print(pokemon)
    return pokemon_data

expanded_pokemon_data = expand_pokemon_with_moves(pokemon_data)

# Save the expanded Pokémon data to a new JSON file
with open('expanded_pokemon_data.json', 'w') as file:
    json.dump(expanded_pokemon_data, file, indent=4)

print("Expanded Pokémon data saved to 'expanded_pokemon_data.json'")
