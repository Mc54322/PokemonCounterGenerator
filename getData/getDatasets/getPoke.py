import requests
from bs4 import BeautifulSoup
import json

# URL of the webpage containing the Pokémon data
url = 'https://game8.co/games/Pokemon-Scarlet-Violet/archives/369171'

# Send a GET request to the webpage
response = requests.get(url)
response.raise_for_status()  # Check for request errors

# Parse the webpage content
soup = BeautifulSoup(response.text, 'html.parser')

# Initialize a list to hold the Pokémon data
pokemonList = []

# Find all rows that contain relevant Pokémon data by locating th tags with the correct class
headers = soup.find_all('th', class_=['Paldea_Dex_cell', 'National_Dex_cell', 'Kitakami_Dex_cell', 'Nat_Dex_cell'])

# Iterate over each header to find corresponding Pokémon data
for header in headers:
    row = header.find_parent('tr')
    if not row:
        continue
    
    # Extract the Pokémon name
    columns = row.find_all('td')
    if len(columns) < 4:
        continue  # Skip rows that don't have enough columns
    
    name_tag = columns[0].find('a')
    name = name_tag.get_text(strip=True)

    # Extract the types
    types = [t.get_text(strip=True) for t in columns[1].find_all('a')]
    primaryType = types[0] if types else 'N/A'
    secondaryType = types[1] if len(types) > 1 else 'N/A'

    # Extract the abilities
    abilities = columns[2].decode_contents().split('<br/>')
    while("" in abilities):
        abilities.remove("")

    stat1_tag = columns[4]
    stat1 = stat1_tag.get_text(strip=True)

    stat2_tag = columns[5]
    stat2 = stat2_tag.get_text(strip=True)

    stat3_tag = columns[6]
    stat3 = stat3_tag.get_text(strip=True)

    stat4_tag = columns[7]
    stat4 = stat4_tag.get_text(strip=True)

    stat5_tag = columns[8]
    stat5 = stat5_tag.get_text(strip=True)

    stat6_tag = columns[9]
    stat6 = stat6_tag.get_text(strip=True)

    if secondaryType == "":
        secondaryType = "N/A"

    # Create a dictionary for the Pokémon
    pokemonData = {
        'name': name,
        'types': {
            'primary': primaryType,
            'secondary': secondaryType
        },
        'abilities': abilities,
        'stats': {
            'HP': stat1,
            'Attack': stat2,
            'Defense': stat3,
            'SpecialAttack': stat4,
            'SpecialDefense': stat5,
            'Speed': stat6
        }
    }

    # Add the Pokémon data to the list
    pokemonList.append(pokemonData)

# Write the Pokémon data to a JSON file
with open('pokemonData.json', 'w', encoding='utf-8') as jsonFile:
    json.dump(pokemonList, jsonFile, ensure_ascii=False, indent=4)

print('Pokémon data has been written to pokemonData.json')
