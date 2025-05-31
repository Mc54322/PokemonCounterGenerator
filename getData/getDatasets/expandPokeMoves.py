import json
import requests

# Standardize names to match API expectations
name_mapping = {
    "Oinkologne (Male)": "oinkologne-male",
    "Paldean Wooper": "wooper-paldea",
    "Maushold": "maushold-family-of-three",
    "Lycanroc (Midday Form)": "lycanroc-midday",
    "Iron Treads": "iron-treads",
    "Oinkologne (Female)": "oinkologne-female",
    "Lycanroc (Midnight Form)": "lycanroc-midnight",
    "Lycanroc (Dusk Form)": "lycanroc-dusk",
    "Paldean Tauros (Fighting Form)": "tauros-paldea-combat-breed",
    "Paldean Tauros (Fighting/Fire Form)": "tauros-paldea-blaze-breed",
    "Paldean Tauros (Fighting/Water Form)": "tauros-paldea-aqua-breed",
    "Tatsugiri": "tatsugiri-curly",
    "Basculin (White Striped Form)": "basculin-white-striped",
    "Oricorio (Baile Style)": "oricorio-baile",
    "Oricorio (Pa'u Style)": "oricorio-pau",
    "Oricorio (Pom-Pom Style)": "oricorio-pom-pom",
    "Oricorio (Sensu Style)": "oricorio-sensu",
    "Squawkabilly": "squawkabilly-green-plumage",
    "Basculin": "basculin-red-striped",
    "Dudunsparce": "dudunsparce-two-segment",
    "Toxtricity (Amped Form)": "toxtricity-amped",
    "Toxtricity (Low Key Form)": "toxtricity-low-key",
    "Mimikyu": "mimikyu-disguised",
    "Indeedee (Male)": "indeedee-male",
    "Indeedee (Female)": "indeedee-female",
    "Palafin": "palafin-zero",
    "Palafin (Hero Form)": "palafin-hero",
    "Heat Rotom": "rotom-heat",
    "Wash Rotom": "rotom-wash",
    "Frost Rotom": "rotom-frost",
    "Fan Rotom": "rotom-fan",
    "Mow Rotom": "rotom-mow",
    "Eiscue (Ice Face)": "eiscue-ice",
    "Great Tusk": "great-tusk",
    "Scream Tail": "scream-tail",
    "Brute Bonnet": "brute-bonnet",
    "Flutter Mane": "flutter-mane",
    "Slither Wing": "slither-wing",
    "Sandy Shocks": "sandy-shocks",
    "Iron Bundle": "iron-bundle",
    "Iron Hands": "iron-hands",
    "Iron Jugulis": "iron-jugulis",
    "Iron Moth": "iron-moth",
    "Iron Thorns": "iron-thorns",
    "Roaring Moon": "roaring-moon",
    "Iron Valiant": "iron-valiant",
    "Alolan Raichu": "raichu-alola",
    "Alolan Meowth": "meowth-alola",
    "Galarian Meowth": "meowth-galar",
    "Alolan Persian": "persian-alola",
    "Hisuian Growlithe": "growlithe-hisui",
    "Hisuian Arcanine": "arcanine-hisui",
    "Hisuian Voltorb": "voltorb-hisui",
    "Hisuian Electrode": "electrode-hisui",
    "Galarian Weezing": "weezing-galar",
    "Galarian Articuno": "articuno-galar",
    "Galarian Zapdos": "zapdos-galar",
    "Galarian Moltres": "moltres-galar",
    "Hisuian Typhlosion": "typhlosion-hisui",
    "Hisuian Sneasel": "sneasel-hisui",
    "Deoxys (Normal Forme)": "deoxys-normal",
    "Deoxys (Attack Forme)": "deoxys-attack",
    "Deoxys (Defense Forme)": "deoxys-defense",
    "Deoxys (Speed Forme)": "deoxys-speed",
    "Dialga (Origin Forme)": "dialga-origin",
    "Palkia (Origin Forme)": "palkia-origin",
    "Giratina (Altered Forme)": "giratina-altered",
    "Giratina (Origin Forme)": "giratina-origin",
    "Shaymin (Land Forme)": "shaymin-land",
    "Shaymin (Sky Forme)": "shaymin-sky",
    "Hisuian Samurott": "samurott-hisui",
    "Hisuian Lilligant": "lilligant-hisui",
    "Hisuian Zorua": "zorua-hisui",
    "Hisuian Zoroark": "zoroark-hisui",
    "Hisuian Braviary": "braviary-hisui",
    "Tornadus (Incarnate Forme)": "tornadus-incarnate",
    "Tornadus (Therian Forme)": "tornadus-therian",
    "Thundurus (Incarnate Forme)": "thundurus-incarnate",
    "Thundurus (Therian Forme)": "thundurus-therian",
    "Landorus (Incarnate Forme)": "landorus-incarnate",
    "Landorus (Therian Forme)": "landorus-therian",
    "Keldeo": "keldeo-ordinary",
    "Meloetta (Aria Forme)": "meloetta-aria",
    "Meloetta (Pirouette Forme)": "meloetta-pirouette",
    "Hisuian Sliggoo": "sliggoo-hisui",
    "Hisuian Goodra": "goodra-hisui",
    "Hisuian Avalugg": "avalugg-hisui",
    "Hoopa Unbound": "hoopa-unbound",
    "Hisuian Decidueye": "decidueye-hisui",
    "Zacian (Crowned Sword)": "zacian-crowned",
    "Zamazenta (Crowned Shield)": "zamazenta-crowned",
    "Calyrex (Ice Rider)": "calyrex-ice",
    "Calyrex (Shadow Rider)": "calyrex-shadow",
    "Enamorus (Incarnate Forme)": "enamorus-incarnate",
    "Enamorus (Therian Forme)": "enamorus-therian",
    "Gimmighoul (Roaming Form)": "gimmighoul-roaming",
    "Morpeko": "morpeko-full-belly",
    "Basculegion (Male)": "basculegion-male",
    "Basculegion (Female)": "basculegion-female",
    "Bloodmoon Ursaluna": "ursaluna-bloodmoon",
    "Ogerpon (Wellspring Mask)": "ogerpon-wellspring-mask",
    "Ogerpon (Hearthflame Mask)": "ogerpon-hearthflame-mask",
    "Ogerpon (Cornerstone Mask)": "ogerpon-cornerstone-mask",
    "Alolan Exeggutor": "exeggutor-alola",
    "Alolan Diglett": "diglett-alola",
    "Alolan Dugtrio": "dugtrio-alola",
    "Alolan Grimer": "grimer-alola",
    "Alolan Muk": "muk-alola",
    "Galarian Slowpoke": "slowpoke-galar",
    "Galarian Slowbro": "slowbro-galar",
    "Galarian Slowking": "slowking-galar",
    "Alolan Geodude": "geodude-alola",
    "Alolan Graveler": "graveler-alola",
    "Alolan Golem": "golem-alola",
    "Meowstic (Male)": "meowstic-male",
    "Meowstic (Female)": "meowstic-female",
    "Minior (Meteor Form)": "minior-red-meteor",
    "Minior": "minior-red",
    "Hisuian Qwilfish": "qwilfish-hisui",
    "Alolan Sandshrew": "sandshrew-alola",
    "Alolan Sandslash": "sandslash-alola",
    "Alolan Vulpix": "vulpix-alola",
    "Alolan Ninetales":  "ninetales-alola",
    "Gouging Fire": "gouging-fire",
    "Raging Bolt": "raging-bolt",
    "Iron Crown": "iron-crown",
    "Iron Boulder": "iron-boulder",
    "Terapagos (Terastal Form)": "terapagos-terastal",
    "Terapagos (Stellar Form)": "terapagos-stellar",
    "Walking Wake": "walking-wake",
    "Iron Leaves": "iron-leaves",
    "Urshifu (Single Strike Style)": "urshifu-single-strike",
    "Urshifu (Rapid Strike Style)": "urshifu-rapid-strike"
}

# Load the Pokémon data (assuming it's stored in a JSON file named "pokemon.json")
with open('expanded_pokemon_data.json', 'r') as file:
    pokemon_data = json.load(file)

# Function to get Pokémon moves from the PokéAPI
def get_pokemon_moves_from_api(pokemon_name):
    # Only proceed if the name is in the mapping
    if pokemon_name in name_mapping:
        api_name = name_mapping[pokemon_name]
        url = f"https://pokeapi.co/api/v2/pokemon/{api_name}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            moves = [move["move"]["name"] for move in data["moves"]]
            return moves
    
    return []

# Expand the original JSON to include learnable moves from the API
def expand_pokemon_with_moves(pokemon_data):
    for pokemon in pokemon_data:
        name = pokemon["name"]
        # Only get moves for mapped names
        if name in name_mapping.keys():
            pokemon["learnable_moves"] = get_pokemon_moves_from_api(name)
            print(pokemon)
    return pokemon_data

expanded_pokemon_data = expand_pokemon_with_moves(pokemon_data)

# Save the expanded Pokémon data to a new JSON file
with open('expanded_pokemon_data.json', 'w') as file:
    json.dump(expanded_pokemon_data, file, indent=4)

print("Expanded Pokémon data saved to 'expanded_pokemon_data.json'")