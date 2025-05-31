import requests
from bs4 import BeautifulSoup
import csv

def scrapePokemonAbilities():
    # URL of the Pokémon abilities page
    url = "https://pokemondb.net/ability"
    
    # Perform an HTTP GET request to fetch the HTML content
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve the webpage.")
        return
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the table containing the abilities
    abilityTable = soup.find("table", {"id": "abilities"})
    if not abilityTable:
        print("Could not find the abilities table.")
        return
    
    # Extract the table rows
    rows = abilityTable.find_all("tr")
    
    # Define the CSV file path
    csvPath = "abilities.csv"
    
    # Open the CSV file and write the header row
    with open(csvPath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Writing the header row
        headers = ["Name", "Description"]
        writer.writerow(headers)
        
        # Loop through the table rows to extract the data
        for row in rows[1:]:  # Skipping the header row
            columns = row.find_all("td")
            if len(columns) < 2:
                continue
            
            # Extracting data from each cell
            abilityName = columns[0].text.strip()
            abilityDescription = columns[2].text.strip()
            
            # Writing the ability data to the CSV file
            print(abilityName, abilityDescription)
            writer.writerow([abilityName, abilityDescription])

    print(f"Data has been successfully scraped and saved to {csvPath}.")

# Run the scraping function
scrapePokemonAbilities()
