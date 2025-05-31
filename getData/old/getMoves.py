import requests
from bs4 import BeautifulSoup
import csv

def scrapePokemonMoves():
    # URL of the Pokémon moves page
    url = "https://pokemondb.net/move/all"
    
    # Perform an HTTP GET request to fetch the HTML content
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve the webpage.")
        return
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the table containing the moves
    moveTable = soup.find("table", {"id": "moves"})
    if not moveTable:
        print("Could not find the moves table.")
        return
    
    # Extract the table rows
    rows = moveTable.find_all("tr")
    
    # Define the CSV file path
    csvPath = "moves.csv"
    
    # Open the CSV file and write the header row
    with open(csvPath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Writing the header row
        headers = ["Name", "Type", "Category", "Power", "Accuracy", "PP", "Effect", "Probability"]
        writer.writerow(headers)
        
        # Loop through the table rows to extract the data
        for row in rows[1:]:  # Skipping the header row
            columns = row.find_all("td")
            if len(columns) < 7:
                continue
            
            # Extracting data from each cell
            moveName = columns[0].text.strip()
            moveType = columns[1].find("a").text.strip()  # The type is in an image link
            moveCategoryTag = columns[2].find("img")
            moveCategory = moveCategoryTag["alt"].strip() if moveCategoryTag else "—"  # Extracting the alt text for category or assigning "—"
            movePower = columns[3].text.strip() if columns[3].text.strip() != "—" else None
            moveAccuracy = columns[4].text.strip() if columns[4].text.strip() != "—" else None
            movePP = columns[5].text.strip()
            moveEffect = columns[6].text.strip()
            moveProbability = columns[7].text.strip() if len(columns) > 7 else None
            
            # Writing the move data to the CSV file
            print(moveName, moveType, moveCategory, movePower, moveAccuracy, movePP, moveEffect, moveProbability)
            writer.writerow([moveName, moveType, moveCategory, movePower, moveAccuracy, movePP, moveEffect, moveProbability])

    print(f"Data has been successfully scraped and saved to {csvPath}.")

# Run the scraping function
scrapePokemonMoves()
