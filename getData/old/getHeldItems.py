import requests
from bs4 import BeautifulSoup
import csv

def scrapeHeldItemsAndBerries():
    # URL of the Serebii held items page
    url = "https://www.serebii.net/scarletviolet/items.shtml"
    
    # Perform an HTTP GET request to fetch the HTML content
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve the webpage.")
        return
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the section containing held items and berries
    itemsSection = soup.find('div', {'id': 'content'}).find_all('h3')
    if not itemsSection:
        print("Could not find the items section.")
        return
    
    heldItemsTable = None
    berriesTable = None
    
    # Loop through all h3 tags to find the correct tables
    for h3_tag in itemsSection:
        if 'Hold Items' in h3_tag.text:
            heldItemsTable = h3_tag.find_next('table')
        elif 'Berries' in h3_tag.text:
            berriesTable = h3_tag.find_next('table')
    
    if not heldItemsTable or not berriesTable:
        print("Could not find the held items or berries tables.")
        return
    
    # Combine the tables into a list
    tables = [heldItemsTable, berriesTable]
    
    # Define the CSV file path
    csvPath = "heldItems.csv"
    
    # Open the CSV file and write the header row
    with open(csvPath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Writing the header row
        headers = ["Name", "Effect"]
        writer.writerow(headers)
        
        # Loop through each table to extract data
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:  # Skipping the header row
                columns = row.find_all('td')
                if len(columns) < 2:
                    continue
                
                # Extracting data from each cell
                itemName = columns[0].text.strip()
                itemEffect = columns[1].text.strip()
                
                # Writing the item data to the CSV file
                print(itemName, itemEffect)
                writer.writerow([itemName, itemEffect])

    print(f"Data has been successfully scraped and saved to {csvPath}.")

# Run the scraping function
scrapeHeldItemsAndBerries()