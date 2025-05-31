from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Import the moveset extraction function from getPokemonMoveset.py.
# We assume the function is defined as extractMovesetsFromLink in that file,
# so we alias it here as getMovesetsFromLink.
from getPokemonMoveset import extractMovesetsFromLink as getMovesetsFromLink

def getAllPokemonLinks():
    driver = webdriver.Chrome()
    driver.get("https://www.smogon.com/dex/sv/pokemon/")

    try:
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Consent')]"))
        ).click()
    except Exception:
        pass

    # Wait for the initial Pokémon links to load.
    WebDriverWait(driver, 15).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a[href^='/dex/sv/pokemon/']"))
    )

    collectedLinks = []
    while True:
        newLinkFound = False
        # Collect links before scrolling
        linkElements = driver.find_elements(By.CSS_SELECTOR, "a[href^='/dex/sv/pokemon/']")
        for elem in linkElements:
            linkHref = elem.get_attribute("href")
            if linkHref and linkHref not in collectedLinks:
                collectedLinks.append(linkHref)
                newLinkFound = True
                if elem.text.strip().lower() == "zweilous":
                    driver.quit()
                    return collectedLinks

        # Slowly scroll to the bottom in increments and collect links after each increment.
        lastHeight = driver.execute_script("return document.body.scrollHeight")
        currentPosition = 0
        while currentPosition < lastHeight:
            currentPosition += 300
            driver.execute_script("window.scrollTo(0, arguments[0]);", currentPosition)
            time.sleep(0.3)
            # Collect links after each scroll increment
            linkElements = driver.find_elements(By.CSS_SELECTOR, "a[href^='/dex/sv/pokemon/']")
            for elem in linkElements:
                linkHref = elem.get_attribute("href")
                if linkHref and linkHref not in collectedLinks:
                    collectedLinks.append(linkHref)
                    newLinkFound = True
                    if elem.text.strip().lower() == "zweilous":
                        driver.quit()
                        return collectedLinks
        if not newLinkFound:
            break
        time.sleep(3)
    driver.quit()
    return collectedLinks

def processAllPokemonMovesets():
    allLinks = getAllPokemonLinks()
    for link in allLinks:
        # For each link, call the moveset extraction function.
        #print(link)
        getMovesetsFromLink(link)

if __name__ == "__main__":
    processAllPokemonMovesets()
