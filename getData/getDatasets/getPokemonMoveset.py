from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from parseIntoJson import saveExtractedMoveset
from parser import parseCompetitiveMoveset

def extractMovesetsFromLink(link):

    driver = webdriver.Chrome()
    driver.get(link)

    # Dismiss consent popup if it appears.
    try:
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Consent')]"))
        ).click()
    except Exception:
        pass

    allMovesets = []

    try:
        exportButtons = driver.find_elements(By.XPATH, "//button[contains(., 'Export')]")
        numButtons = len(exportButtons)

        for i in range(numButtons):
            exportButtons = driver.find_elements(By.XPATH, "//button[contains(., 'Export')]")
            currentButton = exportButtons[i]

            driver.execute_script(
                "document.querySelectorAll('iframe[aria-label=\"ad\"]').forEach(el => el.style.display='none');"
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", currentButton)
            driver.execute_script("arguments[0].click();", currentButton)

            exportTextArea = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, "//textarea"))
            )
            defaultMoveset = exportTextArea.get_attribute("value") or exportTextArea.text
            allMovesets.append(defaultMoveset)

            try:
                evChangerContainer = WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.MovesetExport-EvChanger"))
                )
            except Exception:
                evChangerContainer = None

            if evChangerContainer:
                radioButtons = evChangerContainer.find_elements(By.CSS_SELECTOR, "input[type='radio'][name='export']")
                for radio in radioButtons:
                    isChecked = radio.get_attribute("checked")
                    if isChecked is not None and isChecked.lower() in ["true", "checked"]:
                        continue
                    driver.execute_script("arguments[0].click();", radio)
                    time.sleep(1)
                    exportTextArea = WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.XPATH, "//textarea"))
                    )
                    toggledMoveset = exportTextArea.get_attribute("value") or exportTextArea.text
                    allMovesets.append(toggledMoveset)
            try:
                driver.execute_script("arguments[0].click();", currentButton)
            except Exception:
                pass

    except Exception:
        pass

    try:
        formatList = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "ul.FormatList"))
        )
        liElements = formatList.find_elements(By.TAG_NAME, "li")
        if liElements:
            tier = liElements[0].text.strip()
        else:
            tier = ""
    except Exception:
        tier = ""

    driver.quit()

    uniqueMovesets = []
    for ms in allMovesets:
        if ms not in uniqueMovesets:
            uniqueMovesets.append(ms)

    print("\nAll unique extracted movesets:")
    for idx, ms in enumerate(uniqueMovesets, start=1):
        print(f"\nMoveset {idx}:")
        print(ms)

    finalEntries = []
    for ms in uniqueMovesets:
        pokemonName, evs, ivs, nature, item, ability, teraType, moves = parseCompetitiveMoveset(ms)
        entry = {
            "pokemonName": pokemonName,
            "evs": evs,
            "ivs": ivs,
            "nature": nature,
            "item": item,
            "ability": ability,
            "teraType": teraType,
            "moves": moves,
            "pokemonTier": tier
        }
        saveExtractedMoveset(ms, tier)
        finalEntries.append(entry)

    print("\nFinal extracted movesets with tiers:")
    for idx, entry in enumerate(finalEntries, start=1):
        print(f"\nMoveset {idx}:")
        print(entry)