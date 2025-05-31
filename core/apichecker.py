import openai
import re
from typing import Dict

def score_moveset_with_gpt(moveset_text: str) -> float:
    """
    Sends the moveset text to the GPT API and returns a score out of 10.
    """
    openai.api_key = "API KEY HERE"
    if not openai.api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

    # Construct the prompt
    prompt = (
        "You are a competitive Pokémon battles expert. "
        "You will be given a moveset for pokemon in pokemon scarlet and violet, rate it out of ten given the pokemon, aka a bad miraidon moveset can get a 2 and a good applin moveset can get a 8/9 etc. "
        "Make sure to rate any unoptimal issues like random 0 ivs or stupid ev spread or useless nature or bad moveset or bad tera types very low. "
        "Rate the following moveset on a scale from 1 to 10, where 1 is very poor and 10 is optimal. Do this for each part of the moveset.  "
        "Make sure to be strict when rating and try to rate based on which is the most meta set for a given pokemon. "
        "Rate the moveset high if it is viable in the current meta, give a middling score if it is a niche but usable moveset and give it a low score if it is garbage (only for the parts that are garbage, for example if garchomp has 252sp atk evs but an atk boosting nature give it a higher score for the nature and a lower score for the evs, same for moves. ). "
        """Reply with the strict output format: 
        Nature: (Score the nature of the pokemon out of 10)
        Evs: (Score the ev set of the pokemon out of 10)
        Ivs: (Score the iv set of the pokemon out of 10, make sure to rank stupid iv sets very low)
        Moves: (Score each moves out of 10 and return the average total for all 4 moves, but only output the average score)
        Ability: (Score the ability out of 10 giving 10 for the best ability the pokemon has)
        Item: (Score the item the pokemon is holding out of 10 given the set)
        Tera Type: (Score the tera type the pokemon has out of 10)
        Overall: (Overall score out of 10)
        Only return the numbers and labels, no explanations.\n"""
        "Here is the moveset.\n\n" + moveset_text
    )

    # Call the ChatCompletion API
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=45
    )

    # Extract the numeric score from the assistant's reply
    reply = response.choices[0].message.content.strip()

     # Parse each line into a dictionary
    scores: Dict[str, float] = {}
    for line in reply.splitlines():
        # Match lines like 'Nature: 3' or 'Moves: 5'
        match = re.match(r"^(Nature|Evs|Ivs|Moves|Ability|Item|Tera Type|Overall):\s*(\d+(?:\.\d+)?)", line)
        if match:
            label = match.group(1)
            value = float(match.group(2))
            # Normalize key names if desired (e.g., "Tera Type" -> "Tera_Type")
            key = label.replace(" ", "_")
            scores[key] = max(1.0, min(10.0, value))

    # Ensure we have all expected keys
    expected = ["Nature", "Evs", "Ivs", "Moves", "Ability", "Item", "Tera Type", "Overall"]
    for label in expected:
        key = label.replace(" ", "_")
        if key not in scores:
            raise ValueError(f"Missing score for '{label}' in model response: {reply}")

    return scores