import asyncio
import uuid
import re
from functools import lru_cache
from typing import Tuple
from poke_env.player.random_player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env import AccountConfiguration
from poke_env.exceptions import ShowdownException

# Use custom game so single-mon squads are allowed
battle_format = "gen9customgame"

def sanitize_team(team_str: str) -> str:
    lines = team_str.splitlines()
    clean = []
    for line in lines:
        line = re.sub(r"^[^\w@-]+", "", line).strip()
        if line:
            clean.append(line)
    return "\n".join(clean)

class MovesetPlayer(RandomPlayer):
    def __init__(self, team_str: str):
        uname = f"u{uuid.uuid4().hex[:8]}"
        acct = AccountConfiguration(username=uname, password="")
        super().__init__(
            account_configuration=acct,
            server_configuration=LocalhostServerConfiguration,
            battle_format=battle_format,
            team=sanitize_team(team_str),
            accept_open_team_sheet=True,
            save_replays=False,  
        )

    def on_popup(self, message: str) -> None:
        raise ShowdownException(message)

async def simulate_battle(
    challenger_team: str,
    counter_team: str,
    num_battles: int,
) -> tuple[int, int]:
    """
    Run num_battles matches between challenger_team and counter_team.
    Returns (wins_by_counter, wins_by_challenger).
    """
    # instantiate two fresh players
    playerA = MovesetPlayer(challenger_team)
    playerB = MovesetPlayer(counter_team)

    # this kicks off all the battles but returns None
    await playerA.battle_against(playerB, n_battles=num_battles)

    # now read off the results
    wins_by_challenger = playerA.n_won_battles
    wins_by_counter    = playerB.n_won_battles
    finished           = playerA.n_finished_battles  # should equal playerB.n_finished_battles

    if finished == 0:
        # no battle ever actually ran
        return 0, 0

    return wins_by_counter, wins_by_challenger

@lru_cache(maxsize=2048)
def has_high_winrate(
    challenger_team: str,
    counter_team: str,
    win_rate: Tuple[float, ...],
    num_battles: int = 1000,
) -> bool:
    wins, losses = asyncio.run(
        simulate_battle(challenger_team, counter_team, num_battles)
    )
    total = wins + losses
    if total == 0:
        return False
    for i in win_rate:
        if (wins / total) >= i:
            return i, True
    return None, False

if __name__ == "__main__":
    teamA = """
    Annihilape @ Leftovers
    Ability: Defiant
    EVs: 240 HP / 252 SpD / 16 Spe
    Tera Type: Water
    Careful Nature
    - Bulk Up
    - Taunt
    - Drain Punch
    - Rage Fist
    """
    teamB = """
Hawlucha @ Psychic Seed
Ability: Unburden
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
Tera Type: Fire
- Swords Dance
- Close Combat
- Acrobatics
- Encore
    """

    winRates = (0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5)

    percent, check = has_high_winrate(teamA, teamB, winRates)
    if check == True:
        print(f"Won over {percent * 100}%")
    else: 
        print("Lost")