from typing import Dict
from models import natureModifiers

def validateEvs(evs: Dict[str, int]) -> bool:
    return all(0 <= value <= 252 for value in evs.values()) and sum(evs.values()) <= 510

def validateIvs(ivs: Dict[str, int]) -> bool:
    return all(0 <= value <= 31 for value in ivs.values())

def validateNature(nature: str) -> bool:
    return nature in natureModifiers
