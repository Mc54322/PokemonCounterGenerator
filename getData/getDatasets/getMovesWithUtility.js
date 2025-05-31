"use strict";

const fs = require("fs");
const { Moves } = require("./moves.js");

// Constants (from Python)
const damageMultiplier = 2.0;
const selfBoostMultiplier = 100;
const enemyBoostMultiplier = 50;
const recoilPenaltyFactor = 1.0;
const rechargePenalty = 50;
const healingBonus = 100;
const selfDestructPenalty = 350;

// Field effect mappings
const volatileStatusValues = {
  "flinch": 5, "aquaring": 20, "attract": 10, "confusion": 10,
  "banefulbunker": 80, "bide": 5, "partiallytrapped": 10, "burningbulwark": 70,
  "charge": 5, "curse": 30, "defensecurl": 0, "destinybond": 40,
  "protect": 100, "disable": 10, "dragoncheer": 5, "electrify": 10,
  "embargo": 10, "encore": 40, "endure": 10, "focusenergy": 5,
  "followme": 10, "foresight": 10, "gastroacid": 15, "grudge": 5,
  "healblock": 30, "helpinghand": 10, "imprison": 5, "ingrain": 30,
  "kingsshield": 100, "laserfocus": 5, "leechseed": 30, "magiccoat": 5,
  "magnetrise": 5, "maxguard": 100, "minimize": 5, "miracleeye": 5,
  "nightmare": 10, "noretreat": 5, "obstruct": 100, "octolock": 100,
  "powder": 5, "powershift": 5, "powertrick": 5, "ragepowder": 100,
  "saltcure": 5, "substitute": 20, "silktrap": 5, "smackdown": 5,
  "snatch": 10, "sparklingaria": 5, "spikyshield": 100, "spotlight": 5,
  "stockpile": 20, "syrupbomb": 5, "tarshot": 5, "taunt": 20,
  "telekinesis": 5, "torment": 5, "yawn": 5
};

const statusValues = {
  "psn": 20, "brn": 20, "frz": 20, "par": 15, "tox": 30, "slp": 15
};

const sideConditionValues = {
  "auroraveil": 60, "craftyshield": 50, "lightscreen": 50, "luckychant": 40,
  "matblock": 40, "mist": 30, "quickguard": 50, "reflect": 50,
  "safeguard": 40, "spikes": 40, "stealthrock": 60, "stickyweb": 40,
  "tailwind": 50, "toxicspikes": 50, "wideguard": 50
};

const pseudoWeatherValues = {
  "fairylock": 40, "gravity": 30, "iondeluge": 40, "magicroom": 30,
  "mudsport": 20, "trickroom": 60, "watersport": 20, "wonderroom": 30
};

const weatherValues = {
  "snow": 30, "hail": 30, "raindance": 40, "sandstorm": 40, "sunnyday": 40
};

const terrainValues = {
  "electricterrain": 40, "grassyterrain": 40, "mistyterrain": 40, "psychicterrain": 40
};

// Revised utility calculation function (ported from Python)
function calculateMoveUtility(move, pokemon) {
  let utility = 0.0;

  // 1. Damaging component.
  if (move.category !== "Status" && typeof move.basePower === "number" && move.basePower > 0) {
    const moveType = (move.type || "").toLowerCase();
    const pokeTypes = (pokemon.types) || {};
    const primaryType = (pokeTypes.primary || "").toLowerCase();
    let secondaryType = "";
    if (pokeTypes.secondary && pokeTypes.secondary.toLowerCase() !== "n/a") {
      secondaryType = pokeTypes.secondary.toLowerCase();
    }
    const stab = (moveType && (moveType === primaryType || moveType === secondaryType)) ? 1.5 : 1;
    utility += move.basePower * damageMultiplier * stab;
    if (move.secondary && typeof move.secondary.chance === "number") {
      utility += move.secondary.chance * 0.5;
    }
    if (move.drain) {
      utility += 10;
    }
  }

  // 2. Move-applied boosts (from move.boosts).
  if (move.boosts && typeof move.boosts === "object") {
    const targetValue = move.target || "";
    const isSelf = (targetValue === "self" ||
                    targetValue === "adjacentAllyOrSelf" ||
                    (move.category === "Status" && move.accuracy === true));
    let pos = 0, neg = 0;
    for (const stat in move.boosts) {
      const val = move.boosts[stat];
      if (val > 0) {
        pos += val;
      } else if (val < 0) {
        neg += Math.abs(val);
      }
    }
    if (isSelf) {
      utility += pos * selfBoostMultiplier;
      utility -= neg * selfBoostMultiplier;
    } else {
      utility -= pos * enemyBoostMultiplier;
      utility += neg * enemyBoostMultiplier;
    }
  }

  // 3. Self-effects: move.self.boosts.
  if (move.self && move.self.boosts && typeof move.self.boosts === "object") {
    let pos = 0, neg = 0;
    for (const stat in move.self.boosts) {
      const val = move.self.boosts[stat];
      if (val > 0) {
        pos += val;
      } else if (val < 0) {
        neg += Math.abs(val);
      }
    }
    utility += pos * selfBoostMultiplier;
    utility -= neg * selfBoostMultiplier;
  }

  // 4. Recoil penalty.
  if (move.recoil && Array.isArray(move.recoil) && move.basePower) {
    const [num, den] = move.recoil;
    const fraction = num / den;
    utility -= move.basePower * fraction * recoilPenaltyFactor;
  }

  // 5. Recharge penalty.
  if (move.self && typeof move.self === "object") {
    if (move.self.volatileStatus &&
        typeof move.self.volatileStatus === "string" &&
        move.self.volatileStatus.toLowerCase() === "mustrecharge") {
      utility -= rechargePenalty;
    }
  }

  // 6. Field effects.
  if (typeof move.volatileStatus === "string") {
    utility += volatileStatusValues[move.volatileStatus.toLowerCase()] || 0;
  }
  if (move.secondary && typeof move.secondary === "object") {
    if (move.secondary.volatileStatus && typeof move.secondary.volatileStatus === "string") {
      utility += volatileStatusValues[move.secondary.volatileStatus.toLowerCase()] || 0;
    }
    if (move.secondary.status && typeof move.secondary.status === "string") {
      utility += statusValues[move.secondary.status.toLowerCase()] || 0;
    }
  }
  if (move.sideCondition && typeof move.sideCondition === "string") {
    utility += sideConditionValues[move.sideCondition.toLowerCase()] || 0;
  }
  if (move.pseudoWeather && typeof move.pseudoWeather === "string") {
    utility += pseudoWeatherValues[move.pseudoWeather.toLowerCase()] || 0;
  }
  if (move.weather && typeof move.weather === "string") {
    utility += weatherValues[move.weather.toLowerCase()] || 0;
  }
  if (move.terrain && typeof move.terrain === "string") {
    utility += terrainValues[move.terrain.toLowerCase()] || 0;
  }

  // 7. Healing bonus.
  if (move.flags && typeof move.flags === "object" && move.flags.heal && !("drain" in move)) {
    utility += healingBonus;
  }

  // 8. Self-destruct penalty.
  if (move.selfdestruct) {
    utility -= selfDestructPenalty;
  }

  // 9. PP factor.
  if (typeof move.pp === "number") {
    utility += (100 - move.pp) * 0.05;
  }

  return utility;
}

// Create a new object with annotated moves.
const annotatedMoves = {};

// Process each move in Moves.
for (const moveName in Moves) {
  if (!Moves.hasOwnProperty(moveName)) continue;
  // Make a shallow copy of the move.
  const move = Object.assign({}, Moves[moveName]);

  // Remove any function properties.
  for (const prop in move) {
    if (typeof move[prop] === "function") {
      delete move[prop];
    }
  }

  // Dummy Pokémon for calculations.
  const dummyNoStab = { types: { primary: "none", secondary: "n/a" } };
  const moveType = move.type || "none";
  const dummyStab = { types: { primary: moveType, secondary: "n/a" } };

  // Compute utility without STAB and with STAB.
  move.utility = Math.round(calculateMoveUtility(move, dummyNoStab) * 100) / 100;
  move.stabutility = Math.round(calculateMoveUtility(move, dummyStab) * 100) / 100;

  annotatedMoves[moveName] = move;
}

// Write the annotated moves to a JSON file.
fs.writeFileSync("annotatedMoves.json", JSON.stringify(annotatedMoves, null, 2));
console.log("Annotated moves saved to annotatedMoves.json");
