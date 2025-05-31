"use strict";
/*

Ratings and how they work:

-1: Detrimental
      An ability that severely harms the user.
    ex. Defeatist, Slow Start

 0: Useless
      An ability with no overall benefit in a singles battle.
    ex. Color Change, Plus

 1: Ineffective
      An ability that has minimal effect or is only useful in niche situations.
    ex. Light Metal, Suction Cups

 2: Useful
      An ability that can be generally useful.
    ex. Flame Body, Overcoat

 3: Effective
      An ability with a strong effect on the user or foe.
    ex. Chlorophyll, Sturdy

 4: Very useful
      One of the more popular abilities. It requires minimal support to be effective.
    ex. Adaptability, Magic Bounce

 5: Essential
      The sort of ability that defines metagames.
    ex. Imposter, Shadow Tag

*/
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Abilities = void 0;
exports.Abilities = {
    noability: {
        isNonstandard: "Past",
        flags: {},
        name: "No Ability",
        rating: 0.1,
        num: 0,
    },
    adaptability: {
        onModifySTAB: function (stab, source, target, move) {
            if (move.forceSTAB || source.hasType(move.type)) {
                if (stab === 2) {
                    return 2.25;
                }
                return 2;
            }
        },
        flags: {},
        name: "Adaptability",
        rating: 4,
        num: 91,
    },
    aerilate: {
        onModifyTypePriority: -1,
        onModifyType: function (move, pokemon) {
            var noModifyType = [
                'judgment', 'multiattack', 'naturalgift', 'revelationdance', 'technoblast', 'terrainpulse', 'weatherball',
            ];
            if (move.type === 'Normal' && !noModifyType.includes(move.id) &&
                !(move.isZ && move.category !== 'Status') && !(move.name === 'Tera Blast' && pokemon.terastallized)) {
                move.type = 'Flying';
                move.typeChangerBoosted = this.effect;
            }
        },
        onBasePowerPriority: 23,
        onBasePower: function (basePower, pokemon, target, move) {
            if (move.typeChangerBoosted === this.effect)
                return this.chainModify([4915, 4096]);
        },
        flags: {},
        name: "Aerilate",
        rating: 4,
        num: 184,
    },
    aftermath: {
        onDamagingHitOrder: 1,
        onDamagingHit: function (damage, target, source, move) {
            if (!target.hp && this.checkMoveMakesContact(move, source, target, true)) {
                this.damage(source.baseMaxhp / 4, source, target);
            }
        },
        flags: {},
        name: "Aftermath",
        rating: 2,
        num: 106,
    },
    airlock: {
        onSwitchIn: function (pokemon) {
            // Air Lock does not activate when Skill Swapped or when Neutralizing Gas leaves the field
            this.add('-ability', pokemon, 'Air Lock');
            this.effect.onStart.call(this, pokemon);
        },
        onStart: function (pokemon) {
            pokemon.abilityState.ending = false; // Clear the ending flag
            this.eachEvent('WeatherChange', this.effect);
        },
        onEnd: function (pokemon) {
            pokemon.abilityState.ending = true;
            this.eachEvent('WeatherChange', this.effect);
        },
        suppressWeather: true,
        flags: {},
        name: "Air Lock",
        rating: 1.5,
        num: 76,
    },
    analytic: {
        onBasePowerPriority: 21,
        onBasePower: function (basePower, pokemon) {
            var boosted = true;
            for (var _i = 0, _a = this.getAllActive(); _i < _a.length; _i++) {
                var target = _a[_i];
                if (target === pokemon)
                    continue;
                if (this.queue.willMove(target)) {
                    boosted = false;
                    break;
                }
            }
            if (boosted) {
                this.debug('Analytic boost');
                return this.chainModify([5325, 4096]);
            }
        },
        flags: {},
        name: "Analytic",
        rating: 2.5,
        num: 148,
    },
    angerpoint: {
        onHit: function (target, source, move) {
            if (!target.hp)
                return;
            if ((move === null || move === void 0 ? void 0 : move.effectType) === 'Move' && target.getMoveHitData(move).crit) {
                this.boost({ atk: 12 }, target, target);
            }
        },
        flags: {},
        name: "Anger Point",
        rating: 1,
        num: 83,
    },
    angershell: {
        onDamage: function (damage, target, source, effect) {
            if (effect.effectType === "Move" &&
                !effect.multihit &&
                (!effect.negateSecondary && !(effect.hasSheerForce && source.hasAbility('sheerforce')))) {
                this.effectState.checkedAngerShell = false;
            }
            else {
                this.effectState.checkedAngerShell = true;
            }
        },
        onTryEatItem: function (item) {
            var healingItems = [
                'aguavberry', 'enigmaberry', 'figyberry', 'iapapaberry', 'magoberry', 'sitrusberry', 'wikiberry', 'oranberry', 'berryjuice',
            ];
            if (healingItems.includes(item.id)) {
                return this.effectState.checkedAngerShell;
            }
            return true;
        },
        onAfterMoveSecondary: function (target, source, move) {
            this.effectState.checkedAngerShell = true;
            if (!source || source === target || !target.hp || !move.totalDamage)
                return;
            var lastAttackedBy = target.getLastAttackedBy();
            if (!lastAttackedBy)
                return;
            var damage = move.multihit ? move.totalDamage : lastAttackedBy.damage;
            if (target.hp <= target.maxhp / 2 && target.hp + damage > target.maxhp / 2) {
                this.boost({ atk: 1, spa: 1, spe: 1, def: -1, spd: -1 }, target, target);
            }
        },
        flags: {},
        name: "Anger Shell",
        rating: 3,
        num: 271,
    },
    anticipation: {
        onStart: function (pokemon) {
            for (var _i = 0, _a = pokemon.foes(); _i < _a.length; _i++) {
                var target = _a[_i];
                for (var _b = 0, _c = target.moveSlots; _b < _c.length; _b++) {
                    var moveSlot = _c[_b];
                    var move = this.dex.moves.get(moveSlot.move);
                    if (move.category === 'Status')
                        continue;
                    var moveType = move.id === 'hiddenpower' ? target.hpType : move.type;
                    if (this.dex.getImmunity(moveType, pokemon) && this.dex.getEffectiveness(moveType, pokemon) > 0 ||
                        move.ohko) {
                        this.add('-ability', pokemon, 'Anticipation');
                        return;
                    }
                }
            }
        },
        flags: {},
        name: "Anticipation",
        rating: 0.5,
        num: 107,
    },
    arenatrap: {
        onFoeTrapPokemon: function (pokemon) {
            if (!pokemon.isAdjacent(this.effectState.target))
                return;
            if (pokemon.isGrounded()) {
                pokemon.tryTrap(true);
            }
        },
        onFoeMaybeTrapPokemon: function (pokemon, source) {
            if (!source)
                source = this.effectState.target;
            if (!source || !pokemon.isAdjacent(source))
                return;
            if (pokemon.isGrounded(!pokemon.knownType)) { // Negate immunity if the type is unknown
                pokemon.maybeTrapped = true;
            }
        },
        flags: {},
        name: "Arena Trap",
        rating: 5,
        num: 71,
    },
    armortail: {
        onFoeTryMove: function (target, source, move) {
            var targetAllExceptions = ['perishsong', 'flowershield', 'rototiller'];
            if (move.target === 'foeSide' || (move.target === 'all' && !targetAllExceptions.includes(move.id))) {
                return;
            }
            var armorTailHolder = this.effectState.target;
            if ((source.isAlly(armorTailHolder) || move.target === 'all') && move.priority > 0.1) {
                this.attrLastMove('[still]');
                this.add('cant', armorTailHolder, 'ability: Armor Tail', move, "[of] ".concat(target));
                return false;
            }
        },
        flags: { breakable: 1 },
        name: "Armor Tail",
        rating: 2.5,
        num: 296,
    },
    aromaveil: {
        onAllyTryAddVolatile: function (status, target, source, effect) {
            if (['attract', 'disable', 'encore', 'healblock', 'taunt', 'torment'].includes(status.id)) {
                if (effect.effectType === 'Move') {
                    var effectHolder = this.effectState.target;
                    this.add('-block', target, 'ability: Aroma Veil', "[of] ".concat(effectHolder));
                }
                return null;
            }
        },
        flags: { breakable: 1 },
        name: "Aroma Veil",
        rating: 2,
        num: 165,
    },
    asoneglastrier: {
        onSwitchInPriority: 1,
        onStart: function (pokemon) {
            if (this.effectState.unnerved)
                return;
            this.add('-ability', pokemon, 'As One');
            this.add('-ability', pokemon, 'Unnerve');
            this.effectState.unnerved = true;
        },
        onEnd: function () {
            this.effectState.unnerved = false;
        },
        onFoeTryEatItem: function () {
            return !this.effectState.unnerved;
        },
        onSourceAfterFaint: function (length, target, source, effect) {
            if (effect && effect.effectType === 'Move') {
                this.boost({ atk: length }, source, source, this.dex.abilities.get('chillingneigh'));
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, cantsuppress: 1 },
        name: "As One (Glastrier)",
        rating: 3.5,
        num: 266,
    },
    asonespectrier: {
        onSwitchInPriority: 1,
        onStart: function (pokemon) {
            if (this.effectState.unnerved)
                return;
            this.add('-ability', pokemon, 'As One');
            this.add('-ability', pokemon, 'Unnerve');
            this.effectState.unnerved = true;
        },
        onEnd: function () {
            this.effectState.unnerved = false;
        },
        onFoeTryEatItem: function () {
            return !this.effectState.unnerved;
        },
        onSourceAfterFaint: function (length, target, source, effect) {
            if (effect && effect.effectType === 'Move') {
                this.boost({ spa: length }, source, source, this.dex.abilities.get('grimneigh'));
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, cantsuppress: 1 },
        name: "As One (Spectrier)",
        rating: 3.5,
        num: 267,
    },
    aurabreak: {
        onStart: function (pokemon) {
            if (this.suppressingAbility(pokemon))
                return;
            this.add('-ability', pokemon, 'Aura Break');
        },
        onAnyTryPrimaryHit: function (target, source, move) {
            if (target === source || move.category === 'Status')
                return;
            move.hasAuraBreak = true;
        },
        flags: { breakable: 1 },
        name: "Aura Break",
        rating: 1,
        num: 188,
    },
    baddreams: {
        onResidualOrder: 28,
        onResidualSubOrder: 2,
        onResidual: function (pokemon) {
            if (!pokemon.hp)
                return;
            for (var _i = 0, _a = pokemon.foes(); _i < _a.length; _i++) {
                var target = _a[_i];
                if (target.status === 'slp' || target.hasAbility('comatose')) {
                    this.damage(target.baseMaxhp / 8, target, pokemon);
                }
            }
        },
        flags: {},
        name: "Bad Dreams",
        rating: 1.5,
        num: 123,
    },
    ballfetch: {
        flags: {},
        name: "Ball Fetch",
        rating: 0,
        num: 237,
    },
    battery: {
        onAllyBasePowerPriority: 22,
        onAllyBasePower: function (basePower, attacker, defender, move) {
            if (attacker !== this.effectState.target && move.category === 'Special') {
                this.debug('Battery boost');
                return this.chainModify([5325, 4096]);
            }
        },
        flags: {},
        name: "Battery",
        rating: 0,
        num: 217,
    },
    battlearmor: {
        onCriticalHit: false,
        flags: { breakable: 1 },
        name: "Battle Armor",
        rating: 1,
        num: 4,
    },
    battlebond: {
        onSourceAfterFaint: function (length, target, source, effect) {
            if ((effect === null || effect === void 0 ? void 0 : effect.effectType) !== 'Move')
                return;
            if (source.abilityState.battleBondTriggered)
                return;
            if (source.species.id === 'greninjabond' && source.hp && !source.transformed && source.side.foePokemonLeft()) {
                this.boost({ atk: 1, spa: 1, spe: 1 }, source, source, this.effect);
                this.add('-activate', source, 'ability: Battle Bond');
                source.abilityState.battleBondTriggered = true;
            }
        },
        onModifyMovePriority: -1,
        onModifyMove: function (move, attacker) {
            if (move.id === 'watershuriken' && attacker.species.name === 'Greninja-Ash' &&
                !attacker.transformed) {
                move.multihit = 3;
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, cantsuppress: 1 },
        name: "Battle Bond",
        rating: 3.5,
        num: 210,
    },
    beadsofruin: {
        onStart: function (pokemon) {
            if (this.suppressingAbility(pokemon))
                return;
            this.add('-ability', pokemon, 'Beads of Ruin');
        },
        onAnyModifySpD: function (spd, target, source, move) {
            var _a;
            var abilityHolder = this.effectState.target;
            if (target.hasAbility('Beads of Ruin'))
                return;
            if (!((_a = move.ruinedSpD) === null || _a === void 0 ? void 0 : _a.hasAbility('Beads of Ruin')))
                move.ruinedSpD = abilityHolder;
            if (move.ruinedSpD !== abilityHolder)
                return;
            this.debug('Beads of Ruin SpD drop');
            return this.chainModify(0.75);
        },
        flags: {},
        name: "Beads of Ruin",
        rating: 4.5,
        num: 284,
    },
    beastboost: {
        onSourceAfterFaint: function (length, target, source, effect) {
            var _a;
            if (effect && effect.effectType === 'Move') {
                var bestStat = source.getBestStat(true, true);
                this.boost((_a = {}, _a[bestStat] = length, _a), source);
            }
        },
        flags: {},
        name: "Beast Boost",
        rating: 3.5,
        num: 224,
    },
    berserk: {
        onDamage: function (damage, target, source, effect) {
            if (effect.effectType === "Move" &&
                !effect.multihit &&
                (!effect.negateSecondary && !(effect.hasSheerForce && source.hasAbility('sheerforce')))) {
                this.effectState.checkedBerserk = false;
            }
            else {
                this.effectState.checkedBerserk = true;
            }
        },
        onTryEatItem: function (item) {
            var healingItems = [
                'aguavberry', 'enigmaberry', 'figyberry', 'iapapaberry', 'magoberry', 'sitrusberry', 'wikiberry', 'oranberry', 'berryjuice',
            ];
            if (healingItems.includes(item.id)) {
                return this.effectState.checkedBerserk;
            }
            return true;
        },
        onAfterMoveSecondary: function (target, source, move) {
            this.effectState.checkedBerserk = true;
            if (!source || source === target || !target.hp || !move.totalDamage)
                return;
            var lastAttackedBy = target.getLastAttackedBy();
            if (!lastAttackedBy)
                return;
            var damage = move.multihit && !move.smartTarget ? move.totalDamage : lastAttackedBy.damage;
            if (target.hp <= target.maxhp / 2 && target.hp + damage > target.maxhp / 2) {
                this.boost({ spa: 1 }, target, target);
            }
        },
        flags: {},
        name: "Berserk",
        rating: 2,
        num: 201,
    },
    bigpecks: {
        onTryBoost: function (boost, target, source, effect) {
            if (source && target === source)
                return;
            if (boost.def && boost.def < 0) {
                delete boost.def;
                if (!effect.secondaries && effect.id !== 'octolock') {
                    this.add("-fail", target, "unboost", "Defense", "[from] ability: Big Pecks", "[of] ".concat(target));
                }
            }
        },
        flags: { breakable: 1 },
        name: "Big Pecks",
        rating: 0.5,
        num: 145,
    },
    blaze: {
        onModifyAtkPriority: 5,
        onModifyAtk: function (atk, attacker, defender, move) {
            if (move.type === 'Fire' && attacker.hp <= attacker.maxhp / 3) {
                this.debug('Blaze boost');
                return this.chainModify(1.5);
            }
        },
        onModifySpAPriority: 5,
        onModifySpA: function (atk, attacker, defender, move) {
            if (move.type === 'Fire' && attacker.hp <= attacker.maxhp / 3) {
                this.debug('Blaze boost');
                return this.chainModify(1.5);
            }
        },
        flags: {},
        name: "Blaze",
        rating: 2,
        num: 66,
    },
    bulletproof: {
        onTryHit: function (pokemon, target, move) {
            if (move.flags['bullet']) {
                this.add('-immune', pokemon, '[from] ability: Bulletproof');
                return null;
            }
        },
        flags: { breakable: 1 },
        name: "Bulletproof",
        rating: 3,
        num: 171,
    },
    cheekpouch: {
        onEatItem: function (item, pokemon) {
            this.heal(pokemon.baseMaxhp / 3);
        },
        flags: {},
        name: "Cheek Pouch",
        rating: 2,
        num: 167,
    },
    chillingneigh: {
        onSourceAfterFaint: function (length, target, source, effect) {
            if (effect && effect.effectType === 'Move') {
                this.boost({ atk: length }, source);
            }
        },
        flags: {},
        name: "Chilling Neigh",
        rating: 3,
        num: 264,
    },
    chlorophyll: {
        onModifySpe: function (spe, pokemon) {
            if (['sunnyday', 'desolateland'].includes(pokemon.effectiveWeather())) {
                return this.chainModify(2);
            }
        },
        flags: {},
        name: "Chlorophyll",
        rating: 3,
        num: 34,
    },
    clearbody: {
        onTryBoost: function (boost, target, source, effect) {
            if (source && target === source)
                return;
            var showMsg = false;
            var i;
            for (i in boost) {
                if (boost[i] < 0) {
                    delete boost[i];
                    showMsg = true;
                }
            }
            if (showMsg && !effect.secondaries && effect.id !== 'octolock') {
                this.add("-fail", target, "unboost", "[from] ability: Clear Body", "[of] ".concat(target));
            }
        },
        flags: { breakable: 1 },
        name: "Clear Body",
        rating: 2,
        num: 29,
    },
    cloudnine: {
        onSwitchIn: function (pokemon) {
            // Cloud Nine does not activate when Skill Swapped or when Neutralizing Gas leaves the field
            this.add('-ability', pokemon, 'Cloud Nine');
            this.effect.onStart.call(this, pokemon);
        },
        onStart: function (pokemon) {
            pokemon.abilityState.ending = false; // Clear the ending flag
            this.eachEvent('WeatherChange', this.effect);
        },
        onEnd: function (pokemon) {
            pokemon.abilityState.ending = true;
            this.eachEvent('WeatherChange', this.effect);
        },
        suppressWeather: true,
        flags: {},
        name: "Cloud Nine",
        rating: 1.5,
        num: 13,
    },
    colorchange: {
        onAfterMoveSecondary: function (target, source, move) {
            if (!target.hp)
                return;
            var type = move.type;
            if (target.isActive && move.effectType === 'Move' && move.category !== 'Status' &&
                type !== '???' && !target.hasType(type)) {
                if (!target.setType(type))
                    return false;
                this.add('-start', target, 'typechange', type, '[from] ability: Color Change');
                if (target.side.active.length === 2 && target.position === 1) {
                    // Curse Glitch
                    var action = this.queue.willMove(target);
                    if (action && action.move.id === 'curse') {
                        action.targetLoc = -1;
                    }
                }
            }
        },
        flags: {},
        name: "Color Change",
        rating: 0,
        num: 16,
    },
    comatose: {
        onStart: function (pokemon) {
            this.add('-ability', pokemon, 'Comatose');
        },
        onSetStatus: function (status, target, source, effect) {
            if (effect === null || effect === void 0 ? void 0 : effect.status) {
                this.add('-immune', target, '[from] ability: Comatose');
            }
            return false;
        },
        // Permanent sleep "status" implemented in the relevant sleep-checking effects
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, cantsuppress: 1 },
        name: "Comatose",
        rating: 4,
        num: 213,
    },
    commander: {
        onAnySwitchInPriority: -2,
        onAnySwitchIn: function () {
            this.effect.onUpdate.call(this, this.effectState.target);
        },
        onStart: function (pokemon) {
            this.effect.onUpdate.call(this, pokemon);
        },
        onUpdate: function (pokemon) {
            var _a;
            if (this.gameType !== 'doubles')
                return;
            // don't run between when a Pokemon switches in and the resulting onSwitchIn event
            if (((_a = this.queue.peek()) === null || _a === void 0 ? void 0 : _a.choice) === 'runSwitch')
                return;
            var ally = pokemon.allies()[0];
            if (pokemon.switchFlag || (ally === null || ally === void 0 ? void 0 : ally.switchFlag))
                return;
            if (!ally || pokemon.baseSpecies.baseSpecies !== 'Tatsugiri' || ally.baseSpecies.baseSpecies !== 'Dondozo') {
                // Handle any edge cases
                if (pokemon.getVolatile('commanding'))
                    pokemon.removeVolatile('commanding');
                return;
            }
            if (!pokemon.getVolatile('commanding')) {
                // If Dondozo already was commanded this fails
                if (ally.getVolatile('commanded'))
                    return;
                // Cancel all actions this turn for pokemon if applicable
                this.queue.cancelAction(pokemon);
                // Add volatiles to both pokemon
                this.add('-activate', pokemon, 'ability: Commander', "[of] ".concat(ally));
                pokemon.addVolatile('commanding');
                ally.addVolatile('commanded', pokemon);
                // Continued in conditions.ts in the volatiles
            }
            else {
                if (!ally.fainted)
                    return;
                pokemon.removeVolatile('commanding');
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1 },
        name: "Commander",
        rating: 0,
        num: 279,
    },
    competitive: {
        onAfterEachBoost: function (boost, target, source, effect) {
            if (!source || target.isAlly(source)) {
                return;
            }
            var statsLowered = false;
            var i;
            for (i in boost) {
                if (boost[i] < 0) {
                    statsLowered = true;
                }
            }
            if (statsLowered) {
                this.boost({ spa: 2 }, target, target, null, false, true);
            }
        },
        flags: {},
        name: "Competitive",
        rating: 2.5,
        num: 172,
    },
    compoundeyes: {
        onSourceModifyAccuracyPriority: -1,
        onSourceModifyAccuracy: function (accuracy) {
            if (typeof accuracy !== 'number')
                return;
            this.debug('compoundeyes - enhancing accuracy');
            return this.chainModify([5325, 4096]);
        },
        flags: {},
        name: "Compound Eyes",
        rating: 3,
        num: 14,
    },
    contrary: {
        onChangeBoost: function (boost, target, source, effect) {
            if (effect && effect.id === 'zpower')
                return;
            var i;
            for (i in boost) {
                boost[i] *= -1;
            }
        },
        flags: { breakable: 1 },
        name: "Contrary",
        rating: 4.5,
        num: 126,
    },
    corrosion: {
        // Implemented in sim/pokemon.js:Pokemon#setStatus
        flags: {},
        name: "Corrosion",
        rating: 2.5,
        num: 212,
    },
    costar: {
        onSwitchInPriority: -2,
        onStart: function (pokemon) {
            var ally = pokemon.allies()[0];
            if (!ally)
                return;
            var i;
            for (i in ally.boosts) {
                pokemon.boosts[i] = ally.boosts[i];
            }
            var volatilesToCopy = ['dragoncheer', 'focusenergy', 'gmaxchistrike', 'laserfocus'];
            // we need to be sure to remove all the overlapping crit volatiles before trying to add any
            for (var _i = 0, volatilesToCopy_1 = volatilesToCopy; _i < volatilesToCopy_1.length; _i++) {
                var volatile = volatilesToCopy_1[_i];
                pokemon.removeVolatile(volatile);
            }
            for (var _a = 0, volatilesToCopy_2 = volatilesToCopy; _a < volatilesToCopy_2.length; _a++) {
                var volatile = volatilesToCopy_2[_a];
                if (ally.volatiles[volatile]) {
                    pokemon.addVolatile(volatile);
                    if (volatile === 'gmaxchistrike')
                        pokemon.volatiles[volatile].layers = ally.volatiles[volatile].layers;
                    if (volatile === 'dragoncheer')
                        pokemon.volatiles[volatile].hasDragonType = ally.volatiles[volatile].hasDragonType;
                }
            }
            this.add('-copyboost', pokemon, ally, '[from] ability: Costar');
        },
        flags: {},
        name: "Costar",
        rating: 0,
        num: 294,
    },
    cottondown: {
        onDamagingHit: function (damage, target, source, move) {
            var activated = false;
            for (var _i = 0, _a = this.getAllActive(); _i < _a.length; _i++) {
                var pokemon = _a[_i];
                if (pokemon === target || pokemon.fainted)
                    continue;
                if (!activated) {
                    this.add('-ability', target, 'Cotton Down');
                    activated = true;
                }
                this.boost({ spe: -1 }, pokemon, target, null, true);
            }
        },
        flags: {},
        name: "Cotton Down",
        rating: 2,
        num: 238,
    },
    cudchew: {
        onEatItem: function (item, pokemon) {
            if (item.isBerry && pokemon.addVolatile('cudchew')) {
                pokemon.volatiles['cudchew'].berry = item;
            }
        },
        onEnd: function (pokemon) {
            delete pokemon.volatiles['cudchew'];
        },
        condition: {
            noCopy: true,
            duration: 2,
            onRestart: function () {
                this.effectState.duration = 2;
            },
            onResidualOrder: 28,
            onResidualSubOrder: 2,
            onEnd: function (pokemon) {
                if (pokemon.hp) {
                    var item = this.effectState.berry;
                    this.add('-activate', pokemon, 'ability: Cud Chew');
                    this.add('-enditem', pokemon, item.name, '[eat]');
                    if (this.singleEvent('Eat', item, null, pokemon, null, null)) {
                        this.runEvent('EatItem', pokemon, null, null, item);
                    }
                    if (item.onEat)
                        pokemon.ateBerry = true;
                }
            },
        },
        flags: {},
        name: "Cud Chew",
        rating: 2,
        num: 291,
    },
    curiousmedicine: {
        onStart: function (pokemon) {
            for (var _i = 0, _a = pokemon.adjacentAllies(); _i < _a.length; _i++) {
                var ally = _a[_i];
                ally.clearBoosts();
                this.add('-clearboost', ally, '[from] ability: Curious Medicine', "[of] ".concat(pokemon));
            }
        },
        flags: {},
        name: "Curious Medicine",
        rating: 0,
        num: 261,
    },
    cursedbody: {
        onDamagingHit: function (damage, target, source, move) {
            if (source.volatiles['disable'])
                return;
            if (!move.isMax && !move.flags['futuremove'] && move.id !== 'struggle') {
                if (this.randomChance(3, 10)) {
                    source.addVolatile('disable', this.effectState.target);
                }
            }
        },
        flags: {},
        name: "Cursed Body",
        rating: 2,
        num: 130,
    },
    cutecharm: {
        onDamagingHit: function (damage, target, source, move) {
            if (this.checkMoveMakesContact(move, source, target)) {
                if (this.randomChance(3, 10)) {
                    source.addVolatile('attract', this.effectState.target);
                }
            }
        },
        flags: {},
        name: "Cute Charm",
        rating: 0.5,
        num: 56,
    },
    damp: {
        onAnyTryMove: function (target, source, effect) {
            if (['explosion', 'mindblown', 'mistyexplosion', 'selfdestruct'].includes(effect.id)) {
                this.attrLastMove('[still]');
                this.add('cant', this.effectState.target, 'ability: Damp', effect, "[of] ".concat(target));
                return false;
            }
        },
        onAnyDamage: function (damage, target, source, effect) {
            if (effect && effect.name === 'Aftermath') {
                return false;
            }
        },
        flags: { breakable: 1 },
        name: "Damp",
        rating: 0.5,
        num: 6,
    },
    dancer: {
        flags: {},
        name: "Dancer",
        // implemented in runMove in scripts.js
        rating: 1.5,
        num: 216,
    },
    darkaura: {
        onStart: function (pokemon) {
            if (this.suppressingAbility(pokemon))
                return;
            this.add('-ability', pokemon, 'Dark Aura');
        },
        onAnyBasePowerPriority: 20,
        onAnyBasePower: function (basePower, source, target, move) {
            var _a;
            if (target === source || move.category === 'Status' || move.type !== 'Dark')
                return;
            if (!((_a = move.auraBooster) === null || _a === void 0 ? void 0 : _a.hasAbility('Dark Aura')))
                move.auraBooster = this.effectState.target;
            if (move.auraBooster !== this.effectState.target)
                return;
            return this.chainModify([move.hasAuraBreak ? 3072 : 5448, 4096]);
        },
        flags: {},
        name: "Dark Aura",
        rating: 3,
        num: 186,
    },
    dauntlessshield: {
        onStart: function (pokemon) {
            if (pokemon.shieldBoost)
                return;
            pokemon.shieldBoost = true;
            this.boost({ def: 1 }, pokemon);
        },
        flags: {},
        name: "Dauntless Shield",
        rating: 3.5,
        num: 235,
    },
    dazzling: {
        onFoeTryMove: function (target, source, move) {
            var targetAllExceptions = ['perishsong', 'flowershield', 'rototiller'];
            if (move.target === 'foeSide' || (move.target === 'all' && !targetAllExceptions.includes(move.id))) {
                return;
            }
            var dazzlingHolder = this.effectState.target;
            if ((source.isAlly(dazzlingHolder) || move.target === 'all') && move.priority > 0.1) {
                this.attrLastMove('[still]');
                this.add('cant', dazzlingHolder, 'ability: Dazzling', move, "[of] ".concat(target));
                return false;
            }
        },
        flags: { breakable: 1 },
        name: "Dazzling",
        rating: 2.5,
        num: 219,
    },
    defeatist: {
        onModifyAtkPriority: 5,
        onModifyAtk: function (atk, pokemon) {
            if (pokemon.hp <= pokemon.maxhp / 2) {
                return this.chainModify(0.5);
            }
        },
        onModifySpAPriority: 5,
        onModifySpA: function (atk, pokemon) {
            if (pokemon.hp <= pokemon.maxhp / 2) {
                return this.chainModify(0.5);
            }
        },
        flags: {},
        name: "Defeatist",
        rating: -1,
        num: 129,
    },
    defiant: {
        onAfterEachBoost: function (boost, target, source, effect) {
            if (!source || target.isAlly(source)) {
                return;
            }
            var statsLowered = false;
            var i;
            for (i in boost) {
                if (boost[i] < 0) {
                    statsLowered = true;
                }
            }
            if (statsLowered) {
                this.boost({ atk: 2 }, target, target, null, false, true);
            }
        },
        flags: {},
        name: "Defiant",
        rating: 3,
        num: 128,
    },
    deltastream: {
        onStart: function (source) {
            this.field.setWeather('deltastream');
        },
        onAnySetWeather: function (target, source, weather) {
            var strongWeathers = ['desolateland', 'primordialsea', 'deltastream'];
            if (this.field.getWeather().id === 'deltastream' && !strongWeathers.includes(weather.id))
                return false;
        },
        onEnd: function (pokemon) {
            if (this.field.weatherState.source !== pokemon)
                return;
            for (var _i = 0, _a = this.getAllActive(); _i < _a.length; _i++) {
                var target = _a[_i];
                if (target === pokemon)
                    continue;
                if (target.hasAbility('deltastream')) {
                    this.field.weatherState.source = target;
                    return;
                }
            }
            this.field.clearWeather();
        },
        flags: {},
        name: "Delta Stream",
        rating: 4,
        num: 191,
    },
    desolateland: {
        onStart: function (source) {
            this.field.setWeather('desolateland');
        },
        onAnySetWeather: function (target, source, weather) {
            var strongWeathers = ['desolateland', 'primordialsea', 'deltastream'];
            if (this.field.getWeather().id === 'desolateland' && !strongWeathers.includes(weather.id))
                return false;
        },
        onEnd: function (pokemon) {
            if (this.field.weatherState.source !== pokemon)
                return;
            for (var _i = 0, _a = this.getAllActive(); _i < _a.length; _i++) {
                var target = _a[_i];
                if (target === pokemon)
                    continue;
                if (target.hasAbility('desolateland')) {
                    this.field.weatherState.source = target;
                    return;
                }
            }
            this.field.clearWeather();
        },
        flags: {},
        name: "Desolate Land",
        rating: 4.5,
        num: 190,
    },
    disguise: {
        onDamagePriority: 1,
        onDamage: function (damage, target, source, effect) {
            if ((effect === null || effect === void 0 ? void 0 : effect.effectType) === 'Move' && ['mimikyu', 'mimikyutotem'].includes(target.species.id)) {
                this.add('-activate', target, 'ability: Disguise');
                this.effectState.busted = true;
                return 0;
            }
        },
        onCriticalHit: function (target, source, move) {
            if (!target)
                return;
            if (!['mimikyu', 'mimikyutotem'].includes(target.species.id)) {
                return;
            }
            var hitSub = target.volatiles['substitute'] && !move.flags['bypasssub'] && !(move.infiltrates && this.gen >= 6);
            if (hitSub)
                return;
            if (!target.runImmunity(move.type))
                return;
            return false;
        },
        onEffectiveness: function (typeMod, target, type, move) {
            if (!target || move.category === 'Status')
                return;
            if (!['mimikyu', 'mimikyutotem'].includes(target.species.id)) {
                return;
            }
            var hitSub = target.volatiles['substitute'] && !move.flags['bypasssub'] && !(move.infiltrates && this.gen >= 6);
            if (hitSub)
                return;
            if (!target.runImmunity(move.type))
                return;
            return 0;
        },
        onUpdate: function (pokemon) {
            if (['mimikyu', 'mimikyutotem'].includes(pokemon.species.id) && this.effectState.busted) {
                var speciesid = pokemon.species.id === 'mimikyutotem' ? 'Mimikyu-Busted-Totem' : 'Mimikyu-Busted';
                pokemon.formeChange(speciesid, this.effect, true);
                this.damage(pokemon.baseMaxhp / 8, pokemon, pokemon, this.dex.species.get(speciesid));
            }
        },
        flags: {
            failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, cantsuppress: 1,
            breakable: 1, notransform: 1,
        },
        name: "Disguise",
        rating: 3.5,
        num: 209,
    },
    download: {
        onStart: function (pokemon) {
            var totaldef = 0;
            var totalspd = 0;
            for (var _i = 0, _a = pokemon.foes(); _i < _a.length; _i++) {
                var target = _a[_i];
                totaldef += target.getStat('def', false, true);
                totalspd += target.getStat('spd', false, true);
            }
            if (totaldef && totaldef >= totalspd) {
                this.boost({ spa: 1 });
            }
            else if (totalspd) {
                this.boost({ atk: 1 });
            }
        },
        flags: {},
        name: "Download",
        rating: 3.5,
        num: 88,
    },
    dragonsmaw: {
        onModifyAtkPriority: 5,
        onModifyAtk: function (atk, attacker, defender, move) {
            if (move.type === 'Dragon') {
                this.debug('Dragon\'s Maw boost');
                return this.chainModify(1.5);
            }
        },
        onModifySpAPriority: 5,
        onModifySpA: function (atk, attacker, defender, move) {
            if (move.type === 'Dragon') {
                this.debug('Dragon\'s Maw boost');
                return this.chainModify(1.5);
            }
        },
        flags: {},
        name: "Dragon's Maw",
        rating: 3.5,
        num: 263,
    },
    drizzle: {
        onStart: function (source) {
            if (source.species.id === 'kyogre' && source.item === 'blueorb')
                return;
            this.field.setWeather('raindance');
        },
        flags: {},
        name: "Drizzle",
        rating: 4,
        num: 2,
    },
    drought: {
        onStart: function (source) {
            if (source.species.id === 'groudon' && source.item === 'redorb')
                return;
            this.field.setWeather('sunnyday');
        },
        flags: {},
        name: "Drought",
        rating: 4,
        num: 70,
    },
    dryskin: {
        onTryHit: function (target, source, move) {
            if (target !== source && move.type === 'Water') {
                if (!this.heal(target.baseMaxhp / 4)) {
                    this.add('-immune', target, '[from] ability: Dry Skin');
                }
                return null;
            }
        },
        onSourceBasePowerPriority: 17,
        onSourceBasePower: function (basePower, attacker, defender, move) {
            if (move.type === 'Fire') {
                return this.chainModify(1.25);
            }
        },
        onWeather: function (target, source, effect) {
            if (target.hasItem('utilityumbrella'))
                return;
            if (effect.id === 'raindance' || effect.id === 'primordialsea') {
                this.heal(target.baseMaxhp / 8);
            }
            else if (effect.id === 'sunnyday' || effect.id === 'desolateland') {
                this.damage(target.baseMaxhp / 8, target, target);
            }
        },
        flags: { breakable: 1 },
        name: "Dry Skin",
        rating: 3,
        num: 87,
    },
    earlybird: {
        flags: {},
        name: "Early Bird",
        // Implemented in statuses.js
        rating: 1.5,
        num: 48,
    },
    eartheater: {
        onTryHit: function (target, source, move) {
            if (target !== source && move.type === 'Ground') {
                if (!this.heal(target.baseMaxhp / 4)) {
                    this.add('-immune', target, '[from] ability: Earth Eater');
                }
                return null;
            }
        },
        flags: { breakable: 1 },
        name: "Earth Eater",
        rating: 3.5,
        num: 297,
    },
    effectspore: {
        onDamagingHit: function (damage, target, source, move) {
            if (this.checkMoveMakesContact(move, source, target) && !source.status && source.runStatusImmunity('powder')) {
                var r = this.random(100);
                if (r < 11) {
                    source.setStatus('slp', target);
                }
                else if (r < 21) {
                    source.setStatus('par', target);
                }
                else if (r < 30) {
                    source.setStatus('psn', target);
                }
            }
        },
        flags: {},
        name: "Effect Spore",
        rating: 2,
        num: 27,
    },
    electricsurge: {
        onStart: function (source) {
            this.field.setTerrain('electricterrain');
        },
        flags: {},
        name: "Electric Surge",
        rating: 4,
        num: 226,
    },
    electromorphosis: {
        onDamagingHitOrder: 1,
        onDamagingHit: function (damage, target, source, move) {
            target.addVolatile('charge');
        },
        flags: {},
        name: "Electromorphosis",
        rating: 3,
        num: 280,
    },
    embodyaspectcornerstone: {
        onStart: function (pokemon) {
            if (pokemon.baseSpecies.name === 'Ogerpon-Cornerstone-Tera' &&
                this.effectState.embodied !== pokemon.previouslySwitchedIn) {
                this.effectState.embodied = pokemon.previouslySwitchedIn;
                this.boost({ def: 1 }, pokemon);
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, notransform: 1 },
        name: "Embody Aspect (Cornerstone)",
        rating: 3.5,
        num: 304,
    },
    embodyaspecthearthflame: {
        onStart: function (pokemon) {
            if (pokemon.baseSpecies.name === 'Ogerpon-Hearthflame-Tera' &&
                this.effectState.embodied !== pokemon.previouslySwitchedIn) {
                this.effectState.embodied = pokemon.previouslySwitchedIn;
                this.boost({ atk: 1 }, pokemon);
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, notransform: 1 },
        name: "Embody Aspect (Hearthflame)",
        rating: 3.5,
        num: 303,
    },
    embodyaspectteal: {
        onStart: function (pokemon) {
            if (pokemon.baseSpecies.name === 'Ogerpon-Teal-Tera' &&
                this.effectState.embodied !== pokemon.previouslySwitchedIn) {
                this.effectState.embodied = pokemon.previouslySwitchedIn;
                this.boost({ spe: 1 }, pokemon);
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, notransform: 1 },
        name: "Embody Aspect (Teal)",
        rating: 3.5,
        num: 301,
    },
    embodyaspectwellspring: {
        onStart: function (pokemon) {
            if (pokemon.baseSpecies.name === 'Ogerpon-Wellspring-Tera' &&
                this.effectState.embodied !== pokemon.previouslySwitchedIn) {
                this.effectState.embodied = pokemon.previouslySwitchedIn;
                this.boost({ spd: 1 }, pokemon);
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, notransform: 1 },
        name: "Embody Aspect (Wellspring)",
        rating: 3.5,
        num: 302,
    },
    emergencyexit: {
        onEmergencyExit: function (target) {
            if (!this.canSwitch(target.side) || target.forceSwitchFlag || target.switchFlag)
                return;
            for (var _i = 0, _a = this.sides; _i < _a.length; _i++) {
                var side = _a[_i];
                for (var _b = 0, _c = side.active; _b < _c.length; _b++) {
                    var active = _c[_b];
                    active.switchFlag = false;
                }
            }
            target.switchFlag = true;
            this.add('-activate', target, 'ability: Emergency Exit');
        },
        flags: {},
        name: "Emergency Exit",
        rating: 1,
        num: 194,
    },
    fairyaura: {
        onStart: function (pokemon) {
            if (this.suppressingAbility(pokemon))
                return;
            this.add('-ability', pokemon, 'Fairy Aura');
        },
        onAnyBasePowerPriority: 20,
        onAnyBasePower: function (basePower, source, target, move) {
            var _a;
            if (target === source || move.category === 'Status' || move.type !== 'Fairy')
                return;
            if (!((_a = move.auraBooster) === null || _a === void 0 ? void 0 : _a.hasAbility('Fairy Aura')))
                move.auraBooster = this.effectState.target;
            if (move.auraBooster !== this.effectState.target)
                return;
            return this.chainModify([move.hasAuraBreak ? 3072 : 5448, 4096]);
        },
        flags: {},
        name: "Fairy Aura",
        rating: 3,
        num: 187,
    },
    filter: {
        onSourceModifyDamage: function (damage, source, target, move) {
            if (target.getMoveHitData(move).typeMod > 0) {
                this.debug('Filter neutralize');
                return this.chainModify(0.75);
            }
        },
        flags: { breakable: 1 },
        name: "Filter",
        rating: 3,
        num: 111,
    },
    flamebody: {
        onDamagingHit: function (damage, target, source, move) {
            if (this.checkMoveMakesContact(move, source, target)) {
                if (this.randomChance(3, 10)) {
                    source.trySetStatus('brn', target);
                }
            }
        },
        flags: {},
        name: "Flame Body",
        rating: 2,
        num: 49,
    },
    flareboost: {
        onBasePowerPriority: 19,
        onBasePower: function (basePower, attacker, defender, move) {
            if (attacker.status === 'brn' && move.category === 'Special') {
                return this.chainModify(1.5);
            }
        },
        flags: {},
        name: "Flare Boost",
        rating: 2,
        num: 138,
    },
    flashfire: {
        onTryHit: function (target, source, move) {
            if (target !== source && move.type === 'Fire') {
                move.accuracy = true;
                if (!target.addVolatile('flashfire')) {
                    this.add('-immune', target, '[from] ability: Flash Fire');
                }
                return null;
            }
        },
        onEnd: function (pokemon) {
            pokemon.removeVolatile('flashfire');
        },
        condition: {
            noCopy: true, // doesn't get copied by Baton Pass
            onStart: function (target) {
                this.add('-start', target, 'ability: Flash Fire');
            },
            onModifyAtkPriority: 5,
            onModifyAtk: function (atk, attacker, defender, move) {
                if (move.type === 'Fire' && attacker.hasAbility('flashfire')) {
                    this.debug('Flash Fire boost');
                    return this.chainModify(1.5);
                }
            },
            onModifySpAPriority: 5,
            onModifySpA: function (atk, attacker, defender, move) {
                if (move.type === 'Fire' && attacker.hasAbility('flashfire')) {
                    this.debug('Flash Fire boost');
                    return this.chainModify(1.5);
                }
            },
            onEnd: function (target) {
                this.add('-end', target, 'ability: Flash Fire', '[silent]');
            },
        },
        flags: { breakable: 1 },
        name: "Flash Fire",
        rating: 3.5,
        num: 18,
    },
    flowergift: {
        onSwitchInPriority: -2,
        onStart: function (pokemon) {
            this.singleEvent('WeatherChange', this.effect, this.effectState, pokemon);
        },
        onWeatherChange: function (pokemon) {
            if (!pokemon.isActive || pokemon.baseSpecies.baseSpecies !== 'Cherrim' || pokemon.transformed)
                return;
            if (!pokemon.hp)
                return;
            if (['sunnyday', 'desolateland'].includes(pokemon.effectiveWeather())) {
                if (pokemon.species.id !== 'cherrimsunshine') {
                    pokemon.formeChange('Cherrim-Sunshine', this.effect, false, '0', '[msg]');
                }
            }
            else {
                if (pokemon.species.id === 'cherrimsunshine') {
                    pokemon.formeChange('Cherrim', this.effect, false, '0', '[msg]');
                }
            }
        },
        onAllyModifyAtkPriority: 3,
        onAllyModifyAtk: function (atk, pokemon) {
            if (this.effectState.target.baseSpecies.baseSpecies !== 'Cherrim')
                return;
            if (['sunnyday', 'desolateland'].includes(pokemon.effectiveWeather())) {
                return this.chainModify(1.5);
            }
        },
        onAllyModifySpDPriority: 4,
        onAllyModifySpD: function (spd, pokemon) {
            if (this.effectState.target.baseSpecies.baseSpecies !== 'Cherrim')
                return;
            if (['sunnyday', 'desolateland'].includes(pokemon.effectiveWeather())) {
                return this.chainModify(1.5);
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, breakable: 1 },
        name: "Flower Gift",
        rating: 1,
        num: 122,
    },
    flowerveil: {
        onAllyTryBoost: function (boost, target, source, effect) {
            if ((source && target === source) || !target.hasType('Grass'))
                return;
            var showMsg = false;
            var i;
            for (i in boost) {
                if (boost[i] < 0) {
                    delete boost[i];
                    showMsg = true;
                }
            }
            if (showMsg && !effect.secondaries) {
                var effectHolder = this.effectState.target;
                this.add('-block', target, 'ability: Flower Veil', "[of] ".concat(effectHolder));
            }
        },
        onAllySetStatus: function (status, target, source, effect) {
            if (target.hasType('Grass') && source && target !== source && effect && effect.id !== 'yawn') {
                this.debug('interrupting setStatus with Flower Veil');
                if (effect.name === 'Synchronize' || (effect.effectType === 'Move' && !effect.secondaries)) {
                    var effectHolder = this.effectState.target;
                    this.add('-block', target, 'ability: Flower Veil', "[of] ".concat(effectHolder));
                }
                return null;
            }
        },
        onAllyTryAddVolatile: function (status, target) {
            if (target.hasType('Grass') && status.id === 'yawn') {
                this.debug('Flower Veil blocking yawn');
                var effectHolder = this.effectState.target;
                this.add('-block', target, 'ability: Flower Veil', "[of] ".concat(effectHolder));
                return null;
            }
        },
        flags: { breakable: 1 },
        name: "Flower Veil",
        rating: 0,
        num: 166,
    },
    fluffy: {
        onSourceModifyDamage: function (damage, source, target, move) {
            var mod = 1;
            if (move.type === 'Fire')
                mod *= 2;
            if (move.flags['contact'])
                mod /= 2;
            return this.chainModify(mod);
        },
        flags: { breakable: 1 },
        name: "Fluffy",
        rating: 3.5,
        num: 218,
    },
    forecast: {
        onSwitchInPriority: -2,
        onStart: function (pokemon) {
            this.singleEvent('WeatherChange', this.effect, this.effectState, pokemon);
        },
        onWeatherChange: function (pokemon) {
            if (pokemon.baseSpecies.baseSpecies !== 'Castform' || pokemon.transformed)
                return;
            var forme = null;
            switch (pokemon.effectiveWeather()) {
                case 'sunnyday':
                case 'desolateland':
                    if (pokemon.species.id !== 'castformsunny')
                        forme = 'Castform-Sunny';
                    break;
                case 'raindance':
                case 'primordialsea':
                    if (pokemon.species.id !== 'castformrainy')
                        forme = 'Castform-Rainy';
                    break;
                case 'hail':
                case 'snowscape':
                    if (pokemon.species.id !== 'castformsnowy')
                        forme = 'Castform-Snowy';
                    break;
                default:
                    if (pokemon.species.id !== 'castform')
                        forme = 'Castform';
                    break;
            }
            if (pokemon.isActive && forme) {
                pokemon.formeChange(forme, this.effect, false, '0', '[msg]');
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1 },
        name: "Forecast",
        rating: 2,
        num: 59,
    },
    forewarn: {
        onStart: function (pokemon) {
            var warnMoves = [];
            var warnBp = 1;
            for (var _i = 0, _a = pokemon.foes(); _i < _a.length; _i++) {
                var target = _a[_i];
                for (var _b = 0, _c = target.moveSlots; _b < _c.length; _b++) {
                    var moveSlot = _c[_b];
                    var move = this.dex.moves.get(moveSlot.move);
                    var bp = move.basePower;
                    if (move.ohko)
                        bp = 150;
                    if (move.id === 'counter' || move.id === 'metalburst' || move.id === 'mirrorcoat')
                        bp = 120;
                    if (bp === 1)
                        bp = 80;
                    if (!bp && move.category !== 'Status')
                        bp = 80;
                    if (bp > warnBp) {
                        warnMoves = [[move, target]];
                        warnBp = bp;
                    }
                    else if (bp === warnBp) {
                        warnMoves.push([move, target]);
                    }
                }
            }
            if (!warnMoves.length)
                return;
            var _d = this.sample(warnMoves), warnMoveName = _d[0], warnTarget = _d[1];
            this.add('-activate', pokemon, 'ability: Forewarn', warnMoveName, "[of] ".concat(warnTarget));
        },
        flags: {},
        name: "Forewarn",
        rating: 0.5,
        num: 108,
    },
    friendguard: {
        onAnyModifyDamage: function (damage, source, target, move) {
            if (target !== this.effectState.target && target.isAlly(this.effectState.target)) {
                this.debug('Friend Guard weaken');
                return this.chainModify(0.75);
            }
        },
        flags: { breakable: 1 },
        name: "Friend Guard",
        rating: 0,
        num: 132,
    },
    frisk: {
        onStart: function (pokemon) {
            for (var _i = 0, _a = pokemon.foes(); _i < _a.length; _i++) {
                var target = _a[_i];
                if (target.item) {
                    this.add('-item', target, target.getItem().name, '[from] ability: Frisk', "[of] ".concat(pokemon));
                }
            }
        },
        flags: {},
        name: "Frisk",
        rating: 1.5,
        num: 119,
    },
    fullmetalbody: {
        onTryBoost: function (boost, target, source, effect) {
            if (source && target === source)
                return;
            var showMsg = false;
            var i;
            for (i in boost) {
                if (boost[i] < 0) {
                    delete boost[i];
                    showMsg = true;
                }
            }
            if (showMsg && !effect.secondaries && effect.id !== 'octolock') {
                this.add("-fail", target, "unboost", "[from] ability: Full Metal Body", "[of] ".concat(target));
            }
        },
        flags: {},
        name: "Full Metal Body",
        rating: 2,
        num: 230,
    },
    furcoat: {
        onModifyDefPriority: 6,
        onModifyDef: function (def) {
            return this.chainModify(2);
        },
        flags: { breakable: 1 },
        name: "Fur Coat",
        rating: 4,
        num: 169,
    },
    galewings: {
        onModifyPriority: function (priority, pokemon, target, move) {
            if ((move === null || move === void 0 ? void 0 : move.type) === 'Flying' && pokemon.hp === pokemon.maxhp)
                return priority + 1;
        },
        flags: {},
        name: "Gale Wings",
        rating: 1.5,
        num: 177,
    },
    galvanize: {
        onModifyTypePriority: -1,
        onModifyType: function (move, pokemon) {
            var noModifyType = [
                'judgment', 'multiattack', 'naturalgift', 'revelationdance', 'technoblast', 'terrainpulse', 'weatherball',
            ];
            if (move.type === 'Normal' && !noModifyType.includes(move.id) &&
                !(move.isZ && move.category !== 'Status') && !(move.name === 'Tera Blast' && pokemon.terastallized)) {
                move.type = 'Electric';
                move.typeChangerBoosted = this.effect;
            }
        },
        onBasePowerPriority: 23,
        onBasePower: function (basePower, pokemon, target, move) {
            if (move.typeChangerBoosted === this.effect)
                return this.chainModify([4915, 4096]);
        },
        flags: {},
        name: "Galvanize",
        rating: 4,
        num: 206,
    },
    gluttony: {
        onStart: function (pokemon) {
            pokemon.abilityState.gluttony = true;
        },
        onDamage: function (item, pokemon) {
            pokemon.abilityState.gluttony = true;
        },
        flags: {},
        name: "Gluttony",
        rating: 1.5,
        num: 82,
    },
    goodasgold: {
        onTryHit: function (target, source, move) {
            if (move.category === 'Status' && target !== source) {
                this.add('-immune', target, '[from] ability: Good as Gold');
                return null;
            }
        },
        flags: { breakable: 1 },
        name: "Good as Gold",
        rating: 5,
        num: 283,
    },
    gooey: {
        onDamagingHit: function (damage, target, source, move) {
            if (this.checkMoveMakesContact(move, source, target, true)) {
                this.add('-ability', target, 'Gooey');
                this.boost({ spe: -1 }, source, target, null, true);
            }
        },
        flags: {},
        name: "Gooey",
        rating: 2,
        num: 183,
    },
    gorillatactics: {
        onStart: function (pokemon) {
            pokemon.abilityState.choiceLock = "";
        },
        onBeforeMove: function (pokemon, target, move) {
            if (move.isZOrMaxPowered || move.id === 'struggle')
                return;
            if (pokemon.abilityState.choiceLock && pokemon.abilityState.choiceLock !== move.id) {
                // Fails unless ability is being ignored (these events will not run), no PP lost.
                this.addMove('move', pokemon, move.name);
                this.attrLastMove('[still]');
                this.debug("Disabled by Gorilla Tactics");
                this.add('-fail', pokemon);
                return false;
            }
        },
        onModifyMove: function (move, pokemon) {
            if (pokemon.abilityState.choiceLock || move.isZOrMaxPowered || move.id === 'struggle')
                return;
            pokemon.abilityState.choiceLock = move.id;
        },
        onModifyAtkPriority: 1,
        onModifyAtk: function (atk, pokemon) {
            if (pokemon.volatiles['dynamax'])
                return;
            // PLACEHOLDER
            this.debug('Gorilla Tactics Atk Boost');
            return this.chainModify(1.5);
        },
        onDisableMove: function (pokemon) {
            if (!pokemon.abilityState.choiceLock)
                return;
            if (pokemon.volatiles['dynamax'])
                return;
            for (var _i = 0, _a = pokemon.moveSlots; _i < _a.length; _i++) {
                var moveSlot = _a[_i];
                if (moveSlot.id !== pokemon.abilityState.choiceLock) {
                    pokemon.disableMove(moveSlot.id, false, this.effectState.sourceEffect);
                }
            }
        },
        onEnd: function (pokemon) {
            pokemon.abilityState.choiceLock = "";
        },
        flags: {},
        name: "Gorilla Tactics",
        rating: 4.5,
        num: 255,
    },
    grasspelt: {
        onModifyDefPriority: 6,
        onModifyDef: function (pokemon) {
            if (this.field.isTerrain('grassyterrain'))
                return this.chainModify(1.5);
        },
        flags: { breakable: 1 },
        name: "Grass Pelt",
        rating: 0.5,
        num: 179,
    },
    grassysurge: {
        onStart: function (source) {
            this.field.setTerrain('grassyterrain');
        },
        flags: {},
        name: "Grassy Surge",
        rating: 4,
        num: 229,
    },
    grimneigh: {
        onSourceAfterFaint: function (length, target, source, effect) {
            if (effect && effect.effectType === 'Move') {
                this.boost({ spa: length }, source);
            }
        },
        flags: {},
        name: "Grim Neigh",
        rating: 3,
        num: 265,
    },
    guarddog: {
        onDragOutPriority: 1,
        onDragOut: function (pokemon) {
            this.add('-activate', pokemon, 'ability: Guard Dog');
            return null;
        },
        onTryBoostPriority: 2,
        onTryBoost: function (boost, target, source, effect) {
            if (effect.name === 'Intimidate' && boost.atk) {
                delete boost.atk;
                this.boost({ atk: 1 }, target, target, null, false, true);
            }
        },
        flags: { breakable: 1 },
        name: "Guard Dog",
        rating: 2,
        num: 275,
    },
    gulpmissile: {
        onDamagingHit: function (damage, target, source, move) {
            if (!source.hp || !source.isActive || target.isSemiInvulnerable())
                return;
            if (['cramorantgulping', 'cramorantgorging'].includes(target.species.id)) {
                this.damage(source.baseMaxhp / 4, source, target);
                if (target.species.id === 'cramorantgulping') {
                    this.boost({ def: -1 }, source, target, null, true);
                }
                else {
                    source.trySetStatus('par', target, move);
                }
                target.formeChange('cramorant', move);
            }
        },
        // The Dive part of this mechanic is implemented in Dive's `onTryMove` in moves.ts
        onSourceTryPrimaryHit: function (target, source, effect) {
            if ((effect === null || effect === void 0 ? void 0 : effect.id) === 'surf' && source.hasAbility('gulpmissile') && source.species.name === 'Cramorant') {
                var forme = source.hp <= source.maxhp / 2 ? 'cramorantgorging' : 'cramorantgulping';
                source.formeChange(forme, effect);
            }
        },
        flags: { cantsuppress: 1, notransform: 1 },
        name: "Gulp Missile",
        rating: 2.5,
        num: 241,
    },
    guts: {
        onModifyAtkPriority: 5,
        onModifyAtk: function (atk, pokemon) {
            if (pokemon.status) {
                return this.chainModify(1.5);
            }
        },
        flags: {},
        name: "Guts",
        rating: 3.5,
        num: 62,
    },
    hadronengine: {
        onStart: function (pokemon) {
            if (!this.field.setTerrain('electricterrain') && this.field.isTerrain('electricterrain')) {
                this.add('-activate', pokemon, 'ability: Hadron Engine');
            }
        },
        onModifySpAPriority: 5,
        onModifySpA: function (atk, attacker, defender, move) {
            if (this.field.isTerrain('electricterrain')) {
                this.debug('Hadron Engine boost');
                return this.chainModify([5461, 4096]);
            }
        },
        flags: {},
        name: "Hadron Engine",
        rating: 4.5,
        num: 289,
    },
    harvest: {
        onResidualOrder: 28,
        onResidualSubOrder: 2,
        onResidual: function (pokemon) {
            if (this.field.isWeather(['sunnyday', 'desolateland']) || this.randomChance(1, 2)) {
                if (pokemon.hp && !pokemon.item && this.dex.items.get(pokemon.lastItem).isBerry) {
                    pokemon.setItem(pokemon.lastItem);
                    pokemon.lastItem = '';
                    this.add('-item', pokemon, pokemon.getItem(), '[from] ability: Harvest');
                }
            }
        },
        flags: {},
        name: "Harvest",
        rating: 2.5,
        num: 139,
    },
    healer: {
        onResidualOrder: 5,
        onResidualSubOrder: 3,
        onResidual: function (pokemon) {
            for (var _i = 0, _a = pokemon.adjacentAllies(); _i < _a.length; _i++) {
                var allyActive = _a[_i];
                if (allyActive.status && this.randomChance(3, 10)) {
                    this.add('-activate', pokemon, 'ability: Healer');
                    allyActive.cureStatus();
                }
            }
        },
        flags: {},
        name: "Healer",
        rating: 0,
        num: 131,
    },
    heatproof: {
        onSourceModifyAtkPriority: 6,
        onSourceModifyAtk: function (atk, attacker, defender, move) {
            if (move.type === 'Fire') {
                this.debug('Heatproof Atk weaken');
                return this.chainModify(0.5);
            }
        },
        onSourceModifySpAPriority: 5,
        onSourceModifySpA: function (atk, attacker, defender, move) {
            if (move.type === 'Fire') {
                this.debug('Heatproof SpA weaken');
                return this.chainModify(0.5);
            }
        },
        onDamage: function (damage, target, source, effect) {
            if (effect && effect.id === 'brn') {
                return damage / 2;
            }
        },
        flags: { breakable: 1 },
        name: "Heatproof",
        rating: 2,
        num: 85,
    },
    heavymetal: {
        onModifyWeightPriority: 1,
        onModifyWeight: function (weighthg) {
            return weighthg * 2;
        },
        flags: { breakable: 1 },
        name: "Heavy Metal",
        rating: 0,
        num: 134,
    },
    honeygather: {
        flags: {},
        name: "Honey Gather",
        rating: 0,
        num: 118,
    },
    hospitality: {
        onSwitchInPriority: -2,
        onStart: function (pokemon) {
            for (var _i = 0, _a = pokemon.adjacentAllies(); _i < _a.length; _i++) {
                var ally = _a[_i];
                this.heal(ally.baseMaxhp / 4, ally, pokemon);
            }
        },
        flags: {},
        name: "Hospitality",
        rating: 0,
        num: 299,
    },
    hugepower: {
        onModifyAtkPriority: 5,
        onModifyAtk: function (atk) {
            return this.chainModify(2);
        },
        flags: {},
        name: "Huge Power",
        rating: 5,
        num: 37,
    },
    hungerswitch: {
        onResidualOrder: 29,
        onResidual: function (pokemon) {
            if (pokemon.species.baseSpecies !== 'Morpeko' || pokemon.terastallized)
                return;
            var targetForme = pokemon.species.name === 'Morpeko' ? 'Morpeko-Hangry' : 'Morpeko';
            pokemon.formeChange(targetForme);
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, notransform: 1 },
        name: "Hunger Switch",
        rating: 1,
        num: 258,
    },
    hustle: {
        // This should be applied directly to the stat as opposed to chaining with the others
        onModifyAtkPriority: 5,
        onModifyAtk: function (atk) {
            return this.modify(atk, 1.5);
        },
        onSourceModifyAccuracyPriority: -1,
        onSourceModifyAccuracy: function (accuracy, target, source, move) {
            if (move.category === 'Physical' && typeof accuracy === 'number') {
                return this.chainModify([3277, 4096]);
            }
        },
        flags: {},
        name: "Hustle",
        rating: 3.5,
        num: 55,
    },
    hydration: {
        onResidualOrder: 5,
        onResidualSubOrder: 3,
        onResidual: function (pokemon) {
            if (pokemon.status && ['raindance', 'primordialsea'].includes(pokemon.effectiveWeather())) {
                this.debug('hydration');
                this.add('-activate', pokemon, 'ability: Hydration');
                pokemon.cureStatus();
            }
        },
        flags: {},
        name: "Hydration",
        rating: 1.5,
        num: 93,
    },
    hypercutter: {
        onTryBoost: function (boost, target, source, effect) {
            if (source && target === source)
                return;
            if (boost.atk && boost.atk < 0) {
                delete boost.atk;
                if (!effect.secondaries) {
                    this.add("-fail", target, "unboost", "Attack", "[from] ability: Hyper Cutter", "[of] ".concat(target));
                }
            }
        },
        flags: { breakable: 1 },
        name: "Hyper Cutter",
        rating: 1.5,
        num: 52,
    },
    icebody: {
        onWeather: function (target, source, effect) {
            if (effect.id === 'hail' || effect.id === 'snowscape') {
                this.heal(target.baseMaxhp / 16);
            }
        },
        onImmunity: function (type, pokemon) {
            if (type === 'hail')
                return false;
        },
        flags: {},
        name: "Ice Body",
        rating: 1,
        num: 115,
    },
    iceface: {
        onSwitchInPriority: -2,
        onStart: function (pokemon) {
            if (this.field.isWeather(['hail', 'snowscape']) && pokemon.species.id === 'eiscuenoice') {
                this.add('-activate', pokemon, 'ability: Ice Face');
                this.effectState.busted = false;
                pokemon.formeChange('Eiscue', this.effect, true);
            }
        },
        onDamagePriority: 1,
        onDamage: function (damage, target, source, effect) {
            if ((effect === null || effect === void 0 ? void 0 : effect.effectType) === 'Move' && effect.category === 'Physical' && target.species.id === 'eiscue') {
                this.add('-activate', target, 'ability: Ice Face');
                this.effectState.busted = true;
                return 0;
            }
        },
        onCriticalHit: function (target, type, move) {
            if (!target)
                return;
            if (move.category !== 'Physical' || target.species.id !== 'eiscue')
                return;
            if (target.volatiles['substitute'] && !(move.flags['bypasssub'] || move.infiltrates))
                return;
            if (!target.runImmunity(move.type))
                return;
            return false;
        },
        onEffectiveness: function (typeMod, target, type, move) {
            if (!target)
                return;
            if (move.category !== 'Physical' || target.species.id !== 'eiscue')
                return;
            var hitSub = target.volatiles['substitute'] && !move.flags['bypasssub'] && !(move.infiltrates && this.gen >= 6);
            if (hitSub)
                return;
            if (!target.runImmunity(move.type))
                return;
            return 0;
        },
        onUpdate: function (pokemon) {
            if (pokemon.species.id === 'eiscue' && this.effectState.busted) {
                pokemon.formeChange('Eiscue-Noice', this.effect, true);
            }
        },
        onWeatherChange: function (pokemon, source, sourceEffect) {
            // snow/hail resuming because Cloud Nine/Air Lock ended does not trigger Ice Face
            if (sourceEffect === null || sourceEffect === void 0 ? void 0 : sourceEffect.suppressWeather)
                return;
            if (!pokemon.hp)
                return;
            if (this.field.isWeather(['hail', 'snowscape']) && pokemon.species.id === 'eiscuenoice') {
                this.add('-activate', pokemon, 'ability: Ice Face');
                this.effectState.busted = false;
                pokemon.formeChange('Eiscue', this.effect, true);
            }
        },
        flags: {
            failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, cantsuppress: 1,
            breakable: 1, notransform: 1,
        },
        name: "Ice Face",
        rating: 3,
        num: 248,
    },
    icescales: {
        onSourceModifyDamage: function (damage, source, target, move) {
            if (move.category === 'Special') {
                return this.chainModify(0.5);
            }
        },
        flags: { breakable: 1 },
        name: "Ice Scales",
        rating: 4,
        num: 246,
    },
    illuminate: {
        onTryBoost: function (boost, target, source, effect) {
            if (source && target === source)
                return;
            if (boost.accuracy && boost.accuracy < 0) {
                delete boost.accuracy;
                if (!effect.secondaries) {
                    this.add("-fail", target, "unboost", "accuracy", "[from] ability: Illuminate", "[of] ".concat(target));
                }
            }
        },
        onModifyMove: function (move) {
            move.ignoreEvasion = true;
        },
        flags: { breakable: 1 },
        name: "Illuminate",
        rating: 0.5,
        num: 35,
    },
    illusion: {
        onBeforeSwitchIn: function (pokemon) {
            pokemon.illusion = null;
            // yes, you can Illusion an active pokemon but only if it's to your right
            for (var i = pokemon.side.pokemon.length - 1; i > pokemon.position; i--) {
                var possibleTarget = pokemon.side.pokemon[i];
                if (!possibleTarget.fainted) {
                    // If Ogerpon is in the last slot while the Illusion Pokemon is Terastallized
                    // Illusion will not disguise as anything
                    if (!pokemon.terastallized || possibleTarget.species.baseSpecies !== 'Ogerpon') {
                        pokemon.illusion = possibleTarget;
                    }
                    break;
                }
            }
        },
        onDamagingHit: function (damage, target, source, move) {
            if (target.illusion) {
                this.singleEvent('End', this.dex.abilities.get('Illusion'), target.abilityState, target, source, move);
            }
        },
        onEnd: function (pokemon) {
            if (pokemon.illusion) {
                this.debug('illusion cleared');
                pokemon.illusion = null;
                var details = pokemon.getUpdatedDetails();
                this.add('replace', pokemon, details);
                this.add('-end', pokemon, 'Illusion');
                if (this.ruleTable.has('illusionlevelmod')) {
                    this.hint("Illusion Level Mod is active, so this Pok\u00e9mon's true level was hidden.", true);
                }
            }
        },
        onFaint: function (pokemon) {
            pokemon.illusion = null;
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1 },
        name: "Illusion",
        rating: 4.5,
        num: 149,
    },
    immunity: {
        onUpdate: function (pokemon) {
            if (pokemon.status === 'psn' || pokemon.status === 'tox') {
                this.add('-activate', pokemon, 'ability: Immunity');
                pokemon.cureStatus();
            }
        },
        onSetStatus: function (status, target, source, effect) {
            if (status.id !== 'psn' && status.id !== 'tox')
                return;
            if (effect === null || effect === void 0 ? void 0 : effect.status) {
                this.add('-immune', target, '[from] ability: Immunity');
            }
            return false;
        },
        flags: { breakable: 1 },
        name: "Immunity",
        rating: 2,
        num: 17,
    },
    imposter: {
        onSwitchIn: function (pokemon) {
            // Imposter does not activate when Skill Swapped or when Neutralizing Gas leaves the field
            // Imposter copies across in doubles/triples
            // (also copies across in multibattle and diagonally in free-for-all,
            // but side.foe already takes care of those)
            var target = pokemon.side.foe.active[pokemon.side.foe.active.length - 1 - pokemon.position];
            if (target) {
                pokemon.transformInto(target, this.dex.abilities.get('imposter'));
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1 },
        name: "Imposter",
        rating: 5,
        num: 150,
    },
    infiltrator: {
        onModifyMove: function (move) {
            move.infiltrates = true;
        },
        flags: {},
        name: "Infiltrator",
        rating: 2.5,
        num: 151,
    },
    innardsout: {
        onDamagingHitOrder: 1,
        onDamagingHit: function (damage, target, source, move) {
            if (!target.hp) {
                this.damage(target.getUndynamaxedHP(damage), source, target);
            }
        },
        flags: {},
        name: "Innards Out",
        rating: 4,
        num: 215,
    },
    innerfocus: {
        onTryAddVolatile: function (status, pokemon) {
            if (status.id === 'flinch')
                return null;
        },
        onTryBoost: function (boost, target, source, effect) {
            if (effect.name === 'Intimidate' && boost.atk) {
                delete boost.atk;
                this.add('-fail', target, 'unboost', 'Attack', '[from] ability: Inner Focus', "[of] ".concat(target));
            }
        },
        flags: { breakable: 1 },
        name: "Inner Focus",
        rating: 1,
        num: 39,
    },
    insomnia: {
        onUpdate: function (pokemon) {
            if (pokemon.status === 'slp') {
                this.add('-activate', pokemon, 'ability: Insomnia');
                pokemon.cureStatus();
            }
        },
        onSetStatus: function (status, target, source, effect) {
            if (status.id !== 'slp')
                return;
            if (effect === null || effect === void 0 ? void 0 : effect.status) {
                this.add('-immune', target, '[from] ability: Insomnia');
            }
            return false;
        },
        onTryAddVolatile: function (status, target) {
            if (status.id === 'yawn') {
                this.add('-immune', target, '[from] ability: Insomnia');
                return null;
            }
        },
        flags: { breakable: 1 },
        name: "Insomnia",
        rating: 1.5,
        num: 15,
    },
    intimidate: {
        onStart: function (pokemon) {
            var activated = false;
            for (var _i = 0, _a = pokemon.adjacentFoes(); _i < _a.length; _i++) {
                var target = _a[_i];
                if (!activated) {
                    this.add('-ability', pokemon, 'Intimidate', 'boost');
                    activated = true;
                }
                if (target.volatiles['substitute']) {
                    this.add('-immune', target);
                }
                else {
                    this.boost({ atk: -1 }, target, pokemon, null, true);
                }
            }
        },
        flags: {},
        name: "Intimidate",
        rating: 3.5,
        num: 22,
    },
    intrepidsword: {
        onStart: function (pokemon) {
            if (pokemon.swordBoost)
                return;
            pokemon.swordBoost = true;
            this.boost({ atk: 1 }, pokemon);
        },
        flags: {},
        name: "Intrepid Sword",
        rating: 4,
        num: 234,
    },
    ironbarbs: {
        onDamagingHitOrder: 1,
        onDamagingHit: function (damage, target, source, move) {
            if (this.checkMoveMakesContact(move, source, target, true)) {
                this.damage(source.baseMaxhp / 8, source, target);
            }
        },
        flags: {},
        name: "Iron Barbs",
        rating: 2.5,
        num: 160,
    },
    ironfist: {
        onBasePowerPriority: 23,
        onBasePower: function (basePower, attacker, defender, move) {
            if (move.flags['punch']) {
                this.debug('Iron Fist boost');
                return this.chainModify([4915, 4096]);
            }
        },
        flags: {},
        name: "Iron Fist",
        rating: 3,
        num: 89,
    },
    justified: {
        onDamagingHit: function (damage, target, source, move) {
            if (move.type === 'Dark') {
                this.boost({ atk: 1 });
            }
        },
        flags: {},
        name: "Justified",
        rating: 2.5,
        num: 154,
    },
    keeneye: {
        onTryBoost: function (boost, target, source, effect) {
            if (source && target === source)
                return;
            if (boost.accuracy && boost.accuracy < 0) {
                delete boost.accuracy;
                if (!effect.secondaries) {
                    this.add("-fail", target, "unboost", "accuracy", "[from] ability: Keen Eye", "[of] ".concat(target));
                }
            }
        },
        onModifyMove: function (move) {
            move.ignoreEvasion = true;
        },
        flags: { breakable: 1 },
        name: "Keen Eye",
        rating: 0.5,
        num: 51,
    },
    klutz: {
        // Klutz isn't technically active immediately in-game, but it activates early enough to beat all items
        // we should keep an eye out in future gens for items that activate on switch-in before Unnerve
        onSwitchInPriority: 1,
        // Item suppression implemented in Pokemon.ignoringItem() within sim/pokemon.js
        onStart: function (pokemon) {
            this.singleEvent('End', pokemon.getItem(), pokemon.itemState, pokemon);
        },
        flags: {},
        name: "Klutz",
        rating: -1,
        num: 103,
    },
    leafguard: {
        onSetStatus: function (status, target, source, effect) {
            if (['sunnyday', 'desolateland'].includes(target.effectiveWeather())) {
                if (effect === null || effect === void 0 ? void 0 : effect.status) {
                    this.add('-immune', target, '[from] ability: Leaf Guard');
                }
                return false;
            }
        },
        onTryAddVolatile: function (status, target) {
            if (status.id === 'yawn' && ['sunnyday', 'desolateland'].includes(target.effectiveWeather())) {
                this.add('-immune', target, '[from] ability: Leaf Guard');
                return null;
            }
        },
        flags: { breakable: 1 },
        name: "Leaf Guard",
        rating: 0.5,
        num: 102,
    },
    levitate: {
        // airborneness implemented in sim/pokemon.js:Pokemon#isGrounded
        flags: { breakable: 1 },
        name: "Levitate",
        rating: 3.5,
        num: 26,
    },
    libero: {
        onPrepareHit: function (source, target, move) {
            if (this.effectState.libero === source.previouslySwitchedIn)
                return;
            if (move.hasBounced || move.flags['futuremove'] || move.sourceEffect === 'snatch' || move.callsMove)
                return;
            var type = move.type;
            if (type && type !== '???' && source.getTypes().join() !== type) {
                if (!source.setType(type))
                    return;
                this.effectState.libero = source.previouslySwitchedIn;
                this.add('-start', source, 'typechange', type, '[from] ability: Libero');
            }
        },
        flags: {},
        name: "Libero",
        rating: 4,
        num: 236,
    },
    lightmetal: {
        onModifyWeight: function (weighthg) {
            return this.trunc(weighthg / 2);
        },
        flags: { breakable: 1 },
        name: "Light Metal",
        rating: 1,
        num: 135,
    },
    lightningrod: {
        onTryHit: function (target, source, move) {
            if (target !== source && move.type === 'Electric') {
                if (!this.boost({ spa: 1 })) {
                    this.add('-immune', target, '[from] ability: Lightning Rod');
                }
                return null;
            }
        },
        onAnyRedirectTarget: function (target, source, source2, move) {
            if (move.type !== 'Electric' || move.flags['pledgecombo'])
                return;
            var redirectTarget = ['randomNormal', 'adjacentFoe'].includes(move.target) ? 'normal' : move.target;
            if (this.validTarget(this.effectState.target, source, redirectTarget)) {
                if (move.smartTarget)
                    move.smartTarget = false;
                if (this.effectState.target !== target) {
                    this.add('-activate', this.effectState.target, 'ability: Lightning Rod');
                }
                return this.effectState.target;
            }
        },
        flags: { breakable: 1 },
        name: "Lightning Rod",
        rating: 3,
        num: 31,
    },
    limber: {
        onUpdate: function (pokemon) {
            if (pokemon.status === 'par') {
                this.add('-activate', pokemon, 'ability: Limber');
                pokemon.cureStatus();
            }
        },
        onSetStatus: function (status, target, source, effect) {
            if (status.id !== 'par')
                return;
            if (effect === null || effect === void 0 ? void 0 : effect.status) {
                this.add('-immune', target, '[from] ability: Limber');
            }
            return false;
        },
        flags: { breakable: 1 },
        name: "Limber",
        rating: 2,
        num: 7,
    },
    lingeringaroma: {
        onDamagingHit: function (damage, target, source, move) {
            var sourceAbility = source.getAbility();
            if (sourceAbility.flags['cantsuppress'] || sourceAbility.id === 'lingeringaroma') {
                return;
            }
            if (this.checkMoveMakesContact(move, source, target, !source.isAlly(target))) {
                var oldAbility = source.setAbility('lingeringaroma', target);
                if (oldAbility) {
                    this.add('-activate', target, 'ability: Lingering Aroma', this.dex.abilities.get(oldAbility).name, "[of] ".concat(source));
                }
            }
        },
        flags: {},
        name: "Lingering Aroma",
        rating: 2,
        num: 268,
    },
    liquidooze: {
        onSourceTryHeal: function (damage, target, source, effect) {
            this.debug("Heal is occurring: ".concat(target, " <- ").concat(source, " :: ").concat(effect.id));
            var canOoze = ['drain', 'leechseed', 'strengthsap'];
            if (canOoze.includes(effect.id)) {
                this.damage(damage);
                return 0;
            }
        },
        flags: {},
        name: "Liquid Ooze",
        rating: 2.5,
        num: 64,
    },
    liquidvoice: {
        onModifyTypePriority: -1,
        onModifyType: function (move, pokemon) {
            if (move.flags['sound'] && !pokemon.volatiles['dynamax']) { // hardcode
                move.type = 'Water';
            }
        },
        flags: {},
        name: "Liquid Voice",
        rating: 1.5,
        num: 204,
    },
    longreach: {
        onModifyMove: function (move) {
            delete move.flags['contact'];
        },
        flags: {},
        name: "Long Reach",
        rating: 1,
        num: 203,
    },
    magicbounce: {
        onTryHitPriority: 1,
        onTryHit: function (target, source, move) {
            if (target === source || move.hasBounced || !move.flags['reflectable'] || target.isSemiInvulnerable()) {
                return;
            }
            var newMove = this.dex.getActiveMove(move.id);
            newMove.hasBounced = true;
            newMove.pranksterBoosted = false;
            this.actions.useMove(newMove, target, { target: source });
            return null;
        },
        onAllyTryHitSide: function (target, source, move) {
            if (target.isAlly(source) || move.hasBounced || !move.flags['reflectable'] || target.isSemiInvulnerable()) {
                return;
            }
            var newMove = this.dex.getActiveMove(move.id);
            newMove.hasBounced = true;
            newMove.pranksterBoosted = false;
            this.actions.useMove(newMove, this.effectState.target, { target: source });
            return null;
        },
        condition: {
            duration: 1,
        },
        flags: { breakable: 1 },
        name: "Magic Bounce",
        rating: 4,
        num: 156,
    },
    magicguard: {
        onDamage: function (damage, target, source, effect) {
            if (effect.effectType !== 'Move') {
                if (effect.effectType === 'Ability')
                    this.add('-activate', source, 'ability: ' + effect.name);
                return false;
            }
        },
        flags: {},
        name: "Magic Guard",
        rating: 4,
        num: 98,
    },
    magician: {
        onAfterMoveSecondarySelf: function (source, target, move) {
            if (!move || !target || source.switchFlag === true)
                return;
            if (target !== source && move.category !== 'Status') {
                if (source.item || source.volatiles['gem'] || move.id === 'fling')
                    return;
                var yourItem = target.takeItem(source);
                if (!yourItem)
                    return;
                if (!source.setItem(yourItem)) {
                    target.item = yourItem.id; // bypass setItem so we don't break choicelock or anything
                    return;
                }
                this.add('-item', source, yourItem, '[from] ability: Magician', "[of] ".concat(target));
            }
        },
        flags: {},
        name: "Magician",
        rating: 1,
        num: 170,
    },
    magmaarmor: {
        onUpdate: function (pokemon) {
            if (pokemon.status === 'frz') {
                this.add('-activate', pokemon, 'ability: Magma Armor');
                pokemon.cureStatus();
            }
        },
        onImmunity: function (type, pokemon) {
            if (type === 'frz')
                return false;
        },
        flags: { breakable: 1 },
        name: "Magma Armor",
        rating: 0.5,
        num: 40,
    },
    magnetpull: {
        onFoeTrapPokemon: function (pokemon) {
            if (pokemon.hasType('Steel') && pokemon.isAdjacent(this.effectState.target)) {
                pokemon.tryTrap(true);
            }
        },
        onFoeMaybeTrapPokemon: function (pokemon, source) {
            if (!source)
                source = this.effectState.target;
            if (!source || !pokemon.isAdjacent(source))
                return;
            if (!pokemon.knownType || pokemon.hasType('Steel')) {
                pokemon.maybeTrapped = true;
            }
        },
        flags: {},
        name: "Magnet Pull",
        rating: 4,
        num: 42,
    },
    marvelscale: {
        onModifyDefPriority: 6,
        onModifyDef: function (def, pokemon) {
            if (pokemon.status) {
                return this.chainModify(1.5);
            }
        },
        flags: { breakable: 1 },
        name: "Marvel Scale",
        rating: 2.5,
        num: 63,
    },
    megalauncher: {
        onBasePowerPriority: 19,
        onBasePower: function (basePower, attacker, defender, move) {
            if (move.flags['pulse']) {
                return this.chainModify(1.5);
            }
        },
        flags: {},
        name: "Mega Launcher",
        rating: 3,
        num: 178,
    },
    merciless: {
        onModifyCritRatio: function (critRatio, source, target) {
            if (target && ['psn', 'tox'].includes(target.status))
                return 5;
        },
        flags: {},
        name: "Merciless",
        rating: 1.5,
        num: 196,
    },
    mimicry: {
        onSwitchInPriority: -1,
        onStart: function (pokemon) {
            this.singleEvent('TerrainChange', this.effect, this.effectState, pokemon);
        },
        onTerrainChange: function (pokemon) {
            var types;
            switch (this.field.terrain) {
                case 'electricterrain':
                    types = ['Electric'];
                    break;
                case 'grassyterrain':
                    types = ['Grass'];
                    break;
                case 'mistyterrain':
                    types = ['Fairy'];
                    break;
                case 'psychicterrain':
                    types = ['Psychic'];
                    break;
                default:
                    types = pokemon.baseSpecies.types;
            }
            var oldTypes = pokemon.getTypes();
            if (oldTypes.join() === types.join() || !pokemon.setType(types))
                return;
            if (this.field.terrain || pokemon.transformed) {
                this.add('-start', pokemon, 'typechange', types.join('/'), '[from] ability: Mimicry');
                if (!this.field.terrain)
                    this.hint("Transform Mimicry changes you to your original un-transformed types.");
            }
            else {
                this.add('-activate', pokemon, 'ability: Mimicry');
                this.add('-end', pokemon, 'typechange', '[silent]');
            }
        },
        flags: {},
        name: "Mimicry",
        rating: 0,
        num: 250,
    },
    mindseye: {
        onTryBoost: function (boost, target, source, effect) {
            if (source && target === source)
                return;
            if (boost.accuracy && boost.accuracy < 0) {
                delete boost.accuracy;
                if (!effect.secondaries) {
                    this.add("-fail", target, "unboost", "accuracy", "[from] ability: Mind's Eye", "[of] ".concat(target));
                }
            }
        },
        onModifyMovePriority: -5,
        onModifyMove: function (move) {
            move.ignoreEvasion = true;
            if (!move.ignoreImmunity)
                move.ignoreImmunity = {};
            if (move.ignoreImmunity !== true) {
                move.ignoreImmunity['Fighting'] = true;
                move.ignoreImmunity['Normal'] = true;
            }
        },
        flags: { breakable: 1 },
        name: "Mind's Eye",
        rating: 0,
        num: 300,
    },
    minus: {
        onModifySpAPriority: 5,
        onModifySpA: function (spa, pokemon) {
            for (var _i = 0, _a = pokemon.allies(); _i < _a.length; _i++) {
                var allyActive = _a[_i];
                if (allyActive.hasAbility(['minus', 'plus'])) {
                    return this.chainModify(1.5);
                }
            }
        },
        flags: {},
        name: "Minus",
        rating: 0,
        num: 58,
    },
    mirrorarmor: {
        onTryBoost: function (boost, target, source, effect) {
            // Don't bounce self stat changes, or boosts that have already bounced
            if (!source || target === source || !boost || effect.name === 'Mirror Armor')
                return;
            var b;
            for (b in boost) {
                if (boost[b] < 0) {
                    if (target.boosts[b] === -6)
                        continue;
                    var negativeBoost = {};
                    negativeBoost[b] = boost[b];
                    delete boost[b];
                    if (source.hp) {
                        this.add('-ability', target, 'Mirror Armor');
                        this.boost(negativeBoost, source, target, null, true);
                    }
                }
            }
        },
        flags: { breakable: 1 },
        name: "Mirror Armor",
        rating: 2,
        num: 240,
    },
    mistysurge: {
        onStart: function (source) {
            this.field.setTerrain('mistyterrain');
        },
        flags: {},
        name: "Misty Surge",
        rating: 3.5,
        num: 228,
    },
    moldbreaker: {
        onStart: function (pokemon) {
            this.add('-ability', pokemon, 'Mold Breaker');
        },
        onModifyMove: function (move) {
            move.ignoreAbility = true;
        },
        flags: {},
        name: "Mold Breaker",
        rating: 3,
        num: 104,
    },
    moody: {
        onResidualOrder: 28,
        onResidualSubOrder: 2,
        onResidual: function (pokemon) {
            var stats = [];
            var boost = {};
            var statPlus;
            for (statPlus in pokemon.boosts) {
                if (statPlus === 'accuracy' || statPlus === 'evasion')
                    continue;
                if (pokemon.boosts[statPlus] < 6) {
                    stats.push(statPlus);
                }
            }
            var randomStat = stats.length ? this.sample(stats) : undefined;
            if (randomStat)
                boost[randomStat] = 2;
            stats = [];
            var statMinus;
            for (statMinus in pokemon.boosts) {
                if (statMinus === 'accuracy' || statMinus === 'evasion')
                    continue;
                if (pokemon.boosts[statMinus] > -6 && statMinus !== randomStat) {
                    stats.push(statMinus);
                }
            }
            randomStat = stats.length ? this.sample(stats) : undefined;
            if (randomStat)
                boost[randomStat] = -1;
            this.boost(boost, pokemon, pokemon);
        },
        flags: {},
        name: "Moody",
        rating: 5,
        num: 141,
    },
    motordrive: {
        onTryHit: function (target, source, move) {
            if (target !== source && move.type === 'Electric') {
                if (!this.boost({ spe: 1 })) {
                    this.add('-immune', target, '[from] ability: Motor Drive');
                }
                return null;
            }
        },
        flags: { breakable: 1 },
        name: "Motor Drive",
        rating: 3,
        num: 78,
    },
    moxie: {
        onSourceAfterFaint: function (length, target, source, effect) {
            if (effect && effect.effectType === 'Move') {
                this.boost({ atk: length }, source);
            }
        },
        flags: {},
        name: "Moxie",
        rating: 3,
        num: 153,
    },
    multiscale: {
        onSourceModifyDamage: function (damage, source, target, move) {
            if (target.hp >= target.maxhp) {
                this.debug('Multiscale weaken');
                return this.chainModify(0.5);
            }
        },
        flags: { breakable: 1 },
        name: "Multiscale",
        rating: 3.5,
        num: 136,
    },
    multitype: {
        // Multitype's type-changing itself is implemented in statuses.js
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, cantsuppress: 1 },
        name: "Multitype",
        rating: 4,
        num: 121,
    },
    mummy: {
        onDamagingHit: function (damage, target, source, move) {
            var sourceAbility = source.getAbility();
            if (sourceAbility.flags['cantsuppress'] || sourceAbility.id === 'mummy') {
                return;
            }
            if (this.checkMoveMakesContact(move, source, target, !source.isAlly(target))) {
                var oldAbility = source.setAbility('mummy', target);
                if (oldAbility) {
                    this.add('-activate', target, 'ability: Mummy', this.dex.abilities.get(oldAbility).name, "[of] ".concat(source));
                }
            }
        },
        flags: {},
        name: "Mummy",
        rating: 2,
        num: 152,
    },
    myceliummight: {
        onFractionalPriorityPriority: -1,
        onFractionalPriority: function (priority, pokemon, target, move) {
            if (move.category === 'Status') {
                return -0.1;
            }
        },
        onModifyMove: function (move) {
            if (move.category === 'Status') {
                move.ignoreAbility = true;
            }
        },
        flags: {},
        name: "Mycelium Might",
        rating: 2,
        num: 298,
    },
    naturalcure: {
        onCheckShow: function (pokemon) {
            // This is complicated
            // For the most part, in-game, it's obvious whether or not Natural Cure activated,
            // since you can see how many of your opponent's pokemon are statused.
            // The only ambiguous situation happens in Doubles/Triples, where multiple pokemon
            // that could have Natural Cure switch out, but only some of them get cured.
            if (pokemon.side.active.length === 1)
                return;
            if (pokemon.showCure === true || pokemon.showCure === false)
                return;
            var cureList = [];
            var noCureCount = 0;
            for (var _i = 0, _a = pokemon.side.active; _i < _a.length; _i++) {
                var curPoke = _a[_i];
                // pokemon not statused
                if (!(curPoke === null || curPoke === void 0 ? void 0 : curPoke.status)) {
                    // this.add('-message', "" + curPoke + " skipped: not statused or doesn't exist");
                    continue;
                }
                if (curPoke.showCure) {
                    // this.add('-message', "" + curPoke + " skipped: Natural Cure already known");
                    continue;
                }
                var species = curPoke.species;
                // pokemon can't get Natural Cure
                if (!Object.values(species.abilities).includes('Natural Cure')) {
                    // this.add('-message', "" + curPoke + " skipped: no Natural Cure");
                    continue;
                }
                // pokemon's ability is known to be Natural Cure
                if (!species.abilities['1'] && !species.abilities['H']) {
                    // this.add('-message', "" + curPoke + " skipped: only one ability");
                    continue;
                }
                // pokemon isn't switching this turn
                if (curPoke !== pokemon && !this.queue.willSwitch(curPoke)) {
                    // this.add('-message', "" + curPoke + " skipped: not switching");
                    continue;
                }
                if (curPoke.hasAbility('naturalcure')) {
                    // this.add('-message', "" + curPoke + " confirmed: could be Natural Cure (and is)");
                    cureList.push(curPoke);
                }
                else {
                    // this.add('-message', "" + curPoke + " confirmed: could be Natural Cure (but isn't)");
                    noCureCount++;
                }
            }
            if (!cureList.length || !noCureCount) {
                // It's possible to know what pokemon were cured
                for (var _b = 0, cureList_1 = cureList; _b < cureList_1.length; _b++) {
                    var pkmn = cureList_1[_b];
                    pkmn.showCure = true;
                }
            }
            else {
                // It's not possible to know what pokemon were cured
                // Unlike a -hint, this is real information that battlers need, so we use a -message
                this.add('-message', "(".concat(cureList.length, " of ").concat(pokemon.side.name, "'s pokemon ").concat(cureList.length === 1 ? "was" : "were", " cured by Natural Cure.)"));
                for (var _c = 0, cureList_2 = cureList; _c < cureList_2.length; _c++) {
                    var pkmn = cureList_2[_c];
                    pkmn.showCure = false;
                }
            }
        },
        onSwitchOut: function (pokemon) {
            if (!pokemon.status)
                return;
            // if pokemon.showCure is undefined, it was skipped because its ability
            // is known
            if (pokemon.showCure === undefined)
                pokemon.showCure = true;
            if (pokemon.showCure)
                this.add('-curestatus', pokemon, pokemon.status, '[from] ability: Natural Cure');
            pokemon.clearStatus();
            // only reset .showCure if it's false
            // (once you know a Pokemon has Natural Cure, its cures are always known)
            if (!pokemon.showCure)
                pokemon.showCure = undefined;
        },
        flags: {},
        name: "Natural Cure",
        rating: 2.5,
        num: 30,
    },
    neuroforce: {
        onModifyDamage: function (damage, source, target, move) {
            if (move && target.getMoveHitData(move).typeMod > 0) {
                return this.chainModify([5120, 4096]);
            }
        },
        flags: {},
        name: "Neuroforce",
        rating: 2.5,
        num: 233,
    },
    neutralizinggas: {
        // Ability suppression implemented in sim/pokemon.ts:Pokemon#ignoringAbility
        onSwitchInPriority: 2,
        onSwitchIn: function (pokemon) {
            this.add('-ability', pokemon, 'Neutralizing Gas');
            pokemon.abilityState.ending = false;
            var strongWeathers = ['desolateland', 'primordialsea', 'deltastream'];
            for (var _i = 0, _a = this.getAllActive(); _i < _a.length; _i++) {
                var target = _a[_i];
                if (target.hasItem('Ability Shield')) {
                    this.add('-block', target, 'item: Ability Shield');
                    continue;
                }
                // Can't suppress a Tatsugiri inside of Dondozo already
                if (target.volatiles['commanding']) {
                    continue;
                }
                if (target.illusion) {
                    this.singleEvent('End', this.dex.abilities.get('Illusion'), target.abilityState, target, pokemon, 'neutralizinggas');
                }
                if (target.volatiles['slowstart']) {
                    delete target.volatiles['slowstart'];
                    this.add('-end', target, 'Slow Start', '[silent]');
                }
                if (strongWeathers.includes(target.getAbility().id)) {
                    this.singleEvent('End', this.dex.abilities.get(target.getAbility().id), target.abilityState, target, pokemon, 'neutralizinggas');
                }
            }
        },
        onEnd: function (source) {
            if (source.transformed)
                return;
            for (var _i = 0, _a = this.getAllActive(); _i < _a.length; _i++) {
                var pokemon = _a[_i];
                if (pokemon !== source && pokemon.hasAbility('Neutralizing Gas')) {
                    return;
                }
            }
            this.add('-end', source, 'ability: Neutralizing Gas');
            // FIXME this happens before the pokemon switches out, should be the opposite order.
            // Not an easy fix since we cant use a supported event. Would need some kind of special event that
            // gathers events to run after the switch and then runs them when the ability is no longer accessible.
            // (If you're tackling this, do note extreme weathers have the same issue)
            // Mark this pokemon's ability as ending so Pokemon#ignoringAbility skips it
            if (source.abilityState.ending)
                return;
            source.abilityState.ending = true;
            var sortedActive = this.getAllActive();
            this.speedSort(sortedActive);
            for (var _b = 0, sortedActive_1 = sortedActive; _b < sortedActive_1.length; _b++) {
                var pokemon = sortedActive_1[_b];
                if (pokemon !== source) {
                    if (pokemon.getAbility().flags['cantsuppress'])
                        continue; // does not interact with e.g Ice Face, Zen Mode
                    if (pokemon.hasItem('abilityshield'))
                        continue; // don't restart abilities that weren't suppressed
                    // Will be suppressed by Pokemon#ignoringAbility if needed
                    this.singleEvent('Start', pokemon.getAbility(), pokemon.abilityState, pokemon);
                    if (pokemon.ability === "gluttony") {
                        pokemon.abilityState.gluttony = false;
                    }
                }
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, notransform: 1 },
        name: "Neutralizing Gas",
        rating: 3.5,
        num: 256,
    },
    noguard: {
        onAnyInvulnerabilityPriority: 1,
        onAnyInvulnerability: function (target, source, move) {
            if (move && (source === this.effectState.target || target === this.effectState.target))
                return 0;
        },
        onAnyAccuracy: function (accuracy, target, source, move) {
            if (move && (source === this.effectState.target || target === this.effectState.target)) {
                return true;
            }
            return accuracy;
        },
        flags: {},
        name: "No Guard",
        rating: 4,
        num: 99,
    },
    normalize: {
        onModifyTypePriority: 1,
        onModifyType: function (move, pokemon) {
            var noModifyType = [
                'hiddenpower', 'judgment', 'multiattack', 'naturalgift', 'revelationdance', 'struggle', 'technoblast', 'terrainpulse', 'weatherball',
            ];
            if (!(move.isZ && move.category !== 'Status') && !noModifyType.includes(move.id) &&
                // TODO: Figure out actual interaction
                !(move.name === 'Tera Blast' && pokemon.terastallized)) {
                move.type = 'Normal';
                move.typeChangerBoosted = this.effect;
            }
        },
        onBasePowerPriority: 23,
        onBasePower: function (basePower, pokemon, target, move) {
            if (move.typeChangerBoosted === this.effect)
                return this.chainModify([4915, 4096]);
        },
        flags: {},
        name: "Normalize",
        rating: 0,
        num: 96,
    },
    oblivious: {
        onUpdate: function (pokemon) {
            if (pokemon.volatiles['attract']) {
                this.add('-activate', pokemon, 'ability: Oblivious');
                pokemon.removeVolatile('attract');
                this.add('-end', pokemon, 'move: Attract', '[from] ability: Oblivious');
            }
            if (pokemon.volatiles['taunt']) {
                this.add('-activate', pokemon, 'ability: Oblivious');
                pokemon.removeVolatile('taunt');
                // Taunt's volatile already sends the -end message when removed
            }
        },
        onImmunity: function (type, pokemon) {
            if (type === 'attract')
                return false;
        },
        onTryHit: function (pokemon, target, move) {
            if (move.id === 'attract' || move.id === 'captivate' || move.id === 'taunt') {
                this.add('-immune', pokemon, '[from] ability: Oblivious');
                return null;
            }
        },
        onTryBoost: function (boost, target, source, effect) {
            if (effect.name === 'Intimidate' && boost.atk) {
                delete boost.atk;
                this.add('-fail', target, 'unboost', 'Attack', '[from] ability: Oblivious', "[of] ".concat(target));
            }
        },
        flags: { breakable: 1 },
        name: "Oblivious",
        rating: 1.5,
        num: 12,
    },
    opportunist: {
        onFoeAfterBoost: function (boost, target, source, effect) {
            if ((effect === null || effect === void 0 ? void 0 : effect.name) === 'Opportunist' || (effect === null || effect === void 0 ? void 0 : effect.name) === 'Mirror Herb')
                return;
            if (!this.effectState.boosts)
                this.effectState.boosts = {};
            var boostPlus = this.effectState.boosts;
            var i;
            for (i in boost) {
                if (boost[i] > 0) {
                    boostPlus[i] = (boostPlus[i] || 0) + boost[i];
                }
            }
        },
        onAnySwitchInPriority: -3,
        onAnySwitchIn: function () {
            if (!this.effectState.boosts)
                return;
            this.boost(this.effectState.boosts, this.effectState.target);
            delete this.effectState.boosts;
        },
        onAnyAfterMove: function () {
            if (!this.effectState.boosts)
                return;
            this.boost(this.effectState.boosts, this.effectState.target);
            delete this.effectState.boosts;
        },
        onResidualOrder: 29,
        onResidual: function (pokemon) {
            if (!this.effectState.boosts)
                return;
            this.boost(this.effectState.boosts, this.effectState.target);
            delete this.effectState.boosts;
        },
        onEnd: function () {
            delete this.effectState.boosts;
        },
        flags: {},
        name: "Opportunist",
        rating: 3,
        num: 290,
    },
    orichalcumpulse: {
        onStart: function (pokemon) {
            if (this.field.setWeather('sunnyday')) {
                this.add('-activate', pokemon, 'Orichalcum Pulse', '[source]');
            }
            else if (this.field.isWeather('sunnyday')) {
                this.add('-activate', pokemon, 'ability: Orichalcum Pulse');
            }
        },
        onModifyAtkPriority: 5,
        onModifyAtk: function (atk, pokemon) {
            if (['sunnyday', 'desolateland'].includes(pokemon.effectiveWeather())) {
                this.debug('Orichalcum boost');
                return this.chainModify([5461, 4096]);
            }
        },
        flags: {},
        name: "Orichalcum Pulse",
        rating: 4.5,
        num: 288,
    },
    overcoat: {
        onImmunity: function (type, pokemon) {
            if (type === 'sandstorm' || type === 'hail' || type === 'powder')
                return false;
        },
        onTryHitPriority: 1,
        onTryHit: function (target, source, move) {
            if (move.flags['powder'] && target !== source && this.dex.getImmunity('powder', target)) {
                this.add('-immune', target, '[from] ability: Overcoat');
                return null;
            }
        },
        flags: { breakable: 1 },
        name: "Overcoat",
        rating: 2,
        num: 142,
    },
    overgrow: {
        onModifyAtkPriority: 5,
        onModifyAtk: function (atk, attacker, defender, move) {
            if (move.type === 'Grass' && attacker.hp <= attacker.maxhp / 3) {
                this.debug('Overgrow boost');
                return this.chainModify(1.5);
            }
        },
        onModifySpAPriority: 5,
        onModifySpA: function (atk, attacker, defender, move) {
            if (move.type === 'Grass' && attacker.hp <= attacker.maxhp / 3) {
                this.debug('Overgrow boost');
                return this.chainModify(1.5);
            }
        },
        flags: {},
        name: "Overgrow",
        rating: 2,
        num: 65,
    },
    owntempo: {
        onUpdate: function (pokemon) {
            if (pokemon.volatiles['confusion']) {
                this.add('-activate', pokemon, 'ability: Own Tempo');
                pokemon.removeVolatile('confusion');
            }
        },
        onTryAddVolatile: function (status, pokemon) {
            if (status.id === 'confusion')
                return null;
        },
        onHit: function (target, source, move) {
            if ((move === null || move === void 0 ? void 0 : move.volatileStatus) === 'confusion') {
                this.add('-immune', target, 'confusion', '[from] ability: Own Tempo');
            }
        },
        onTryBoost: function (boost, target, source, effect) {
            if (effect.name === 'Intimidate' && boost.atk) {
                delete boost.atk;
                this.add('-fail', target, 'unboost', 'Attack', '[from] ability: Own Tempo', "[of] ".concat(target));
            }
        },
        flags: { breakable: 1 },
        name: "Own Tempo",
        rating: 1.5,
        num: 20,
    },
    parentalbond: {
        onPrepareHit: function (source, target, move) {
            if (move.category === 'Status' || move.multihit || move.flags['noparentalbond'] || move.flags['charge'] ||
                move.flags['futuremove'] || move.spreadHit || move.isZ || move.isMax)
                return;
            move.multihit = 2;
            move.multihitType = 'parentalbond';
        },
        // Damage modifier implemented in BattleActions#modifyDamage()
        onSourceModifySecondaries: function (secondaries, target, source, move) {
            if (move.multihitType === 'parentalbond' && move.id === 'secretpower' && move.hit < 2) {
                // hack to prevent accidentally suppressing King's Rock/Razor Fang
                return secondaries.filter(function (effect) { return effect.volatileStatus === 'flinch'; });
            }
        },
        flags: {},
        name: "Parental Bond",
        rating: 4.5,
        num: 185,
    },
    pastelveil: {
        onStart: function (pokemon) {
            for (var _i = 0, _a = pokemon.alliesAndSelf(); _i < _a.length; _i++) {
                var ally = _a[_i];
                if (['psn', 'tox'].includes(ally.status)) {
                    this.add('-activate', pokemon, 'ability: Pastel Veil');
                    ally.cureStatus();
                }
            }
        },
        onUpdate: function (pokemon) {
            if (['psn', 'tox'].includes(pokemon.status)) {
                this.add('-activate', pokemon, 'ability: Pastel Veil');
                pokemon.cureStatus();
            }
        },
        onAnySwitchIn: function () {
            this.effect.onStart.call(this, this.effectState.target);
        },
        onSetStatus: function (status, target, source, effect) {
            if (!['psn', 'tox'].includes(status.id))
                return;
            if (effect === null || effect === void 0 ? void 0 : effect.status) {
                this.add('-immune', target, '[from] ability: Pastel Veil');
            }
            return false;
        },
        onAllySetStatus: function (status, target, source, effect) {
            if (!['psn', 'tox'].includes(status.id))
                return;
            if (effect === null || effect === void 0 ? void 0 : effect.status) {
                var effectHolder = this.effectState.target;
                this.add('-block', target, 'ability: Pastel Veil', "[of] ".concat(effectHolder));
            }
            return false;
        },
        flags: { breakable: 1 },
        name: "Pastel Veil",
        rating: 2,
        num: 257,
    },
    perishbody: {
        onDamagingHit: function (damage, target, source, move) {
            if (!this.checkMoveMakesContact(move, source, target) || source.volatiles['perishsong'])
                return;
            this.add('-ability', target, 'Perish Body');
            source.addVolatile('perishsong');
            target.addVolatile('perishsong');
        },
        flags: {},
        name: "Perish Body",
        rating: 1,
        num: 253,
    },
    pickpocket: {
        onAfterMoveSecondary: function (target, source, move) {
            if (source && source !== target && (move === null || move === void 0 ? void 0 : move.flags['contact'])) {
                if (target.item || target.switchFlag || target.forceSwitchFlag || source.switchFlag === true) {
                    return;
                }
                var yourItem = source.takeItem(target);
                if (!yourItem) {
                    return;
                }
                if (!target.setItem(yourItem)) {
                    source.item = yourItem.id;
                    return;
                }
                this.add('-enditem', source, yourItem, '[silent]', '[from] ability: Pickpocket', "[of] ".concat(source));
                this.add('-item', target, yourItem, '[from] ability: Pickpocket', "[of] ".concat(source));
            }
        },
        flags: {},
        name: "Pickpocket",
        rating: 1,
        num: 124,
    },
    pickup: {
        onResidualOrder: 28,
        onResidualSubOrder: 2,
        onResidual: function (pokemon) {
            if (pokemon.item)
                return;
            var pickupTargets = this.getAllActive().filter(function (target) { return (target.lastItem && target.usedItemThisTurn && pokemon.isAdjacent(target)); });
            if (!pickupTargets.length)
                return;
            var randomTarget = this.sample(pickupTargets);
            var item = randomTarget.lastItem;
            randomTarget.lastItem = '';
            this.add('-item', pokemon, this.dex.items.get(item), '[from] ability: Pickup');
            pokemon.setItem(item);
        },
        flags: {},
        name: "Pickup",
        rating: 0.5,
        num: 53,
    },
    pixilate: {
        onModifyTypePriority: -1,
        onModifyType: function (move, pokemon) {
            var noModifyType = [
                'judgment', 'multiattack', 'naturalgift', 'revelationdance', 'technoblast', 'terrainpulse', 'weatherball',
            ];
            if (move.type === 'Normal' && !noModifyType.includes(move.id) &&
                !(move.isZ && move.category !== 'Status') && !(move.name === 'Tera Blast' && pokemon.terastallized)) {
                move.type = 'Fairy';
                move.typeChangerBoosted = this.effect;
            }
        },
        onBasePowerPriority: 23,
        onBasePower: function (basePower, pokemon, target, move) {
            if (move.typeChangerBoosted === this.effect)
                return this.chainModify([4915, 4096]);
        },
        flags: {},
        name: "Pixilate",
        rating: 4,
        num: 182,
    },
    plus: {
        onModifySpAPriority: 5,
        onModifySpA: function (spa, pokemon) {
            for (var _i = 0, _a = pokemon.allies(); _i < _a.length; _i++) {
                var allyActive = _a[_i];
                if (allyActive.hasAbility(['minus', 'plus'])) {
                    return this.chainModify(1.5);
                }
            }
        },
        flags: {},
        name: "Plus",
        rating: 0,
        num: 57,
    },
    poisonheal: {
        onDamagePriority: 1,
        onDamage: function (damage, target, source, effect) {
            if (effect.id === 'psn' || effect.id === 'tox') {
                this.heal(target.baseMaxhp / 8);
                return false;
            }
        },
        flags: {},
        name: "Poison Heal",
        rating: 4,
        num: 90,
    },
    poisonpoint: {
        onDamagingHit: function (damage, target, source, move) {
            if (this.checkMoveMakesContact(move, source, target)) {
                if (this.randomChance(3, 10)) {
                    source.trySetStatus('psn', target);
                }
            }
        },
        flags: {},
        name: "Poison Point",
        rating: 1.5,
        num: 38,
    },
    poisonpuppeteer: {
        onAnyAfterSetStatus: function (status, target, source, effect) {
            if (source.baseSpecies.name !== "Pecharunt")
                return;
            if (source !== this.effectState.target || target === source || effect.effectType !== 'Move')
                return;
            if (status.id === 'psn' || status.id === 'tox') {
                target.addVolatile('confusion');
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1 },
        name: "Poison Puppeteer",
        rating: 3,
        num: 310,
    },
    poisontouch: {
        onSourceDamagingHit: function (damage, target, source, move) {
            // Despite not being a secondary, Shield Dust / Covert Cloak block Poison Touch's effect
            if (target.hasAbility('shielddust') || target.hasItem('covertcloak'))
                return;
            if (this.checkMoveMakesContact(move, target, source)) {
                if (this.randomChance(3, 10)) {
                    target.trySetStatus('psn', source);
                }
            }
        },
        flags: {},
        name: "Poison Touch",
        rating: 2,
        num: 143,
    },
    powerconstruct: {
        onResidualOrder: 29,
        onResidual: function (pokemon) {
            if (pokemon.baseSpecies.baseSpecies !== 'Zygarde' || pokemon.transformed || !pokemon.hp)
                return;
            if (pokemon.species.id === 'zygardecomplete' || pokemon.hp > pokemon.maxhp / 2)
                return;
            this.add('-activate', pokemon, 'ability: Power Construct');
            pokemon.formeChange('Zygarde-Complete', this.effect, true);
            pokemon.baseMaxhp = Math.floor(Math.floor(2 * pokemon.species.baseStats['hp'] + pokemon.set.ivs['hp'] + Math.floor(pokemon.set.evs['hp'] / 4) + 100) * pokemon.level / 100 + 10);
            var newMaxHP = pokemon.volatiles['dynamax'] ? (2 * pokemon.baseMaxhp) : pokemon.baseMaxhp;
            pokemon.hp = newMaxHP - (pokemon.maxhp - pokemon.hp);
            pokemon.maxhp = newMaxHP;
            this.add('-heal', pokemon, pokemon.getHealth, '[silent]');
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, cantsuppress: 1 },
        name: "Power Construct",
        rating: 5,
        num: 211,
    },
    powerofalchemy: {
        onAllyFaint: function (target) {
            if (!this.effectState.target.hp)
                return;
            var ability = target.getAbility();
            if (ability.flags['noreceiver'] || ability.id === 'noability')
                return;
            if (this.effectState.target.setAbility(ability)) {
                this.add('-ability', this.effectState.target, ability, '[from] ability: Power of Alchemy', "[of] ".concat(target));
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1 },
        name: "Power of Alchemy",
        rating: 0,
        num: 223,
    },
    powerspot: {
        onAllyBasePowerPriority: 22,
        onAllyBasePower: function (basePower, attacker, defender, move) {
            if (attacker !== this.effectState.target) {
                this.debug('Power Spot boost');
                return this.chainModify([5325, 4096]);
            }
        },
        flags: {},
        name: "Power Spot",
        rating: 0,
        num: 249,
    },
    prankster: {
        onModifyPriority: function (priority, pokemon, target, move) {
            if ((move === null || move === void 0 ? void 0 : move.category) === 'Status') {
                move.pranksterBoosted = true;
                return priority + 1;
            }
        },
        flags: {},
        name: "Prankster",
        rating: 4,
        num: 158,
    },
    pressure: {
        onStart: function (pokemon) {
            this.add('-ability', pokemon, 'Pressure');
        },
        onDeductPP: function (target, source) {
            if (target.isAlly(source))
                return;
            return 1;
        },
        flags: {},
        name: "Pressure",
        rating: 2.5,
        num: 46,
    },
    primordialsea: {
        onStart: function (source) {
            this.field.setWeather('primordialsea');
        },
        onAnySetWeather: function (target, source, weather) {
            var strongWeathers = ['desolateland', 'primordialsea', 'deltastream'];
            if (this.field.getWeather().id === 'primordialsea' && !strongWeathers.includes(weather.id))
                return false;
        },
        onEnd: function (pokemon) {
            if (this.field.weatherState.source !== pokemon)
                return;
            for (var _i = 0, _a = this.getAllActive(); _i < _a.length; _i++) {
                var target = _a[_i];
                if (target === pokemon)
                    continue;
                if (target.hasAbility('primordialsea')) {
                    this.field.weatherState.source = target;
                    return;
                }
            }
            this.field.clearWeather();
        },
        flags: {},
        name: "Primordial Sea",
        rating: 4.5,
        num: 189,
    },
    prismarmor: {
        onSourceModifyDamage: function (damage, source, target, move) {
            if (target.getMoveHitData(move).typeMod > 0) {
                this.debug('Prism Armor neutralize');
                return this.chainModify(0.75);
            }
        },
        flags: {},
        name: "Prism Armor",
        rating: 3,
        num: 232,
    },
    propellertail: {
        onModifyMovePriority: 1,
        onModifyMove: function (move) {
            // most of the implementation is in Battle#getTarget
            move.tracksTarget = move.target !== 'scripted';
        },
        flags: {},
        name: "Propeller Tail",
        rating: 0,
        num: 239,
    },
    protean: {
        onPrepareHit: function (source, target, move) {
            if (this.effectState.protean === source.previouslySwitchedIn)
                return;
            if (move.hasBounced || move.flags['futuremove'] || move.sourceEffect === 'snatch' || move.callsMove)
                return;
            var type = move.type;
            if (type && type !== '???' && source.getTypes().join() !== type) {
                if (!source.setType(type))
                    return;
                this.effectState.protean = source.previouslySwitchedIn;
                this.add('-start', source, 'typechange', type, '[from] ability: Protean');
            }
        },
        flags: {},
        name: "Protean",
        rating: 4,
        num: 168,
    },
    protosynthesis: {
        onSwitchInPriority: -2,
        onStart: function (pokemon) {
            this.singleEvent('WeatherChange', this.effect, this.effectState, pokemon);
        },
        onWeatherChange: function (pokemon) {
            var _a;
            // Protosynthesis is not affected by Utility Umbrella
            if (this.field.isWeather('sunnyday')) {
                pokemon.addVolatile('protosynthesis');
            }
            else if (!((_a = pokemon.volatiles['protosynthesis']) === null || _a === void 0 ? void 0 : _a.fromBooster) && !this.field.isWeather('sunnyday')) {
                pokemon.removeVolatile('protosynthesis');
            }
        },
        onEnd: function (pokemon) {
            delete pokemon.volatiles['protosynthesis'];
            this.add('-end', pokemon, 'Protosynthesis', '[silent]');
        },
        condition: {
            noCopy: true,
            onStart: function (pokemon, source, effect) {
                if ((effect === null || effect === void 0 ? void 0 : effect.name) === 'Booster Energy') {
                    this.effectState.fromBooster = true;
                    this.add('-activate', pokemon, 'ability: Protosynthesis', '[fromitem]');
                }
                else {
                    this.add('-activate', pokemon, 'ability: Protosynthesis');
                }
                this.effectState.bestStat = pokemon.getBestStat(false, true);
                this.add('-start', pokemon, 'protosynthesis' + this.effectState.bestStat);
            },
            onModifyAtkPriority: 5,
            onModifyAtk: function (atk, pokemon) {
                if (this.effectState.bestStat !== 'atk' || pokemon.ignoringAbility())
                    return;
                this.debug('Protosynthesis atk boost');
                return this.chainModify([5325, 4096]);
            },
            onModifyDefPriority: 6,
            onModifyDef: function (def, pokemon) {
                if (this.effectState.bestStat !== 'def' || pokemon.ignoringAbility())
                    return;
                this.debug('Protosynthesis def boost');
                return this.chainModify([5325, 4096]);
            },
            onModifySpAPriority: 5,
            onModifySpA: function (spa, pokemon) {
                if (this.effectState.bestStat !== 'spa' || pokemon.ignoringAbility())
                    return;
                this.debug('Protosynthesis spa boost');
                return this.chainModify([5325, 4096]);
            },
            onModifySpDPriority: 6,
            onModifySpD: function (spd, pokemon) {
                if (this.effectState.bestStat !== 'spd' || pokemon.ignoringAbility())
                    return;
                this.debug('Protosynthesis spd boost');
                return this.chainModify([5325, 4096]);
            },
            onModifySpe: function (spe, pokemon) {
                if (this.effectState.bestStat !== 'spe' || pokemon.ignoringAbility())
                    return;
                this.debug('Protosynthesis spe boost');
                return this.chainModify(1.5);
            },
            onEnd: function (pokemon) {
                this.add('-end', pokemon, 'Protosynthesis');
            },
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, notransform: 1 },
        name: "Protosynthesis",
        rating: 3,
        num: 281,
    },
    psychicsurge: {
        onStart: function (source) {
            this.field.setTerrain('psychicterrain');
        },
        flags: {},
        name: "Psychic Surge",
        rating: 4,
        num: 227,
    },
    punkrock: {
        onBasePowerPriority: 7,
        onBasePower: function (basePower, attacker, defender, move) {
            if (move.flags['sound']) {
                this.debug('Punk Rock boost');
                return this.chainModify([5325, 4096]);
            }
        },
        onSourceModifyDamage: function (damage, source, target, move) {
            if (move.flags['sound']) {
                this.debug('Punk Rock weaken');
                return this.chainModify(0.5);
            }
        },
        flags: { breakable: 1 },
        name: "Punk Rock",
        rating: 3.5,
        num: 244,
    },
    purepower: {
        onModifyAtkPriority: 5,
        onModifyAtk: function (atk) {
            return this.chainModify(2);
        },
        flags: {},
        name: "Pure Power",
        rating: 5,
        num: 74,
    },
    purifyingsalt: {
        onSetStatus: function (status, target, source, effect) {
            if (effect === null || effect === void 0 ? void 0 : effect.status) {
                this.add('-immune', target, '[from] ability: Purifying Salt');
            }
            return false;
        },
        onTryAddVolatile: function (status, target) {
            if (status.id === 'yawn') {
                this.add('-immune', target, '[from] ability: Purifying Salt');
                return null;
            }
        },
        onSourceModifyAtkPriority: 6,
        onSourceModifyAtk: function (atk, attacker, defender, move) {
            if (move.type === 'Ghost') {
                this.debug('Purifying Salt weaken');
                return this.chainModify(0.5);
            }
        },
        onSourceModifySpAPriority: 5,
        onSourceModifySpA: function (spa, attacker, defender, move) {
            if (move.type === 'Ghost') {
                this.debug('Purifying Salt weaken');
                return this.chainModify(0.5);
            }
        },
        flags: { breakable: 1 },
        name: "Purifying Salt",
        rating: 4,
        num: 272,
    },
    quarkdrive: {
        onSwitchInPriority: -2,
        onStart: function (pokemon) {
            this.singleEvent('TerrainChange', this.effect, this.effectState, pokemon);
        },
        onTerrainChange: function (pokemon) {
            var _a;
            if (this.field.isTerrain('electricterrain')) {
                pokemon.addVolatile('quarkdrive');
            }
            else if (!((_a = pokemon.volatiles['quarkdrive']) === null || _a === void 0 ? void 0 : _a.fromBooster)) {
                pokemon.removeVolatile('quarkdrive');
            }
        },
        onEnd: function (pokemon) {
            delete pokemon.volatiles['quarkdrive'];
            this.add('-end', pokemon, 'Quark Drive', '[silent]');
        },
        condition: {
            noCopy: true,
            onStart: function (pokemon, source, effect) {
                if ((effect === null || effect === void 0 ? void 0 : effect.name) === 'Booster Energy') {
                    this.effectState.fromBooster = true;
                    this.add('-activate', pokemon, 'ability: Quark Drive', '[fromitem]');
                }
                else {
                    this.add('-activate', pokemon, 'ability: Quark Drive');
                }
                this.effectState.bestStat = pokemon.getBestStat(false, true);
                this.add('-start', pokemon, 'quarkdrive' + this.effectState.bestStat);
            },
            onModifyAtkPriority: 5,
            onModifyAtk: function (atk, pokemon) {
                if (this.effectState.bestStat !== 'atk' || pokemon.ignoringAbility())
                    return;
                this.debug('Quark Drive atk boost');
                return this.chainModify([5325, 4096]);
            },
            onModifyDefPriority: 6,
            onModifyDef: function (def, pokemon) {
                if (this.effectState.bestStat !== 'def' || pokemon.ignoringAbility())
                    return;
                this.debug('Quark Drive def boost');
                return this.chainModify([5325, 4096]);
            },
            onModifySpAPriority: 5,
            onModifySpA: function (spa, pokemon) {
                if (this.effectState.bestStat !== 'spa' || pokemon.ignoringAbility())
                    return;
                this.debug('Quark Drive spa boost');
                return this.chainModify([5325, 4096]);
            },
            onModifySpDPriority: 6,
            onModifySpD: function (spd, pokemon) {
                if (this.effectState.bestStat !== 'spd' || pokemon.ignoringAbility())
                    return;
                this.debug('Quark Drive spd boost');
                return this.chainModify([5325, 4096]);
            },
            onModifySpe: function (spe, pokemon) {
                if (this.effectState.bestStat !== 'spe' || pokemon.ignoringAbility())
                    return;
                this.debug('Quark Drive spe boost');
                return this.chainModify(1.5);
            },
            onEnd: function (pokemon) {
                this.add('-end', pokemon, 'Quark Drive');
            },
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, notransform: 1 },
        name: "Quark Drive",
        rating: 3,
        num: 282,
    },
    queenlymajesty: {
        onFoeTryMove: function (target, source, move) {
            var targetAllExceptions = ['perishsong', 'flowershield', 'rototiller'];
            if (move.target === 'foeSide' || (move.target === 'all' && !targetAllExceptions.includes(move.id))) {
                return;
            }
            var dazzlingHolder = this.effectState.target;
            if ((source.isAlly(dazzlingHolder) || move.target === 'all') && move.priority > 0.1) {
                this.attrLastMove('[still]');
                this.add('cant', dazzlingHolder, 'ability: Queenly Majesty', move, "[of] ".concat(target));
                return false;
            }
        },
        flags: { breakable: 1 },
        name: "Queenly Majesty",
        rating: 2.5,
        num: 214,
    },
    quickdraw: {
        onFractionalPriorityPriority: -1,
        onFractionalPriority: function (priority, pokemon, target, move) {
            if (move.category !== "Status" && this.randomChance(3, 10)) {
                this.add('-activate', pokemon, 'ability: Quick Draw');
                return 0.1;
            }
        },
        flags: {},
        name: "Quick Draw",
        rating: 2.5,
        num: 259,
    },
    quickfeet: {
        onModifySpe: function (spe, pokemon) {
            if (pokemon.status) {
                return this.chainModify(1.5);
            }
        },
        flags: {},
        name: "Quick Feet",
        rating: 2.5,
        num: 95,
    },
    raindish: {
        onWeather: function (target, source, effect) {
            if (target.hasItem('utilityumbrella'))
                return;
            if (effect.id === 'raindance' || effect.id === 'primordialsea') {
                this.heal(target.baseMaxhp / 16);
            }
        },
        flags: {},
        name: "Rain Dish",
        rating: 1.5,
        num: 44,
    },
    rattled: {
        onDamagingHit: function (damage, target, source, move) {
            if (['Dark', 'Bug', 'Ghost'].includes(move.type)) {
                this.boost({ spe: 1 });
            }
        },
        onAfterBoost: function (boost, target, source, effect) {
            if ((effect === null || effect === void 0 ? void 0 : effect.name) === 'Intimidate' && boost.atk) {
                this.boost({ spe: 1 });
            }
        },
        flags: {},
        name: "Rattled",
        rating: 1,
        num: 155,
    },
    receiver: {
        onAllyFaint: function (target) {
            if (!this.effectState.target.hp)
                return;
            var ability = target.getAbility();
            if (ability.flags['noreceiver'] || ability.id === 'noability')
                return;
            if (this.effectState.target.setAbility(ability)) {
                this.add('-ability', this.effectState.target, ability, '[from] ability: Receiver', "[of] ".concat(target));
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1 },
        name: "Receiver",
        rating: 0,
        num: 222,
    },
    reckless: {
        onBasePowerPriority: 23,
        onBasePower: function (basePower, attacker, defender, move) {
            if (move.recoil || move.hasCrashDamage) {
                this.debug('Reckless boost');
                return this.chainModify([4915, 4096]);
            }
        },
        flags: {},
        name: "Reckless",
        rating: 3,
        num: 120,
    },
    refrigerate: {
        onModifyTypePriority: -1,
        onModifyType: function (move, pokemon) {
            var noModifyType = [
                'judgment', 'multiattack', 'naturalgift', 'revelationdance', 'technoblast', 'terrainpulse', 'weatherball',
            ];
            if (move.type === 'Normal' && !noModifyType.includes(move.id) &&
                !(move.isZ && move.category !== 'Status') && !(move.name === 'Tera Blast' && pokemon.terastallized)) {
                move.type = 'Ice';
                move.typeChangerBoosted = this.effect;
            }
        },
        onBasePowerPriority: 23,
        onBasePower: function (basePower, pokemon, target, move) {
            if (move.typeChangerBoosted === this.effect)
                return this.chainModify([4915, 4096]);
        },
        flags: {},
        name: "Refrigerate",
        rating: 4,
        num: 174,
    },
    regenerator: {
        onSwitchOut: function (pokemon) {
            pokemon.heal(pokemon.baseMaxhp / 3);
        },
        flags: {},
        name: "Regenerator",
        rating: 4.5,
        num: 144,
    },
    ripen: {
        onTryHeal: function (damage, target, source, effect) {
            if (!effect)
                return;
            if (effect.name === 'Berry Juice' || effect.name === 'Leftovers') {
                this.add('-activate', target, 'ability: Ripen');
            }
            if (effect.isBerry)
                return this.chainModify(2);
        },
        onChangeBoost: function (boost, target, source, effect) {
            if (effect && effect.isBerry) {
                var b = void 0;
                for (b in boost) {
                    boost[b] *= 2;
                }
            }
        },
        onSourceModifyDamagePriority: -1,
        onSourceModifyDamage: function (damage, source, target, move) {
            if (target.abilityState.berryWeaken) {
                target.abilityState.berryWeaken = false;
                return this.chainModify(0.5);
            }
        },
        onTryEatItemPriority: -1,
        onTryEatItem: function (item, pokemon) {
            this.add('-activate', pokemon, 'ability: Ripen');
        },
        onEatItem: function (item, pokemon) {
            var weakenBerries = [
                'Babiri Berry', 'Charti Berry', 'Chilan Berry', 'Chople Berry', 'Coba Berry', 'Colbur Berry', 'Haban Berry', 'Kasib Berry', 'Kebia Berry', 'Occa Berry', 'Passho Berry', 'Payapa Berry', 'Rindo Berry', 'Roseli Berry', 'Shuca Berry', 'Tanga Berry', 'Wacan Berry', 'Yache Berry',
            ];
            // Record if the pokemon ate a berry to resist the attack
            pokemon.abilityState.berryWeaken = weakenBerries.includes(item.name);
        },
        flags: {},
        name: "Ripen",
        rating: 2,
        num: 247,
    },
    rivalry: {
        onBasePowerPriority: 24,
        onBasePower: function (basePower, attacker, defender, move) {
            if (attacker.gender && defender.gender) {
                if (attacker.gender === defender.gender) {
                    this.debug('Rivalry boost');
                    return this.chainModify(1.25);
                }
                else {
                    this.debug('Rivalry weaken');
                    return this.chainModify(0.75);
                }
            }
        },
        flags: {},
        name: "Rivalry",
        rating: 0,
        num: 79,
    },
    rkssystem: {
        // RKS System's type-changing itself is implemented in statuses.js
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, cantsuppress: 1 },
        name: "RKS System",
        rating: 4,
        num: 225,
    },
    rockhead: {
        onDamage: function (damage, target, source, effect) {
            if (effect.id === 'recoil') {
                if (!this.activeMove)
                    throw new Error("Battle.activeMove is null");
                if (this.activeMove.id !== 'struggle')
                    return null;
            }
        },
        flags: {},
        name: "Rock Head",
        rating: 3,
        num: 69,
    },
    rockypayload: {
        onModifyAtkPriority: 5,
        onModifyAtk: function (atk, attacker, defender, move) {
            if (move.type === 'Rock') {
                this.debug('Rocky Payload boost');
                return this.chainModify(1.5);
            }
        },
        onModifySpAPriority: 5,
        onModifySpA: function (atk, attacker, defender, move) {
            if (move.type === 'Rock') {
                this.debug('Rocky Payload boost');
                return this.chainModify(1.5);
            }
        },
        flags: {},
        name: "Rocky Payload",
        rating: 3.5,
        num: 276,
    },
    roughskin: {
        onDamagingHitOrder: 1,
        onDamagingHit: function (damage, target, source, move) {
            if (this.checkMoveMakesContact(move, source, target, true)) {
                this.damage(source.baseMaxhp / 8, source, target);
            }
        },
        flags: {},
        name: "Rough Skin",
        rating: 2.5,
        num: 24,
    },
    runaway: {
        flags: {},
        name: "Run Away",
        rating: 0,
        num: 50,
    },
    sandforce: {
        onBasePowerPriority: 21,
        onBasePower: function (basePower, attacker, defender, move) {
            if (this.field.isWeather('sandstorm')) {
                if (move.type === 'Rock' || move.type === 'Ground' || move.type === 'Steel') {
                    this.debug('Sand Force boost');
                    return this.chainModify([5325, 4096]);
                }
            }
        },
        onImmunity: function (type, pokemon) {
            if (type === 'sandstorm')
                return false;
        },
        flags: {},
        name: "Sand Force",
        rating: 2,
        num: 159,
    },
    sandrush: {
        onModifySpe: function (spe, pokemon) {
            if (this.field.isWeather('sandstorm')) {
                return this.chainModify(2);
            }
        },
        onImmunity: function (type, pokemon) {
            if (type === 'sandstorm')
                return false;
        },
        flags: {},
        name: "Sand Rush",
        rating: 3,
        num: 146,
    },
    sandspit: {
        onDamagingHit: function (damage, target, source, move) {
            this.field.setWeather('sandstorm');
        },
        flags: {},
        name: "Sand Spit",
        rating: 1,
        num: 245,
    },
    sandstream: {
        onStart: function (source) {
            this.field.setWeather('sandstorm');
        },
        flags: {},
        name: "Sand Stream",
        rating: 4,
        num: 45,
    },
    sandveil: {
        onImmunity: function (type, pokemon) {
            if (type === 'sandstorm')
                return false;
        },
        onModifyAccuracyPriority: -1,
        onModifyAccuracy: function (accuracy) {
            if (typeof accuracy !== 'number')
                return;
            if (this.field.isWeather('sandstorm')) {
                this.debug('Sand Veil - decreasing accuracy');
                return this.chainModify([3277, 4096]);
            }
        },
        flags: { breakable: 1 },
        name: "Sand Veil",
        rating: 1.5,
        num: 8,
    },
    sapsipper: {
        onTryHitPriority: 1,
        onTryHit: function (target, source, move) {
            if (target !== source && move.type === 'Grass') {
                if (!this.boost({ atk: 1 })) {
                    this.add('-immune', target, '[from] ability: Sap Sipper');
                }
                return null;
            }
        },
        onAllyTryHitSide: function (target, source, move) {
            if (source === this.effectState.target || !target.isAlly(source))
                return;
            if (move.type === 'Grass') {
                this.boost({ atk: 1 }, this.effectState.target);
            }
        },
        flags: { breakable: 1 },
        name: "Sap Sipper",
        rating: 3,
        num: 157,
    },
    schooling: {
        onSwitchInPriority: -1,
        onStart: function (pokemon) {
            if (pokemon.baseSpecies.baseSpecies !== 'Wishiwashi' || pokemon.level < 20 || pokemon.transformed)
                return;
            if (pokemon.hp > pokemon.maxhp / 4) {
                if (pokemon.species.id === 'wishiwashi') {
                    pokemon.formeChange('Wishiwashi-School');
                }
            }
            else {
                if (pokemon.species.id === 'wishiwashischool') {
                    pokemon.formeChange('Wishiwashi');
                }
            }
        },
        onResidualOrder: 29,
        onResidual: function (pokemon) {
            if (pokemon.baseSpecies.baseSpecies !== 'Wishiwashi' || pokemon.level < 20 ||
                pokemon.transformed || !pokemon.hp)
                return;
            if (pokemon.hp > pokemon.maxhp / 4) {
                if (pokemon.species.id === 'wishiwashi') {
                    pokemon.formeChange('Wishiwashi-School');
                }
            }
            else {
                if (pokemon.species.id === 'wishiwashischool') {
                    pokemon.formeChange('Wishiwashi');
                }
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, cantsuppress: 1 },
        name: "Schooling",
        rating: 3,
        num: 208,
    },
    scrappy: {
        onModifyMovePriority: -5,
        onModifyMove: function (move) {
            if (!move.ignoreImmunity)
                move.ignoreImmunity = {};
            if (move.ignoreImmunity !== true) {
                move.ignoreImmunity['Fighting'] = true;
                move.ignoreImmunity['Normal'] = true;
            }
        },
        onTryBoost: function (boost, target, source, effect) {
            if (effect.name === 'Intimidate' && boost.atk) {
                delete boost.atk;
                this.add('-fail', target, 'unboost', 'Attack', '[from] ability: Scrappy', "[of] ".concat(target));
            }
        },
        flags: {},
        name: "Scrappy",
        rating: 3,
        num: 113,
    },
    screencleaner: {
        onStart: function (pokemon) {
            var activated = false;
            for (var _i = 0, _a = ['reflect', 'lightscreen', 'auroraveil']; _i < _a.length; _i++) {
                var sideCondition = _a[_i];
                for (var _b = 0, _c = __spreadArray([pokemon.side], pokemon.side.foeSidesWithConditions(), true); _b < _c.length; _b++) {
                    var side = _c[_b];
                    if (side.getSideCondition(sideCondition)) {
                        if (!activated) {
                            this.add('-activate', pokemon, 'ability: Screen Cleaner');
                            activated = true;
                        }
                        side.removeSideCondition(sideCondition);
                    }
                }
            }
        },
        flags: {},
        name: "Screen Cleaner",
        rating: 2,
        num: 251,
    },
    seedsower: {
        onDamagingHit: function (damage, target, source, move) {
            this.field.setTerrain('grassyterrain');
        },
        flags: {},
        name: "Seed Sower",
        rating: 2.5,
        num: 269,
    },
    serenegrace: {
        onModifyMovePriority: -2,
        onModifyMove: function (move) {
            var _a;
            if (move.secondaries) {
                this.debug('doubling secondary chance');
                for (var _i = 0, _b = move.secondaries; _i < _b.length; _i++) {
                    var secondary = _b[_i];
                    if (secondary.chance)
                        secondary.chance *= 2;
                }
            }
            if ((_a = move.self) === null || _a === void 0 ? void 0 : _a.chance)
                move.self.chance *= 2;
        },
        flags: {},
        name: "Serene Grace",
        rating: 3.5,
        num: 32,
    },
    shadowshield: {
        onSourceModifyDamage: function (damage, source, target, move) {
            if (target.hp >= target.maxhp) {
                this.debug('Shadow Shield weaken');
                return this.chainModify(0.5);
            }
        },
        flags: {},
        name: "Shadow Shield",
        rating: 3.5,
        num: 231,
    },
    shadowtag: {
        onFoeTrapPokemon: function (pokemon) {
            if (!pokemon.hasAbility('shadowtag') && pokemon.isAdjacent(this.effectState.target)) {
                pokemon.tryTrap(true);
            }
        },
        onFoeMaybeTrapPokemon: function (pokemon, source) {
            if (!source)
                source = this.effectState.target;
            if (!source || !pokemon.isAdjacent(source))
                return;
            if (!pokemon.hasAbility('shadowtag')) {
                pokemon.maybeTrapped = true;
            }
        },
        flags: {},
        name: "Shadow Tag",
        rating: 5,
        num: 23,
    },
    sharpness: {
        onBasePowerPriority: 19,
        onBasePower: function (basePower, attacker, defender, move) {
            if (move.flags['slicing']) {
                this.debug('Sharpness boost');
                return this.chainModify(1.5);
            }
        },
        flags: {},
        name: "Sharpness",
        rating: 3.5,
        num: 292,
    },
    shedskin: {
        onResidualOrder: 5,
        onResidualSubOrder: 3,
        onResidual: function (pokemon) {
            if (pokemon.hp && pokemon.status && this.randomChance(33, 100)) {
                this.debug('shed skin');
                this.add('-activate', pokemon, 'ability: Shed Skin');
                pokemon.cureStatus();
            }
        },
        flags: {},
        name: "Shed Skin",
        rating: 3,
        num: 61,
    },
    sheerforce: {
        onModifyMove: function (move, pokemon) {
            if (move.secondaries) {
                delete move.secondaries;
                // Technically not a secondary effect, but it is negated
                delete move.self;
                if (move.id === 'clangoroussoulblaze')
                    delete move.selfBoost;
                // Actual negation of `AfterMoveSecondary` effects implemented in scripts.js
                move.hasSheerForce = true;
            }
        },
        onBasePowerPriority: 21,
        onBasePower: function (basePower, pokemon, target, move) {
            if (move.hasSheerForce)
                return this.chainModify([5325, 4096]);
        },
        flags: {},
        name: "Sheer Force",
        rating: 3.5,
        num: 125,
    },
    shellarmor: {
        onCriticalHit: false,
        flags: { breakable: 1 },
        name: "Shell Armor",
        rating: 1,
        num: 75,
    },
    shielddust: {
        onModifySecondaries: function (secondaries) {
            this.debug('Shield Dust prevent secondary');
            return secondaries.filter(function (effect) { return !!(effect.self || effect.dustproof); });
        },
        flags: { breakable: 1 },
        name: "Shield Dust",
        rating: 2,
        num: 19,
    },
    shieldsdown: {
        onSwitchInPriority: -1,
        onStart: function (pokemon) {
            if (pokemon.baseSpecies.baseSpecies !== 'Minior' || pokemon.transformed)
                return;
            if (pokemon.hp > pokemon.maxhp / 2) {
                if (pokemon.species.forme !== 'Meteor') {
                    pokemon.formeChange('Minior-Meteor');
                }
            }
            else {
                if (pokemon.species.forme === 'Meteor') {
                    pokemon.formeChange(pokemon.set.species);
                }
            }
        },
        onResidualOrder: 29,
        onResidual: function (pokemon) {
            if (pokemon.baseSpecies.baseSpecies !== 'Minior' || pokemon.transformed || !pokemon.hp)
                return;
            if (pokemon.hp > pokemon.maxhp / 2) {
                if (pokemon.species.forme !== 'Meteor') {
                    pokemon.formeChange('Minior-Meteor');
                }
            }
            else {
                if (pokemon.species.forme === 'Meteor') {
                    pokemon.formeChange(pokemon.set.species);
                }
            }
        },
        onSetStatus: function (status, target, source, effect) {
            if (target.species.id !== 'miniormeteor' || target.transformed)
                return;
            if (effect === null || effect === void 0 ? void 0 : effect.status) {
                this.add('-immune', target, '[from] ability: Shields Down');
            }
            return false;
        },
        onTryAddVolatile: function (status, target) {
            if (target.species.id !== 'miniormeteor' || target.transformed)
                return;
            if (status.id !== 'yawn')
                return;
            this.add('-immune', target, '[from] ability: Shields Down');
            return null;
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, cantsuppress: 1 },
        name: "Shields Down",
        rating: 3,
        num: 197,
    },
    simple: {
        onChangeBoost: function (boost, target, source, effect) {
            if (effect && effect.id === 'zpower')
                return;
            var i;
            for (i in boost) {
                boost[i] *= 2;
            }
        },
        flags: { breakable: 1 },
        name: "Simple",
        rating: 4,
        num: 86,
    },
    skilllink: {
        onModifyMove: function (move) {
            if (move.multihit && Array.isArray(move.multihit) && move.multihit.length) {
                move.multihit = move.multihit[1];
            }
            if (move.multiaccuracy) {
                delete move.multiaccuracy;
            }
        },
        flags: {},
        name: "Skill Link",
        rating: 3,
        num: 92,
    },
    slowstart: {
        onStart: function (pokemon) {
            pokemon.addVolatile('slowstart');
        },
        onEnd: function (pokemon) {
            delete pokemon.volatiles['slowstart'];
            this.add('-end', pokemon, 'Slow Start', '[silent]');
        },
        condition: {
            duration: 5,
            onResidualOrder: 28,
            onResidualSubOrder: 2,
            onStart: function (target) {
                this.add('-start', target, 'ability: Slow Start');
            },
            onResidual: function (pokemon) {
                if (!pokemon.activeTurns) {
                    this.effectState.duration += 1;
                }
            },
            onModifyAtkPriority: 5,
            onModifyAtk: function (atk, pokemon) {
                return this.chainModify(0.5);
            },
            onModifySpe: function (spe, pokemon) {
                return this.chainModify(0.5);
            },
            onEnd: function (target) {
                this.add('-end', target, 'Slow Start');
            },
        },
        flags: {},
        name: "Slow Start",
        rating: -1,
        num: 112,
    },
    slushrush: {
        onModifySpe: function (spe, pokemon) {
            if (this.field.isWeather(['hail', 'snowscape'])) {
                return this.chainModify(2);
            }
        },
        flags: {},
        name: "Slush Rush",
        rating: 3,
        num: 202,
    },
    sniper: {
        onModifyDamage: function (damage, source, target, move) {
            if (target.getMoveHitData(move).crit) {
                this.debug('Sniper boost');
                return this.chainModify(1.5);
            }
        },
        flags: {},
        name: "Sniper",
        rating: 2,
        num: 97,
    },
    snowcloak: {
        onImmunity: function (type, pokemon) {
            if (type === 'hail')
                return false;
        },
        onModifyAccuracyPriority: -1,
        onModifyAccuracy: function (accuracy) {
            if (typeof accuracy !== 'number')
                return;
            if (this.field.isWeather(['hail', 'snowscape'])) {
                this.debug('Snow Cloak - decreasing accuracy');
                return this.chainModify([3277, 4096]);
            }
        },
        flags: { breakable: 1 },
        name: "Snow Cloak",
        rating: 1.5,
        num: 81,
    },
    snowwarning: {
        onStart: function (source) {
            this.field.setWeather('snowscape');
        },
        flags: {},
        name: "Snow Warning",
        rating: 4,
        num: 117,
    },
    solarpower: {
        onModifySpAPriority: 5,
        onModifySpA: function (spa, pokemon) {
            if (['sunnyday', 'desolateland'].includes(pokemon.effectiveWeather())) {
                return this.chainModify(1.5);
            }
        },
        onWeather: function (target, source, effect) {
            if (target.hasItem('utilityumbrella'))
                return;
            if (effect.id === 'sunnyday' || effect.id === 'desolateland') {
                this.damage(target.baseMaxhp / 8, target, target);
            }
        },
        flags: {},
        name: "Solar Power",
        rating: 2,
        num: 94,
    },
    solidrock: {
        onSourceModifyDamage: function (damage, source, target, move) {
            if (target.getMoveHitData(move).typeMod > 0) {
                this.debug('Solid Rock neutralize');
                return this.chainModify(0.75);
            }
        },
        flags: { breakable: 1 },
        name: "Solid Rock",
        rating: 3,
        num: 116,
    },
    soulheart: {
        onAnyFaintPriority: 1,
        onAnyFaint: function () {
            this.boost({ spa: 1 }, this.effectState.target);
        },
        flags: {},
        name: "Soul-Heart",
        rating: 3.5,
        num: 220,
    },
    soundproof: {
        onTryHit: function (target, source, move) {
            if (target !== source && move.flags['sound']) {
                this.add('-immune', target, '[from] ability: Soundproof');
                return null;
            }
        },
        onAllyTryHitSide: function (target, source, move) {
            if (move.flags['sound']) {
                this.add('-immune', this.effectState.target, '[from] ability: Soundproof');
            }
        },
        flags: { breakable: 1 },
        name: "Soundproof",
        rating: 2,
        num: 43,
    },
    speedboost: {
        onResidualOrder: 28,
        onResidualSubOrder: 2,
        onResidual: function (pokemon) {
            if (pokemon.activeTurns) {
                this.boost({ spe: 1 });
            }
        },
        flags: {},
        name: "Speed Boost",
        rating: 4.5,
        num: 3,
    },
    stakeout: {
        onModifyAtkPriority: 5,
        onModifyAtk: function (atk, attacker, defender) {
            if (!defender.activeTurns) {
                this.debug('Stakeout boost');
                return this.chainModify(2);
            }
        },
        onModifySpAPriority: 5,
        onModifySpA: function (atk, attacker, defender) {
            if (!defender.activeTurns) {
                this.debug('Stakeout boost');
                return this.chainModify(2);
            }
        },
        flags: {},
        name: "Stakeout",
        rating: 4.5,
        num: 198,
    },
    stall: {
        onFractionalPriority: -0.1,
        flags: {},
        name: "Stall",
        rating: -1,
        num: 100,
    },
    stalwart: {
        onModifyMovePriority: 1,
        onModifyMove: function (move) {
            // most of the implementation is in Battle#getTarget
            move.tracksTarget = move.target !== 'scripted';
        },
        flags: {},
        name: "Stalwart",
        rating: 0,
        num: 242,
    },
    stamina: {
        onDamagingHit: function (damage, target, source, effect) {
            this.boost({ def: 1 });
        },
        flags: {},
        name: "Stamina",
        rating: 4,
        num: 192,
    },
    stancechange: {
        onModifyMovePriority: 1,
        onModifyMove: function (move, attacker, defender) {
            if (attacker.species.baseSpecies !== 'Aegislash' || attacker.transformed)
                return;
            if (move.category === 'Status' && move.id !== 'kingsshield')
                return;
            var targetForme = (move.id === 'kingsshield' ? 'Aegislash' : 'Aegislash-Blade');
            if (attacker.species.name !== targetForme)
                attacker.formeChange(targetForme);
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, cantsuppress: 1 },
        name: "Stance Change",
        rating: 4,
        num: 176,
    },
    static: {
        onDamagingHit: function (damage, target, source, move) {
            if (this.checkMoveMakesContact(move, source, target)) {
                if (this.randomChance(3, 10)) {
                    source.trySetStatus('par', target);
                }
            }
        },
        flags: {},
        name: "Static",
        rating: 2,
        num: 9,
    },
    steadfast: {
        onFlinch: function (pokemon) {
            this.boost({ spe: 1 });
        },
        flags: {},
        name: "Steadfast",
        rating: 1,
        num: 80,
    },
    steamengine: {
        onDamagingHit: function (damage, target, source, move) {
            if (['Water', 'Fire'].includes(move.type)) {
                this.boost({ spe: 6 });
            }
        },
        flags: {},
        name: "Steam Engine",
        rating: 2,
        num: 243,
    },
    steelworker: {
        onModifyAtkPriority: 5,
        onModifyAtk: function (atk, attacker, defender, move) {
            if (move.type === 'Steel') {
                this.debug('Steelworker boost');
                return this.chainModify(1.5);
            }
        },
        onModifySpAPriority: 5,
        onModifySpA: function (atk, attacker, defender, move) {
            if (move.type === 'Steel') {
                this.debug('Steelworker boost');
                return this.chainModify(1.5);
            }
        },
        flags: {},
        name: "Steelworker",
        rating: 3.5,
        num: 200,
    },
    steelyspirit: {
        onAllyBasePowerPriority: 22,
        onAllyBasePower: function (basePower, attacker, defender, move) {
            if (move.type === 'Steel') {
                this.debug('Steely Spirit boost');
                return this.chainModify(1.5);
            }
        },
        flags: {},
        name: "Steely Spirit",
        rating: 3.5,
        num: 252,
    },
    stench: {
        onModifyMovePriority: -1,
        onModifyMove: function (move) {
            if (move.category !== "Status") {
                this.debug('Adding Stench flinch');
                if (!move.secondaries)
                    move.secondaries = [];
                for (var _i = 0, _a = move.secondaries; _i < _a.length; _i++) {
                    var secondary = _a[_i];
                    if (secondary.volatileStatus === 'flinch')
                        return;
                }
                move.secondaries.push({
                    chance: 10,
                    volatileStatus: 'flinch',
                });
            }
        },
        flags: {},
        name: "Stench",
        rating: 0.5,
        num: 1,
    },
    stickyhold: {
        onTakeItem: function (item, pokemon, source) {
            if (!this.activeMove)
                throw new Error("Battle.activeMove is null");
            if (!pokemon.hp || pokemon.item === 'stickybarb')
                return;
            if ((source && source !== pokemon) || this.activeMove.id === 'knockoff') {
                this.add('-activate', pokemon, 'ability: Sticky Hold');
                return false;
            }
        },
        flags: { breakable: 1 },
        name: "Sticky Hold",
        rating: 1.5,
        num: 60,
    },
    stormdrain: {
        onTryHit: function (target, source, move) {
            if (target !== source && move.type === 'Water') {
                if (!this.boost({ spa: 1 })) {
                    this.add('-immune', target, '[from] ability: Storm Drain');
                }
                return null;
            }
        },
        onAnyRedirectTarget: function (target, source, source2, move) {
            if (move.type !== 'Water' || move.flags['pledgecombo'])
                return;
            var redirectTarget = ['randomNormal', 'adjacentFoe'].includes(move.target) ? 'normal' : move.target;
            if (this.validTarget(this.effectState.target, source, redirectTarget)) {
                if (move.smartTarget)
                    move.smartTarget = false;
                if (this.effectState.target !== target) {
                    this.add('-activate', this.effectState.target, 'ability: Storm Drain');
                }
                return this.effectState.target;
            }
        },
        flags: { breakable: 1 },
        name: "Storm Drain",
        rating: 3,
        num: 114,
    },
    strongjaw: {
        onBasePowerPriority: 19,
        onBasePower: function (basePower, attacker, defender, move) {
            if (move.flags['bite']) {
                return this.chainModify(1.5);
            }
        },
        flags: {},
        name: "Strong Jaw",
        rating: 3.5,
        num: 173,
    },
    sturdy: {
        onTryHit: function (pokemon, target, move) {
            if (move.ohko) {
                this.add('-immune', pokemon, '[from] ability: Sturdy');
                return null;
            }
        },
        onDamagePriority: -30,
        onDamage: function (damage, target, source, effect) {
            if (target.hp === target.maxhp && damage >= target.hp && effect && effect.effectType === 'Move') {
                this.add('-ability', target, 'Sturdy');
                return target.hp - 1;
            }
        },
        flags: { breakable: 1 },
        name: "Sturdy",
        rating: 3,
        num: 5,
    },
    suctioncups: {
        onDragOutPriority: 1,
        onDragOut: function (pokemon) {
            this.add('-activate', pokemon, 'ability: Suction Cups');
            return null;
        },
        flags: { breakable: 1 },
        name: "Suction Cups",
        rating: 1,
        num: 21,
    },
    superluck: {
        onModifyCritRatio: function (critRatio) {
            return critRatio + 1;
        },
        flags: {},
        name: "Super Luck",
        rating: 1.5,
        num: 105,
    },
    supersweetsyrup: {
        onStart: function (pokemon) {
            if (pokemon.syrupTriggered)
                return;
            pokemon.syrupTriggered = true;
            this.add('-ability', pokemon, 'Supersweet Syrup');
            for (var _i = 0, _a = pokemon.adjacentFoes(); _i < _a.length; _i++) {
                var target = _a[_i];
                if (target.volatiles['substitute']) {
                    this.add('-immune', target);
                }
                else {
                    this.boost({ evasion: -1 }, target, pokemon, null, true);
                }
            }
        },
        flags: {},
        name: "Supersweet Syrup",
        rating: 1.5,
        num: 306,
    },
    supremeoverlord: {
        onStart: function (pokemon) {
            if (pokemon.side.totalFainted) {
                this.add('-activate', pokemon, 'ability: Supreme Overlord');
                var fallen = Math.min(pokemon.side.totalFainted, 5);
                this.add('-start', pokemon, "fallen".concat(fallen), '[silent]');
                this.effectState.fallen = fallen;
            }
        },
        onEnd: function (pokemon) {
            this.add('-end', pokemon, "fallen".concat(this.effectState.fallen), '[silent]');
        },
        onBasePowerPriority: 21,
        onBasePower: function (basePower, attacker, defender, move) {
            if (this.effectState.fallen) {
                var powMod = [4096, 4506, 4915, 5325, 5734, 6144];
                this.debug("Supreme Overlord boost: ".concat(powMod[this.effectState.fallen], "/4096"));
                return this.chainModify([powMod[this.effectState.fallen], 4096]);
            }
        },
        flags: {},
        name: "Supreme Overlord",
        rating: 4,
        num: 293,
    },
    surgesurfer: {
        onModifySpe: function (spe) {
            if (this.field.isTerrain('electricterrain')) {
                return this.chainModify(2);
            }
        },
        flags: {},
        name: "Surge Surfer",
        rating: 3,
        num: 207,
    },
    swarm: {
        onModifyAtkPriority: 5,
        onModifyAtk: function (atk, attacker, defender, move) {
            if (move.type === 'Bug' && attacker.hp <= attacker.maxhp / 3) {
                this.debug('Swarm boost');
                return this.chainModify(1.5);
            }
        },
        onModifySpAPriority: 5,
        onModifySpA: function (atk, attacker, defender, move) {
            if (move.type === 'Bug' && attacker.hp <= attacker.maxhp / 3) {
                this.debug('Swarm boost');
                return this.chainModify(1.5);
            }
        },
        flags: {},
        name: "Swarm",
        rating: 2,
        num: 68,
    },
    sweetveil: {
        onAllySetStatus: function (status, target, source, effect) {
            if (status.id === 'slp') {
                this.debug('Sweet Veil interrupts sleep');
                var effectHolder = this.effectState.target;
                this.add('-block', target, 'ability: Sweet Veil', "[of] ".concat(effectHolder));
                return null;
            }
        },
        onAllyTryAddVolatile: function (status, target) {
            if (status.id === 'yawn') {
                this.debug('Sweet Veil blocking yawn');
                var effectHolder = this.effectState.target;
                this.add('-block', target, 'ability: Sweet Veil', "[of] ".concat(effectHolder));
                return null;
            }
        },
        flags: { breakable: 1 },
        name: "Sweet Veil",
        rating: 2,
        num: 175,
    },
    swiftswim: {
        onModifySpe: function (spe, pokemon) {
            if (['raindance', 'primordialsea'].includes(pokemon.effectiveWeather())) {
                return this.chainModify(2);
            }
        },
        flags: {},
        name: "Swift Swim",
        rating: 3,
        num: 33,
    },
    symbiosis: {
        onAllyAfterUseItem: function (item, pokemon) {
            if (pokemon.switchFlag)
                return;
            var source = this.effectState.target;
            var myItem = source.takeItem();
            if (!myItem)
                return;
            if (!this.singleEvent('TakeItem', myItem, source.itemState, pokemon, source, this.effect, myItem) ||
                !pokemon.setItem(myItem)) {
                source.item = myItem.id;
                return;
            }
            this.add('-activate', source, 'ability: Symbiosis', myItem, "[of] ".concat(pokemon));
        },
        flags: {},
        name: "Symbiosis",
        rating: 0,
        num: 180,
    },
    synchronize: {
        onAfterSetStatus: function (status, target, source, effect) {
            if (!source || source === target)
                return;
            if (effect && effect.id === 'toxicspikes')
                return;
            if (status.id === 'slp' || status.id === 'frz')
                return;
            this.add('-activate', target, 'ability: Synchronize');
            // Hack to make status-prevention abilities think Synchronize is a status move
            // and show messages when activating against it.
            source.trySetStatus(status, target, { status: status.id, id: 'synchronize' });
        },
        flags: {},
        name: "Synchronize",
        rating: 2,
        num: 28,
    },
    swordofruin: {
        onStart: function (pokemon) {
            if (this.suppressingAbility(pokemon))
                return;
            this.add('-ability', pokemon, 'Sword of Ruin');
        },
        onAnyModifyDef: function (def, target, source, move) {
            var _a;
            var abilityHolder = this.effectState.target;
            if (target.hasAbility('Sword of Ruin'))
                return;
            if (!((_a = move.ruinedDef) === null || _a === void 0 ? void 0 : _a.hasAbility('Sword of Ruin')))
                move.ruinedDef = abilityHolder;
            if (move.ruinedDef !== abilityHolder)
                return;
            this.debug('Sword of Ruin Def drop');
            return this.chainModify(0.75);
        },
        flags: {},
        name: "Sword of Ruin",
        rating: 4.5,
        num: 285,
    },
    tabletsofruin: {
        onStart: function (pokemon) {
            if (this.suppressingAbility(pokemon))
                return;
            this.add('-ability', pokemon, 'Tablets of Ruin');
        },
        onAnyModifyAtk: function (atk, source, target, move) {
            var abilityHolder = this.effectState.target;
            if (source.hasAbility('Tablets of Ruin'))
                return;
            if (!move.ruinedAtk)
                move.ruinedAtk = abilityHolder;
            if (move.ruinedAtk !== abilityHolder)
                return;
            this.debug('Tablets of Ruin Atk drop');
            return this.chainModify(0.75);
        },
        flags: {},
        name: "Tablets of Ruin",
        rating: 4.5,
        num: 284,
    },
    tangledfeet: {
        onModifyAccuracyPriority: -1,
        onModifyAccuracy: function (accuracy, target) {
            if (typeof accuracy !== 'number')
                return;
            if (target === null || target === void 0 ? void 0 : target.volatiles['confusion']) {
                this.debug('Tangled Feet - decreasing accuracy');
                return this.chainModify(0.5);
            }
        },
        flags: { breakable: 1 },
        name: "Tangled Feet",
        rating: 1,
        num: 77,
    },
    tanglinghair: {
        onDamagingHit: function (damage, target, source, move) {
            if (this.checkMoveMakesContact(move, source, target, true)) {
                this.add('-ability', target, 'Tangling Hair');
                this.boost({ spe: -1 }, source, target, null, true);
            }
        },
        flags: {},
        name: "Tangling Hair",
        rating: 2,
        num: 221,
    },
    technician: {
        onBasePowerPriority: 30,
        onBasePower: function (basePower, attacker, defender, move) {
            var basePowerAfterMultiplier = this.modify(basePower, this.event.modifier);
            this.debug("Base Power: ".concat(basePowerAfterMultiplier));
            if (basePowerAfterMultiplier <= 60) {
                this.debug('Technician boost');
                return this.chainModify(1.5);
            }
        },
        flags: {},
        name: "Technician",
        rating: 3.5,
        num: 101,
    },
    telepathy: {
        onTryHit: function (target, source, move) {
            if (target !== source && target.isAlly(source) && move.category !== 'Status') {
                this.add('-activate', target, 'ability: Telepathy');
                return null;
            }
        },
        flags: { breakable: 1 },
        name: "Telepathy",
        rating: 0,
        num: 140,
    },
    teraformzero: {
        onAfterTerastallization: function (pokemon) {
            if (pokemon.baseSpecies.name !== 'Terapagos-Stellar')
                return;
            if (this.field.weather || this.field.terrain) {
                this.add('-ability', pokemon, 'Teraform Zero');
                this.field.clearWeather();
                this.field.clearTerrain();
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1 },
        name: "Teraform Zero",
        rating: 3,
        num: 309,
    },
    terashell: {
        // effectiveness implemented in sim/pokemon.ts:Pokemon#runEffectiveness
        // needs two checks to reset between regular moves and future attacks
        onAnyBeforeMove: function () {
            delete this.effectState.resisted;
        },
        onAnyAfterMove: function () {
            delete this.effectState.resisted;
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, breakable: 1 },
        name: "Tera Shell",
        rating: 3.5,
        num: 308,
    },
    terashift: {
        onSwitchInPriority: 2,
        onSwitchIn: function (pokemon) {
            if (pokemon.baseSpecies.baseSpecies !== 'Terapagos')
                return;
            if (pokemon.species.forme !== 'Terastal') {
                this.add('-activate', pokemon, 'ability: Tera Shift');
                pokemon.formeChange('Terapagos-Terastal', this.effect, true);
                pokemon.regressionForme = false;
                pokemon.baseMaxhp = Math.floor(Math.floor(2 * pokemon.species.baseStats['hp'] + pokemon.set.ivs['hp'] + Math.floor(pokemon.set.evs['hp'] / 4) + 100) * pokemon.level / 100 + 10);
                var newMaxHP = pokemon.baseMaxhp;
                pokemon.hp = newMaxHP - (pokemon.maxhp - pokemon.hp);
                pokemon.maxhp = newMaxHP;
                this.add('-heal', pokemon, pokemon.getHealth, '[silent]');
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, cantsuppress: 1, notransform: 1 },
        name: "Tera Shift",
        rating: 3,
        num: 307,
    },
    teravolt: {
        onStart: function (pokemon) {
            this.add('-ability', pokemon, 'Teravolt');
        },
        onModifyMove: function (move) {
            move.ignoreAbility = true;
        },
        flags: {},
        name: "Teravolt",
        rating: 3,
        num: 164,
    },
    thermalexchange: {
        onDamagingHit: function (damage, target, source, move) {
            if (move.type === 'Fire') {
                this.boost({ atk: 1 });
            }
        },
        onUpdate: function (pokemon) {
            if (pokemon.status === 'brn') {
                this.add('-activate', pokemon, 'ability: Thermal Exchange');
                pokemon.cureStatus();
            }
        },
        onSetStatus: function (status, target, source, effect) {
            if (status.id !== 'brn')
                return;
            if (effect === null || effect === void 0 ? void 0 : effect.status) {
                this.add('-immune', target, '[from] ability: Thermal Exchange');
            }
            return false;
        },
        flags: { breakable: 1 },
        name: "Thermal Exchange",
        rating: 2.5,
        num: 270,
    },
    thickfat: {
        onSourceModifyAtkPriority: 6,
        onSourceModifyAtk: function (atk, attacker, defender, move) {
            if (move.type === 'Ice' || move.type === 'Fire') {
                this.debug('Thick Fat weaken');
                return this.chainModify(0.5);
            }
        },
        onSourceModifySpAPriority: 5,
        onSourceModifySpA: function (atk, attacker, defender, move) {
            if (move.type === 'Ice' || move.type === 'Fire') {
                this.debug('Thick Fat weaken');
                return this.chainModify(0.5);
            }
        },
        flags: { breakable: 1 },
        name: "Thick Fat",
        rating: 3.5,
        num: 47,
    },
    tintedlens: {
        onModifyDamage: function (damage, source, target, move) {
            if (target.getMoveHitData(move).typeMod < 0) {
                this.debug('Tinted Lens boost');
                return this.chainModify(2);
            }
        },
        flags: {},
        name: "Tinted Lens",
        rating: 4,
        num: 110,
    },
    torrent: {
        onModifyAtkPriority: 5,
        onModifyAtk: function (atk, attacker, defender, move) {
            if (move.type === 'Water' && attacker.hp <= attacker.maxhp / 3) {
                this.debug('Torrent boost');
                return this.chainModify(1.5);
            }
        },
        onModifySpAPriority: 5,
        onModifySpA: function (atk, attacker, defender, move) {
            if (move.type === 'Water' && attacker.hp <= attacker.maxhp / 3) {
                this.debug('Torrent boost');
                return this.chainModify(1.5);
            }
        },
        flags: {},
        name: "Torrent",
        rating: 2,
        num: 67,
    },
    toughclaws: {
        onBasePowerPriority: 21,
        onBasePower: function (basePower, attacker, defender, move) {
            if (move.flags['contact']) {
                return this.chainModify([5325, 4096]);
            }
        },
        flags: {},
        name: "Tough Claws",
        rating: 3.5,
        num: 181,
    },
    toxicboost: {
        onBasePowerPriority: 19,
        onBasePower: function (basePower, attacker, defender, move) {
            if ((attacker.status === 'psn' || attacker.status === 'tox') && move.category === 'Physical') {
                return this.chainModify(1.5);
            }
        },
        flags: {},
        name: "Toxic Boost",
        rating: 3,
        num: 137,
    },
    toxicchain: {
        onSourceDamagingHit: function (damage, target, source, move) {
            // Despite not being a secondary, Shield Dust / Covert Cloak block Toxic Chain's effect
            if (target.hasAbility('shielddust') || target.hasItem('covertcloak'))
                return;
            if (this.randomChance(3, 10)) {
                target.trySetStatus('tox', source);
            }
        },
        flags: {},
        name: "Toxic Chain",
        rating: 4.5,
        num: 305,
    },
    toxicdebris: {
        onDamagingHit: function (damage, target, source, move) {
            var side = source.isAlly(target) ? source.side.foe : source.side;
            var toxicSpikes = side.sideConditions['toxicspikes'];
            if (move.category === 'Physical' && (!toxicSpikes || toxicSpikes.layers < 2)) {
                this.add('-activate', target, 'ability: Toxic Debris');
                side.addSideCondition('toxicspikes', target);
            }
        },
        flags: {},
        name: "Toxic Debris",
        rating: 3.5,
        num: 295,
    },
    trace: {
        onStart: function (pokemon) {
            this.effectState.seek = true;
            // n.b. only affects Hackmons
            // interaction with No Ability is complicated: https://www.smogon.com/forums/threads/pokemon-sun-moon-battle-mechanics-research.3586701/page-76#post-7790209
            if (pokemon.adjacentFoes().some(function (foeActive) { return foeActive.ability === 'noability'; })) {
                this.effectState.seek = false;
            }
            // interaction with Ability Shield is similar to No Ability
            if (pokemon.hasItem('Ability Shield')) {
                this.add('-block', pokemon, 'item: Ability Shield');
                this.effectState.seek = false;
            }
            if (this.effectState.seek) {
                this.singleEvent('Update', this.effect, this.effectState, pokemon);
            }
        },
        onUpdate: function (pokemon) {
            if (!this.effectState.seek)
                return;
            var possibleTargets = pokemon.adjacentFoes().filter(function (target) { return !target.getAbility().flags['notrace'] && target.ability !== 'noability'; });
            if (!possibleTargets.length)
                return;
            var target = this.sample(possibleTargets);
            var ability = target.getAbility();
            if (pokemon.setAbility(ability)) {
                this.add('-ability', pokemon, ability, '[from] ability: Trace', "[of] ".concat(target));
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1 },
        name: "Trace",
        rating: 2.5,
        num: 36,
    },
    transistor: {
        onModifyAtkPriority: 5,
        onModifyAtk: function (atk, attacker, defender, move) {
            if (move.type === 'Electric') {
                this.debug('Transistor boost');
                return this.chainModify([5325, 4096]);
            }
        },
        onModifySpAPriority: 5,
        onModifySpA: function (atk, attacker, defender, move) {
            if (move.type === 'Electric') {
                this.debug('Transistor boost');
                return this.chainModify([5325, 4096]);
            }
        },
        flags: {},
        name: "Transistor",
        rating: 3.5,
        num: 262,
    },
    triage: {
        onModifyPriority: function (priority, pokemon, target, move) {
            if (move === null || move === void 0 ? void 0 : move.flags['heal'])
                return priority + 3;
        },
        flags: {},
        name: "Triage",
        rating: 3.5,
        num: 205,
    },
    truant: {
        onStart: function (pokemon) {
            pokemon.removeVolatile('truant');
            if (pokemon.activeTurns && (pokemon.moveThisTurnResult !== undefined || !this.queue.willMove(pokemon))) {
                pokemon.addVolatile('truant');
            }
        },
        onBeforeMovePriority: 9,
        onBeforeMove: function (pokemon) {
            if (pokemon.removeVolatile('truant')) {
                this.add('cant', pokemon, 'ability: Truant');
                return false;
            }
            pokemon.addVolatile('truant');
        },
        condition: {},
        flags: {},
        name: "Truant",
        rating: -1,
        num: 54,
    },
    turboblaze: {
        onStart: function (pokemon) {
            this.add('-ability', pokemon, 'Turboblaze');
        },
        onModifyMove: function (move) {
            move.ignoreAbility = true;
        },
        flags: {},
        name: "Turboblaze",
        rating: 3,
        num: 163,
    },
    unaware: {
        onAnyModifyBoost: function (boosts, pokemon) {
            var unawareUser = this.effectState.target;
            if (unawareUser === pokemon)
                return;
            if (unawareUser === this.activePokemon && pokemon === this.activeTarget) {
                boosts['def'] = 0;
                boosts['spd'] = 0;
                boosts['evasion'] = 0;
            }
            if (pokemon === this.activePokemon && unawareUser === this.activeTarget) {
                boosts['atk'] = 0;
                boosts['def'] = 0;
                boosts['spa'] = 0;
                boosts['accuracy'] = 0;
            }
        },
        flags: { breakable: 1 },
        name: "Unaware",
        rating: 4,
        num: 109,
    },
    unburden: {
        onAfterUseItem: function (item, pokemon) {
            if (pokemon !== this.effectState.target)
                return;
            pokemon.addVolatile('unburden');
        },
        onTakeItem: function (item, pokemon) {
            pokemon.addVolatile('unburden');
        },
        onEnd: function (pokemon) {
            pokemon.removeVolatile('unburden');
        },
        condition: {
            onModifySpe: function (spe, pokemon) {
                if (!pokemon.item && !pokemon.ignoringAbility()) {
                    return this.chainModify(2);
                }
            },
        },
        flags: {},
        name: "Unburden",
        rating: 3.5,
        num: 84,
    },
    unnerve: {
        onSwitchInPriority: 1,
        onStart: function (pokemon) {
            if (this.effectState.unnerved)
                return;
            this.add('-ability', pokemon, 'Unnerve');
            this.effectState.unnerved = true;
        },
        onEnd: function () {
            this.effectState.unnerved = false;
        },
        onFoeTryEatItem: function () {
            return !this.effectState.unnerved;
        },
        flags: {},
        name: "Unnerve",
        rating: 1,
        num: 127,
    },
    unseenfist: {
        onModifyMove: function (move) {
            if (move.flags['contact'])
                delete move.flags['protect'];
        },
        flags: {},
        name: "Unseen Fist",
        rating: 2,
        num: 260,
    },
    vesselofruin: {
        onStart: function (pokemon) {
            if (this.suppressingAbility(pokemon))
                return;
            this.add('-ability', pokemon, 'Vessel of Ruin');
        },
        onAnyModifySpA: function (spa, source, target, move) {
            var abilityHolder = this.effectState.target;
            if (source.hasAbility('Vessel of Ruin'))
                return;
            if (!move.ruinedSpA)
                move.ruinedSpA = abilityHolder;
            if (move.ruinedSpA !== abilityHolder)
                return;
            this.debug('Vessel of Ruin SpA drop');
            return this.chainModify(0.75);
        },
        flags: {},
        name: "Vessel of Ruin",
        rating: 4.5,
        num: 284,
    },
    victorystar: {
        onAnyModifyAccuracyPriority: -1,
        onAnyModifyAccuracy: function (accuracy, target, source) {
            if (source.isAlly(this.effectState.target) && typeof accuracy === 'number') {
                return this.chainModify([4506, 4096]);
            }
        },
        flags: {},
        name: "Victory Star",
        rating: 2,
        num: 162,
    },
    vitalspirit: {
        onUpdate: function (pokemon) {
            if (pokemon.status === 'slp') {
                this.add('-activate', pokemon, 'ability: Vital Spirit');
                pokemon.cureStatus();
            }
        },
        onSetStatus: function (status, target, source, effect) {
            if (status.id !== 'slp')
                return;
            if (effect === null || effect === void 0 ? void 0 : effect.status) {
                this.add('-immune', target, '[from] ability: Vital Spirit');
            }
            return false;
        },
        onTryAddVolatile: function (status, target) {
            if (status.id === 'yawn') {
                this.add('-immune', target, '[from] ability: Vital Spirit');
                return null;
            }
        },
        flags: { breakable: 1 },
        name: "Vital Spirit",
        rating: 1.5,
        num: 72,
    },
    voltabsorb: {
        onTryHit: function (target, source, move) {
            if (target !== source && move.type === 'Electric') {
                if (!this.heal(target.baseMaxhp / 4)) {
                    this.add('-immune', target, '[from] ability: Volt Absorb');
                }
                return null;
            }
        },
        flags: { breakable: 1 },
        name: "Volt Absorb",
        rating: 3.5,
        num: 10,
    },
    wanderingspirit: {
        onDamagingHit: function (damage, target, source, move) {
            if (source.getAbility().flags['failskillswap'] || target.volatiles['dynamax'])
                return;
            if (this.checkMoveMakesContact(move, source, target)) {
                var targetCanBeSet = this.runEvent('SetAbility', target, source, this.effect, source.ability);
                if (!targetCanBeSet)
                    return targetCanBeSet;
                var sourceAbility = source.setAbility('wanderingspirit', target);
                if (!sourceAbility)
                    return;
                if (target.isAlly(source)) {
                    this.add('-activate', target, 'Skill Swap', '', '', "[of] ".concat(source));
                }
                else {
                    this.add('-activate', target, 'ability: Wandering Spirit', this.dex.abilities.get(sourceAbility).name, 'Wandering Spirit', "[of] ".concat(source));
                }
                target.setAbility(sourceAbility);
            }
        },
        flags: {},
        name: "Wandering Spirit",
        rating: 2.5,
        num: 254,
    },
    waterabsorb: {
        onTryHit: function (target, source, move) {
            if (target !== source && move.type === 'Water') {
                if (!this.heal(target.baseMaxhp / 4)) {
                    this.add('-immune', target, '[from] ability: Water Absorb');
                }
                return null;
            }
        },
        flags: { breakable: 1 },
        name: "Water Absorb",
        rating: 3.5,
        num: 11,
    },
    waterbubble: {
        onSourceModifyAtkPriority: 5,
        onSourceModifyAtk: function (atk, attacker, defender, move) {
            if (move.type === 'Fire') {
                return this.chainModify(0.5);
            }
        },
        onSourceModifySpAPriority: 5,
        onSourceModifySpA: function (atk, attacker, defender, move) {
            if (move.type === 'Fire') {
                return this.chainModify(0.5);
            }
        },
        onModifyAtk: function (atk, attacker, defender, move) {
            if (move.type === 'Water') {
                return this.chainModify(2);
            }
        },
        onModifySpA: function (atk, attacker, defender, move) {
            if (move.type === 'Water') {
                return this.chainModify(2);
            }
        },
        onUpdate: function (pokemon) {
            if (pokemon.status === 'brn') {
                this.add('-activate', pokemon, 'ability: Water Bubble');
                pokemon.cureStatus();
            }
        },
        onSetStatus: function (status, target, source, effect) {
            if (status.id !== 'brn')
                return;
            if (effect === null || effect === void 0 ? void 0 : effect.status) {
                this.add('-immune', target, '[from] ability: Water Bubble');
            }
            return false;
        },
        flags: { breakable: 1 },
        name: "Water Bubble",
        rating: 4.5,
        num: 199,
    },
    watercompaction: {
        onDamagingHit: function (damage, target, source, move) {
            if (move.type === 'Water') {
                this.boost({ def: 2 });
            }
        },
        flags: {},
        name: "Water Compaction",
        rating: 1.5,
        num: 195,
    },
    waterveil: {
        onUpdate: function (pokemon) {
            if (pokemon.status === 'brn') {
                this.add('-activate', pokemon, 'ability: Water Veil');
                pokemon.cureStatus();
            }
        },
        onSetStatus: function (status, target, source, effect) {
            if (status.id !== 'brn')
                return;
            if (effect === null || effect === void 0 ? void 0 : effect.status) {
                this.add('-immune', target, '[from] ability: Water Veil');
            }
            return false;
        },
        flags: { breakable: 1 },
        name: "Water Veil",
        rating: 2,
        num: 41,
    },
    weakarmor: {
        onDamagingHit: function (damage, target, source, move) {
            if (move.category === 'Physical') {
                this.boost({ def: -1, spe: 2 }, target, target);
            }
        },
        flags: {},
        name: "Weak Armor",
        rating: 1,
        num: 133,
    },
    wellbakedbody: {
        onTryHit: function (target, source, move) {
            if (target !== source && move.type === 'Fire') {
                if (!this.boost({ def: 2 })) {
                    this.add('-immune', target, '[from] ability: Well-Baked Body');
                }
                return null;
            }
        },
        flags: { breakable: 1 },
        name: "Well-Baked Body",
        rating: 3.5,
        num: 273,
    },
    whitesmoke: {
        onTryBoost: function (boost, target, source, effect) {
            if (source && target === source)
                return;
            var showMsg = false;
            var i;
            for (i in boost) {
                if (boost[i] < 0) {
                    delete boost[i];
                    showMsg = true;
                }
            }
            if (showMsg && !effect.secondaries && effect.id !== 'octolock') {
                this.add("-fail", target, "unboost", "[from] ability: White Smoke", "[of] ".concat(target));
            }
        },
        flags: { breakable: 1 },
        name: "White Smoke",
        rating: 2,
        num: 73,
    },
    wimpout: {
        onEmergencyExit: function (target) {
            if (!this.canSwitch(target.side) || target.forceSwitchFlag || target.switchFlag)
                return;
            for (var _i = 0, _a = this.sides; _i < _a.length; _i++) {
                var side = _a[_i];
                for (var _b = 0, _c = side.active; _b < _c.length; _b++) {
                    var active = _c[_b];
                    active.switchFlag = false;
                }
            }
            target.switchFlag = true;
            this.add('-activate', target, 'ability: Wimp Out');
        },
        flags: {},
        name: "Wimp Out",
        rating: 1,
        num: 193,
    },
    windpower: {
        onDamagingHitOrder: 1,
        onDamagingHit: function (damage, target, source, move) {
            if (move.flags['wind']) {
                target.addVolatile('charge');
            }
        },
        onAllySideConditionStart: function (target, source, sideCondition) {
            var pokemon = this.effectState.target;
            if (sideCondition.id === 'tailwind') {
                pokemon.addVolatile('charge');
            }
        },
        flags: {},
        name: "Wind Power",
        rating: 1,
        num: 277,
    },
    windrider: {
        onStart: function (pokemon) {
            if (pokemon.side.sideConditions['tailwind']) {
                this.boost({ atk: 1 }, pokemon, pokemon);
            }
        },
        onTryHit: function (target, source, move) {
            if (target !== source && move.flags['wind']) {
                if (!this.boost({ atk: 1 }, target, target)) {
                    this.add('-immune', target, '[from] ability: Wind Rider');
                }
                return null;
            }
        },
        onAllySideConditionStart: function (target, source, sideCondition) {
            var pokemon = this.effectState.target;
            if (sideCondition.id === 'tailwind') {
                this.boost({ atk: 1 }, pokemon, pokemon);
            }
        },
        flags: { breakable: 1 },
        name: "Wind Rider",
        rating: 3.5,
        // We do not want Brambleghast to get Infiltrator in Randbats
        num: 274,
    },
    wonderguard: {
        onTryHit: function (target, source, move) {
            if (target === source || move.category === 'Status' || move.type === '???' || move.id === 'struggle')
                return;
            if (move.id === 'skydrop' && !source.volatiles['skydrop'])
                return;
            this.debug('Wonder Guard immunity: ' + move.id);
            if (target.runEffectiveness(move) <= 0) {
                if (move.smartTarget) {
                    move.smartTarget = false;
                }
                else {
                    this.add('-immune', target, '[from] ability: Wonder Guard');
                }
                return null;
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, failskillswap: 1, breakable: 1 },
        name: "Wonder Guard",
        rating: 5,
        num: 25,
    },
    wonderskin: {
        onModifyAccuracyPriority: 10,
        onModifyAccuracy: function (accuracy, target, source, move) {
            if (move.category === 'Status' && typeof accuracy === 'number') {
                this.debug('Wonder Skin - setting accuracy to 50');
                return 50;
            }
        },
        flags: { breakable: 1 },
        name: "Wonder Skin",
        rating: 2,
        num: 147,
    },
    zenmode: {
        onResidualOrder: 29,
        onResidual: function (pokemon) {
            if (pokemon.baseSpecies.baseSpecies !== 'Darmanitan' || pokemon.transformed) {
                return;
            }
            if (pokemon.hp <= pokemon.maxhp / 2 && !['Zen', 'Galar-Zen'].includes(pokemon.species.forme)) {
                pokemon.addVolatile('zenmode');
            }
            else if (pokemon.hp > pokemon.maxhp / 2 && ['Zen', 'Galar-Zen'].includes(pokemon.species.forme)) {
                pokemon.addVolatile('zenmode'); // in case of base Darmanitan-Zen
                pokemon.removeVolatile('zenmode');
            }
        },
        onEnd: function (pokemon) {
            if (!pokemon.volatiles['zenmode'] || !pokemon.hp)
                return;
            pokemon.transformed = false;
            delete pokemon.volatiles['zenmode'];
            if (pokemon.species.baseSpecies === 'Darmanitan' && pokemon.species.battleOnly) {
                pokemon.formeChange(pokemon.species.battleOnly, this.effect, false, '0', '[silent]');
            }
        },
        condition: {
            onStart: function (pokemon) {
                if (!pokemon.species.name.includes('Galar')) {
                    if (pokemon.species.id !== 'darmanitanzen')
                        pokemon.formeChange('Darmanitan-Zen');
                }
                else {
                    if (pokemon.species.id !== 'darmanitangalarzen')
                        pokemon.formeChange('Darmanitan-Galar-Zen');
                }
            },
            onEnd: function (pokemon) {
                if (['Zen', 'Galar-Zen'].includes(pokemon.species.forme)) {
                    pokemon.formeChange(pokemon.species.battleOnly);
                }
            },
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, cantsuppress: 1 },
        name: "Zen Mode",
        rating: 0,
        num: 161,
    },
    zerotohero: {
        onSwitchOut: function (pokemon) {
            if (pokemon.baseSpecies.baseSpecies !== 'Palafin')
                return;
            if (pokemon.species.forme !== 'Hero') {
                pokemon.formeChange('Palafin-Hero', this.effect, true);
                pokemon.regressionForme = false;
            }
        },
        onSwitchIn: function (pokemon) {
            if (pokemon.baseSpecies.baseSpecies !== 'Palafin')
                return;
            if (!this.effectState.heroMessageDisplayed && pokemon.species.forme === 'Hero') {
                this.add('-activate', pokemon, 'ability: Zero to Hero');
                this.effectState.heroMessageDisplayed = true;
            }
        },
        flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1, cantsuppress: 1, notransform: 1 },
        name: "Zero to Hero",
        rating: 5,
        num: 278,
    },
    // CAP
    mountaineer: {
        onDamage: function (damage, target, source, effect) {
            if (effect && effect.id === 'stealthrock') {
                return false;
            }
        },
        onTryHit: function (target, source, move) {
            if (move.type === 'Rock' && !target.activeTurns) {
                this.add('-immune', target, '[from] ability: Mountaineer');
                return null;
            }
        },
        isNonstandard: "CAP",
        flags: { breakable: 1 },
        name: "Mountaineer",
        rating: 3,
        num: -2,
    },
    rebound: {
        isNonstandard: "CAP",
        onTryHitPriority: 1,
        onTryHit: function (target, source, move) {
            if (this.effectState.target.activeTurns)
                return;
            if (target === source || move.hasBounced || !move.flags['reflectable']) {
                return;
            }
            var newMove = this.dex.getActiveMove(move.id);
            newMove.hasBounced = true;
            this.actions.useMove(newMove, target, { target: source });
            return null;
        },
        onAllyTryHitSide: function (target, source, move) {
            if (this.effectState.target.activeTurns)
                return;
            if (target.isAlly(source) || move.hasBounced || !move.flags['reflectable']) {
                return;
            }
            var newMove = this.dex.getActiveMove(move.id);
            newMove.hasBounced = true;
            this.actions.useMove(newMove, this.effectState.target, { target: source });
            return null;
        },
        condition: {
            duration: 1,
        },
        flags: { breakable: 1 },
        name: "Rebound",
        rating: 3,
        num: -3,
    },
    persistent: {
        isNonstandard: "CAP",
        // implemented in the corresponding move
        flags: {},
        name: "Persistent",
        rating: 3,
        num: -4,
    },
};
