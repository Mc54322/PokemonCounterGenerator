// convert_moves.js
const fs = require('fs');
const path = require('path');


// Adjust this require according to how your module exports the moves.
// For example, if moves.js exports { Moves: {...} }:
const movesData = require(path.join(__dirname, 'items')).Items;

// Convert the moves data to a JSON string with pretty-printing.
const jsonString = JSON.stringify(movesData, null, 2);

// Write the JSON string to moves.json in the same folder.
fs.writeFileSync(path.join(__dirname, 'items.json'), jsonString, 'utf8');

console.log('Conversion complete: abilities.json created.');
