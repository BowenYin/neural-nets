/**
 * Bowen Yin
 * 9/10/2019
 * 
 */
const LAYERS = 3;
const INPUT_NODES = 2;
const OUTPUT_NODES = 1;
let inputs = new Array(LAYERS).fill(0).map(x => new Array(INPUT_NODES));
let weights = new Array(LAYERS).fill(0).map(x => new Array(INPUT_NODES).map(x => new Array()));
for (let i=0; i<weights.length; i++)
  for (let j=0; j<weights[i].length; j++)
    weights[i][j]=[];

inputs[0][0] = 0;
inputs[0][1] = 1;

weights[0][0][0] = 0.55;
weights[0][0][1] = 0.2;
weights[0][1][0] = 0.825;
weights[0][1][1] = 0.08;
weights[1][0][0] = 0.3;
weights[1][1][0] = 0.9;

function getThreshold(a, w) {
  return 
}
/**
 * 
 */
function populateInputs() {
  for (let i = 1; i < inputs.length-1; i++) {
    for (let j = 0; j < i; j++) {
      inputs[i][j] = inputs[i-1][j]*weights[0][j][j]+inputs[i-1][j];
    }
  }
  inputs[1][0] = inputs[0][0]*weights[0][0][0]+inputs[0][1]*weights[0][1][0];
  inputs[1][1] = inputs[0][0]*weights[0][0][1]+inputs[0][1]*weights[0][1][1];
  inputs[2][0] = inputs[1][0]*weights[1][0][0]+inputs[1][1]*weights[1][1][0];
}

/**
 * 
 */
function populateWeights() {
  
}

populateInputs();
console.log("OUTPUT: "+inputs[2][0]);