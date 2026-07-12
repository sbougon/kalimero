import { ROWS, COLS, CELLS, applyMove, checkWin, getLegalMoves, resetBoard } from "./core.js";

export const POLICY_SCHEMA = "connect4-solver-policy-v3";
export const LAYER_SIZES = [98, 96, 64, 7];

function randn() {
  const u = Math.max(Number.EPSILON, Math.random());
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function createPolicyModel() {
  const layers = [];
  for (let l = 1; l < LAYER_SIZES.length; l++) {
    const inputSize = LAYER_SIZES[l - 1];
    const outputSize = LAYER_SIZES[l];
    const weights = new Float32Array(inputSize * outputSize);
    const scale = Math.sqrt(2 / inputSize);
    for (let i = 0; i < weights.length; i++) weights[i] = randn() * scale;
    layers.push({ inputSize, outputSize, weights, biases: new Float32Array(outputSize) });
  }
  return {
    schema: POLICY_SCHEMA,
    sizes: [...LAYER_SIZES],
    layers,
    skipWeights: new Float32Array(COLS * COLS * 2),
    training: { positions: 0, epochs: 0, averageLoss: 0, teacherAgreement: 0, solverDepth: 0 }
  };
}

export function encodePosition(board, player) {
  const input = new Float32Array(CELLS * 2 + COLS * 2);
  for (let i = 0; i < CELLS; i++) {
    if (board[i] === player) input[i] = 1;
    else if (board[i] === -player) input[CELLS + i] = 1;
  }
  for (let col=0;col<COLS;col++) {
    const mine=new Int8Array(board), myRow=applyMove(mine,col,player);
    if(myRow>=0 && checkWin(mine,myRow,col,player)) input[CELLS*2+col]=1;
    const theirs=new Int8Array(board), theirRow=applyMove(theirs,col,-player);
    if(theirRow>=0 && checkWin(theirs,theirRow,col,-player)) input[CELLS*2+COLS+col]=1;
  }
  return input;
}

export function policyForward(model, board, player) {
  const activations = [encodePosition(board, player)];
  const preActivations = [];
  let current = activations[0];
  for (let l = 0; l < model.layers.length; l++) {
    const layer = model.layers[l];
    const z = new Float32Array(layer.outputSize);
    const next = new Float32Array(layer.outputSize);
    for (let o = 0; o < layer.outputSize; o++) {
      let sum = layer.biases[o];
      const offset = o * layer.inputSize;
      for (let i = 0; i < layer.inputSize; i++) sum += layer.weights[offset + i] * current[i];
      if (l === model.layers.length - 1) {
        for (let t=0;t<COLS*2;t++) sum += model.skipWeights[o*COLS*2+t] * activations[0][CELLS*2+t];
      }
      z[o] = sum;
      next[o] = l === model.layers.length - 1 ? sum : Math.max(0, sum);
    }
    preActivations.push(z);
    activations.push(next);
    current = next;
  }
  const legal = getLegalMoves(board);
  const probabilities = new Float32Array(COLS);
  if (legal.length) {
    let maxLogit = -Infinity;
    for (const col of legal) maxLogit = Math.max(maxLogit, current[col]);
    let total = 0;
    for (const col of legal) { probabilities[col] = Math.exp(current[col] - maxLogit); total += probabilities[col]; }
    for (const col of legal) probabilities[col] /= total;
  }
  return { logits: current, probabilities, activations, preActivations };
}

export function choosePolicyMove(model, board, player) {
  const legal = getLegalMoves(board);
  if (!legal.length) return null;
  const probabilities = policyForward(model, board, player).probabilities;
  let best = legal[0];
  for (const col of legal) if (probabilities[col] > probabilities[best]) best = col;
  return best;
}

function immediateWins(board, player) {
  const wins = [];
  for (const col of getLegalMoves(board)) {
    const copy = new Int8Array(board);
    const row = applyMove(copy, col, player);
    if (checkWin(copy, row, col, player)) wins.push(col);
  }
  return wins;
}

export function chooseEngineRandomMove(board, player) {
  const wins = immediateWins(board, player);
  if (wins.length) return wins[Math.floor(Math.random() * wins.length)];
  const blocks = immediateWins(board, -player);
  if (blocks.length) return blocks[Math.floor(Math.random() * blocks.length)];
  return chooseRandomMove(board);
}

export function chooseRandomMove(board) {
  const legal = getLegalMoves(board);
  return legal.length ? legal[Math.floor(Math.random() * legal.length)] : null;
}

const WINDOWS = (() => {
  const result = [];
  const dirs = [[1,0],[0,1],[1,1],[1,-1]];
  for (let r=0;r<ROWS;r++) for (let c=0;c<COLS;c++) for (const [dr,dc] of dirs) {
    const endR=r+3*dr, endC=c+3*dc;
    if (endR>=0&&endR<ROWS&&endC>=0&&endC<COLS) result.push([0,1,2,3].map(n=>(r+n*dr)*COLS+c+n*dc));
  }
  return result;
})();

function heuristic(board, player) {
  let score = 0;
  for (let r=0;r<ROWS;r++) if (board[r*COLS+3] === player) score += 5; else if (board[r*COLS+3] === -player) score -= 5;
  for (const window of WINDOWS) {
    let mine=0, theirs=0, empty=0;
    for (const idx of window) { if(board[idx]===player) mine++; else if(board[idx]===-player) theirs++; else empty++; }
    if (!theirs) score += mine===3&&empty===1 ? 80 : mine===2&&empty===2 ? 12 : mine===1 ? 1 : 0;
    if (!mine) score -= theirs===3&&empty===1 ? 95 : theirs===2&&empty===2 ? 14 : theirs===1 ? 1 : 0;
  }
  return score;
}

function orderedMoves(board) {
  const legal = getLegalMoves(board);
  return [3,2,4,1,5,0,6].filter(col => legal.includes(col));
}

function negamax(board, player, depth, alpha, beta, table) {
  const legal = getLegalMoves(board);
  if (!legal.length) return 0;
  if (depth <= 0) return heuristic(board, player);
  const key = `${depth}:${player}:${board.join("")}`;
  if (table.has(key)) return table.get(key);
  let best = -Infinity;
  for (const col of orderedMoves(board)) {
    const copy = new Int8Array(board);
    const row = applyMove(copy, col, player);
    let score;
    if (checkWin(copy, row, col, player)) score = 100000 + depth;
    else score = -negamax(copy, -player, depth - 1, -beta, -alpha, table);
    best = Math.max(best, score); alpha = Math.max(alpha, score);
    if (alpha >= beta) break;
  }
  table.set(key, best);
  return best;
}

export function solverScores(board, player, depth = 4) {
  const scores = Array(COLS).fill(null);
  const table = new Map();
  for (const col of orderedMoves(board)) {
    const copy = new Int8Array(board);
    const row = applyMove(copy, col, player);
    scores[col] = checkWin(copy, row, col, player) ? 100000 + depth : -negamax(copy, -player, depth - 1, -Infinity, Infinity, table);
  }
  return scores;
}

export function chooseSolverMove(board, player, depth = 5) {
  const scores = solverScores(board, player, depth);
  let best = null;
  for (let col=0;col<COLS;col++) if (scores[col] !== null && (best===null || scores[col] > scores[best])) best=col;
  return best;
}

export function teacherTarget(scores) {
  const target = new Float32Array(COLS);
  let best = -Infinity, count = 0;
  for (const score of scores) if (score !== null) best = Math.max(best, score);
  for (let c=0;c<COLS;c++) if (scores[c] === best) count++;
  for (let c=0;c<COLS;c++) if (scores[c] === best) target[c] = 1 / count;
  return target;
}

export function serializePolicy(model) {
  return { schema:POLICY_SCHEMA, sizes:[...model.sizes], training:{...model.training}, skipWeights:Array.from(model.skipWeights), layers:model.layers.map(l=>({inputSize:l.inputSize,outputSize:l.outputSize,weights:Array.from(l.weights),biases:Array.from(l.biases)})) };
}

export function deserializePolicy(raw) {
  if (!raw || raw.schema !== POLICY_SCHEMA || JSON.stringify(raw.sizes) !== JSON.stringify(LAYER_SIZES)) throw new Error("Expected a solver policy model with shape 98 -> 96 -> 64 -> 7");
  return { schema:POLICY_SCHEMA, sizes:[...raw.sizes], training:{ positions:0,epochs:0,averageLoss:0,teacherAgreement:0,solverDepth:0,...raw.training }, skipWeights:new Float32Array(raw.skipWeights), layers:raw.layers.map(l=>({inputSize:l.inputSize,outputSize:l.outputSize,weights:new Float32Array(l.weights),biases:new Float32Array(l.biases)})) };
}

export function newBoard() { return resetBoard(); }
