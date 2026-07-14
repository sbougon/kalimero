import { applyMove, checkWin, getLegalMoves } from "./core.js";
import { policyForward } from "./solver-core.js";

export function immediateWinningMoves(board, player) {
  const wins = [];
  for (const col of getLegalMoves(board)) {
    const copy = new Int8Array(board);
    const row = applyMove(copy, col, player);
    if (checkWin(copy, row, col, player)) wins.push(col);
  }
  return wins;
}

export function forkMoves(board, player) {
  const forks = [];
  for (const col of getLegalMoves(board)) {
    const copy = new Int8Array(board);
    applyMove(copy, col, player);
    if (immediateWinningMoves(copy, player).length >= 2) forks.push(col);
  }
  return forks;
}

function bestPolicyMove(model, board, player, candidates) {
  if (!candidates.length) return null;
  const probabilities = policyForward(model, board, player).probabilities;
  return candidates.reduce((best, col) => probabilities[col] > probabilities[best] ? col : best);
}

export function tacticallySafeMoves(board, player) {
  const legal = getLegalMoves(board);
  const safe = legal.filter(col => {
    const copy = new Int8Array(board);
    applyMove(copy, col, player);
    return immediateWinningMoves(copy, -player).length === 0;
  });
  return safe.length ? safe : legal;
}

export function chooseTacticalPolicyMove(model, board, player) {
  const wins = immediateWinningMoves(board, player);
  if (wins.length) return bestPolicyMove(model, board, player, wins);

  const forcedBlocks = immediateWinningMoves(board, -player);
  if (forcedBlocks.length === 1) return forcedBlocks[0];

  const safe = tacticallySafeMoves(board, player);
  const forks = forkMoves(board, player).filter(col => safe.includes(col));
  return bestPolicyMove(model, board, player, forks.length ? forks : safe);
}

export function hardTarget(columns, size = 7) {
  const target = new Float32Array(size);
  if (!columns.length) return target;
  for (const col of columns) target[col] = 1 / columns.length;
  return target;
}

export function mirrorSample(sample) {
  const board = new Int8Array(sample.board.length);
  for (let row = 0; row < 6; row++) {
    for (let col = 0; col < 7; col++) board[row * 7 + col] = sample.board[row * 7 + 6 - col];
  }
  return {
    ...sample,
    board,
    target: Float32Array.from(sample.target).reverse(),
    scores: sample.scores ? [...sample.scores].reverse() : sample.scores
  };
}
