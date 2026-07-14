import { applyMove, checkWin } from "./core.js";
import { chooseEngineRandomMove, choosePolicyMove, chooseRandomMove, chooseSolverMove, deserializePolicy, newBoard } from "./solver-core.js";
import { chooseTacticalPolicyMove } from "./tactical-core.js";

function move(kind, model, board, player) {
  if (kind === "v3") return chooseTacticalPolicyMove(model, board, player);
  if (kind === "ai") return choosePolicyMove(model, board, player);
  if (kind === "solver") return chooseSolverMove(board, player, 5);
  if (kind === "engine-random") return chooseEngineRandomMove(board, player);
  return chooseRandomMove(board);
}

function game(model, aiPlayer, aiKind, opponent) {
  const board = newBoard();
  let player = 1;
  for (let turn = 0; turn < 42; turn++) {
    const col = move(player === aiPlayer ? aiKind : opponent, model, board, player);
    if (col === null) return 0;
    const row = applyMove(board, col, player);
    if (checkWin(board, row, col, player)) return player;
    player = -player;
  }
  return 0;
}

self.onmessage = event => {
  if (event.data.type !== "benchmark") return;
  const model = deserializePolicy(event.data.model);
  const games = event.data.games;
  const opponent = event.data.opponent;
  const aiKind = event.data.aiKind || "v3";
  const red = { games: 0, wins: 0, draws: 0, losses: 0 };
  const yellow = { games: 0, wins: 0, draws: 0, losses: 0 };
  for (let index = 0; index < games; index++) {
    const aiPlayer = index % 2 === 0 ? 1 : -1;
    const bucket = aiPlayer === 1 ? red : yellow;
    const winner = game(model, aiPlayer, aiKind, opponent);
    bucket.games++;
    if (winner === aiPlayer) bucket.wins++;
    else if (winner === 0) bucket.draws++;
    else bucket.losses++;
    if (index % 100 === 99) self.postMessage({ type: "progress", completed: index + 1, total: games });
  }
  for (const bucket of [red, yellow]) bucket.score = (bucket.wins + 0.5 * bucket.draws) / bucket.games;
  const wins = red.wins + yellow.wins;
  const draws = red.draws + yellow.draws;
  const losses = red.losses + yellow.losses;
  self.postMessage({ type: "done", result: { games, opponent, aiKind, wins, draws, losses, score: (wins + 0.5 * draws) / games, red, yellow } });
};
