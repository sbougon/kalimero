import { Worker } from "node:worker_threads";
import { applyMove, checkWin } from "../src/core.js";
import { createPolicyModel, deserializePolicy, serializePolicy, choosePolicyMove, chooseRandomMove, chooseEngineRandomMove, newBoard } from "../src/solver-core.js";

function play(model, aiPlayer, opponent) {
  const board = newBoard();
  let player = 1;
  for (let turn = 0; turn < 42; turn++) {
    const col = player === aiPlayer
      ? choosePolicyMove(model, board, player)
      : opponent === "random"
        ? chooseRandomMove(board)
        : chooseEngineRandomMove(board, player);
    if (col === null) return 0;
    const row = applyMove(board, col, player);
    if (checkWin(board, row, col, player)) return player;
    player = -player;
  }
  return 0;
}

function benchmark(model, opponent, games = 200) {
  const result = { wins: 0, draws: 0, losses: 0 };
  for (let i = 0; i < games; i++) {
    const aiPlayer = i % 2 ? -1 : 1;
    const winner = play(model, aiPlayer, opponent);
    if (winner === aiPlayer) result.wins++;
    else if (winner === 0) result.draws++;
    else result.losses++;
  }
  result.score = (result.wins + 0.5 * result.draws) / games;
  return result;
}

function train(model, positions, depth) {
  const moduleUrl = new URL("../src/solver-trainer-worker.js", import.meta.url).href;
  const source = `
    import { parentPort } from "node:worker_threads";
    globalThis.self = {
      postMessage: message => parentPort.postMessage(message),
      onmessage: null
    };
    parentPort.on("message", data => globalThis.self.onmessage({ data }));
    await import(${JSON.stringify(moduleUrl)});
  `;
  const url = new URL(`data:text/javascript,${encodeURIComponent(source)}`);
  const worker = new Worker(url, { type: "module" });
  return new Promise((resolve, reject) => {
    worker.on("message", message => {
      if (message.type === "done") {
        worker.terminate();
        resolve(deserializePolicy(message.model));
      } else if (message.type === "error") reject(new Error(message.message));
    });
    worker.on("error", reject);
    worker.postMessage({ type: "train", model: serializePolicy(model), positions, depth, learningRate: 0.002 });
  });
}

let model = createPolicyModel();
console.log("untrained random", benchmark(model, "random"));
console.log("untrained engine-random", benchmark(model, "engine-random"));
const runs=Number(process.argv[4]||1);
for(let run=1;run<=runs;run++) {
  model = await train(model, Number(process.argv[2] || 2000), Number(process.argv[3] || 3));
  console.log(`training run ${run}`, model.training);
  console.log("trained random", benchmark(model, "random"));
  console.log("trained engine-random", benchmark(model, "engine-random"));
}
