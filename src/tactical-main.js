import { ROWS, COLS, applyMove, checkWin, getLegalMoves } from "./core.js";
import {
  chooseEngineRandomMove,
  choosePolicyMove,
  chooseRandomMove,
  chooseSolverMove,
  createPolicyModel,
  deserializePolicy,
  modelLayers,
  newBoard,
  parameterCount,
  policyForward,
  serializePolicy,
  solverScores
} from "./solver-core.js";
import { chooseTacticalPolicyMove } from "./tactical-core.js";

const STORAGE_KEY = "connect4.tacticalPolicy.v3";
const V2_STORAGE_KEY = "connect4.solverPolicy.v2";
const HISTORY_KEY = "connect4.tacticalPolicy.benchmarks.v3";
const EXPORT_SCHEMA = "connect4-tactical-pipeline-state-v1";
const $ = id => document.getElementById(id);
const els = {
  status: $("appStatus"), board: $("board"), columns: $("columnMoves"), gameStatus: $("gameStatus"),
  red: $("redPlayer"), yellow: $("yellowPlayer"), newGame: $("newGameBtn"), moveReadout: $("moveReadout"),
  save: $("saveBtn"), export: $("exportBtn"), exportState: $("exportStateBtn"), import: $("importInput"), reset: $("resetBtn"),
  positions: $("positionsInput"), depth: $("depthInput"), lr: $("learningRateInput"), train: $("trainBtn"),
  quick: $("quickTrainBtn"), stop: $("stopBtn"), trainState: $("trainState"), progress: $("trainProgress"), trainStats: $("trainStats"),
  benchmark: $("benchmarkBtn"), benchmarkStatus: $("benchmarkStatus"), benchmarkOpponent: $("benchmarkOpponent"),
  benchmarkGames: $("benchmarkGames"), benchmarkMode: $("benchmarkMode"), benchmarkStats: $("benchmarkStats"), benchmarkCanvas: $("benchmarkCanvas"),
  networkCanvas: $("networkCanvas"), weightsCanvas: $("weightsCanvas"), layer: $("layerSelect"), neuron: $("neuronInput"),
  neuronOutput: $("neuronOutput"), neuronStats: $("neuronStats"), activationRows: $("activationRows"), refresh: $("refreshBtn")
};

let initialModelSource = "fresh";
let model = loadModel();
let history = loadHistory();
let board = newBoard();
let player = 1;
let gameOver = false;
let autoToken = 0;
let trainingWorker = null;
let benchmarkWorker = null;
let trainingStatus = idleStatus();
let benchmarkRunStatus = idleStatus();

function idleStatus() {
  return { active: false, state: "idle", stage: "idle", completed: 0, total: 0, startedAt: null, phaseStartedAt: null, lastProgressAt: null, finishedAt: null, requested: null, error: null };
}

function loadModel() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      initialModelSource = "v3";
      return deserializePolicy(JSON.parse(raw));
    }
    const v2 = localStorage.getItem(V2_STORAGE_KEY);
    if (v2) {
      initialModelSource = "v2";
      const imported = deserializePolicy(JSON.parse(v2));
      localStorage.setItem(STORAGE_KEY, JSON.stringify(serializePolicy(imported)));
      return imported;
    }
    return createPolicyModel();
  } catch (error) {
    console.error(error);
    return createPolicyModel();
  }
}

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); }
  catch { return []; }
}

function saveModel() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(serializePolicy(model)));
  els.status.textContent = `Saved V3 at ${model.training.positions.toLocaleString()} labeled positions`;
}

function setModel(next) {
  model = next;
  saveModel();
  renderAll();
}

function label(kind) {
  return { human: "Human", v3: "AI V3", ai: "Pure NN", solver: "Solver", random: "Random", "engine-random": "Engine + random" }[kind];
}

function playerKind() { return player === 1 ? els.red.value : els.yellow.value; }

function choose(kind) {
  if (kind === "v3") return chooseTacticalPolicyMove(model, board, player);
  if (kind === "ai") return choosePolicyMove(model, board, player);
  if (kind === "solver") return chooseSolverMove(board, player, 5);
  if (kind === "engine-random") return chooseEngineRandomMove(board, player);
  return chooseRandomMove(board);
}

function renderBoard() {
  const cells = [];
  for (let row = ROWS - 1; row >= 0; row--) {
    for (let col = 0; col < COLS; col++) {
      const value = board[row * COLS + col];
      cells.push(`<div class="cell ${value === 1 ? "red" : value === -1 ? "yellow" : ""}"></div>`);
    }
  }
  els.board.innerHTML = cells.join("");
  const legal = getLegalMoves(board);
  const human = !gameOver && playerKind() === "human";
  els.columns.innerHTML = Array.from({ length: COLS }, (_, col) => `<button data-col="${col}" ${human && legal.includes(col) ? "" : "disabled"}>${col + 1}</button>`).join("");
  els.columns.querySelectorAll("button").forEach(button => button.addEventListener("click", () => playMove(Number(button.dataset.col))));
}

function renderGameInfo() {
  if (!gameOver) els.gameStatus.textContent = `${player === 1 ? "Red" : "Yellow"} to move — ${label(playerKind())}`;
  const output = policyForward(model, board, player);
  const legal = getLegalMoves(board);
  let scores;
  try { scores = solverScores(board, player, 3); }
  catch { scores = Array(COLS).fill(null); }
  const tacticalChoice = chooseTacticalPolicyMove(model, board, player);
  els.moveReadout.innerHTML = Array.from({ length: COLS }, (_, col) => `<div class="move-score ${col === tacticalChoice ? "best" : ""}"><b>${col + 1}</b><br>${legal.includes(col) ? `${(output.probabilities[col] * 100).toFixed(1)}%` : "--"}<br><span>${scores[col] === null ? "--" : scores[col]}</span></div>`).join("");
}

function playMove(col) {
  if (gameOver || !getLegalMoves(board).includes(col)) return;
  const moving = player;
  const row = applyMove(board, col, moving);
  if (checkWin(board, row, col, moving)) {
    gameOver = true;
    els.gameStatus.textContent = `${moving === 1 ? "Red" : "Yellow"} wins (${label(moving === 1 ? els.red.value : els.yellow.value)})`;
  } else if (!getLegalMoves(board).length) {
    gameOver = true;
    els.gameStatus.textContent = "Draw";
  } else player = -player;
  renderAll();
  scheduleAuto();
}

function newGame() {
  autoToken++;
  board = newBoard();
  player = 1;
  gameOver = false;
  renderAll();
  scheduleAuto();
}

function scheduleAuto() {
  const token = ++autoToken;
  if (gameOver || playerKind() === "human") return;
  setTimeout(() => {
    if (token !== autoToken || gameOver) return;
    playMove(choose(playerKind()));
  }, 260);
}

function statRows(rows) { return rows.map(([term, value]) => `<dt>${term}</dt><dd>${value}</dd>`).join(""); }

function accuracy(category) {
  const bucket = model.training.tacticalValidation?.[category];
  return bucket?.accuracy == null ? "-" : `${(bucket.accuracy * 100).toFixed(1)}%`;
}

function renderTraining() {
  const training = model.training;
  els.trainStats.innerHTML = statRows([
    ["Seen / replay", `${training.positions.toLocaleString()} / ${training.replaySize.toLocaleString()}`],
    ["Optimizer updates", training.updates.toLocaleString()],
    ["Train / validation loss", `${training.averageLoss.toFixed(4)} / ${training.validationLoss?.toFixed(4) ?? "-"}`],
    ["Train / validation agreement", `${(training.teacherAgreement * 100).toFixed(1)}% / ${training.validationAgreement == null ? "-" : `${(training.validationAgreement * 100).toFixed(1)}%`}`],
    ["Win / block accuracy", `${accuracy("win")} / ${accuracy("block")}`],
    ["Fork / prevention", `${accuracy("fork")} / ${accuracy("fork-prevention")}`],
    ["Learning rate", training.learningRate.toExponential(2)],
    ["Generation", training.generation || "fresh / V2 warm start"]
  ]);
}

function phaseFor(stage = "") {
  if (stage.startsWith("building")) return "validation";
  if (stage.startsWith("curriculum")) return "generation";
  if (stage.includes("learning") || stage.startsWith("final")) return "learning";
  return stage || "unknown";
}

function startTraining(override) {
  if (trainingWorker) return;
  const positions = override || Math.max(128, Math.min(100000, Number(els.positions.value) || 6000));
  const now = Date.now();
  trainingStatus = { ...idleStatus(), active: true, state: "running", stage: "starting", phase: "starting", total: positions, startedAt: now, phaseStartedAt: now, lastProgressAt: now, requested: { positions, solverDepth: Number(els.depth.value), learningRate: Number(els.lr.value), tacticalFraction: 0.75, tacticalWeight: 2, batchSize: 128 } };
  trainingWorker = new Worker("./src/tactical-trainer-worker.js", { type: "module" });
  els.train.disabled = els.quick.disabled = true;
  els.stop.disabled = false;
  els.progress.value = 0;
  els.trainState.textContent = "Starting V3";
  els.status.textContent = `Building V3 curriculum for ${positions.toLocaleString()} positions`;

  trainingWorker.onmessage = event => {
    const data = event.data;
    if (data.type === "progress") {
      const now = Date.now();
      const phase = phaseFor(data.stage);
      if (phase !== trainingStatus.phase) trainingStatus.phaseStartedAt = now;
      trainingStatus = { ...trainingStatus, active: true, state: "running", stage: data.stage, phase, completed: data.completed, total: data.total, lastProgressAt: now, liveLoss: Number.isFinite(data.loss) ? data.loss : null, liveAgreement: Number.isFinite(data.agreement) ? data.agreement : null };
      els.progress.value = data.completed / data.total;
      els.trainState.textContent = data.stage;
      els.trainStats.innerHTML = statRows([["Stage progress", `${data.completed.toLocaleString()} / ${data.total.toLocaleString()}`], ["Live loss", data.loss ? data.loss.toFixed(4) : "-"], ["Live agreement", data.agreement ? `${(data.agreement * 100).toFixed(1)}%` : "-"]]);
    } else if (data.type === "done") {
      model = deserializePolicy(data.model);
      trainingStatus = { ...trainingStatus, active: false, state: data.stopped ? "stopped" : "complete", stage: data.stopped ? "stopped" : "complete", finishedAt: Date.now(), result: { completedExamples: data.completed, improved: data.improved, validation: data.validation, replaySize: data.replaySize, learningRate: data.learningRate } };
      finishTraining(data.stopped ? "V3 training stopped and saved" : "V3 tactical training complete");
      saveModel();
      renderAll();
    } else if (data.type === "error") {
      trainingStatus = { ...trainingStatus, active: false, state: "error", stage: "error", finishedAt: Date.now(), error: data.message };
      finishTraining(`V3 training error: ${data.message}`);
    }
  };
  trainingWorker.onerror = error => {
    trainingStatus = { ...trainingStatus, active: false, state: "error", stage: "error", finishedAt: Date.now(), error: error.message };
    finishTraining(`V3 training error: ${error.message}`);
  };
  trainingWorker.postMessage({ type: "train", model: serializePolicy(model), positions, depth: Number(els.depth.value), learningRate: Number(els.lr.value), batchSize: 128, replayCapacity: 100000 });
}

function finishTraining(message) {
  trainingWorker?.terminate();
  trainingWorker = null;
  els.train.disabled = els.quick.disabled = false;
  els.stop.disabled = true;
  els.trainState.textContent = "Idle";
  els.status.textContent = message;
}

function runBenchmark() {
  if (benchmarkWorker) return;
  const games = Math.max(100, Math.min(50000, Number(els.benchmarkGames.value) || 5000));
  const opponent = els.benchmarkOpponent.value;
  const aiKind = els.benchmarkMode.value;
  const now = Date.now();
  benchmarkRunStatus = { ...idleStatus(), active: true, state: "running", stage: "benchmark", total: games, startedAt: now, phaseStartedAt: now, lastProgressAt: now, requested: { games, opponent, aiKind, modelPositions: model.training.positions } };
  benchmarkWorker = new Worker("./src/tactical-benchmark-worker.js", { type: "module" });
  els.benchmark.disabled = true;
  els.benchmarkStatus.textContent = `Running ${games.toLocaleString()} ${label(aiKind)} games...`;
  benchmarkWorker.onmessage = event => {
    if (event.data.type === "progress") {
      benchmarkRunStatus = { ...benchmarkRunStatus, completed: event.data.completed, lastProgressAt: Date.now() };
      els.benchmarkStatus.textContent = `Running ${event.data.completed.toLocaleString()} / ${games.toLocaleString()} games...`;
      return;
    }
    if (event.data.type !== "done") return;
    const result = { ...event.data.result, positions: model.training.positions, createdAt: Date.now() };
    benchmarkRunStatus = { ...benchmarkRunStatus, active: false, state: "complete", stage: "complete", completed: games, finishedAt: Date.now(), result };
    history.push(result);
    history = history.slice(-80);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    benchmarkWorker.terminate();
    benchmarkWorker = null;
    els.benchmark.disabled = false;
    renderBenchmark(result);
  };
  benchmarkWorker.onerror = error => {
    benchmarkRunStatus = { ...benchmarkRunStatus, active: false, state: "error", stage: "error", finishedAt: Date.now(), error: error.message };
    els.benchmarkStatus.textContent = error.message;
    benchmarkWorker.terminate();
    benchmarkWorker = null;
    els.benchmark.disabled = false;
  };
  benchmarkWorker.postMessage({ type: "benchmark", model: serializePolicy(model), games, opponent, aiKind });
}

function renderBenchmark(last = history.at(-1)) {
  if (last) {
    els.benchmarkStatus.textContent = `Last: ${(last.score * 100).toFixed(1)}% ${label(last.aiKind || "v3")} vs ${label(last.opponent)}`;
    els.benchmarkStats.innerHTML = statRows([["Overall W/D/L", `${last.wins} / ${last.draws} / ${last.losses}`], ["Overall score", `${(last.score * 100).toFixed(1)}%`], ["As Red", `${last.red.wins} / ${last.red.draws} / ${last.red.losses}`], ["As Yellow", `${last.yellow.wins} / ${last.yellow.draws} / ${last.yellow.losses}`]]);
  } else els.benchmarkStats.innerHTML = "";
  const canvas = els.benchmarkCanvas;
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const pad = 34;
  context.fillStyle = "#141a1d";
  context.fillRect(0, 0, width, height);
  context.font = "11px system-ui";
  context.strokeStyle = "#39444a";
  context.fillStyle = "#9eabb1";
  for (let index = 0; index <= 4; index++) {
    const y = 12 + (height - 40) * index / 4;
    context.beginPath(); context.moveTo(pad, y); context.lineTo(width - 8, y); context.stroke(); context.fillText(`${100 - index * 25}%`, 2, y + 4);
  }
  if (!history.length) return;
  const points = history.slice(-40);
  context.strokeStyle = "#55b982";
  context.lineWidth = 3;
  context.beginPath();
  points.forEach((point, index) => {
    const x = pad + index / Math.max(1, points.length - 1) * (width - pad - 12);
    const y = 12 + (1 - point.score) * (height - 40);
    if (index) context.lineTo(x, y); else context.moveTo(x, y);
  });
  context.stroke();
}

function renderNetwork() {
  const canvas = els.networkCanvas;
  const context = canvas.getContext("2d");
  context.fillStyle = "#141a1d"; context.fillRect(0, 0, canvas.width, canvas.height);
  const boxes = [[18, 90, 74, 82, "2×6×7\ninput"], [112, 60, 76, 142, "64 channel\nstem"], [210, 30, 105, 202, "4 residual\n3×3 blocks"], [342, 48, 74, 76, "policy\n7 cols"], [342, 146, 74, 76, "value\n[-1, 1]"]];
  context.font = "12px system-ui"; context.textAlign = "center";
  for (const [x, y, width, height, name] of boxes) {
    context.strokeStyle = "#55b982"; context.strokeRect(x, y, width, height); context.fillStyle = "#dbe4e7";
    name.split("\n").forEach((line, index) => context.fillText(line, x + width / 2, y + height / 2 + index * 16));
  }
  context.fillStyle = "#9eabb1"; context.fillText(`${parameterCount(model).toLocaleString()} parameters + tactical shield`, canvas.width - 105, 250);
  renderWeights(); renderActivations();
}

function renderWeights() {
  const layers = modelLayers(model);
  const layerIndex = Math.min(Number(els.layer.value), layers.length - 1);
  const layer = layers[layerIndex];
  els.neuron.max = layer.outputSize - 1;
  const neuron = Math.min(Number(els.neuron.value), layer.outputSize - 1);
  els.neuron.value = neuron; els.neuronOutput.value = neuron;
  const stride = layer.inputSize * layer.kernel * layer.kernel;
  const values = layer.weights.slice(neuron * stride, (neuron + 1) * stride);
  const canvas = els.weightsCanvas; const context = canvas.getContext("2d");
  const columns = Math.ceil(Math.sqrt(values.length)); const rows = Math.ceil(values.length / columns);
  const cellWidth = canvas.width / columns; const cellHeight = canvas.height / rows; const max = Math.max(...values.map(Math.abs), 0.001);
  for (let index = 0; index < columns * rows; index++) {
    const value = values[index] || 0; const intensity = Math.round(35 + Math.abs(value) / max * 220);
    context.fillStyle = value >= 0 ? `rgb(25,${intensity},110)` : `rgb(${intensity},55,65)`;
    context.fillRect((index % columns) * cellWidth, Math.floor(index / columns) * cellHeight, cellWidth + 0.5, cellHeight + 0.5);
  }
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
  els.neuronStats.innerHTML = statRows([["Selected", `Layer ${layerIndex + 1}, channel ${neuron}`], ["Kernel", `${layer.kernel}×${layer.kernel}`], ["Bias", layer.biases[neuron].toFixed(5)], ["Weight mean", mean.toFixed(5)], ["L2 norm", norm.toFixed(4)]]);
}

function renderActivations() {
  const forward = policyForward(model, board, player);
  let best = 0;
  for (let col = 1; col < COLS; col++) if (forward.probabilities[col] > forward.probabilities[best]) best = col;
  const tactical = chooseTacticalPolicyMove(model, board, player);
  els.activationRows.innerHTML = `<div class="activation-row"><span>pure[${best}]</span><div class="activation-bar"><div class="activation-fill" style="width:${forward.probabilities[best] * 100}%"></div></div><span>${forward.probabilities[best].toFixed(4)}</span></div><div class="activation-row"><span>V3 move</span><div class="activation-bar"><div class="activation-fill" style="width:${(tactical + 1) / 7 * 100}%"></div></div><span>col ${tactical + 1}</span></div>`;
}

function iso(time) { return time ? new Date(time).toISOString() : null; }
function liveStatus(status) {
  const now = Date.now();
  const elapsedMs = status.startedAt ? Math.max(0, (status.finishedAt || now) - status.startedAt) : 0;
  const phaseElapsedMs = status.phaseStartedAt ? Math.max(0, (status.finishedAt || now) - status.phaseStartedAt) : 0;
  const ratePerMinute = status.completed > 0 && phaseElapsedMs > 0 ? status.completed / (phaseElapsedMs / 60000) : null;
  return { ...status, startedAt: iso(status.startedAt), phaseStartedAt: iso(status.phaseStartedAt), lastProgressAt: iso(status.lastProgressAt), finishedAt: iso(status.finishedAt), elapsedMs, phaseElapsedMs, ratePerMinute, estimatedRemainingMs: ratePerMinute && status.total > status.completed ? (status.total - status.completed) / ratePerMinute * 60000 : null, secondsSinceProgress: status.active && status.lastProgressAt ? (now - status.lastProgressAt) / 1000 : null };
}

function pipelineExport() {
  return {
    schema: EXPORT_SCHEMA,
    exportedAt: new Date().toISOString(),
    note: "Weights, biases, optimizer tensors, replay samples, and validation samples omitted.",
    strategy: { generation: "tactical-curriculum-v3", tacticalFraction: 0.75, tacticalLossWeight: 2, strategicTemperature: 0.1, mirrorAugmentation: 0.5, guardedInference: true },
    model: { schema: model.schema, modelId: model.modelId, architecture: { ...model.architecture }, parameterCount: parameterCount(model), training: { ...model.training } },
    pipeline: { training: liveStatus(trainingStatus), benchmark: liveStatus(benchmarkRunStatus) },
    settings: { positions: Number(els.positions.value), strategicDepth: Number(els.depth.value), learningRate: Number(els.lr.value), benchmarkOpponent: els.benchmarkOpponent.value, benchmarkGames: Number(els.benchmarkGames.value), benchmarkMode: els.benchmarkMode.value },
    benchmarks: { count: history.length, history: history.map(result => ({ ...result })) },
    runtime: { page: location.pathname, userAgent: navigator.userAgent, visibility: document.visibilityState }
  };
}

function downloadJson(value, name) {
  const blob = new Blob([`${JSON.stringify(value, null, 2)}\n`], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = name; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

async function exportState() {
  const state = pipelineExport();
  const json = `${JSON.stringify(state, null, 2)}\n`;
  downloadJson(state, `connect4-v3-pipeline-${model.training.positions}.json`);
  try { await navigator.clipboard.writeText(json); els.status.textContent = "V3 pipeline state copied and downloaded"; }
  catch { els.status.textContent = "V3 pipeline state downloaded"; }
}

function renderAll() { renderBoard(); renderGameInfo(); renderTraining(); renderBenchmark(); renderNetwork(); }

els.newGame.addEventListener("click", newGame);
els.red.addEventListener("change", newGame);
els.yellow.addEventListener("change", newGame);
els.save.addEventListener("click", saveModel);
els.export.addEventListener("click", () => downloadJson(serializePolicy(model), `connect4-tactical-v3-${model.training.positions}.json`));
els.exportState.addEventListener("click", exportState);
els.import.addEventListener("change", () => {
  const file = els.import.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try { setModel(deserializePolicy(JSON.parse(reader.result))); els.status.textContent = `Warm-started V3 from ${model.training.positions.toLocaleString()} positions`; }
    catch (error) { alert(error.message); }
  };
  reader.readAsText(file);
});
els.reset.addEventListener("click", () => {
  if (!confirm("Reset the separate V3 model, replay, optimizer, and validation set?")) return;
  localStorage.removeItem(STORAGE_KEY);
  const worker = new Worker("./src/tactical-trainer-worker.js", { type: "module" });
  worker.postMessage({ type: "clear" });
  worker.onmessage = () => worker.terminate();
  model = createPolicyModel();
  trainingStatus = idleStatus();
  newGame();
  els.status.textContent = "V3 model and persistent curriculum state reset";
});
els.train.addEventListener("click", () => startTraining());
els.quick.addEventListener("click", () => startTraining(1000));
els.stop.addEventListener("click", () => trainingWorker?.postMessage({ type: "stop" }));
els.benchmark.addEventListener("click", runBenchmark);
els.layer.addEventListener("change", () => { els.neuron.value = 0; renderWeights(); });
els.neuron.addEventListener("input", renderWeights);
els.refresh.addEventListener("click", renderNetwork);

renderAll();
if (initialModelSource === "v2") els.status.textContent = `Warm-started V3 automatically from the saved V2 checkpoint at ${model.training.positions.toLocaleString()} positions`;
else if (initialModelSource === "v3") els.status.textContent = `Loaded saved V3 at ${model.training.positions.toLocaleString()} positions`;
scheduleAuto();
