import {
  ARCHITECTURE_PRESETS,
  CELLS,
  COLS,
  ROWS,
  applyMetadata,
  applyMove,
  architectureLabel,
  averageModels,
  boardFromArray,
  checkWin,
  choosePureNeuralMove,
  copyWeights,
  createModel,
  deserializeModel,
  getFirstLayerOutgoingWeights,
  getLegalMoves,
  getMoveScores,
  isDraw,
  resetBoard,
  serializeModel,
  TRAINING_CURRICULUM
} from "./core.js";

const STORAGE_KEY = "connect4.valueNet.v1";
const BENCHMARK_STORAGE_KEY = "connect4.valueNet.v1.benchmarks";
const SAVE_EVERY_GAMES = 100;

const els = {
  statusText: document.getElementById("statusText"),
  board: document.getElementById("board"),
  arrowRow: document.getElementById("arrowRow"),
  columnButtons: document.getElementById("columnButtons"),
  gameStatus: document.getElementById("gameStatus"),
  aiMoveBtn: document.getElementById("aiMoveBtn"),
  newGameBtn: document.getElementById("newGameBtn"),
  testModeSelect: document.getElementById("testModeSelect"),
  gameCountInput: document.getElementById("gameCountInput"),
  workerCountInput: document.getElementById("workerCountInput"),
  batchSizeInput: document.getElementById("batchSizeInput"),
  learningRateInput: document.getElementById("learningRateInput"),
  trainBtn: document.getElementById("trainBtn"),
  pauseBtn: document.getElementById("pauseBtn"),
  trainOneBtn: document.getElementById("trainOneBtn"),
  trainPresetBtn: document.getElementById("trainPresetBtn"),
  runBenchmarkBtn: document.getElementById("runBenchmarkBtn"),
  downloadBenchmarkBtn: document.getElementById("downloadBenchmarkBtn"),
  clearBenchmarkBtn: document.getElementById("clearBenchmarkBtn"),
  benchmarkScenarioSelect: document.getElementById("benchmarkScenarioSelect"),
  quickBenchmarkGamesInput: document.getElementById("quickBenchmarkGamesInput"),
  benchmarkIntervalInput: document.getElementById("benchmarkIntervalInput"),
  finalBenchmarkGamesInput: document.getElementById("finalBenchmarkGamesInput"),
  autoBenchmarkInput: document.getElementById("autoBenchmarkInput"),
  benchmarkCanvas: document.getElementById("benchmarkCanvas"),
  benchmarkStatus: document.getElementById("benchmarkStatus"),
  benchmarkStats: document.getElementById("benchmarkStats"),
  saveBtn: document.getElementById("saveBtn"),
  exportBtn: document.getElementById("exportBtn"),
  importInput: document.getElementById("importInput"),
  resetModelBtn: document.getElementById("resetModelBtn"),
  architectureSelect: document.getElementById("architectureSelect"),
  curriculumText: document.getElementById("curriculumText"),
  refreshWeightsBtn: document.getElementById("refreshWeightsBtn"),
  weightsMeta: document.getElementById("weightsMeta"),
  networkCanvas: document.getElementById("networkCanvas"),
  featureCanvas: document.getElementById("featureCanvas"),
  hiddenNeuronInput: document.getElementById("hiddenNeuronInput"),
  hiddenNeuronLabel: document.getElementById("hiddenNeuronLabel"),
  featureMeta: document.getElementById("featureMeta"),
  statsList: document.getElementById("statsList"),
  scoreList: document.getElementById("scoreList"),
  w1Canvas: document.getElementById("w1Canvas"),
  b1Canvas: document.getElementById("b1Canvas"),
  w2Canvas: document.getElementById("w2Canvas"),
  tournamentInputs: Array.from(document.querySelectorAll("[data-tournament-slot]")),
  tournamentGamesInput: document.getElementById("tournamentGamesInput"),
  runTournamentBtn: document.getElementById("runTournamentBtn"),
  clearTournamentBtn: document.getElementById("clearTournamentBtn"),
  tournamentStatus: document.getElementById("tournamentStatus"),
  tournamentBracket: document.getElementById("tournamentBracket"),
  tournamentStandings: document.getElementById("tournamentStandings")
};

let model = loadModel();
let gameBoard = resetBoard();
let gamePlayer = 1;
let gameOver = false;
let interactionLocked = false;
let activeArrowCol = null;
let activeArrowPlayer = null;
let turnTimerId = null;
let confettiTimerId = null;
let trainingRun = null;
let playbackFrames = [];
let playbackIndex = 0;
let playbackLastTs = 0;
let lastSavedAtGame = model.training.games;
let benchmarkHistory = loadBenchmarkHistory();
let benchmarkRun = null;
let lastBenchmarkAtGame = getLastBenchmarkGame();
let tournamentModels = [null, null, null, null];
let tournamentRun = null;

function saveModel() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(serializeModel(model)));
  lastSavedAtGame = model.training.games;
  setStatus(`Saved at ${model.training.games.toLocaleString()} games`);
}

function loadModel() {
  const json = localStorage.getItem(STORAGE_KEY);

  if (!json) {
    return createModel();
  }

  try {
    return deserializeModel(JSON.parse(json));
  } catch (err) {
    console.error("Failed to load model", err);
    return createModel();
  }
}

function loadBenchmarkHistory() {
  const json = localStorage.getItem(BENCHMARK_STORAGE_KEY);

  if (!json) {
    return [];
  }

  try {
    const parsed = JSON.parse(json);
    return Array.isArray(parsed) ? parsed.filter(isValidBenchmarkPoint) : [];
  } catch (err) {
    console.error("Failed to load benchmark history", err);
    return [];
  }
}

function isValidBenchmarkPoint(point) {
  return (
    point &&
    Number.isFinite(point.gamesTrained) &&
    Number.isFinite(point.score) &&
    Number.isFinite(point.games)
  );
}

function saveBenchmarkHistory() {
  localStorage.setItem(BENCHMARK_STORAGE_KEY, JSON.stringify(benchmarkHistory.slice(-400)));
}

function getLastBenchmarkGame() {
  if (benchmarkHistory.length === 0) {
    return model.training.games;
  }

  return benchmarkHistory[benchmarkHistory.length - 1].gamesTrained;
}

function resetModel() {
  pauseTraining();
  window.clearTimeout(turnTimerId);
  clearConfetti();
  localStorage.removeItem(STORAGE_KEY);
  localStorage.removeItem(BENCHMARK_STORAGE_KEY);
  model = createModel(getSelectedArchitectureId());
  benchmarkHistory = [];
  lastSavedAtGame = 0;
  lastBenchmarkAtGame = model.training.games;
  gameBoard = resetBoard();
  gamePlayer = 1;
  gameOver = false;
  interactionLocked = false;
  renderAll();
  scheduleAutoTurn();
  setStatus(`Model reset to ${model.architectureName}`);
}

function getSelectedArchitectureId() {
  return els.architectureSelect?.value || "player-a";
}

function populateArchitectureSelect() {
  if (!els.architectureSelect) {
    return;
  }

  els.architectureSelect.innerHTML = ARCHITECTURE_PRESETS.map(
    preset =>
      `<option value="${preset.id}">${preset.name}: ${preset.description}</option>`
  ).join("");
  els.architectureSelect.value = model.architectureId ?? "player-a";
}

function handleArchitectureChange() {
  if (trainingRun?.active) {
    els.architectureSelect.value = model.architectureId ?? "player-a";
    setStatus("Pause training before switching architecture");
    return;
  }

  resetModel();
}

function getModePlayers() {
  switch (els.testModeSelect.value) {
    case "ai-human":
      return { 1: "ai", "-1": "human" };
    case "ai-random":
      return { 1: "ai", "-1": "random" };
    case "random-ai":
      return { 1: "random", "-1": "ai" };
    case "ai-ai":
      return { 1: "ai", "-1": "ai" };
    case "human-ai":
    default:
      return { 1: "human", "-1": "ai" };
  }
}

function getPlayerKind(player = gamePlayer) {
  return getModePlayers()[player];
}

function playerLabel(player = gamePlayer) {
  return player === 1 ? "Red" : "Yellow";
}

function playerKindLabel(kind) {
  return kind === "ai" ? "AI" : kind === "random" ? "Random" : "Human";
}

function maybeSaveModel() {
  if (model.training.games - lastSavedAtGame >= SAVE_EVERY_GAMES) {
    saveModel();
  }
}

function exportModel() {
  const json = JSON.stringify(serializeModel(model), null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const slug = (model.architectureName ?? "model").toLowerCase().replace(/[^a-z0-9]+/g, "-");
  a.download = `connect4-${slug}-${model.training.games}-games.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function importModelFromFile(file) {
  const reader = new FileReader();

  reader.onload = () => {
    try {
      model = deserializeModel(JSON.parse(reader.result));
      benchmarkHistory = [];
      lastBenchmarkAtGame = model.training.games;
      saveBenchmarkHistory();
      saveModel();
      if (els.architectureSelect) {
        els.architectureSelect.value = model.architectureId ?? "player-a";
      }
      renderAll();
      setStatus(`Imported ${model.architectureName ?? "model"}`);
    } catch (err) {
      alert(`Invalid model file: ${err.message}`);
    }
  };

  reader.onerror = () => {
    alert("Could not read file");
  };

  reader.readAsText(file);
}

function setStatus(text) {
  els.statusText.textContent = text;
}

function formatPct(numerator, denominator) {
  if (!denominator) {
    return "0.0%";
  }

  return `${((numerator / denominator) * 100).toFixed(1)}%`;
}

function formatPercent(value) {
  if (!Number.isFinite(value)) {
    return "n/a";
  }

  return `${(value * 100).toFixed(1)}%`;
}

function scenarioLabel(pointOrScenario) {
  const scenario =
    typeof pointOrScenario === "string"
      ? pointOrScenario
      : pointOrScenario.scenario ?? legacyScenario(pointOrScenario);

  switch (scenario) {
    case "pure-vs-random":
      return "Pure NN vs random";
    case "engine-vs-tactical":
      return "Engine vs win/block";
    case "engine-vs-random":
      return "Engine vs random";
    case "pure-vs-engine":
      return "Pure NN vs engine";
    case "pure-vs-tactical":
    default:
      return "Pure NN vs win/block";
  }
}

function legacyScenario(point) {
  if (!point) {
    return "pure-vs-tactical";
  }

  if (point.opponent === "random") {
    return "engine-vs-random";
  }

  return "engine-vs-tactical";
}

function renderStats() {
  const totalResults = model.stats.redWins + model.stats.yellowWins + model.stats.draws;
  const rows = [
    ["Architecture", architectureLabel(model)],
    ["Training phase", model.training.curriculumPhase ?? "n/a"],
    ["Games trained", model.training.games.toLocaleString()],
    ["Positions trained", model.training.positions.toLocaleString()],
    ["Current epsilon", model.training.epsilon.toFixed(4)],
    ["Learning rate", model.training.learningRate.toString()],
    ["Average loss", model.stats.averageLoss.toFixed(5)],
    ["Red wins", formatPct(model.stats.redWins, totalResults)],
    ["Yellow wins", formatPct(model.stats.yellowWins, totalResults)],
    ["Draws", formatPct(model.stats.draws, totalResults)]
  ];

  els.statsList.innerHTML = rows
    .map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`)
    .join("");
}

function renderBoard(board = gameBoard, options = {}) {
  const html = [];
  const omitIndex = options.omitIndex ?? -1;

  for (let displayRow = ROWS - 1; displayRow >= 0; displayRow--) {
    for (let col = 0; col < COLS; col++) {
      const idx = displayRow * COLS + col;
      const value = idx === omitIndex ? 0 : board[idx];
      const cls = value === 1 ? " red" : value === -1 ? " yellow" : "";
      html.push(`<div class="cell${cls}" data-row="${displayRow}" data-col="${col}"></div>`);
    }
  }

  els.board.innerHTML = html.join("");
}

function renderArrowButtons() {
  els.arrowRow.innerHTML = "";
  const legalMoves = getLegalMoves(gameBoard);
  els.aiMoveBtn.disabled = interactionLocked || gameOver;
  const humanTurn = getPlayerKind() === "human";

  for (let col = 0; col < COLS; col++) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "column-arrow";
    btn.title = `Drop in column ${col + 1}`;
    btn.setAttribute("aria-label", `Drop in column ${col + 1}`);

    if (activeArrowCol === col) {
      btn.classList.add("ai-active");
      btn.classList.add(activeArrowPlayer === 1 ? "red-active" : "yellow-active");
    }

    btn.disabled =
      interactionLocked ||
      gameOver ||
      !humanTurn ||
      !legalMoves.includes(col);

    btn.addEventListener("click", () => playHumanMove(col));
    els.arrowRow.append(btn);
  }
}

function renderColumnButtons() {
  els.columnButtons.innerHTML = "";
  const legalMoves = getLegalMoves(gameBoard);

  for (let col = 0; col < COLS; col++) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = String(col + 1);
    btn.disabled =
      interactionLocked || gameOver || getPlayerKind() !== "human" || !legalMoves.includes(col);
    btn.addEventListener("click", () => playHumanMove(col));
    els.columnButtons.append(btn);
  }
}

function renderGameStatus() {
  if (gameOver) {
    return;
  }

  const kind = getPlayerKind();

  if (interactionLocked && kind !== "human") {
    els.gameStatus.textContent = `${playerLabel()} ${playerKindLabel(kind)} thinking`;
  } else {
    els.gameStatus.textContent = `${playerLabel()} ${playerKindLabel(kind)} to move`;
  }
}

function renderScores() {
  const scores = getMoveScores(model, gameBoard, gamePlayer, true);

  els.scoreList.innerHTML = scores
    .map((score, idx) => {
      const value = score === null ? "full" : score.toFixed(3);
      return `<div class="score-item">Col ${idx + 1}: ${value}</div>`;
    })
    .join("");
}

function drawWeightCanvas(canvas, values, columns, rows, scale = 1) {
  if (canvas.width !== columns) {
    canvas.width = columns;
  }

  if (canvas.height !== rows) {
    canvas.height = rows;
  }

  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(columns, rows);

  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    const [r, g, b] = weightRgb(value, scale);

    image.data[i * 4] = r;
    image.data[i * 4 + 1] = g;
    image.data[i * 4 + 2] = b;
    image.data[i * 4 + 3] = 255;
  }

  ctx.putImageData(image, 0, 0);
}

function renderWeights() {
  const firstHiddenSize = model.hiddenLayers[0] ?? 0;
  const selectedHidden = clampInt(els.hiddenNeuronInput.value, 0, Math.max(0, firstHiddenSize - 1));

  els.hiddenNeuronInput.max = String(Math.max(0, firstHiddenSize - 1));
  els.hiddenNeuronInput.value = String(selectedHidden);
  els.weightsMeta.textContent = `${model.architectureName ?? "Model"}: ${architectureLabel(model)}`;
  els.curriculumText.textContent = TRAINING_CURRICULUM
    .map((phase, index) => {
      const start = index === 0 ? 0 : TRAINING_CURRICULUM[index - 1].end;
      return `${Math.round(start * 100)}-${Math.round(phase.end * 100)}% ${phase.label}`;
    })
    .join(" | ");
  renderNetworkDiagram();
  renderHiddenNeuronInspector();
  drawWeightCanvas(els.w1Canvas, model.layers[0].weights, model.inputSize, firstHiddenSize, 1);
  drawWeightCanvas(els.b1Canvas, model.layers[0].biases, firstHiddenSize, 1, 1);
  drawWeightCanvas(els.w2Canvas, getFirstLayerOutgoingWeights(model), firstHiddenSize, 1, 1);
}

function weightShade(value, scale = 1) {
  const [r, g, b] = weightRgb(value, scale);
  return `rgb(${r}, ${g}, ${b})`;
}

function weightRgb(value, scale = 1) {
  const scaled = Math.max(-1, Math.min(1, value / scale));

  if (scaled >= 0) {
    const rb = Math.round(255 * (1 - scaled));
    const g = Math.round(255 - 120 * scaled);
    return [rb, g, rb];
  }

  const amount = -scaled;
  const gb = Math.round(255 * (1 - amount));
  const r = Math.round(255 - 35 * amount);
  return [r, gb, gb];
}

function renderNetworkDiagram() {
  const canvas = els.networkCanvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const selectedHidden = Number.parseInt(els.hiddenNeuronInput.value, 10) || 0;
  const firstHiddenSize = model.hiddenLayers[0] ?? 0;
  const hiddenCols = Math.ceil(Math.sqrt(firstHiddenSize));
  const hiddenRows = Math.ceil(firstHiddenSize / hiddenCols);
  const outgoing = getFirstLayerOutgoingWeights(model);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#171c20";
  ctx.fillRect(0, 0, width, height);

  const inputX = 78;
  const hiddenX = firstHiddenSize > 100 ? 250 : 290;
  const outputX = 452;
  const inputTop = 46;
  const inputSize = 12;
  const gap = 4;
  const hiddenTop = 32;
  const hiddenSize = Math.max(
    4,
    Math.min(9, Math.floor((height - hiddenTop - 48) / Math.max(1, hiddenRows)) - gap)
  );

  ctx.strokeStyle = "rgba(220, 226, 230, 0.16)";
  ctx.lineWidth = 1;

  for (let i = 0; i < CELLS; i += 3) {
    const row = Math.floor(i / COLS);
    const col = i % COLS;
    const x = inputX + col * (inputSize + gap) + inputSize / 2;
    const y = inputTop + (ROWS - 1 - row) * (inputSize + gap) + inputSize / 2;
    const hRow = Math.floor(selectedHidden / hiddenCols);
    const hCol = selectedHidden % hiddenCols;
    const hx = hiddenX + hCol * (hiddenSize + gap) + hiddenSize / 2;
    const hy = hiddenTop + hRow * (hiddenSize + gap) + hiddenSize / 2;
    const weight = model.layers[0].weights[selectedHidden * model.inputSize + i];

    ctx.strokeStyle = weight >= 0 ? "rgba(255,255,255,0.24)" : "rgba(0,0,0,0.55)";
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(hx, hy);
    ctx.stroke();
  }

  const selectedOutputWeight = outgoing[selectedHidden] ?? 0;
  const hRow = Math.floor(selectedHidden / hiddenCols);
  const hCol = selectedHidden % hiddenCols;
  const hx = hiddenX + hCol * (hiddenSize + gap) + hiddenSize / 2;
  const hy = hiddenTop + hRow * (hiddenSize + gap) + hiddenSize / 2;
  ctx.strokeStyle =
    selectedOutputWeight >= 0 ? "rgba(255,255,255,0.42)" : "rgba(0,0,0,0.7)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(hx, hy);
  ctx.lineTo(outputX, height / 2);
  ctx.stroke();

  for (let i = 0; i < CELLS; i++) {
    const row = Math.floor(i / COLS);
    const col = i % COLS;
    const x = inputX + col * (inputSize + gap);
    const y = inputTop + (ROWS - 1 - row) * (inputSize + gap);
    ctx.fillStyle = weightShade(model.layers[0].weights[selectedHidden * model.inputSize + i], 1);
    ctx.beginPath();
    ctx.arc(x + inputSize / 2, y + inputSize / 2, inputSize / 2, 0, Math.PI * 2);
    ctx.fill();
  }

  for (let i = 0; i < firstHiddenSize; i++) {
    const row = Math.floor(i / hiddenCols);
    const col = i % hiddenCols;
    const x = hiddenX + col * (hiddenSize + gap);
    const y = hiddenTop + row * (hiddenSize + gap);

    ctx.fillStyle = weightShade(outgoing[i], 1);
    ctx.beginPath();
    ctx.arc(x + hiddenSize / 2, y + hiddenSize / 2, hiddenSize / 2, 0, Math.PI * 2);
    ctx.fill();

    if (i === selectedHidden) {
      ctx.strokeStyle = "#f1c84b";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  ctx.fillStyle = weightShade(model.layers.at(-1).biases[0], 1);
  ctx.beginPath();
  ctx.arc(outputX, height / 2, 18, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#dfe5e8";
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText("42 board inputs", 34, 24);
  ctx.fillText(`${firstHiddenSize} first hidden`, 244, 24);
  ctx.fillText("1 value", 430, 24);
  ctx.fillStyle = "#9ca8b0";
  ctx.fillText("red = negative, white = neutral, green = positive", 104, height - 18);
}

function renderHiddenNeuronInspector() {
  const hidden = Number.parseInt(els.hiddenNeuronInput.value, 10) || 0;
  const start = hidden * model.inputSize;
  const values = model.layers[0].weights.slice(start, start + model.inputSize);
  const displayValues = new Float32Array(CELLS);
  const outgoing = getFirstLayerOutgoingWeights(model);

  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLS; col++) {
      displayValues[(ROWS - 1 - row) * COLS + col] = values[row * COLS + col];
    }
  }

  els.hiddenNeuronLabel.textContent = `Hidden ${hidden + 1}`;
  els.featureMeta.textContent =
    `Outgoing ${outgoing[hidden].toFixed(4)} | bias ${model.layers[0].biases[hidden].toFixed(4)}`;
  drawWeightCanvas(els.featureCanvas, displayValues, COLS, ROWS, 1);
}

function renderAll() {
  if (els.architectureSelect) {
    els.architectureSelect.value = model.architectureId ?? "player-a";
  }
  renderStats();
  renderBoard();
  renderArrowButtons();
  renderColumnButtons();
  renderGameStatus();
  renderScores();
  renderWeights();
  renderBenchmark();
  els.learningRateInput.value = model.training.learningRate;
}

function renderBenchmark() {
  renderBenchmarkGraph();
  renderBenchmarkStats();
}

function renderBenchmarkStats() {
  const primaryScenario = getPrimaryBenchmarkScenario();
  const primaryPoints = getScenarioPoints(primaryScenario);
  const last = primaryPoints[primaryPoints.length - 1];

  if (!last) {
    els.benchmarkStatus.textContent = benchmarkRun ? "Benchmark running" : "No benchmark yet";
    els.benchmarkStats.innerHTML = "";
    return;
  }

  const plateau = getPlateauState();
  els.benchmarkStatus.textContent = benchmarkRun
    ? "Benchmark running"
    : plateau.isPlateau
      ? "Plateau suspected"
      : `Last benchmark at ${last.gamesTrained.toLocaleString()} games`;

  const rows = getBenchmarkStatRows(last, plateau);

  els.benchmarkStats.innerHTML = rows
    .map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`)
    .join("");
}

function getBenchmarkStatRows(point, plateau = getPlateauState()) {
  return [
    ["Score", formatPercent(point.score)],
    ["Benchmark", scenarioLabel(point)],
    ["Bench games", point.games.toLocaleString()],
    ["Best score", formatPercent(plateau.bestScore)],
    ["Red score", formatPercent(point.redScore)],
    ["Yellow score", formatPercent(point.yellowScore)],
    [
      "Win / Draw / Loss",
      `${formatPercent(point.winRate)} / ${formatPercent(point.drawRate)} / ${formatPercent(point.lossRate)}`
    ],
    ["Flat checkpoints", `${plateau.flatCount} / ${plateau.patience}`]
  ];
}

function renderBenchmarkGraph() {
  const canvas = els.benchmarkCanvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const padLeft = 58;
  const padRight = 18;
  const padTop = 20;
  const padBottom = 42;
  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#171c20";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "#384149";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#9ca8b0";
  ctx.font = "12px system-ui, sans-serif";

  for (let i = 0; i <= 4; i++) {
    const y = padTop + (plotHeight * i) / 4;
    const score = 1 - i / 4;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(width - padRight, y);
    ctx.stroke();
    ctx.fillText(`${Math.round(score * 100)}%`, 12, y + 4);
  }

  ctx.fillText("training games", width / 2 - 36, height - 10);
  ctx.save();
  ctx.translate(14, height / 2 + 34);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("benchmark score", 0, 0);
  ctx.restore();

  const primaryScenario = getPrimaryBenchmarkScenario();
  const primaryPoints = getScenarioPoints(primaryScenario);
  const rollingPoints = rollingSeries(primaryPoints);
  const comparisonPoints =
    primaryScenario === "engine-vs-tactical" ? [] : getScenarioPoints("engine-vs-tactical");
  const allPoints = [...primaryPoints, ...comparisonPoints];

  if (allPoints.length === 0) {
    ctx.fillStyle = "#dfe5e8";
    ctx.fillText("Run a benchmark to start the quality graph", padLeft + 18, height / 2);
    return;
  }

  const minGame = Math.min(...allPoints.map(point => point.gamesTrained));
  const maxGame = Math.max(...allPoints.map(point => point.gamesTrained), minGame + 1);

  drawBenchmarkLine(ctx, primaryPoints, minGame, maxGame, padLeft, padTop, plotWidth, plotHeight, "#38a169", 1.5);
  drawBenchmarkLine(ctx, rollingPoints, minGame, maxGame, padLeft, padTop, plotWidth, plotHeight, "#ffffff", 3);
  drawBenchmarkLine(ctx, comparisonPoints, minGame, maxGame, padLeft, padTop, plotWidth, plotHeight, "#f1c84b", 1.5);

  for (const point of primaryPoints) {
    const x = padLeft + ((point.gamesTrained - minGame) / (maxGame - minGame)) * plotWidth;
    const y = padTop + (1 - point.score) * plotHeight;
    ctx.fillStyle = "#38a169";
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "#38a169";
  ctx.fillText(scenarioLabel(primaryScenario), padLeft + 10, padTop + 16);
  ctx.fillStyle = "#ffffff";
  ctx.fillText("5-point average", padLeft + 10, padTop + 32);
  if (comparisonPoints.length > 0) {
    ctx.fillStyle = "#f1c84b";
    ctx.fillText("Engine vs win/block", padLeft + 10, padTop + 48);
  }

  ctx.fillStyle = "#9ca8b0";
  ctx.fillText(minGame.toLocaleString(), padLeft, height - 24);
  ctx.fillText(maxGame.toLocaleString(), width - padRight - 78, height - 24);
}

function drawBenchmarkLine(ctx, points, minGame, maxGame, padLeft, padTop, plotWidth, plotHeight, color, width) {
  if (points.length === 0) {
    return;
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();

  points.forEach((point, index) => {
    const x = padLeft + ((point.gamesTrained - minGame) / (maxGame - minGame)) * plotWidth;
    const y = padTop + (1 - point.score) * plotHeight;

    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });

  ctx.stroke();
}

function getPrimaryBenchmarkScenario() {
  return els.benchmarkScenarioSelect.value;
}

function getScenarioPoints(scenario) {
  return benchmarkHistory
    .filter(point => (point.scenario ?? legacyScenario(point)) === scenario)
    .sort((a, b) => a.gamesTrained - b.gamesTrained);
}

function rollingSeries(points) {
  const result = [];
  const windowSize = 5;

  for (let i = 0; i < points.length; i++) {
    const start = Math.max(0, i - windowSize + 1);
    let total = 0;
    let count = 0;

    for (let j = start; j <= i; j++) {
      total += points[j].score;
      count++;
    }

    result.push({
      ...points[i],
      score: total / count
    });
  }

  return result;
}

function getPlateauState() {
  const minImprovement = 0.005;
  const patience = 10;
  let bestScore = 0;
  let bestIndex = -1;
  const points = getScenarioPoints(getPrimaryBenchmarkScenario());

  for (let i = 0; i < points.length; i++) {
    const score = rollingBenchmarkScore(points, i);

    if (score > bestScore + minImprovement) {
      bestScore = score;
      bestIndex = i;
    }
  }

  const flatCount = bestIndex === -1 ? 0 : points.length - 1 - bestIndex;

  return {
    bestScore,
    flatCount,
    patience,
    isPlateau: points.length >= patience + 1 && flatCount >= patience
  };
}

function rollingBenchmarkScore(points, index) {
  const windowSize = 5;
  const start = Math.max(0, index - windowSize + 1);
  let total = 0;
  let count = 0;

  for (let i = start; i <= index; i++) {
    total += points[i].score;
    count++;
  }

  return count ? total / count : 0;
}

async function runBenchmark(games, reason = "manual") {
  if (benchmarkRun) {
    return benchmarkRun.promise;
  }

  const primaryScenario = getPrimaryBenchmarkScenario();
  const points = [];
  points.push(await runBenchmarkScenario(primaryScenario, games, reason));

  if (primaryScenario !== "engine-vs-tactical") {
    points.push(await runBenchmarkScenario("engine-vs-tactical", games, `${reason}-comparison`));
  }

  return points;
}

function runBenchmarkScenario(scenario, games, reason) {
  if (benchmarkRun) {
    return benchmarkRun.promise;
  }

  const jobId = `${Date.now()}-${Math.random()}`;
  const worker = new Worker("./src/benchmark-worker.js", { type: "module" });
  const promise = new Promise((resolve, reject) => {
    worker.onmessage = event => {
      if (event.data.type !== "done" || event.data.jobId !== jobId) {
        return;
      }

      worker.terminate();
      const point = {
        ...event.data.result,
        reason,
        gamesTrained: model.training.games,
        createdAt: new Date().toISOString()
      };
      benchmarkHistory.push(point);
      benchmarkHistory = benchmarkHistory.slice(-400);
      if (scenario === getPrimaryBenchmarkScenario()) {
        lastBenchmarkAtGame = point.gamesTrained;
      }
      saveBenchmarkHistory();
      benchmarkRun = null;
      updateBenchmarkButtons();
      renderBenchmark();
      resolve(point);
    };

    worker.onerror = err => {
      worker.terminate();
      benchmarkRun = null;
      updateBenchmarkButtons();
      renderBenchmark();
      reject(err);
    };
  });

  benchmarkRun = { worker, promise };
  updateBenchmarkButtons();
  els.benchmarkStatus.textContent = `Benchmarking ${scenarioLabel(scenario)}: ${games.toLocaleString()} games`;
  worker.postMessage({
    type: "benchmark",
    jobId,
    model: serializeModel(model),
    options: {
      games,
      scenario
    }
  });

  return promise;
}

async function maybeRunAutoBenchmark(remainingTrainingGames) {
  if (!els.autoBenchmarkInput.checked || benchmarkRun) {
    return;
  }

  const interval = clampInt(els.benchmarkIntervalInput.value, 100, 10_000_000);

  if (model.training.games - lastBenchmarkAtGame < interval) {
    return;
  }

  const quickGames = clampInt(els.quickBenchmarkGamesInput.value, 100, 100_000);

  try {
    const points = await runBenchmark(quickGames, "checkpoint");
    const point = points[0];
    const plateau = getPlateauState();

    if (plateau.isPlateau && remainingTrainingGames > 0) {
      const finalGames = clampInt(els.finalBenchmarkGamesInput.value, 1000, 500_000);
      await runBenchmark(finalGames, "plateau-confirm");
      setStatus(
        `Plateau check complete at ${formatPercent(point.score)} after ${model.training.games.toLocaleString()} games`
      );
    }
  } catch (err) {
    console.error(err);
    setStatus(`Benchmark error: ${err.message}`);
  }
}

function clearBenchmarkHistory() {
  benchmarkHistory = [];
  lastBenchmarkAtGame = model.training.games;
  saveBenchmarkHistory();
  renderBenchmark();
  updateBenchmarkButtons();
}

function downloadBenchmarkImage() {
  const primaryScenario = getPrimaryBenchmarkScenario();
  const primaryPoints = getScenarioPoints(primaryScenario);
  const last = primaryPoints[primaryPoints.length - 1];

  if (!last) {
    return;
  }

  const scale = window.devicePixelRatio || 1;
  const sourceCanvas = els.benchmarkCanvas;
  const width = sourceCanvas.width;
  const height = 430;
  const canvas = document.createElement("canvas");
  canvas.width = width * scale;
  canvas.height = height * scale;

  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  ctx.fillStyle = "#171c20";
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = "#f3f5f6";
  ctx.font = "bold 20px system-ui, sans-serif";
  ctx.fillText("Benchmark", 20, 32);
  ctx.fillStyle = "#9ca8b0";
  ctx.font = "16px system-ui, sans-serif";
  ctx.fillText(
    `${scenarioLabel(primaryScenario)} at ${last.gamesTrained.toLocaleString()} training games`,
    20,
    58
  );

  ctx.drawImage(sourceCanvas, 20, 78, width - 40, 240);
  drawBenchmarkStatsForDownload(ctx, last, 20, 346, width - 40);

  const a = document.createElement("a");
  a.href = canvas.toDataURL("image/png");
  a.download = benchmarkFileName(last, primaryScenario);
  a.click();
}

function drawBenchmarkStatsForDownload(ctx, last, x, y, width) {
  const rows = getBenchmarkStatRows(last);
  const columns = 4;
  const columnWidth = width / columns;
  const rowGap = 44;

  ctx.textBaseline = "top";

  rows.forEach(([key, value], index) => {
    const col = index % columns;
    const row = Math.floor(index / columns);
    const cellX = x + col * columnWidth;
    const cellY = y + row * rowGap;

    ctx.fillStyle = "#9ca8b0";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText(key, cellX, cellY);
    ctx.fillStyle = "#f3f5f6";
    ctx.font = fontThatFits(ctx, value, "system-ui, sans-serif", 20, 13, columnWidth - 10);
    ctx.fillText(value, cellX, cellY + 18);
  });
}

function fontThatFits(ctx, text, family, maxSize, minSize, maxWidth) {
  for (let size = maxSize; size >= minSize; size--) {
    const font = `${size}px ${family}`;
    ctx.font = font;

    if (ctx.measureText(text).width <= maxWidth) {
      return font;
    }
  }

  return `${minSize}px ${family}`;
}

function benchmarkFileName(point, scenario) {
  const slug = scenario.replace(/-/g, "_");
  return `connect4-benchmark-${slug}-${point.gamesTrained}-games.png`;
}

function handleBenchmarkScenarioChange() {
  clearBenchmarkHistory();
  setStatus(`Benchmark reset for ${scenarioLabel(getPrimaryBenchmarkScenario())}`);
}

function updateBenchmarkButtons() {
  const active = Boolean(benchmarkRun);
  els.runBenchmarkBtn.disabled = active;
  els.clearBenchmarkBtn.disabled = active;
  els.downloadBenchmarkBtn.disabled =
    active || getScenarioPoints(getPrimaryBenchmarkScenario()).length === 0;
}

function loadTournamentModel(file, index) {
  const reader = new FileReader();

  reader.onload = () => {
    try {
      const loadedModel = deserializeModel(JSON.parse(reader.result));
      tournamentModels[index] = {
        name: file.name.replace(/\.json$/i, ""),
        model: loadedModel
      };
      renderTournament();
      setStatus(`Loaded tournament slot ${index + 1}: ${tournamentModels[index].name}`);
    } catch (err) {
      tournamentModels[index] = null;
      renderTournament();
      alert(`Invalid tournament model: ${err.message}`);
    }
  };

  reader.onerror = () => {
    alert("Could not read tournament model");
  };

  reader.readAsText(file);
}

function clearTournament() {
  if (tournamentRun) {
    tournamentRun.worker.terminate();
    tournamentRun = null;
  }

  tournamentModels = [null, null, null, null];

  for (const input of els.tournamentInputs) {
    input.value = "";
  }

  renderTournament();
  updateTournamentButtons();
}

function updateTournamentButtons() {
  const loadedCount = tournamentModels.filter(Boolean).length;
  const active = Boolean(tournamentRun);

  els.runTournamentBtn.disabled = active || loadedCount !== 4;
  els.clearTournamentBtn.disabled = active;
}

function renderTournament(result = null) {
  const loadedCount = tournamentModels.filter(Boolean).length;
  els.tournamentStatus.textContent = tournamentRun
    ? "Tournament running"
    : `${loadedCount} / 4 models loaded`;

  if (!result) {
    els.tournamentBracket.innerHTML = renderEmptyBracket();
    els.tournamentStandings.innerHTML = "";
    updateTournamentButtons();
    return;
  }

  const name = index => tournamentModels[index]?.name ?? `Slot ${index + 1}`;
  els.tournamentBracket.innerHTML = `
    <div class="bracket-round">
      <h3>Semifinals</h3>
      ${renderMatch(result.semifinals[0], name)}
      ${renderMatch(result.semifinals[1], name)}
    </div>
    <div class="bracket-round">
      <h3>Final</h3>
      ${renderMatch(result.final, name)}
      <h3>Third Place</h3>
      ${renderMatch(result.thirdPlace, name)}
    </div>
  `;
  els.tournamentStandings.innerHTML = result.standings
    .map(
      (row, index) => `
        <tr>
          <td>${index + 1}</td>
          <td>${escapeHtml(row.name)}</td>
          <td>${formatPercent(row.scoreRate)}</td>
          <td>${row.wins}/${row.draws}/${row.losses}</td>
          <td>${row.matchesWon}</td>
        </tr>
      `
    )
    .join("");
  updateTournamentButtons();
}

function renderEmptyBracket() {
  const slot = index => escapeHtml(tournamentModels[index]?.name ?? `Load slot ${index + 1}`);

  return `
    <div class="bracket-round">
      <h3>Semifinals</h3>
      <div class="match"><span>${slot(0)}</span><span>vs</span><span>${slot(3)}</span></div>
      <div class="match"><span>${slot(1)}</span><span>vs</span><span>${slot(2)}</span></div>
    </div>
    <div class="bracket-round">
      <h3>Final</h3>
      <div class="match"><span>Winner SF1</span><span>vs</span><span>Winner SF2</span></div>
      <h3>Third Place</h3>
      <div class="match"><span>Loser SF1</span><span>vs</span><span>Loser SF2</span></div>
    </div>
  `;
}

function renderMatch(match, nameForIndex) {
  const leftName = nameForIndex(match.leftIndex);
  const rightName = nameForIndex(match.rightIndex);
  const leftWinner = match.winnerIndex === match.leftIndex;
  const rightWinner = match.winnerIndex === match.rightIndex;

  return `
    <div class="match">
      <span class="${leftWinner ? "winner" : ""}">${escapeHtml(leftName)}</span>
      <span>${match.leftWins}-${match.draws}-${match.rightWins}</span>
      <span class="${rightWinner ? "winner" : ""}">${escapeHtml(rightName)}</span>
    </div>
  `;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function runTournament() {
  if (tournamentRun || tournamentModels.filter(Boolean).length !== 4) {
    return;
  }

  const jobId = `${Date.now()}-${Math.random()}`;
  const worker = new Worker("./src/benchmark-worker.js", { type: "module" });
  const gamesPerMatch = clampInt(els.tournamentGamesInput.value, 2, 100_000);

  tournamentRun = { worker };
  updateTournamentButtons();
  els.tournamentStatus.textContent = `Running ${gamesPerMatch.toLocaleString()} games per match`;

  worker.onmessage = event => {
    if (event.data.jobId !== jobId) {
      return;
    }

    worker.terminate();
    tournamentRun = null;

    if (event.data.type === "tournament-error") {
      els.tournamentStatus.textContent = `Tournament error: ${event.data.message}`;
      updateTournamentButtons();
      return;
    }

    els.tournamentStatus.textContent = `Complete: ${gamesPerMatch.toLocaleString()} games per match`;
    renderTournament(event.data.result);
  };

  worker.onerror = err => {
    worker.terminate();
    tournamentRun = null;
    els.tournamentStatus.textContent = `Tournament error: ${err.message}`;
    updateTournamentButtons();
  };

  worker.postMessage({
    type: "tournament",
    jobId,
    models: tournamentModels.map(entry => ({
      name: entry.name,
      model: serializeModel(entry.model)
    })),
    options: {
      gamesPerMatch
    }
  });
}

function finishMove(row, col, player) {
  if (checkWin(gameBoard, row, col, player)) {
    gameOver = true;
    els.gameStatus.textContent = `${player === 1 ? "Red" : "Yellow"} wins`;
    return "win";
  } else if (isDraw(gameBoard)) {
    gameOver = true;
    els.gameStatus.textContent = "Draw";
    return "draw";
  } else {
    gamePlayer = -gamePlayer;
    return "continue";
  }
}

function getCellElement(row, col) {
  return els.board.querySelector(`[data-row="${row}"][data-col="${col}"]`);
}

function delay(ms) {
  return new Promise(resolve => {
    window.setTimeout(resolve, ms);
  });
}

async function animateDrop(row, col, player) {
  const targetCell = getCellElement(row, col);

  if (!targetCell) {
    return;
  }

  const boardRect = els.board.getBoundingClientRect();
  const cellRect = targetCell.getBoundingClientRect();
  const piece = document.createElement("div");
  const top = cellRect.top - boardRect.top;
  const left = cellRect.left - boardRect.left;
  const startOffset = -cellRect.height - top - 8;

  piece.className = `falling-piece ${player === 1 ? "red" : "yellow"}`;
  piece.style.width = `${cellRect.width}px`;
  piece.style.height = `${cellRect.height}px`;
  piece.style.left = `${left}px`;
  piece.style.top = `${top}px`;
  els.board.append(piece);

  const animation = piece.animate(
    [
      { transform: `translateY(${startOffset}px)` },
      { transform: "translateY(0)" }
    ],
    {
      duration: 360 + (ROWS - 1 - row) * 42,
      easing: "cubic-bezier(.16,.8,.25,1)"
    }
  );

  await animation.finished;
  piece.remove();
}

function clearConfetti() {
  window.clearTimeout(confettiTimerId);
  els.board.querySelectorAll(".confetti-piece").forEach(piece => piece.remove());
}

function launchConfetti(player) {
  clearConfetti();

  const colors =
    player === 1
      ? ["#ff6b6b", "#ffd166", "#ffffff", "#38a169"]
      : ["#f1c84b", "#ffffff", "#38a169", "#e54b4b"];
  const boardRect = els.board.getBoundingClientRect();
  const count = 90;

  for (let i = 0; i < count; i++) {
    const piece = document.createElement("div");
    const x = (Math.random() - 0.5) * boardRect.width * 1.15;
    const y = boardRect.height + 40 + Math.random() * 120;
    const delay = Math.random() * 220;
    const duration = 1200 + Math.random() * 850;
    const rotation = (Math.random() > 0.5 ? 1 : -1) * (240 + Math.random() * 720);

    piece.className = "confetti-piece";
    piece.style.left = `${Math.random() * 100}%`;
    piece.style.background = colors[i % colors.length];
    piece.style.setProperty("--x", `${x}px`);
    piece.style.setProperty("--y", `${y}px`);
    piece.style.setProperty("--rotation", `${rotation}deg`);
    piece.style.setProperty("--duration", `${duration}ms`);
    piece.style.animationDelay = `${delay}ms`;

    if (i % 3 === 0) {
      piece.style.width = "6px";
      piece.style.height = "10px";
    }

    els.board.append(piece);
  }

  confettiTimerId = window.setTimeout(clearConfetti, 2600);
}

async function playMoveAnimated(col, player) {
  if (gameOver || interactionLocked || !getLegalMoves(gameBoard).includes(col)) {
    return false;
  }

  interactionLocked = true;
  activeArrowCol = getPlayerKind(player) === "human" ? null : col;
  activeArrowPlayer = getPlayerKind(player) === "human" ? null : player;
  renderArrowButtons();
  renderColumnButtons();

  const row = applyMove(gameBoard, col, player);

  if (row === -1) {
    interactionLocked = false;
    activeArrowCol = null;
    activeArrowPlayer = null;
    renderAll();
    return false;
  }

  renderBoard(gameBoard, { omitIndex: row * COLS + col });
  await animateDrop(row, col, player);
  const outcome = finishMove(row, col, player);
  interactionLocked = false;
  activeArrowCol = null;
  activeArrowPlayer = null;
  renderBoard();
  if (outcome === "win") {
    launchConfetti(player);
  }
  renderArrowButtons();
  renderColumnButtons();
  renderGameStatus();
  renderScores();
  return true;
}

async function playHumanMove(col) {
  if (getPlayerKind() !== "human") {
    return;
  }

  const moved = await playMoveAnimated(col, gamePlayer);

  if (moved && !gameOver) {
    scheduleAutoTurn();
  }
}

function chooseAutomaticMove(kind, player) {
  const legalMoves = getLegalMoves(gameBoard);

  if (legalMoves.length === 0) {
    return null;
  }

  if (kind === "random") {
    return legalMoves[Math.floor(Math.random() * legalMoves.length)];
  }

  return choosePureNeuralMove(model, gameBoard, player, 0);
}

function scheduleAutoTurn() {
  window.clearTimeout(turnTimerId);

  if (gameOver || getPlayerKind() === "human") {
    interactionLocked = false;
    renderAll();
    return;
  }

  const player = gamePlayer;
  const kind = getPlayerKind(player);
  interactionLocked = true;
  renderArrowButtons();
  renderColumnButtons();
  renderGameStatus();

  turnTimerId = window.setTimeout(async () => {
    if (gameOver || gamePlayer !== player || getPlayerKind(player) === "human") {
      interactionLocked = false;
      renderAll();
      return;
    }

    const col = chooseAutomaticMove(kind, player);

    if (col === null) {
      gameOver = true;
      interactionLocked = false;
      els.gameStatus.textContent = "Draw";
      renderArrowButtons();
      renderColumnButtons();
      return;
    }

    activeArrowCol = col;
    activeArrowPlayer = player;
    renderArrowButtons();
    await delay(260);
    interactionLocked = false;
    const moved = await playMoveAnimated(col, player);

    if (moved && !gameOver) {
      scheduleAutoTurn();
    }
  }, 2000);
}

async function playAiMove() {
  if (gameOver || interactionLocked) {
    return;
  }

  const col = choosePureNeuralMove(model, gameBoard, gamePlayer, 0);

  if (col === null) {
    gameOver = true;
    els.gameStatus.textContent = "Draw";
    renderArrowButtons();
    renderColumnButtons();
    return;
  }

  await playMoveAnimated(col, gamePlayer);
}

function newGame() {
  window.clearTimeout(turnTimerId);
  clearConfetti();
  gameBoard = resetBoard();
  gamePlayer = 1;
  gameOver = false;
  interactionLocked = false;
  activeArrowCol = null;
  activeArrowPlayer = null;
  playbackFrames = [];
  renderBoard();
  renderArrowButtons();
  renderColumnButtons();
  renderGameStatus();
  renderScores();
  scheduleAutoTurn();
}

function distributeGames(totalGames, workerCount, batchSize) {
  const activeWorkers = Math.min(workerCount, Math.ceil(totalGames / batchSize));
  const jobs = [];
  let remaining = totalGames;

  for (let i = 0; i < activeWorkers; i++) {
    const slotsLeft = activeWorkers - i;
    const games = Math.min(batchSize, Math.ceil(remaining / slotsLeft));
    jobs.push(games);
    remaining -= games;
  }

  return jobs;
}

function runWorkerJob(workerIndex, games, baseModel, learningRate, startGame, totalRunGames, runGameOffset) {
  return new Promise((resolve, reject) => {
    const worker = new Worker("./src/trainer-worker.js", { type: "module" });
    const jobId = `${Date.now()}-${workerIndex}-${Math.random()}`;
    let settled = false;
    const cancelJob = () => {
      if (settled) {
        return;
      }

      settled = true;
      worker.terminate();
      resolve({ cancelled: true });
    };

    worker.onmessage = event => {
      if (event.data.type === "done" && event.data.jobId === jobId) {
        settled = true;
        worker.terminate();
        trainingRun?.workers.delete(worker);
        trainingRun?.cancelJobs.delete(cancelJob);
        resolve(event.data);
      }
    };

    worker.onerror = err => {
      settled = true;
      worker.terminate();
      trainingRun?.workers.delete(worker);
      trainingRun?.cancelJobs.delete(cancelJob);
      reject(err);
    };

    worker.postMessage({
      type: "train",
      jobId,
      model: baseModel,
      games,
      learningRate,
      startGame,
      totalRunGames,
      runGameOffset
    });

    if (trainingRun) {
      trainingRun.workers.add(worker);
      trainingRun.cancelJobs.add(cancelJob);
    }
  });
}

async function trainGames(totalGames) {
  if (trainingRun?.active) {
    setStatus("Training is already running");
    return;
  }

  const workerCount = clampInt(els.workerCountInput.value, 1, 12);
  const batchSize = clampInt(els.batchSizeInput.value, 1, 1000);
  const learningRate = clampFloat(els.learningRateInput.value, 0.0001, 1);
  let remaining = Math.max(1, Math.floor(totalGames));
  const requestedGames = remaining;
  let completedInRun = 0;

  model.training.learningRate = learningRate;
  trainingRun = {
    active: true,
    workers: new Set(),
    cancelJobs: new Set()
  };
  updateTrainingButtons();
  setStatus(`Training ${remaining.toLocaleString()} games`);

  try {
    while (remaining > 0 && trainingRun?.active) {
      const roundGames = Math.min(remaining, workerCount * batchSize);
      const jobs = distributeGames(roundGames, workerCount, batchSize);
      const baseModel = serializeModel(model);
      const startGame = model.training.games;
      let jobOffset = 0;
      const results = await Promise.all(
        jobs.map((games, index) => {
          const runGameOffset = completedInRun + jobOffset;
          const jobStartGame = startGame + jobOffset;
          jobOffset += games;
          return runWorkerJob(
            index,
            games,
            baseModel,
            learningRate,
            jobStartGame,
            requestedGames,
            runGameOffset
          );
        })
      );

      if (!trainingRun?.active) {
        break;
      }

      const completedResults = results.filter(result => !result.cancelled);

      if (completedResults.length === 0) {
        break;
      }

      const trainedModels = completedResults.map(result => ({
        model: deserializeModel(result.model),
        games: result.summary.games
      }));
      const merged = averageModels(trainedModels);
      copyWeights(model, merged);

      const roundSummary = mergeSummaries(completedResults.map(result => result.summary));
      applyMetadata(model, roundSummary);
      remaining -= roundSummary.games;
      completedInRun += roundSummary.games;

      const sample = completedResults.find(result => result.sampleFrames?.length);
      if (sample) {
        setPlaybackFrames(sample.sampleFrames);
      }

      renderStats();
      renderWeights();
      renderScores();
      maybeSaveModel();
      setStatus(
        `Training: ${remaining.toLocaleString()} left, ${model.training.games.toLocaleString()} total`
      );
      await maybeRunAutoBenchmark(remaining);

      await new Promise(requestAnimationFrame);
    }
  } catch (err) {
    console.error(err);
    setStatus(`Training error: ${err.message}`);
  } finally {
    saveModel();
    if (trainingRun) {
      for (const worker of trainingRun.workers) {
        worker.terminate();
      }
    }
    trainingRun = null;
    updateTrainingButtons();
    renderAll();
  }
}

function mergeSummaries(summaries) {
  const merged = {
    games: 0,
    positions: 0,
    redWins: 0,
    yellowWins: 0,
    draws: 0,
    averageLoss: 0
  };
  let phaseLabel = null;

  for (const summary of summaries) {
    merged.games += summary.games;
    merged.positions += summary.positions;
    merged.redWins += summary.redWins;
    merged.yellowWins += summary.yellowWins;
    merged.draws += summary.draws;
    merged.averageLoss += summary.averageLoss * summary.games;
    phaseLabel = summary.phaseLabel ?? phaseLabel;
  }

  if (merged.games > 0) {
    merged.averageLoss /= merged.games;
  }

  merged.phaseLabel = phaseLabel;

  return merged;
}

function pauseTraining() {
  if (!trainingRun) {
    return;
  }

  trainingRun.active = false;

  for (const worker of trainingRun.workers) {
    worker.terminate();
  }

  for (const cancelJob of trainingRun.cancelJobs) {
    cancelJob();
  }

  setStatus("Pausing training");
}

function updateTrainingButtons() {
  const active = Boolean(trainingRun?.active);
  els.trainBtn.disabled = active;
  els.trainOneBtn.disabled = active;
  els.trainPresetBtn.disabled = active;
  els.pauseBtn.disabled = !active;
}

function setPlaybackFrames(frames) {
  playbackFrames = frames;
  playbackIndex = 0;
  playbackLastTs = 0;
}

function animatePlayback(ts) {
  if (playbackFrames.length > 0 && ts - playbackLastTs >= 16) {
    const board = boardFromArray(playbackFrames[playbackIndex]);
    renderBoard(board);
    playbackIndex++;
    playbackLastTs = ts;

    if (playbackIndex >= playbackFrames.length) {
      playbackFrames = [];
      renderBoard();
      renderArrowButtons();
    }
  }

  requestAnimationFrame(animatePlayback);
}

function clampInt(value, min, max) {
  const parsed = Number.parseInt(value, 10);

  if (!Number.isFinite(parsed)) {
    return min;
  }

  return Math.min(max, Math.max(min, parsed));
}

function clampFloat(value, min, max) {
  const parsed = Number.parseFloat(value);

  if (!Number.isFinite(parsed)) {
    return min;
  }

  return Math.min(max, Math.max(min, parsed));
}

els.trainBtn.addEventListener("click", () => {
  trainGames(clampInt(els.gameCountInput.value, 1, 1_000_000_000));
});

els.trainOneBtn.addEventListener("click", () => {
  trainGames(1);
});

els.trainPresetBtn.addEventListener("click", () => {
  els.gameCountInput.value = "100000";
  trainGames(100_000);
});

els.runBenchmarkBtn.addEventListener("click", async () => {
  const games = clampInt(els.finalBenchmarkGamesInput.value, 100, 500_000);

  try {
    await runBenchmark(games, "manual");
  } catch (err) {
    console.error(err);
    setStatus(`Benchmark error: ${err.message}`);
  }
});

els.clearBenchmarkBtn.addEventListener("click", clearBenchmarkHistory);
els.downloadBenchmarkBtn.addEventListener("click", downloadBenchmarkImage);
els.benchmarkScenarioSelect.addEventListener("change", handleBenchmarkScenarioChange);
els.pauseBtn.addEventListener("click", pauseTraining);
els.saveBtn.addEventListener("click", saveModel);
els.exportBtn.addEventListener("click", exportModel);
els.resetModelBtn.addEventListener("click", resetModel);
els.architectureSelect.addEventListener("change", handleArchitectureChange);
els.refreshWeightsBtn.addEventListener("click", renderWeights);
els.newGameBtn.addEventListener("click", newGame);
els.aiMoveBtn.addEventListener("click", playAiMove);
els.hiddenNeuronInput.addEventListener("input", renderWeights);
els.testModeSelect.addEventListener("change", newGame);

els.tournamentInputs.forEach(input => {
  input.addEventListener("change", event => {
    const index = Number.parseInt(input.dataset.tournamentSlot, 10);
    const file = event.target.files[0];

    if (file && Number.isInteger(index)) {
      loadTournamentModel(file, index);
    }
  });
});
els.runTournamentBtn.addEventListener("click", runTournament);
els.clearTournamentBtn.addEventListener("click", clearTournament);

els.importInput.addEventListener("change", event => {
  const file = event.target.files[0];

  if (file) {
    importModelFromFile(file);
  }

  event.target.value = "";
});

window.addEventListener("beforeunload", () => {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(serializeModel(model)));
});

populateArchitectureSelect();
renderAll();
renderTournament();
updateTrainingButtons();
updateBenchmarkButtons();
updateTournamentButtons();
scheduleAutoTurn();
requestAnimationFrame(animatePlayback);
