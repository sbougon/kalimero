export const ROWS = 6;
export const COLS = 7;
export const CELLS = ROWS * COLS;
export const INPUT_SIZE = CELLS;
export const DEFAULT_ARCHITECTURE_ID = "player-a";
export const MODEL_SCHEMA = "connect4-value-net-v2";
export const LEGACY_MODEL_SCHEMA = "connect4-value-net-v1";

export const ARCHITECTURE_PRESETS = [
  {
    id: "player-a",
    name: "Player A",
    description: "42 -> 64 -> 1",
    hiddenLayers: [64]
  },
  {
    id: "player-b",
    name: "Player B",
    description: "42 -> 128 -> 1",
    hiddenLayers: [128]
  },
  {
    id: "player-c",
    name: "Player C",
    description: "42 -> 256 -> 1",
    hiddenLayers: [256]
  },
  {
    id: "player-d",
    name: "Player D",
    description: "42 -> 128 -> 64 -> 1",
    hiddenLayers: [128, 64]
  }
];

export const TRAINING_CURRICULUM = [
  {
    label: "Random openings",
    end: 0.05,
    policy: "random",
    epsilon: 1
  },
  {
    label: "Win/block + random",
    end: 0.25,
    policy: "tactical-random",
    epsilon: 1
  },
  {
    label: "Full random",
    end: 0.35,
    policy: "random",
    epsilon: 1
  },
  {
    label: "Win/block + current NN",
    end: 0.65,
    policy: "tactical-nn",
    epsilon: 0.2
  },
  {
    label: "Full random refresh",
    end: 0.75,
    policy: "random",
    epsilon: 1
  },
  {
    label: "Win/block + current NN",
    end: 0.85,
    policy: "tactical-nn",
    epsilon: 0.08
  },
  {
    label: "Random with win/block",
    end: 0.95,
    policy: "tactical-random",
    epsilon: 1
  },
  {
    label: "Pure NN finish",
    end: 1,
    policy: "pure-nn",
    epsilon: 0
  }
];

export function randomWeight(fanIn = INPUT_SIZE) {
  const scale = Math.sqrt(2 / Math.max(1, fanIn));
  return (Math.random() * 2 - 1) * scale;
}

export function getArchitecturePreset(id = DEFAULT_ARCHITECTURE_ID) {
  return (
    ARCHITECTURE_PRESETS.find(preset => preset.id === id) ??
    ARCHITECTURE_PRESETS[0]
  );
}

export function createModel(architectureId = DEFAULT_ARCHITECTURE_ID) {
  const preset = getArchitecturePreset(architectureId);
  const model = {
    schema: MODEL_SCHEMA,
    architectureId: preset.id,
    architectureName: preset.name,
    inputSize: INPUT_SIZE,
    hiddenLayers: [...preset.hiddenLayers],
    layers: createLayers([INPUT_SIZE, ...preset.hiddenLayers, 1]),
    training: {
      games: 0,
      positions: 0,
      learningRate: 0.01,
      epsilon: 1,
      curriculumPhase: TRAINING_CURRICULUM[0].label
    },
    stats: {
      redWins: 0,
      yellowWins: 0,
      draws: 0,
      averageLoss: 0
    }
  };

  applyLegacyAliases(model);
  return model;
}

function createLayers(sizes) {
  const layers = [];

  for (let i = 1; i < sizes.length; i++) {
    const inputSize = sizes[i - 1];
    const outputSize = sizes[i];
    const weights = new Float32Array(inputSize * outputSize);
    const biases = new Float32Array(outputSize);

    for (let j = 0; j < weights.length; j++) {
      weights[j] = randomWeight(inputSize);
    }

    layers.push({
      inputSize,
      outputSize,
      weights,
      biases
    });
  }

  return layers;
}

function applyLegacyAliases(model) {
  model.hiddenSize = model.hiddenLayers[0] ?? 0;
  model.w1 = model.layers[0]?.weights ?? new Float32Array();
  model.b1 = model.layers[0]?.biases ?? new Float32Array();

  if (model.layers.length === 2 && model.layers[1].outputSize === 1) {
    model.w2 = model.layers[1].weights;
    model.b2 = model.layers[1].biases[0] ?? 0;
  } else {
    model.w2 = getFirstLayerOutgoingWeights(model);
    model.b2 = model.layers[model.layers.length - 1]?.biases[0] ?? 0;
  }
}

export function architectureLabel(model) {
  return `${model.inputSize} -> ${[...model.hiddenLayers, 1].join(" -> ")}`;
}

export function cloneBoard(board) {
  return new Int8Array(board);
}

export function resetBoard() {
  return new Int8Array(CELLS);
}

export function boardToArray(board) {
  return Array.from(board);
}

export function boardFromArray(values) {
  return new Int8Array(values);
}

export function getLegalMoves(board) {
  const moves = [];

  for (let col = 0; col < COLS; col++) {
    const topIdx = (ROWS - 1) * COLS + col;

    if (board[topIdx] === 0) {
      moves.push(col);
    }
  }

  return moves;
}

export function applyMove(board, col, player) {
  for (let row = 0; row < ROWS; row++) {
    const idx = row * COLS + col;

    if (board[idx] === 0) {
      board[idx] = player;
      return row;
    }
  }

  return -1;
}

export function countDirection(board, row, col, dr, dc, player) {
  let count = 0;
  let r = row + dr;
  let c = col + dc;

  while (r >= 0 && r < ROWS && c >= 0 && c < COLS) {
    if (board[r * COLS + c] !== player) {
      break;
    }

    count++;
    r += dr;
    c += dc;
  }

  return count;
}

export function checkWin(board, row, col, player) {
  if (row < 0 || col < 0) {
    return false;
  }

  const directions = [
    [1, 0],
    [0, 1],
    [1, 1],
    [1, -1]
  ];

  for (const [dr, dc] of directions) {
    const count =
      1 +
      countDirection(board, row, col, dr, dc, player) +
      countDirection(board, row, col, -dr, -dc, player);

    if (count >= 4) {
      return true;
    }
  }

  return false;
}

export function isDraw(board) {
  return getLegalMoves(board).length === 0;
}

export function getImmediateWinningMoves(board, player) {
  const winningMoves = [];

  for (const col of getLegalMoves(board)) {
    const nextBoard = cloneBoard(board);
    const row = applyMove(nextBoard, col, player);

    if (checkWin(nextBoard, row, col, player)) {
      winningMoves.push(col);
    }
  }

  return winningMoves;
}

export function encodeBoard(board, playerToMove) {
  const input = new Float32Array(CELLS);

  for (let i = 0; i < CELLS; i++) {
    if (board[i] === playerToMove) {
      input[i] = 1;
    } else if (board[i] === -playerToMove) {
      input[i] = -1;
    }
  }

  return input;
}

export function forward(model, input) {
  const activations = [input];
  let current = input;

  for (const layer of model.layers) {
    const next = new Float32Array(layer.outputSize);

    for (let out = 0; out < layer.outputSize; out++) {
      let sum = layer.biases[out];
      const offset = out * layer.inputSize;

      for (let i = 0; i < layer.inputSize; i++) {
        sum += current[i] * layer.weights[offset + i];
      }

      next[out] = Math.tanh(sum);
    }

    activations.push(next);
    current = next;
  }

  return {
    output: current[0],
    hidden: activations[1] ?? new Float32Array(),
    activations
  };
}

export function scoreMove(model, board, player, col) {
  const nextBoard = cloneBoard(board);
  const row = applyMove(nextBoard, col, player);

  if (row === -1) {
    return -Infinity;
  }

  if (checkWin(nextBoard, row, col, player)) {
    return 1;
  }

  const opponent = -player;

  if (getImmediateWinningMoves(nextBoard, opponent).length > 0) {
    return -1;
  }

  return rawNetworkScoreBoard(model, nextBoard, player);
}

export function rawNetworkScoreMove(model, board, player, col) {
  const nextBoard = cloneBoard(board);
  const row = applyMove(nextBoard, col, player);

  if (row === -1) {
    return -Infinity;
  }

  return rawNetworkScoreBoard(model, nextBoard, player);
}

function rawNetworkScoreBoard(model, board, player) {
  const opponent = -player;
  const input = encodeBoard(board, opponent);
  const opponentValue = forward(model, input).output;

  return -opponentValue;
}

export function getMoveScores(model, board, player, pure = false) {
  const scores = Array(COLS).fill(null);
  const scorer = pure ? rawNetworkScoreMove : scoreMove;

  for (const col of getLegalMoves(board)) {
    scores[col] = scorer(model, board, player, col);
  }

  return scores;
}

export function chooseMove(model, board, player, epsilon = 0) {
  const legalMoves = getLegalMoves(board);

  if (legalMoves.length === 0) {
    return null;
  }

  const winningMoves = getImmediateWinningMoves(board, player);

  if (winningMoves.length > 0) {
    return chooseCenterMost(winningMoves);
  }

  const opponentWinningMoves = getImmediateWinningMoves(board, -player);

  if (opponentWinningMoves.length > 0) {
    const safeBlockingMoves = legalMoves.filter(col => {
      const nextBoard = cloneBoard(board);
      applyMove(nextBoard, col, player);
      return getImmediateWinningMoves(nextBoard, -player).length === 0;
    });

    if (safeBlockingMoves.length > 0) {
      return chooseBestScoredMove(model, board, player, safeBlockingMoves);
    }

    return chooseCenterMost(opponentWinningMoves);
  }

  if (Math.random() < epsilon) {
    return chooseRandomMove(board);
  }

  return chooseBestScoredMove(model, board, player, legalMoves);
}

export function choosePureNeuralMove(model, board, player, epsilon = 0) {
  const legalMoves = getLegalMoves(board);

  if (legalMoves.length === 0) {
    return null;
  }

  if (Math.random() < epsilon) {
    return chooseRandomMove(board);
  }

  let bestCol = legalMoves[0];
  let bestScore = -Infinity;

  for (const col of legalMoves) {
    const score = rawNetworkScoreMove(model, board, player, col);

    if (score > bestScore) {
      bestScore = score;
      bestCol = col;
    }
  }

  return bestCol;
}

export function chooseTacticalRandomMove(board, player) {
  const winningMoves = getImmediateWinningMoves(board, player);

  if (winningMoves.length > 0) {
    return chooseCenterMost(winningMoves);
  }

  const opponentWinningMoves = getImmediateWinningMoves(board, -player);

  if (opponentWinningMoves.length > 0) {
    return chooseCenterMost(opponentWinningMoves);
  }

  return chooseRandomMove(board);
}

export function chooseRandomMove(board) {
  const legalMoves = getLegalMoves(board);

  if (legalMoves.length === 0) {
    return null;
  }

  return legalMoves[Math.floor(Math.random() * legalMoves.length)];
}

export function chooseTrainingMove(model, board, player, phase) {
  switch (phase.policy) {
    case "pure-nn":
      return choosePureNeuralMove(model, board, player, phase.epsilon ?? 0);
    case "tactical-nn":
      return chooseMove(model, board, player, phase.epsilon ?? 0);
    case "tactical-random":
      return chooseTacticalRandomMove(board, player);
    case "random":
    default:
      return chooseRandomMove(board);
  }
}

export function trainingPhaseForProgress(progress) {
  const clamped = Math.max(0, Math.min(1, progress));
  return TRAINING_CURRICULUM.find(phase => clamped < phase.end) ?? TRAINING_CURRICULUM.at(-1);
}

export function chooseCenterMost(moves) {
  let bestCol = moves[0];
  let bestDistance = Math.abs(bestCol - 3);

  for (const col of moves) {
    const distance = Math.abs(col - 3);

    if (distance < bestDistance) {
      bestCol = col;
      bestDistance = distance;
    }
  }

  return bestCol;
}

export function chooseBestScoredMove(model, board, player, moves) {
  let bestCol = moves[0];
  let bestScore = -Infinity;

  for (const col of moves) {
    const score = scoreMove(model, board, player, col);

    if (score > bestScore) {
      bestScore = score;
      bestCol = col;
    }
  }

  return bestCol;
}

export function targetForState(winner, statePlayer) {
  if (winner === 0) {
    return 0;
  }

  return winner === statePlayer ? 1 : -1;
}

export function trainOne(model, input, target, learningRate) {
  const { output, activations } = forward(model, input);
  const error = output - target;
  let delta = new Float32Array([error * (1 - output * output)]);
  const oldWeights = model.layers.map(layer => layer.weights.slice());

  for (let layerIndex = model.layers.length - 1; layerIndex >= 0; layerIndex--) {
    const layer = model.layers[layerIndex];
    const previousActivation = activations[layerIndex];

    for (let out = 0; out < layer.outputSize; out++) {
      const offset = out * layer.inputSize;

      for (let i = 0; i < layer.inputSize; i++) {
        layer.weights[offset + i] -= learningRate * delta[out] * previousActivation[i];
      }

      layer.biases[out] -= learningRate * delta[out];
    }

    if (layerIndex > 0) {
      const previousLayer = model.layers[layerIndex - 1];
      const previousDelta = new Float32Array(previousLayer.outputSize);
      const previousOutput = activations[layerIndex];

      for (let i = 0; i < previousLayer.outputSize; i++) {
        let sum = 0;

        for (let out = 0; out < layer.outputSize; out++) {
          sum += delta[out] * oldWeights[layerIndex][out * layer.inputSize + i];
        }

        previousDelta[i] = sum * (1 - previousOutput[i] * previousOutput[i]);
      }

      delta = previousDelta;
    }
  }

  syncLegacyScalarAliases(model);
  return error * error;
}

function syncLegacyScalarAliases(model) {
  model.b2 = model.layers.at(-1)?.biases[0] ?? 0;
}

export function epsilonForGame(gameNumber) {
  const start = 1;
  const end = 0.05;
  const decayGames = 200_000;
  const t = Math.min(gameNumber / decayGames, 1);

  return start + t * (end - start);
}

export function playTrainingGame(model, options = {}) {
  const board = resetBoard();
  const trajectory = [];
  const frames = options.captureFrames ? [boardToArray(board)] : null;
  const phase = options.phase ?? {
    label: "Legacy mixed training",
    policy: "tactical-nn",
    epsilon: options.epsilon ?? 1
  };
  let player = 1;
  let winner = 0;

  for (let turn = 0; turn < CELLS; turn++) {
    trajectory.push({
      input: encodeBoard(board, player),
      player
    });

    const col = chooseTrainingMove(model, board, player, phase);

    if (col === null) {
      break;
    }

    const row = applyMove(board, col, player);

    if (frames) {
      frames.push(boardToArray(board));
    }

    if (checkWin(board, row, col, player)) {
      winner = player;
      break;
    }

    player = -player;
  }

  let totalLoss = 0;

  for (const state of trajectory) {
    const target = targetForState(winner, state.player);
    totalLoss += trainOne(model, state.input, target, options.learningRate ?? 0.01);
  }

  return {
    winner,
    turns: trajectory.length,
    loss: trajectory.length ? totalLoss / trajectory.length : 0,
    frames,
    phaseLabel: phase.label
  };
}

export function serializeModel(model) {
  return {
    schema: MODEL_SCHEMA,
    savedAt: new Date().toISOString(),
    architectureId: model.architectureId,
    architectureName: model.architectureName,
    inputSize: model.inputSize,
    hiddenLayers: [...model.hiddenLayers],
    layers: model.layers.map(layer => ({
      inputSize: layer.inputSize,
      outputSize: layer.outputSize,
      weights: Array.from(layer.weights),
      biases: Array.from(layer.biases)
    })),
    training: { ...model.training },
    stats: { ...model.stats }
  };
}

export function validateModelData(data) {
  if (!data || (data.schema !== MODEL_SCHEMA && data.schema !== LEGACY_MODEL_SCHEMA)) {
    throw new Error("Invalid model schema");
  }

  if (data.inputSize !== INPUT_SIZE) {
    throw new Error("Invalid input size");
  }

  if (data.schema === LEGACY_MODEL_SCHEMA) {
    validateLegacyModelData(data);
    return;
  }

  if (!Array.isArray(data.hiddenLayers) || data.hiddenLayers.length === 0) {
    throw new Error("Invalid hidden layers");
  }

  if (!Array.isArray(data.layers) || data.layers.length !== data.hiddenLayers.length + 1) {
    throw new Error("Invalid layers");
  }

  let expectedInput = INPUT_SIZE;

  data.layers.forEach((layer, index) => {
    const expectedOutput =
      index < data.hiddenLayers.length ? data.hiddenLayers[index] : 1;

    if (layer.inputSize !== expectedInput || layer.outputSize !== expectedOutput) {
      throw new Error("Invalid layer shape");
    }

    if (
      !Array.isArray(layer.weights) ||
      layer.weights.length !== layer.inputSize * layer.outputSize
    ) {
      throw new Error("Invalid layer weights");
    }

    if (!Array.isArray(layer.biases) || layer.biases.length !== layer.outputSize) {
      throw new Error("Invalid layer biases");
    }

    expectedInput = layer.outputSize;
  });

  for (const layer of data.layers) {
    assertFiniteArray(layer.weights);
    assertFiniteArray(layer.biases);
  }
}

function validateLegacyModelData(data) {
  if (!Number.isInteger(data.hiddenSize) || data.hiddenSize < 1) {
    throw new Error("Invalid hidden size");
  }

  if (!Array.isArray(data.w1) || data.w1.length !== INPUT_SIZE * data.hiddenSize) {
    throw new Error("Invalid w1");
  }

  if (!Array.isArray(data.b1) || data.b1.length !== data.hiddenSize) {
    throw new Error("Invalid b1");
  }

  if (!Array.isArray(data.w2) || data.w2.length !== data.hiddenSize) {
    throw new Error("Invalid w2");
  }

  if (typeof data.b2 !== "number" || !Number.isFinite(data.b2)) {
    throw new Error("Invalid b2");
  }

  for (const arr of [data.w1, data.b1, data.w2]) {
    assertFiniteArray(arr);
  }
}

function assertFiniteArray(values) {
  for (const value of values) {
    if (!Number.isFinite(value)) {
      throw new Error("Model contains invalid number");
    }
  }
}

export function deserializeModel(data) {
  validateModelData(data);

  if (data.schema === LEGACY_MODEL_SCHEMA) {
    return deserializeLegacyModel(data);
  }

  const model = {
    schema: MODEL_SCHEMA,
    architectureId: data.architectureId ?? "custom",
    architectureName: data.architectureName ?? "Custom",
    inputSize: data.inputSize,
    hiddenLayers: [...data.hiddenLayers],
    layers: data.layers.map(layer => ({
      inputSize: layer.inputSize,
      outputSize: layer.outputSize,
      weights: new Float32Array(layer.weights),
      biases: new Float32Array(layer.biases)
    })),
    training: {
      games: data.training?.games ?? 0,
      positions: data.training?.positions ?? 0,
      learningRate: data.training?.learningRate ?? 0.01,
      epsilon: data.training?.epsilon ?? 1,
      curriculumPhase: data.training?.curriculumPhase ?? TRAINING_CURRICULUM[0].label
    },
    stats: {
      redWins: data.stats?.redWins ?? 0,
      yellowWins: data.stats?.yellowWins ?? 0,
      draws: data.stats?.draws ?? 0,
      averageLoss: data.stats?.averageLoss ?? 0
    }
  };

  applyLegacyAliases(model);
  return model;
}

function deserializeLegacyModel(data) {
  const outputBiases = new Float32Array(1);
  outputBiases[0] = data.b2;
  const model = {
    schema: MODEL_SCHEMA,
    architectureId: data.hiddenSize === 64 ? "player-a" : "legacy",
    architectureName: data.hiddenSize === 64 ? "Player A" : "Legacy",
    inputSize: data.inputSize,
    hiddenLayers: [data.hiddenSize],
    layers: [
      {
        inputSize: data.inputSize,
        outputSize: data.hiddenSize,
        weights: new Float32Array(data.w1),
        biases: new Float32Array(data.b1)
      },
      {
        inputSize: data.hiddenSize,
        outputSize: 1,
        weights: new Float32Array(data.w2),
        biases: outputBiases
      }
    ],
    training: {
      games: data.training?.games ?? 0,
      positions: data.training?.positions ?? 0,
      learningRate: data.training?.learningRate ?? 0.01,
      epsilon: data.training?.epsilon ?? 1,
      curriculumPhase: data.training?.curriculumPhase ?? TRAINING_CURRICULUM[0].label
    },
    stats: {
      redWins: data.stats?.redWins ?? 0,
      yellowWins: data.stats?.yellowWins ?? 0,
      draws: data.stats?.draws ?? 0,
      averageLoss: data.stats?.averageLoss ?? 0
    }
  };

  applyLegacyAliases(model);
  return model;
}

export function cloneModel(model) {
  const cloned = {
    schema: MODEL_SCHEMA,
    architectureId: model.architectureId,
    architectureName: model.architectureName,
    inputSize: model.inputSize,
    hiddenLayers: [...model.hiddenLayers],
    layers: model.layers.map(layer => ({
      inputSize: layer.inputSize,
      outputSize: layer.outputSize,
      weights: new Float32Array(layer.weights),
      biases: new Float32Array(layer.biases)
    })),
    training: { ...model.training },
    stats: { ...model.stats }
  };

  applyLegacyAliases(cloned);
  return cloned;
}

export function applyMetadata(model, summary) {
  model.training.games += summary.games;
  model.training.positions += summary.positions;
  model.training.epsilon = epsilonForGame(model.training.games);
  model.training.curriculumPhase = summary.phaseLabel ?? model.training.curriculumPhase;
  model.stats.redWins += summary.redWins;
  model.stats.yellowWins += summary.yellowWins;
  model.stats.draws += summary.draws;

  if (summary.games > 0) {
    model.stats.averageLoss =
      model.stats.averageLoss === 0
        ? summary.averageLoss
        : model.stats.averageLoss * 0.98 + summary.averageLoss * 0.02;
  }
}

export function averageModels(models) {
  if (models.length === 0) {
    throw new Error("Cannot average zero models");
  }

  const result = cloneModel(models[0].model);
  const totalGames = models.reduce((sum, item) => sum + item.games, 0) || models.length;

  for (const layer of result.layers) {
    layer.weights.fill(0);
    layer.biases.fill(0);
  }

  for (const item of models) {
    const weight = totalGames === models.length ? 1 / models.length : item.games / totalGames;

    assertSameArchitecture(result, item.model);

    for (let layerIndex = 0; layerIndex < result.layers.length; layerIndex++) {
      const resultLayer = result.layers[layerIndex];
      const sourceLayer = item.model.layers[layerIndex];

      for (let i = 0; i < resultLayer.weights.length; i++) {
        resultLayer.weights[i] += sourceLayer.weights[i] * weight;
      }

      for (let i = 0; i < resultLayer.biases.length; i++) {
        resultLayer.biases[i] += sourceLayer.biases[i] * weight;
      }
    }
  }

  applyLegacyAliases(result);
  return result;
}

function assertSameArchitecture(a, b) {
  if (
    a.inputSize !== b.inputSize ||
    a.layers.length !== b.layers.length ||
    a.layers.some(
      (layer, index) =>
        layer.inputSize !== b.layers[index].inputSize ||
        layer.outputSize !== b.layers[index].outputSize
    )
  ) {
    throw new Error("Cannot average models with different architectures");
  }
}

export function copyWeights(target, source) {
  assertSameArchitecture(target, source);

  for (let layerIndex = 0; layerIndex < target.layers.length; layerIndex++) {
    target.layers[layerIndex].weights = new Float32Array(source.layers[layerIndex].weights);
    target.layers[layerIndex].biases = new Float32Array(source.layers[layerIndex].biases);
  }

  applyLegacyAliases(target);
}

export function getFirstLayerOutgoingWeights(model) {
  const firstHiddenSize = model.hiddenLayers[0] ?? 0;
  const outgoing = new Float32Array(firstHiddenSize);
  const nextLayer = model.layers[1];

  if (!nextLayer) {
    return outgoing;
  }

  for (let hidden = 0; hidden < firstHiddenSize; hidden++) {
    let sum = 0;

    for (let out = 0; out < nextLayer.outputSize; out++) {
      sum += nextLayer.weights[out * nextLayer.inputSize + hidden];
    }

    outgoing[hidden] = sum / nextLayer.outputSize;
  }

  return outgoing;
}
