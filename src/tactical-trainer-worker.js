import { ROWS, COLS, CELLS, applyMove, checkWin, getLegalMoves } from "./core.js";
import {
  chooseEngineRandomMove,
  choosePolicyMove,
  chooseRandomMove,
  deserializePolicy,
  modelLayers,
  newBoard,
  policyValueForward,
  serializePolicy,
  solverScores,
  teacherTarget,
  valueTarget
} from "./solver-core.js";
import { forkMoves, hardTarget, immediateWinningMoves, mirrorSample } from "./tactical-core.js";

const DB_NAME = "connect4-tactical-training-v3";
const DB_VERSION = 1;
const STATE_KEY = "trainer";
const VALIDATION_SIZE = 384;
const DEFAULT_REPLAY = 100000;
const CURRICULUM = ["win", "block", "fork", "fork-prevention", "win", "block", "strategic", "strategic"];
let stopped = false;

self.onmessage = event => {
  if (event.data.type === "stop") stopped = true;
  if (event.data.type === "clear") clearState();
  if (event.data.type === "train") {
    stopped = false;
    train(event.data).catch(error => self.postMessage({ type: "error", message: error.stack || error.message }));
  }
};

const tick = () => new Promise(resolve => setTimeout(resolve, 0));

function openDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => request.result.createObjectStore("state");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function dbGet() {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("state", "readonly");
    const request = tx.objectStore("state").get(STATE_KEY);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    tx.oncomplete = () => db.close();
  });
}

async function dbPut(value) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("state", "readwrite");
    tx.objectStore("state").put(value, STATE_KEY);
    tx.oncomplete = () => { db.close(); resolve(); };
    tx.onerror = () => reject(tx.error);
  });
}

async function clearState() {
  const db = await openDb();
  const tx = db.transaction("state", "readwrite");
  tx.objectStore("state").delete(STATE_KEY);
  tx.oncomplete = () => { db.close(); self.postMessage({ type: "cleared" }); };
}

function emptyGrad(model) {
  return modelLayers(model).map(layer => ({
    weights: new Float32Array(layer.weights.length),
    biases: new Float32Array(layer.biases.length)
  }));
}

function createAdam(model) {
  return {
    step: 0,
    layers: modelLayers(model).map(layer => ({
      mw: new Float32Array(layer.weights.length),
      vw: new Float32Array(layer.weights.length),
      mb: new Float32Array(layer.biases.length),
      vb: new Float32Array(layer.biases.length)
    }))
  };
}

function denseBackward(layer, input, gradOut, grad) {
  const gradIn = new Float32Array(layer.inputSize);
  for (let out = 0; out < layer.outputSize; out++) {
    const delta = gradOut[out];
    grad.biases[out] += delta;
    const offset = out * layer.inputSize;
    for (let i = 0; i < layer.inputSize; i++) {
      grad.weights[offset + i] += delta * input[i];
      gradIn[i] += layer.weights[offset + i] * delta;
    }
  }
  return gradIn;
}

function convBackward(layer, input, gradOut, grad) {
  const gradIn = new Float32Array(layer.inputSize * CELLS);
  const kernel = layer.kernel;
  const pad = kernel >> 1;
  for (let oc = 0; oc < layer.outputSize; oc++) {
    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col < COLS; col++) {
        const delta = gradOut[oc * CELLS + row * COLS + col];
        grad.biases[oc] += delta;
        for (let ic = 0; ic < layer.inputSize; ic++) {
          for (let kr = 0; kr < kernel; kr++) {
            for (let kc = 0; kc < kernel; kc++) {
              const rr = row + kr - pad;
              const cc = col + kc - pad;
              if (rr < 0 || rr >= ROWS || cc < 0 || cc >= COLS) continue;
              const wi = ((oc * layer.inputSize + ic) * kernel + kr) * kernel + kc;
              const ii = ic * CELLS + rr * COLS + cc;
              grad.weights[wi] += delta * input[ii];
              gradIn[ii] += layer.weights[wi] * delta;
            }
          }
        }
      }
    }
  }
  return gradIn;
}

function trainExample(model, sample, grads) {
  const weight = sample.tactical ? 2 : 1;
  const forward = policyValueForward(model, sample.board, sample.player, true);
  const cache = forward.cache;
  const layers = modelLayers(model);
  const dLogits = new Float32Array(COLS);
  let policyLoss = 0;
  for (let col = 0; col < COLS; col++) {
    dLogits[col] = weight * (forward.probabilities[col] - sample.target[col]);
    if (sample.target[col]) policyLoss -= sample.target[col] * Math.log(Math.max(1e-8, forward.probabilities[col]));
  }

  let gradIn = denseBackward(model.policy.dense, cache.policyA, dLogits, grads[layers.indexOf(model.policy.dense)]);
  for (let i = 0; i < gradIn.length; i++) if (cache.policyZ[i] <= 0) gradIn[i] = 0;
  const trunkPolicy = convBackward(model.policy.conv, cache.trunk, gradIn, grads[layers.indexOf(model.policy.conv)]);

  const valueError = forward.value - sample.valueTarget;
  const valueLoss = valueError * valueError;
  let valueGrad = new Float32Array([weight * 0.5 * valueError * (1 - forward.value * forward.value)]);
  gradIn = denseBackward(model.value.output, cache.hidden, valueGrad, grads[layers.indexOf(model.value.output)]);
  for (let i = 0; i < gradIn.length; i++) if (cache.hiddenZ[i] <= 0) gradIn[i] = 0;
  gradIn = denseBackward(model.value.hidden, cache.valueA, gradIn, grads[layers.indexOf(model.value.hidden)]);
  for (let i = 0; i < gradIn.length; i++) if (cache.valueZ[i] <= 0) gradIn[i] = 0;
  const trunkValue = convBackward(model.value.conv, cache.trunk, gradIn, grads[layers.indexOf(model.value.conv)]);

  let trunkGrad = new Float32Array(trunkPolicy.length);
  for (let i = 0; i < trunkGrad.length; i++) trunkGrad[i] = trunkPolicy[i] + trunkValue[i];
  for (let blockIndex = model.blocks.length - 1; blockIndex >= 0; blockIndex--) {
    const block = model.blocks[blockIndex];
    const blockCache = cache.blocks[blockIndex];
    for (let i = 0; i < trunkGrad.length; i++) if (blockCache.out[i] <= 0) trunkGrad[i] = 0;
    const skip = new Float32Array(trunkGrad);
    const grad1 = convBackward(block.conv2, blockCache.a1, trunkGrad, grads[layers.indexOf(block.conv2)]);
    for (let i = 0; i < grad1.length; i++) if (blockCache.z1[i] <= 0) grad1[i] = 0;
    const through = convBackward(block.conv1, blockCache.input, grad1, grads[layers.indexOf(block.conv1)]);
    for (let i = 0; i < trunkGrad.length; i++) trunkGrad[i] = skip[i] + through[i];
  }
  for (let i = 0; i < trunkGrad.length; i++) if (cache.stemZ[i] <= 0) trunkGrad[i] = 0;
  convBackward(model.stem, cache.input, trunkGrad, grads[0]);

  let predicted = 0;
  let expected = 0;
  for (let col = 1; col < COLS; col++) {
    if (forward.probabilities[col] > forward.probabilities[predicted]) predicted = col;
    if (sample.target[col] > sample.target[expected]) expected = col;
  }
  return { policyLoss, valueLoss, correct: sample.tactical ? (sample.target[predicted] > 0 ? 1 : 0) : (predicted === expected ? 1 : 0) };
}

function adamUpdate(model, grads, adam, learningRate, batchSize) {
  const beta1 = 0.9;
  const beta2 = 0.999;
  adam.step++;
  const layers = modelLayers(model);
  for (let layerIndex = 0; layerIndex < layers.length; layerIndex++) {
    const params = layers[layerIndex];
    const grad = grads[layerIndex];
    const moments = adam.layers[layerIndex];
    for (const kind of ["weights", "biases"]) {
      const m = kind === "weights" ? moments.mw : moments.mb;
      const v = kind === "weights" ? moments.vw : moments.vb;
      const values = params[kind];
      const gradient = grad[kind];
      for (let i = 0; i < values.length; i++) {
        const delta = Math.max(-2, Math.min(2, gradient[i] / batchSize));
        m[i] = beta1 * m[i] + (1 - beta1) * delta;
        v[i] = beta2 * v[i] + (1 - beta2) * delta * delta;
        const correctedM = m[i] / (1 - Math.pow(beta1, adam.step));
        const correctedV = v[i] / (1 - Math.pow(beta2, adam.step));
        values[i] -= learningRate * correctedM / (Math.sqrt(correctedV) + 1e-8);
      }
    }
  }
}

function randomPosition(model) {
  const board = newBoard();
  let player = 1;
  const plies = 6 + Math.floor(Math.random() * 28);
  for (let turn = 0; turn < plies; turn++) {
    const roll = Math.random();
    const col = roll < 0.5 ? chooseRandomMove(board) : roll < 0.8 ? chooseEngineRandomMove(board, player) : choosePolicyMove(model, board, player);
    if (col === null) return null;
    const row = applyMove(board, col, player);
    if (checkWin(board, row, col, player)) return null;
    player = -player;
  }
  return { board, player };
}

function preventionMoves(board, player) {
  const legal = getLegalMoves(board);
  const safe = legal.filter(col => {
    const copy = new Int8Array(board);
    applyMove(copy, col, player);
    return immediateWinningMoves(copy, -player).length === 0 && forkMoves(copy, -player).length === 0;
  });
  return safe.length < legal.length ? safe : [];
}

function classify(position) {
  const wins = immediateWinningMoves(position.board, position.player);
  if (wins.length) return { source: "win", moves: wins, value: 1 };
  const threats = immediateWinningMoves(position.board, -position.player);
  if (threats.length === 1) return { source: "block", moves: threats };
  const forks = forkMoves(position.board, position.player);
  if (forks.length) return { source: "fork", moves: forks, value: 0.8 };
  if (forkMoves(position.board, -position.player).length) {
    const prevention = preventionMoves(position.board, position.player);
    if (prevention.length) return { source: "fork-prevention", moves: prevention };
  }
  return null;
}

function tacticalSample(position, kind, depth) {
  const scores = solverScores(position.board, position.player, Math.min(3, depth));
  return {
    ...position,
    scores,
    target: hardTarget(kind.moves),
    valueTarget: kind.value ?? valueTarget(scores),
    source: kind.source,
    tactical: true
  };
}

function strategicSample(position, depth) {
  const scores = solverScores(position.board, position.player, depth);
  return {
    ...position,
    scores,
    target: teacherTarget(scores, 0.1),
    valueTarget: valueTarget(scores),
    source: "strategic",
    tactical: false
  };
}

function generateSample(model, depth, index) {
  const wanted = CURRICULUM[index % CURRICULUM.length];
  for (let attempt = 0; attempt < 100; attempt++) {
    const position = randomPosition(model);
    if (!position) continue;
    const kind = classify(position);
    if (wanted === "strategic" && !kind) return strategicSample(position, depth);
    if (kind?.source === wanted) return tacticalSample(position, kind, depth);
  }
  for (;;) {
    const position = randomPosition(model);
    if (position) return strategicSample(position, depth);
  }
}

function reservoirAdd(replay, sample, seen, capacity) {
  if (replay.length < capacity) replay.push(sample);
  else {
    const index = Math.floor(Math.random() * seen);
    if (index < capacity) replay[index] = sample;
  }
}

function pickReplay(replay, tactical) {
  for (let attempt = 0; attempt < 12; attempt++) {
    const sample = replay[Math.floor(Math.random() * replay.length)];
    if (sample.tactical === tactical) return sample;
  }
  return replay[Math.floor(Math.random() * replay.length)];
}

function sampleBatch(replay, size) {
  const batch = [];
  for (let i = 0; i < size; i++) {
    const sample = pickReplay(replay, Math.random() < 0.7);
    batch.push(Math.random() < 0.5 ? mirrorSample(sample) : sample);
  }
  return batch;
}

function validationMetrics(model, validation) {
  let policyLoss = 0;
  let valueLoss = 0;
  let correct = 0;
  const categories = {};
  for (const sample of validation) {
    const forward = policyValueForward(model, sample.board, sample.player);
    for (let col = 0; col < COLS; col++) if (sample.target[col]) policyLoss -= sample.target[col] * Math.log(Math.max(1e-8, forward.probabilities[col]));
    valueLoss += (forward.value - sample.valueTarget) ** 2;
    let predicted = 0;
    let expected = 0;
    for (let col = 1; col < COLS; col++) {
      if (forward.probabilities[col] > forward.probabilities[predicted]) predicted = col;
      if (sample.target[col] > sample.target[expected]) expected = col;
    }
    const hit = sample.tactical ? (sample.target[predicted] > 0 ? 1 : 0) : (predicted === expected ? 1 : 0);
    correct += hit;
    const bucket = categories[sample.source] ||= { correct: 0, total: 0 };
    bucket.correct += hit;
    bucket.total++;
  }
  for (const bucket of Object.values(categories)) bucket.accuracy = bucket.total ? bucket.correct / bucket.total : null;
  return {
    loss: policyLoss / validation.length + 0.25 * valueLoss / validation.length,
    policyLoss: policyLoss / validation.length,
    valueLoss: valueLoss / validation.length,
    agreement: correct / validation.length,
    categories
  };
}

function applyTrainingMetadata(model, state, metrics, totals, depth) {
  model.training.positions = state.seen;
  model.training.updates = state.adam.step;
  model.training.averageLoss = totals.examples ? (totals.policy + 0.25 * totals.value) / totals.examples : 0;
  model.training.policyLoss = totals.examples ? totals.policy / totals.examples : 0;
  model.training.valueLoss = totals.examples ? totals.value / totals.examples : 0;
  model.training.teacherAgreement = totals.examples ? totals.correct / totals.examples : 0;
  model.training.validationAgreement = metrics?.agreement ?? null;
  model.training.validationLoss = metrics?.loss ?? null;
  model.training.learningRate = state.learningRate;
  model.training.solverDepth = depth;
  model.training.replaySize = state.replay.length;
  model.training.bestValidationLoss = state.bestLoss;
  model.training.plateauCount = state.plateau;
  model.training.generation = "tactical-curriculum-v3";
  model.training.tacticalValidation = metrics?.categories ?? null;
}

async function train(message) {
  const incoming = deserializePolicy(message.model);
  let state = await dbGet();
  if (!state || state.schema !== incoming.schema || state.modelId !== incoming.modelId || message.fresh) {
    state = {
      schema: incoming.schema,
      modelId: incoming.modelId,
      current: serializePolicy(incoming),
      best: serializePolicy(incoming),
      adam: createAdam(incoming),
      replay: [],
      validation: [],
      seen: incoming.training.positions || 0,
      bestLoss: Infinity,
      plateau: 0,
      learningRate: message.learningRate || 0.0005
    };
  }

  const model = deserializePolicy(state.current);
  const depth = Math.max(3, message.depth || 5);
  const count = Math.max(128, message.positions || 6000);
  const batchSize = Math.max(16, message.batchSize || 128);
  const replayCapacity = Math.max(10000, message.replayCapacity || DEFAULT_REPLAY);
  const totals = { policy: 0, value: 0, correct: 0, examples: 0 };

  while (state.validation.length < VALIDATION_SIZE && !stopped) {
    const index = state.validation.length;
    state.validation.push(generateSample(model, depth, index));
    self.postMessage({ type: "progress", stage: "building balanced validation", completed: index + 1, total: VALIDATION_SIZE, loss: 0, agreement: 0 });
    await tick();
  }

  for (let index = 0; index < count && !stopped; index++) {
    const sample = generateSample(model, depth, state.seen);
    state.seen++;
    reservoirAdd(state.replay, sample, state.seen, replayCapacity);
    self.postMessage({ type: "progress", stage: `curriculum ${sample.source}`, completed: index + 1, total: count, loss: totals.examples ? (totals.policy + 0.25 * totals.value) / totals.examples : 0, agreement: totals.examples ? totals.correct / totals.examples : 0 });
    await tick();

    if ((index + 1) % 128 === 0) {
      for (let update = 0; update < 10 && !stopped; update++) {
        const grads = emptyGrad(model);
        const batch = sampleBatch(state.replay, Math.min(batchSize, state.replay.length));
        for (const item of batch) {
          const result = trainExample(model, item, grads);
          totals.policy += result.policyLoss;
          totals.value += result.valueLoss;
          totals.correct += result.correct;
          totals.examples++;
        }
        adamUpdate(model, grads, state.adam, state.learningRate, batch.length);
        self.postMessage({ type: "progress", stage: "learning balanced replay", completed: index + 1, total: count, loss: (totals.policy + 0.25 * totals.value) / totals.examples, agreement: totals.correct / totals.examples });
        await tick();
      }
    }

    if ((index + 1) % 1024 === 0) {
      state.current = serializePolicy(model);
      await dbPut(state);
    }
  }

  if (!stopped) {
    for (let update = 0; update < 32 && !stopped; update++) {
      const grads = emptyGrad(model);
      const batch = sampleBatch(state.replay, Math.min(batchSize, state.replay.length));
      for (const item of batch) {
        const result = trainExample(model, item, grads);
        totals.policy += result.policyLoss;
        totals.value += result.valueLoss;
        totals.correct += result.correct;
        totals.examples++;
      }
      adamUpdate(model, grads, state.adam, state.learningRate, batch.length);
      self.postMessage({ type: "progress", stage: "final tactical consolidation", completed: update + 1, total: 32, loss: (totals.policy + 0.25 * totals.value) / totals.examples, agreement: totals.correct / totals.examples });
      await tick();
    }
  }

  const metrics = validationMetrics(model, state.validation);
  const improved = metrics.loss < state.bestLoss - 1e-4;
  if (improved) {
    state.bestLoss = metrics.loss;
    state.plateau = 0;
  } else {
    state.plateau++;
    if (state.plateau >= 2) {
      state.learningRate = Math.max(1e-5, state.learningRate * 0.5);
      state.plateau = 0;
    }
  }
  applyTrainingMetadata(model, state, metrics, totals, depth);
  state.current = serializePolicy(model);
  if (improved) state.best = state.current;
  await dbPut(state);

  const deployed = deserializePolicy(improved ? state.current : state.best);
  deployed.training = { ...model.training, bestValidationLoss: state.bestLoss };
  self.postMessage({
    type: "done",
    model: serializePolicy(deployed),
    stopped,
    completed: totals.examples,
    improved,
    validation: metrics,
    replaySize: state.replay.length,
    learningRate: state.learningRate
  });
}
