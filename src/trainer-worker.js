import {
  deserializeModel,
  playTrainingGame,
  serializeModel,
  trainingPhaseForProgress
} from "./core.js";

self.onmessage = event => {
  const message = event.data;

  if (message.type !== "train") {
    return;
  }

  const model = deserializeModel(message.model);
  const learningRate = message.learningRate;
  const summary = {
    games: 0,
    positions: 0,
    redWins: 0,
    yellowWins: 0,
    draws: 0,
    averageLoss: 0
  };
  let sampleFrames = null;
  let lastPhaseLabel = null;

  for (let game = 0; game < message.games; game++) {
    const totalRunGames = Math.max(1, message.totalRunGames ?? message.games);
    const runGameOffset = message.runGameOffset ?? game;
    const phase = trainingPhaseForProgress((runGameOffset + game) / totalRunGames);
    const captureFrames = game === message.games - 1;
    const result = playTrainingGame(model, {
      learningRate,
      phase,
      captureFrames
    });

    model.training.games++;
    model.training.positions += result.turns;
    model.training.epsilon = phase.epsilon ?? model.training.epsilon;
    model.training.learningRate = learningRate;
    model.training.curriculumPhase = phase.label;

    summary.games++;
    summary.positions += result.turns;
    summary.averageLoss += result.loss;
    lastPhaseLabel = phase.label;

    if (result.winner === 1) {
      summary.redWins++;
    } else if (result.winner === -1) {
      summary.yellowWins++;
    } else {
      summary.draws++;
    }

    if (captureFrames) {
      sampleFrames = result.frames;
    }
  }

  if (summary.games > 0) {
    summary.averageLoss /= summary.games;
  }

  summary.phaseLabel = lastPhaseLabel;

  self.postMessage({
    type: "done",
    jobId: message.jobId,
    model: serializeModel(model),
    summary,
    sampleFrames
  });
};
