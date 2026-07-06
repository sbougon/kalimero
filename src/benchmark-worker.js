import {
  applyMove,
  checkWin,
  chooseMove,
  choosePureNeuralMove,
  chooseRandomMove,
  chooseTacticalRandomMove,
  deserializeModel,
  resetBoard
} from "./core.js";

function chooseBenchmarkMove(model, board, player, kind) {
  if (kind === "engine") {
    return chooseMove(model, board, player, 0);
  }

  if (kind === "pure") {
    return choosePureNeuralMove(model, board, player, 0);
  }

  if (kind === "tactical") {
    return chooseTacticalRandomMove(board, player);
  }

  return chooseRandomMove(board);
}

function scenarioPlayers(scenario) {
  switch (scenario) {
    case "pure-vs-random":
      return { testedKind: "pure", opponentKind: "random" };
    case "engine-vs-tactical":
      return { testedKind: "engine", opponentKind: "tactical" };
    case "engine-vs-random":
      return { testedKind: "engine", opponentKind: "random" };
    case "pure-vs-engine":
      return { testedKind: "pure", opponentKind: "engine" };
    case "pure-vs-tactical":
    default:
      return { testedKind: "pure", opponentKind: "tactical" };
  }
}

function playBenchmarkGame(model, testedPlayer, testedKind, opponentKind) {
  const board = resetBoard();
  let player = 1;

  for (let turn = 0; turn < 42; turn++) {
    const kind = player === testedPlayer ? testedKind : opponentKind;
    const col = chooseBenchmarkMove(model, board, player, kind);

    if (col === null) {
      return 0;
    }

    const row = applyMove(board, col, player);

    if (checkWin(board, row, col, player)) {
      return player;
    }

    player = -player;
  }

  return 0;
}

function runBenchmark(model, options) {
  const games = Math.max(2, Math.floor(options.games));
  const redGames = Math.floor(games / 2);
  const yellowGames = games - redGames;
  const scenario = options.scenario ?? "pure-vs-tactical";
  const { testedKind, opponentKind } = scenarioPlayers(scenario);
  const result = {
    games,
    redGames,
    yellowGames,
    scenario,
    testedKind,
    opponentKind,
    wins: 0,
    losses: 0,
    draws: 0,
    redWins: 0,
    redLosses: 0,
    redDraws: 0,
    yellowWins: 0,
    yellowLosses: 0,
    yellowDraws: 0
  };

  for (let i = 0; i < redGames; i++) {
    recordWinner(result, playBenchmarkGame(model, 1, testedKind, opponentKind), 1);
  }

  for (let i = 0; i < yellowGames; i++) {
    recordWinner(result, playBenchmarkGame(model, -1, testedKind, opponentKind), -1);
  }

  result.score = (result.wins + result.draws * 0.5) / result.games;
  result.winRate = result.wins / result.games;
  result.lossRate = result.losses / result.games;
  result.drawRate = result.draws / result.games;
  result.redScore = (result.redWins + result.redDraws * 0.5) / result.redGames;
  result.yellowScore =
    (result.yellowWins + result.yellowDraws * 0.5) / result.yellowGames;

  return result;
}

function recordWinner(result, winner, enginePlayer) {
  if (winner === 0) {
    result.draws++;

    if (enginePlayer === 1) {
      result.redDraws++;
    } else {
      result.yellowDraws++;
    }

    return;
  }

  const engineWon = winner === enginePlayer;

  if (engineWon) {
    result.wins++;

    if (enginePlayer === 1) {
      result.redWins++;
    } else {
      result.yellowWins++;
    }
  } else {
    result.losses++;

    if (enginePlayer === 1) {
      result.redLosses++;
    } else {
      result.yellowLosses++;
    }
  }
}

function playModelGame(redModel, yellowModel) {
  const board = resetBoard();
  let player = 1;

  for (let turn = 0; turn < 42; turn++) {
    const model = player === 1 ? redModel : yellowModel;
    const col = choosePureNeuralMove(model, board, player, 0);

    if (col === null) {
      return 0;
    }

    const row = applyMove(board, col, player);

    if (checkWin(board, row, col, player)) {
      return player;
    }

    player = -player;
  }

  return 0;
}

function runMatch(players, leftIndex, rightIndex, games) {
  const result = {
    leftIndex,
    rightIndex,
    games,
    leftWins: 0,
    rightWins: 0,
    draws: 0,
    leftScore: 0,
    rightScore: 0,
    winnerIndex: leftIndex,
    loserIndex: rightIndex
  };

  for (let i = 0; i < games; i++) {
    const leftIsRed = i % 2 === 0;
    const redModel = leftIsRed ? players[leftIndex].model : players[rightIndex].model;
    const yellowModel = leftIsRed ? players[rightIndex].model : players[leftIndex].model;
    const winner = playModelGame(redModel, yellowModel);
    const leftWon = (winner === 1 && leftIsRed) || (winner === -1 && !leftIsRed);
    const rightWon = (winner === 1 && !leftIsRed) || (winner === -1 && leftIsRed);

    if (leftWon) {
      result.leftWins++;
    } else if (rightWon) {
      result.rightWins++;
    } else {
      result.draws++;
    }
  }

  result.leftScore = (result.leftWins + result.draws * 0.5) / games;
  result.rightScore = (result.rightWins + result.draws * 0.5) / games;

  if (result.rightScore > result.leftScore) {
    result.winnerIndex = rightIndex;
    result.loserIndex = leftIndex;
  }

  return result;
}

function runTournament(rawModels, options) {
  const gamesPerMatch = Math.max(2, Math.floor(options.gamesPerMatch ?? 200));
  const players = rawModels.map((entry, index) => ({
    name: entry.name || `Player ${index + 1}`,
    model: deserializeModel(entry.model),
    wins: 0,
    losses: 0,
    draws: 0,
    score: 0,
    matchesWon: 0
  }));

  if (players.length !== 4) {
    throw new Error("Tournament requires exactly 4 models");
  }

  const semifinals = [
    runMatch(players, 0, 3, gamesPerMatch),
    runMatch(players, 1, 2, gamesPerMatch)
  ];
  const final = runMatch(
    players,
    semifinals[0].winnerIndex,
    semifinals[1].winnerIndex,
    gamesPerMatch
  );
  const thirdPlace = runMatch(
    players,
    semifinals[0].loserIndex,
    semifinals[1].loserIndex,
    gamesPerMatch
  );
  const matches = [...semifinals, final, thirdPlace];

  for (const match of matches) {
    const left = players[match.leftIndex];
    const right = players[match.rightIndex];
    left.wins += match.leftWins;
    left.losses += match.rightWins;
    left.draws += match.draws;
    right.wins += match.rightWins;
    right.losses += match.leftWins;
    right.draws += match.draws;
    left.score += match.leftWins + match.draws * 0.5;
    right.score += match.rightWins + match.draws * 0.5;
    players[match.winnerIndex].matchesWon++;
  }

  const standings = players
    .map((player, index) => ({
      index,
      name: player.name,
      wins: player.wins,
      losses: player.losses,
      draws: player.draws,
      score: player.score,
      scoreRate: player.score / (player.wins + player.losses + player.draws),
      matchesWon: player.matchesWon
    }))
    .sort(
      (a, b) =>
        b.matchesWon - a.matchesWon ||
        b.scoreRate - a.scoreRate ||
        b.wins - a.wins
    );

  return {
    gamesPerMatch,
    semifinals,
    final,
    thirdPlace,
    standings
  };
}

self.onmessage = event => {
  const message = event.data;

  if (message.type === "benchmark") {
    const model = deserializeModel(message.model);
    const result = runBenchmark(model, message.options);

    self.postMessage({
      type: "done",
      jobId: message.jobId,
      result
    });
    return;
  }

  if (message.type === "tournament") {
    try {
      const result = runTournament(message.models, message.options ?? {});

      self.postMessage({
        type: "tournament-done",
        jobId: message.jobId,
        result
      });
    } catch (err) {
      self.postMessage({
        type: "tournament-error",
        jobId: message.jobId,
        message: err.message
      });
    }
  }
};
