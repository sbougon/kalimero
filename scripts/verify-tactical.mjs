import assert from "node:assert/strict";
import fs from "node:fs";
import { applyMove } from "../src/core.js";
import { createPolicyModel, newBoard } from "../src/solver-core.js";
import { chooseTacticalPolicyMove, immediateWinningMoves, mirrorSample } from "../src/tactical-core.js";

function play(board, moves) {
  let player = 1;
  for (const col of moves) {
    applyMove(board, col, player);
    player = -player;
  }
  return player;
}

const model = createPolicyModel();

const winningBoard = newBoard();
const winningPlayer = play(winningBoard, [0, 6, 1, 6, 2, 5]);
assert.deepEqual(immediateWinningMoves(winningBoard, winningPlayer), [3]);
assert.equal(chooseTacticalPolicyMove(model, winningBoard, winningPlayer), 3, "V3 must take an immediate win");

const blockingBoard = newBoard();
const blockingPlayer = play(blockingBoard, [6, 0, 6, 1, 5, 2]);
assert.deepEqual(immediateWinningMoves(blockingBoard, -blockingPlayer), [3]);
assert.equal(chooseTacticalPolicyMove(model, blockingBoard, blockingPlayer), 3, "V3 must block a single immediate loss");

const sample = { board: Int8Array.from({ length: 42 }, (_, i) => i), target: Float32Array.from([1, 2, 3, 4, 5, 6, 7]), scores: [1, 2, 3, 4, 5, 6, 7] };
const mirrored = mirrorSample(sample);
assert.deepEqual(Array.from(mirrored.board.slice(0, 7)), [6, 5, 4, 3, 2, 1, 0]);
assert.deepEqual(Array.from(mirrored.target), [7, 6, 5, 4, 3, 2, 1]);

const html = fs.readFileSync(new URL("../tactical-policy-nn.html", import.meta.url), "utf8");
const main = fs.readFileSync(new URL("../src/tactical-main.js", import.meta.url), "utf8");
const htmlIds = new Set([...html.matchAll(/id="([^"]+)"/g)].map(match => match[1]));
const requestedIds = [...main.matchAll(/\$\("([^"]+)"\)/g)].map(match => match[1]);
assert.deepEqual(requestedIds.filter(id => !htmlIds.has(id)), [], "V3 main must only request DOM IDs present in its app");

console.log("V3 tactical policy verified: immediate wins, forced blocks, and mirror augmentation.");
