import assert from "node:assert/strict";
import { applyMove } from "../src/core.js";
import {
  POLICY_SCHEMA,
  createPolicyModel,
  deserializePolicy,
  newBoard,
  parameterCount,
  policyValueForward,
  serializePolicy,
  teacherTarget,
  valueTarget
} from "../src/solver-core.js";

const model=createPolicyModel();
const count=parameterCount(model);
assert.equal(model.schema,POLICY_SCHEMA);
assert.ok(count>=300000&&count<=600000,`parameter count ${count} is outside the V2 target`);

const board=newBoard();
for(let i=0;i<6;i++)applyMove(board,0,i%2?1:-1);
const output=policyValueForward(model,board,1);
assert.equal(output.probabilities[0],0,"full columns must be masked");
assert.ok(Math.abs(output.probabilities.reduce((a,b)=>a+b,0)-1)<1e-5,"legal policy must normalize");
assert.ok(output.value>=-1&&output.value<=1,"value head must be bounded");

const scores=[100,-20,0,10,null,-100,50];
const target=teacherTarget(scores);
assert.equal(target[4],0,"illegal actions must receive no target mass");
assert.ok(target.filter(x=>x>0).length>1,"solver target must be soft, not one-hot");
assert.ok(Math.abs(target.reduce((a,b)=>a+b,0)-1)<1e-5,"solver target must normalize");
assert.ok(valueTarget(scores)>0,"value target must follow the best legal action value");

const restored=deserializePolicy(serializePolicy(model));
assert.equal(restored.modelId,model.modelId,"serialization must preserve training identity");
assert.equal(parameterCount(restored),count,"serialization must preserve all parameters");
assert.equal(restored.stem.weights[17],model.stem.weights[17],"serialization must preserve exact float32 weights");

console.log(`V2 solver policy verified: ${count.toLocaleString()} parameters, soft targets, policy/value heads, compact round trip.`);
