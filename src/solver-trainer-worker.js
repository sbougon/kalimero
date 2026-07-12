import { applyMove, checkWin } from "./core.js";
import { chooseEngineRandomMove, choosePolicyMove, chooseRandomMove, chooseSolverMove, deserializePolicy, policyForward, serializePolicy, solverScores, teacherTarget, newBoard } from "./solver-core.js";

let stopped = false;
self.onmessage = event => {
  if (event.data.type === "stop") { stopped = true; return; }
  if (event.data.type !== "train") return;
  stopped = false;
  train(event.data).catch(error => self.postMessage({type:"error",message:error.message}));
};

function createAdam(model) {
  const layers=model.layers.map(layer => ({ mw:new Float32Array(layer.weights.length), vw:new Float32Array(layer.weights.length), mb:new Float32Array(layer.biases.length), vb:new Float32Array(layer.biases.length) }));
  layers.skip={m:new Float32Array(model.skipWeights.length),v:new Float32Array(model.skipWeights.length)};
  return layers;
}

function trainSample(model, board, player, target, adam, lr, step) {
  const result = policyForward(model, board, player);
  const deltas = new Array(model.layers.length);
  const outputDelta = new Float32Array(7);
  let loss = 0;
  for (let c=0;c<7;c++) { outputDelta[c] = result.probabilities[c] - target[c]; if (target[c]) loss -= target[c] * Math.log(Math.max(1e-8,result.probabilities[c])); }
  deltas[2] = outputDelta;
  for (let l=2;l>0;l--) {
    const layer=model.layers[l], previous=result.activations[l], delta=new Float32Array(layer.inputSize);
    for (let i=0;i<layer.inputSize;i++) { let sum=0; for(let o=0;o<layer.outputSize;o++) sum += layer.weights[o*layer.inputSize+i]*deltas[l][o]; delta[i]=previous[i]>0?sum:0; }
    deltas[l-1]=delta;
  }
  const b1=.9,b2=.999,eps=1e-8;
  for(let l=0;l<model.layers.length;l++) {
    const layer=model.layers[l], state=adam[l], previous=result.activations[l], delta=deltas[l];
    for(let o=0;o<layer.outputSize;o++) {
      const gradB=Math.max(-5,Math.min(5,delta[o])); state.mb[o]=b1*state.mb[o]+(1-b1)*gradB; state.vb[o]=b2*state.vb[o]+(1-b2)*gradB*gradB;
      layer.biases[o]-=lr*(state.mb[o]/(1-Math.pow(b1,step)))/(Math.sqrt(state.vb[o]/(1-Math.pow(b2,step)))+eps);
      const offset=o*layer.inputSize;
      for(let i=0;i<layer.inputSize;i++) { const idx=offset+i, grad=Math.max(-5,Math.min(5,delta[o]*previous[i])); state.mw[idx]=b1*state.mw[idx]+(1-b1)*grad; state.vw[idx]=b2*state.vw[idx]+(1-b2)*grad*grad; layer.weights[idx]-=lr*(state.mw[idx]/(1-Math.pow(b1,step)))/(Math.sqrt(state.vw[idx]/(1-Math.pow(b2,step)))+eps); }
    }
  }
  const skip=adam.skip, input=result.activations[0];
  for(let o=0;o<7;o++) for(let t=0;t<14;t++) { const idx=o*14+t,grad=outputDelta[o]*input[84+t];skip.m[idx]=b1*skip.m[idx]+(1-b1)*grad;skip.v[idx]=b2*skip.v[idx]+(1-b2)*grad*grad;model.skipWeights[idx]-=lr*(skip.m[idx]/(1-Math.pow(b1,step)))/(Math.sqrt(skip.v[idx]/(1-Math.pow(b2,step)))+eps); }
  let predicted=0, expected=0; for(let c=1;c<7;c++){if(result.probabilities[c]>result.probabilities[predicted])predicted=c;if(target[c]>target[expected])expected=c;}
  return {loss, correct:target[predicted]>0?1:0};
}

function randomPosition(model, depth) {
  for (;;) {
    const board=newBoard(); let player=1; const plies=Math.floor(Math.random()*32);
    const seekTactical = Math.random() < .7;
    let terminal=false;
    for(let turn=0;turn<(seekTactical?36:plies);turn++) {
      if(seekTactical && turn>=4 && hasTacticalChoice(board,player)) return {board,player};
      let col;
      const roll=Math.random(), trained=model.training.positions>1000;
      if(roll<(trained?.45:.05) && trained) col=choosePolicyMove(model,board,player);
      else if(roll<(trained?.8:.45)) col=chooseEngineRandomMove(board,player);
      else if(roll<(trained?.85:.52) && turn<18) col=chooseSolverMove(board,player,Math.max(2,depth-1));
      else col=chooseRandomMove(board);
      if(col===null){terminal=true;break;}
      const row=applyMove(board,col,player); if(checkWin(board,row,col,player)){terminal=true;break;} player=-player;
    }
    if(!terminal) return {board,player};
  }
}

function hasTacticalChoice(board, player) {
  for (const side of [player, -player]) {
    for (let col=0;col<7;col++) {
      const copy=new Int8Array(board), row=applyMove(copy,col,side);
      if(row>=0 && checkWin(copy,row,col,side)) return true;
    }
  }
  return false;
}

function mirror(board,target) {
  const mirrored=new Int8Array(42), mirroredTarget=new Float32Array(7);
  for(let r=0;r<6;r++) for(let c=0;c<7;c++) mirrored[r*7+(6-c)]=board[r*7+c];
  for(let c=0;c<7;c++) mirroredTarget[6-c]=target[c];
  return {board:mirrored,target:mirroredTarget};
}

async function train(message) {
  const model=deserializePolicy(message.model), adam=createAdam(model), count=message.positions, depth=message.depth, lr=message.learningRate;
  const samples=[];
  for(let n=0;n<count && !stopped;n++) {
    const sample=randomPosition(model,depth), scores=solverScores(sample.board,sample.player,depth);
    samples.push({...sample,target:teacherTarget(scores)});
    if(n%100===0 || n===count-1) { self.postMessage({type:"progress",stage:"labeling",completed:n+1,total:count,loss:0,agreement:0}); await new Promise(resolve=>setTimeout(resolve,0)); }
  }
  let lossSum=0, correct=0, completed=0, step=0;
  const epochs=4;
  for(let epoch=0;epoch<epochs && !stopped;epoch++) {
    for(let i=samples.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[samples[i],samples[j]]=[samples[j],samples[i]];}
    for(let n=0;n<samples.length && !stopped;n++) {
      const sample=samples[n];
      let result=trainSample(model,sample.board,sample.player,sample.target,adam,lr,++step); lossSum+=result.loss; correct+=result.correct; completed++;
      const flipped=mirror(sample.board,sample.target); result=trainSample(model,flipped.board,sample.player,flipped.target,adam,lr,++step); lossSum+=result.loss; correct+=result.correct; completed++;
      if(n%100===0 || n===samples.length-1) { self.postMessage({type:"progress",stage:`learning ${epoch+1}/${epochs}`,completed:epoch*samples.length+n+1,total:epochs*samples.length,loss:lossSum/completed,agreement:correct/completed}); await new Promise(resolve=>setTimeout(resolve,0)); }
    }
  }
  model.training.positions += samples.length;
  model.training.epochs += 1;
  model.training.averageLoss = completed?lossSum/completed:0;
  model.training.teacherAgreement = completed?correct/completed:0;
  model.training.solverDepth = depth;
  self.postMessage({type:"done",model:serializePolicy(model),stopped,completed});
}
