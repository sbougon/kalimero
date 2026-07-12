import { ROWS, COLS, CELLS, applyMove, checkWin, getLegalMoves, resetBoard } from "./core.js";

export const POLICY_SCHEMA = "connect4-residual-policy-v4";
export const CHANNELS = 64;
export const RESIDUAL_BLOCKS = 4;
export const POLICY_CHANNELS = 2;
export const VALUE_HIDDEN = 64;

function randn() {
  const u=Math.max(Number.EPSILON,Math.random()),v=Math.random();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
}
function parameter(inputSize, outputSize, kernel=1, scale=Math.sqrt(2/(inputSize*kernel*kernel))) {
  const weights=new Float32Array(inputSize*outputSize*kernel*kernel);
  for(let i=0;i<weights.length;i++) weights[i]=randn()*scale;
  return {inputSize,outputSize,kernel,weights,biases:new Float32Array(outputSize)};
}
function trainingDefaults(){return{positions:0,updates:0,averageLoss:0,policyLoss:0,valueLoss:0,teacherAgreement:0,validationLoss:null,learningRate:0.001,solverDepth:0,replaySize:0,bestValidationLoss:null,plateauCount:0};}

export function createPolicyModel() {
  return {
    schema:POLICY_SCHEMA,
    modelId:`${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`,
    architecture:{rows:ROWS,cols:COLS,channels:CHANNELS,blocks:RESIDUAL_BLOCKS},
    stem:parameter(2,CHANNELS,3),
    blocks:Array.from({length:RESIDUAL_BLOCKS},()=>({conv1:parameter(CHANNELS,CHANNELS,3),conv2:parameter(CHANNELS,CHANNELS,3,0.02)})),
    policy:{conv:parameter(CHANNELS,POLICY_CHANNELS),dense:parameter(POLICY_CHANNELS*CELLS,COLS)},
    value:{conv:parameter(CHANNELS,1),hidden:parameter(CELLS,VALUE_HIDDEN),output:parameter(VALUE_HIDDEN,1,1,0.02)},
    training:trainingDefaults()
  };
}

export function encodePosition(board,player) {
  const x=new Float32Array(CELLS*2);
  for(let i=0;i<CELLS;i++){if(board[i]===player)x[i]=1;else if(board[i]===-player)x[CELLS+i]=1;}
  return x;
}

function convForward(layer,input) {
  const out=new Float32Array(layer.outputSize*CELLS),k=layer.kernel,pad=k>>1;
  for(let oc=0;oc<layer.outputSize;oc++) for(let r=0;r<ROWS;r++) for(let c=0;c<COLS;c++) {
    let sum=layer.biases[oc];
    for(let ic=0;ic<layer.inputSize;ic++) for(let kr=0;kr<k;kr++) for(let kc=0;kc<k;kc++) {
      const rr=r+kr-pad,cc=c+kc-pad;if(rr<0||rr>=ROWS||cc<0||cc>=COLS)continue;
      sum+=layer.weights[(((oc*layer.inputSize+ic)*k+kr)*k+kc)]*input[ic*CELLS+rr*COLS+cc];
    }
    out[oc*CELLS+r*COLS+c]=sum;
  }
  return out;
}
function denseForward(layer,input) {
  const out=new Float32Array(layer.outputSize);
  for(let o=0;o<layer.outputSize;o++){let sum=layer.biases[o],off=o*layer.inputSize;for(let i=0;i<layer.inputSize;i++)sum+=layer.weights[off+i]*input[i];out[o]=sum;}
  return out;
}
const relu=x=>{const y=new Float32Array(x.length);for(let i=0;i<x.length;i++)y[i]=Math.max(0,x[i]);return y;};

export function policyValueForward(model,board,player,withCache=false) {
  const input=encodePosition(board,player),stemZ=convForward(model.stem,input),stem=relu(stemZ),blocks=[];let trunk=stem;
  for(const block of model.blocks){const z1=convForward(block.conv1,trunk),a1=relu(z1),z2=convForward(block.conv2,a1),out=new Float32Array(z2.length);for(let i=0;i<out.length;i++)out[i]=Math.max(0,trunk[i]+z2[i]);blocks.push({input:trunk,z1,a1,z2,out});trunk=out;}
  const policyZ=convForward(model.policy.conv,trunk),policyA=relu(policyZ),logits=denseForward(model.policy.dense,policyA);
  const legal=getLegalMoves(board),probabilities=new Float32Array(COLS);
  if(legal.length){let max=-Infinity,total=0;for(const c of legal)max=Math.max(max,logits[c]);for(const c of legal){probabilities[c]=Math.exp(logits[c]-max);total+=probabilities[c];}for(const c of legal)probabilities[c]/=total;}
  const valueZ=convForward(model.value.conv,trunk),valueA=relu(valueZ),hiddenZ=denseForward(model.value.hidden,valueA),hidden=relu(hiddenZ),valueRaw=denseForward(model.value.output,hidden)[0],value=Math.tanh(valueRaw);
  const result={logits,probabilities,value};
  if(withCache)result.cache={input,stemZ,stem,blocks,trunk,policyZ,policyA,valueZ,valueA,hiddenZ,hidden,valueRaw};
  return result;
}
export const policyForward=(model,board,player)=>policyValueForward(model,board,player);

export function choosePolicyMove(model,board,player){const legal=getLegalMoves(board);if(!legal.length)return null;const p=policyForward(model,board,player).probabilities;let best=legal[0];for(const c of legal)if(p[c]>p[best])best=c;return best;}
function immediateWins(board,player){const wins=[];for(const col of getLegalMoves(board)){const copy=new Int8Array(board),row=applyMove(copy,col,player);if(checkWin(copy,row,col,player))wins.push(col);}return wins;}
export function chooseEngineRandomMove(board,player){const wins=immediateWins(board,player);if(wins.length)return wins[Math.floor(Math.random()*wins.length)];const blocks=immediateWins(board,-player);if(blocks.length)return blocks[Math.floor(Math.random()*blocks.length)];return chooseRandomMove(board);}
export function chooseRandomMove(board){const legal=getLegalMoves(board);return legal.length?legal[Math.floor(Math.random()*legal.length)]:null;}

const WINDOWS=(()=>{const a=[],dirs=[[1,0],[0,1],[1,1],[1,-1]];for(let r=0;r<ROWS;r++)for(let c=0;c<COLS;c++)for(const[dr,dc]of dirs){const er=r+3*dr,ec=c+3*dc;if(er>=0&&er<ROWS&&ec>=0&&ec<COLS)a.push([0,1,2,3].map(n=>(r+n*dr)*COLS+c+n*dc));}return a;})();
function heuristic(board,player){let s=0;for(let r=0;r<ROWS;r++)if(board[r*COLS+3]===player)s+=5;else if(board[r*COLS+3]===-player)s-=5;for(const w of WINDOWS){let m=0,t=0,e=0;for(const i of w){if(board[i]===player)m++;else if(board[i]===-player)t++;else e++;}if(!t)s+=m===3&&e===1?80:m===2&&e===2?12:m===1?1:0;if(!m)s-=t===3&&e===1?95:t===2&&e===2?14:t===1?1:0;}return s;}
function orderedMoves(board){const legal=getLegalMoves(board);return[3,2,4,1,5,0,6].filter(c=>legal.includes(c));}
function negamax(board,player,depth,alpha,beta,table){const legal=getLegalMoves(board);if(!legal.length)return 0;if(depth<=0)return heuristic(board,player);const key=`${depth}:${player}:${board.join("")}`;if(table.has(key))return table.get(key);let best=-Infinity;for(const col of orderedMoves(board)){const copy=new Int8Array(board),row=applyMove(copy,col,player);const score=checkWin(copy,row,col,player)?100000+depth:-negamax(copy,-player,depth-1,-beta,-alpha,table);best=Math.max(best,score);alpha=Math.max(alpha,score);if(alpha>=beta)break;}table.set(key,best);return best;}
export function solverScores(board,player,depth=7){const scores=Array(COLS).fill(null),table=new Map();for(const col of orderedMoves(board)){const copy=new Int8Array(board),row=applyMove(copy,col,player);scores[col]=checkWin(copy,row,col,player)?100000+depth:-negamax(copy,-player,depth-1,-Infinity,Infinity,table);}return scores;}
export function chooseSolverMove(board,player,depth=7){const s=solverScores(board,player,depth);let best=null;for(let c=0;c<COLS;c++)if(s[c]!==null&&(best===null||s[c]>s[best]))best=c;return best;}

export function scoreUtility(score){if(score===null)return-Infinity;if(score>10000)return 1;if(score< -10000)return-1;return Math.tanh(score/120);}
export function teacherTarget(scores,temperature=0.22){const target=new Float32Array(COLS);let max=-Infinity,total=0;const u=scores.map(scoreUtility);for(const x of u)max=Math.max(max,x);for(let c=0;c<COLS;c++)if(scores[c]!==null){target[c]=Math.exp((u[c]-max)/temperature);total+=target[c];}for(let c=0;c<COLS;c++)target[c]/=total;return target;}
export function valueTarget(scores){let best=-Infinity;for(const s of scores)if(s!==null)best=Math.max(best,scoreUtility(s));return best;}

function encodeFloats(values){const bytes=new Uint8Array(values.buffer,values.byteOffset,values.byteLength);let s="";for(let i=0;i<bytes.length;i+=0x8000)s+=String.fromCharCode(...bytes.subarray(i,i+0x8000));return btoa(s);}
function decodeFloats(value){if(Array.isArray(value))return new Float32Array(value);const s=atob(value),b=new Uint8Array(s.length);for(let i=0;i<s.length;i++)b[i]=s.charCodeAt(i);return new Float32Array(b.buffer);}
const packLayer=l=>({inputSize:l.inputSize,outputSize:l.outputSize,kernel:l.kernel,weights:encodeFloats(l.weights),biases:encodeFloats(l.biases)});
const unpackLayer=l=>({...l,weights:decodeFloats(l.weights),biases:decodeFloats(l.biases)});
export function serializePolicy(m){return{schema:POLICY_SCHEMA,modelId:m.modelId,architecture:{...m.architecture},training:{...m.training},stem:packLayer(m.stem),blocks:m.blocks.map(b=>({conv1:packLayer(b.conv1),conv2:packLayer(b.conv2)})),policy:{conv:packLayer(m.policy.conv),dense:packLayer(m.policy.dense)},value:{conv:packLayer(m.value.conv),hidden:packLayer(m.value.hidden),output:packLayer(m.value.output)}};}
export function deserializePolicy(raw){if(!raw||raw.schema!==POLICY_SCHEMA)throw new Error("Expected a v4 residual Connect Four policy/value model");return{schema:POLICY_SCHEMA,modelId:raw.modelId||`import-${Date.now().toString(36)}`,architecture:{...raw.architecture},training:{...trainingDefaults(),...raw.training},stem:unpackLayer(raw.stem),blocks:raw.blocks.map(b=>({conv1:unpackLayer(b.conv1),conv2:unpackLayer(b.conv2)})),policy:{conv:unpackLayer(raw.policy.conv),dense:unpackLayer(raw.policy.dense)},value:{conv:unpackLayer(raw.value.conv),hidden:unpackLayer(raw.value.hidden),output:unpackLayer(raw.value.output)}};}
export function modelLayers(model){return[model.stem,...model.blocks.flatMap(b=>[b.conv1,b.conv2]),model.policy.conv,model.policy.dense,model.value.conv,model.value.hidden,model.value.output];}
export function parameterCount(model){return modelLayers(model).reduce((n,l)=>n+l.weights.length+l.biases.length,0);}
export function newBoard(){return resetBoard();}
