import { applyMove, checkWin } from "./core.js";
import { chooseEngineRandomMove, choosePolicyMove, chooseRandomMove, chooseSolverMove, deserializePolicy, newBoard } from "./solver-core.js";

function move(kind,model,board,player) {
  if(kind==="ai") return choosePolicyMove(model,board,player);
  if(kind==="solver") return chooseSolverMove(board,player,5);
  if(kind==="engine-random") return chooseEngineRandomMove(board,player);
  return chooseRandomMove(board);
}
function game(model,aiPlayer,opponent) {
  const board=newBoard(); let player=1;
  for(let turn=0;turn<42;turn++) { const col=move(player===aiPlayer?"ai":opponent,model,board,player); if(col===null)return 0; const row=applyMove(board,col,player); if(checkWin(board,row,col,player))return player; player=-player; }
  return 0;
}
self.onmessage=e=>{
  if(e.data.type!=="benchmark")return;
  const model=deserializePolicy(e.data.model), games=e.data.games, opponent=e.data.opponent;
  const red={games:0,wins:0,draws:0,losses:0},yellow={games:0,wins:0,draws:0,losses:0};
  for(let i=0;i<games;i++){const aiPlayer=i%2===0?1:-1,bucket=aiPlayer===1?red:yellow,winner=game(model,aiPlayer,opponent);bucket.games++;if(winner===aiPlayer)bucket.wins++;else if(winner===0)bucket.draws++;else bucket.losses++;if(i%100===99)self.postMessage({type:"progress",completed:i+1,total:games});}
  for(const bucket of[red,yellow])bucket.score=(bucket.wins+.5*bucket.draws)/bucket.games;
  const wins=red.wins+yellow.wins,draws=red.draws+yellow.draws,losses=red.losses+yellow.losses;
  self.postMessage({type:"done",result:{games,opponent,wins,draws,losses,score:(wins+.5*draws)/games,red,yellow}});
};
