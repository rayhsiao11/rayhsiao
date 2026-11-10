<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>規則 × 機器學習 — 條件式預測</title>
<style>
  :root { --bg:#0B0E14; --panel:#12161F; --border:#1E2530; --text:#E6EAF2;
    --dim:#8B93A7; --faint:#5A6172; --up:#0ECB81; --down:#F6465D; --acc:#F0B90B; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font-family:'Noto Sans TC','PingFang TC','Microsoft JhengHei',system-ui,sans-serif; }
  .wrap { max-width:860px; margin:0 auto; padding:16px 12px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:12px;
    padding:14px; margin-top:12px; }
  .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
  label { font-size:12px; color:var(--dim); }
  select, input { padding:8px 9px; border-radius:7px; border:1px solid var(--border);
    background:var(--bg); color:var(--text); font-size:13px; font-family:inherit; }
  input { width:60px; text-align:center; }
  button { padding:11px 18px; border-radius:8px; border:1px solid var(--acc);
    background:var(--acc); color:#0B0E14; font-size:14px; font-weight:700;
    cursor:pointer; font-family:inherit; }
  button:disabled { opacity:.4; }
  .mono { font-family:ui-monospace,Menlo,Consolas,monospace; }
  .hint { font-size:11px; color:var(--faint); line-height:1.6; }
  #prog { height:8px; background:var(--bg); border-radius:4px; overflow:hidden;
    border:1px solid var(--border); margin-top:8px; }
  #progFill { height:100%; width:0%; background:var(--acc); transition:width .2s; }
  table { border-collapse:collapse; font-size:12.5px; width:100%; }
  th { color:var(--dim); font-size:11px; font-weight:600; text-align:right;
    padding:4px 8px; border-bottom:1px solid var(--border); }
  th:first-child, td:first-child { text-align:left; }
  td { padding:6px 8px; text-align:right; }
  tr.hl td { background:#F0B90B10; }
  .sec { font-size:12.5px; color:var(--dim); margin-top:16px; margin-bottom:6px; }
  .bar { height:9px; border-radius:3px; background:var(--acc); display:inline-block;
    vertical-align:middle; }
  .verdict { margin-top:10px; padding:10px 12px; border-radius:8px; font-size:13px;
    border:1px solid var(--acc); color:var(--acc); background:#F0B90B12; }
</style>
</head>
<body>
<div class="wrap">
  <div style="font-size:19px;font-weight:700;letter-spacing:1px">規則 × 機器學習
    <span class="hint">條件式預測：只在規則觸發的時點上訓練模型</span></div>

  <div class="card">
    <div class="row">
      <label>幣種 <select id="sym"><option>BTCUSDT</option><option>ETHUSDT</option></select></label>
      <label>期間 <select id="months"><option value="24">近 2 年</option>
        <option value="60" selected>近 5 年</option><option value="120">近 10 年</option></select></label>
      <label>押 <select id="hz"><option value="10" selected>10 分</option>
        <option value="30">30 分</option><option value="60">60 分</option></select></label>
    </div>
    <div class="row" style="margin-top:8px">
      <span class="hint">規則（產生候選事件）</span>
      <label>1H MA <input id="h1ma" type="number" value="3" min="2" max="200"></label>
      <label>BOLL <input id="bolln" type="number" value="99" min="5" max="300"></label>
      <label>×σ <input id="bollk" type="number" value="3" min="0.5" max="5" step="0.1"></label>
      <label>暴走門檻 <input id="filtTh" type="number" value="30" min="5" max="90">%</label>
    </div>
    <div class="row" style="margin-top:8px">
      <span class="hint">驗證</span>
      <label>折數 <input id="folds" type="number" value="5" min="3" max="10"></label>
      <label>gap <input id="gapD" type="number" value="1" min="0" max="30" step="1"> 天</label>
      <label>樹深 <input id="depth" type="number" value="4" min="2" max="12"></label>
      <label>葉最小樣本 <input id="leaf" type="number" value="40" min="5" max="500"></label>
      <label>森林樹數 <input id="ntree" type="number" value="60" min="10" max="300"></label>
    </div>
    <div class="row" style="margin-top:12px">
      <button id="runBtn">開始實驗</button>
      <span class="hint" id="status">會先用本機快取；沒有才下載（跟回測器共用，不會重抓）。</span>
    </div>
    <div id="prog"><div id="progFill"></div></div>
  </div>

  <div id="out"></div>

  <div class="card hint">
    做法：規則先圈出「極端急殺」的候選事件，只在這些時點上算特徵、訓練模型，用 TimeSeriesSplit
    K 折 ＋ gap（永遠用過去訓練、預測未來，訓練與測試之間留空白）驗證。此處基準線不是 50%，
    而是「規則本身無腦押漲」的勝率 — 模型要贏的是它。所有特徵只用進場當下以前的資料計算。
    僅供研究驗證，非投資建議。
  </div>
</div>

<script>
"use strict";
const HOSTS=["https://data-api.binance.vision","https://api.binance.com"];
const PAYOUT=0.8, BREAKEVEN=1/1.8, STAKE=20;
const $=id=>document.getElementById(id);
const DB_NAME="btc-backtest-cache", STORE="klines";
const mem={};

// ─── IndexedDB（與回測器共用）─────────────────────────
function idb(){ return new Promise((res,rej)=>{
  const r=indexedDB.open(DB_NAME,1);
  r.onupgradeneeded=()=>{ if(!r.result.objectStoreNames.contains(STORE)) r.result.createObjectStore(STORE); };
  r.onsuccess=()=>res(r.result); r.onerror=()=>rej(r.error); }); }
async function idbGet(k){ try{ const db=await idb();
  return await new Promise((res,rej)=>{ const q=db.transaction(STORE,"readonly").objectStore(STORE).get(k);
    q.onsuccess=()=>res(q.result||null); q.onerror=()=>rej(q.error); }); }catch(e){ return null; } }
async function idbPut(k,v){ try{ const db=await idb();
  await new Promise((res,rej)=>{ const q=db.transaction(STORE,"readwrite").objectStore(STORE).put(v,k);
    q.onsuccess=()=>res(); q.onerror=()=>rej(q.error); }); return true; }catch(e){ return false; } }

let throttleMs=220;
async function apiOnce(host,path){
  const r=await fetch(host+path);
  if(r.status===429||r.status===418){ const w=(+r.headers.get("retry-after")||5)*1000;
    throttleMs=Math.min(2000,throttleMs*2); const e=new Error("rate"); e.wait=w; throw e; }
  if(!r.ok) throw new Error("http"+r.status);
  const j=await r.json(); if(!Array.isArray(j)) throw new Error("bad"); return j;
}
async function api(path,tries=6){
  let last=null;
  for(let a=0;a<tries;a++){
    for(const h of HOSTS){
      try{ return await apiOnce(h,path); }
      catch(e){ last=e; if(e.wait){ $("status").textContent=`幣安限流，暫停 ${Math.round(e.wait/1000)} 秒…`;
        await new Promise(r=>setTimeout(r,e.wait)); } }
    }
    await new Promise(r=>setTimeout(r,400*Math.pow(2,a)));
  }
  throw last||new Error("fetch failed");
}
async function loadData(sym,months){
  const key=sym+"_"+months;
  if(mem[key]) return mem[key];
  const st=await idbGet(key);
  if(st && st.t && st.t.length>5000){
    const d={t:Array.from(st.t), c:Array.from(st.c)}; mem[key]=d;
    $("status").textContent=`本機快取 ${d.t.length.toLocaleString()} 根，免下載`;
    $("progFill").style.width="100%"; return d;
  }
  const end=Date.now()-2*86400e3, start=end-months*30*86400e3, CHUNK=1000*60000;
  const starts=[]; for(let c=start;c<end;c+=CHUNK) starts.push(Math.floor(c));
  const parts=new Array(starts.length).fill(null); let done=0,failed=0;
  for(let b=0;b<starts.length;b+=4){
    const idxs=[]; for(let k=b;k<Math.min(b+4,starts.length);k++) idxs.push(k);
    await Promise.all(idxs.map(async k=>{
      try{ parts[k]=await api(`/api/v3/klines?symbol=${sym}&interval=1m&startTime=${starts[k]}&endTime=${starts[k]+CHUNK-1}&limit=1000`); }
      catch(e){ parts[k]=[]; failed++; } }));
    done+=idxs.length;
    $("progFill").style.width=(done/starts.length*100).toFixed(1)+"%";
    if(!/限流/.test($("status").textContent))
      $("status").textContent=`下載中… ${done}/${starts.length} 段${failed?`（失敗 ${failed}）`:""}`;
    await new Promise(r=>setTimeout(r,throttleMs));
  }
  const T=[],C=[]; let lt=-1;
  for(const raw of parts){ if(!raw) continue; for(const k of raw){ if(k[0]>lt){ T.push(k[0]); C.push(+k[4]); lt=k[0]; } } }
  if(failed>starts.length*0.02) throw new Error(`有 ${failed} 段抓不到，請稍後重試`);
  const d={t:T,c:C}; mem[key]=d;
  await idbPut(key,{t:new Float64Array(T), c:new Float64Array(C), ts:Date.now()});
  return d;
}

// ─── 指標 ───────────────────────────────────────────────
function sma(a,n){ const o=new Array(a.length).fill(null); let s=0;
  for(let i=0;i<a.length;i++){ s+=a[i]; if(i>=n) s-=a[i-n]; if(i>=n-1) o[i]=s/n; } return o; }
function rollStd(a,n){ const o=new Array(a.length).fill(null); let s=0,s2=0;
  for(let i=0;i<a.length;i++){ s+=a[i]; s2+=a[i]*a[i]; if(i>=n){ s-=a[i-n]; s2-=a[i-n]*a[i-n]; }
    if(i>=n-1){ const m=s/n; o[i]=Math.sqrt(Math.max(0,s2/n-m*m)); } } return o; }
function ema(a,n){ const k=2/(n+1), o=new Array(a.length).fill(null); let p=null;
  for(let i=0;i<a.length;i++){ p=p===null?a[i]:a[i]*k+p*(1-k); if(i>=n-1) o[i]=p; } return o; }
function rsiArr(v,n){ const o=new Array(v.length).fill(null); let ag=0,al=0;
  for(let i=1;i<v.length;i++){ const d=v[i]-v[i-1], g=Math.max(d,0), l=Math.max(-d,0);
    if(i<=n){ ag+=g/n; al+=l/n; if(i===n) o[i]=100-100/(1+ag/(al||1e-9)); }
    else { ag=(ag*(n-1)+g)/n; al=(al*(n-1)+l)/n; o[i]=100-100/(1+ag/(al||1e-9)); } }
  return o; }
function trendMap(T,C,bucketMin,maN){
  const ms=bucketMin*60000, bT=[], bC=[];
  for(let i=0;i<T.length;i++){ const b=Math.floor(T[i]/ms)*ms;
    if(!bT.length||bT[bT.length-1]!==b){ bT.push(b); bC.push(C[i]); } else bC[bC.length-1]=C[i]; }
  const bMA=sma(bC,maN), keys=[], vals=[];
  for(let j=0;j<bT.length;j++){ if(bMA[j]===null) continue;
    keys.push(bT[j]+ms); vals.push({d:bC[j]>bMA[j]?1:-1, gap:bC[j]/bMA[j]-1}); }
  const out=new Array(T.length).fill(null); let ki=0,cur=null;
  for(let i=0;i<T.length;i++){ while(ki<keys.length&&keys[ki]<=T[i]){ cur=vals[ki]; ki++; } out[i]=cur; }
  return out;
}
function ret30Map(T,C){
  const ms=86400e3, dT=[], dC=[];
  for(let i=0;i<T.length;i++){ const b=Math.floor(T[i]/ms)*ms;
    if(!dT.length||dT[dT.length-1]!==b){ dT.push(b); dC.push(C[i]); } else dC[dC.length-1]=C[i]; }
  const keys=[],vals=[];
  for(let j=30;j<dT.length;j++){ keys.push(dT[j]+ms); vals.push(dC[j]/dC[j-30]-1); }
  const out=new Array(T.length).fill(null); let ki=0,cur=null;
  for(let i=0;i<T.length;i++){ while(ki<keys.length&&keys[ki]<=T[i]){ cur=vals[ki]; ki++; } out[i]=cur; }
  return out;
}
function normCdf(z){ const t=1/(1+0.2316419*Math.abs(z)), d=0.3989423*Math.exp(-z*z/2);
  const p=d*t*(0.3193815+t*(-0.3565638+t*(1.781478+t*(-1.821256+t*1.330274)))); return z>0?1-p:p; }

// ─── 決策樹（CART / Gini）────────────────────────────────
function giniOf(p){ return 1-p*p-(1-p)*(1-p); }
function buildTree(X,y,idx,feats,depth,maxDepth,minLeaf,imp){
  const n=idx.length;
  let pos=0; for(const i of idx) pos+=y[i];
  const p=pos/n;
  if(depth>=maxDepth || n<2*minLeaf || p===0 || p===1) return {leaf:true, p};
  const parentG=giniOf(p);
  let best=null;
  for(const f of feats){
    const vals=idx.map(i=>[X[i][f],y[i]]).sort((a,b)=>a[0]-b[0]);
    let lp=0;
    for(let k=0;k<n-1;k++){
      lp+=vals[k][1];
      const lc=k+1, rc=n-lc;
      if(lc<minLeaf||rc<minLeaf) continue;
      if(vals[k][0]===vals[k+1][0]) continue;
      const g=(lc/n)*giniOf(lp/lc)+(rc/n)*giniOf((pos-lp)/rc);
      const gain=parentG-g;
      if(!best||gain>best.gain) best={gain, f, thr:(vals[k][0]+vals[k+1][0])/2};
    }
  }
  if(!best||best.gain<=1e-9) return {leaf:true,p};
  if(imp) imp[best.f]=(imp[best.f]||0)+best.gain*n;
  const L=[],R=[];
  for(const i of idx) (X[i][best.f]<=best.thr?L:R).push(i);
  if(!L.length||!R.length) return {leaf:true,p};
  return { leaf:false, f:best.f, thr:best.thr,
    L:buildTree(X,y,L,feats,depth+1,maxDepth,minLeaf,imp),
    R:buildTree(X,y,R,feats,depth+1,maxDepth,minLeaf,imp) };
}
function treePred(node,x){ while(!node.leaf) node = x[node.f]<=node.thr ? node.L : node.R; return node.p; }

// ─── 隨機森林 ───────────────────────────────────────────
function mulberry(a){ return function(){ a|=0; a=a+0x6D2B79F5|0;
  let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t;
  return ((t^t>>>14)>>>0)/4294967296; }; }
function buildForest(X,y,idx,nFeat,nTree,maxDepth,minLeaf,imp,seed){
  const rnd=mulberry(seed||42), F=X[0].length, m=Math.max(1,Math.round(Math.sqrt(F)));
  const trees=[];
  for(let t=0;t<nTree;t++){
    const boot=[]; for(let i=0;i<idx.length;i++) boot.push(idx[(rnd()*idx.length)|0]);
    const pool=[]; for(let f=0;f<F;f++) pool.push(f);
    for(let i=pool.length-1;i>0;i--){ const j=(rnd()*(i+1))|0; [pool[i],pool[j]]=[pool[j],pool[i]]; }
    trees.push(buildTree(X,y,boot,pool.slice(0,m),0,maxDepth,minLeaf,imp));
  }
  return trees;
}
function forestPred(trees,x){ let s=0; for(const t of trees) s+=treePred(t,x); return s/trees.length; }

// ─── 羅吉斯迴歸（標準化 + 梯度下降）──────────────────────
function fitLogistic(X,y,idx,iters=300,lr=0.3){
  const F=X[0].length, mu=new Array(F).fill(0), sd=new Array(F).fill(0);
  for(const i of idx) for(let f=0;f<F;f++) mu[f]+=X[i][f];
  for(let f=0;f<F;f++) mu[f]/=idx.length;
  for(const i of idx) for(let f=0;f<F;f++) sd[f]+=(X[i][f]-mu[f])**2;
  for(let f=0;f<F;f++) sd[f]=Math.sqrt(sd[f]/idx.length)||1;
  const w=new Array(F).fill(0); let b=0;
  for(let it=0;it<iters;it++){
    const gw=new Array(F).fill(0); let gb=0;
    for(const i of idx){
      let z=b; for(let f=0;f<F;f++) z+=w[f]*(X[i][f]-mu[f])/sd[f];
      const pr=1/(1+Math.exp(-z)), e=pr-y[i];
      for(let f=0;f<F;f++) gw[f]+=e*(X[i][f]-mu[f])/sd[f];
      gb+=e;
    }
    for(let f=0;f<F;f++) w[f]-=lr*gw[f]/idx.length;
    b-=lr*gb/idx.length;
  }
  return {w,b,mu,sd};
}
function logiPred(m,x){ let z=m.b;
  for(let f=0;f<x.length;f++) z+=m.w[f]*(x[f]-m.mu[f])/m.sd[f];
  return 1/(1+Math.exp(-z)); }

function auc(ys,ps){
  const n=ys.length, ord=ys.map((_,i)=>i).sort((a,b)=>ps[a]-ps[b]);
  let r=new Array(n), i=0;
  while(i<n){ let j=i; while(j+1<n&&ps[ord[j+1]]===ps[ord[i]]) j++;
    const avg=(i+j)/2+1; for(let k=i;k<=j;k++) r[ord[k]]=avg; i=j+1; }
  let sp=0,np=0,nn=0;
  for(let k=0;k<n;k++){ if(ys[k]===1){ sp+=r[k]; np++; } else nn++; }
  if(!np||!nn) return 0.5;
  return (sp-np*(np+1)/2)/(np*nn);
}

// ─── 主流程 ─────────────────────────────────────────────
const FNAMES=["超出下軌σ倍","連續外軌根數","RSI14","近5分報酬","近15分報酬","近60分報酬",
  "1m波動率60","布林帶寬","距1H均線%","30日報酬","RSI5","距EMA60%","小時sin","小時cos"];

async function run(){
  $("runBtn").disabled=true; $("out").innerHTML="";
  const sym=$("sym").value, months=+$("months").value, H=+$("hz").value;
  const h1ma=+$("h1ma").value, bn=+$("bolln").value, bk=+$("bollk").value;
  const th=(+$("filtTh").value||30)/100;
  const K=+$("folds").value, gapMs=(+$("gapD").value||0)*86400e3;
  const maxDepth=+$("depth").value, minLeaf=+$("leaf").value, nTree=+$("ntree").value;
  try{
    const {t:T,c:C}=await loadData(sym,months);
    $("status").textContent="計算特徵與訊號中…";
    await new Promise(r=>setTimeout(r,20));

    const mid=sma(C,bn), sd=rollStd(C,bn), t1h=trendMap(T,C,60,h1ma), r30=ret30Map(T,C);
    const rsi14=rsiArr(C,14), rsi5=rsiArr(C,5), e60=ema(C,60);
    const ret1=new Array(C.length).fill(0);
    for(let i=1;i<C.length;i++) ret1[i]=C[i]/C[i-1]-1;
    const vol60=rollStd(ret1,60);

    // 連續在下軌外的根數
    const outRun=new Array(C.length).fill(0);
    for(let i=0;i<C.length;i++){
      if(mid[i]===null){ outRun[i]=0; continue; }
      outRun[i] = (C[i]<mid[i]-bk*sd[i]) ? (i>0?outRun[i-1]+1:1) : 0;
    }

    const X=[], Y=[], TS=[];
    let busy=-1, blocked=0;
    for(let i=1;i<C.length-H-2;i++){
      const e=i+1;
      if(mid[i]===null||mid[e]===null||t1h[e]===null||r30[e]===null) continue;
      if(rsi14[e]===null||rsi5[e]===null||e60[e]===null||vol60[e]===null) continue;
      if(T[e]<=busy) continue;
      // 規則：連兩根跌破下軌 ＋ 1H 偏空
      const dn = C[i]<mid[i]-bk*sd[i] && C[e]<mid[e]-bk*sd[e];
      if(!dn || t1h[e].d!==-1) continue;
      if(Math.abs(r30[e])>=th){ blocked++; continue; }
      const targetT=T[e]+H*60000;
      let j=e+H; while(j<T.length&&T[j]<targetT) j++;
      if(j>=T.length||T[j]!==targetT) continue;
      if(C[j]===C[e]) continue;                     // 持平剔除
      const d=new Date(T[e]), hr=d.getUTCHours()+d.getUTCMinutes()/60;
      X.push([
        (mid[e]-C[e])/(sd[e]||1e-9),               // 超出下軌幾個 σ
        outRun[e],                                  // 連續外軌根數
        rsi14[e],
        C[e]/C[Math.max(0,e-5)]-1,
        C[e]/C[Math.max(0,e-15)]-1,
        C[e]/C[Math.max(0,e-60)]-1,
        vol60[e],
        sd[e]/mid[e],                               // 布林帶寬（波動體質）
        t1h[e].gap,                                 // 距 1H 均線 %
        r30[e],                                     // 30 日報酬（市場狀態）
        rsi5[e],
        C[e]/e60[e]-1,
        Math.sin(2*Math.PI*hr/24), Math.cos(2*Math.PI*hr/24),
      ]);
      Y.push(C[j]>C[e]?1:0);                        // 押漲是否成功
      TS.push(T[e]);
      busy=T[e]+H*60000;
    }

    if(X.length<300) throw new Error(`訊號只有 ${X.length} 筆，樣本太少。請放寬參數或拉長期間。`);
    $("status").textContent=`訊號 ${X.length} 筆，開始 ${K} 折驗證…`;
    await new Promise(r=>setTimeout(r,20));

    // ─── TimeSeriesSplit K 折 ＋ gap ───
    const n=X.length, chunk=Math.floor(n/(K+1));
    const folds=[];
    for(let k=0;k<K;k++){
      const trEnd=chunk*(k+1), teStart=trEnd, teEnd=(k===K-1)?n:chunk*(k+2);
      const cut=TS[teStart]-gapMs;
      const tr=[]; for(let i=0;i<trEnd;i++) if(TS[i]<cut) tr.push(i);
      const te=[]; for(let i=teStart;i<teEnd;i++) te.push(i);
      if(tr.length<100||te.length<20) continue;
      folds.push({tr,te});
    }
    if(!folds.length) throw new Error("折數或 gap 設定導致沒有有效折，請調小 gap 或折數。");

    const models={ "規則基準（全押漲）":null, "DecisionTree":"dt", "RandomForest":"rf", "Logistic":"lg" };
    const res={}; const impAgg={};
    const foldBase=[];
    for(const name in models) res[name]={acc:[], auc:[], kept:[], picked:[]};

    for(let fi=0; fi<folds.length; fi++){
      const {tr,te}=folds[fi];
      $("status").textContent=`第 ${fi+1}/${folds.length} 折訓練中…`;
      $("progFill").style.width=((fi+1)/folds.length*100).toFixed(0)+"%";
      await new Promise(r=>setTimeout(r,10));

      const baseAcc=te.reduce((s,i)=>s+Y[i],0)/te.length;
      foldBase.push(baseAcc);
      res["規則基準（全押漲）"].acc.push(baseAcc);
      res["規則基準（全押漲）"].auc.push(0.5);
      res["規則基準（全押漲）"].kept.push(1);
      res["規則基準（全押漲）"].picked.push(baseAcc);

      const imp={};
      const dt=buildTree(X,Y,tr,X[0].map((_,f)=>f),0,maxDepth,minLeaf,null);
      const rf=buildForest(X,Y,tr,null,nTree,maxDepth,minLeaf,imp,1234+fi);
      const lg=fitLogistic(X,Y,tr);
      for(const f in imp) impAgg[f]=(impAgg[f]||0)+imp[f];

      for(const [name,kind] of [["DecisionTree","dt"],["RandomForest","rf"],["Logistic","lg"]]){
        const ps=te.map(i=> kind==="dt"?treePred(dt,X[i]) : kind==="rf"?forestPred(rf,X[i]) : logiPred(lg,X[i]));
        const ys=te.map(i=>Y[i]);
        // 預測準確率（模型判斷「這筆會贏嗎」）
        let correct=0; for(let k=0;k<ys.length;k++) correct += ((ps[k]>=0.5?1:0)===ys[k])?1:0;
        res[name].acc.push(correct/ys.length);
        res[name].auc.push(auc(ys,ps));
        // 交易意義：只在模型說會贏（p≥0.5）時下注，這些單的實際勝率
        const sel=[]; for(let k=0;k<ys.length;k++) if(ps[k]>=0.5) sel.push(ys[k]);
        res[name].kept.push(sel.length/ys.length);
        res[name].picked.push(sel.length? sel.reduce((a,b)=>a+b,0)/sel.length : NaN);
      }
    }

    // ─── 門檻掃描（用最後一折外推，所有折合併）───
    const sweep=[];
    {
      const allP=[], allY=[];
      for(const {tr,te} of folds){
        const imp={};
        const rf=buildForest(X,Y,tr,null,nTree,maxDepth,minLeaf,imp,999);
        for(const i of te){ allP.push(forestPred(rf,X[i])); allY.push(Y[i]); }
      }
      for(const t of [0.50,0.55,0.60,0.65]){
        const sel=[]; for(let k=0;k<allY.length;k++) if(allP[k]>=t) sel.push(allY[k]);
        if(sel.length<30){ sweep.push({t, n:sel.length, wr:NaN, ev:NaN, pnl:NaN}); continue; }
        const wr=sel.reduce((a,b)=>a+b,0)/sel.length;
        sweep.push({t, n:sel.length, wr, ev:wr*PAYOUT-(1-wr),
          pnl:sel.reduce((a,b)=>a+b,0)*STAKE*PAYOUT-(sel.length-sel.reduce((a,b)=>a+b,0))*STAKE});
      }
    }

    render({res, folds:folds.length, n, blocked, sweep, impAgg, X, sym, months, H, foldBase});
    $("status").textContent=`完成：${sym} 近 ${months} 個月，訊號 ${n} 筆`;
  }catch(e){
    $("out").innerHTML=`<div class="card" style="color:var(--down)">錯誤：${e.message}</div>`;
    $("status").textContent="失敗";
  }
  $("runBtn").disabled=false;
}

const mean=a=>a.filter(x=>!isNaN(x)).reduce((s,x)=>s+x,0)/a.filter(x=>!isNaN(x)).length;
const std=a=>{ const v=a.filter(x=>!isNaN(x)); const m=mean(v);
  return Math.sqrt(v.reduce((s,x)=>s+(x-m)**2,0)/v.length); };

function render(D){
  const {res, folds, n, blocked, sweep, impAgg, X, foldBase}=D;
  const base=mean(foldBase);
  const pct=x=>isNaN(x)?"—":(x*100).toFixed(1)+"%";

  let html=`<div class="card">
    <div class="sec" style="margin-top:0">樣本</div>
    <div class="row"><span class="mono">訊號 ${n} 筆</span>
      <span class="hint">濾網擋下 ${blocked}・${folds} 折 TimeSeriesSplit＋gap・特徵 ${X[0].length} 個</span></div>

    <div class="sec">各模型表現（${folds} 折平均 ± 標準差）</div>
    <table><tr><th>模型</th><th>預測準確率</th><th>AUC</th><th>下注比例</th><th>下注單勝率</th><th>每1U期望值</th></tr>`;

  for(const name in res){
    const r=res[name];
    const picked=mean(r.picked), ev=picked*PAYOUT-(1-picked);
    const isBase=name.startsWith("規則");
    const better = !isBase && picked>base;
    html+=`<tr class="${better?"hl":""}">
      <td><b>${name}</b></td>
      <td class="mono">${pct(mean(r.acc))}<span class="hint"> ±${(std(r.acc)*100).toFixed(1)}</span></td>
      <td class="mono">${isBase?"—":mean(r.auc).toFixed(3)}</td>
      <td class="mono">${pct(mean(r.kept))}</td>
      <td class="mono" style="color:${picked>BREAKEVEN?"var(--up)":"var(--down)"}">${pct(picked)}</td>
      <td class="mono" style="color:${ev>=0?"var(--up)":"var(--down)"}">${isNaN(ev)?"—":(ev>=0?"+":"")+ev.toFixed(3)}</td></tr>`;
  }
  html+=`</table>
    <div class="hint" style="margin-top:8px">「下注單勝率」才是交易上有意義的數字：模型說會贏的那些單，實際贏了多少。
    基準線＝規則全押漲 ${pct(base)}，兩平線 55.6%。黃底＝勝過規則基準。</div>`;

  // 各折明細
  html+=`<div class="sec">各折勝率（看穩定度，不是看平均）</div><table>
    <tr><th>模型</th>${res["規則基準（全押漲）"].picked.map((_,i)=>`<th>第${i+1}折</th>`).join("")}</tr>`;
  for(const name in res){
    html+=`<tr><td>${name}</td>${res[name].picked.map(v=>
      `<td class="mono" style="color:${v>BREAKEVEN?"var(--up)":"var(--down)"}">${pct(v)}</td>`).join("")}</tr>`;
  }
  html+=`</table>`;

  // 門檻掃描
  html+=`<div class="sec">信心門檻掃描（RandomForest，全折外推合併）</div><table>
    <tr><th>門檻</th><th>下注筆數</th><th>勝率</th><th>每1U期望值</th><th>固定20U損益</th></tr>`;
  for(const s of sweep){
    html+=`<tr><td class="mono">p ≥ ${s.t.toFixed(2)}</td><td class="mono">${s.n}</td>
      <td class="mono" style="color:${s.wr>BREAKEVEN?"var(--up)":"var(--down)"}">${pct(s.wr)}</td>
      <td class="mono" style="color:${s.ev>=0?"var(--up)":"var(--down)"}">${isNaN(s.ev)?"—":(s.ev>=0?"+":"")+s.ev.toFixed(3)}</td>
      <td class="mono" style="color:${s.pnl>=0?"var(--up)":"var(--down)"}">${isNaN(s.pnl)?"—":(s.pnl>=0?"+":"")+s.pnl.toFixed(0)}U</td></tr>`;
  }
  html+=`</table><div class="hint" style="margin-top:6px">門檻拉高＝更挑剔＝筆數變少。要看的是勝率升幅能不能補償筆數損失。</div>`;

  // 特徵重要性
  const imps=Object.entries(impAgg).map(([f,v])=>[+f,v]).sort((a,b)=>b[1]-a[1]);
  const tot=imps.reduce((s,x)=>s+x[1],0)||1;
  html+=`<div class="sec">特徵重要性（RandomForest，Gini 增益加總）</div>`;
  for(const [f,v] of imps.slice(0,10)){
    html+=`<div style="display:flex;gap:8px;align-items:center;font-size:12px;padding:2px 0">
      <span class="mono" style="width:110px">${FNAMES[f]||("f"+f)}</span>
      <span class="bar" style="width:${(v/tot*300).toFixed(0)}px"></span>
      <span class="mono hint">${(v/tot*100).toFixed(1)}%</span></div>`;
  }

  // 判定
  const rfPicked=mean(res["RandomForest"].picked), dtPicked=mean(res["DecisionTree"].picked);
  const bestPicked=Math.max(rfPicked,dtPicked,mean(res["Logistic"].picked));
  let v;
  if(bestPicked>base+0.02) v=`模型有效：最佳模型把下注單勝率從規則的 ${pct(base)} 提升到 ${pct(bestPicked)}，而且是在 ${folds} 折外推驗證下。值得把它當成下注前的過濾器。`;
  else if(bestPicked>base) v=`模型略有幫助（${pct(base)} → ${pct(bestPicked)}），但幅度小於 2 個百分點，可能只是雜訊。看各折是否一致，不一致就別用。`;
  else v=`模型沒有勝過規則本身（規則 ${pct(base)} vs 模型 ${pct(bestPicked)}）。代表規則已經把能用的訊息抓完了，加機器學習反而多此一舉 — 這本身就是有價值的結論。`;
  html+=`<div class="verdict">${v}</div></div>`;

  $("out").innerHTML=html;
}
$("runBtn").onclick=run;
</script>
</body>
</html>
