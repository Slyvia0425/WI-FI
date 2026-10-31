
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync('web/app.js','utf8'),html=fs.readFileSync('web/index.html','utf8');
const original=[[.1,1,.45],[.1,5,.65],[5.9,1,.65],[5.9,5,.45]],changed=structuredClone(original);changed[0]=[1.1,1.2,.45];
const elements={};const field=id=>elements[id]??=( {value:'',textContent:'',innerHTML:'',hidden:true,dataset:{}} );
const checks=[];function check(name,fn){fn();checks.push({name,passed:true});}
const ctx=vm.createContext({console,Math,Number,JSON,Infinity,Set,Array,$:field,busy:false,liveTx:structuredClone(original),running:true,epoch:0,result:null,history:[],records:[],cursor:0,
 state:{config:{home_environment:true,tx_m:original,rx_m:[],fixed_objects:[],available_layouts:[{id:'default',ready:true,tx_positions:original},{id:'t1_v1',ready:true,tx_positions:changed}]},records:[{id:'h5_test',layout_id:'default',count:3,split:'test'},{id:'t2_test',layout_id:'t1_v1',count:3,split:'test'}]},
 t:(zh,en)=>zh,esc:s=>String(s),pct:x=>(100*x).toFixed(1)+'%',occupancyMode:()=>true,statusLabel:s=>s,
 inputs:()=>[{x:1.1,y:1.2,size:[.6,.5,.8]}],render:()=>{},renderHistory:()=>{},setStatus:(...x)=>ctx.lastStatus=x,
 stop:()=>{ctx.running=false;ctx.result=null;ctx.epoch++;},invalidate:()=>{field('source').value='live';ctx.running=false;ctx.result=null;},
 renderTxFields:()=>ctx.liveTx.forEach((p,i)=>p.forEach((v,j)=>field(`tx-${i}-${j}`).value=String(v))),
 renderLayoutScore:()=>{field('trained-layout').value=ctx.state.config.available_layouts.find(e=>JSON.stringify(e.tx_positions)===JSON.stringify(ctx.liveTx))?.id||'';}
});
vm.runInContext(source.slice(source.indexOf('function englishMessage('),source.indexOf('const names=')),ctx);
vm.runInContext(source.slice(source.indexOf('function replayLayoutId()'),source.indexOf('let layoutCount=')),ctx);
vm.runInContext(source.slice(source.indexOf('function resetRecords()'),source.indexOf("$('source').onchange=")),ctx);
vm.runInContext(source.slice(source.indexOf('function applyTx('),source.indexOf("$('tx-apply').onclick=")),ctx);
vm.runInContext(source.slice(source.indexOf('function initLayoutPresets()'),source.indexOf('function renderLayoutScore()')),ctx);
const run=s=>vm.runInContext(s,ctx);
check('One top-level layout selector',()=>{assert.equal((html.match(/id="trained-layout"/g)||[]).length,1);assert(html.indexOf('id="trained-layout"')<html.indexOf('id="truth-map"'));});
check('Replay preset preserves source and selects matching records',()=>{field('source').value='3';field('trained-layout').value='t1_v1';ctx.result={old:true};run("applyTx(false,state.config.available_layouts[1])");assert.equal(field('source').value,'3');assert.equal(ctx.records[0].id,'t2_test');assert.equal(ctx.result,null);});
check('Reset during replay selects default records before detection',()=>{run('applyTx(true)');assert.equal(field('trained-layout').value,'default');assert.equal(ctx.records[0].id,'h5_test');assert.equal(field('source').value,'3');});
check('Invalid manual coordinates preserve applied layout',()=>{ctx.renderTxFields();field('tx-0-0').value='7';const before=JSON.stringify(ctx.liveTx);run('applyTx()');assert.equal(JSON.stringify(ctx.liveTx),before);assert.equal(field('error').hidden,false);});
check('Busy preset change does not replace active layout',()=>{ctx.renderTxFields();run('initLayoutPresets()');ctx.busy=true;field('trained-layout').value='t1_v1';field('trained-layout').onchange();assert.equal(field('trained-layout').value,'default');assert.deepEqual(ctx.liveTx,original);ctx.busy=false;});
check('Manual valid coordinates switch replay to custom mode',()=>{ctx.renderTxFields();field('tx-0-0').value='1.6';field('tx-0-1').value='1.2';run('applyTx()');assert.equal(field('source').value,'live');assert.equal(ctx.liveTx[0][0],1.6);});
vm.runInContext(source.slice(source.indexOf('function renderHistory(){'),source.indexOf('function renderEvaluation(){')),ctx);
check('History denominator uses latest25 and current revision only',()=>{
 const row=(lid,rev,match=true)=>({model_layout_id:lid,inference_model_revision:rev,truth:[{zone:0}],true_count:1,occupancy:{per_zone_object_count:[match?1:0,0,0,0,0,0,0,0,0],estimated_object_count:1,confirmed_objects:1,status:'Accepted regions',elapsed_ms:2}});
 ctx.history=Array.from({length:40},()=>row('t1_v1','old',false));ctx.history.push(row('default','default_v5'),row('t1_v1','new',true),row('t1_v1','new',false));ctx.result={model_layout_id:'t1_v1',inference_model_revision:'new'};field('show-truth').checked=true;run('renderHistory()');assert(field('summary').textContent.includes('1/2'));assert.equal((field('history').innerHTML.match(/<tr>/g)||[]).length,25);assert(field('history').innerHTML.includes('T1预设'));
});
async function modelUpdates(){
 ctx.state.config.available_layouts=ctx.state.config.available_layouts.map(e=>({...e,revision:e.id==='default'?'default_v5':'t1_v1'}));
 ctx.liveTx=structuredClone(changed);field('source').value='3';field('trained-layout').value='t1_v1';ctx.result={model_layout_id:'t1_v1',inference_model_revision:'t1_v1'};ctx.running=true;
 const fresh=structuredClone(ctx.state);fresh.config.available_layouts.find(e=>e.id==='t1_v1').revision='t1_count_v3';fresh.records.find(e=>e.layout_id==='t1_v1').id='t3_test';
 ctx.setInterval=fn=>ctx.modelPoll=fn;ctx.fetch=async path=>({ok:true,json:async()=>path==='/api/layouts'?{layouts:fresh.config.available_layouts.map(({id,ready,revision})=>({id,ready,revision}))}:fresh});
 vm.runInContext(source.slice(source.indexOf('let layoutPollBusy=false;'),source.indexOf('function renderTxFields()')),ctx);
 await ctx.modelPoll();
 check('Model revision update invalidates stale result and refreshes records',()=>{assert.equal(ctx.result,null);assert.equal(ctx.running,false);assert.equal(ctx.records[0].id,'t3_test');assert.equal(field('source').value,'3');assert(!field('summary').textContent.includes('布局匹配'));});
 const snapshot={model_layout_id:'t1_v1',inference_model_revision:'t1_count_v3'};ctx.result=snapshot;ctx.running=true;await ctx.modelPoll();
 check('Unchanged model version does not interrupt processing',()=>{assert.strictEqual(ctx.result,snapshot);assert.equal(ctx.running,true);});

 check('Unvalidated custom layout has no default replay records',()=>{field('trained-layout').value='';field('source').value='3';run('resetRecords()');assert.equal(ctx.records.length,0);});
 vm.runInContext(source.slice(source.indexOf("$('upload-button').onclick="),source.indexOf("$('signal-download').onclick=")),ctx);
 field('upload').files=[{size:8,text:async()=>'{"real":[]}'}];ctx.busy=false;
 let resolve;ctx.api=()=>new Promise(r=>resolve=r);
 const upload=field('upload-button').onclick();await new Promise(r=>setImmediate(r));
 assert(resolve,'upload request started');ctx.epoch++;const current={newScene:true};ctx.result=current;const before=ctx.history.length;resolve({model_layout_id:'default'});await upload;
 check('Stale upload cannot overwrite a newer scene or append history',()=>{assert.strictEqual(ctx.result,current);assert.equal(ctx.history.length,before);assert.equal(ctx.busy,false);});
 ctx.api=async()=>({model_layout_id:'default'});await field('upload-button').onclick();
 check('Fresh upload still displays and records the response',()=>{assert.equal(ctx.result.model_layout_id,'default');assert.equal(ctx.history.length,before+1);});
 fs.mkdirSync('outputs/ui_checks',{recursive:true});fs.writeFileSync('outputs/ui_checks/layout_logic.json',JSON.stringify({passed:true,scope:'Code-level state/HTML checks only; no browser rendering is claimed.',checks},null,2));console.log(JSON.stringify({passed:true,checks:checks.length}));
}
modelUpdates().catch(e=>{console.error(e);process.exitCode=1;});
