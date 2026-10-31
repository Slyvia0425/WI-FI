'use strict';
const $=id=>document.getElementById(id),pct=x=>x==null?'—':(100*x).toFixed(1)+'%';
let lang='en',state=null,result=null,history=[],running=false,busy=false,epoch=0,timer=null,runDone=0,cursor=0,records=[];
let mapDrag=null;
let liveTx=null;
function placementConfig(){return {...state.config,tx_m:liveTx||state.config.tx_m};}
function replayLayoutId(){return state.config.home_environment?($('trained-layout').value||null):'default';}
let layoutCount=2;
let layoutBeforeExample=null;
const layoutKey='roomsense-custom-layout-v1';
const t=(zh,en)=>en;
function englishMessage(value){const parts=String(value||'').split(/\s+\/\s+/);return parts.find(p=>p.trim()&&!/[\u3400-\u9fff]/.test(p))||'Unable to process this input.';}
const names=()=>[t('小型','Small'),t('中型','Medium'),t('大型','Large')];
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const zone=z=>'Z'+(z+1);
const occupancyMode=()=>$('output-mode').value==='occupancy';
let statusCopy={zh:'就绪',en:'Ready',tone:'idle'};
function setStatus(zh,en,tone='idle'){statusCopy={zh,en,tone};$('status').textContent=t(zh,en);$('status').dataset.tone=tone;$('start').disabled=running||busy;}
function statusLabel(s){return s==='Accepted regions'||s==='Accepted'?t('已识别','Detected'):s==='Uncertain'?t('不确定','Uncertain'):s==='Unknown'?t('未知','Unknown'):t('等待检测','Awaiting input');}
function translate(){
  document.querySelectorAll('[data-zh][data-en]').forEach(e=>{e.textContent=t(e.dataset.zh,e.dataset.en);});
  document.documentElement.lang=lang==='en'?'en':'zh-CN';
  setStatus(statusCopy.zh,statusCopy.en,statusCopy.tone);
  if(state){$('environment-note').hidden=!state.config.home_environment;$('environment-note').textContent=t('家居场景 · 门窗与固定柜体 · 统计待测物体','Furnished room · door, window & fixed cabinet · counts refer to targets');updateDimensions();render();initLayoutPresets();renderEvaluation();renderHistory();}
}
localStorage.setItem('multi-language','en');$('lang').value='en';$('lang').onchange=()=>{lang='en';translate();};
function initInputs(){
  liveTx=state.config.tx_m.map(p=>[...p]);$('tx-panel').hidden=!state.config.home_environment;$('radio-control').hidden=!state.config.home_environment;renderTxFields();initLayoutPresets();
  if(!state.config.home_environment)$('single-target').remove();else $('single-target').hidden=false;
  if(state.config.home_environment){let o=document.createElement('option');o.value='1';o.dataset.zh='测试：1个物体';o.dataset.en='Test: 1 object';$('source').append(o);}
  if(state.model_version>=2&&!$('boundary-source')){let o=document.createElement('option');o.id='boundary-source';o.value='boundary';o.dataset.zh='交界测试';o.dataset.en='Boundary test';$('source').append(o);}
  if(state.config.supports_shared_zones){
    $('shared-examples').hidden=false;
    for(let n of [2,3]){let opt=document.createElement('option');opt.value='shared-'+n;opt.dataset.zh='同区测试：'+n+'件';opt.dataset.en='Shared-zone test: '+n+' objects';$('source').append(opt);}
    $('placement-rule').dataset.zh='每档最多1件；允许同区，物体不可重叠。拖动自动避让设备。';$('placement-rule').dataset.en='One per size band; shared zones allowed, no overlap. Dragging avoids radios.';
    $('output-mode').querySelector('option[value="typed"]').disabled=true;$('output-mode').value='occupancy';
  }
  let defaults=[{kind:0,x:1,y:1},{kind:2,x:5,y:5},{kind:1,x:3,y:3}];
  let saved=null;try{saved=JSON.parse(localStorage.getItem(layoutKey));if(!saved||!(state.config.home_environment?[1,2,3]:[2,3]).includes(saved.count)||saved.objects?.length!==3||saved.objects.some(o=>![0,1,2].includes(o.kind)||![o.x,o.y,o.scale].every(Number.isFinite)||o.x<0||o.x>6||o.y<0||o.y>6||o.scale<.92||o.scale>1.08))saved=null;}catch{}
  if(saved){defaults=saved.objects;$('count').value=String(saved.count);$('edit').value=String(saved.edit>=0&&saved.edit<saved.count?saved.edit:0);}
  $('placements').innerHTML=defaults.map((o,i)=>`<tr id="row-${i}"><td><button type="button" class="object-pick" data-select-object="${i}">${'ABC'[i]}</button></td><td><select id="kind-${i}">${[0,1,2].map(k=>`<option value="${k}" ${k===o.kind?'selected':''}>${['S / Small','M / Medium','L / Large'][k]}</option>`).join('')}</select></td><td><input id="x-${i}" type="number" min="0" max="6" step=".01" value="${o.x}"></td><td><input id="y-${i}" type="number" min="0" max="6" step=".01" value="${o.y}"></td><td><input id="scale-${i}" type="range" min=".92" max="1.08" step=".01" value="${o.scale||1}"><span id="scale-label-${i}" class="scale-label">1.00</span></td><td id="dims-${i}"></td></tr>`).join('');
  for(let i=0;i<3;i++)for(let key of ['kind','x','y','scale'])$(`${key}-${i}`).oninput=()=>{$('edit').value=String(i);updateDimensions();invalidate();};
  for(let i=0;i<3;i++)for(let key of ['kind','x','y','scale']){const commit=()=>{if(i<Number($('count').value)){let o=inputs()[i],p=nearestPlacement(inputs(),i,o.x,o.y,placementConfig());if(p&&(Math.abs(p.x-o.x)>.005||Math.abs(p.y-o.y)>.005))moveObject(i,o.x,o.y);}};$(`${key}-${i}`).onchange=commit;$(`${key}-${i}`).onblur=commit;}
  updateDimensions();
  layoutCount=Number($('count').value);
}
function updateDimensions(){
  if(!state)return;let n=Number($('count').value);
  for(let i=0;i<3;i++){let k=Number($(`kind-${i}`).value),scale=Number($(`scale-${i}`).value);$(`row-${i}`).hidden=i>=n;$('edit').options[i].textContent='ABC'[i]+' · '+names()[k];$(`scale-label-${i}`).textContent=scale.toFixed(2);$(`dims-${i}`).textContent=state.config.canonical_sizes_m[k].map(v=>(v*scale).toFixed(2)).join(' × ');}
  for(let i=0;i<3;i++)$('edit').options[i].disabled=i>=n;if(Number($('edit').value)>=n)$('edit').value='0';
}
function inputs(){return Array.from({length:Number($('count').value)},(_,i)=>{let k=Number($(`kind-${i}`).value),scale=Number($(`scale-${i}`).value);return {kind:k,x:Number($(`x-${i}`).value),y:Number($(`y-${i}`).value),size:state.config.canonical_sizes_m[k].map(v=>v*scale)};});}
function capturePlacement(){return {count:$('count').value,edit:$('edit').value,objects:[0,1,2].map(i=>({kind:$('kind-'+i).value,x:$('x-'+i).value,y:$('y-'+i).value,scale:$('scale-'+i).value}))};}
function applyPlacement(v){$('count').value=String(v.count);$('edit').value=String(v.edit);v.objects.forEach((o,i)=>{for(let k of ['kind','x','y','scale'])$(k+'-'+i).value=String(o[k]);});layoutCount=Number(v.count);updateDimensions();invalidate();}
function sharedExample(n){if(!layoutBeforeExample)layoutBeforeExample=capturePlacement();let xy=n===2?[[3,3.05],[3,2.15],[3.48,3.65]]:[[2.45,3.65],[3,2.60],[3.48,3.65]];applyPlacement({count:n,edit:0,objects:[0,2,1].map((kind,i)=>({kind,x:xy[i][0],y:xy[i][1],scale:1}))});$('undo-layout').disabled=false;}
$('preset-two').onclick=()=>sharedExample(2);$('preset-three').onclick=()=>sharedExample(3);$('undo-layout').onclick=()=>{if(layoutBeforeExample){applyPlacement(layoutBeforeExample);layoutBeforeExample=null;$('undo-layout').disabled=true;}};
function saveLayout(){
  if(!state)return;let objects=[0,1,2].map(i=>({kind:Number($(`kind-${i}`).value),x:Number($(`x-${i}`).value),y:Number($(`y-${i}`).value),scale:Number($(`scale-${i}`).value)}));
  if(objects.every(o=>[o.x,o.y,o.scale].every(Number.isFinite)))try{localStorage.setItem(layoutKey,JSON.stringify({count:Number($('count').value),edit:Number($('edit').value),objects}));}catch{}
}
function selectObject(i){if(i<0||i>=Number($('count').value))return;$('edit').value=String(i);saveLayout();render();}
function mapPoint(e){let svg=$('truth-map').querySelector('svg'),p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;p=p.matrixTransform(svg.getScreenCTM().inverse());return {x:(p.x-40)/60,y:(400-p.y)/60};}

function nearestPlacement(objects,index,x,y,cfg){
  const o=objects[index],pad=cfg.clearance_m??.08,room=cfg.room_m||[6,6],hw=o.size[0]/2,hh=o.size[1]/2;
  if(![x,y,hw,hh].every(Number.isFinite))return null;
  const low=[Math.ceil((hw+.01)*100),Math.ceil((hh+.01)*100)],high=[Math.floor((room[0]-hw-.01)*100),Math.floor((room[1]-hh-.01)*100)];
  const blocks=[...(cfg.tx_m||[]),...(cfg.rx_m||[])].map(p=>[p[0]-hw-pad,p[0]+hw+pad,p[1]-hh-pad,p[1]+hh+pad]);
  (cfg.fixed_objects||[]).forEach(p=>blocks.push([p.x-(o.size[0]+p.size[0])/2-pad,p.x+(o.size[0]+p.size[0])/2+pad,p.y-(o.size[1]+p.size[1])/2-pad,p.y+(o.size[1]+p.size[1])/2+pad]));
  objects.forEach((p,j)=>{if(j!==index)blocks.push([p.x-(o.size[0]+p.size[0])/2-pad,p.x+(o.size[0]+p.size[0])/2+pad,p.y-(o.size[1]+p.size[1])/2-pad,p.y+(o.size[1]+p.size[1])/2+pad]);});
  const clamp=(v,a)=>Math.max(low[a],Math.min(high[a],v));
  const xs=new Set([low[0],high[0],clamp(Math.round(x*100),0)]),ys=new Set([low[1],high[1],clamp(Math.round(y*100),1)]);
  for(const b of blocks){xs.add(clamp(Math.ceil(b[0]*100-1e-8)-1,0));xs.add(clamp(Math.floor(b[1]*100+1e-8)+1,0));ys.add(clamp(Math.ceil(b[2]*100-1e-8)-1,1));ys.add(clamp(Math.floor(b[3]*100+1e-8)+1,1));}
  let best=null,bestDistance=Infinity;
  for(const xx of xs)for(const yy of ys){const px=xx/100,py=yy/100;if(blocks.some(b=>px>=b[0]-1e-10&&px<=b[1]+1e-10&&py>=b[2]-1e-10&&py<=b[3]+1e-10))continue;const d=(px-x)**2+(py-y)**2;if(d<bestDistance){bestDistance=d;best={x:px,y:py,adjusted:Math.hypot(px-x,py-y)>.015};}}
  return best;
}

function moveObject(i,x,y){const p=nearestPlacement(inputs(),i,x,y,placementConfig());if(!p){setStatus('没有可用位置','No free placement','warning');return false;}$(`x-${i}`).value=p.x.toFixed(2);$(`y-${i}`).value=p.y.toFixed(2);invalidate();if(p.adjusted)setStatus('已自动避让','Placement adjusted');return true;}
function pickButton(e){let b=e.target.closest('[data-select-object]');if(b)selectObject(Number(b.dataset.selectObject));}
$('placements').onclick=pickButton;$('object-buttons').onclick=pickButton;
$('truth-map').onpointerdown=e=>{
  if($('source').value!=='live'||!$('show-truth').checked||e.button!==0)return;
  let p=mapPoint(e);if(p.x<0||p.x>6||p.y<0||p.y>6)return;
  let object=e.target.closest('[data-object-index]');e.preventDefault();
  if(object){let i=Number(object.dataset.objectIndex),o=inputs()[i];selectObject(i);mapDrag={id:e.pointerId,index:i,x:o.x,y:o.y,start:p,screenX:e.clientX,screenY:e.clientY,moved:false};$('truth-map').setPointerCapture(e.pointerId);}
  else moveObject(Number($('edit').value),p.x,p.y);
};
$('truth-map').onpointermove=e=>{
  if(!mapDrag||mapDrag.id!==e.pointerId)return;
  if(!mapDrag.moved&&Math.hypot(e.clientX-mapDrag.screenX,e.clientY-mapDrag.screenY)<4)return;
  let p=mapPoint(e);mapDrag.moved=true;moveObject(mapDrag.index,mapDrag.x+p.x-mapDrag.start.x,mapDrag.y+p.y-mapDrag.start.y);
};
function finishMapDrag(e){if(!mapDrag||mapDrag.id!==e.pointerId)return;mapDrag=null;if($('truth-map').hasPointerCapture(e.pointerId))$('truth-map').releasePointerCapture(e.pointerId);}
$('truth-map').onpointerup=finishMapDrag;$('truth-map').onpointercancel=finishMapDrag;window.addEventListener('blur',()=>{mapDrag=null;});
$('truth-map').onkeydown=e=>{let o=e.target.closest('[data-object-index]');if(!o||$('source').value!=='live')return;let i=Number(o.dataset.objectIndex);if(e.key==='Enter'||e.key===' '){e.preventDefault();selectObject(i);$('truth-map').querySelector(`[data-object-index="${i}"]`)?.focus({preventScroll:true});}};
function stop(clear=true){running=false;epoch++;clearTimeout(timer);if(clear){result=null;render();}setStatus('已暂停','Paused');}
function invalidate(){$('source').value='live';$('input-panel').hidden=false;saveLayout();stop();$('error').hidden=true;setStatus('位置已更新','Layout changed');}
function placeAddedObjects(){
  let n=Number($('count').value);
  for(let i=layoutCount;i<n;i++){
    let previous=inputs().slice(0,i),usedKinds=previous.map(o=>o.kind),k=Number($('kind-'+i).value);
    if(usedKinds.includes(k)){$('kind-'+i).value=String([0,1,2].find(a=>!usedKinds.includes(a)));}
    let added=inputs()[i];
    const valid=(x,y)=>x>=added.size[0]/2+.01&&x<=6-added.size[0]/2-.01&&y>=added.size[1]/2+.01&&y<=6-added.size[1]/2-.01&&!(state.config.fixed_objects||[]).some(b=>Math.abs(b.x-x)<=(b.size[0]+added.size[0])/2+.08&&Math.abs(b.y-y)<=(b.size[1]+added.size[1])/2+.08)&&!previous.some(o=>(!state.config.supports_shared_zones&&Math.floor(o.x/2)===Math.floor(x/2)&&Math.floor(o.y/2)===Math.floor(y/2))||Math.abs(o.x-x)<=(o.size[0]+added.size[0])/2+.08&&Math.abs(o.y-y)<=(o.size[1]+added.size[1])/2+.08)&&![...liveTx,...state.config.rx_m].some(p=>Math.abs(p[0]-x)<=added.size[0]/2+.08&&Math.abs(p[1]-y)<=added.size[1]/2+.08);
    if(!valid(added.x,added.y))for(let z of [4,0,8,2,6,1,3,5,7]){let x=1+2*(z%3),y=1+2*Math.floor(z/3);if(valid(x,y)){$('x-'+i).value=x.toFixed(2);$('y-'+i).value=y.toFixed(2);break;}}
  }
  layoutCount=n;
}
$('count').onchange=()=>{placeAddedObjects();updateDimensions();invalidate();};
$('edit').onchange=()=>{saveLayout();render();};
function resetRecords(){let source=$('source').value;records=state.records.filter(r=>(r.layout_id||'default')===replayLayoutId()).filter(r=>source==='boundary'?r.split==='boundary_test':source.startsWith('shared-')?r.shared&&r.count===Number(source.slice(7))&&r.split==='test':r.count===Number(source)&&r.split==='test');for(let i=records.length-1;i>0;i--){let j=Math.floor(Math.random()*(i+1));[records[i],records[j]]=[records[j],records[i]];}cursor=0;}
$('source').onchange=()=>{stop();if($('source').value!=='live'&&!replayLayoutId()){$('source').value='live';$('error').hidden=false;$('error').textContent=t('请先选择已验证设备布局，再使用测试回放。','Select a validated radio preset before test replay.');}$('input-panel').hidden=$('source').value!=='live';resetRecords();render();setStatus('就绪','Ready');};
async function api(path,body){let response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await response.json();if(!response.ok)throw Error(d.error||response.statusText);return d;}
async function detect(){
  if(busy)return;
  if($('source').value==='live'){let i=Number($('edit').value),a=inputs()[i],p=nearestPlacement(inputs(),i,a.x,a.y,placementConfig());if(p){$(`x-${i}`).value=p.x.toFixed(2);$(`y-${i}`).value=p.y.toFixed(2);if(p.adjusted){result=null;saveLayout();render();}}}
  busy=true;let token=epoch,continuous=running;$('error').hidden=true;$('once').disabled=true;setStatus('检测中…','Processing…','running');
  try{let body={seed:Math.floor(Math.random()*2147483647),snr_db:Number($('snr').value)},path;
    if($('source').value==='live'){body.objects=inputs();if(state.config.home_environment)body.tx_positions=liveTx;path='/api/simulate';}else{if(!replayLayoutId())throw Error(t('请选择已验证布局','Select a validated layout'));if(!records.length)resetRecords();if(!records.length)throw Error(t('没有匹配的测试场景','No matching test scenes'));body.id=records[cursor++%records.length].id;body.layout_id=replayLayoutId();path='/api/record';}
    let d=await api(path,body);if(token!==epoch)return;d.timestamp=new Date().toISOString();result=d;history.push(d);if(history.length>10000)history.shift();render();renderHistory();
    if(continuous){runDone++;let limit=$('limit').value==='unlimited'?Infinity:Number($('limit').value);if(runDone>=limit){running=false;setStatus('已完成 '+runDone+' 次','Completed '+runDone);}else setStatus('检测 '+runDone+'/'+(Number.isFinite(limit)?limit:'∞'),'Processing '+runDone+'/'+(Number.isFinite(limit)?limit:'∞'),'running');}
    else setStatus('检测完成','Complete');
  }catch(e){if(token===epoch){result=null;running=false;render();$('error').hidden=false;$('error').textContent=englishMessage(e.message);setStatus('请检查摆放','Check input','warning');}}
  finally{busy=false;$('once').disabled=false;$('start').disabled=running;if(running)timer=setTimeout(detect,1500);}
}
$('start').onclick=()=>{if(running||busy)return;runDone=0;epoch++;running=true;detect();};
$('pause').onclick=()=>{stop(false);setStatus(result?'已暂停 · 快照':'已暂停',result?'Paused · snapshot':'Paused','warning');};$('once').onclick=()=>{stop(false);detect();};
$('snr').onchange=()=>{stop();setStatus('信号设置已更新','Signal settings changed');};$('show-truth').onchange=()=>{render();renderHistory();};$('view3d').onchange=render;
$('output-mode').onchange=()=>{if(occupancyMode())$('view3d').checked=false;$('view3d').disabled=occupancyMode();render();renderEvaluation();renderHistory();};
$('export').onclick=()=>{let payload={scope:state.config.supports_shared_zones?'Shared center zones supported; physical object count and occupied-zone count are separate signal-derived predictions. Within-zone individual positions/dimensions are not recovered.':'Multi-object simulated signals; distinct center zones. Individual dimensions are not recovered.',model_version:state.model_version,active_layout:result?.model_layout_id||$('trained-layout').value,active_revision:result?.inference_model_revision||(state.config.available_layouts||[]).find(e=>e.id===$('trained-layout').value)?.revision,models:{occupancy:(state.config.available_layouts||[]).find(e=>e.id===(result?.model_layout_id||$('trained-layout').value))?.evaluation?.selected_model||state.occupancy_results.selected_model,experimental_size:state.results.selected_model},display_mode:$('output-mode').value,config:state.config,history};let url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=url;a.download='multi-object-run.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('upload-button').onclick=async()=>{
  stop();if(busy){setStatus('检测中，请稍后上传','Processing; retry upload shortly','warning');return;}
  const token=epoch;
  try{let file=$('upload').files[0];if(!file)throw Error(t('请选择JSON文件。','Choose a JSON file.'));if(file.size>8000000)throw Error('File too large');busy=true;$('error').hidden=true;setStatus('读取信号…','Reading signal…','running');
    const payload=JSON.parse(await file.text());if(token!==epoch)return;
    let d=await api('/api/predict',payload);if(token!==epoch)return;
    d.timestamp=new Date().toISOString();result=d;history.push(d);if(history.length>10000)history.shift();render();renderHistory();setStatus('上传结果 · 快照','Uploaded · snapshot');
  }catch(e){if(token===epoch){$('error').hidden=false;$('error').textContent=englishMessage(e.message);setStatus('上传失败','Upload failed','warning');}}
  finally{busy=false;$('start').disabled=running;}
};
$('signal-download').onclick=async e=>{e.preventDefault();try{if(!replayLayoutId())throw Error(t('请选择已验证布局以下载对应示例','Select a validated layout to download its signal example'));let r=await fetch('/api/example-json?layout_id='+encodeURIComponent(replayLayoutId()));if(!r.ok)throw Error('Cannot load signal example');let url=URL.createObjectURL(await r.blob()),a=document.createElement('a');a.href=url;a.download='multi_signal_only.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1500);}catch(error){$('error').hidden=false;$('error').textContent=englishMessage(error.message);}};

function map(truth){
  let show=$('show-truth').checked,cfg={...state.config,tx_m:result?result.tx_positions||state.config.tx_m:(liveTx||state.config.tx_m)},d=result,X=x=>40+x*60,Y=y=>400-y*60;
  let editable=truth&&show&&$('source').value==='live';
  let s='<svg viewBox="0 0 440 440" '+(editable?'class="interactive"':'')+' role="'+(editable?'group':'img')+'" aria-label="'+(truth?'Reference multi-object scene':'Model inferred multi-object scene')+'"><rect x="40" y="40" width="360" height="360" fill="#102333" stroke="#5c7b93" stroke-width="2"/>';
  for(let z=0;z<9;z++){let x=z%3,y=Math.floor(z/3);s+=`<rect x="${X(x*2)}" y="${Y(y*2+2)}" width="120" height="120" fill="none" stroke="#2d4558"/><text x="${X(x*2)+8}" y="${Y(y*2+2)+18}" fill="#7593aa" font-size="13">Z${z+1}</text>`;}
  for(let b of cfg.fixed_objects||[]){s+=`<rect x="${X(b.x-b.size[0]/2)}" y="${Y(b.y+b.size[1]/2)}" width="${b.size[0]*60}" height="${b.size[1]*60}" fill="#657d90" fill-opacity=".6" stroke="#b5c2cd" pointer-events="none"><title>${esc(t('固定背景柜体，不计入待测物体','Fixed background cabinet, excluded from targets'))}</title></rect><text x="${X(b.x)}" y="${Y(b.y)+3}" fill="#e1e8ed" font-size="9" text-anchor="middle" pointer-events="none">BG</text>`;}
  if(cfg.home_environment)s+=`<path d="M${X(2.1)},40 H${X(3.9)}" stroke="#70bdf0" stroke-width="4"><title>${esc(t('玻璃窗','Glass window'))}</title></path><path d="M${X(2.5)},400 H${X(3.4)}" stroke="#cead78" stroke-width="4"><title>${esc(t('木门','Wooden door'))}</title></path>`;
  if(editable){let size=inputs()[Number($('edit').value)].size,pad=cfg.clearance_m??.08;for(let p of [...cfg.tx_m,...cfg.rx_m]){let a=Math.max(0,p[0]-size[0]/2-pad),b=Math.min(6,p[0]+size[0]/2+pad),c=Math.max(0,p[1]-size[1]/2-pad),e=Math.min(6,p[1]+size[1]/2+pad);s+=`<rect class="device-keepout" aria-hidden="true" x="${X(a)}" y="${Y(e)}" width="${(b-a)*60}" height="${(e-c)*60}" fill="#ffbf73" fill-opacity=".065" stroke="#ffbf73" stroke-opacity=".2" stroke-dasharray="2,3" pointer-events="none"><title>${esc(t('所选物体中心避让区','Selected-object center clearance'))}</title></rect>`;}}
  if(!truth&&occupancyMode()&&d?.occupancy){
    for(let o of d.occupancy.regions){let z=o.zone,col=z%3,row=Math.floor(z/3),actual=show&&d.truth?d.truth.filter(a=>a.zone===z).length:null,correct=actual==null?null:o.object_count!=null?actual===o.object_count:actual>0,c=o.accepted?(correct===false?'#ff977e':'#67dcc4'):'#ffbf73';s+=`<rect x="${X(col*2)}" y="${Y(row*2+2)}" width="120" height="120" fill="${c}" fill-opacity=".2" stroke="${c}" stroke-width="2" ${!o.accepted?'stroke-dasharray="5,3"':''}/>`;if(o.object_count!=null)s+=`<text x="${X(col*2+1)}" y="${Y(row*2+1)-6}" text-anchor="middle" fill="${c}" font-size="20"><title>${esc(t(o.object_count+'件物体；下行为区域占用概率',o.object_count+' '+(o.object_count===1?'object':'objects')+'; occupied-zone probability below'))}</title>× ${o.object_count}${!o.accepted?' ?':''}</text>`;s+=`<text x="${X(col*2+1)}" y="${Y(row*2+1)+(o.object_count!=null?15:5)}" text-anchor="middle" fill="${c}" font-size="${o.object_count!=null?11:15}">${pct(o.probability)}${!o.accepted&&o.object_count==null?' ?':''}</text>`;}
  }
  let os=truth?(show?(d?d.truth||[]:$('source').value==='live'?inputs():[]):[]):occupancyMode()?[]:d?.objects||[];
  os.forEach((o,index)=>{
    let k=o.kind,size=truth?o.size:o.size_m,xy=truth?[o.x,o.y]:o.center_m;
    if(!xy||!Number.isFinite(xy[0])||!Number.isFinite(xy[1]))return;
    let accepted=truth||o.accepted,color=truth?'#bc99fa':!o.accepted?'#ffbf73':show&&o.correct_zone_and_size_band===false?'#ff977e':'#67dcc4';
    let x=X(xy[0]-size[0]/2),y=Y(xy[1]+size[1]/2),w=size[0]*60,h=size[1]*60,high=$('view3d').checked?size[2]*23:0;
    let selected=editable&&index===Number($('edit').value);
    s+=`<g class="map-object${selected?' is-selected':''}" ${editable?`data-object-index="${index}" role="button" tabindex="0" aria-label="${esc('ABC'[index]+' · '+names()[k])}" aria-pressed="${selected}"`:''}>`;
    s+=`<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${color}" fill-opacity=".2" stroke="${color}" stroke-width="2" ${truth||!accepted?'stroke-dasharray="5,3"':''}/>`;
    if(high)s+=`<path d="M${x},${y} v-${high} h${w} v${h} h-${w} v-${h} M${x},${y+h} v-${high} M${x+w},${y+h} v-${high}" fill="${color}" fill-opacity=".12" stroke="${color}"/>`;
    s+=`<text x="${X(xy[0])}" y="${Y(xy[1])+4-high}" text-anchor="middle" fill="${color}" font-size="12">${editable?'ABC'[index]+' · ':''}${['S','M','L'][k]}${!truth&&!accepted?' ?':''}</text></g>`;
  });
  for(let [label,ps,color]of[['T',cfg.tx_m,'#68bbff'],['R',cfg.rx_m,'#ffbf73']])ps.forEach((p,i)=>{s+=`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="4" fill="${color}"/><text x="${X(p[0])+6}" y="${Y(p[1])-5}" fill="${color}" font-size="9">${label}${i+1}</text>`;});
  s+='<text x="220" y="424" text-anchor="middle" fill="#7593aa" font-size="13">x · 6 m</text><text x="15" y="220" text-anchor="middle" transform="rotate(-90,15,220)" fill="#7593aa" font-size="13">y · 6 m</text></svg>';return s;
}
function render(){
  if(!state)return;let d=result,show=$('show-truth').checked,o=d?.occupancy,active=occupancyMode()?o:d,s=active?.status,tone=s==='Accepted regions'||s==='Accepted'?'success':s?'warning':'idle';
  $('truth-map').innerHTML=map(true);$('pred-map').innerHTML=map(false);
  let editing=$('source').value==='live'&&show,selected=Number($('edit').value),inputObjects=inputs();
  $('input-panel').hidden=$('source').value!=='live';$('placement-details').hidden=$('source').value!=='live';
  $('object-buttons').hidden=!editing;
  $('object-buttons').innerHTML=inputObjects.map((a,i)=>'<button type="button" data-select-object="'+i+'" class="object-pick'+(i===selected?' is-selected':'')+'" aria-pressed="'+(i===selected)+'">'+'ABC'[i]+' · '+esc(names()[a.kind])+'</button>').join('');
  for(let i=0;i<3;i++)$('row-'+i).classList.toggle('selected-object',i===selected);
  $('edit-hint').textContent=editing?t('拖动自动避让 · 阴影为中心禁放区','Drag to place · shaded center clearance'):t('参考位置已隐藏','Reference hidden');
  $('truth-caption').textContent=!show?t('参考已隐藏','Reference hidden'):d&&!d.truth?t('无参考位置','No reference positions'):d?.truth?d.truth.map(a=>['S','M','L'][a.kind]+' · '+zone(a.zone)).join('   /   '):$('source').value==='live'?t('6 × 6 m · 可拖动摆放','6 × 6 m · drag to place'):t('测试场景 · 只读','Test scene · read-only');
  $('prediction-badge').textContent=statusLabel(s);$('prediction-badge').dataset.tone=tone;
  $('pred-caption').textContent=d?.source_positions_changed&&!d.model_position_validated?t('信号源已移动 · 预测未验证','Moved transmitters · prediction unvalidated'):!d?t('点击“开始检测”','Press Start'):s==='Unknown'?t('信号不可用','Signal unavailable'):!occupancyMode()?t('尺寸示意（实验）','Prototype sizes (experimental)'):s==='Uncertain'?(o?.reason==='nonstandard_frame_count_not_16_frame_validated'?t('需要16帧输入','Requires 16 frames'):t('候选结果 · 置信度不足','Candidate · low confidence')):t('占用中心区域','Occupied center zones');
  if(d?.model_layout_id&&d.model_layout_id!=='unvalidated')$('pred-caption').textContent+=' · '+(d.model_layout_id==='t1_v1'?t('T1预设','T1 preset'):t('默认布局','Default layout'));
  let regions=occupancyMode()?(o?.regions||[]).map(a=>zone(a.zone)):(d?.objects||[]).map(a=>zone(a.zone));
  let confidence=occupancyMode()?(o?.joint_confidence??o?.layout_confidence):d?.heads?.length?Math.min(...d.heads.map(a=>a.confidence)):null;
  let items=[[t('预测区域','Predicted zones'),regions.length?regions.join(' · '):'—'],[t('预测数量','Count'),(occupancyMode()?o?.estimated_region_count:d?.estimated_count)??'—'],[occupancyMode()?t('布局置信度','Layout confidence'):t('最低档位分数','Lowest head score'),s==='Unknown'?'—':pct(confidence)],[t('判断状态','Status'),statusLabel(s)]];
  if(state.config.separate_object_count&&occupancyMode())items=[[t('预测区域','Predicted zones'),regions.length?regions.join(' · '):'—'],[state.config.home_environment?t('待测物体数','Targets'):t('物体数','Objects'),o?.estimated_object_count??'—'],[t('占用区域数','Occupied zones'),o?.estimated_region_count??'—'],[t('联合置信度','Joint confidence'),s==='Unknown'?'—':pct(confidence)],[t('判断状态','Status'),statusLabel(s)]];
  $('decision').classList.toggle('separate-counts',!!state.config.separate_object_count);
  $('decision').innerHTML=items.map(([k,v],i)=>'<div><span class="key">'+k+'</span><strong '+(i===0?'class="region-value" data-tone="'+tone+'"':'')+'>'+v+'</strong></div>').join('');
  $('confidence-summary').textContent=d?(active?.elapsed_ms?.toFixed(1)+' ms'+(d.frames?' · '+d.frames+' '+t('帧','frames'):'')):'';
  $('alternatives').textContent=occupancyMode()&&o?.alternative_layouts?t('候选：','Candidates: ')+o.alternative_layouts.map(a=>a.zones.map(z=>zone(z)+(a.per_zone_object_count?' ×'+a.per_zone_object_count[z]:'')).join(' + ')+' ('+pct(a.probability)+')').join(' · '):'';
  $('window-info').textContent=state.model_version>=2?t('16帧静态CSI · 32链路','16-frame static CSI · 32 links')+(o?.feature_view?' · '+(o.feature_view==='phase_free'?t('相位对齐分支','Phase-aligned branch'):o.feature_view==='ordinal_count_v6'?t('区域件数分类','Ordinal zone counts'):o.feature_view==='count_refiner'?t('区域计数增强','Per-zone count refinement'):o.feature_view==='fusion'?t('融合特征','Fused features'):t('校准分支','Calibrated branch')):''):'';
  $('heads').innerHTML=occupancyMode()?Array.from({length:9},(_,z)=>{let p=o?.probabilities?.[z]||0;return '<article class="head"><strong>'+zone(z)+' · '+pct(p)+'</strong><div class="bar"><i style="width:'+100*p+'%"></i></div></article>';}).join(''):Array.from({length:3},(_,k)=>{let h=d?.heads?.[k];return '<article class="head"><strong>'+names()[k]+'</strong><p>'+(h?(h.raw_label?zone(h.raw_label-1):t('未检测到','Not detected'))+' · '+pct(h.confidence):'—')+'</p></article>';}).join('');
  if(state.config.separate_object_count&&occupancyMode())$('heads').innerHTML=[1,2,3].map(n=>{let p=o?.count_probabilities?.[n-1]||0;return '<article class="head"><strong>'+t(n+'件物体',n+' objects')+' · '+pct(p)+'</strong><div class="bar"><i style="width:'+100*p+'%"></i></div></article>';}).join('')+$('heads').innerHTML;
  const activeLayout=d?.model_layout_id||($('trained-layout').value||'unvalidated');$('model-binding').textContent=activeLayout==='unvalidated'?t('当前设备布局未验证','Current radio layout is unvalidated'):t('当前结果模型：','Result model: ')+(activeLayout==='t1_v1'?t('T1预设','T1 preset'):t('默认布局','Default layout'))+(d?.inference_model_revision==='spatial_count_v6'?' · '+t('计数分类版','Count classifier'):d?.inference_model_revision==='t1_count_v3'?' · '+t('计数增强版','Enhanced counts'):d?.inference_model_revision==='t1_fusion_v2'?' · '+t('融合版','Fusion'):'');
  renderEvaluation();
}
function renderHistory(){
  let show=$('show-truth').checked,matched=d=>{if(!d.truth)return null;if(d.occupancy?.per_zone_object_count){let c=Array(9).fill(0);d.truth.forEach(o=>c[o.zone]++);return JSON.stringify(c)===JSON.stringify(d.occupancy.per_zone_object_count);}let a=[...new Set(d.truth.map(o=>o.zone))].sort((a,b)=>a-b),b=(d.occupancy?.regions||[]).map(o=>o.zone).sort((a,b)=>a-b);return JSON.stringify(a)===JSON.stringify(b)&&(d.occupancy?.estimated_object_count==null||d.occupancy.estimated_object_count===d.true_count);};
  let recent=history.slice(-25),lid=result?.model_layout_id||$('trained-layout').value,revision=result?.inference_model_revision||(state.config.available_layouts||[]).find(e=>e.id===lid)?.revision;let checkable=recent.filter(d=>d.truth&&d.occupancy&&d.model_layout_id===lid&&lid!=='unvalidated'&&(!revision||d.inference_model_revision===revision)),correct=checkable.filter(d=>matched(d)).length;
  $('history-count').textContent=t(history.length+'条',history.length+' records');
  $('summary').textContent=t('最近25条','Latest 25')+(show&&occupancyMode()&&checkable.length?' · '+t('当前布局匹配 ','Current-layout matches ')+correct+'/'+checkable.length+' ('+pct(correct/checkable.length)+')':'');
  $('history').innerHTML=history.slice(-25).reverse().map((d,i)=>{let o=d.occupancy,match=occupancyMode()?matched(d):d.exact_layout_correct;return `<tr><td>${history.length-i}</td><td>${esc(d.model_layout_id==='t1_v1'?(d.inference_model_revision==='t1_count_v3'?t('T1增强版','T1 enhanced'):t('T1预设','T1 preset')):d.model_layout_id==='unvalidated'?t('未验证','Unvalidated'):t('默认','Default'))}</td><td>${show?(d.true_count??'—'):'—'}</td><td>${(occupancyMode()?(o?.estimated_object_count??o?.estimated_region_count):d.estimated_count)??'Unknown'}</td><td>${occupancyMode()?(o?.confirmed_objects??o?.confirmed_regions):d.confirmed_count}</td><td>${statusLabel(occupancyMode()?o?.status:d.status)}</td><td>${(occupancyMode()?o?.elapsed_ms:d.elapsed_ms)?.toFixed(1)} ms</td><td>${show&&match!=null?(match?t('匹配','Match'):t('不匹配','Mismatch')):'—'}</td></tr>`;}).join('');
}
function renderEvaluation(){
  let metricsLayout=result?.model_layout_id&&result.model_layout_id!=='unvalidated'?result.model_layout_id:$('trained-layout').value;
  let chosenLayout=(state.config.available_layouts||[]).find(e=>e.id===metricsLayout);
  const unavailable=state.config.home_environment&&!chosenLayout;
  $('evaluation').closest('table').hidden=unavailable;
  if(unavailable){$('stats').innerHTML='';$('evaluation').innerHTML='';$('comparison').innerHTML='';$('coverage').textContent='';$('before-after').hidden=true;$('evaluation-summary').textContent=t('未验证布局 · 暂无测试分数','Unvalidated layout · no test scores');$('evaluation-note').textContent=t('选择已验证设备布局可查看对应测试结果。','Select a validated radio preset to view its test results.');return;}

  let r=state.results,occ=chosenLayout?.evaluation||state.occupancy_results,m=occ.splits.test;
  let counts=occ.split_counts||r.split_counts,shared=!!state.config.separate_object_count;
  $('stats').innerHTML=[[t('训练场景','Training scenes'),counts.train,t('独立测试 '+m.n,'Held-out test '+m.n)],[t('区域 Macro-F1','Zone Macro-F1'),pct(m.occupied_zone_macro_f1),t('占用区域','Occupied zones')],[t('物体数准确率','Object count accuracy'),pct(m.count_accuracy),shared?t('独立学习物体数量','Learned physical count'):t('中心区域互不相同','Distinct center zones')],[shared?t('联合正确率','Zones + objects'):t('完整布局','Exact layout'),pct(shared?m.exact_joint_accuracy:m.exact_occupied_layout_accuracy),shared?t('每个区域件数均正确','Every zone count correct'):t('所有区域同时正确','All zones match')]].map(([k,v,s])=>`<article class="stat"><div class="label">${k}</div><div class="value">${v}</div><div class="sub">${s}</div></article>`).join('');
  let rows=[[t('全部测试','All test scenes'),m],[t('2个物体','2 objects'),m.by_true_count['2']],[t('3个物体','3 objects'),m.by_true_count['3']]];
  if(state.config.home_environment)rows.splice(1,0,[t('1个物体','1 object'),m.by_true_count['1']]);
  if(shared)for(let [key,zh,en] of [['distinct','不同区布局','Distinct-zone layouts'],['shared','存在同区物体','Shared-zone layouts'],['all_in_one_zone','全部物体同区','All objects in one zone']])if(m.by_sharing?.[key])rows.push([t(zh,en),m.by_sharing[key]]);
  const tableRow=([name,a])=>`<tr><td>${name}</td><td>${a.n}</td><td>${pct(a.occupied_zone_macro_f1)}</td><td>${pct(a.count_accuracy)}</td><td>${pct(a.exact_occupied_layout_accuracy)}</td>${shared?'<td>'+pct(a.exact_joint_accuracy)+'</td>':''}</tr>`;
  $('joint-col').hidden=!shared;$('baseline-joint-col').hidden=!shared;
  $('evaluation').innerHTML=rows.map(tableRow).join('');
  $('evaluation-note').textContent=shared?t('物体数为总数；联合指标要求每个区域的物体数均正确。包含全部测试。','Object count is the total. Joint accuracy requires the correct count in every zone; all test scenes included.'):t('区域F1、物体数与完整布局为不同指标；包含全部测试。','Zone F1, count and exact layout differ; all test scenes included.');
  $('evaluation-summary').textContent=(chosenLayout?.id==='t1_v1'?t('T1预设 · ','T1 preset · '):state.config.home_environment?t('默认布局 · ','Default layout · '):'')+'F1 '+pct(m.occupied_zone_macro_f1)+' · '+(shared?t('联合 ','Joint '):t('布局 ','Layout '))+pct(shared?m.exact_joint_accuracy:m.exact_occupied_layout_accuracy);
  $('coverage').textContent=state.model_version>=2?t('通过门限：','Confidence-gated subset: ')+pct(m.accepted_layout_coverage)+(shared?t('；子集联合准确率：','; joint accuracy within subset: '):t('；子集布局准确率：','; layout accuracy within subset: '))+pct(m.accepted_layout_accuracy):'';
  $('comparison-scope').textContent=state.config.home_environment?t('同批新场景；旧权重使用相同背景标定，不重新训练。','Same new scenes; previous weights use the same background calibration without retraining.'):t('同批 '+m.n+' 个独立测试；交界测试单独报告。','Same '+m.n+' held-out scenes; boundary stress test reported separately.');
  $('before-after').hidden=state.model_version<2;
  if(state.model_version===2){let base=occ.baseline_comparison.test;$('baseline-table').innerHTML=[[t('旧模型：单帧','Old model: single frame'),base.v1_single_frame],[t('旧模型：同样16帧平均','Old model: same 16-frame average'),base.v1_same_16_frame_average],[t('优化模型：16帧','Optimized model: 16 frames'),m],[t('优化模型：交界压力测试','Optimized: boundary stress test'),occ.splits.boundary_test]].map(([name,a])=>`<tr><td>${name}</td><td>${a.n}</td><td>${pct(a.occupied_zone_macro_f1)}</td><td>${pct(a.count_accuracy)}</td><td>${pct(a.exact_occupied_layout_accuracy)}</td></tr>`).join('');}
  if(state.model_version>=3){let base=occ.baseline_comparison.test.previous_t1_model||occ.baseline_comparison.test.v4_same_background_reference||occ.baseline_comparison.test.v3_region_count_as_object_count||occ.baseline_comparison.test.v2_same_frames;$('baseline-table').innerHTML=[[t(occ.baseline_comparison.test.previous_t1_model&&chosenLayout?.id==='default'?'上一版默认布局：同批新测试':occ.baseline_comparison.test.previous_t1_model?'上一版T1：同批新测试':occ.baseline_comparison.test.v4_same_background_reference?'V4：相同新测试':'上一版：相同新测试',occ.baseline_comparison.test.previous_t1_model&&chosenLayout?.id==='default'?'Previous default: paired fresh test':occ.baseline_comparison.test.previous_t1_model?'Previous T1: paired fresh test':occ.baseline_comparison.test.v4_same_background_reference?'V4: same new test':'Previous: same new test'),base],[t('当前模型','Current model'),m],[t('当前：交界压力测试','Current: boundary stress test'),occ.splits.boundary_test]].map(tableRow).join('');}
  let selected=occupancyMode()?occ:r;
  let decoder=selected.selected_decoder;
  let chosen=state.model_version===2&&occupancyMode()?(decoder?'count='+decoder.count_weight+', layout='+decoder.layout_weight+', bias='+decoder.odds_bias:selected.comparison[0].name):selected.selected_model;
  $('comparison').innerHTML=(selected.comparison||[]).map(c=>`<div class="comparison"><span>${esc(c.name)}${c.selected||c.name===chosen?' ✓':''}</span><strong>${pct(occupancyMode()?c.validation.occupied_zone_macro_f1:c.validation.typed_object_f1)}</strong></div>`).join('');
}
async function load(){try{let response=await fetch('/api/state'),d=await response.json();if(!response.ok)throw Error('Service unavailable');if(!d.ready){$('loading').textContent=t('模型准备中…','Preparing model…');setTimeout(load,2500);return;}state=d;$('loading').hidden=true;$('app').hidden=false;initInputs();$('view3d').disabled=occupancyMode();translate();setStatus('就绪','Ready');}catch(e){$('loading').textContent=englishMessage(e.message);}}
translate();load();
let layoutPollBusy=false;
setInterval(async()=>{if(!state||layoutPollBusy)return;layoutPollBusy=true;try{const r=await fetch('/api/layouts');if(!r.ok)return;const stamps=(await r.json()).layouts,previous=(state.config.available_layouts||[]).map(({id,ready,revision})=>({id,ready,revision}));if(JSON.stringify(stamps)===JSON.stringify(previous))return;const full=await fetch('/api/state');if(!full.ok)return;const fresh=await full.json(),active=result?.model_layout_id||$('trained-layout').value,old=previous.find(e=>e.id===active),next=stamps.find(e=>e.id===active),changed=old&&JSON.stringify(old)!==JSON.stringify(next);if(changed)stop();state.config.available_layouts=fresh.config.available_layouts;state.records=fresh.records;initLayoutPresets();resetRecords();render();renderHistory();if(changed)setStatus('模型已更新，请重新检测','Model updated; start a new detection');}catch(e){}finally{layoutPollBusy=false;}},15000);

function renderTxFields(){if(!state)return;$('tx-fields').innerHTML=(liveTx||state.config.tx_m).map((p,i)=>`<tr><td>T${i+1}</td>${p.map((v,j)=>`<td><input id="tx-${i}-${j}" aria-label="T${i+1} ${['x','y','z'][j]}" type="number" min="${j===2?.1:.08}" max="${j===2?2.9:5.92}" step=".01" value="${v}" style="width:100%;max-width:140px"></td>`).join('')}</tr>`).join('');}
function applyTx(reset=false,preset=null){
 if(busy){setStatus('检测中，请稍后应用','Detection in progress; retry shortly','warning');return;}
 const replayPreset=(preset||reset)&&$('source').value!=='live';
 const ps=preset?preset.tx_positions.map(p=>[...p]):reset?state.config.tx_m.map(p=>[...p]):[0,1,2,3].map(i=>[0,1,2].map(j=>Number($(`tx-${i}-${j}`).value)));
 try{
 if(ps.some(p=>p.some((v,j)=>!Number.isFinite(v)||v<(j===2?.1:.08)||v>(j===2?2.9:5.92))))throw Error(t('坐标超出房间范围','Coordinates outside room'));
 for(let i=0;i<4;i++){
  if([...ps.slice(0,i),...state.config.rx_m].some(q=>Math.hypot(...ps[i].map((v,j)=>v-q[j]))<.08))throw Error(t('设备位置过近','Devices too close'));
  if([...(replayPreset?[]:inputs()),...(state.config.fixed_objects||[])].some(o=>Math.abs(ps[i][0]-o.x)<=o.size[0]/2+.08&&Math.abs(ps[i][1]-o.y)<=o.size[1]/2+.08))throw Error(t('信号源不可放入物体或柜体','Transmitter overlaps an object or cabinet'));
 }
 liveTx=ps;renderLayoutScore();if(replayPreset){stop();resetRecords();render();renderHistory();}else invalidate();renderTxFields();renderLayoutScore();setStatus(reset?'已恢复默认信号源':'信号源已更新',reset?'Default transmitters restored':'Transmitters updated');
 }catch(e){$('error').hidden=false;$('error').textContent=englishMessage(e.message);}
}
$('tx-apply').onclick=()=>applyTx();$('tx-reset').onclick=()=>applyTx(true);

function initLayoutPresets(){
 const es=state.config.available_layouts||[];
 $('trained-layout').innerHTML=`<option value="" disabled>${esc(t('自定义位置（未验证）','Custom positions (unvalidated)'))}</option>`+es.map(e=>`<option value="${e.id}" ${e.ready?'':'disabled'}>${esc(e.revision==='spatial_count_v6'?t(e.id==='default'?'默认 · 计数分类版':'T1 · 计数分类版',e.id==='default'?'Default · count classifier':'T1 · count classifier'):e.id==='default'?t('默认布局','Default layout'):e.revision==='t1_count_v3'?t('T1 · 增强版','T1 · enhanced'):t('新位置 T1','New T1 layout'))}${e.ready?'':t('（训练中）',' (training)')}</option>`).join('');
 $('trained-layout').onchange=()=>{const e=es.find(e=>e.id===$('trained-layout').value);if(!e?.ready)return;if(busy){renderLayoutScore();setStatus('检测中，请稍后切换','Detection in progress; retry shortly','warning');return;}applyTx(false,e);renderTxFields();renderLayoutScore();};renderLayoutScore();
}
function renderLayoutScore(){if(!state)return;const e=(state.config.available_layouts||[]).find(e=>e.tx_positions.every((p,i)=>p.every((v,j)=>Math.abs(v-liveTx[i][j])<1e-7)));
 $('trained-layout').value=e?.ready?e.id:'';
 $('layout-score').textContent=e?.ready?(e.metrics?t('新布局独立测试','New-layout held-out test')+`: N=${e.metrics.n} · F1 ${pct(e.metrics.occupied_zone_macro_f1)} · `+t('数量','Count')+` ${pct(e.metrics.count_accuracy)} · `+t('完整空间计数','Exact spatial counts')+` ${pct(e.metrics.exact_joint_accuracy)}`:t('默认布局 · 原模型','Default layout · original model')):t('此设备布局未验证','This radio layout is unvalidated');renderEvaluation();
}
