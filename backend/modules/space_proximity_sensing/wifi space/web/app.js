'use strict';
const $=id=>document.getElementById(id);
let lang=localStorage.getItem('static-language')||'both', experiment=null, current=null, history=[], running=false, busy=false, pendingSingle=false, epoch=0, timer=null, cursor=0, order=[];
let runDone=0,runTarget=20;
const t=(zh,en)=>lang==='zh'?zh:lang==='en'?en:zh+' / '+en;
const pct=x=>(100*x).toFixed(1)+'%';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const zone=z=>z==null?'Unknown':'Z'+(z+1);
const decisionLabel=d=>d?.prediction!=null?zone(d.prediction):(d?.reason==='ambiguous_zones'&&d.candidate_zones?.length<=3?t('候选 ','Candidates ')+d.candidate_zones.map(zone).join(' / '):'Unknown');
function translate(){
  document.body.dataset.language=lang;document.documentElement.lang=lang==='en'?'en':'zh-CN';
  document.querySelectorAll('[data-zh][data-en]').forEach(e=>{
    if(lang==='both'&&e.tagName!=='OPTION'&&e.tagName!=='SPAN'&&e.tagName!=='TH')e.innerHTML=esc(e.dataset.zh)+'<span class="subenglish">'+esc(e.dataset.en)+'</span>';
    else e.textContent=lang==='zh'?e.dataset.zh:lang==='en'?e.dataset.en:e.dataset.zh+' / '+e.dataset.en;
  });
  if(experiment){renderEvaluation();renderCurrent();renderHistory();}
}
function protocolText(){
  if(experiment.results.protocol!==3)return;
  const set=(e,zh,en)=>{if(e){e.dataset.zh=zh;e.dataset.en=en;}};
  if(!$('source').querySelector('[value="boundary_test"]')){let option=document.createElement('option');option.value='boundary_test';set(option,'区域交界测试','Zone-boundary test');$('source').insertBefore(option,$('source').children[1]);}
  set(document.querySelector('.legend > span:last-child'),'外虚线：有效中心范围；阴影：设备禁放区','Outer dashes: center bounds; shaded: device exclusion');
  set(document.querySelector('[data-en^="Click the left map"]'),'在左图点击可生成新场景。新版覆盖房间内所有有效中心范围，阴影区禁止放置箱体。区域标签按物体中心定义；证据不足时仅显示候选区域。','Click the left map to generate a new scene. Full valid center bounds are covered; shaded device areas are forbidden. Labels refer to the object center; uncertain evidence produces candidate zones.');
  set(document.querySelector('.flow span[data-zh]'),'204 维信号特征','204 signal features');
  set(document.querySelector('[data-en^="Low confidence or features"]'),'新版使用独立数据调整分数与候选集合。候选区域不唯一、分数不足或特征超出参考范围时，不绘制确定位置。这仍不能保证识别陌生房间、空房间或多个物体。','Separate data adjusts scores and candidate sets. Multiple candidates, low scores or unusual features do not produce an accepted position. This still does not guarantee detection of new rooms, empty rooms or multiple objects.');
}
$('language').value=lang;$('language').onchange=()=>{lang=$('language').value;localStorage.setItem('static-language',lang);translate();};
async function api(path,body){let response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});let d=await response.json();if(!response.ok)throw Error(d.error||response.statusText);return d;}
function resetOrder(){order=experiment.records.filter(r=>r.split===$('source').value);for(let i=order.length-1;i>0;i--){let j=Math.floor(Math.random()*(i+1));[order[i],order[j]]=[order[j],order[i]];}cursor=0;}
function stop(clear=true){running=false;pendingSingle=false;epoch++;clearTimeout(timer);if(clear){current=null;renderCurrent();}$('run-status').textContent=t('已暂停 · 位置已清空','Paused · current decision cleared');}
function seed(){return Math.floor(Math.random()*2147483647);}
async function detect(){
  if(busy){pendingSingle=true;return;}busy=true;let requestEpoch=epoch;let continuous=running;let mode=$('source').value;
  $('error').hidden=true;$('run-status').textContent=t('处理中…','Processing…');$('once').disabled=true;
  try{
    let body={seed:seed(),snr_db:Number($('noise').value)},path;
    if(mode==='live'){body.x=Number($('x').value);body.y=Number($('y').value);path='/api/simulate';}
    else{if(!order.length)resetOrder();body.id=order[cursor++%order.length].id;path='/api/record';}
    let start=performance.now();let d=await api(path,body);d.roundtrip_ms=performance.now()-start;d.timestamp=new Date().toISOString();
    if(requestEpoch!==epoch)return;
    current=d;history.push(d);if(history.length>10000)history.shift();renderCurrent();renderHistory();
    if(continuous){runDone++;if(runDone>=runTarget){running=false;$('run-status').textContent=t('本轮已完成 · 最后一次结果快照','Run complete · last-result snapshot');}else $('run-status').textContent=t('持续检测 ','Processing ')+runDone+'/'+(Number.isFinite(runTarget)?runTarget:'∞');}
    else $('run-status').textContent=t('单次结果 · 快照','Single result · snapshot');
  }catch(e){if(requestEpoch===epoch){current=null;renderCurrent();$('error').hidden=false;$('error').textContent=e.message;running=false;$('run-status').textContent=t('输入失败 · Unknown','Input failed · Unknown');}}
  finally{busy=false;$('once').disabled=false;if(pendingSingle){pendingSingle=false;timer=setTimeout(detect,0);}else if(running)timer=setTimeout(detect,Number($('interval').value));}
}
$('start').onclick=()=>{if(running)return;let limit=$('run-limit').value;runDone=0;runTarget=limit==='unlimited'?Infinity:limit==='pass'?($('source').value==='live'?1:order.length):Number(limit);running=true;epoch++;detect();};
$('pause').onclick=()=>stop();
$('once').onclick=()=>{stop(false);detect();};
$('random').onclick=()=>{stop();$('source').value='test';resetOrder();detect();};
$('simulate').onclick=()=>{stop();$('source').value='live';detect();};
$('reset').onclick=()=>{stop();history=[];renderHistory();};
$('source').onchange=()=>{stop();resetOrder();};
$('noise').onchange=()=>{if(!running){current=null;renderCurrent();}};
$('upload-button').onclick=async()=>{
  stop();if(busy)return;
  try{let file=$('upload').files[0];if(!file)throw Error(t('先选择 JSON 文件。','Select a JSON file first.'));if(file.size>256000)throw Error('File too large');
    busy=true;$('error').hidden=true;let d=await api('/api/predict',JSON.parse(await file.text()));d.timestamp=new Date().toISOString();current=d;history.push(d);renderCurrent();renderHistory();$('run-status').textContent=t('上传信号 · 单次结果','Uploaded signal · single result');
  }catch(e){current=null;renderCurrent();$('error').hidden=false;$('error').textContent=e.message;}finally{busy=false;}
};
$('truth-visible').onchange=()=>{renderCurrent();renderHistory();};
$('three-d').onchange=renderCurrent;
$('export').onclick=()=>{let payload={scope:'Simulated CSI; one known-size object. Not real-world furniture reconstruction.',model:experiment.results.selected_model,threshold:experiment.results.threshold,config:experiment.config,history};let url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=url;a.download='static-obstacle-run.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};

function map(kind){
  let truth=kind==='truth',show=$('truth-visible').checked,d=current,cfg=experiment.config;
  const X=x=>40+x*60,Y=y=>400-y*60;
  let s='<svg viewBox="0 0 440 440" '+(truth?'class="interactive" role="img" aria-label="Ground truth placement map"':'role="img" aria-label="Signal-based predicted zone"')+'><rect x="40" y="40" width="360" height="360" rx="3" fill="#0e2030" stroke="#5a7a94" stroke-width="2"/>';
  for(let z=0;z<9;z++){
    let cx=1+2*(z%3),cy=1+2*Math.floor(z/3),p=d?.probabilities?.[z]||0;
    let selected=!truth&&d?.prediction===z,candidate=!truth&&d?.reason==='ambiguous_zones'&&d.candidate_zones?.includes(z);
    s+=`<rect x="${X(cx-1)}" y="${Y(cy+1)}" width="120" height="120" fill="${selected?'#21564f':candidate?'#f8bf70':'#68bfae'}" fill-opacity="${selected?.65:candidate?.15:truth?0:p*.25}" stroke="${candidate?'#f8bf70':'#2b4357'}" stroke-width="${candidate?2:1}" ${candidate?'stroke-dasharray="5,3"':''}/>`;
    if(truth&&cfg.coverage_mode!=='full_valid')s+=`<rect x="${X(cx-.38)}" y="${Y(cy+.48)}" width="45.6" height="57.6" rx="3" fill="none" stroke="#45647e" stroke-dasharray="3,4"/>`;
    s+=`<text x="${X(cx-1)+9}" y="${Y(cy+1)+19}" fill="${selected?'#8cefd9':'#6f8da6'}" font-size="12">Z${z+1}</text>`;
  }
  if(truth&&cfg.coverage_mode==='full_valid'){
    s+=`<rect x="${X(.61)}" y="${Y(5.59)}" width="${(5.39-.61)*60}" height="${(5.59-.41)*60}" fill="none" stroke="#557a85" stroke-dasharray="4,4"/>`;
    for(let p of [...(d?.device_positions?.tx||cfg.tx_m),...(d?.device_positions?.rx||cfg.rx_m)]){
      let x0=Math.max(.61,p[0]-.68),x1=Math.min(5.39,p[0]+.68),y0=Math.max(.41,p[1]-.48),y1=Math.min(5.59,p[1]+.48);
      if(x1>x0&&y1>y0)s+=`<rect x="${X(x0)}" y="${Y(y1)}" width="${(x1-x0)*60}" height="${(y1-y0)*60}" fill="#c97667" fill-opacity=".22" stroke="#a66b64" stroke-dasharray="2,3"/>`;
    }
  }
  let objectXY=truth?(show?d?.truth?.xy:null):d?.center_m;
  if(objectXY){let[x,y]=objectXY,left=X(x-.6),top=Y(y+.4);let c=truth?'#c59aff':show&&d.correct===false?'#f8bf70':'#65dcc2';let height=$('three-d').checked?28:0;
    s+=`<rect x="${left}" y="${top}" width="72" height="48" rx="3" fill="${c}" fill-opacity=".20" stroke="${c}" stroke-width="2" ${truth?'stroke-dasharray="6,3"':''}/>`;
    if(height){s+=`<path d="M${left},${top} l0,-${height} l72,0 l0,${height} M${left},${top-height} l0,48 l72,0 l0,-48 M${left},${top+48} l0,-${height} M${left+72},${top+48} l0,-${height}" fill="${c}" fill-opacity=".15" stroke="${c}" stroke-width="1.5"/>`;}
    s+=`<circle cx="${X(x)}" cy="${Y(y)}" r="3" fill="${c}"/>`;
  }
  for(let[k,positions]of[['TX',d?.device_positions?.tx||cfg.tx_m],['RX',d?.device_positions?.rx||cfg.rx_m]])positions.forEach((p,i)=>{let c=k==='TX'?'#64b9ff':'#f8bf70';s+=`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="5" fill="${c}" stroke="#0b1725" stroke-width="2"/><text x="${X(p[0])+8}" y="${Y(p[1])-7}" fill="${c}" font-size="9">${k}${i+1}</text>`;});
  s+='<text x="220" y="425" text-anchor="middle" fill="#7f9bb4" font-size="11">x · 6 m</text><text x="17" y="220" text-anchor="middle" fill="#7f9bb4" font-size="11" transform="rotate(-90,17,220)">y · 6 m</text><text x="40" y="418" fill="#68849c" font-size="9">0</text></svg>';
  return s;
}
function renderCurrent(){
  if(!experiment)return;
  $('truth-map').innerHTML=map('truth');$('prediction-map').innerHTML=map('prediction');
  $('truth-map').classList.toggle('iso',$('three-d').checked);$('prediction-map').classList.toggle('iso',$('three-d').checked);
  $('truth-map').querySelector('svg').onclick=e=>{
    let svg=e.currentTarget,p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;p=p.matrixTransform(svg.getScreenCTM().inverse());
    let x=Math.max(.61,Math.min(5.39,(p.x-40)/60)),y=Math.max(.41,Math.min(5.59,(400-p.y)/60));
    $('x').value=x.toFixed(2);$('y').value=y.toFixed(2);stop();$('source').value='live';detect();
  };
  let d=current,show=$('truth-visible').checked;
  $('truth-caption').textContent=d?.truth?(show?`${zone(d.truth.zone)} · (${d.truth.xy.map(x=>x.toFixed(2)).join(', ')}) m · ${d.truth.id}`:t('真值已隐藏，可用于盲测。','Ground truth hidden for a blind check.')):t('选择测试样本，或点击图中位置生成新信号。','Choose a test sample, or click to generate a new scene.');
  $('prediction-caption').textContent=d?(d.prediction!=null?t('信号判为 '+zone(d.prediction)+'；箱体绘制在区域中心，尺寸为已知先验。','Signal predicts '+zone(d.prediction)+'; box at zone center, dimensions known in advance.'):d.reason==='ambiguous_zones'?t('区域证据不够明确：'+d.candidate_zones.map(zone).join(' / ')+'。显示候选范围，不绘制确定物体位置。','Ambiguous evidence: '+d.candidate_zones.map(zone).join(' / ')+'. Candidate areas only; no accepted object position.'):t('Unknown：'+(d.reason==='outside_signal_reference_range'?'信号特征偏离参考范围。':'置信度不足或信号无效。'),'Unknown: '+(d.reason==='outside_signal_reference_range'?'signal features outside reference range.':'low confidence or invalid signal.'))):t('等待信号输入。暂停或输入失败后不保留有效位置。','Waiting for input. Pausing or failed input clears the current location.');
  if(d?.in_training_placement_range===false)$('truth-caption').textContent+=' · '+t('训练摆放范围外','Outside trained placement ranges');
  $('decision-badge').textContent=d?decisionLabel(d):'—';
  let pred=d?decisionLabel(d):'—',conf=d?pct(d.confidence):'—',err=d?.error_m!=null&&show?d.error_m.toFixed(2)+' m':'—';
  let effectiveThreshold=Math.max(experiment.results.threshold,experiment.results.set_cutoff||0);
  $('decision-strip').innerHTML=[[t('区域判断','Zone'),pred],[t('置信度 / 分数门限','Confidence / score gate'),conf+(d?' / '+pct(effectiveThreshold):'')],[t('区域中心误差','Zone-center error'),err],[t('特征 + 模型耗时','Features + model'),d?d.elapsed_ms.toFixed(1)+' ms':'—']].map(([key,val])=>`<div><div class="key">${key}</div><div class="val">${val}</div></div>`).join('');
  $('probabilities').innerHTML=Array.from({length:9},(_,z)=>{let p=d?.probabilities?.[z]||0;return `<div class="prob"><span>Z${z+1}</span><div class="track"><div class="fill" style="width:${100*p}%"></div></div><span>${pct(p)}</span></div>`;}).join('');
  $('model-name').textContent=experiment.results.selected_model;
  renderSignal();
}
function renderSignal(){
  let values=current?.reference_delta_db||Array(12).fill(0),max=Math.max(10,...values.map(Math.abs));
  let s='<svg class="signal-svg" viewBox="0 0 560 205" role="img" aria-label="Amplitude differences per link"><line x1="34" y1="95" x2="550" y2="95" stroke="#466079"/><text x="3" y="99" fill="#90a8bd" font-size="10">0 dB</text>';
  values.forEach((v,i)=>{let height=Math.abs(v)/max*75,x=43+i*42,y=v>=0?95-height:95;s+=`<rect x="${x}" y="${y}" width="23" height="${Math.max(1,height)}" rx="3" fill="${v>=0?'#65dcc2':'#64b9ff'}"/><text x="${x+11}" y="185" text-anchor="middle" fill="#90a8bd" font-size="10">L${i+1}</text><text x="${x+11}" y="${v>=0?y-5:y+height+13}" text-anchor="middle" fill="#bdd1e1" font-size="9">${v.toFixed(1)}</text>`;});
  $('signal-chart').innerHTML=s+'</svg>';
}
function renderHistory(){
  let show=$('truth-visible').checked,accepted=history.filter(d=>d.prediction!=null),withTruth=accepted.filter(d=>d.truth),correct=withTruth.filter(d=>d.correct);
  $('run-summary').textContent=`${t('已处理','Processed')} ${history.length} · Unknown ${history.length-accepted.length} · ${t('接受样本准确率','Accepted accuracy')} ${show&&withTruth.length?pct(correct.length/withTruth.length):'—'}`;
  $('history').innerHTML=history.slice(-25).reverse().map((d,i)=>`<tr><td>${history.length-i}</td><td>${show?esc(d.truth?.id||'signal_upload'):t('样本 ID 已隐藏','Sample ID hidden')}</td><td>${show&&d.truth?zone(d.truth.zone):'—'}</td><td class="${!show?'':d.correct?'positive':d.prediction==null?'muted':'negative'}">${decisionLabel(d)}</td><td>${pct(d.confidence)}</td><td>${show&&d.error_m!=null?d.error_m.toFixed(2)+' m':'—'}</td><td>${d.elapsed_ms.toFixed(1)} ms</td></tr>`).join('')||`<tr><td colspan="7" class="muted">${t('尚无处理记录。','No processed records yet.')}</td></tr>`;
}
function renderEvaluation(){
  let r=experiment.results,test=r.splits.test;
  $('stats').innerHTML=[['独立物理位置','Physical positions',Object.values(r.split_counts).reduce((a,b)=>a+b,0),t('训练 '+r.train_positions+' · 测试 '+test.n,'Train '+r.train_positions+' · test '+test.n)],['标准测试 Macro-F1','Standard-test Macro-F1',pct(test.macro_f1),t('全部原始预测，含低置信度','All raw predictions, including low confidence')],['测试均定位误差','Mean test location error',test.zone_center_error_m.mean.toFixed(2)+' m',t('预测区域中心至真实物体中心','Predicted zone center to true center')],['模型推理中位数','Median model inference',r.inference_ms.median.toFixed(1)+' ms',t('不含射线追踪与网络传输','Excludes ray tracing and HTTP')]].map(([zh,en,v,sub])=>`<article class="stat"><div class="label">${t(zh,en)}</div><div class="value">${v}</div><div class="sub">${sub}</div></article>`).join('');
  let labels={validation:t('验证集（用于选型）','Validation (selection)'),cal_temperature:t('独立温度校准','Temperature calibration'),cal_guard:t('独立候选/拒绝校准','Candidate/guard calibration'),test:r.protocol===3?t('全有效范围 · 新位置测试','Full valid support · new positions'):t('标准测试 · 新位置','Standard test · new positions'),boundary_test:t('交界处专项测试','Boundary-only test'),device_shift:t('设备位置 ±6/4 cm 偏移','Device positions shifted ±6/4 cm'),material_shift:t('物体木材 → 刨花板','Object wood → chipboard')};
  $('evaluation').innerHTML=Object.entries(r.splits).map(([k,v])=>`<tr><td>${labels[k]||k}</td><td>${v.n}</td><td>${pct(v.macro_f1)}</td><td>${pct(v.accuracy)}</td><td>${pct(v.coverage)}</td><td>${v.zone_center_error_m.mean.toFixed(2)} m</td></tr>`).join('')+Object.entries(r.noise_stress).map(([k,v])=>`<tr><td>${t('标准测试加噪','Test + noise')} · ${k} dB</td><td>${v.n}</td><td>${pct(v.macro_f1)}</td><td>${pct(v.accuracy)}</td><td>${pct(v.coverage)}</td><td>${v.zone_center_error_m.mean.toFixed(2)} m</td></tr>`).join('');
  let matrix=test.confusion_matrix,max=Math.max(...matrix.flat()),html='<span></span>';
  for(let z=1;z<=9;z++)html+=`<span class="matrix-label">Z${z}</span>`;
  matrix.forEach((row,i)=>{html+=`<span class="matrix-label">Z${i+1}</span>`;row.forEach((n,j)=>html+=`<span title="True Z${i+1}, predicted Z${j+1}: ${n}" style="background:rgba(101,220,194,${.04+.75*n/max});color:${n>max*.6?'#062a26':'#d4e8ed'}">${n}</span>`);});
  $('confusion').innerHTML='<div class="mini-matrix">'+html+'</div>';
  $('comparison').innerHTML=r.comparison.map(c=>`<div class="comparison"><span>${c.name}${c.name===r.selected_model?' ✓':''}</span><strong>${pct(c.validation_macro_f1)}</strong></div>`).join('')+`<p class="explain">${t('模型文件','Model file')}: ${(r.model_bytes/1048576).toFixed(2)} MiB · ${r.feature_dimension} features · ${t('有效分数门限','Effective score gate')} ${pct(Math.max(r.threshold,r.set_cutoff||0))}</p>`;
  $('paired-section').hidden=!r.baseline_comparison;
  if(r.baseline_comparison)$('paired-results').innerHTML=['test','boundary_test','device_shift','material_shift'].map(k=>{let v=r.baseline_comparison[k];return `<tr><td>${labels[k]}</td><td>${v.new.n}</td><td>${pct(v.old.accuracy)}</td><td>${pct(v.new.accuracy)}</td><td>${pct(v.new.coverage)}</td><td>${v.new.accepted_accuracy==null?'—':pct(v.new.accepted_accuracy)}</td></tr>`;}).join('');
}
async function load(){
  try{let response=await fetch('/api/state');experiment=await response.json();if(!response.ok)throw Error('Server unavailable');if(!experiment.ready){$('loading').textContent=JSON.stringify(experiment.progress);setTimeout(load,2500);return;}$('loading').hidden=true;$('application').hidden=false;protocolText();resetOrder();translate();$('run-status').textContent=t('就绪 · 模型已训练完成','Ready · model already trained');}
  catch(e){$('loading').textContent=e.message;}
}
translate();load();
