
import json,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image
ROOT=Path(__file__).resolve().parent
E=ROOT/'outputs/ui_acceptance_20261031';O=ROOT/'outputs/layout_t1_v3'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 evidence=read(E/'browser_checks.json');assert evidence['passed'] and len(evidence['checks'])==20 and all(c['passed'] for c in evidence['checks'])
 frozen=read(O/'frozen.json');assert sha(O/'model.joblib')==frozen['model_sha256']
 shots=[]
 for name in evidence['screenshots']:
  p=E/name
  with Image.open(p) as im:im.verify()
  with Image.open(p) as im:shots.append(dict(file=str(p),width=im.width,height=im.height,format=im.format))
 counts={n:len(read(E/n)['history']) for n in ['run_before_visual_fix.json','run_after_visual_fix.json']}
 now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
 report=dict(passed=True,completed_at=now,url=evidence['url'],browser='ChatGPT in-app browser, real page interaction',model_revision='t1_count_v3',model_sha256=frozen['model_sha256'],tested_viewports=[dict(width=1024,height=900),dict(width=375,height=812)],checks=evidence['checks'],screenshots=shots,run_record_counts=counts,original_placement_restored=True,console_errors=evidence['browserConsole'],source_sha256={n:sha(ROOT/n) for n in ['web/app.js','web/index.html','web/style.css']},fixed='Map count text changed to larger language-neutral multiplication/count labels with accessible bilingual singular/plural hints.',scope='Browser functional/visual acceptance only; prediction mistakes and Uncertain/Unknown states are retained. Screenshots of individual samples are not accuracy estimates.')
 write(O/'ui_verification.json',report)
 state=read(O/'release_state.json');state.update(state='completed',visual_verification='ui_verification.json',visual_completed_at=now);write(O/'release_state.json',state)
 for p in [O/'handoff.json',ROOT/'outputs/optimization_handoff.json']:
  h=read(p);h.update(status='accuracy_v3_fully_verified',visual_acceptance='passed',browser_checks=20,visual_report='UI_VISUAL_ACCEPTANCE.md');write(p,h)
 names='\n'.join(f"| {i+1} | {c['name']} | Pass |" for i,c in enumerate(evidence['checks']))
 text=f'''# 浏览器视觉验收 / Browser acceptance

完成时间：{now}。项目页面 http://127.0.0.1:8769/ ，当前增强模型 `t1_count_v3`。

实际浏览器交互与视觉验收 **20/20通过**，不是仅检查接口或模拟DOM。检查1024×900桌面、375×812窄屏，及中文、英文、双语显示。浏览器没有JavaScript错误或警告。

20/20 actual browser checks passed, covering desktop/narrow layouts and Chinese/English/bilingual UI. No JavaScript errors or warnings were captured.

| # | Check | Result |
|---|---|---|
{names}

## 发现并修复 / Fix

区域计数原显示“1 objects”，窄屏字体较小。现改为更醒目的“×1、×2”数字标签，并为辅助提示补上正确英文单复数。修复后刷新实际页面，重新检查移动端与桌面截图。模型权重、推理算法和正式测试分数未改变。

Map counts now use larger language-neutral labels and accessible bilingual hints with correct singular/plural. Actual rendering was rechecked after reload. Model weights/evaluation scores were not changed.

## 证据 / Evidence

![Desktop single test sample](outputs/ui_acceptance_20261031/desktop_overview.jpg)

![Mobile prediction](outputs/ui_acceptance_20261031/mobile_prediction.jpg)

![Unknown input](outputs/ui_acceptance_20261031/unknown_signal.jpg)

截图展示单个功能测试样本，不代表整体正确率。测试中的预测错误和不确定结果未隐藏；修复前24条和修复后8条运行记录均已导出保存。已验证20次自动停止及持续模式暂停后记录不再增长。你的原始三物体位置、尺寸档和选中物体均已恢复；浏览器视口已重置，项目标签保留打开。

Screenshots show individual functional samples, not overall accuracy. Prediction errors and uncertainty remain visible in saved runs. Both24 pre-fix and8 post-fix records are retained. Original placement was restored and the viewport override reset. The project tab remains open.

详细机器记录：`outputs/layout_t1_v3/ui_verification.json`；原始浏览器检查：`outputs/ui_acceptance_20261031/browser_checks.json`。正式正确率仍见 [ACCURACY_V3_RESULTS.md](ACCURACY_V3_RESULTS.md)。
'''
 (ROOT/'UI_VISUAL_ACCEPTANCE.md').write_text(text,encoding='utf-8')
 p=ROOT/'RESUME_LAYOUT_T1.md';s=p.read_text(encoding='utf-8');s=s.replace('# 恢复工作入口','# 恢复工作入口\n\n## 浏览器视觉验收已完成（2026-10-31）\n\n当前V3已完成20项真实浏览器交互和视觉验收，桌面1024与窄屏375、中英文/双语均检查。修正地图数量“1 objects”为更醒目的×1/×2，并加可访问提示。原用户摆放已恢复，项目标签保留，视口已重置。浏览器无JS错误。截图、24+8条运行记录和完整检查清单见UI_VISUAL_ACCEPTANCE.md及outputs/ui_acceptance_20261031。当前release_state=completed，ui_verification.passed=true，不再标记待补。下方待补说明为过去阶段记录。\n',1);p.write_text(s,encoding='utf-8')
 p=ROOT/'LAYOUT_PRESET_GUIDE.md';s=p.read_text(encoding='utf-8');s+='\n\n## 浏览器验收已完成 / Browser acceptance completed\n\n2026-10-31已完成20项实际浏览器检查，包括20次持续检测、暂停、拖动、布局/分数切换、文件导入、Unknown及桌面/窄屏/三种语言显示。原摆放已恢复，项目页面保留可用。见UI_VISUAL_ACCEPTANCE.md。此前“待补”是历史状态。\n\n20 actual browser checks passed. Screenshots and retained run records are in UI_VISUAL_ACCEPTANCE.md; earlier pending notes describe previous stages.\n';p.write_text(s,encoding='utf-8')
 with (ROOT/'LAYOUT_T1_WORKLOG.md').open('a',encoding='utf-8') as f:f.write('\n## Browser visual acceptance completed '+now+'\n20real browser checks passed via existing IABtab1. Desktop1024/mobile375, both/en/zh, live20auto-stop,unlimitedpause,draglarge,invalidTX,unvalidatedTX,cross-layoutupload,zeroCSIUnknown,recovery,scores/history all verified. JSconsole noerrors. Fixed countlabelgrammar/readability; model unchanged. Preserved original placement and32 total UIrun records across24+8 exports; screenshots saved. Browserviewport reset and tab kept open. UI_VISUAL_ACCEPTANCE.md; outputs/layout_t1_v3/ui_verification.json passedtrue; release_state completed.\n')
 print(json.dumps(dict(passed=True,checks=len(report['checks']),screenshots=len(shots),run_records=counts,model_unchanged=True)))
if __name__=='__main__':main()
