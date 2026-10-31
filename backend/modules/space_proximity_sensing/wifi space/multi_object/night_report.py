
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from night_common import NIGHT,now
from multi_common import ROOT,OUT

def pct(x):return '—' if x is None else f'{100*x:.1f}%'
def write(name,text):(ROOT/name).write_text(text,encoding='utf-8')

def figures(r):
    folder=ROOT/'figures'/'overnight';folder.mkdir(exist_ok=True,parents=True)
    new=r['splits']['test'];old=r['baseline_comparison']['test']['v2_same_frames'];fig,ax=plt.subplots(figsize=(8.4,4.6),layout='constrained');x=np.arange(4);width=.35
    names=['All scenes','2 objects','3 objects','Boundary']
    os=[old['exact_occupied_layout_accuracy']]+[old['by_true_count'][str(k)]['exact_occupied_layout_accuracy'] for k in [2,3]]+[r['baseline_comparison']['boundary_test']['v2_same_frames']['exact_occupied_layout_accuracy']]
    ns=[new['exact_occupied_layout_accuracy']]+[new['by_true_count'][str(k)]['exact_occupied_layout_accuracy'] for k in [2,3]]+[r['splits']['boundary_test']['exact_occupied_layout_accuracy']]
    for off,values,label,color in [(-width/2,os,'Previous v2','#7894ab'),(width/2,ns,'Candidate v3','#16a38c')]:
        bars=ax.bar(x+off,100*np.array(values),width,label=label,color=color);ax.bar_label(bars,fmt='%.1f%%',padding=3,fontsize=9)
    ax.set(xticks=x,xticklabels=names,ylabel='Exact occupied-layout accuracy (%)',ylim=(0,100),title='Same-scene comparison, all inputs included');ax.legend(loc='upper right');ax.spines[['top','right']].set_visible(False);fig.savefig(folder/'paired_layout_comparison.png',dpi=180);plt.close(fig)
    cm=np.asarray(new['count_confusion']);fig,ax=plt.subplots(figsize=(5.5,4.5),layout='constrained');im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax)
    for i in range(3):
        for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set(xticks=range(3),yticks=range(3),xticklabels=[1,2,3],yticklabels=[1,2,3],xlabel='Predicted count',ylabel='True count',title='New final test: count confusion');fig.savefig(folder/'count_confusion.png',dpi=180);plt.close(fig)
    cms=np.asarray(new['per_zone_confusion']);fig,axes=plt.subplots(3,3,figsize=(8,8),layout='constrained')
    for z,ax in enumerate(axes.flat):
        ax.imshow(cms[z],cmap='Blues',vmin=0,vmax=cms.max());ax.set(title=f'Z{z+1}',xticks=[0,1],yticks=[0,1],xticklabels=['Empty','Occupied'],yticklabels=['Empty','Occupied'])
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cms[z,i,j]),ha='center',va='center',color='white' if cms[z,i,j]>cms.max()/2 else 'black')
    fig.suptitle('Per-zone confusion: rows = truth, columns = prediction');fig.savefig(folder/'zone_confusions.png',dpi=180);plt.close(fig)
    with np.load(NIGHT/'predictions_test.npz') as a:conf=a['posterior'].max(1);correct=np.all(a['pred']==a['truth'],axis=1)
    bins=[];ece=0
    for lo in np.arange(0,1,.1):
        keep=(conf>=lo)&(conf<lo+.1 if lo<.9 else conf<=1)
        if not keep.any():continue
        avg=float(conf[keep].mean());acc=float(correct[keep].mean());n=int(keep.sum());bins.append(dict(mean_confidence=avg,exact_accuracy=acc,n=n));ece+=n/len(conf)*abs(avg-acc)
    fig,ax=plt.subplots(figsize=(5.7,4.7),layout='constrained');ax.plot([0,1],[0,1],'--',color='#94a3b8');ax.scatter([b['mean_confidence'] for b in bins],[b['exact_accuracy'] for b in bins],s=[max(20,b['n']) for b in bins],color='#16a38c',alpha=.75)
    for bb in bins:ax.annotate(str(bb['n']),(bb['mean_confidence'],bb['exact_accuracy']),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Mean layout confidence',ylabel='Observed exact-layout accuracy',title='Frozen-test reliability; point labels = sample counts',xlim=(0,1),ylim=(0,1));fig.savefig(folder/'confidence_reliability.png',dpi=180);plt.close(fig)
    (NIGHT/'reliability.json').write_text(json.dumps(dict(bins=bins,expected_calibration_error=ece,n=len(conf),note='Post-freeze diagnostic only; no test-based recalibration.'),indent=2))

def main():
    r=json.loads((NIGHT/'results.json').read_text(encoding='utf-8'));old=r['baseline_comparison']['test']['v2_same_frames'];new=r['splits']['test'];boundary=r['splits']['boundary_test'];ci=r['paired_comparison']['test'];active=json.loads((OUT/'active_profile.json').read_text(encoding='utf-8'))['version'] if (OUT/'active_profile.json').exists() else 2
    audit=json.loads((NIGHT/'data_audit_full.json').read_text(encoding='utf-8'));fr=json.loads((NIGHT/'candidate_frozen.json').read_text(encoding='utf-8'));figures(r)
    rows=[]
    for name,o,n in [('全部1500 / All1500',old,new),('两件 / Two',old['by_true_count']['2'],new['by_true_count']['2']),('三件 / Three',old['by_true_count']['3'],new['by_true_count']['3']),('交界450 / Boundary450',r['baseline_comparison']['boundary_test']['v2_same_frames'],boundary)]:rows.append(f'| {name} | {pct(o["occupied_zone_macro_f1"])} → {pct(n["occupied_zone_macro_f1"])} | {pct(o["count_accuracy"])} → {pct(n["count_accuracy"])} | {pct(o["exact_occupied_layout_accuracy"])} → {pct(n["exact_occupied_layout_accuracy"])} |')
    table='\n'.join(rows);phase=r['phase_diagnostics'];phase_rows='\n'.join(f'| {k} | {pct(v["v2_raw"]["exact_occupied_layout_accuracy"])} | {pct(v["exact_occupied_layout_accuracy"])} | {v["phase_free_routed"]}/90 |' for k,v in phase.items() if isinstance(v,dict))
    report=f'''# 晨间成果 / Morning handoff

生成时间 / Generated: {now()}。当前启用模型 / Active profile: **v{active}**。新候选发布检查 / Candidate release check: **{r['release_check']['eligible']}**。

**尚未解决：你的小物体位置(3.84,3.42)仍有高置信区域误判，详见末尾个例表。 / The user placement(3.84,3.42) still has a confident zone error; see the case table.**

## 现在怎样打开 / Open it

双击本目录 `START_MULTI.bat`，或项目根目录 `OPEN_MULTIOBJECT.bat`。地址 **http://127.0.0.1:8769/**。先看 `TEACHER_DEMO.md` 的三分钟演示。页面主流程：拖动摆放 → 开始检测 → 查看右图。坐标、置信度、历史、测试结果和设置默认折叠。

Double-click `START_MULTI.bat` or the root `OPEN_MULTIOBJECT.bat`. The launcher warms the model before opening the browser. See `TEACHER_DEMO.md` for a three-minute walkthrough.

## 本次完成 / Completed work

1. 新增24,750个真实联合射线追踪的模拟场景：训练21,000、验证1,200、校准600、最终测试1,500、交界测试450。复用旧数据时只用原训练部分15,000条，因此训练共36,000个物理场景、72,000个增强窗口。每窗口16帧不是16个新场景。
2. 训练中加入3,000组固定背景对照，每组含原两件物体及两种增加第三件的布局；强调小物体和区域交界。原始信号审核通过，无训练/评估完全相同信号交叠，无相同信号对应冲突区域标签。
3. 比较校准复数特征与逐帧相位对齐特征，分别训练占用、数量、布局网络及提升树。验证集选模型和解码参数；校准集决定温度、接受门限和路由规则。相位对齐分支同样有独立数量和布局模型。
4. 冻结候选后才打开新最终测试；旧v2和候选使用完全相同的新测试信号比较。保留v2原权重、回退命令、所有失败/不确定结果。没有用左图真值修改右图。
5. 修复冷启动、增加服务就绪检查与候选独立端口、补充纯信号接口验证、复现说明和演示指南。

Added24,750 independent joint-RT scenes and3,000 counterfactual training groups. Training now uses36,000 physical configurations and72,000 augmented16-frame windows. Separate calibrated/phase-aligned experts learn occupancy, count and layouts. Selection/calibration/final testing remain separate, and v2 is retained for rollback.

## 独立测试结果 / Independent final results

下表均为旧v2 → 新v3，在相同全新数据上的比较；包括不确定/拒绝样本。区域F1不是整套布局正确率，旧报告的69.1%来自另一批旧测试，不能和这里直接相减。

Every row compares v2 → v3 on the SAME new signals, including uncertain/rejected cases. Previous69.1% came from a different older test set.

| 条件 / Condition | Zone Macro-F1 | 数量 / Count accuracy | 完整布局 / Exact layout |
|---|---|---|---|
{table}

完整布局差值 / Exact-layout difference: **{100*ci['exact_layout_difference']:.2f} percentage points**，分层配对bootstrap95%区间 / paired stratified bootstrap95%interval: **[{100*ci['paired_bootstrap_95_interval'][0]:.2f}, {100*ci['paired_bootstrap_95_interval'][1]:.2f}] points**。

新候选通过布局门限的覆盖率为{pct(new['accepted_layout_coverage'])}，该子集完整布局准确率{pct(new['accepted_layout_accuracy'])}。这只是子集结果，不能代替上表全部测试。质量拒绝也算失败时，全输入完整布局准确率为{pct(new['all_inputs_operational_exact_accuracy'])}。

Confidence-gated coverage is{pct(new['accepted_layout_coverage'])}; exact accuracy within that subset is{pct(new['accepted_layout_accuracy'])}. This is not the all-input score. Quality-adjusted operational exact accuracy across all inputs is{pct(new['all_inputs_operational_exact_accuracy'])}.

信号推断耗时 / Signal-only latency: median **{r['inference_ms']['median']:.1f}ms**, p95 **{r['inference_ms']['p95']:.1f}ms**，不含采集、射线追踪或HTTP。模型文件 / model bundle: **{r['model_bytes']/1024**2:.1f}MiB**。

## 相位扰动 / Phase diagnostics

90个新测试几何场景，单独施加模拟相位扰动；不是换真实设备或换房间实验。

| 条件 / Condition | v2 exact | v3 exact | 对齐分支 / Phase-aligned routed |
|---|---|---|---|
{phase_rows}

独立链路/帧相位的特征不变性通过数值检查；整体路由后的准确率仍取决于分支模型。不能据此宣称已解决时钟延迟、频率相关相位、换天线布局或真实硬件。

Feature invariance covers constant per-link/per-frame phase rotations only. It does not establish invariance to timing delays, frequency-dependent phase errors, antenna layouts or real hardware.

## 如何测试 / What to test

打开页面后先在自定义场景拖动A/B，点单次检测；再切换两件/三件测试并持续运行。暂停应保留快照，修改坐标应清空预测。下载JSON示例并重新上传，右图仍能推断且不显示参考位置。更多设置可降低SNR，观察不确定或拒绝；运行记录可导出。完整步骤见 `TEACHER_DEMO.md`；脚本检查见 `night_verify.py`。

Drag and infer; run held-out two/three-object scenes; pause and edit; export history; round-trip a signal-only JSON; lower SNR. These check workflow, while the frozen offline test checks performance.

## 仍然不能承诺 / Remaining limits

边界、小物体遮挡和多件布局仍会误判，不能保证每个自定义位置正确。当前输出是中心所属2米格区，无法分开同一区域多件物体，不能确认空房。辅助尺寸/立体示意未恢复真实长宽高或沙发、桌子语义。房间、射频位置、材质固定；没有真实家庭采集和跨房间部署证据。

Boundary and occluded-small-object errors remain. Outputs are2m center zones, not exact coordinates, measured dimensions or furniture semantics. Empty rooms and multiple objects in one zone are unvalidated. Real-world/generalized deployment remains future work.

课程材料仍需学生完成团队身份整合、最终PPT、10–15分钟视频、个人反思和正式提交，不能把技术原型完成等同于课程材料全部交完。见 `COURSE_MAP.md`。

The student/team still needs to finalize presentation/video, personal reflection and formal submission. Technical implementation is not a claim that every human coursework deliverable has been submitted.

## 证据与复现 / Evidence and reproduction

- `outputs/overnight/results.json`: 全部指标、分组、置信度覆盖率、相位诊断、发布检查。
- `outputs/overnight/data_audit_full.json`: 原始信号/标签/划分审计。
- `outputs/overnight/candidate_frozen.json`、`test_opened.json`: 冻结与解封记录。
- `outputs/overnight/verification.json`、`ui_verification.json`: 接口和浏览器检查（以文件存在且passed为准）。
- `SETUP.md`、`requirements-night.txt`、`outputs/overnight/environment-freeze.txt`: 安装、版本与复现命令。
- `figures/overnight/`: 独立对比、数量混淆、九区域混淆和置信度可靠性图。
- `OVERNIGHT_LOG.md`: 包括探索性旧验证pilot；不是最终测试成绩。

候选模型SHA256 / Candidate model SHA256: `{fr['model_sha256']}`。
'''
    write('MORNING_REPORT.md',report)
    extra='\n## 进一步实验与未解决个例 / Additional experiments and unresolved cases\n\n'
    cd=NIGHT/'custom_diagnostics.json'
    if cd.exists():
        cases=json.loads(cd.read_text(encoding='utf-8'))['cases']
        extra+='以下每个位置仅一个物理几何场景、32个噪声窗口，不是32个独立测试场景。\n\nEach row is one development geometry with32 receiver-noise windows.\n\n| 场景 / Scene | v2 correct /32 | v3 correct /32 | v3 wrong accepted /32 |\n|---|---|---|---|\n'
        for c in cases:extra+=f'| {c["name"]} | {c["metrics"]["v2"]["correct_windows"]} | {c["metrics"]["v3"]["correct_windows"]} | {c["metrics"]["v3"]["wrong_accepted"]} |\n'
        extra+='\n**明确未解决：小物体(3.84,3.42)+大物体(5,5)仍会把Z5判成Z6，且32个窗口均被高置信接受。总体校准不能保证每个局部位置正确；页面没有用真值修正这个错误。**\n\n**Unresolved: the user placement still predictsZ6 instead ofZ5 and confidently accepts32/32 noisy windows. Global calibration does not ensure local correctness. The frontend does not replace predictions with truth.**\n'
    gp=NIGHT/'guard_pilot'/'results.json';rp=NIGHT/'risk_pilot'/'results.json'
    if gp.exists():
        g=json.loads(gp.read_text(encoding='utf-8'));mae=float(np.mean([a['validation_coordinate_mae_m'] for a in g['regressors']]))
        extra+=f'\n边界回归试验：训练9个区域条件坐标回归器，验证集平均每轴MAE为{mae:.3f}m；这是**已知正确区域条件下**的误差，不是端到端坐标精度。单纯按预测边界距离拒绝，未能同时达到预设95%接受子集精度和25%覆盖率，因此未部署。\n\nBoundary pilot trained9 conditional coordinate regressors; validation per-axis MAE={mae:.3f}m assumes the correct zone. It is not end-to-end coordinate accuracy. The boundary-only gate failed its predeclared95% accuracy/25% coverage target and was not deployed.\n'
    if rp.exists():
        q=json.loads(rp.read_text(encoding='utf-8'));best=max(a['development_auc'] for a in q['comparisons'])
        extra+=f'\n进一步比较逻辑回归与浅层提升树学习误判风险，输入包括两分支差异、概率熵及边界估计。独立开发子集最好AUC={best:.3f}，低于直接置信度基线{q["development_baseline_confidence_auc"]:.3f}，未部署。两个后续试验均未使用最终测试调参，保留负结果，继续启用通过独立验证的v3。\n\nA learned18-feature error ranker was also tested. Its best held-out development AUC={best:.3f}, below confidence alone({q["development_baseline_confidence_auc"]:.3f}). It was not deployed. Both negative experiments used development/calibration only; the proven v3 remains active.\n'
    with (ROOT/'MORNING_REPORT.md').open('a',encoding='utf-8') as f:f.write(extra)

    marker='<!-- PRESERVED_PREVIOUS_DOCUMENTATION -->'
    readme=ROOT/'README.md';existing=readme.read_text(encoding='utf-8')
    history=existing.split(marker,1)[-1] if marker in existing else existing.replace('# Current optimized version / 当前优化版','# Previous optimized_v2 / 旧版v2',1)
    readme.write_text(f'''# RoomSense · 当前多物体版本 / Current multi-object project

当前启用 **v{active}**，网页 **http://127.0.0.1:8769/**。双击 `START_MULTI.bat`；根目录也可双击 `OPEN_MULTIOBJECT.bat`。

Active profile **v{active}**. Launch once, drag objects, start sensing and inspect predicted center zones. Main UI is bilingual and compact.

- [晨间成果与完整新测试 / Results](MORNING_REPORT.md)
- [三分钟演示与验证 / Demo guide](TEACHER_DEMO.md)
- [安装、复现、回退 / Setup](SETUP.md)
- [答辩问题 / FAQ](FAQ.md)
- [课程对应与待提交材料 / Course mapping](COURSE_MAP.md)

主要输出：固定模拟房间中1–3件箱体的中心占用区域与数量。尺寸/立体示意是辅助实验；真实家具形状和家庭部署未验证。完整布局与区域F1要分别报告。

Primary output is center-zone occupancy and count in a fixed simulated room. Experimental size drawings are not measured furniture reconstruction.

## 历史文档 / Previous experiments

以下保留旧版v2/v1说明和成绩，不能作为当前模型在新测试上的成绩。 / The following reports are historical.

{marker}
'''+history,encoding='utf-8')
    root_readme=ROOT.parent/'README.md';previous=root_readme.read_text(encoding='utf-8');old_doc=previous.split(marker,1)[-1] if marker in previous else previous
    root_readme.write_text(f'''# RoomSense · 项目入口 / Project entry

**主要演示 / Main demo:** 双击 `OPEN_MULTIOBJECT.bat`，打开 http://127.0.0.1:8769/ 。当前多物体模型为 **v{active}**。

先阅读 [晨间成果](multi_object/MORNING_REPORT.md) 和 [三分钟操作](multi_object/TEACHER_DEMO.md)；复现方法见 [SETUP](multi_object/SETUP.md)。不需要重新下载或重新训练即可演示现有模型。

Open the multi-object launcher, then follow the bilingual demo guide. Raw simulated data and trained models are included. The Python environment is installed on this computer and excluded from the archive.

## 保留的单物体/历史说明 / Preserved single-object and historical documentation

单物体版本继续保留在8766；下面的原说明与旧成绩不是当前多物体新测试成绩。 / Single-object8766 remains separate; historical scores below do not describe the current multi-object final test.

{marker}
'''+old_doc,encoding='utf-8')
    write('COURSE_MAP.md','''# 课程对应 / Course mapping

依据已提供课程PDF（PRS Module7.0，第9、15、16页）及项目内 `../reports/COURSE_CHECKLIST.md`。本表说明技术证据，不承诺老师评分。

| 要求 / Requirement | 本项目证据 / Evidence | 状态 / Status |
|---|---|---|
| 实际问题 / Application | 隐私友好的室内粗定位；固定房间仿真验证 / Privacy-conscious coarse spatial sensing | 可演示研究原型 / Demonstrable prototype |
| 分类 / Classification | 9区域多标签、物体数量、129种1–3区域布局 / Zone, count and layout classification | 已实现 / Implemented |
| 学习方法 / Learning methods | MLP、提升树，保留原Extra Trees对比 / MLP, boosted trees, historical Extra Trees | 实际训练 / Trained models |
| 智能感知与解释 / Sensing and sense-making | CSI→特征→分类→置信度→区域图 / CSI to confidence and spatial output | 已实现 / Implemented |
| 混合/集成（若宣称） / Hybrid if claimed | 树/网络占用融合及网络数量/布局联合解码，参数在验证集选择 / Learned ensemble and structured decoding | 以当前冻结decoder为准 / See frozen decoder |
| 数据、工具、方法 / Data and method | 数据协议、manifest、复数NPZ、分组划分、源代码 / Protocol, raw channels, splits, code | 已提供 / Included |
| 评价与讨论 / Evaluation | 全输入F1、召回、混淆、数量、完整布局、延时、边界/相位压力测试 / Raw metrics and stress tests | 见晨间报告 / See report |
| 可运行系统 / Runnable system | 一键启动、中英文网页、拖动、纯信号导入、回放 / Launcher and bilingual UI | 需以验证记录为准 / See verification |
| 最终报告 / Team report | 晨间技术报告、图表、操作说明 / Technical report and figures | 学生加入团队信息并整合 / Team integration remains |
| 两次演示PPT / Presentation slides | 当前原型与既有设计有变化 / Scope changed from earlier slides | 学生更新并核对 / Update required |
| 10–15分钟视频 / Video | 三分钟模块演示指南可作分镜 / Demo guide available | 学生/团队录制 / Human recording required |
| 个人反思1–2页 / Individual reflection | 本人学习与贡献 / Personal learning and contribution | 由学生真实撰写 / Student-authored |
| 同伴评价、ZIP与Canvas / Submission | 本地技术包支持 / Local technical package | 团队审阅并提交 / Human submission |

使用静态物体模拟数据的成绩说明静态模拟任务；原CSI-Bench人体定位成绩不能混作家具定位成绩。真实家具3D重建和任意房间泛化未完成，汇报中必须保留该边界。

Static-object simulation results must not be replaced by earlier human-localization scores. Real furniture reconstruction and arbitrary-room generalization remain unvalidated.
''')
    print('Morning report and four exported figures generated.')

if __name__=='__main__':main()
