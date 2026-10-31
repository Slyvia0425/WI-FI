
from pathlib import Path
import json
from decimal import Decimal,ROUND_HALF_UP
O=Path('outputs/layout_t1_v3')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def pct(v):return str(Decimal(str(100*v)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))+'%'
def main():
 r=read(O/'results.json');old=r['baseline_comparison']['test']['previous_t1_model'];new=r['splits']['test'];sel=read(O/'selection.json');pair=r['paired_comparison'];bench=read(O/'inference_benchmark.json');f=read(O/'frozen.json')
 ui=read(O/'ui_verification.json') if (O/'ui_verification.json').exists() else {'passed':False}
 ui_zh=(f"另已完成{len(ui.get('checks',[]))}项真实浏览器交互与视觉验收，桌面与窄屏截图见UI_VISUAL_ACCEPTANCE.md。" if ui.get('passed') else '浏览器视觉验收仍待补，不声称生成新的界面截图。')
 ui_en=(f"Browser visual acceptance passed ({len(ui.get('checks',[]))} actual checks); see UI_VISUAL_ACCEPTANCE.md for screenshots." if ui.get('passed') else 'Browser visual acceptance remains pending.')
 rows=[('All test scenes',old,new)]+[(str(i)+' object(s)',old['by_true_count'][str(i)],new['by_true_count'][str(i)]) for i in [1,2,3]]+[('Shared zones',old['by_sharing']['shared'],new['by_sharing']['shared']),('Boundary stress',r['baseline_comparison']['boundary_test']['previous_t1_model'],r['splits']['boundary_test'])]
 table='\n'.join(f'| {name} | {a["n"]} | {pct(a["exact_joint_accuracy"])} | {pct(b["exact_joint_accuracy"])} |' for name,a,b in rows)
 comparison='\n'.join(f'| {label} | {pct(old[k])} | {pct(new[k])} |' for k,label in [('occupied_zone_macro_f1','Zone Macro-F1'),('count_accuracy','Total object-count accuracy'),('exact_joint_accuracy','Exact spatial counts')])
 gain=str(Decimal(str(100*pair['joint_gain'])).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
 text=f'''# 空间计数正确率优化 / Spatial-count accuracy improvement

T1固定布局的计数增强模型已启用，原模型与默认布局模型保留。这里的正确率只针对已定义的在线仿真场景，不代表真实家用设备或任意设备位置。

The enhanced model is active for the fixed T1 layout. Original weights/default-layout model remain available. Results apply to the defined simulation, not physical home devices or arbitrary radio placements.

## 实际改动 / Model change

增加直接预测九个区域物体件数的CatBoost多输出回归与MLP回归模型。两者平均后的信号预测，对原模型219种合法计数布局重新排序。仅当校准特征可用且质量通过时应用增强；相位异常仍回退原相位对齐分支。模型不读取参考物体数量、大小或坐标。

New CatBoost multioutput and MLP regressors directly estimate nine zone counts. Their mean prediction reranks the219 legal count vectors. Refinement applies only to usable calibrated features; phase faults retain the original phase-aligned fallback. No target geometry/count hints enter inference.

训练使用原40,000个物理场景、80,000个噪声观测窗口。ExtraTrees也参与验证比较，但没有进入上线模型。训练/选模仅使用训练集和原验证集；之前已打开的最终测试不用于调参。

Training uses the original40,000 scenes and80,000 receiver windows. ExtraTrees was compared on validation but excluded from deployment. Selection uses training/validation only, not prior final tests.

验证集完整计数由72.58%升至74.63%，三物体由61.67%升至64.44%。最终选择：{sel['best']['regressor']}，输出缩放{sel['best']['scale']}，计数证据权重{sel['best']['weight']}。这些是开发结果，独立验证见下表。

## 同一批全新测试 / Paired fresh evaluation

新增4,200个场景：1,200校准、2,400最终测试、600边界压力测试。与原50,800个场景无精确几何/信道重叠。先冻结模型再提取最终测试特征；两模型使用相同新信号与接收噪声。

4,200 new scenes provide1,200 confidence-calibration,2,400 final-test and600 boundary cases, with no exact overlap against50,800 prior scenes. Weights freeze before final-test features are opened. Both models see identical inputs.

| All-input metric, N=2400 | Previous model, same inputs | Enhanced model |
|---|---:|---:|
{comparison}

| Condition | N | Previous exact spatial counts | Enhanced exact spatial counts |
|---|---:|---:|---:|
{table}

主要指标是九个区域的件数全部正确，包括错误、不确定和质量不通过输入。提升为+{gain}个百分点；配对bootstrap95%区间[{100*pair['ci95'][0]:.2f}, {100*pair['ci95'][1]:.2f}]个百分点，修正{pair['improved_scenes']}条、使{pair['worsened_scenes']}条原本正确的记录变错，净改善27条。3,000次bootstrap；提升门限在新测试打开前确定，全部通过。

The primary metric requires every zone count to match and includes all inputs. Gain:+{gain} percentage points, paired-bootstrap95% CI[{100*pair['ci95'][0]:.2f}, {100*pair['ci95'][1]:.2f}] points. It fixes{pair['improved_scenes']} cases and worsens{pair['worsened_scenes']}, a net27. The3,000-resample comparison passes preregistered improvement gates.

不要用旧报告另一测试集的71.83%直接与74.96%计算改进幅度。正确的同批比较是73.83%→74.96%。

Do not compare the old report's71.83% from another test set against74.96% to claim a larger gain. The paired comparison is73.83%→74.96%.

![Paired fresh evaluation](figures/layout_t1_v3/paired_accuracy.png)

## 仍有局限 / Tradeoffs and limits

- 区域F1略降：{pct(old['occupied_zone_macro_f1'])}→{pct(new['occupied_zone_macro_f1'])}；不是所有指标都提高。
- 三物体的总数准确率略降：{pct(old['by_true_count']['3']['count_accuracy'])}→{pct(new['by_true_count']['3']['count_accuracy'])}，但完整空间计数提高。
- 三物体完整计数仍只有{pct(new['by_true_count']['3']['exact_joint_accuracy'])}，边界场景为{pct(r['splits']['boundary_test']['exact_joint_accuracy'])}，不能声称复杂布局已解决。
- 通过置信门限的比例{pct(new['accepted_layout_coverage'])}，该子集完整计数为{pct(new['accepted_joint_accuracy'])}；校准集目标不保证新测试或实际场景始终达到90%。不能把子集分数当作全输入正确率。
- 仅1–3件木质长方体、每档尺寸最多一件；不识别家具类别、精确尺寸、单件轮廓或连续三维坐标。
- 默认设备布局仍使用原模型；增强只针对T1=(1.10,1.20,0.45)m这一固定布局。

Zone F1 and three-object total-count accuracy decline slightly. Three-object exact counts and boundary performance remain imperfect. Confidence-gated subset scores are not all-input accuracy or a guaranteed90% precision. Only the fixed T1 preset is upgraded; target/material/geometry assumptions remain unchanged.

## 耗时与验收 / Runtime and checks

纯信号推理中位{bench['median_ms']:.2f}ms，p95 {bench['p95_ms']:.2f}ms；不含射线追踪、采集、HTTP、JSON和首次加载。模型文件{r['model_bytes']:,}bytes（约{r['model_bytes']/1e6:.2f}MB）。

Signal-only inference median{bench['median_ms']:.2f}ms, p95 {bench['p95_ms']:.2f}ms, excluding simulation/acquisition/transport/first load.

候选服务及主服务各29项接口检查通过，默认模型25项回归通过，9项前端状态逻辑检查通过。另核对全部三种评估划分与冻结权重一致、相位故障回退及零信号Unknown。{ui_zh}

29 checks passed on each candidate/public service,25 default-model regression checks and9 UI logic checks passed. Frozen metrics, phase fallback and zero-signal rejection were verified. {ui_en}

模型SHA256：`{f['model_sha256']}`。

证据：`outputs/layout_t1_v3/results.json`、`data_audit.json`、`evaluation_binding.json`、`api_verification_public.json`、`inference_benchmark.json`。

## 使用 / Use

刷新 http://127.0.0.1:8769/ ，顶部“设备布局”选择 **T1·增强版**，再开始检测。结果会标明“计数增强版”。切换到默认布局使用原默认模型。模型版本变化时页面会清除旧结果，避免把旧预测和新分数混在一起。

Refresh the page and choose **T1·enhanced** before starting. Results identify the enhanced count model. The default layout retains its original model. Version changes invalidate stale outputs.
'''
 Path('ACCURACY_V3_RESULTS.md').write_text(text,encoding='utf-8')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.ticker import PercentFormatter
 import numpy as np
 out=Path('figures/layout_t1_v3');out.mkdir(exist_ok=True,parents=True)
 fig,ax=plt.subplots(figsize=(11,4.7),layout='constrained');x=np.arange(len(rows));w=.36
 for offset,key,color in [(-w/2,1,'#60788c'),(w/2,2,'#1c9584')]:
  vals=[a[key]['exact_joint_accuracy'] for a in rows];bars=ax.bar(x+offset,vals,w,color=color,label='Previous model' if key==1 else 'Enhanced counts')
  ax.bar_label(bars,labels=[f'{100*v:.1f}%' for v in vals],fontsize=9,padding=3)
 ax.set_xticks(x,[a[0] for a in rows],rotation=12,ha='right');ax.set_ylim(0,1.08);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel('All zones and counts correct');ax.set_title(f'Paired new scenes: +{gain}pp exact spatial counts');ax.legend(loc='lower left');ax.spines[['top','right']].set_visible(False)
 fig.savefig(out/'paired_accuracy.png',dpi=180);fig.savefig(out/'paired_accuracy.svg');plt.close(fig);print('Written ACCURACY_V3_RESULTS.md and paired accuracy figures.')
if __name__=='__main__':main()
