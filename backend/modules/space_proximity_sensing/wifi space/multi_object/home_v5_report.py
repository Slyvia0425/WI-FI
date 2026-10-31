
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from home_v5_common import D5,O5,read,now,sha
from home_v5_physics import SPEC

def pct(x):return '—' if x is None else f'{100*x:.1f}%'

def main():
 r=read(O5/'results.json');m=r['splits']['test'];old=r['baseline_comparison']['test']['v4_same_background_reference'];active=read(O5.parent/'active_profile.json')['version'];folder=Path('figures/home_v5');folder.mkdir(parents=True,exist_ok=True)
 cm=np.array(m['object_count_confusion']);fig,ax=plt.subplots(figsize=(5.6,4.4),layout='constrained');ax.imshow(cm,cmap='Blues')
 for i in range(3):
  for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
 ax.set(title='Target count: furnished-room final test',xlabel='Predicted target count',ylabel='True target count',xticks=range(3),yticks=range(3),xticklabels=[1,2,3],yticklabels=[1,2,3]);fig.savefig(folder/'target_count_confusion.png',dpi=180);plt.close(fig)
 names=['All','Distinct zones','Shared zones','All in one zone'];keys=[None,'distinct','shared','all_in_one_zone'];x=np.arange(4);fig,ax=plt.subplots(figsize=(9.2,4.8),layout='constrained')
 for off,model,color,label in [(-.18,old,'#849bad','Previous v4 weights + new background reference'),(.18,m,'#16a38c','Retrained v5')]:
  vals=[100*(model if k is None else model['by_sharing'][k])['count_accuracy'] for k in keys];bars=ax.bar(x+off,vals,.36,label=label,color=color);ax.bar_label(bars,fmt='%.1f%%',padding=3,fontsize=9)
 ax.set(xticks=x,xticklabels=names,ylabel='Target-count accuracy (%)',ylim=(0,100),title='Same new furnished-room test scenes; all inputs included');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),fontsize=8,ncol=2,frameon=False);ax.spines[['top','right']].set_visible(False);fig.savefig(folder/'count_comparison.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7.4,3.8),layout='constrained');a=np.asarray(m['per_zone_recall']);bars=ax.bar(np.arange(9),100*a,color='#16a38c');ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=8);ax.set(xticks=range(9),xticklabels=[f'Z{i+1}' for i in range(9)],ylim=(0,105),ylabel='Recall (%)',title='Occupied-zone recall: all final-test inputs');ax.spines[['top','right']].set_visible(False);fig.savefig(folder/'zone_recall.png',dpi=180);plt.close(fig)
 lines=[]
 for name,key in zip(['全部 / All','不同区 / Distinct','存在同区 / Shared','全部同区 / All in one'],keys):
  a=m if key is None else m['by_sharing'][key];lines.append(f'| {name} | {a["n"]} | {pct(a["occupied_zone_macro_f1"])} | {pct(a["count_accuracy"])} | {pct(a["exact_occupied_layout_accuracy"])} | {pct(a["exact_joint_accuracy"])} |')
 for count,a in m['by_true_count'].items():lines.append(f'| {count}件 / {count} targets | {a["n"]} | {pct(a["occupied_zone_macro_f1"])} | {pct(a["count_accuracy"])} | {pct(a["exact_occupied_layout_accuracy"])} | {pct(a["exact_joint_accuracy"])} |')
 stress=read(O5/'stress_results.json') if (O5/'stress_results.json').exists() else None
 stresslines=[]
 if stress:
  for name,a in stress['conditions'].items():stresslines.append(f'| {name} | {a["n"]} | {pct(a["occupied_zone_macro_f1"])} | {pct(a["count_accuracy"])} | {pct(a["quality_coverage"])} | {pct(a["accepted_layout_coverage"])} |')
 report=f"""# 家居场景模拟升级 / Furnished-room simulation upgrade

生成 / Generated: {now()}。当前活动模型 / Active model: v{active}。本候选质量门槛 / Candidate quality check: **{r['release_check']['eligible']}**。

## 实际改动 / Actual changes

仍为纯线上仿真。重新构造带玻璃窗、木门和固定背景柜体的6×6×3米房间；启用漫反射、穿透和偶极子天线方向性，同时保留直达、遮挡与镜面反射。柜体标记为BG，是已标定背景，**不计入待测物体数**。4发射点/8接收点保留，以32条链路提供空间观测；不能称为单台路由器仿真系统。

This remains an online simulator. A new mesh includes a glass window, wooden door and fixed background cabinet. Diffuse reflection, refraction and dipole antenna patterns are enabled. BG is calibrated background and excluded from target counts. The distributed4TX/8RX layout is retained.

信道为5.24GHz、20MHz频带的64个复数频率采样点；不是完整802.11数据包/射频收发链路。最多3件木质轴对齐箱体，每个尺寸档位最多1件，尺寸在档位±8%内变化；区域按中心定义。支持单件、同区多件。没有恢复家具语义、实际轮廓或同区内单件坐标。

Channel responses are sampled at64 frequencies over20MHz around5.24GHz. The simulator is not a complete802.11 PHY or measured RF receiver. Targets remain1–3 wooden axis-aligned cuboids, one per size band. Center-zone counts are recovered, not furniture semantics, shapes or exact within-zone coordinates.

## 数据与算法 / Data and algorithm

全新46,600个联合射线追踪场景：训练40,000、验证2,400、校准1,200、最终测试2,400、交界600。没有把旧信号改标签后当作新场景。每个训练场景2个独立接收扰动窗口，共80,000训练窗口；每窗16帧。25–40dB噪声以及轻微幅度/相位/频率斜率扰动只是一种接收误差近似。

All46,600 joint-RT geometries are new. Training uses40,000 scenes/80,000 windows. Validation, calibration, final test and boundary groups are separate. Receiver augmentation includes25–40dB SNR and small gain/phase/slope drift; this is a simplified error model.

双信号特征分支中，提升树预测区域占用，神经网络预测区域数、物体总数和219类空间计数；联合解码保持每区数量、总数和区域数一致。模型选择仅用验证集，温度和接受门限仅用独立校准集。最终测试前冻结权重。位置、数量、尺寸和类别真值不进入推断。

Two feature views combine boosted occupancy trees with neural region/target-count and219-state spatial-count models. Validation selects checkpoints/decoder; separate calibration fits confidence. Final tests are opened only after freezing. Ground-truth geometry/counts never enter inference.

## 独立测试 / Independent results

| 条件 / Condition | N | 区域Macro-F1 | 待测总数正确 | 完整区域正确 | 每区件数全部正确 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

完整区域只要求格区正确；每区件数全部正确还要求数量分配正确。总数正确不能代替完整空间计数正确。包含全部测试输入，不能只挑选已识别样例。质量可用覆盖率{pct(m['quality_coverage'])}，通过联合置信门限的覆盖率{pct(m['accepted_layout_coverage'])}；通过门限子集的完整计数正确率{pct(m['accepted_layout_accuracy'])}，只是子集指标。

Exact zones and exact per-zone counts differ. All-input scores include uncertain/rejected examples. Quality coverage={pct(m['quality_coverage'])}; confidence acceptance coverage={pct(m['accepted_layout_coverage'])}. Exact spatial-count accuracy among accepted inputs={pct(m['accepted_layout_accuracy'])}, not the headline all-input score.

上一版权重在同批新场景、相同新背景参考下的区域F1为{pct(old['occupied_zone_macro_f1'])}、总数准确率{pct(old['count_accuracy'])}。旧权重没有重训；该比较不是原版简化房间的成绩。交界600场景区域F1 {pct(r['splits']['boundary_test']['occupied_zone_macro_f1'])}、总数准确率{pct(r['splits']['boundary_test']['count_accuracy'])}、完整空间计数{pct(r['splits']['boundary_test']['exact_joint_accuracy'])}。

The previous v4 weights use the same new background reference but are not retrained. These are new-scene comparisons, not old simple-room scores. Boundary scenes are reported separately.

纯推断中位数{r['inference_ms']['median']:.1f}ms、p95 {r['inference_ms']['p95']:.1f}ms，不含传播仿真、信号采集或HTTP；模型{r['model_bytes']/1024**2:.1f}MiB。

## 压力诊断 / Stress diagnostics

以下条件复用同一批300个独立测试几何，不能累计为多批独立场景，也不参与选参。F1和数量列描述原始分类候选；质量/置信门限决定是否输出Unknown或Uncertain。

| 条件 | N | 区域F1 | 总数准确率 | 质量可用覆盖 | 置信接受覆盖 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(stresslines) if stresslines else '| Pending | — | — | — | — | — |'}

缺少必需链路会输出Unknown，不能宣称本模型已支持更少设备。标准测试成绩不代表压力条件、其他房间或真实家庭的成绩。

Stress variants reuse the same300 geometries. Missing required links returnUnknown; reduced-device localization is not supported by this model. Standard scores do not establish different-room or real-home generalization.

## 使用、复现与限制 / Use, reproduction and limits

打开 http://127.0.0.1:8769/ 或双击START_MULTI.bat。选择1/2/3件目标并拖动摆放，固定柜体/设备会自动避让。开始检测持续生成信号并推断，暂停保留快照。灰色BG不计数，右图是预测中心区域与每区件数。低置信结果保留不确定；不能保证放入任意物体都能定位。

Choose1/2/3 targets, drag to place, and Start. Fixed background/radio clearances are handled automatically. Pause retains a snapshot. BG is excluded from target counts. Predictions are coarse center regions; uncertainty and errors remain visible.

复现顺序：home_v5_data.py --workers5 → home_v5_audit.py --files → home_v5_cache.py → home_v5_train.py → home_v5_evaluate.py → home_v5_stress.py。候选服务 app.py --port8773 --model-version5，运行home_v5_verify.py，实际页面检查后才可home_v5_release.py activate；rollback返回保留的v4。不要删除冻结标记后利用已打开最终测试调参。

For a new experiment, use fresh versioned outputs and independently held-out final data. Existing complete stages resume; opened final tests must not guide retuning. The release helper requires quality/data/API/browser evidence. Rollback restores v4.

当前仍是**固定房间、已知背景**验证，不是跨房间泛化。材质使用薄层近似，不是完整全波解算；没有完整家居材质、运动人体、实际射频误差或真实CSI验证。至少1件目标，空场仅作背景标定，没有训练0件目标类别。模型和高置信结果仍可能出错。

This is a fixed-room, known-background simulation, not demonstrated room generalization. Material slab approximations and simplified receiver errors remain. No zero-target class is trained; the empty room is used only for calibration. Confident errors remain possible.

证据 / Evidence: outputs/home_v5/results.json、frozen.json、data_audit_full.json、verification.json、ui_verification.json。模型SHA256: {sha(O5/'model.joblib')}。图表与截图 / Figures: figures/home_v5/。
"""
 report=report.replace('--workers5','--workers 5').replace('--port8773 --model-version5','--port 8773 --model-version 5')
 Path('HOME_SIMULATION_RESULTS.md').write_text(report,encoding='utf-8');print('Bilingual home-v5 report and3 figures generated')
if __name__=='__main__':main()
