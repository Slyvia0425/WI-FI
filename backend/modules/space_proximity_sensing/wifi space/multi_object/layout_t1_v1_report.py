
from pathlib import Path
import numpy as np
from layout_t1_v1_common import O5,read,sha

def main():
    r=read(O5/'results.json');f=read(O5/'frozen.json');m=r['splits']['test'];b=r['splits']['boundary_test']
    assert sha(O5/'model.joblib')==f['model_sha256']
    p=lambda v:f'{100*v:.2f}%'
    groups=[('All test scenes',m),('1 object',m['by_true_count']['1']),('2 objects',m['by_true_count']['2']),('3 objects',m['by_true_count']['3']),('Shared zones',m['by_sharing']['shared']),('Boundary stress test',b)]
    table='\n'.join('| '+name+' | '+str(a['n'])+' | '+p(a['occupied_zone_macro_f1'])+' | '+p(a['count_accuracy'])+' | '+p(a['exact_joint_accuracy'])+' |' for name,a in groups)
    text=f'''# T1 新布局实验结果 / New fixed-T1 experiment

T1 = (1.10, 1.20, 0.45) m；其他设备和房间背景固定。原布局模型独立保留。
T1 = (1.10, 1.20, 0.45) m; all other radios and the room background are fixed. The original-layout model is retained separately.

## 数据与算法 / Data and algorithm

本布局独立生成 46,600 个仿真场景：40,000 训练、2,400 验证、1,200 置信度校准、2,400 最终测试、600 交界压力测试。训练场景生成两个接收噪声观测，共 80,000 个窗口；同一几何场景不跨划分。每窗口 16 帧，32 条链路、64 个复数频点。包含木质目标、固定背景柜体、门窗、反射/散射/透射；不是完整的真实家用 Wi-Fi PHY。

46,600 independently generated simulation scenes: 40,000 training, 2,400 validation, 1,200 confidence calibration, 2,400 final test and 600 boundary stress cases. Two receiver observations per training geometry yield 80,000 windows. Each window has 16 frames, 32 links and 64 complex frequency samples. Furnishings, door/window and multipath are simulated; this is not a complete real home Wi-Fi PHY.

CSI 分别形成校准特征和相位对齐特征。CatBoost 学习区域占用，MLP 学习占用区域数、物体总数和空间计数状态，联合解码 219 种区域计数组合。模型选择仅用验证集；独立校准集设置温度与置信门限；冻结后打开最终测试集。

Calibrated and phase-aligned CSI features feed CatBoost occupancy and MLP region-count, physical-count and spatial-state models. Joint decoding covers 219 count vectors. Selection uses validation only, confidence uses separate calibration, and final tests open after model freezing.

## 全输入结果 / All-input results

| Group | N | Zone Macro-F1 | Object-count accuracy | Exact spatial counts |
|---|---:|---:|---:|---:|
{table}

区域 F1 衡量哪些区域有物体；数量准确率只要求总件数正确；完整空间计数要求九个区域的件数全部正确。它们不能相互替代。全部结果保留错误和不确定样本，没有只统计通过置信门限的结果。

Zone F1 measures occupancy, count accuracy requires the correct total, and exact spatial counts require every zone's count to match. These are different metrics. The table includes errors and uncertain inputs rather than only confidence-accepted cases.

通过置信门限的比例 / Accepted coverage: **{p(m['accepted_layout_coverage'])}**.
该子集联合正确率 / Exact spatial counts within that subset: **{p(m['accepted_joint_accuracy']) if m['accepted_joint_accuracy'] is not None else 'N/A'}**. 此分数不能代表全部输入 / This is not an all-input accuracy.

纯信号推理时间 / Signal-only inference: median **{r['inference_ms']['median']:.2f} ms**, p95 **{r['inference_ms']['p95']:.2f} ms**. 不含射线追踪、采集与 HTTP / Excludes ray tracing, acquisition and HTTP.

![Independent evaluation](figures/layout_t1_v1/evaluation.png)

## 验收与限制 / Acceptance and limits

预先约定门限 / Precommitted gates: zone F1 >=85%, count >=85%, exact spatial counts >=65%, shared-zone count >=80%.
最终测试门限通过 / Final-test gate passed: **{r['release_check']['eligible']}**.
这里只验证这一组固定位置；手动改成其他位置仍属于未验证布局。它不能证明任意设备摆放、真实家庭、任意家具材质或精确三维重建的性能。

Only this fixed preset is evaluated. Other manually edited positions remain unvalidated. These results do not establish arbitrary-layout, real-home, arbitrary-material or precise 3D reconstruction performance.

输出为九个区域的计数及置信度，支持 1–3 件目标及同区多物体；每档尺寸最多一件。图中颜色表示预测区域，不代表恢复了单件物体的真实轮廓。

Outputs are counts/confidence for nine zones, with one to three targets and shared zones, at most one target per size band. Colored regions are predictions, not recovered individual object shapes.

数据审计 / Raw audit: `outputs/layout_t1_v1/data_audit_full.json`.
信号推理核对 / Offline checks: `outputs/layout_t1_v1/offline_verification.json`.
接口验收 / API checks: `outputs/layout_t1_v1/api_verification.json`.
浏览器验收 / Browser checks: `outputs/layout_t1_v1/ui_verification.json`（单独记录 / recorded separately）.
模型 SHA-256 / Model SHA-256: `{f['model_sha256']}`.
使用方法见 / Usage: [LAYOUT_PRESET_GUIDE.md](LAYOUT_PRESET_GUIDE.md).
'''
    ui=read(O5/'ui_verification.json') if (O5/'ui_verification.json').exists() else {'status':'pending'}
    text+=f'\n\n## Interface acceptance / 界面验收状态\n\nPublic API:29 checks passed; original-model regression:25 checks passed. Browser status: **{ui["status"]}**. Browser automation could not reliably identify the current URL; no new passed browser test or UI screenshot is claimed.\n\n模型文件 / Model file: {r["model_bytes"]:,} bytes. 三物体完整空间计数 / Three-object exact spatial counts: {p(m["by_true_count"]["3"]["exact_joint_accuracy"])}; 交界压力测试 / Boundary exact spatial counts: {p(b["exact_joint_accuracy"])}. 困难布局仍会出错 / Difficult layouts remain imperfect.\n'
    Path('LAYOUT_T1_RESULTS.md').write_text(text,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    folder=Path('figures/layout_t1_v1');folder.mkdir(exist_ok=True,parents=True)
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    names=[a[0] for a in groups[:5]];x=np.arange(5);w=.25
    for j,(key,label,color) in enumerate([('occupied_zone_macro_f1','Zone Macro-F1','#167d9a'),('count_accuracy','Object count','#6c54b8'),('exact_joint_accuracy','Exact spatial counts','#c88a24')]):
        vals=[a[key] for _,a in groups[:5]];bars=axs[0].bar(x+(j-1)*w,vals,w,label=label,color=color)
        axs[0].bar_label(bars,labels=[f'{100*v:.1f}%' for v in vals],fontsize=7,padding=2)
    axs[0].set_xticks(x,names,rotation=20,ha='right');axs[0].set_ylim(0,1.12);axs[0].set_ylabel('Score (all inputs)');axs[0].yaxis.set_major_formatter(PercentFormatter(1));axs[0].legend(fontsize=8,loc='lower left');axs[0].set_title('New T1 layout: independent test')
    cm=np.array(m['object_count_confusion']);im=axs[1].imshow(cm,cmap='Blues')
    for i in range(3):
        for j in range(3):axs[1].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    axs[1].set_xticks(range(3),['1','2','3']);axs[1].set_yticks(range(3),['1','2','3']);axs[1].set_xlabel('Predicted object count');axs[1].set_ylabel('True object count');axs[1].set_title(f'Object count confusion (N={m["n"]})');fig.colorbar(im,ax=axs[1],shrink=.8)
    fig.savefig(folder/'evaluation.png',dpi=180);fig.savefig(folder/'evaluation.svg');plt.close(fig)
    print('Written LAYOUT_T1_RESULTS.md and evaluation plots.')

if __name__=='__main__':main()
