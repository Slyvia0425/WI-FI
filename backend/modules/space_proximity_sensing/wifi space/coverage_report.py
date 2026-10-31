
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coverage_common import ROOT,DATA,OUT,devices,valid_placement

def pct(x):return '—' if x is None else f'{100*x:.1f}%'
def main():
    r=json.loads((OUT/'results.json').read_text(encoding='utf-8'));dest=ROOT/'reports';fig=dest/'figures/coverage_v3';fig.mkdir(parents=True,exist_ok=True)
    labels={'validation':'模型选型验证 / Model validation','cal_temperature':'温度校准 / Temperature calibration','cal_guard':'候选与拒绝校准 / Candidate calibration','test':'全有效范围 / Full valid support','boundary_test':'交界专项 / Boundary only','device_shift':'设备偏移 / Shifted devices','material_shift':'材料变化 / Material change'}
    table='\n'.join(f"| {labels[s]} | {d['n']} | {pct(d['accuracy'])} | {d['macro_f1']:.4f} | {pct(d['coverage'])} | {pct(d['accepted_accuracy'])} | {d['zone_center_error_m']['mean']:.3f} |" for s,d in r['splits'].items())
    paired='\n'.join(f"| {labels[s]} | {d['new']['n']} | {pct(d['old']['accuracy'])} | {pct(d['new']['accuracy'])} | {pct(d['new']['coverage'])} | {pct(d['new']['accepted_accuracy'])} |" for s,d in r['baseline_comparison'].items() if s in ['test','boundary_test','device_shift','material_shift'])
    old=json.loads((ROOT/'data/manifest.json').read_text());collisions=0
    for row in old:
        offsets=row.get('device_offset_xy')
        if offsets is None and row['split']=='diagnostic_original_device_shift':offsets=[[.06,-.04] if i%2==0 else [-.06,.04] for i in range(8)]
        p=devices(offsets);collisions+=bool(np.any(np.all(np.abs(p[:,:2]-row['xy'])<=[.6,.4],axis=1)))
    cm=np.array(r['splits']['test']['confusion_matrix']);f,ax=plt.subplots(figsize=(7,6));im=ax.imshow(cm,cmap='Blues');ax.set(xticks=range(9),yticks=range(9),xticklabels=[f'Z{i+1}' for i in range(9)],yticklabels=[f'Z{i+1}' for i in range(9)],xlabel='Predicted center zone',ylabel='True center zone',title=f'Full valid support · {cm.sum()} held-out positions')
    for (i,j),n in np.ndenumerate(cm):ax.text(j,i,str(n),ha='center',va='center',color='white' if n>cm.max()/2 else '#29445b')
    f.colorbar(im,ax=ax);f.tight_layout();f.savefig(fig/'confusion.png',dpi=150);plt.close(f)
    keys=['test','boundary_test','device_shift','material_shift'];xx=np.arange(4);f,ax=plt.subplots(figsize=(10,4.5))
    a=ax.bar(xx-.18,[r['baseline_comparison'][k]['old']['accuracy']*100 for k in keys],.36,color='#8a9dae',label='Old central-range model');b=ax.bar(xx+.18,[r['splits'][k]['accuracy']*100 for k in keys],.36,color='#288b94',label='New full-support model');ax.bar_label(a,fmt='%.1f',padding=3);ax.bar_label(b,fmt='%.1f',padding=3);ax.set(xticks=xx,xticklabels=['Full support','Boundary only','Shifted devices','Material change'],ylim=(0,112),ylabel='Raw accuracy (%)',title='Paired comparison on identical new test signals');ax.legend(frameon=False);f.tight_layout();f.savefig(fig/'paired_comparison.png',dpi=150);plt.close(f)
    rows=json.loads((DATA/'manifest.json').read_text());f,axs=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
    for ax,rs,title in [(axs[0],old,'Previous central ranges'),(axs[1],rows,'New full valid support')]:
        a=np.array([q['xy'] for q in rs if q['split']=='train']);ax.scatter(a[:,0],a[:,1],s=2,alpha=.3,color='#288b94');ax.set(xlim=(0,6),ylim=(0,6),aspect='equal',xlabel='x (m)',title=title);ax.set_ylabel('y (m)')
        for v in [2,4]:ax.axvline(v,color='#abc0cb',lw=.6);ax.axhline(v,color='#abc0cb',lw=.6)
    f.tight_layout();f.savefig(fig/'training_coverage.png',dpi=150);plt.close(f)
    example=r['reported_boundary_example'];exold=example['old'];exnew=example['new']
    example_new='Z'+str(exnew['prediction']+1) if exnew['prediction'] is not None else '候选 / Candidates '+','.join('Z'+str(i+1) for i in exnew.get('candidate_zones',[])) if exnew.get('candidate_zones') else 'Unknown'
    report=f'''# 当前结果：全有效范围改进版 / Current full-support results

这是对“物体在区域交界，模型却判到远处”的实际修正。旧版 99.5% 仅对应中央摆放的小范围，不适用于新版全范围测试。下面所有新旧比较使用**同一批新信号、同一组噪声种子**。

This upgrade addresses the observed boundary failure. The earlier 99.5% belonged to a restricted central-range test. Paired comparisons below use identical new signals and noise seeds.

## 数据与问题复核 / Data and audit

- 新生成 10,035 个独立物理配置：6,480 训练、1,080 选型验证、225 温度校准、225 候选/拒绝校准、900 全范围测试、450 交界测试、405 设备偏移、270 材料变化。
- 四次噪声增强产生 25,920 个训练观测，仍只算 6,480 个物理配置。
- 合法中心范围 x=[0.61,5.39] m、y=[0.41,5.59] m，排除箱体与设备及 8 cm 间距重叠的部分。允许跨越区域线，类别按**箱体中心**所属区域定义。
- 旧数据精确复核：{collisions}/{len(old)} 个配置存在箱体与设备位置重叠。新版全部排除。旧分数保留为历史，不将其用于全范围性能宣传。
- 旧训练中央矩形总面积为 6.5664 m²；有效中心外框面积为 24.7604 m²，外框内仍需扣除设备禁放区。仅扩大 UI 可点击范围不等于扩大模型训练覆盖。

The scene, fixed box dimensions, carrier and 12-link/64-subcarrier format remain controlled simulation assumptions. This is not a real-home furniture scanner.

![Training coverage](figures/coverage_v3/training_coverage.png)

## 模型与校准 / Models and calibration

Selected on validation: **{r['selected_model']}**. Feature count: 204, including amplitude descriptors and within-link frequency correlations that cancel a constant per-link phase offset. Positions, labels, device coordinates and truth overlays are excluded from the classifier input.

Temperature uses its own 225 records. A different 225-record set defines candidate-score cutoff, confidence and feature-distance checks. Multiple candidate zones produce an uncertain result rather than an accepted box location. These empirical checks have no guaranteed conditional or shifted-domain reliability.

温度参数 {r['temperature']:.3f}；基础接受阈值 {r['threshold']:.2f}；单一区域证据门限 {r['set_cutoff']:.4f}，实际分数门限取两者较大值。没有区域满足证据门限时，显示分数最高的两个候选，但不接受物体位置。校准集上的条件准确率目标是否达到：{r['threshold_target_met']}。校准和模型选型不是使用同一批数据，最终测试未用于调参。

| Condition / 条件 | N | Raw accuracy | Macro-F1 | Accepted coverage | Accepted accuracy | Raw mean center error (m) |
|---|---:|---:|---:|---:|---:|---:|
{table}

All raw scores include rejected predictions. Coverage must accompany accepted accuracy. Zone-center error is not a continuously regressed coordinate error.

![Confusion matrix](figures/coverage_v3/confusion.png)

## 同信号比较 / Paired comparison

| Condition | N | Old raw accuracy | New raw accuracy | New coverage | New accepted accuracy |
|---|---:|---:|---:|---:|---:|
{paired}

![Paired comparison](figures/coverage_v3/paired_comparison.png)

Validation Macro-F1 improvement: {r['validation_gain']:.4f}. The predeclared activation gate requires at least 0.05 improvement on validation, not a favorable hand-picked test example.

## 用户交界示例 / Reported boundary example

Diagnostic placement (2,3) m, 30 dB, seed 42: old raw zone Z{exold['raw_prediction']+1}; new output **{example_new}**. This point is not used for fitting or headline scoring. A candidate set represents uncertainty about the center zone, not a recovered object footprint.

该精确边界点按标签规则属于 Z5，新版的 Z4 仍是相邻区域误判，不能宣称这个点已经完全识别正确。这里保留失败结果，不用真值修改模型输出。

## 使用上的变化 / UI changes

默认连续检测 20 次后自动结束；1.5 秒间隔下约 30 秒，加上实际计算耗时。可以改为一轮测试或一直运行。自动结束保留最后一次结果快照，手动暂停清空有效位置。页面明确提示模型已经训练，不需要等待它“学会”当前位置。

新版左图显示完整有效中心外框及设备禁放阴影，去掉旧的九个中央训练小框。右图只在接受单一区域时画确定箱体；证据不足时显示候选区域，阴影不代表已确定的物体。新增交界测试选择和同信号新旧比较表。

## 复现与限制 / Reproduce and limits

Run `coverage_rebuild.ps1` for generation, training and activation. Start the service, then run `verify.py`; its report follows the active profile. New evidence is in `outputs/coverage_v3/`, protocol/data in `data/coverage_v3/`. The original data and models are preserved for comparison.

Even an improved classifier can misclassify, especially near boundaries and under device changes. Scores still describe an idealized simulator, one fixed-size axis-aligned object and a fixed room. Size/height are known priors; no semantic furniture recognition, arbitrary 3D reconstruction or live Wi-Fi input is claimed.

Warm inference median: {r['inference_ms']['median']:.2f} ms, p95 {r['inference_ms']['p95']:.2f} ms, excluding simulation/HTTP. Model bundle: {r['model_bytes']/1048576:.2f} MiB.
'''
    (dest/'LATEST_RESULTS.md').write_text(report,encoding='utf-8')
    print('Full-support report generated:',dest/'LATEST_RESULTS.md',flush=True)

if __name__=='__main__':main()
