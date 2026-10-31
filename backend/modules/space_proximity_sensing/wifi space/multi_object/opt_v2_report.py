
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opt_v2_common import ROOT,OUT2

def pc(x):return '—' if x is None else f'{100*x:.1f}%'
def main():
    r=json.loads((OUT2/'results.json').read_text());diag=json.loads((OUT2/'diagnostics.json').read_text());rows=[]
    for split in ['test','boundary_test']:
        m=r['splits'][split]
        for count in ['all','2','3']:
            a=m if count=='all' else m['by_true_count'][count];rows.append(f"| {split} / {count} | {a['n']} | {pc(a['occupied_zone_macro_f1'])} | {pc(a['count_accuracy'])} | {pc(a['exact_occupied_layout_accuracy'])} |")
    base=r['baseline_comparison']['test'];table=[]
    for name,a in [('Original single frame',base['v1_single_frame']),('Original with same 16-frame mean',base['v1_same_16_frame_average']),('Optimized 16 frames',r['splits']['test'])]:table.append(f"| {name} | {pc(a['occupied_zone_macro_f1'])} | {pc(a['count_accuracy'])} | {pc(a['exact_occupied_layout_accuracy'])} |")
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));colors=['#8396a5','#dea64e','#148f85'];labels=['Old / 1 frame','Old / 16 frames','Optimized / 16 frames'];models=[base['v1_single_frame'],base['v1_same_16_frame_average'],r['splits']['test']]
    for ax,key,title in [(axes[0],'exact_occupied_layout_accuracy','Exact whole-layout accuracy'),(axes[1],'count_accuracy','Object-center-zone count accuracy')]:
        vals=[100*a[key] for a in models];ax.bar(labels,vals,color=colors);ax.set_ylim(0,100);ax.set_ylabel('% of all 1,500 held-out scenes');ax.set_title(title);ax.grid(axis='y',alpha=.15)
        for i,v in enumerate(vals):ax.text(i,v+1,f'{v:.1f}%',ha='center')
    fig.suptitle('Same new physical test scenes — no true object count supplied');fig.tight_layout();folder=ROOT/'figures';folder.mkdir(exist_ok=True);fig.savefig(folder/'optimized_v2_comparison.png',dpi=160);plt.close(fig)
    stress='\n'.join(f"| {name} | {a['metrics']['n']} | {pc(a['metrics']['exact_occupied_layout_accuracy'])} | {a['unknown']} | {a['accepted']} |" for name,a in diag['stress'].items())
    examples='\n'.join(f"| {a['name']} | {','.join('Z'+str(o['zone']+1) for o in a['truth'])} | {','.join('Z'+str(o['zone']+1) for o in a['optimized']['regions']) or 'Unknown'} | {a['optimized']['status']} |" for a in diag['known_scenes'])
    m=r['splits']['test'];text=f'''# Multi-object optimization / 多物体优化结果

Current evidence is in `outputs/optimized_v2/`; original v1 evidence is retained separately.
当前结果来自重新训练与独立新测试，不是修改页面上的数字。

## Changes / 做了什么

1. Added 12,000 training scenes, emphasizing three objects and boundaries. Together with **only** the original 3,000 training scenes: 15,000 physical training scenes, 30,000 augmented windows. Original validation/calibration/test scenes never entered training.
2. Each observation window contains **16 distinct noisy receiver frames of one static geometry**. Averaging reduces receiver noise. This is extra observation time/information; the old model is also tested on the same frame averages to separate its benefit.
3. Extracted 1,600 amplitude, relative complex-response and frequency-correlation features. Requires a fixed, synchronized/phase-calibrated sensing layout plus the empty-room reference. This does not work unchanged with arbitrary receiver clock phases.
4. Two occupancy MLPs, a learned count classifier and a 129-class layout classifier are combined by a structured decoder. The predictor sees only CSI. It infers the count; the user's simulator count is never used to force the answer.
5. Model checkpoints and decoder weights use a new **900-scene validation** split. Temperature, confidence gate and quality checks use **600 separate calibration scenes**. New test 1,500 and boundary test 450 are used after selection.

训练物理场景从3,000增加到15,000，增强后的30,000个窗口不是30,000个独立摆放。模型不读取物体数量、坐标和尺寸；1–3件、各中心区域不同是预先限定的任务范围。新模型以完整布局表现选型，未把测试样本改作训练数据。

## Same-test comparison / 同一批测试对照

| Method | Zone Macro-F1 | Count accuracy | Exact whole layout |
|---|---:|---:|---:|
{chr(10).join(table)}

![Same-test comparison](figures/optimized_v2_comparison.png)

## All scenes, including uncertain cases / 所有场景均计入

| Split / true count | N | Zone Macro-F1 | Count accuracy | Exact whole layout |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Zone F1 is not the probability that a whole room is correct. Exact layout requires all nine occupied/empty labels to match. Boundary test is a different, deliberately difficult placement distribution; do not compare it with ordinary test as if the records were the same.

测试中通过完整布局置信度门限的比例为 **{pc(m['accepted_layout_coverage'])}**，其中完整布局正确率为 **{pc(m['accepted_layout_accuracy'])}**。这是一个筛选后的子集，不能替代上表全部样本的成绩，也不能保证自定义场景全部正确。

## Supplemental diagnostics / 补充检查

120 fixed new test scenes; these are diagnostics, not the official full-test score. Phase drift intentionally violates the calibrated-phase assumption.

| Condition | N | Exact layout | Unknown | Accepted |
|---|---:|---:|---:|---:|
{stress}

| Previously observed scene | True zones | New estimated zones | Status |
|---|---|---|---|
{examples}

GPU ray-trace recomputation evidence: `outputs/optimized_v2/diagnostics.json` → `physical_recompute`. Full manifest/file checks and API tests: `outputs/optimized_v2/verification.json`.

## How to test / 在前端怎么验证

Open http://127.0.0.1:8767/ using `START_MULTI.bat`, refresh the existing tab.

1. 自定义2或3个不同档位的物体，点击“生成联合信号并检测”。左边是真值，右边是模型估计，不会用真值修正。比较区域和数量，再移动到区域边缘重复。
2. 切换“独立测试：2件/3件”，开始持续检测，默认20次。也可选“交界位置：独立压力测试”。记录是独立摆放，不是连续物体运动。
3. 隐藏参考真值，确认模型结果不变。实线表示完整布局通过门限，黄虚线表示不确定候选；高分数仍可能出错。
4. 下载纯信号JSON再上传，不提供物体信息，应得到一致的预测。优化版格式为real/imag [16,32,64]，旧单帧[32,64]仍接受，但不会冒充16帧已验证结果。
5. 把SNR从30改为20/10 dB，查看退化、Unknown和误判，并导出运行记录。不能把弱信号下输出Unknown说成准确识别。
6. 查看“改进前后”表，核对两件、三件、交界测试的完整布局准确率。正式成绩使用固定完整测试，前端随机20次只是演示。

## Limits / 仍然做不到

Fixed 6×6 m room, four transmitters/eight receivers, 32 links, axis-aligned solid wood boxes in three known size bands, at most one per band and distinct center zones. Simplified opaque/specular Sionna scene, no diffraction/refraction/diffuse scattering. No real-home, device or arbitrary furniture generalization has been established. Output is center-zone occupancy, not exact coordinates, furniture identity or recovered 3D shape. The auxiliary size-band model was not retrained and remains experimental.

主输出仍是区域级判断，不是恢复沙发/桌子的真实外形。空房参考、固定设备和相位校准都是当前模型的前提。真实采集与新房间部署需要另外验证，不能用这些仿真分数代替。

Inference median {r['inference_ms']['median']:.2f} ms, p95 {r['inference_ms']['p95']:.2f} ms; excludes RT, acquisition and HTTP. Model bundle {r['model_bytes']/1048576:.2f} MiB.
'''
    ui_path=OUT2/'ui_verification.json'
    if ui_path.exists():
        ui=json.loads(ui_path.read_text());demo=ui['run'];text+=f"\n## Browser demonstration / 前端实际演示\n\n20-scene random three-object run: {demo['exact_matches']}/{demo['n']} complete layouts correct ({pc(demo['exact_accuracy'])}); {demo['accepted']} accepted layouts and {demo['unknown']} Unknown. All errors remain in `outputs/optimized_v2/ui_run_20.json`. Functional UI checks passed; they do not imply perfect spatial predictions.\n\n刚完成的20次随机三物体演示，完整匹配{demo['exact_matches']}次；不能用这20次替代完整675个三物体测试场景的58.2%成绩。纯信号上传/下载、隐藏真值、英文/双语、低SNR拒绝、自动结束和暂停清空均已在浏览器操作验证。\n\n![Actual frontend](figures/optimized_v2_three_objects.png)\n\n![Same-test comparison in UI](figures/optimized_v2_results_ui.png)\n"
    (ROOT/'OPTIMIZATION_RESULTS.md').write_text(text,encoding='utf-8');print('Report and figure saved')
if __name__=='__main__':main()
