
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multi_common import ROOT,OUT

def pc(x):return '—' if x is None else f'{x*100:.1f}%'
def main():
    r=json.loads((OUT/'results.json').read_text());test=r['splits']['test'];fig=ROOT/'figures';fig.mkdir(exist_ok=True)
    rows=[('All held-out scenes / 全部测试',test),('Two objects / 两件',test['by_true_count']['2']),('Three objects / 三件',test['by_true_count']['3'])]
    table='\n'.join(f"| {n} | {m['n']} | {pc(m['occupied_zone_macro_f1'])} | {pc(m['typed_object_f1'])} | {pc(m['count_accuracy'])} | {pc(m['exact_typed_layout_accuracy'])} |" for n,m in rows)
    compare='\n'.join(f"| {c['name']} | {pc(c['validation']['typed_object_f1'])} | {pc(c['validation']['occupied_zone_macro_f1'])} | {c['training_s']:.1f} s |" for c in r['comparison'])
    f,ax=plt.subplots(figsize=(10,4.5));x=np.arange(3);w=.2
    for offset,key,label,color in [(-1.5,'occupied_zone_macro_f1','Occupied-zone F1','#288da1'),(-.5,'typed_object_f1','Size + zone F1','#5ebfa6'),(.5,'count_accuracy','Count accuracy','#bd99df'),(1.5,'exact_typed_layout_accuracy','Exact layout','#e6ac6d')]:
        bars=ax.bar(x+offset*w,[m[key]*100 for _,m in rows],w,label=label,color=color);ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
    ax.set(xticks=x,xticklabels=['All test scenes','Two objects','Three objects'],ylim=(0,112),ylabel='Percent',title='Different multi-object metrics describe different capabilities');ax.legend(ncol=2,frameon=False,fontsize=9);f.tight_layout();f.savefig(fig/'multi_object_results.png',dpi=150);plt.close(f)
    report=f'''# Multi-object measured results / 多物体实验结果

The system now jointly simulates 2–3 differently sized objects and makes signal-based region/size-category proposals. It does not recover arbitrary dimensions. Current performance must be judged using the actual held-out numbers below.

当前已能联合模拟2–3件不同大小物体，并根据整组信号输出区域与尺寸档位候选。是否能可靠识别必须看下面的独立测试，不能引用单物体版本的准确率。

## Scope / 实验范围

Fixed 6×6 m room; 4 TX / 8 RX and 32 links; 64 complex frequencies/link; three nominal size bands with ±8% dimension variation. At most one item per band, different center zones, no collisions. The map renders canonical dimensions for the predicted category. It is not exact dimension regression, furniture semantics, arbitrary 3D imaging or real-home sensing.

固定房间、小中大三档、每档最多一件、中心位于不同区域。Small=0.6×0.5×0.8 m；Medium=1.1×0.8×1.1 m；Large=1.6×1.0×1.4 m。右图尺寸是预测类别的典型值，不是真值尺寸，也不是精确测量值。

## Dataset / 数据

4,800 independent physical scenes: 3,000 train, 600 validation, 300 calibration, 900 test. Four receiver-noise observations/scene give 12,000 training observations, not 12,000 independent scenes. Each raw record stores complex CSI [32,64]. The full scene is ray traced with all objects present simultaneously, including occlusion and reflections. The target count, geometry, dimensions and record ID are not classifier inputs.

原始信道跨训练/评估重复数：{r['duplicate_channels']}。场景的对象尺寸、坐标及相互碰撞由数据检查验证。模型只接收544维信号特征。

## Model comparison / 模型比较

| Model | Validation typed-object F1 | Validation occupied-zone F1 | Fit time |
|---|---:|---:|---:|
{compare}

Selected using validation only: **{r['selected_model']}**. Three heads predict whether each size band is absent or in zone Z1–Z9. Calibration records are split into independent temperature and gate subsets. Test records are not used for fitting or model selection.

## Frozen evaluation / 独立评估

| Condition | N | Occupied-zone macro-F1 | Size + zone F1 | Count accuracy | Exact typed layout |
|---|---:|---:|---:|---:|---:|
{table}

![Measured multi-object metrics](figures/multi_object_results.png)

Occupied-zone F1 ignores size category. Typed-object F1 requires matching category and center zone; a wrong zone or wrong size category is an error. Exact-layout accuracy requires every head, including absence, to match. These scores are not interchangeable.

占用区域F1不要求尺寸类别正确；联合F1要求尺寸档位与区域同时正确；完整布局准确率要求全部物体及缺席类别都正确。因此不能把其中最高的数当作整体识别准确率。

Fully accepted scene coverage: {pc(test['fully_accepted_scene_coverage'])}. Exact-layout accuracy among those scenes: {pc(test['fully_accepted_exact_accuracy'])}. This filtered score must always be accompanied by its coverage. Uncertain objects are shown as dashed candidates; they are not accepted detections. No-object outputs are uncertain because empty rooms are outside the training scope.

## Runtime / 耗时

Warm feature/model/rejection median {r['inference_ms']['median']:.1f} ms, p95 {r['inference_ms']['p95']:.1f} ms. Ray tracing and HTTP are excluded. The joint-scene simulation takes additional time and the first call initializes the GPU engine. Model bundle {r['model_bytes']/1048576:.1f} MiB.

## How to verify / 如何验证

1. Open http://127.0.0.1:8767/ and simulate the two default objects.
2. Select three objects and move each one separately. Inspect misses, wrong size bands, wrong zones and count disagreement.
3. Choose the held-out two- or three-object replay; those records were not used for training.
4. Hide truth, note predictions, then reveal truth. No predicted object is moved using ground-truth coordinates.
5. Download and re-upload the signal-only JSON. It has no count, size or position fields. Metadata fields are rejected by the inference API.
6. Run `verify.py` with the service active. Evidence is in `outputs/verification.json`.

## Limitations / 限制

This is an experimental multi-object extension. Multiple objects make the inverse problem harder and can occlude each other's signal effects. Prediction of arbitrary size, repeated size categories, multiple objects in one region, arbitrary orientations/materials, changed sensor layouts and real-home performance are not validated. If the count or layout is wrong, waiting in continuous mode does not train or correct the model.

模型可能漏检、选错尺寸档位或区域。持续模式只是重复检测，不会自动学会当前摆放。不支持同一区域内分开识别多个物体，也不能把本实验说成真实客厅家具扫描。
'''
    opath=OUT/'occupancy_results.json'
    if opath.exists():
        o=json.loads(opath.read_text());m=o['splits']['test'];orows=[('All test scenes / 全部测试',m),('Two objects / 两件',m['by_true_count']['2']),('Three objects / 三件',m['by_true_count']['3'])]
        ot='\n'.join(f"| {name} | {a['n']} | {pc(a['occupied_zone_macro_f1'])} | {pc(a['zone_precision'])} | {pc(a['zone_recall'])} | {pc(a['count_accuracy'])} | {pc(a['exact_occupied_layout_accuracy'])} |" for name,a in orows)
        f,ax=plt.subplots(figsize=(9,4.5));xx=np.arange(3)
        for delta,key,label,color in [(-.25,'occupied_zone_macro_f1','Occupied-zone macro-F1','#2b91a0'),(0,'count_accuracy','Region-count accuracy','#73bba7'),(.25,'exact_occupied_layout_accuracy','Exact occupied layout','#e2aa72')]:
            bars=ax.bar(xx+delta,[a[key]*100 for _,a in orows],.25,label=label,color=color);ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        ax.set(xticks=xx,xticklabels=['All test scenes','Two objects','Three objects'],ylim=(0,108),ylabel='Percent',title='Primary multi-object occupancy model');ax.legend(frameon=False,ncol=1,fontsize=9);f.tight_layout();f.savefig(fig/'occupancy_results.png',dpi=150);plt.close(f)
        prefix=f'''# Primary result: multi-object occupied zones / 主输出：多物体占用区域

不同大小的2–3件物体已经可以进入同一个射线追踪场景，模型从联合信号估计占用区域。自动恢复每件物体尺寸并不可靠，因此主要界面使用独立训练的 **{o['selected_model']}**，输出区域分数与候选，而不画成已恢复的物体轮廓。

Joint simulation supports 2–3 differently sized cuboids. The primary **{o['selected_model']}** predicts occupied center zones, not individual dimensions. Size/zone assignment is a harder auxiliary experiment; its measured results are retained below.

| Test condition | N | Occupied-zone macro-F1 | Zone precision | Zone recall | Region-count accuracy | Exact occupied layout |
|---|---:|---:|---:|---:|---:|---:|
{ot}

![Primary occupied-zone results](figures/occupancy_results.png)

For three objects, exact full-layout accuracy is {pc(m['by_true_count']['3']['exact_occupied_layout_accuracy'])}. This must not be replaced by the higher F1 score. Missed objects and false occupied zones remain common enough that this is an experimental prototype, not reliable furniture reconstruction.

三件物体的完整区域布局正确率只有 {pc(m['by_true_count']['3']['exact_occupied_layout_accuracy'])}，不能把 {pc(m['by_true_count']['3']['occupied_zone_macro_f1'])} 的F1说成“每次全部识别正确”。尺寸由用户设置用于生成信号，主要预测图只高亮模型识别的区域，不照抄这些尺寸与坐标。

The scene assumptions, size bands, no-collision and distinct-center-zone constraints apply to both models. Selection and thresholds use validation; positive evidence gates use calibration. All numbers above use the same frozen 900-record test split and noise seeds. Neither target count nor object dimensions/positions enter the occupancy model.

---

# Auxiliary experiment: joint size category and location

'''
        qpath=OUT/'quality_audit.json'
        if qpath.exists():
            q=json.loads(qpath.read_text());prefix+=f"\nQuality-reference pass fraction: {pc(q['quality_accepted_scene_fraction'])}. Strong-evidence region precision: {pc(q['strong_region_precision'])}, recall: {pc(q['strong_region_recall'])}. These region-level scores do not establish complete-layout correctness. / 通过质量参考检查的场景比例为{pc(q['quality_accepted_scene_fraction'])}；较强区域的精确率{pc(q['strong_region_precision'])}、召回率{pc(q['strong_region_recall'])}，仍不能保证全部物体均被识别。\n\n"
        report=prefix+report
    if (OUT/'optimized_v2'/'ACTIVE').exists():report='> Historical v1 results / 旧版结果。当前优化版见 [OPTIMIZATION_RESULTS.md](OPTIMIZATION_RESULTS.md)。\n\n'+report
    (ROOT/'RESULTS.md').write_text(report,encoding='utf-8');print('Multi-object report saved:',ROOT/'RESULTS.md',flush=True)

if __name__=='__main__':main()
