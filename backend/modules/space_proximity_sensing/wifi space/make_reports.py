
import json,hashlib,subprocess,sys,zipfile,platform
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from learning import ROOT

def pc(x):return f'{100*x:.2f}%'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=ROOT/'outputs';reports=ROOT/'reports';fig=reports/'figures';fig.mkdir(exist_ok=True)
    r=json.loads((out/'results.json').read_text());old=json.loads((out/'phase1/results.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    cm=np.array(r['splits']['test']['confusion_matrix'])
    f,ax=plt.subplots(figsize=(7,6));im=ax.imshow(cm,cmap='Blues',vmin=0);ax.set(xticks=range(9),yticks=range(9),xticklabels=[f'Z{i+1}' for i in range(9)],yticklabels=[f'Z{i+1}' for i in range(9)],xlabel='Predicted zone',ylabel='True zone',title=f'Frozen standard test · {cm.sum()} unseen positions')
    for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else '#26435b')
    f.colorbar(im,ax=ax,label='Configurations');f.tight_layout();f.savefig(fig/'confusion_matrix.png');plt.close(f)
    labels=['Fixed layout','Device shift','Wood → chipboard','20 dB noise','10 dB noise','0 dB noise']
    values=[r['splits']['test'],r['splits']['device_shift'],r['splits']['material_shift'],*[r['noise_stress'][str(k)] for k in [20,10,0]]]
    f,ax=plt.subplots(figsize=(10,4.5));x=np.arange(len(labels));a=ax.bar(x-.17,[v['accuracy']*100 for v in values],width=.34,label='Raw accuracy',color='#287c9a');b=ax.bar(x+.17,[v['coverage']*100 for v in values],width=.34,label='Accepted coverage',color='#53ba9f');ax.bar_label(a,fmt='%.1f',padding=3,fontsize=9);ax.bar_label(b,fmt='%.1f',padding=3,fontsize=9);ax.set(xticks=x,xticklabels=labels,ylim=(0,114),ylabel='Percent',title='Accuracy and coverage must be reported together');ax.legend(loc='upper right',frameon=False);f.tight_layout();f.savefig(fig/'conditions_and_coverage.png');plt.close(f)
    before=old['splits']['device_shift']['accuracy'];after=r['diagnostics']['diagnostic_original_device_shift']['accuracy']
    f,ax=plt.subplots(figsize=(6,4));bars=ax.bar(['Fixed-layout training','Device-randomized training'],[before*100,after*100],color=['#8095a8','#287c9a']);ax.bar_label(bars,fmt='%.1f%%',padding=4);ax.set(ylim=(0,100),ylabel='Raw accuracy (%)',title='Same 135 diagnostic device-shift records');f.text(.5,.01,'Diagnostic comparison only; final shifted-device test uses 270 new positions.',ha='center',fontsize=8);f.tight_layout(rect=(0,.03,1,1));f.savefig(fig/'device_randomization_comparison.png');plt.close(f)
    rows=[]
    names={'validation':'Validation / 联合验证','test':'Standard test / 固定布局测试','device_shift':'Shifted devices / 设备坐标偏移','material_shift':'Wood → chipboard / 材料变化'}
    for k,v in r['splits'].items():rows.append(f"| {names[k]} | {v['n']} | {pc(v['accuracy'])} | {v['macro_f1']:.4f} | {pc(v['coverage'])} | {pc(v['accepted_accuracy']) if v['accepted_accuracy'] is not None else '—'} | {v['zone_center_error_m']['mean']:.3f} m |")
    for k in [20,10,0]:
        v=r['noise_stress'][str(k)];rows.append(f"| Test + {k} dB noise / 加噪 | {v['n']} | {pc(v['accuracy'])} | {v['macro_f1']:.4f} | {pc(v['coverage'])} | {pc(v['accepted_accuracy']) if v['accepted_accuracy'] is not None else '—'} | {v['zone_center_error_m']['mean']:.3f} m |")
    valcompare='\n'.join(f"| {v['name']} | {v['validation_macro_f1']:.4f} | {v['training_seconds']:.2f} s |" for v in r['comparison'])
    recalls='\n'.join(f"| Z{i+1} | {r['splits']['test']['per_class'][str(i)]['support']:.0f} | {r['splits']['test']['per_class'][str(i)]['recall']:.4f} |" for i in range(9))
    report=f'''# Measured results / 实测结果

Scope: Sionna RT simulation; one known-size cuboid, nine zones, one simplified room. These are not results from real furniture sensing.

边界：Sionna RT 仿真，一个固定尺寸箱体、九个区域和一个简化房间。所有分数来自保存的信号数据和实际模型推断。

## Data / 数据

4,050 distinct physical configurations: 2,250 training, 450 joint validation, 405 final base-layout test, 270 final shifted-device test, 135 final material test, and 540 original diagnostic configurations. Four noisy observations per training configuration give 9,000 training observations, not 9,000 independent positions. Each raw record stores complex64 CSI [12 links,64 subcarriers]. All manifest rows are audited; train/evaluation raw-channel overlap is zero.

共 4,050 个配置；训练增强只增加接收观测，不增加独立位置数量。原第一阶段测试集转为诊断用途，最终测试使用新位置。中心摆放只覆盖各区域中央的小范围，不代表任意客厅全空间扫描。

## Final evaluation / 最终评估

| Condition / 条件 | N | Raw accuracy | Macro-F1 | Coverage | Accepted accuracy | Mean zone-center error |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Raw accuracy and Macro-F1 include all classifier predictions before rejection. Coverage is the accepted fraction. Accepted accuracy is undefined when all samples are rejected. Errors are measured from the predicted **zone center**, not a learned continuous coordinate, and include raw predictions before rejection.

原始分数包含低置信度和被拒绝输入的原始分类结果。接受比例不能省略；0 条被接受时，接受样本准确率记为“—”。标准测试正确区域对应的中心误差下限均值为 {r['splits']['test']['position_floor_m']:.3f} m，说明摆放范围本身影响误差。

![Standard-test confusion matrix](figures/confusion_matrix.png)

![Conditions and coverage](figures/conditions_and_coverage.png)

## Selection / 模型选型

| Model | Joint-validation Macro-F1 | Fitting time |
|---|---:|---:|
{valcompare}

Selected: **{r['selected_model']}**, using validation only. The frozen fixed-layout test is easier than joint validation because joint validation includes random device offsets. Its higher score is not evidence of test-driven selection.

联合验证包含设备坐标随机变化，所以联合验证分数低于固定设备测试并不矛盾。最终选型没有使用新测试标签。

Confidence threshold: {r['threshold']:.2f}. The validation precision target was not achieved at sufficient coverage, so the documented 0.60 fallback applies. It does **not** establish 90% reliability under device shifts. Nearest-neighbor feature-distance limit: {r['novelty_limit']:.3f}, selected as the 99th percentile on joint validation. This is a heuristic and cannot guarantee rejection of all unfamiliar rooms or objects.

置信度未做概率校准。特征距离超限同样输出 Unknown，即使原始模型分数较高。高噪声下全部拒绝说明系统停止给出有效位置，不能写成“噪声下仍然准确”。

## Device-shift improvement / 设备偏移改进

On the **same 135 original diagnostic records**, accuracy changed from {pc(before)} to {pc(after)} after device-coordinate randomization. These records informed the development direction and therefore remain diagnostic. The new independent shifted-device test reports {pc(r['splits']['device_shift']['accuracy'])}. Different datasets must not be silently compared as if they were the same test.

同一批诊断数据有改善，但新偏移测试仍存在明显错误，不能宣称泛化已解决。该测试只改变模拟设备坐标，不涉及不同真实硬件型号。

![Diagnostic comparison](figures/device_randomization_comparison.png)

## Per-class recall / 各类召回率

| Zone | N | Recall |
|---|---:|---:|
{recalls}

## Runtime and model / 耗时与模型

Warm feature extraction, classification and rejection: median {r['inference_ms']['median']:.3f} ms, p95 {r['inference_ms']['p95']:.3f} ms over 50 calls. This excludes ray tracing, HTTP, startup and rendering. Model bundle: {r['model_bytes']/1048576:.2f} MiB; its nearest-neighbor reference accounts for much of the size. The (96,48) MLP has 17,865 learned weights and biases for 132 inputs and 9 outputs.

模型参数量与模型文件大小不同；模型包还包含标准化、参考信号和异常检查的训练特征。不能将模型推断时间作为完整实时系统端到端延迟。

## Verification / 验证

Automated check evidence: `outputs/verification.json`. It validates actual files, shape/finite values, split separation, signal-only API, truth independence, deterministic noise, new-scene RT and severe-noise rejection. Frontend evidence and screenshots are recorded separately in `outputs/ui_verification.json` and `reports/figures/` when available.

## Limits / 限制

- Fixed room, one axis-aligned known box; no semantic identification, object dimensions or arbitrary 3D reconstruction.
- Simplified specular, opaque propagation; no refraction, diffraction, other furniture or real hardware effects.
- Restricted central placement ranges; independent positions are sampled within the same defined task.
- Simulated device-coordinate robustness is improved but incomplete. No claims about different real devices, real users, pets or new homes.
- Displayed 3D box uses known height. Continuous processing is serial simulation/test playback, not live Wi-Fi capture or a physical movement trajectory.

These limitations should stay visible in the presentation and final report. The original human CSI-Bench model is separate and cannot validate the furniture claim.
'''
    if (ROOT/'active_experiment.json').exists():report='> Historical central-range experiment. Current full-support results: [LATEST_RESULTS.md](LATEST_RESULTS.md). / 本文件为早期中央范围实验，当前结果见最新报告。\n\n'+report
    (reports/'RESULTS.md').write_text(report,encoding='utf-8')
    freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (reports/'environment-freeze.txt').write_text(freeze,encoding='utf-8')
    files={str(p.relative_to(ROOT)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in ROOT.rglob('*') if p.is_file() and '.venv' not in p.parts and '__pycache__' not in p.parts and p.name!='provenance.json'}
    provenance=dict(generated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),scope=r['scope'],engine='sionna-rt 1.2.2 / CUDA',python=platform.python_version(),files=files)
    (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    target=ROOT.parent/'static_obstacle_submission.zip';tmp=target.with_suffix('.zip.partial')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in ROOT.rglob('*'):
            if p.is_file() and '.venv' not in p.parts and '__pycache__' not in p.parts:z.write(p,'static_obstacle/'+p.relative_to(ROOT).as_posix())
    tmp.replace(target)
    print('Report:',reports/'RESULTS.md');print('Package:',target,'bytes',target.stat().st_size)

if __name__=='__main__':main()
