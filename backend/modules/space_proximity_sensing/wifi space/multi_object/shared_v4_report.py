
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,recall_score
from multi_common import ROOT,OUT
from shared_v4_common import O4,read,atomic,sha,now
def pct(v):return '—' if v is None else f'{100*v:.1f}%'

def main():
    r=read(O4/'results.json');m=r['splits']['test'];old=r['baseline_comparison']['test']['v3_region_count_as_object_count'];active=read(OUT/'active_profile.json')['version'];figdir=ROOT/'figures'/'shared_v4';figdir.mkdir(exist_ok=True,parents=True)
    with np.load(O4/'predictions_test.npz') as p:
        yc=p['true_object_count'];pc=p['pred_object_count'];z=p['true_zone_count'];pz=p['pred_zone_count']
        diagnostics=dict(object_count_recall=recall_score(yc,pc,labels=[1,2,3],average=None,zero_division=0).tolist(),per_zone_count_confusions=[confusion_matrix(z[:,i],pz[:,i],labels=[0,1,2,3]).tolist() for i in range(9)],scope='Derived after freezing; no tuning from final-test diagnostics.')
    atomic('count_diagnostics.json',diagnostics)
    cm=np.array(m['object_count_confusion']);fig,ax=plt.subplots(figsize=(5.6,4.5),layout='constrained');im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax)
    for i in range(3):
        for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set(title='Physical object count: new final test',xlabel='Predicted count',ylabel='True count',xticks=[0,1,2],yticks=[0,1,2],xticklabels=[1,2,3],yticklabels=[1,2,3]);fig.savefig(figdir/'object_count_confusion.png',dpi=180);plt.close(fig)
    cms=np.asarray(diagnostics['per_zone_count_confusions']);fig,axes=plt.subplots(3,3,figsize=(9,8),layout='constrained')
    for i,ax in enumerate(axes.flat):
        ax.imshow(cms[i],cmap='Blues',vmin=0,vmax=cms.max());ax.set(title=f'Z{i+1}',xticks=range(4),yticks=range(4));
        for a in range(4):
            for b in range(4):ax.text(b,a,str(cms[i,a,b]),ha='center',va='center',fontsize=8,color='white' if cms[i,a,b]>cms.max()/2 else 'black')
    fig.suptitle('Per-zone object counts: rows=true, columns=predicted');fig.savefig(figdir/'zone_count_confusions.png',dpi=180);plt.close(fig)
    groups=[('Distinct zones','distinct'),('Shared zones','shared'),('All in one zone','all_in_one_zone')];x=np.arange(len(groups));fig,ax=plt.subplots(figsize=(8.6,4.6),layout='constrained')
    for offset,model,color,label in [(-.18,old,'#7f96ab','v3: occupied-zone count'),(.18,m,'#16a38c','v4: learned object count')]:
        values=[100*model['by_sharing'][key]['count_accuracy'] for _,key in groups];bars=ax.bar(x+offset,values,.36,color=color,label=label);ax.bar_label(bars,fmt='%.1f%%',padding=3,fontsize=9)
    ax.set(xticks=x,xticklabels=[n for n,_ in groups],ylabel='Physical-count accuracy (%)',ylim=(0,100),title='Same new test scenes, including all uncertain/rejected inputs');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2,fontsize=9,frameon=False);ax.spines[['top','right']].set_visible(False);fig.savefig(figdir/'shared_count_comparison.png',dpi=180);plt.close(fig)
    rows=[]
    for name,key in [('全部 / All',None),('不同区 / Distinct','distinct'),('存在同区 / Shared','shared'),('全部同区 / All in one','all_in_one_zone')]:
        a=m if key is None else m['by_sharing'][key];b=old if key is None else old['by_sharing'][key];rows.append(f'| {name} | {a["n"]} | {pct(a["occupied_zone_macro_f1"])} | {pct(b["count_accuracy"])} → {pct(a["count_accuracy"])} | {pct(a["exact_occupied_layout_accuracy"])} | {pct(a["exact_joint_accuracy"])} |')
    report=f'''# 同区域多物体升级 / Shared-zone upgrade

生成 / Generated: {now()}。当前启用 / Active profile: **v{active}**。候选发布检查 / Release check: **{r['release_check']['eligible']}**。

## 两处修改 / What changed

设备限制改为**自动避让**：拖动或提交坐标时，仅微调所选物体，避开设备、墙面和其他物体；阴影表示所选物体中心的禁放位置。设备不能被放进实心箱体内部，实体重叠也仍然禁止。305个前端投影案例通过实际后端几何校验，浏览器里确认B从(5.15,3.4)自动调整为(5.01,3.4)，A未移动。

Device obstruction is handled by automatic placement adjustment and visible center clearances. Only the edited object moves. Physical interpenetration and embedding radios inside solid objects remain invalid.

同一区域可以有多件物体。新增信号和标签后重新训练，分别输出**物体总数、占用区域数、每个区域的物体数**。例如Z5有两件，正确输出是总数2、区域数1、Z5×2。三件分布在两个区域时，模型也学习哪个区域有两件，而不是把总数直接当区域数。

Shared center zones are supported by retraining. Physical total, number of occupied zones, and per-zone counts are separate outputs. Two objects inZ5 means2 objects,1 occupied zone,Z5×2. The model also learns count allocation when3 objects occupy2 zones.

## 数据与训练 / Data and training

新增16,250个真实联合射线追踪场景：训练12,000个同区布局，验证1,200、校准800、最终测试1,800、交界450。只复用旧版本训练部分36,000条，所以总训练48,000个物理场景、96,000个16帧增强窗口。原始信号均已核查，未发现训练/评估完全相同信号交叠或冲突计数标签。

Added16,250 joint-RT geometries. Training reuses old TRAIN data only:48,000 physical scenes and96,000 receiver windows in total. Validation, calibration and new final tests are separate. Counts/positions/size labels never enter the predictor as inputs.

两个信号分支分别训练占用提升树、区域数网络、物体总数网络、219类空间计数网络。219种组合覆盖1–3件物体分布在9个区域的合法计数向量。验证集选择模型与解码权重，校准集决定温度和接受门限；模型固定后才给最终测试打分。每区计数由信号模型预测，未复制左图真值。

Each feature view trains occupancy, region-count, physical-count and219-state spatial-count models. Validation selects the decoder; separate calibration fits confidence. The candidate is frozen before final-test scoring.

## 新测试结果 / New final results

旧版与新版在**相同新测试信号**上比较。旧版的“区域数”没有能力代表同区布局的真实物体数，因此表中特别标明该基线。包含全部输入，不确定和质量拒绝也计入。

| 条件 / Condition | N | v4区域F1 | v3→v4物体总数准确率 | v4完整区域 | v4每区计数全部正确 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

“完整区域”只要求占用格区正确；“每区计数全部正确”还要求每格具体数量都对。数量恰好正确，位置或分配错误，仍算完整空间计数错误。通过置信门限的覆盖率为{pct(m['accepted_layout_coverage'])}，其中每区计数全部正确率为{pct(m['accepted_layout_accuracy'])}，不能用这个子集成绩代替上表。

Exact zones and exact spatial counts differ. A correct total can accompany wrong zones or count allocation. Accepted coverage={pct(m['accepted_layout_coverage'])}; exact spatial-count accuracy in that subset={pct(m['accepted_layout_accuracy'])}. These are subset metrics, not the all-input score.

推断 / Inference: median {r['inference_ms']['median']:.1f}ms，p95 {r['inference_ms']['p95']:.1f}ms；不包含信号采集、射线追踪或HTTP。模型文件 / bundle: {r['model_bytes']/1024**2:.1f}MiB。

## 使用 / Use

打开 http://127.0.0.1:8769/ 或双击 `START_MULTI.bat`。将A/B拖入同一个Z区域，自动避让保持实体不重叠，再点开始检测。右图显示预测区域和每区件数，下方分开显示物体总数及区域数。选择“同区测试：2件/3件”可持续处理独立测试布局。展开测试结果核对同区/不同区分组；置信度不足会保留虚线候选。

Drag A/B into one center zone, keeping bodies separated, and start. The right map displays predicted counts per zone; cards separate physical total and occupied-zone count. Shared-zone test sources replay held-out layouts; evaluation reports sharing strata separately.

## 限制 / Limits

仍限固定6×6米模拟房间、固定4TX/8RX、最多3件木质箱体，每个尺寸档位最多1件。区域按物体**中心**定义，不代表整件物体轮廓。模型不恢复同区内每件物体的精确坐标、真实长宽高或家具语义；真实家庭/设备泛化没有验证。旧尺寸辅助模型未覆盖同区布局，新版界面禁用该旧实验输出。预测和置信度仍可能出错。

This is a fixed-room simulated prototype with at most3 cuboids and one per size band. Labels refer to object centers. Exact within-zone individual coordinates, measured dimensions, furniture semantics and real-home generalization remain unvalidated. The legacy size head is disabled for the expanded task. Confident mistakes remain possible.

## 复现与回退 / Reproduction and rollback

在本目录依次运行 `shared_v4_data.py`、`shared_v4_cache.py`、`shared_v4_train.py`、`shared_v4_finalize.py`、`shared_v4_audit.py --files`。已有缓存/模型将复用；不要同时另跑评估器，也不要用已解封测试重新选参。候选以 `app.py --port 8772 --model-version 4` 启动，运行 `shared_v4_verify.py` 后才可 `shared_v4_release.py activate`。`shared_v4_release.py rollback` 恢复v3，原权重保留。

See the saved scripts/protocol for exact settings. Do not retune against an opened final test. The release helper verifies model/hash/data/API evidence; rollback restores the preserved v3.

证据 / Evidence: `outputs/shared_v4/results.json`, `frozen.json`, `data_audit_full.json`, `verification.json`, `ui_verification.json`。图表 / Figures: `figures/shared_v4/`。模型SHA256: `{sha(O4/'model.joblib')}`。
'''
    (ROOT/'SHARED_ZONE_RESULTS.md').write_text(report,encoding='utf-8')
    if active==4:
        p=ROOT/'README.md';s=p.read_text(encoding='utf-8');marker='<!-- SHARED_ZONE_PREVIOUS_DOC -->';history=s.split(marker,1)[-1] if marker in s else s
        p.write_text('# 当前同区多物体版 / Current shared-zone version\n\n当前启用 **v4**：http://127.0.0.1:8769/ 。双击 `START_MULTI.bat`。\n\n允许多件物体中心位于同一区域，分开预测总数、区域数与每区件数；拖动自动避让。\n\n[本次结果、使用与限制 / Results and use](SHARED_ZONE_RESULTS.md) · [安装 / Setup](SETUP.md)\n\n旧版v3成绩见晨间报告；不是本次新任务成绩。 / The v3 report below is historical.\n\n'+marker+'\n'+history,encoding='utf-8')
        p=ROOT.parent/'README.md';s=p.read_text(encoding='utf-8');history=s.split(marker,1)[-1] if marker in s else s;p.write_text('# RoomSense · 同区多物体版 v4\n\n双击 `OPEN_MULTIOBJECT.bat`，打开 http://127.0.0.1:8769/ 。\n\n[当前结果与操作 / Current results and use](multi_object/SHARED_ZONE_RESULTS.md)。下方旧版说明与成绩保留作历史记录。\n\n'+marker+'\n'+history,encoding='utf-8')
    print('Shared-zone bilingual report and3 figures generated.')
if __name__=='__main__':main()
