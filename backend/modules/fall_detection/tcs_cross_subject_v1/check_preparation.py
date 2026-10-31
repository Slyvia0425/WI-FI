"""Verify actual prepared features and save the pre-training comparison protocol."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

HERE=Path(__file__).resolve().parent
MODULE=HERE.parent
sys.path.insert(0,str(HERE))
from dataset import GCVDataset
sys.path.insert(0,str(MODULE/'tcs_fall_audit_v1'))
from audit_local_archives import sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features',type=Path,required=True)
    ap.add_argument('--eligibility',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    if args.report.exists():raise FileExistsError('Use a fresh verification/report directory')
    summary=json.loads((args.features/'preparation.json').read_text())
    rows=json.loads((args.features/'samples.json').read_text())
    if sha(args.features/'features.f32')!=summary['feature_cache_sha256']:raise ValueError('Feature cache changed')
    if sha(args.features/'samples.json')!=summary['sample_manifest_sha256']:raise ValueError('Sample manifest changed')
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Duplicate sample IDs')
    if {r['subject'] for r in rows if r['role']=='train'}!={1,2}:raise ValueError('Training subject assignment')
    if {r['subject'] for r in rows if r['role']=='validation'}!={3}:raise ValueError('Validation subject assignment')
    if any(r['role']=='test' or r['subject']==4 for r in rows):raise ValueError('Test features included')
    if any(r['label']!=0 or r['end_seconds']>120 for r in rows if r['kind']=='continuous_background'):
        raise ValueError('Continuous negative violates complete-window background rule')
    cache=np.memmap(args.features/'features.f32',dtype=np.float32,mode='r',shape=tuple(summary['shape']))
    seen={};cross_role_features=[]
    for i,r in enumerate(rows):
        digest=hashlib.sha256(cache[i].tobytes()).hexdigest()
        if digest in seen and seen[digest]['role']!=r['role']:
            cross_role_features.append([seen[digest]['sample_id'],r['sample_id']])
        seen[digest]=r
    if cross_role_features:raise ValueError('Identical features across training/validation require source review')
    old_features=np.load(MODULE/'outputs/tcs_subject01_gcv_cnn_seed42_v1/features.npz',allow_pickle=False)
    old_ids=old_features['ids'].tolist();old_x=old_features['x']
    by_source={r['source_member']:i for i,r in enumerate(rows) if r['kind']=='single_action'}
    old_feature_difference=max(float(np.max(np.abs(cache[by_source[name]]-old_x[i]))) for i,name in enumerate(old_ids))
    if old_feature_difference>1e-7:raise ValueError('Existing subject01 raw features differ from prior preparation')
    ranges=[]
    for name in ('normalization_sa.json','normalization_sa_ct.json'):
        norm=json.loads((args.features/name).read_text())
        indices=[i for i,r in enumerate(rows) if r['role']=='train' and (norm['includes_continuous_negatives'] or r['kind']=='single_action')]
        if norm['fit_sample_count']!=len(indices) or norm['fit_subjects']!=[1,2]:raise ValueError('Normalization fitting scope')
        mean=np.asarray(norm['mean'],dtype=np.float32);scale=np.asarray(norm['scale'],dtype=np.float32)
        low=float('inf');high=float('-inf')
        for start in range(0,len(rows),16):
            x=(np.asarray(cache[start:start+16])-mean)/scale
            if not np.isfinite(x).all():raise ValueError('Nonfinite train/validation normalized output')
            low=min(low,float(x.min()));high=max(high,float(x.max()))
        ranges.append({'normalization':name,'all_train_validation_outputs_finite':True,'range':[low,high]})
    ds=GCVDataset(args.features,'train',True,args.features/'normalization_sa_ct.json')
    chosen=[]
    for kind,label,n in (('single_action',1,4),('single_action',0,2),('continuous_background',0,2)):
        chosen.extend([j for j,i in enumerate(ds.indices) if rows[i]['kind']==kind and rows[i]['label']==label][:n])
    bx,by,ids=next(iter(DataLoader(Subset(ds,chosen),batch_size=8,shuffle=False)))
    old=json.loads((MODULE/'final_selection_v1/preserved_artifacts_before.json').read_text())
    changed=[rel for rel,digest in old.items() if sha(MODULE/rel)!=digest]
    if changed:raise ValueError('Protected files changed: '+str(changed))
    old_tcs_sha=sha(MODULE/'outputs/tcs_subject01_gcv_cnn_seed42_v1/run/best.pt')
    if old_tcs_sha!='4d5d40831f37d3a226f7552811487c2e74b6333cfce46a6478953cf1d931fe39':raise ValueError('Original TCS checkpoint changed')
    counts=Counter((r['role'],r['kind'],r['label']) for r in rows)
    ntrain_sa=sum(n for (role,kind,label),n in counts.items() if role=='train' and kind=='single_action')
    protocol={'status':'prepared before new model training or validation predictions','source_protocol_sha256':sha(HERE/'protocol.json'),
              'feature_cache_sha256':summary['feature_cache_sha256'],'subject_roles':{'train':[1,2],'validation':[3],'test':[4]},
              'model':'unchanged official five-convolution CNN; each arm starts from scratch with same seed',
              'arms':json.loads((HERE/'protocol.json').read_text())['planned_comparison_arms'],
              'seed':42,'max_epochs':100,'early_stop_patience':10,'optimizer':'Adam','learning_rate':0.0001,
              'batch_size':8,'steps_per_epoch':math.ceil(ntrain_sa/8),'threshold':0.5,
              'sampling':'Each batch4 Fall SA +4 Nonfall. SA-only arms:4 SA negatives. CT arms:2 SA negatives +2 CT background negatives; draw CT recording uniformly then window uniformly. Use replacement and a seeded generator. Same step budget in all arms.',
              'criterion':'unweighted CrossEntropyLoss on identically class-balanced batches',
              'validation_loss':'0.5*meanCE(Fall SA)+0.25*meanCE(Nonfall SA)+0.25*meanCE(CT background), identical validation membership for all arms',
              'checkpoint_selection':'lowest common validation loss; strict improvement resets patience',
              'normalization_fit':'only the corresponding arm training pool; equal feature-timepoint weighting; no validation/test statistics',
              'test_policy':'no model evaluation until comparison and all final choices frozen',
              'scope':'new four-subject controlled experiment; differs from paper CV and earlier one-subject pilot'}
    args.report.mkdir(parents=True)
    (args.report/'training_protocol.json').write_text(json.dumps(protocol,indent=2))
    check={'cache_shape':summary['shape'],'sample_shape':list(bx[0].shape),'batch_shape':list(bx.shape),
           'labels':by.tolist(),'label_dtype':str(by.dtype),'batch_range':[float(bx.min()),float(bx.max())],
           'sample_ids':list(ids),'finite_normalization_checks':ranges,'train_only_normalization_scope_verified':True,
           'no_test_features':True,'all_background_windows_end_by120s':True,'protected_original_files_unchanged':True,
           'previous_subject01_features_compared':len(old_ids),'previous_features_max_abs_difference':old_feature_difference,
           'cross_role_exact_feature_duplicate_pairs':cross_role_features,
           'old_tcs_checkpoint_sha256':old_tcs_sha,'training_run':False,'test_model_evaluated':False,
           'free_disk_gib':shutil.disk_usage(args.features).free/1024**3}
    check['recommended_free_disk_gib_before_training']=5.0
    check['training_resource_ready']=check['free_disk_gib']>=5.0
    (args.report/'completion_check.json').write_text(json.dumps(check,indent=2))
    eligibility=json.loads((args.eligibility/'eligibility.json').read_text())
    table='\n'.join(f"| {r['subject']:02d} | {r['role']} | {r['single_action']} | {r['fall']} | {r['nonfall']} | {r['continuous']} |" for r in eligibility['subject_summary'])
    feature_table='\n'.join(f'| {role} | {kind} | {label} | {n} |' for (role,kind,label),n in sorted(counts.items()))
    text=f'''# 四人 TCS-Fall 数据审计与训练输入准备（2026-10-29）

已收到1.zip至4.zip。新包2/3/4逐文件CRC通过；原始CSI审计完成，数据按人分为训练01/02、验证03、测试04。当前完成数据准备，尚未运行本轮模型训练或测试评价。

## 实际记录

| 人员 | 用途 | 可用单动作 | Fall | Nonfall | 连续录制 |
|---|---|---:|---:|---:|---:|
{table}

不是每人都有论文表格描述的210条单动作：02包实际150条，03/04各140条。没有填造缺失样本，也没有据此重新分配受试者。原始包共704条DAT，其中684条满足本轮数据用途；其余为3条既知ZIP损坏、1条短记录和16条未知活动16/17。

02/a_02_03_05.dat只有6973包、约6.976秒，原始数据可读，但不满足固定8000包输入。它作为“输入规格不适用”列出，未补零或拉伸；排除的是一条训练Nonfall。

03/a_03_05_04.dat包含8980个完整CSI帧，最后只有下一帧的2字节长度头。逐帧核对及完整前缀数值/时间审计通过，前8000包不受影响。原文件不修改；读取适配明确记录原始SHA、完整前缀SHA与末尾2字节边界情况，保留该验证Nonfall。详见[完整入选清单]({(args.eligibility/'record_manifest.csv').resolve()})和[处理依据]({(args.eligibility/'eligibility.json').resolve()})。

可解码CSI均为[Tx3,Rx3,报告位置30]，取Tx0后90特征；检查数值有限。规范化原始I/Q内容比较未发现完全重复，包括跨训练/验证/测试重复。它不排除近似重复或其他采集关联。48条连续录制均可读；标注时间零点和onset/impact含义仍未同步验证。

## 新增连续负样本

每个输入8000包，每次前移1000包。连续窗口使用真实包时间，要求整个窗口结束于前120秒内，并交叉检查标注跌倒在120秒之后。训练只使用01/02；03窗口用于共同验证；04没有生成模型特征。

| 用途 | 样本类型 | 标签（Fall=1） | 数量 |
|---|---|---:|---:|
{feature_table}

这些重叠窗口高度相关，不能把它们当作新增独立用户。连续负样本覆盖站立、站立伴日常活动、走路、走路伴日常活动、坐姿及坐姿伴日常活动，其中走路类别是之前误报集中处。

特征缓存形状为{summary['shape']}，占{summary['cache_bytes']/1024**2:.1f}MiB。原GCV范围为{summary['feature_range']}，全部有限。用同一实际连续窗口对照原单动作变换，最大绝对差为{summary['continuous_transform_reference_check']['max_abs_feature_difference']}。

标准化是明确声明的实验分支：对每个90维特征，只用相应训练池的全部GCV时间点拟合均值/标准差，验证和后续推理复用。SA-only和SA+CT分别保存参数。没有使用04数据，没有对现有模型临时添加标准化，也没有把此选择称为缺失官方预处理的精确复原。

实际混合batch形状{list(bx.shape)}，标签{by.tolist()}，包含4个跌倒单动作、2个非跌倒单动作、2个连续背景窗口。标准化batch范围[{float(bx.min()):.4f},{float(bx.max()):.4f}]；两套标准化作用于全部已准备训练/验证输入后都无NaN/Inf。

![训练Fall与正常走路的原始幅度检查]({(args.features/'input_examples.png').resolve()})

曲线用于核对读取和时间轴，不用两张曲线预测模型性能。

## 下一轮固定比较

四组均使用相同CNN、种子42、阈值0.5、最多100轮、验证损失10轮不改善早停。比较SA与SA+连续负样本，各自再比较是否使用训练拟合标准化。添加连续负样本会显著改变数量比例，因此四组统一每batch4个Fall/4个Nonfall、每轮{protocol['steps_per_epoch']}步；加入CT的组将4个负样本分为2个SA和2个CT。验证采用相同样本和固定的分组加权损失，避免长连续录制的窗口数量掩盖跌倒性能。完整设置已在任何新模型预测之前保存到[训练协议]({(args.report/'training_protocol.json').resolve()})。

当前结果证明新数据可按明确标签、受试者和窗口规则生成模型输入；还没有证明加入背景或标准化会改善误报。这需要下一阶段四组训练与验证。原HP及TCS模型保留。

训练/验证4528个GCV窗口还进行了独立特征内容哈希检查，未发现跨集合完全相同特征。受试者01的207条旧SA特征逐条比对，最大绝对差为{old_feature_difference}。

## 本机训练资源

检查时磁盘剩余约{check['free_disk_gib']:.2f}GiB，本轮特征缓存约622MiB。建议先留出至少5GiB再启动本机训练；低于此余量时本阶段仅完成数据准备，不启动四组训练。可将完整原始ZIP移至外置盘，保留校验信息和本机特征缓存；不要直接删除唯一原始文件。新训练尚未运行，已有模型和测试结果不变。
'''
    (args.report/'report_zh.md').write_text(text)
    print(json.dumps(check,indent=2),flush=True)


if __name__=='__main__':main()
