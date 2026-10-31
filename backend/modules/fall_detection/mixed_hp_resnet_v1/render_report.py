"""Render actual frozen-comparison artifacts into a portable Chinese HTML report."""
import base64,html,json
from pathlib import Path
M=Path(__file__).resolve().parent.parent
R=M/'reports/mixed_hp_resnet_v1'

def main():
 result=json.loads((R/'comparison/comparison.json').read_text())
 status=json.loads((M/'outputs/hp_mixed_resnet18_v1/status.json').read_text())
 audit=json.loads((R/'data_audit.json').read_text())
 predictions=json.loads((R/'comparison/validation_predictions.json').read_text())
 e=html.escape
 def image(p):return '<img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
 def table(head,rows):return '<div class="table"><table><thead><tr>'+''.join('<th>'+e(str(v))+'</th>' for v in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+e(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
 def section(t,title):
  rows=[]
  for name,label in [('original','原 ResNet18'),('mixed_HP','HP 混合训练 ResNet18')]:
   d=result['results'][str(t)][name];c=d['FallDetection'];h=d['HAR'];rows.append([label,f"{c['TP']}/{c['Fall']}",c['FN'],f"{c['FP']}/{c['Nonfall']}",f"{c['precision']:.2%}",f"{c['recall']:.2%}",f"{c['F1']:.2%}",f"{h['FP']}/{h['Nonfall']}"])
  return '<h2>'+title+'</h2>'+table(['模型','跌倒检出','漏检','原 Nonfall 误报','Fall Precision','Fall Recall','Fall F1','普通活动触发'],rows)
 primary=result['results']['0.5'];o=primary['original'];n=primary['mixed_HP']
 intro=f"同一阈值 0.5：普通活动触发 {o['HAR']['FP']}/250 → {n['HAR']['FP']}/250；原 Nonfall 误报 {o['FallDetection']['FP']}/536 → {n['FallDetection']['FP']}/536；跌倒检出 {o['FallDetection']['TP']}/390 → {n['FallDetection']['TP']}/390。"
 body='<div class="eyebrow">MODULE C · HP MIXED RESNET18 · SEED 42</div><h1>加入普通活动训练后，误报和跌倒检出怎样变化？</h1><p class="lead">'+intro+'</p>'
 body+='<p class="notice">这是验证比较：验证集参与早停选模型。结果不能当作独立测试、跨场景泛化或连续监控每小时误报率。现有生产模型与 B/C 门控规则保留。</p>'
 body+='<div class="cards">'+''.join('<div><b>'+e(str(v))+'</b><span>'+e(k)+'</span></div>' for k,v in [('训练记录',audit['train_n']),('固定验证记录',audit['val_n']),('最佳轮次',status['best_epoch']),('实际训练轮次',status['epochs'])])+'</div>'
 body+=section(.5,'共同固定阈值 0.5：主要比较')
 body+=section(0.574872225522995,'共同使用原阈值 0.574872：补充比较')
 activity=[]
 for label,old in o['HAR_by_activity'].items():activity.append([label,old['n'],old['FP'],n['HAR_by_activity'][label]['FP']])
 body+='<h2>五类普通活动分别触发多少次？</h2><p>每类取官方验证样本50条，原模型和新模型使用相同ID与原始内容。活动标签作为分段级弱 Nonfall，不提供逐帧跌倒缺席标注。</p>'+table(['活动','验证数','原模型触发','新模型触发'],activity)
 transitions=[]
 for src in ['FallDetection','HAR']:
  transitions.append([src,*[sum(r['source']==src and r['transition_05']==v for r in predictions) for v in ['fixed','regressed','both_wrong','both_correct']]])
 body+='<h2>哪些误判被修正，哪些样本反而退步？</h2>'+table(['来源','旧错新对','旧对新错','两者都错','两者都对'],transitions)
 by_label=[]
 for value,label in [(0,'Fall'),(1,'Nonfall')]:
  subset=[r for r in predictions if r['source']=='FallDetection' and r['label']==value]
  by_label.append([label,*[sum(r['transition_05']==v for r in subset) for v in ['fixed','regressed','both_wrong','both_correct']]])
 body+=table(['真实类别','旧错新对','旧对新错','两者都错','两者都对'],by_label)
 fall=by_label[0]
 body+='<p class="notice">跌倒检出总数相同，但具体漏检对象发生变化：新模型修正原模型漏检的 '+str(fall[1])+' 条跌倒，同时新增 '+str(fall[2])+' 条原模型能够检出的跌倒漏检；共同漏检 '+str(fall[3])+' 条。需要连同这些样本一起评估候选模型。</p>'
 regressions=[r for r in predictions if r['source']=='FallDetection' and r['transition_05']=='regressed']
 body+='<p>原跌倒验证集中，旧模型正确而新模型错误的记录如下（最多展示10条；完整逐样本结果保存在 CSV）。</p>'+table(['样本ID','真实标签','原 Fall score','新 Fall score'],[[r['sample_id'],r['original_label'],f"{r['original_fall_score']:.6f}",f"{r['mixed_fall_score']:.6f}"] for r in regressions[:10]])
 body+='<h2>训练曲线与评分分布</h2>'+image(M/'outputs/hp_mixed_resnet18_v1/training_curves.png')+image(R/'comparison/score_distributions.png')
 body+='<h2>训练与验证怎样隔离？</h2><p>从官方HP跌倒训练记录中去除6条训练内部重复，以及1条与保留验证内容相同的记录，剩4341条；另取官方HP活动训练集每类200条，共1000条。最终训练含Fall1727、Nonfall3614。验证保留Fall390、原Nonfall536，加活动250条。</p><p>训练/验证ID交叉为0，原始数值内容交叉为0，验证1176条内容均不同；先前25条活动诊断样本的相同内容也不进入训练。不随机重分官方集合，不访问测试样本。</p>'
 body+='<h2>哪些因素仍需区分？</h2><ul><li>原模型训练包含HP与ESP32；新模型只用HP，同时增加活动数据、去重并采用训练计数计算的类别权重。因此比较评价的是整体新训练方案，不是单一数据因素的消融。</li><li>两来源的环境不同；训练与验证也包含相同用户和环境。低活动触发率不能证明在活动来源场景里仍能检出真实跌倒。</li><li>两个模型分别推理，B门控不参与本次比较。阈值均提前固定，没有用本次结果调阈值。</li><li>归一化保持官方每完整分段计算均值/标准差的方式，无补零或尺寸拼凑。Fall score是分类评分，不宣称已校准的事故概率。</li></ul>'
 check=result['reload_check'];body+='<p>独立新进程重新加载检查：同一样本评分绝对误差 '+f"{check['absolute_error']:.3g}"+'，容差 '+str(check['tolerance'])+'。</p>'
 clean=result.get('common_subset_without_original_train_duplicates')
 if clean:
  rows=[]
  for name,d in clean['threshold_05'].items():rows.append([name,d['n'],d['TP'],d['FP'],d['FN'],d['TN'],f"{d['F1']:.2%}"])
  body+='<h2>排除原模型训练内容重复后的共同验证子集</h2><p>原模型历史训练中有1条与验证内容相同；新模型已剔除对应训练内容。额外排除此验证内容后，两模型仍在相同子集比较：</p>'+table(['模型','样本数','TP','FP','FN','TN','Fall F1'],rows)
 body+='<p class="footer">数据版本 CSI-Bench Kaggle v12 · 训练最多100轮，验证损失连续10轮不改善早停 · 最低验证损失保存最佳模型。完整证据：data_audit.json、comparison/comparison.json、comparison/validation_predictions.csv 与训练 history.json。</p>'
 css='body{font:16px/1.7 system-ui,sans-serif;background:#f3f6fb;color:#14283d;max-width:1180px;margin:40px auto;padding:32px}h1{font-size:36px;line-height:1.3;max-width:950px}h2{margin-top:42px;font-size:23px}.eyebrow{font-size:12px;letter-spacing:2px;color:#48738d}.lead{font-size:21px}.notice{padding:18px;background:#e6eef9;border-left:4px solid #40709c}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:28px 0}.cards div{background:white;padding:18px;border-radius:12px}.cards b{display:block;font-size:30px}.cards span{color:#5b6b79}.table{overflow:auto}table{width:100%;border-collapse:collapse;background:white;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #dce4ec;white-space:nowrap}th{background:#173e58;color:white}img{width:100%;margin:18px 0;background:white;border-radius:10px}.footer{color:#586c7c;font-size:13px;margin-top:40px}@media(max-width:700px){body{padding:15px;margin:15px auto}h1{font-size:27px}.cards{grid-template-columns:repeat(2,1fr)}}'
 (R/'report_zh.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HP 混合 ResNet18：实际验证比较</title><style>'+css+'</style>'+body+'</html>')
 print(intro)
if __name__=='__main__':main()
