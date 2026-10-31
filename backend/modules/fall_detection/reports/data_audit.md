# FallDetection 官方数据审计

版本：kaggle-guozhenjennzhu-csi-bench-v12；样本数：6700。

官方数字 Fall=0、Nonfall=1；内部二分类 Fall=1、Non-fall=0，保留原始字符串。

## 官方集合

|集合|样本数|Fall|Nonfall|Fall 比例|
|---|---:|---:|---:|---:|
|test_easy|915|362|553|39.56%|
|test_hard|73|38|35|52.05%|
|test_id|1005|409|596|40.70%|
|test_medium|17|9|8|52.94%|
|train_id|4690|1922|2768|40.98%|
|val_id|1005|439|566|43.68%|

split 文件实际为字符串 sample ID 的 JSON 列表。manifest 的 split 使用分号保存所有成员关系，不把重叠测试集算成互斥划分。

## 设备与标签

|设备|原始标签|数量|
|---|---|---:|
|ESP32|Fall|277|
|ESP32|Nonfall|217|
|HP|Fall|2493|
|HP|Nonfall|3713|

## 全量输入形状

```json
{
  "[232, 500, 1]": 6206,
  "[64, 500, 1]": 494
}
```

dtype：{'float32': 6700}。

## 集合间 ID 交集

|集合 A|集合 B|交集|
|---|---|---:|
|test_easy|test_hard|0|
|test_easy|test_id|915|
|test_easy|test_medium|0|
|test_easy|train_id|0|
|test_easy|val_id|0|
|test_hard|test_id|73|
|test_hard|test_medium|0|
|test_hard|train_id|0|
|test_hard|val_id|0|
|test_id|test_medium|17|
|test_id|train_id|0|
|test_id|val_id|0|
|test_medium|train_id|0|
|test_medium|val_id|0|
|train_id|val_id|0|

## 集合间内容交集

|集合 A|集合 B|共同内容哈希数|
|---|---|---:|
|test_easy|test_hard|0|
|test_easy|test_id|915|
|test_easy|test_medium|0|
|test_easy|train_id|1|
|test_easy|val_id|0|
|test_hard|test_id|73|
|test_hard|test_medium|0|
|test_hard|train_id|0|
|test_hard|val_id|0|
|test_id|test_medium|17|
|test_id|train_id|2|
|test_id|val_id|0|
|test_medium|train_id|1|
|test_medium|val_id|0|
|train_id|val_id|1|

## 数据质量与来源

异常项：0；重复内容组：9；候选来源字段组合跨主集合组：15。

每项异常与每个重复组的样本、标签和来源见 audit_details.json；未静默跳过文件，未修改官方划分。

内容相同只证明数组一致，须进一步核查录制来源，不能直接把所有重复组认定为泄漏。

同源检查使用 user/environment/device/session 组合。缺少经过确认的连续录制父 ID；不能证明相邻片段已按父录制隔离。

哈希使用统一 float64（复数使用 complex128）、小端、C 顺序及 shape，并统一符号零和 NaN；不用 HDF5 文件字节哈希判定内容重复。

## 边界

本审计不训练模型、不选择阈值、不报告性能。轴语义证据与审计解释见 inspection_notes.md。

## 当前判断

文件可用性检查完成；不能据此宣称官方划分没有泄漏风险。跨集合重复内容计数见上表；本次人工重新读取与来源核查见 duplicate_review.md。

来源字段组合碰撞可能来自活动内 session 编号重复使用，不能直接作为同源录制的定论。

后续官方实验保留现有划分并披露风险；如另行进行按内容/经核实父录制分组的纠正实验，必须另存划分和结果，不能称为官方基准。当前没有生成纠正划分。

各测试子集设备组成见 audit_details.json；难度与设备可能混杂，解读难度分层结果时必须一并检查。
