# 当前规则：B低置信度才调用C

用户指定的路由已启用，B、C权重和C分类阈值均未训练或修改：

1. 读取同一HP原始幅度，先运行B。
2. B置信度严格小于0.6，且输入身份配对/质量有效时，加载ResNet18 C并判断。
3. B置信度大于或等于0.6：C不加载、不前向；输出`C_ran=false`、`state=not_evaluated`，`fall_score/prediction/is_fall=null`。这不表示C确认无跌倒。
4. B缺失、无效或无法配对：同样没有C判断，原因明确标为B不可用。
5. C被调用时仍采用`0.574872225522995`，`score>=threshold`为Fall。

单文件命令：

```bash
python -m backend.modules.fall_detection.resnet_b_integration.infer --file /path/to/HP_segment.h5
```

`--no-b`是显式的C独立诊断旁路，不属于以上门控流程。`--b-output`可提供实际同输入B输出。

## 验证回放结果及风险

现有926条HP验证分数用于规则回放模拟：621条选择C，305条跳过C；390条真实Fall中51条被跳过。这51条原本C均检出，因此回放中的跌倒检出从384条减少到333条。C原误报23条中，该门控只排除了2条，剩21条。低置信度子集的TP/FP/FN/TN为333/21/6/261；这个条件子集矩阵不含51条被门控跳过的真实Fall，不能当作整条系统表现。

这是用户指定规则在已使用验证数据上的事后模拟，不是新的独立测试或经验证的安全改进。B置信度不是分布外检测器，也不是Fall评分；高置信度普通动作仍可能对应实际跌倒。页面突出展示真实Fall被跳过数量，不将未执行C标作Nonfall。

原有并行回放保存在`backend/modules/fall_detection/outputs/before_confidence_gate_v1.json`；旧权重和研究报告保留。网页是已保存验证预测应用门控后的回放，不是切换样本时实时执行C。真实CLI在B低置信度时才加载/前向C。

规则文件：`backend/modules/fall_detection/confidence_gate_v1/policy.json`。
重导出：`python -m backend.modules.fall_detection.confidence_gate_v1.export_replay`。
结果与检查：`backend/modules/fall_detection/reports/confidence_gate_v1/`。
