# WiFi Sense：无摄像头、无穿戴的 WiFi 人体活动与跌倒感知

## B + C 集成演示（2026-10-30）

B/C训练权重和配套模型数据见 [模型发布与恢复说明](model_artifacts/README.md)。运行 `python3 scripts/restore_bc_models.py` 恢复当前模型，添加 `--all` 可恢复所有35个历史/当前权重文件。

代码交付、模型文件位置及独立HP混合ResNet18候选的结果见 [B/C代码交付说明](docs/BC_RESNET18_CODE_DELIVERY.md)。候选模型未替换当前运行模型。

最新默认规则：B置信度<0.6才运行C；高置信度时C未评估，不表示未跌倒。验证回放会跳过51条真实Fall，具体见[门控说明](docs/B_LOW_CONFIDENCE_GATE.md)。下段保留此前并行接口背景。

Module B已接入作者新发布的频谱ExtraTrees seed42参数；Module C使用CSI-Bench官方ResNet18，显示926条HP验证分数，阈值0.574872。B实际读取同一原始分段并按内容身份配对，只作为活动参考，不修改C分数。B无Fall类，当前分段无五类动作真值，未评价联合准确率。此前TCS D回放已单独归档。运行 `python3 scripts/serve_demo.py` 后打开 http://localhost:5173/#/module/fall_detection 。

最新接口、启动与缺项见 [ResNet18 + B交接说明](docs/RESNET_B_HANDOFF.md)；[旧版TCS交接](docs/BC_DEMO_HANDOFF.md)保留作历史记录。B/C使用同一原始HP输入；其余模块数据未同步，五模块融合页只展示预期规则效果。


基于 WiFi CSI 的室内感知研究原型，面向单人房间的非临床安全监测（模式识别系统实践模块课程项目）。
五个感知模块分别处理 CSI 数据，再由一层透明的规则式融合把各模块输出合成房间状态和安全提示。
网站展示每个模块的工作过程，以及一张"集成后预期效果"的平面图。

> 各模块使用各自的数据子集独立训练和评价，集成页只演示预期效果，**不是联合真值**。

## 快速开始（本地启动）

**环境要求**：Node.js 20+、npm 10+、Python 3.10+（`node -v`、`npm -v`、`python3 --version`；Windows 用 `python`）。后端只用 Python 标准库，不需要 `pip install`。

```bash
git clone <仓库地址>
cd WI-FI
npm run install:frontend      # 安装前端依赖，只需一次
npm run dev                   # 同时启动后端和网站，Ctrl+C 一起关闭
```

| 服务 | 地址 | 说明 |
|---|---|---|
| 网站 | http://localhost:5173 | 浏览器打开这个地址 |
| 后端接口 | http://localhost:8000/api/health | 返回 `{"ok": true}` 说明后端正常 |

**确认成功**：首页应显示深色页面和房间平面图，图上的人会自动循环走动。进入任一模块页，顶部"OUTPUT PREVIEW"卡片右侧会显示 `source: backend` 或 `source: bundled`；`bundled` 表示该模块还没有导出结果，页面用的是自带示例数据，属正常。

**其他启动方式**

```bash
# 只启动网站（用自带示例数据，不需要 Python）
cd frontend && npm install && npm run dev

# 只启动后端（在仓库根目录）
python3 -m backend.api                          # 所有模块，端口 8000
python3 -m backend.api --port 9000              # 换端口
python3 -m backend.api --only fall_detection    # 开发时只启动自己的模块
```

分开启动时，让网站读后端需要创建 `frontend/.env.local`，写入 `VITE_BACKEND_URL=http://localhost:8000`。

**配置数据位置**：CSI-Bench 数据集很大，不进 git，用环境变量告诉后端它在哪里。

```bash
export CSI_DATA_ROOT="/path/to/MotionSourceRecognition"      # macOS / Linux
$env:CSI_DATA_ROOT = "D:\data\MotionSourceRecognition"       # Windows PowerShell
```

**提交前检查**：`npm run check`（后端测试 + 前端构建）。

**常见问题**

| 现象 | 处理 |
|---|---|
| `npm run dev` 报找不到 python | 指定解释器：`PYTHON=python3 npm run dev`（Windows：`$env:PYTHON="python"; npm run dev`） |
| 端口 5173 或 8000 被占用 | 关掉占用的程序，或分开启动并换端口 |
| 页面一直显示 `source: bundled` | 该模块后端还没有输出，属正常；导出结果后会变成 `backend` |
| 浏览器控制台出现 404 | 网站先问后端，后端没有输出时回退到自带数据，不影响页面 |
| 页面缺模块或空白 | 前后端文件夹名必须一致，运行 `npm test` 会指出缺了什么 |

## 架构

```text
 CSI-Bench 数据（CSI_DATA_ROOT，不进 git）
        |
        v
 backend/modules/<模块>/   preprocessing -> training -> experiments -> evaluation
        |                                                        |  write_replay()
        |                                                        v
        |                                              outputs/replay.json
        v                                                        |
 backend/api（routes.py 路由表）  <------------------------------+    GET /api/<模块>/replay
        |
        v                       后端没有输出时回退
 frontend/src/api/replay.js  <----  frontend/src/modules/<模块>/data/replay.json
        |
        v
 网站页面：首页平面图  |  各模块页  |  融合与改进页
```

| 部分 | 目录 | 职责 |
|---|---|---|
| 后端（Python） | `backend/` | 数据处理、训练、实验、评估，以及把各模块回放结果提供给网站的小接口 |
| 前端（React + Vite） | `frontend/` | 网站：带平面图的首页、每个模块一页、融合与改进页 |
| 文档 | `docs/` | 回放数据格式、接口说明、按角色的团队交接 |

前后端互相独立：前端用自带示例数据就能单独运行；后端单独运行也能提供接口。

## 目录结构

```text
README.md  package.json  .env.example  .gitignore
scripts/dev.mjs            同时启动后端和前端
docs/                      data-contract.md  API_INTEGRATION.md  TEAM_HANDOFF.md
backend/
  api/                     routes.py（路由表）、server.py、__main__.py
  shared/                  contract.py（校验）、export.py（write_replay）、paths.py
  tests/                   契约、路由、模块完整性测试
  modules/<模块>/          每个模块一个文件夹，见下
frontend/src/
  main.jsx                 入口
  app/                     router.jsx（全部网站路由）、Layout.jsx（导航）
  api/                     replay.js（唯一取数据的地方）
  components/              图表、模块页模板、通用小组件
  features/landing/        首页和平面图
  features/fusion/         融合页、rules.js（融合规则）、content.js（文案）
  modules/<模块>/          每个模块一个文件夹，registry.js 自动发现
  styles/global.css        主题
```

## 模块与开发位置

五个模块的流程相同：数据预处理 → 模型训练 → 实验与对比 → 评估。
**每人在自己的分支上，只改自己模块的两个文件夹**，合并时不会冲突。文件夹名就是模块名（`snake_case`），前后端一致。

| 模块 | 输出 | 负责人 |
|---|---|---|
| `motion_source_recognition` 运动来源识别 | Human、Pet、Robot、Fan 的概率，置信度，Unknown | |
| `human_activity_recognition` 人体活动识别 | 活动类别、置信度、稳定性 | |
| `fall_detection` 跌倒检测 | 跌倒评分，状态：normal、suspected、alert | |
| `breathing_detection` 呼吸检测 | 呼吸证据，或"当前不适合判断" | |
| `space_proximity_sensing` 空间与接近程度感知 | 区域，近、中、远 | |

每个模块在两处各有一个文件夹：

```text
backend/modules/<模块>/
  preprocessing/ training/ experiments/ evaluation/   四个阶段的代码
  outputs/         生成的文件（replay.json、模型、图）；大文件不进 git
  api.py           GET /api/<模块>/replay 返回什么（默认读 outputs/replay.json）
  contract.py      本模块每一帧要包含哪些字段，由模块自己维护
  requirements.txt 可选，只有本模块需要的 Python 包
  README.md        本模块的字段说明

frontend/src/modules/<模块>/
  meta.js          名称（title 全名、navLabel 菜单短名）、简介、负责人、导航顺序、summarize（一帧显示成什么文字）
  plan.js          四个阶段的计划清单（planned、doing、done）
  data/replay.json 随网站自带的回放和结果表，没有后端时页面也能用
  index.jsx        模块页面；用 stages={{ evaluation: <MyChart /> }} 往任意阶段加自己的内容
```

额外的组件、图表、图片放在自己的文件夹里。模块由程序自动发现，新增或修改模块**不需要改任何共享文件**。
`space_proximity_sensing/meta.js` 还保存首页平面图用的区域名（`zones`）。

**降低集成成本**：
- 字段由各模块自己的 `contract.py` 声明，共享代码不记录任何模块字段。
- `write_replay("<模块>", payload, bundle=True)` 一次调用同时写好后端输出和网站自带的那一份，并按本模块契约校验。
- `npm test` 检查前后端文件夹是否对齐、两边是否完整、自带数据是否符合契约；通过就说明模块可以接入。
- 唯一读取其他模块字段的是融合规则（`frontend/src/features/fusion/rules.js`），改动字段时通知融合负责人。

## 路由

| 层 | 位置 | 说明 |
|---|---|---|
| 网站路由 | `frontend/src/app/router.jsx` | `HashRouter`，地址形如 `/#/module/fall_detection`。`src/modules/` 里每个文件夹自动得到 `/module/<文件夹名>` |
| 导航菜单 | `frontend/src/app/Layout.jsx` | 按各模块 `meta.js` 的 `order` 排列，最后是融合页 |
| 后端路由 | `backend/api/routes.py` | `ROUTES` 路由表：`/api/health`、`/api/modules`、`/api/<模块>/replay` |
| 数据读取 | `frontend/src/api/replay.js` | 设置了 `VITE_BACKEND_URL` 时先读后端，失败回退到自带文件 |

## 共享内容（需要团队一起商量着改）

`docs/`，`backend/api/`、`backend/shared/`、`backend/tests/`，`frontend/src/` 下除 `modules/<模块>/` 以外的全部内容（`app/`、`api/`、`components/`、`features/`、`styles/`、`modules/registry.js`）。

## 把模块结果发布到网站

```python
from backend.shared.export import write_replay

write_replay("fall_detection", {"version": "0.1", "frames": [...], "metrics": {...}}, bundle=True)
```

数据格式见 [docs/data-contract.md](docs/data-contract.md)。

## 约定

- 回放帧用统一的时间轴 `t`（秒）；各模块的帧不需要在同一个 `t` 上。
- `Unknown`、`not_assessable` 是决策状态，不是训练类别。
- 某个模块没有数据时页面显示 "no data"，其他模块照常工作。
- 示例数据必须带 `"mock": true`，页面会显示"占位数据"提示。
- 不要提交 `.env.local`、`node_modules`、`dist`、CSI-Bench 数据和大模型文件。
- 数据来自 CSI-Bench（Zhu 等，NeurIPS 2025），分享衍生数据前请核对数据集许可证。

## 更多文档

- [docs/TEAM_HANDOFF.md](docs/TEAM_HANDOFF.md)：按角色说明各自改什么
- [docs/API_INTEGRATION.md](docs/API_INTEGRATION.md)：请求流程、路由、配置
- [docs/data-contract.md](docs/data-contract.md)：回放数据格式
