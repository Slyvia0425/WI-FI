# WiFi Sense：Wi-Fi 室内感知项目

五个模块共用一个 React 网站：运动来源、人体活动、跌倒、呼吸，以及静态物体空间感知。前端沿用 Fang Meiheng 的深色青蓝设计，首页导航和模块卡片可以进入各自工作区。

A–D 展示各自的录制数据预测与评估结果；E 使用仿真信号执行区域计数推理。首页房间动画和融合规则页是明确标注的示意，各模块不是同一次同步录制，不报告五模块联合准确率。

## 本地启动

需要 Node.js 20+、npm 10+、Python 3.10+。在仓库根目录执行：

```bash
npm ci --prefix frontend
npm run build
python scripts/serve_integrated.py
```

打开 **http://127.0.0.1:5173/**。Windows 构建后也可双击 `START_TEAM.bat`。

### 空间模块的模型与数据

首次使用 E 模块，在 `backend/modules/space_proximity_sensing/wifi space` 中运行 `install.bat`（Windows，Python 3.12）。它会下载并校验已有 Release 的全部分卷，恢复模型和仿真数据，并创建 Python 环境。随后启动统一网站，空间服务会自动运行。

其他系统可在该目录运行 `python3 fetch_data.py`，创建 `.venv` 并安装 `multi_object/requirements-night.txt`。空间模型缺失时，A–D 的保存结果页面仍可浏览，E 页面会显示服务离线并提供重连按钮。

已有完整空间项目可通过 `WIFI_SPACE_ROOT` 指定；解释器可通过 `WIFI_SPATIAL_PYTHON` 指定。网站默认端口为 5173，内部空间服务默认端口为 8780；使用 `--port` 和 `--spatial-port` 修改。

### 其他模块的数据和模型

- B/C 模型恢复：`python scripts/restore_bc_models.py`；完整文件清单在 `model_artifacts/manifest.json`。
- A–D 的原始数据下载说明、脚本和少量原始 H5 示例在 `datasets/`。完整原始 CSI-Bench 数据未提交到 Git；D 的 WiRe-Breath 数据需另行取得。
- 浏览已保存的回放不需要先下载全部原始训练数据。E 页面执行仿真和推理，需要它自己的完整运行环境与模型。

### 开发与检查

```bash
npm ci --prefix frontend
npm run dev
python -m unittest discover -s backend/tests -t .
npm run build
```

开发模式：网站 5173，回放 API 8000，空间服务 8780。Vite 代理 `/api/` 与 `/spatial/`，统一入口会启动空间服务。部署说明见 [整合说明](docs/INTEGRATION.md)。

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
| `space_proximity_sensing` 静态物体空间感知 | 仿真信号、占用区域、区域内物体数 | |

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
