# EXAM 提交规范（学生须知）

> 本指南面向「**26 muscle 3D 行走 RL 训练实训题目**」的提交与评分，请在开始训练前通读一遍，并严格按规范打包提交。

---

## 0. 一页速览（TL;DR）

| 你需要做的事 | 一行说明 |
|---|---|
| **训练你的策略** | `bash EXAM/train.sh` 或 `EXAM\train.bat`，在 `EXAM/results/train_session_<时间戳>/` 下产出权重 |
| **评估你的最佳模型** | `bash EXAM/eval.sh results/train_session_<时间戳> --best-eval` |
| **打包提交** | 按 §1 的目录结构压成 `submission_<学号>_<姓名>.zip`，控制在 **≤45 MB** |
| **唯一可改文件** | `configs/leg26_baseline.json`（详见 §3） |
| **不可修改的内容** | 其它所有文件、目录、脚本，详见 §3.2 禁改清单 |
| **提交前自检** | 跑 §5 的命令（必须 PASS） |
| **教师如何复现** | 详见 §6，与你 §5 自检完全一致 |

---

## 1. 提交目录结构

把以下结构压成一个 zip：

```
submission_<学号>_<姓名>/
├── README.md                                # 1 句话速览（训练步数 + 复现命令）
├── session_config.json                      # 必交：训练时实际生效的完整配置
├── train_log.json                           # 必交：完整训练曲线
├── best_checkpoint.json                     # 必交：最优 checkpoint 历史
├── best_model.zip                           # 必交：最佳模型权重
├── eval_results/                            # 必交：最佳 ckpt 的 analyze_results_<best_step>_00/ 整个目录
│   ├── gait_evaluated_data.json
│   ├── replay.mp4
│   ├── return.png
│   ├── kinematics_data_right_based.png
│   ├── joint_angle_cmap_by_velocity.png
│   ├── segmented_joint_data.png
│   ├── segmented_joint_data_avg.png
│   ├── left_right_comparison_avg.png
│   ├── sim_ref_joints_comparison_with_shade.png
│   └── sim_ref_joints_right_metrics.csv
└── report.pdf                               # 必交：3–5 页报告
```

**总大小目标 ≤100 MB**。所有学生格式完全一致——15 项必交文件，无 `code/` 目录。

---

## 2. 必交文件清单

### 2.1 训练产物（顶层 4 个文件）

来源：`EXAM/results/train_session_<时间戳>/`。

| 文件 | 大小 | 用途 |
|------|------|------|
| `session_config.json` | ~10 KB | 训练时实际生效的完整配置（含奖励权重、PPO 超参、网络结构、地形） |
| `train_log.json` | 2–10 MB | 每段 rollout 的 PPO 指标 + 奖励分量演化 |
| `best_checkpoint.json` | ~5 KB | 自动选优历史 |
| `best_model.zip` | ~1 MB | **最佳模型权重 —— 教师据此复现评估** |

> `best_model.zip` 来源：`EXAM/results/train_session_<时间戳>/trained_models/best_model.zip`。
> **不要**提交 `trained_models/model_<其它 step>.zip`（一次训练能产生 100+ 个、数百 MB）。

### 2.2 `eval_results/` —— 评估产物（10 个文件全部必交）

把最佳 checkpoint 的 `analyze_results_<best_step>_00/` 目录**整个内容**复制进来：

| 文件 | 评分用途 |
|------|----------|
| `gait_evaluated_data.json` | 教师可基于此重出图、重算指标，无需重跑 rollout |
| `replay.mp4` | 步态质量直观证据 |
| `return.png` | 训练收敛曲线 |
| `kinematics_data_right_based.png` | 整段关节角 + 骨盆 + 足底力 |
| `joint_angle_cmap_by_velocity.png` | 速度着色关节角 |
| `segmented_joint_data.png` | 步态周期分段 |
| `segmented_joint_data_avg.png` | 步态周期 mean ± SD |
| `left_right_comparison_avg.png` | 左右对称性 |
| `sim_ref_joints_comparison_with_shade.png` | 仿真 vs 参考步态 |
| `sim_ref_joints_right_metrics.csv` | 髋/膝/踝量化指标（Pearson r / NMSE / RMSE） |

### 2.3 `report.pdf`（3–5 页）

| 章节 | 内容 |
|------|------|
| 1. 实验配置 | 实际 `total_timesteps`、`num_envs`、地形类型、随机种子、训练硬件、训练时长 |
| 2. 设计决定 | 你为何这样选奖励权重；不同权重组合的 ablation 对比 |
| 3. 训练曲线分析 | 解读 `return.png`，简评 `train_log.json` 中的 KL / clip_fraction / explained_variance / std 演化 |
| 4. 评估结果 | 引用 `sim_ref_joints_right_metrics.csv`、回放视频；分析仍存在的步态问题 |
| 5. 复现命令 | 给出从 `submission/` 还原 + 评估的完整命令（教师会照抄） |

### 2.4 `README.md` —— 教师快速入口

最少包含：

```text
- 学号 / 姓名：
- 训练步数（实际跑了多少 step）：
- 关键改动概要（你在 configs/leg26_baseline.json 里调整了什么）：
- 复现命令：
  bash eval.sh results/train_session_submitted --best-eval
```

---

## 3. 禁改清单

**学生唯一可修改的文件是 `configs/leg26_baseline.json`**。其它任何文件 / 目录 / 脚本都禁止修改。

### 3.1 唯一允许修改的文件

`EXAM/configs/leg26_baseline.json` —— 你可以在其中按作业需要调整各项**已有字段**，核心是下面几类：

1. **`reward_keys_and_weights`**：奖励分项及各分项权重。
2. **`ppo_params`**：**全部** PPO 相关超参数（学习率、并行 rollout 步数、批大小、训练轮数、折扣与 GAE、`clip_range`、熵/价值系数、梯度裁剪等，以模板字段为准）。
3. **训练规模**：`total_timesteps`、`num_envs`、`segment_timesteps`（也可用 CLI 覆盖，见下文）。


> CLI 仅能覆盖：`--total_timesteps`、`--num_envs`、`--segment_timesteps`。除此三项外，**所有**可调内容都应写在 JSON 里；不要新增配置文件未约定的自定义顶层键以免加载失败。

### 3.2 全面禁改清单

```text
EXAM/walk/                             ← 所有 .py 文件（env / policy / train / utils / analyzer 全部）
EXAM/models/                           ← 所有 XML 与 mesh
EXAM/reference_data/                   ← 参考步态 .npz
EXAM/train.bat / .sh                   ← 启动脚本
EXAM/eval.bat / .sh                    ← 同上
EXAM/smoke_test.bat / .sh              ← 同上
```

### 注：测试时使用eval可执行下边代码：unset MUJOCO_GL PYOPENGL_PLATFORM unset EXAM_SKIP_REPLAY MUJOCO_GL=egl PYOPENGL_PLATFORM=egl bash eval_one.sh

---


**祝实训顺利！**
![Alt text](%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20260517223937_13_30-1.png)