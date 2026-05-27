# stable-worldmodel 代码理解 Memory

最后更新：2026-05-27

## 维护规则

这个文件用于记录对仓库结构、核心链路和维护注意事项的长期理解。以后如果改动以下内容，请同步更新本文件：`World` 运行循环、policy/solver/model 协议、数据格式读写、环境注册、训练/评估脚本入口、缓存路径或 CLI 行为。

## 项目定位

`stable-worldmodel` 是一个用于 world model 研究的 Python 包。它把数据采集、世界模型训练、基于模型预测控制的评估统一到一套接口里。包入口是 `stable_worldmodel/__init__.py`，主要暴露：

- `World`：环境池、预处理包装、rollout、collect/evaluate 的总入口。
- `PlanConfig`：MPC 规划配置。
- 子包：`data`、`envs`、`policy`、`solver`、`spaces`、`wm`、`wrapper`、`utils`。

命令行入口在 `pyproject.toml` 中注册为 `swm = stable_worldmodel.cli:app`。

## 目录地图

- `stable_worldmodel/world/`：`World` 和 `EnvPool`，负责并行环境运行、重置模式、采集、评估。
- `stable_worldmodel/wrapper/`：环境信息规范化和视觉扰动包装器。`MegaWrapper` 是默认组合包装器。
- `stable_worldmodel/policy.py`：随机策略、feed-forward 策略、world-model MPC 策略，以及 checkpoint 自动扫描工具。
- `stable_worldmodel/solver/`：CEM、iCEM、MPPI、GD、PGD、Lagrangian、Categorical CEM、Predictive Sampling 等规划器。
- `stable_worldmodel/wm/`：参考世界模型和 GCRL/TD-MPC2 模型，核心推理接口通常是 `get_cost`，有些模型还实现 `get_action`。
- `stable_worldmodel/data/`：统一 dataset 抽象、format registry、Lance/HDF5/folder/video/LeRobot 后端、ReplayBuffer、normalization。
- `stable_worldmodel/envs/`：Gymnasium 环境注册和各类环境适配器，导入 `stable_worldmodel.envs` 会注册 `swm/...` 环境。
- `stable_worldmodel/envs/kinder/`：KinDER 环境族的轻量适配包，当前只有 Motion2D；优先使用已安装的 `kindergarden[kinematic2d]`，本地开发时可回退到 checkout。
- `scripts/`：数据采集、训练、评估、可视化、benchmark 脚本。训练脚本大多用 Hydra 配置。
- `scripts/data/collect_kinder_motion2d.py`：KinDER Motion2D 的数据脚本，支持单 seed、多 seed batch 或 `--num-episodes/--start-seed` 连续 seed；默认用 2D grid A* 避开障碍物，streaming 写出可被训练脚本读取的 `dataset_folder/`，并可用 `--artifact-limit` 控制只为前 N 条保存 `episode.npz`、`episode.gif`、`trajectory.png`。Folder writer 每条 episode 后会刷新 `ep_len/ep_offset/*.npz`，避免中断后只留下 pixels 的半成品目录。
- `scripts/data/collect_kinder_motion2d_500.sh`：收集 500 条 KinDER Motion2D episode 的便捷脚本，默认 seeds 0..499、A* policy、`max_steps=300`、只保存前 20 条可视化且不保存 gif，完整数据进入 `outputs/kinder_motion2d_500eps/dataset_folder/`。
- `scripts/eval/evaluate_kinder_motion2d_lewm.py`：KinDER Motion2D 的 LEWM eval 脚本，默认评测 held-out seeds `1000 1001 1002`。脚本先用 A* 收集 eval folder dataset 和 trajectory/gif artifacts，再加载 `lewm_kinder_motion2d_500eps` 的最新 `weights_epoch_*.pt`，计算 latent embedding MSE：teacher-forced one-step 和 autoregressive rollout 两条曲线。LEWM 没有 decoder，所以这是 representation-space dynamics metric，不是 pixel reconstruction metric。加 `--run-planning` 会进一步用 LEWM `get_cost` + CEM/PredictiveSampling 做 dataset-driven MPC：从 eval episode 的起点出发，以未来 `goal_offset` 的 state/image 为 goal，输出 `planning_metrics.json`，包含 success rate、per-episode success、final goal distance 和可选 planning videos；默认 `--planning-horizon=20`，`--planning-receding-horizon=1`。加 `--planner-debug-video` 会保存 MPC rollout debug video，蓝色 top-k 候选 rollout，红黄高亮 final selected rollout，并写出对应 npz/json。加 `--plot-oracle-cost` 会画 oracle/A* planner cost-to-go 曲线，默认 seeds 1000..1019 共 20 条，输出 `oracle_cost_curves.png/csv`；若指定 `--oracle-cost-goal-offset`，cost 目标就是该 offset 的中间 goal，否则在 planning 模式下跟随 `planning_goal_offset`，无 planning 时使用 episode 原始 target。
- `scripts/train/config/data/kinder_motion2d.yaml`：LEWM 等训练脚本使用的 KinDER Motion2D 数据配置，默认 `frameskip=1`，加载 `pixels/action/proprio/state`。
- `docs/`：MkDocs 文档和 API/教程页面。
- `tests/`：单元测试，覆盖 world loop、policy、solver、dataset、format、wrapper、env pool 等。

## Evaluation

- KinDER Motion2D 的 MPC rollout debug video 在 `scripts/eval/evaluate_kinder_motion2d_lewm.py` 里通过 `--planner-debug-video` 开启。它记录每次 CEM replan 的 top-k candidate action sequence 和最终 selected action sequence，再把这些 action 放回真实 Motion2D simulator 从当前 state rollout，可视化成轨迹 overlay：蓝色细线是 top-k candidates，红黄粗线是 final selected rollout，绿色星号是当前 goal。输出在 `<out-dir>/planner_debug/seed_<seed>_mpc_rollouts.{mp4,npz,json}`。
- 相关默认：`--planning-horizon=20` 控制每次 MPC 预测/优化多少个未来 step，`--planning-receding-horizon=1` 控制执行几步后重新规划，`--planning-goal-offset` 不传时通常是 50，表示从 `planning_start_step` 往后第 50 帧作为 goal。若想让 debug video 的 rollout 更长，优先调大 `--planning-horizon`，例如 `--planning-horizon 40`；若想看 CEM 每轮分布收缩，用 `--planner-debug-cem-iters all`。

## 核心运行链路

`World(env_name, num_envs, image_shape, ...)` 会：

1. 用 `gym.make` 创建多个环境。
2. 应用 `pre_wrappers`。
3. 应用 `MegaWrapper`，把 observation、reward、action、terminated/truncated、pixels、goal 等统一塞进 `info`。
4. 应用 `extra_wrappers`。
5. 用 `EnvPool` 管理并行环境。

`World.set_policy(policy)` 会保存策略并调用 `policy.set_env(self.envs)`。MPC 策略会在这里把 action space、env 数量和 `PlanConfig` 配置进 solver。

`World.collect(...)` 和 `World.evaluate(...)` 都走 `_run_iter(...)`：

- 每步调用 `policy.get_action(self.infos)`。
- 再调用 `EnvPool.step(actions, mask=...)`。
- `auto` 模式会在 env 完成后立刻 reset，并给 `infos['_needs_flush'] = done`，让 `WorldModelPolicy` 清空对应 env 的 action buffer。
- `wait` 模式不会 reset，已完成 env 会 frozen，直到所有 env 完成或达到步数上限。

重要形状约定：`EnvPool` 把每个 info key 堆成 `(num_envs, 1, ...)`，中间的 `1` 是时间维。很多 policy、dataset、模型都依赖这个 batch/time 约定。

## Policy / Solver / Model 协议

主要协议：

- Policy：`set_env(env)` 和 `get_action(info_dict)`。
- Solver：`configure(action_space, n_envs, config)` 和 `solve(info_dict, init_action=None)`；solver 实例通常也实现 `__call__` 转发到 `solve`。
- Costable model：`get_cost(info_dict, action_candidates)`，返回 `(B, N)` cost，越小越好。
- Actionable model：`get_action(info_dict, horizon=1, prefix_actions=None)`，可用于 actor policy 或 solver warm-start。

`WorldModelPolicy` 的工作方式：

- 按 `PlanConfig.receding_horizon` 维护每个 env 的 action buffer。
- buffer 空时调用 solver 重新规划。
- `warm_start=True` 时保存上一轮 plan 的剩余部分作为下一次 solver 的 `init_action`。
- 如果收到 `_needs_flush`，对应 env 的 buffer 和 warm-start 状态会被清掉。
- 如果 info 里 `terminated=True`，该 env 输出 NaN action，配合 `wait` 模式避免继续踩死环境。

`PlanConfig` 是 frozen dataclass，字段包括 `horizon`、`receding_horizon`、`history_len`、`action_block`、`warm_start`。`plan_len = horizon * action_block`。

## 数据系统

核心抽象是 `data.Dataset`：

- episode-based，子类实现 `column_names` 和 `_load_slice(ep_idx, start, end)`。
- `__getitem__` 基于 `clip_indices` 返回固定长度 clip。
- `frameskip` 控制观测采样 stride。
- `action` 会 reshape 成 `(num_steps, -1)`，因此 action block/frameskip 相关逻辑要小心。

Format registry 在 `data/format.py`：

- 用 `@register_format` 注册 `Format` 子类。
- 内置格式由 `data/formats/__init__.py` 导入注册。
- `load_dataset` 会解析本地路径、HF repo id、scheme URL，再自动检测格式。

现有格式：

- `lance`：默认格式。LanceDB 表，episode-contiguous flat rows，图片列存 JPEG bytes。Lance 不接受字段名中的 `.`，writer 会把 `foo.bar` 改成 `foo_bar`。
- `hdf5`：单文件。
- `folder`：`.npz` tabular 列 + 每步图片文件。
- `video`：`.npz` tabular 列 + 每 episode 视频。
- `lerobot`：只读 adapter，scheme 类似 `lerobot://...`。

`World.collect(path, format='lance')` 要求 `path` 和 `writer` 二选一。写出 episode 时会跳过 `_` 开头的 info key，只记录 ndarray/tensor；如果 episode 有 `action`，会把初始 dummy action 移到末尾以对齐 transition。

缓存目录由 `STABLEWM_HOME` 控制，默认是 `~/.stable_worldmodel/`，dataset 在 `datasets/`，checkpoint 在 `checkpoints/`。

### 数据处理

KinDER Motion2D 当前采集和训练默认使用 `folder` format。500 条数据默认位置：

```text
outputs/kinder_motion2d_500eps/dataset_folder/
  ep_len.npz
  ep_offset.npz
  action.npz
  reward.npz
  terminated.npz
  truncated.npz
  step_idx.npz
  state.npz
  proprio.npz
  goal_state.npz
  goal_proprio.npz
  pixels/
    ep_<i>_step_<j>.jpeg
```

训练 LEWM 时使用：

```bash
python scripts/train/lewm.py \
  data=kinder_motion2d \
  data.dataset.name=outputs/kinder_motion2d_500eps/dataset_folder \
  ...
```

`data=kinder_motion2d` 会加载 `scripts/train/config/data/kinder_motion2d.yaml`，默认 `frameskip=1`，`keys_to_load=[pixels, action, proprio, state]`，`keys_to_cache=[action, proprio, state]`。`load_dataset` 会自动检测 `folder` format；如果自动检测失败，通常说明 `dataset_folder` 半成品或缺少 `ep_len.npz/ep_offset.npz`。

注册的数据格式和读写能力：

- `folder`：可读可写；当前 KinDER Motion2D 采集默认格式。
- `hdf5`：可读可写；单个 `.h5/.hdf5` 文件。
- `lance`：可读可写；stable-worldmodel 默认常用格式。
- `video`：可读可写；tabular `.npz` + 每 episode 视频。
- `lerobot`：只读 adapter；支持从 `lerobot://...` 读入，但当前没有 LeRobot writer，不能直接导出成 LeRobot 格式。

通用转换入口：

```bash
python scripts/data/convert.py \
  --source <source-path-or-id> \
  --source-format <optional-source-format> \
  --dest <dest-path> \
  --dest-format <folder|hdf5|lance|video> \
  --mode overwrite
```

KinDER Motion2D folder 转 HDF5：

```bash
python scripts/data/convert.py \
  --source outputs/kinder_motion2d_500eps/dataset_folder \
  --source-format folder \
  --dest outputs/kinder_motion2d_500eps.h5 \
  --dest-format hdf5 \
  --mode overwrite
```

KinDER Motion2D folder 转 Lance：

```bash
python scripts/data/convert.py \
  --source outputs/kinder_motion2d_500eps/dataset_folder \
  --source-format folder \
  --dest outputs/kinder_motion2d_500eps.lance \
  --dest-format lance \
  --mode overwrite
```

从 LeRobot 读入并转成 stable-worldmodel 格式是支持的，例如：

```bash
python scripts/data/convert.py \
  --source lerobot://lerobot/pusht \
  --dest outputs/pusht_from_lerobot.lance \
  --dest-format lance \
  --mode overwrite
```

转换后的 HDF5/Lance 可以直接用于训练，只需把 `data.dataset.name` 改成转换后的路径，例如 `outputs/kinder_motion2d_500eps.h5` 或 `outputs/kinder_motion2d_500eps.lance`。

Folder dataset 现在对训练脚本兼容：

- `FolderDataset` 接受 `keys_to_cache`，并会把这些 tabular `.npz` 列缓存起来；这避免 Hydra 默认数据配置把 `keys_to_cache` 传入 folder reader 时报错。
- `FolderDataset.get_dim(col)` 与 Lance/HDF5 一致，返回除 batch 维外的扁平维度；LEWM 用它设置 action encoder 输入维度。
- `load_dataset` 会先检查当前工作目录下的本地相对路径，再退回到 `STABLEWM_HOME/datasets` 和 HuggingFace 解析，因此 `outputs/.../dataset_folder` 这类相对路径可以直接用。

## 环境与 Wrapper

`envs/__init__.py` 注册 `swm/...` 环境，并维护：

- `WORLDS`：所有注册环境 id。
- `DISCRETE_WORLDS`：离散 action 环境 id。

目前覆盖 PushT、TwoRoom、OGBench cube/scene/maze、DMControl、Gymnasium control、Fetch robotics、Craftax、ALE 等。

KinDER 目前只做了 Motion2D 的轻量适配：

- 注册 id：`swm/KinderMotion2D-v0`，以及 `swm/KinderMotion2D-p0-v0` 到 `swm/KinderMotion2D-p5-v0`。
- 适配类：`stable_worldmodel.envs.kinder.motion2d:KinderMotion2D`。
- KinDER 是环境族，不是单个环境；新增 KinDER env 时应在 `stable_worldmodel/envs/kinder/` 下加平行模块，并复用 `_utils.py` 的懒加载/本地 checkout fallback。
- 推荐安装：`pip install "kindergarden[kinematic2d]"`；stable 也提供 `stable-worldmodel[kinder]` optional extra。
- 本地开发回退：如果 import 失败，会尝试 `/home/robin_wang/kindergarden/src`，也可用 `KINDERGARDEN_HOME` 指向其他 checkout。
- KinDER 和 kinematic2d 依赖是懒加载的，只有实际 `gym.make` 这个 env 时才需要。
- Motion2D 的 raw observation 是完整向量状态；适配器保持 observation 原样，并在 info 中写 `state = observation`。`goal_state` 也保持完整向量形状，派生的 `proprio/goal_proprio` 只取 robot 的 `(x, y, theta, arm_joint, vacuum)`。
- Dataset-driven eval 时可按 PushT 的模式传 callables：先 `_set_state(state=state)`，再 `_set_goal_state(goal_state=goal_state)`；full-vector `goal_state` 会用其中 robot 的 `(x, y)` 设置 Motion2D 的 target region。
- `unwrapped` 必须保持为 stable 适配器本身，供 `World.evaluate(..., callables=[_set_state, _set_goal_state])` 调到 wrapper 的 dataset-eval hooks；底层 KinDER env 通过 `kinder_env` 属性访问。
- 数据采集 policy 的重点坑：Motion2D 机器人有 base + gripper/arm，单纯对 base center 做 grid A* 会在窄 passage 中失败，尤其随机初始 `theta` 会让 gripper 撞墙。当前 collect 脚本默认先从竖墙障碍中解析 passage center waypoints，再用 heading controller 让机器人先朝向运动方向；grid A* 只作为 fallback。seed 0..19、`max_steps=300` 的 smoke benchmark 为 20/20 success。

KinDER ClutteredStorage2D 的轻量适配：

- 注册 id：`swm/KinderClutteredStorage2D-v0`，以及官方 block variants `swm/KinderClutteredStorage2D-b1-v0`、`b3`、`b7`、`b15`；默认采集和训练先以 `b1` 为准。
- 适配类：`stable_worldmodel.envs.kinder.cluttered_storage2d:KinderClutteredStorage2D`。observation 保持 KinDER full vector；`info['state']` 等于完整 observation；`proprio/goal_proprio` 是 robot 的 `(x, y, theta, arm_joint, vacuum)`；`goal_state` 是 full-vector shelf storage goal。
- 采集脚本：`scripts/data/collect_kinder_cluttered_storage2d.py`，默认 `--num-blocks 1`、`--policy scripted`、`--max-steps 400`，写出 Motion2D 同款 stable folder dataset，并可保存 `episode.npz`、`trajectory.png`、`episode.gif`、`metadata.json`。500 条数据便捷脚本是 `scripts/data/collect_kinder_cluttered_storage2d_500.sh`，默认 seeds 0..499，输出到 `outputs/kinder_cluttered_storage2d_b1_500eps/dataset_folder/`。
- Scripted expert 只承诺 b1：先 motion plan 到 block 自身长边面的 pregrasp pose，arm retracted；再沿该面法线伸出吸盘并开 vacuum；随后带物体 motion plan 到 shelf 开口下方的 vertical preinsert pose；最后保持 robot heading 向上，用 `darm` 竖直插入 shelf。失败时不 teleport，metadata 记录 failure reason。
- 几何注意：KinDER rectangle 的 `x/y/theta` 是局部左下角 pose，不是中心点。做 grasp/insert 候选时需要用 local axes 或从目标中心反推出 lower-left pose，否则 `is_inside_shelf` 会误判。
- smoke 验收：2026-05-27 本地跑 `seeds 0..9`、b1 scripted、`max_steps=400` 为 10/10 success；单 seed 会保存 gif，例如 `outputs/kinder_cluttered_storage2d_b1_policy_check/episode.gif`。

`MegaWrapper` 组合顺序：

1. 可选 `AddPixelsWrapper`：从 `render()` 或 `render_multiview()` 取图，resize 后写入 `info['pixels']` 或 `info['pixels.<view>']`。
2. `EverythingToInfoWrapper`：把 observation、reward、action、terminated、truncated、step_idx、id 等都写进 info。
3. `EnsureInfoKeysWrapper`：校验必要 key。
4. 可选 `ResizeGoalWrapper`：resize `info['goal']`。

`stable_worldmodel/spaces.py` 扩展 Gymnasium spaces，增加 `value/init_value/reset/update/check` 等状态追踪能力，主要用于 factors of variation。`reset_variation_space` 会按 reset options 采样或设置 FoV。

## Solver 重点

连续 action 常用：

- `CEMSolver`：采样候选 action sequence，按 model cost 选 top-k 更新 mean/std。
- `GradientSolver`：把 action sequence 当可优化参数，对 model cost 反向传播。
- `ICEMSolver`、`MPPISolver`、`PGDSolver`、`LagrangianSolver`、`PredictiveSamplingSolver` 提供变体。

离散 action：

- `CategoricalCEMSolver` 要求 `gymnasium.spaces.Discrete`，输出 action index，支持 `action_block`。

`prepare_init_action` 是 solver warm-start 共享逻辑：如果 model 是 `Actionable`，缺失的 horizon 会用 `model.get_action(..., prefix_actions=...)` 补齐；否则用 0 padding。

## World Model / Baseline

主要参考实现：

- `wm/lewm/LeWM`：图像 encoder + action encoder + predictor。`get_cost` 编码 goal，rollout action candidates，按最终 latent/goal MSE 算 cost。
- `wm/prejepa/PreJEPA`：DINO-WM 风格，支持额外 modality encoder，按 patch latent rollout，`get_cost` 对预测 latent 与 goal latent 算 MSE。
- `wm/pldm/PLDM`：类似 JEPA latent dynamics。
- `wm/tdmpc2/TDMPC2`：latent dynamics + reward/Q/actor，既可 `get_action` actor rollout，又可 `get_cost` 对候选轨迹算负回报。
- `wm/gcrl/GCRL`：goal-conditioned action/value 模型，`get_action` 从 observation/goal embedding 预测动作。

训练脚本常用 `stable_pretraining`、Lightning、Hydra。`wm/utils.py` 的 `save_pretrained/load_pretrained` 用 `config.json + weights.pt` 形式保存/加载，路径默认在 checkpoint cache 下，也支持 HuggingFace repo。

LEWM 训练 KinDER Motion2D 的最小稳定命令建议：

- 使用 `data=kinder_motion2d data.dataset.name=<dataset_folder>`，避免手写 `keys_to_load/keys_to_cache`。
- CPU 或无可用 CUDA 时设置 `trainer.accelerator=cpu trainer.precision=32`。
- 本机 smoke run 建议 `num_workers=0 '~loader.prefetch_factor' loader.persistent_workers=false`，避免 DataLoader 多进程和 `prefetch_factor` 的兼容问题。
- 如果 home 目录或默认 cache 不可写，设置 `STABLEWM_HOME=outputs/stablewm_cache` 和 `SPT_CACHE_DIR=outputs/spt_cache`。
- `SIGReg` 必须跟随输入 tensor 的 device/dtype 创建随机投影，不能硬编码 `device='cuda'`，否则 CPU 训练会失败。
- `scripts/train/lewm.py` 支持 `resume_ckpt_path=<abs path>` 和 `resume_weights_only=false`。用 stable-pretraining 的 `outputs/spt_cache/runs/.../checkpoints/last.ckpt` 恢复时要 `resume_weights_only=false`，这样 epoch、optimizer、scheduler 一起恢复；默认的权重文件 fallback 仍按 weights-only 加载。
- 2026-05-27 的 KinDER Motion2D 500eps 训练 run `outputs/spt_cache/runs/20260527/043013/c227d49c3fc9/metrics.csv`：训练中断在 epoch 19，验证只到 epoch 18。`validate/loss_epoch` 从 0.2235 降到 epoch 16 的最低 0.1253，epoch 18 为 0.1290；`validate/pred_loss_epoch` 继续降到 epoch 18 的 0.00126；`validate/sigreg_loss_epoch` 在 epoch 16 最低 1.3785 后轻微回升到 1.4198。判断：prediction loss 基本收敛，总 loss 已进入平台期，继续训练收益可能有限；现有 checkpoint 只有 `epoch=18-step=41249.ckpt` 和 `last.ckpt`。

## CLI

`swm` 命令基于 Typer：

- `swm datasets`：列出缓存数据集。
- `swm inspect <name>`：检查 Lance/HDF5/folder/video 数据集列、大小、episode 信息。
- `swm envs`：列出已注册环境。
- `swm fovs <env>`：列出指定环境 variation space。
- `swm convert <name> --dest-format video`：转换数据格式。
- `swm checkpoints [filter]`：列出缓存 checkpoint。

## 常用命令

开发安装：

```bash
uv venv --python=3.10
uv sync --extra all --group dev
```

测试：

```bash
pytest
pytest tests/test_world.py tests/test_policy.py
pytest tests/data
pytest tests/solver
```

示例：

```bash
swm envs
swm datasets
swm inspect <dataset_name>
```

## 维护注意事项

- `World` 的 reset/flush 逻辑很敏感，改 `_run_iter` 后优先跑 `tests/test_world.py` 和 `tests/test_new_world.py`。
- `EnvPool` 预分配并原地更新 stacked info，新增 info key 目前不会自动进入已存在 buffer。
- `WorldModelPolicy.get_action` 会 `pop('_needs_flush')`，传入的 info dict 可能被策略修改；相关测试会模拟这个行为。
- `add_pixels=True` 时 `World` 必须传 `image_shape`。
- `World.evaluate(dataset=...)` 要求 `num_envs == len(episodes_idx)`，默认 reset mode 是 `wait`。
- 数据读写新增列或改 shape 时要检查 append mode 的 schema 兼容性。
- Lance writer 会把点号列名改成下划线，依赖原始 FoV key 时要留意。
- 部分格式和环境是 optional dependency；导入 HDF5/video/ALE/Craftax/DMControl 相关模块时要考虑缺依赖路径。
- `CEMSolver` 和 `GradientSolver` 默认会 print solve time，测试或批量评估时可能产生较多输出。
