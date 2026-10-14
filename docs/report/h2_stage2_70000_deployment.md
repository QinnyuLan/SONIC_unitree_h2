# H2 第二阶段 70,000 轮部署（2026-09-18）

> 当日晚回退：用户停止测试 70000 轮。`config/h2_deploy.env` 和
> `scripts/run_h2_sim2sim.sh` 均已切回 GPU3 stage2 第 9450 轮
> `sonic_h2_native_release/experiments/gpu3_20260907_stage2_step009450/`。
> 该版本由 2026-09-08 实机日志确认。当前动作集仍为 `reference/example`；
> 实机 Kp scale 为左/右 pitch 1.2/1.4、roll 0.7/0.7、膝 1.2/1.2。
> sim2sim 仍使用独立配置及 nominal 增益。下文记录 70000 轮部署历史；
> `exported/` 仍保留 70000 轮，回退后的两个启动脚本显式指定 9450 轮目录。

模型已从 gpu3 的 `sonic_transfer/models/h2_stage2_070000_20260918.tar.gz`
下载并验证，保存在 `sonic_h2_native_release/versions/h2_stage2_070000_20260918/`。
`sonic_h2_native_release/exported/` 默认模型也已更新。

## 修改配置

左右脚踝 roll 也可独立配置：`LEFT_ANKLE_ROLL_KP=0.7`、
`RIGHT_ANKLE_ROLL_KP=0.7`，对应 Kp 各 28，Kd 保持 2。
它们仅由 sim2real 配置脚本使用，修改后重启生效。其他现有参数保留用户的最新设置。

实机参数修改 [h2_deploy.env](../../gear_sonic_deploy/config/h2_deploy.env)，保存后重启
sim2real 控制脚本。此文件只供 sim2real 使用：

```bash
MODEL=../sonic_h2_native_release/versions/h2_stage2_070000_20260918
MOTION_DATA=reference/example
INITIAL_MOTION=""
LEFT_ANKLE_KP=1.4
RIGHT_ANKLE_KP=1.6
LEFT_KNEE_KP=1.3
RIGHT_KNEE_KP=1.3
NETWORK_INTERFACE=eno1
DDS_DOMAIN=0
LOG_ROOT=logs
```

- 当前 Kp scale 沿用用户此前修改的实机脚本设置；动作集已按用户要求改为常规 SONIC demo；对应左右脚踝
  Kp=56/64、左右膝 Kp=260/260。Kd 不变。恢复 nominal 时四个 scale 都设为 1。
- `MOTION_DATA=reference/example_h2/deploy`：H2 原生动作，selector 0。
- `MOTION_DATA=reference/example`：常规 SONIC demo，G1 动作输入，selector 3，控制 H2。
- `MOTION_DATA=reference/example_smpl_h2/deploy`：SMPL 输入，selector 2。
- `INITIAL_MOTION` 留空表示集合默认动作；指定名称时必须在当前集合中存在。
- `MODEL` 可以换为之前保留的版本；路径相对于 gear_sonic_deploy 或填写绝对路径。
- `NETWORK_INTERFACE` 只供实机使用，仿真控制端固定为 `sim`（loopback）。
  实机配置不影响仿真；仿真窗口与仿真控制脚本各自的 `DDS_DOMAIN` 需一致。

镜头构图仍在 [run_h2_paper_sim.sh](../../scripts/run_h2_paper_sim.sh) 开头修改。

## sim2sim

仿真独立使用 [run_h2_sim2sim.sh](../../gear_sonic_deploy/scripts/run_h2_sim2sim.sh)
开头的模型、动作集、初始动作和 DDS 参数。仿真 Kp scale 均为 1.0（nominal），
不读取 `h2_deploy.env` 或 `H2_DEPLOY_CONFIG`。

先关闭旧控制器和旧仿真，终端 1：

```bash
bash /home/qingyul/workspace/SONIC-H2/scripts/run_h2_paper_sim.sh
```

终端 2：

```bash
bash /home/qingyul/workspace/SONIC-H2/gear_sonic_deploy/scripts/run_h2_sim2sim.sh
```

`y` → 等待 `Init Done` → `]` → 仿真窗口 `9` 解除弹性支撑 → 稳定后 `t`。
`n/p` 切换动作、`r` 重置、`o` 停止。

## sim2real

```bash
bash /home/qingyul/workspace/SONIC-H2/gear_sonic_deploy/scripts/run_h2_sim2real_four_kp.sh
```

准备机器人支撑，关闭旧控制器及仿真。`y` → `Init Done` → `]` → 稳定后 `t`。
实机没有 `9` 步骤；`o` 进入阻尼并退出姿态保持。

两个控制脚本都支持 `--print-command`，只显示配置与命令，不启动 DDS 或控制。
仅实机脚本可通过 `H2_DEPLOY_CONFIG=/absolute/path/custom.env` 选择另一份实机配置。

## 验证结果与范围

- 归档 SHA256：`0d4b7db6be40a86454f61e95932dca340d79a7223f594de393d3398d5cb84ffb`。
  包内 23 个文件校验通过；3 个 ONNX 完整检查通过。
- 模型维度：compact encoder 640→64，full encoder 1751→64，decoder 994→29。
  四路 selector 各 5 次 CPU 推理输出有限；compact 与 full selector 0 误差为 0。
- 参考朝向为 `sonic_full_root_relative_v0`；本地 C++ 中 29 个关节名顺序、
  DDS 映射、q0、action scale、nominal Kp/Kd 与 runtime_contract.json 一致。
  H2 输入输出 adapter 已折叠，不重复进行 G1/H2 映射。
- 本地构建通过，14 项 H2Native 测试通过；TensorRT 引擎在新版本目录重新生成。
- 独立 loopback DDS domain 73、paper 场景、弹性支撑下完成两组动作播放：
  原生 H2 弓步（selector 0、nominal gains）以及 G1 祈祷动作
  `prayer_R_001__A430`（selector 3、当前配置的四个 Kp scale）。
  每组 749 条推理样本全部有限，实时倍率约 1，最大延迟分别 1.46/1.65 ms。
  第二组实际 LowCmd 验证脚踝 56/64、膝 260/260 和原有 Kd。
  该次测试记录的是配置拆分前的实机增益组合；当前 sim2sim 已按用户要求恢复 nominal 增益。
- 以上是支撑仿真和部署链路检查，不代表无支撑动作稳定性。未执行实机控制。

证据保存在版本目录中的 `local_*validation.json`、`smoke_supported/` 和
`smoke_configured_g1/`。上游包内 eval 文件属于训练服务器的评估，区别于本地检查。

## 备份与回退

旧默认 step 20000 的模型、说明、元数据和缓存已完整备份到：

`sonic_h2_native_release/exported/backup_before_stage2_step070000_20260918_181453/`

如只需对比旧模型，把配置中的 `MODEL` 指向此目录即可；历史 step 9450 仍在：

`sonic_h2_native_release/experiments/gpu3_20260907_stage2_step009450/`

实机脚本原文件保存在新版本目录的 `sim2real_script_before_update.sh`。
默认 exported 副本更新时同步替换了两个 TensorRT 缓存，旧导出元数据归入备份。
