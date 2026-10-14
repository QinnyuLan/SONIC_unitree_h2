# 发布复现记录

## 发布模型

sim2sim、sim2real、VR 遥操作和通用 `deploy.sh --robot-profile h2_native` 默认使用 `models/released`。这是直接微调的原生 H2 模型，无需 LoRA 合并。发布路径、配置、终端和本文不显示训练轮次、机器名和实验名称；ONNX 保持原始字节，内部导出元数据用于溯源。

完整编码器 `obs_dict float32[1,1751] → encoded_tokens float32[1,64]`；精简编码器 `obs_dict float32[1,640] → encoded_tokens float32[1,64]`；解码器 `obs_dict float32[1,994] → action float32[1,29]`。完整编码器 selector 为 H2 / VR / SMPL / G1。三份 ONNX 文件的 SHA256 见 `models/released/SHA256SUMS`。

H2 本体有 31 个关节，策略输出其中 29 维动作，头部由控制层处理。policy 50 Hz、LowCmd 500 Hz、physics 1000 Hz。观测 YAML、关节顺序、增益与三个流程的设置保持匹配。

## 本次验证

- 三份文件与源模型 SHA256 一致，ONNX full checker 通过。
- 与原 sim2sim、sim2real、VR 模型逐项比较：输入输出数量、名称、类型、维度，以及观测、朝向和 selector 契约一致；策略观测、动作配置、机器人资产和关节默认值一致。
- 每个 selector 进行 5 次 CPU encoder→decoder 推理，共 20 次，输出全部有限。H2 完整/精简编码器最大绝对误差为 0。
- C++ 部署程序与测试重新编译通过，47 个单元测试通过；新增测试覆盖直接微调模型加载，以及错配的 encoder、decoder、模型角色、朝向契约被拒绝。
- TensorRT 在本机 GPU 上新建引擎，并对四个 selector 执行 encoder→decoder 推理，输出全部有限。FP32（关闭 TF32）与 CPU 对照的 token 最大绝对误差为 0，动作最大绝对误差小于 `5e-6`。
- 三个主启动流程的 `--print-command` 通过，模型路径均为 `../models/released`。

结果见 `release/model_validation.json`、`release/gpu_model_validation.json`、`release/cpp_tests.json`。这些检查使用合成输入与单元测试，没有验证本模型的实时仿真跟踪或实机稳定性。

## 复现命令

从 `SONIC_unitree_h2` 根目录：

```bash
python3 tools/verify_release.py
uv venv .venv_model_validation --python 3.10
uv pip install --python .venv_model_validation/bin/python -r requirements/model_validation.txt
.venv_model_validation/bin/python tools/validate_released_models.py
export TensorRT_ROOT="$HOME/TensorRT"
./verify_deploy.sh
cd gear_sonic_deploy
just build
./target/release/run_tests
./scripts/run_h2_sim2sim.sh --print-command
```

C++ 测试以 `gear_sonic_deploy` 为工作目录，读取相对路径下的配置、动作和模型。实际 sim2sim 需要同时运行 MuJoCo 窗口和控制端，见根目录 README。仿真两端 DDS domain 需一致；VR 的设备配对和校准需在现场完成。

## 清单说明

`release/source_manifest.json` 保留初始源工作区的复制清单；`release_artifacts` 记录发布目录内新增/修改文件的最终 SHA256，覆盖对应路径的初始副本。`published_model` 记录本次选定模型的中性名称、相对路径和文件 SHA256。历史模型和报告用于追溯，不是当前默认模型。

普通 Git 仓库保留正式 H2 ONNX；共享 G1 planner 保留本地供测试，由 Git 忽略，新克隆工程按 `docs/PLANNER.md` 下载到同一路径，再执行 `python3 tools/verify_release.py --require-planner`。超过单文件限制的可选资产见 `docs/GIT_WORKFLOW.md`。TensorRT engine 和编译产物不作为发布资产，需在目标机器生成。VLA 扩展依赖数据集对应的 decoder，仍独立于这三个主流程进行配置与验证。
