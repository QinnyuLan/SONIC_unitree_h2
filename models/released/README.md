# released

这是 sim2sim、sim2real、VR 遥操作共用的正式发布模型。模型直接微调，原生接收 H2 观测并输出 H2 动作，无需 LoRA 合并或输入输出适配器合并步骤。

发布配置统一写作 `MODEL=../models/released`（相对于 `gear_sonic_deploy`）。终端和发布文档使用 `released`，不展示训练轮次、机器名和实验名称。三个 ONNX 文件按原始字节复制，内部导出元数据保留用于溯源；文件完整性由 `SHA256SUMS` 记录。

## 输入输出

所有输入输出均为 float32，batch 固定为 1；名称、类型、维度和运行契约与此前仿真、实机、VR 三套模型一致。

| 文件 | 输入 | 输出 | 用途 |
|---|---|---|---|
| `model_encoder_full.onnx` | `obs_dict [1,1751]` | `encoded_tokens [1,64]` | 默认完整编码器，支持 H2 / VR / SMPL / G1 |
| `model_encoder.onnx` | `obs_dict [1,640]` | `encoded_tokens [1,64]` | 精简 H2 编码器，需配套 compact observation 配置 |
| `model_decoder.onnx` | `obs_dict [1,994]` | `action [1,29]` | 64 维 token + 930 维历史观测 → 29 维归一化动作 |

完整编码器 selector：0=H2、1=VR 三点、2=SMPL、3=G1。控制层将 29 维策略动作映射到 H2 本体，头部另行处理。观测契约为 `sonic_h2_native_v1`，朝向契约为 `sonic_full_root_relative_v0`。契约中的后缀属于接口标识，不是模型训练版本。

运行时保留输入输出数量/维度、模型角色、policy interface、selector、观测及朝向契约检查；移除了 `h2_native_io_folded` 的限制。此字段描述导出时的适配器处理，不能用来要求直接微调模型执行 LoRA 合并。

## 验证与复现

从 `SONIC_unitree_h2` 根目录执行：

```bash
python3 tools/verify_release.py
uv venv .venv_model_validation --python 3.10
uv pip install --python .venv_model_validation/bin/python -r requirements/model_validation.txt
.venv_model_validation/bin/python tools/validate_released_models.py
```

验证脚本检查 SHA256、ONNX 图、输入输出、运行契约及四个 selector 的 CPU encoder→decoder 推理；不连接机器人。可用重复的 `--compare <模型目录>` 对照历史模型；只比较接口语义，不要求新旧模型输出数值相同。

完整与精简编码器在 H2 模式下的输出必须一致。具体测试结果见 `../../release/model_validation.json`。目标机器会重新生成 TensorRT 缓存；本目录不分发机器相关的缓存文件。
