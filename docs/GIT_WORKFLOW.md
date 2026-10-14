# 独立 Git 管理

远端：`git@github.com:QinnyuLan/SONIC_unitree_h2.git`。默认分支：`main`。

所有发布启动入口使用 `models/released`。部署目录可自行选择。

## 克隆与验证

```bash
git clone git@github.com:QinnyuLan/SONIC_unitree_h2.git
cd SONIC_unitree_h2
python3 tools/verify_release.py
```

正式 H2 模型随 Git 提供，检查脚本校验其 SHA256。VR 使用的 G1 planner 不上传 Git，新克隆工程需按 [Planner 下载说明](PLANNER.md) 单独获取，随后执行 `python3 tools/verify_release.py --require-planner`。已有 planner 文件可放入相同路径复用。

## 日常同步

```bash
cd ~/workspace/SONIC_unitree_h2
git status
git pull --ff-only
# 修改并验证之后，暂存需要提交的文件，例如：
git add -- README.md README_CN.md
git commit -m "说明本次修改"
git push origin main
```

请勿对下表中的大文件使用 `git add -f`。GitHub 普通 Git 不接受超过 100 MiB 的单个文件，工作目录中的这些副本由 `.gitignore` 排除。

## 资产范围

仿真、实机、VR 源码、正式发布 ONNX、MuJoCo/H2 机器人网格、运动参考和随附 SDK 纳入普通 Git。虚拟环境、编译产物、日志、TensorRT 缓存和 XRoboToolkit Python 扩展由目标机器生成。

| 本地保留、Git 忽略的文件 | 用途与恢复方式 |
|---|---|
| `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx` | VR 共享 G1 planner；运行 `download_from_hf.py --planner-only` 恢复至同一路径 |
| `sonic_h2_native_v1_1_release/exported/model_decoder.onnx` 和 `model_decoder_h2_native.onnx` | 历史实验模型；只保留本地副本，三个正式入口使用 `models/released` |
| `gear_sonic/data/robots/g1/g1_43dof_s3.usda` | 可选 G1 Isaac 训练资产；正式 H2 MuJoCo/实机/VR 入口不用它；需要训练时另行提供 |
| `data/body_models/smplx/SMPLX_NEUTRAL.npz` | 可选 SMPL-X 训练/人体处理资产；三个默认部署入口不用它；使用相关训练工具时需自行取得对应人体模型并放回此路径 |

原工作区四份可选 MotionBricks 训练权重只有未下载的指针，不纳入本仓库；MotionBricks 源码保留，使用其训练或独立演示需另行获取权重。

GitHub Pages 文档工作流仅手动触发。`release/source_manifest.json` 的初始复制记录用于溯源；`release_artifacts` 中相同路径的 SHA256 是发布修改后的值。`standalone_repository` 记录独立仓库位置及未纳入 Git 的本地文件。
