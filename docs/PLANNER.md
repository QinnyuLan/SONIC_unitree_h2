# G1 planner 下载与本地测试

H2 VR 部署沿用 SONIC 的共享 G1 kinematic planner。正式 H2 控制器位于 `models/released/`，本下载操作仅补齐 planner。

## 本地路径与 Git

下载脚本和 VR 启动脚本使用相同路径；也可将已有权重放入此位置：

```text
gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
```

此文件已加入根目录 `.gitignore`，不会随普通 `git add` / `git push` 上传。新克隆的仓库不含该权重。文件大小为 773,952,989 字节（约 738 MiB）。sim2sim、sim2real 的动作回放入口不需要 planner；VR 部署端需要。

## 推荐下载方法

在工程根目录运行；`uv` 安装方法见 [依赖说明](../requirements/README.md)。

```bash
uv venv .venv_download --python 3.10
uv pip install --python .venv_download/bin/python -r requirements/download.txt
.venv_download/bin/python download_from_hf.py --planner-only
python3 tools/verify_release.py --require-planner
```

也可以在已有 Python 环境中执行 `python -m pip install -r requirements/download.txt`，然后执行 `python download_from_hf.py --planner-only`。默认目标路径以脚本所在目录为基准，执行命令时的当前目录不影响它。

若本地文件的 SHA256 已匹配，脚本直接复用，不访问网络，也不要求安装下载依赖。需要下载时，脚本使用 Hugging Face 缓存，先写临时文件，通过校验后才替换目标文件；下载或校验失败不会覆盖原文件。缓存通常在 `~/.cache/huggingface`，可通过 `HF_HOME` 指定其他磁盘。首次下载为缓存和目标副本预留至少 1.6 GB 空间。

## 固定来源与校验

官方仓库：[nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC)。文件：[planner_sonic.onnx](https://huggingface.co/nvidia/GEAR-SONIC/blob/72935a37d3860204617ebaaf6c855a1c5aff7b8c/planner_sonic.onnx)。

```text
revision: 72935a37d3860204617ebaaf6c855a1c5aff7b8c
SHA256:   39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea
```

下载脚本固定 revision 和 SHA256，确保复现同一份 planner。上述 SHA256 已与本机保留的权重及 Hugging Face 文件信息核对一致。

如果需要手动获取，可下载上述链接中的原始文件到同一路径，再执行 `python3 tools/verify_release.py --require-planner`。不要将网页保存为 `.onnx`。

## 检查与运行

```bash
# 确认文件被 Git 忽略
git check-ignore gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
# 校验发布模型与 planner，不连接机器人
python3 tools/verify_release.py --require-planner
# 只检查 VR 部署命令
bash gear_sonic_deploy/scripts/run_h2_teleop_head_adjust.sh deploy --print-command
```

不带 `--require-planner` 的检查允许新克隆仓库缺少 planner，并给出下载提示；若文件已经存在但校验不符，检查仍会失败。完整 VR 环境、设备连接和启动步骤见 [主 README](../README.md)。
