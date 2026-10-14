# H2 脚踝温升记录与分析

代替原 sim2real 启动指令执行（不要同时开两个控制器）：

```bash
bash /home/qingyul/workspace/SONIC-H2/gear_sonic_deploy/scripts/run_h2_sim2real_record_ankles.sh
```

仍读取 `gear_sonic_deploy/config/h2_deploy.env`，按原流程 `y` → `Init Done` →
`]` → 稳定后 `t`。脚本创建唯一记录目录并保存本次配置快照。按 `o` 停止控制，
退出后自动生成报告；`o` 会退出姿态保持，先确保支撑。Ctrl+C 停止后也会尝试分析。
本脚本由用户在实机测试时运行，本次实现过程中没有启动实机控制。

日志路径为 `gear_sonic_deploy/logs/h2_ankle_capture_日期时间_随机串/h2_real_日期时间/`。
其中 `ankle_heat_analysis/` 包含：

- `report.md`：左右 roll/pitch 的温升、误差、估计力矩和增益比较。
- `summary.json`：区分动作播放/策略保持的统计、两路温度、状态字、时间对齐质量。
- `ankle_comparison.png`：温度、目标/实际角度、误差、力矩和速度曲线。
- `aligned_ankles.csv`：对齐后的四个关节明细。

原始部署日志约50 Hz记录实际角度、速度、估计力矩、两路温度、状态字；
LowCmd 日志另记录实际下发的目标、Kp、Kd、前馈，包含更高频控制发布。
分析使用状态记录时间之前最近一条命令，超过30 ms不匹配；两者是本机日志时间，
不是精确的电机采样时间。脚踝 DDS 序号：左 roll=4、左 pitch=5、右 roll=10、右 pitch=11。
q/dq 日志使用 H2 local 顺序，这四个脚踝索引与 DDS 一致，不能对所有关节照搬。

单独分析现有日志：

```bash
cd /home/qingyul/workspace/SONIC-H2
.venv_teleop/bin/python scripts/analyze_h2_ankle_heat.py \
  gear_sonic_deploy/logs/h2_real_20260918_190706
```

## 已有 19:07:06 记录的初步结果

76 秒，3802个控制样本，整段 `play=0`，参考名为 dance_in_da_party_001__A464。
左右 roll 均 Kp=28、Kd=2（scale 0.7）。

| 指标 | 左 roll | 右 roll |
| --- | ---: | ---: |
| 温度0，起始→结束/峰值 °C | 40→41/43 | 43→50/52 |
| 温度1，起始→结束/峰值 °C | 32→46/46 | 35→94/94 |
| 跟踪误差 RMS ° | 9.38 | 18.87 |
| 估计力矩 RMS Nm | 4.51 | 9.31 |
| PD 请求 RMS Nm | 4.58 | 9.23 |

10秒后，左 roll 平均目标约 +9.48°、实际 +0.12°；右 roll 平均目标约
−18.93°、实际 +0.09°。右侧 P 项 RMS 9.22，D 项约0.16，说明记录中的请求
主要由持续位置误差贡献。已有数据支持优先检查保持姿态下脚踝为什么几乎不动、
策略目标却持续偏离实际，结合机器人承重/吊装情况、脚底接触限制与标定解释。
这不能单独证明机械卡滞或散热故障。双侧相同的 roll Kp 不等于双侧相同负载。

温度通道1已达到94°C，不应为了复现而长时间维持相同状态；先停止该状态并
待温度回落，再安排短时记录。未修改控制器的保护阈值或增益。

SDK 温度0/1原样保留；本地代码标为绕组/驱动，但具体含义应核对固件。
`tau_est` 是驱动估计量，PD 请求是输出关节坐标下计算值，未计饱和和闭链转换。
当前日志没有电流、电压、电阻或实际热损耗，不能据此计算真实发热功率；
50 Hz 状态日志也不能排除高频振动。
