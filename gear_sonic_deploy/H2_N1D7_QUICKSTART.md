# H2 N1.7 真机测试入口

完整调用链、接口、时序、暂停与故障行为见
[`../docs/report/h2_n1d7_sim2real.md`](../docs/report/h2_n1d7_sim2real.md)。

在 m21 的 `/home/qingyul/workspace/SONIC-H2` 执行：

```bash
# 不连接硬件的代码/协议检查
bash gear_sonic_deploy/scripts/check_h2_vla.sh

# 终端 A：N1.7 模型服务
bash gear_sonic_deploy/scripts/run_h2_vla.sh server

# 服务就绪后，另一个终端：采集帧回放与预热，不发布机器人指令
bash gear_sonic_deploy/scripts/run_h2_vla.sh replay --samples 12

# 终端 B：SONIC H2 控制器；现场按原程序提示确认初始化
bash gear_sonic_deploy/scripts/run_h2_vla.sh deploy

# 终端 C：默认先观察预测，退出 shadow 后再启动 live
bash gear_sonic_deploy/scripts/run_h2_vla.sh shadow
# Ctrl+C 退出 shadow
bash gear_sonic_deploy/scripts/run_h2_vla.sh live
```

进入 live 后输入 `stand` 并回车，确认实体稳定站立、头部到达 40°/0°，
再输入 `run` 并回车。`pause` 回到站立参考；再次 `run` 明确恢复。
`quit`/Ctrl+C 退出客户端，接收器回退到站立参考；`stop` 会结束整个控制程序。

先关闭占用动作端口 5556 的 PICO/旧客户端，以及占用视频 UDP 5002 的采集器/查看器。
相机源 RTP 发送服务和现有 Inspire 驱动继续使用。

N1.7 模型：`models/h2_n1d7/latest`，实际版本以 `deployment_manifest.json` 为准。
SONIC：固定采集所用 `gpu3_20260909_stage2_step005000`。
本交付完成离线/协议/模型推理验证，实体动作及任务效果由现场测试确认。
