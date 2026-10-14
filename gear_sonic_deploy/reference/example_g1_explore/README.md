# G1 exploration references

7 条 G1 原生参考动作。每条在清单与 metadata 中标记为 `g1`，对应通用编码器 selector 3。

在 gear_sonic_deploy 目录运行部署程序时，设置：

```text
--motion-data reference/example_g1_explore
```

加载顺序：

1. prayer_R_001__A430: 381 frames, 7.60 s — 单膝下跪祈祷后站起
2. dejection_R_001__A430: 764 frames, 15.26 s — 双膝跪地后站起
3. show_bicep_R_on_all_fours_360_start_R_idle_right_R_002__A474: 780 frames, 15.58 s — 站立、四足向后爬行、恢复站立（A474 采集者）
4. jump_and_land_light_002__A001: 439 frames, 8.76 s — 站立、原地轻跳、落地站立
5. broad_jump_002__A359: 264 frames, 5.26 s — 站立、立定跳远、落地站立
6. jump_twice_R_002__A533: 165 frames, 3.28 s — 站立、连续向前两跳、站稳
7. high_jump_R_opt_2_001__A477: 157 frames, 3.12 s — 站立、原地高跳、落地站立

数据为 50 Hz、29 维 G1 IsaacLab 关节顺序，包含与 reference/example 一致的 14 个身体部位。
位置单位 m，角度 rad，四元数 wxyz，身体速度在世界坐标系。
源 CSV 的 120 Hz 轨迹线性/SLERP 重采样；末尾最多保持不足 20 ms 以保留原始终点。
重采样后首尾各 13 帧通过站姿、足底高度和速度检查。尚未进行策略闭环测试。
需要支持 G1 参考分支的模型/配置；H2-only 编码器无法读取此参考类型。

复现：在仓库根目录运行 python3 scripts/export_bones_g1_deploy_reference.py --output <新目录>。
来源、帧数、关节与身体顺序、SHA-256 见 h2_lora_reference_manifest.json。

2026-09-08 爬行替换：用户明确要求爬行类，第 3 项由 `my_lord_R_002__A428` 改为 `show_bicep_R_on_all_fours_360_start_R_idle_right_R_002__A474`。

本条是此前 A472 同一后爬动作族的不同采集者版本。单条原始完整轨迹，站立→四足后爬→站立；原始 120 Hz 时长 15.575 s，50 Hz 导出时长 15.58 s。约 5.13–10.10 s 为连续爬行，骨盆在此区间的水平净位移约 1.895 m。未拼接或改变速度。

首尾各 13 帧站立检查通过；C++ 数据读取器确认 7 条 G1、50 Hz、29 关节、14 部位；C++ 与 MuJoCo FK 最大位置差约 1.2 微米。以上为数据和运动学核验，不表示闭环跟随成功。

replacement_preview.mp4 是固定镜头的参考轨迹回放，可直接检查爬行位移。

上一版完整 bundle 备份：`/home/qingyul/workspace/SONIC-H2/gear_sonic_deploy/reference/.example_g1_explore_before_crawl_A474_hj449q5e/bundle`。
