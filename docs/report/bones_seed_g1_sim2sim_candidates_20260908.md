# BONES-SEED G1：站立起止的 sim2sim 候选动作（2026-09-08）

已选出 9 条优先动作和 11 条备选。优先动作均来自单条完整原始轨迹，首尾满足下述站立筛选条件，无需拼接、倒放或补造站立帧。这是参考轨迹筛选结果，尚未进行策略闭环 sim2sim 测试。

检查了现有 `reference/extreme_h2/deploy`、`reference/example_h2/deploy` 和 `reference/example` 共 27 条 demo；25 条通过首尾姿态初筛，以下 2 条不满足每条 demo 都从站立开始、站立结束的要求：

| 现有 demo | 开头 | 结尾 | 结论 |
| --- | --- | --- | --- |
| `kneeling_start_003__A167_M` | 骨盆 1.052 m，双膝 5.8°/4.6° | 骨盆 0.595 m，双膝 91.0°/96.2° | 结尾跪地，不能单独作为完整 demo |
| `kneeling_stop_003__A167_M` | 骨盆 0.594 m，双膝 95.3°/97.1° | 骨盆 1.046 m，双膝 3.5°/3.9° | 开头跪地，不能单独作为完整 demo |

现有轻跳 `jump_and_land_light_002__A001` 的 H2 版本通过首尾姿态初筛，其原始 G1 版本也入选本次清单。现有目录中的姿态初筛没有验证足部接触或控制器跟随。两段下跪/站起即使名字对应，也不能未经关节、速度和坐标对齐就直接拼接。

磁盘最初只剩约 16 GB，而 `g1.tar.gz` 约 22 GB，因此采用定向解压：结合文件名、自然语言描述和技术描述筛选非镜像动作，流式遍历完整压缩包，提取 10,769 条相关 CSV（2.74 GiB），缺失 0 条。未全量解压 G1。原始路径保留为 `/home/qingyul/Datasets/bones_seed/g1/csv/<date>/<motion>.csv`；`recommended_csv` 中为 9 条优先动作的符号链接。

| 优先级 | 动作 | 文件 | 时长 | 预览 |
| --- | --- | --- | --- | --- |
| 01_single_knee | 单膝下跪祈祷后站起 | [prayer_R_001__A430](/home/qingyul/Datasets/bones_seed/g1/csv/230713/prayer_R_001__A430.csv) | 7.59 s | [MP4](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/previews/01_single_knee.mp4) |
| 02_double_knee | 双膝跪地后站起 | [dejection_R_001__A430](/home/qingyul/Datasets/bones_seed/g1/csv/230713/dejection_R_001__A430.csv) | 15.24 s | [MP4](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/previews/02_double_knee.mp4) |
| 03_crawl | 站立、四足后爬、恢复站立 | [show_bicep_R_on_all_fours_360_start_R_idle_right_R_002__A472](/home/qingyul/Datasets/bones_seed/g1/csv/231010/show_bicep_R_on_all_fours_360_start_R_idle_right_R_002__A472.csv) | 15.57 s | [MP4](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/previews/03_crawl.mp4) |
| 04_inchworm | 站立、手向前爬至平板支撑、收回站起 | [inchworm_001__A359](/home/qingyul/Datasets/bones_seed/g1/csv/230509/inchworm_001__A359.csv) | 7.08 s | [MP4](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/previews/04_inchworm.mp4) |
| 05_burpee | 站立开始、三次波比跳、回到站立 | [burpee_001__A361](/home/qingyul/Datasets/bones_seed/g1/csv/230509/burpee_001__A361.csv) | 10.70 s | [MP4](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/previews/05_burpee.mp4) |
| 06_light_jump | 站立、原地轻跳、落地站立 | [jump_and_land_light_002__A001](/home/qingyul/Datasets/bones_seed/g1/csv/210531/jump_and_land_light_002__A001.csv) | 8.75 s | [MP4](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/previews/06_light_jump.mp4) |
| 07_broad_jump | 站立、立定跳远、落地站立 | [broad_jump_002__A359](/home/qingyul/Datasets/bones_seed/g1/csv/230509/broad_jump_002__A359.csv) | 5.25 s | [MP4](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/previews/07_broad_jump.mp4) |
| 08_jump_twice | 站立、连续向前两跳、站稳 | [jump_twice_R_002__A533](/home/qingyul/Datasets/bones_seed/g1/csv/240327/jump_twice_R_002__A533.csv) | 3.27 s | [MP4](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/previews/08_jump_twice.mp4) |
| 09_high_jump | 站立、原地高跳、落地站立 | [high_jump_R_opt_2_001__A477](/home/qingyul/Datasets/bones_seed/g1/csv/231011/high_jump_R_opt_2_001__A477.csv) | 3.11 s | [MP4](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/previews/09_high_jump.mp4) |

建议先试单膝祈祷、双膝跪地和轻跳，再试四足爬行、inchworm、立定跳远、连续两跳、高跳和波比跳。这个顺序按参考动作复杂度安排，未测量策略成功率。

单膝祈祷保持单膝低位约 4.13 s；双膝跪地保持双膝低位约 7.66 s；四足后爬动作包含约 6.19 s 的躯干前倾、双手低位姿态，标注中的后爬段约为 5.0–10.4 s，随后恢复站立。inchworm 是双手爬出到平板支撑后收回站起；波比跳是三次循环，作为扩展候选。

跳跃的实测指标如下。“双脚离地高度”使用 G1 足底碰撞球最低点，取两脚中较低的一脚的峰值，并非骨盆抬升高度。离地时长按足底高于地面 4 cm 且膝、手、骨盆不处于地面支撑姿态计数，因此是带阈值的运动学指标。

| 跳跃 | 起止骨盆高度 | 双脚离地高度峰值 | 最长连续离地时间 |
| --- | --- | --- | --- |
| 站立、原地轻跳、落地站立 | 0.783 → 0.782 m | 9.1 cm | 0.192 s |
| 站立、立定跳远、落地站立 | 0.783 → 0.779 m | 22.0 cm | 0.375 s |
| 站立、连续向前两跳、站稳 | 0.777 → 0.779 m | 18.1 cm | 0.183 s |
| 站立、原地高跳、落地站立 | 0.786 → 0.773 m | 29.0 cm | 0.250 s |

全部 10,769 条完成关节/骨盆轨迹初筛，其中 6,534 条通过首尾姿态初筛；这不意味着它们都属于本次需要的动作。随后选取 229 条涵盖跪地、爬行及多类跳跃的候选，逐帧进行 120 Hz G1 正向运动学核验，216 条通过首尾站立检查；最终从中挑选 9 条优先动作和 11 条备选。未对其他初筛候选作完整运动学认证。

优先动作的首尾各连续 30 帧（120 Hz 下为 0.25 s）均满足：骨盆高度 ≥ 0.70 m；骨盆相对竖直倾角 ≤ 20°；两膝绝对角度 ≤ 35°；骨盆速度 ≤ 0.25 m/s；躯干倾角 ≤ 25°；双脚各自最低足底碰撞球距地面绝对值 ≤ 3.5 cm；双脚踝速度 ≤ 0.25 m/s。整个窗口每帧都必须满足，而不只看第一帧与最后一帧。阈值是本次筛选约定，不是部署系统已有的强制标准。

低位姿态辅助检查：双膝低位要求两膝关节位置高度 < 0.16 m 且骨盆 < 0.55 m；单膝低位要求低侧膝 < 0.16 m、另一侧膝 > 0.22 m、骨盆 < 0.60 m；手撑地姿态要求躯干倾角 > 55°、双腕高度 < 0.30 m、骨盆 < 0.60 m。动作语义再与时间标注和渲染关键帧交叉检查。这些几何条件不能代替接触力验证。

原始 CSV 为 120 Hz，根位置单位 cm，根欧拉角与关节角单位 degree；根旋转按现有转换器的 SciPy `Rotation.from_euler("xyz", ...)` 解码，关节根据列名映射至 G1 MuJoCo 模型。优先 9 条在该模型的关节位置限制内。足底局部误差仍存在，例如立定跳远中最低足底碰撞球约 -2.3 cm，故预览和阈值通过不能证明物理接触完全合理。

`kneeling_start/stop/loop`、`crawl_ff_*`、`spider_crawl` 等名字不构成入选依据。10 条 `spider_crawl` 和 600 条 `crawl_ff_*` 均未通过首尾姿态初筛，不能直接当作站立起止的完整 demo。

这批文件是原始 G1 动作数据，并非部署程序直接读取的 `joint_pos/body_pos/...` 多文件 bundle。导入当前 H2 sim2sim 时，需要按目标策略支持的参考类型正确转换；不能将原始 G1 CSV 直接标记成 H2 原生参考。所有原始采样保留，本次没有转换或替换现有部署 demo。

备选清单：

| 文件 | 时长 |
| --- | --- |
| [prayer_R_001__A428](/home/qingyul/Datasets/bones_seed/g1/csv/230713/prayer_R_001__A428.csv) | 5.42 s |
| [prayer_R_001__A429](/home/qingyul/Datasets/bones_seed/g1/csv/230713/prayer_R_001__A429.csv) | 6.12 s |
| [prayer_R_001__A432](/home/qingyul/Datasets/bones_seed/g1/csv/230713/prayer_R_001__A432.csv) | 5.96 s |
| [dejection_R_001__A431](/home/qingyul/Datasets/bones_seed/g1/csv/230713/dejection_R_001__A431.csv) | 14.69 s |
| [desperate_prayer_R_001__A430](/home/qingyul/Datasets/bones_seed/g1/csv/230713/desperate_prayer_R_001__A430.csv) | 11.72 s |
| [show_bicep_R_on_all_fours_360_start_R_idle_right_R_002__A475](/home/qingyul/Datasets/bones_seed/g1/csv/231010/show_bicep_R_on_all_fours_360_start_R_idle_right_R_002__A475.csv) | 15.57 s |
| [inchworm_003__A359](/home/qingyul/Datasets/bones_seed/g1/csv/230509/inchworm_003__A359.csv) | 7.97 s |
| [jump_and_land_light_003__A001](/home/qingyul/Datasets/bones_seed/g1/csv/210531/jump_and_land_light_003__A001.csv) | 9.74 s |
| [jump_for_joy_R_001__A429](/home/qingyul/Datasets/bones_seed/g1/csv/230713/jump_for_joy_R_001__A429.csv) | 6.83 s |
| [broad_jump_003__A362](/home/qingyul/Datasets/bones_seed/g1/csv/230509/broad_jump_003__A362.csv) | 4.42 s |
| [high_jump_R_003__A371](/home/qingyul/Datasets/bones_seed/g1/csv/230516/high_jump_R_003__A371.csv) | 5.53 s |

![跪地与爬行动作关键帧](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/previews/ground_overview.jpg)

![跳跃动作关键帧](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/previews/jump_overview.jpg)

所有 MP4 均是原始参考轨迹的运动学回放，视频为 30 fps，仅统一可视化平面朝向与原点，镜头跟随骨盆；不是策略输出，也不施加动力学。统计及原始 CSV 保持 120 Hz、原始世界坐标。

结果与复现文件：

- [recommendations.json](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/recommendations.json)：9 条优先动作、11 条备选、源路径、SHA-256、时间标注和核验指标
- [recommendations.csv](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/recommendations.csv)：便于表格查看的优先清单
- [recommended_csv](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/recommended_csv)：优先动作原始 CSV 的符号链接目录
- [existing_demo_audit.json](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/existing_demo_audit.json)：现有 27 条 demo 的首尾姿态检查
- [extraction_manifest.json](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/extraction_manifest.json)：10,769 条已提取文件的路径、大小和 SHA-256
- [kinematic_screen.csv](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/kinematic_screen.csv)：全部提取动作的初筛结果及可裁剪站立窗口
- [verified_candidates.json](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/verified_candidates.json)：229 条候选的逐帧运动学核验结果
- [extract_candidates.py](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/extract_candidates.py)：定向解压脚本，可重复运行并核对已存在文件
- [screen_candidates.py](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/screen_candidates.py)：全量初筛脚本
- [verify_candidates.py](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/verify_candidates.py)：G1 正向运动学核验脚本
- [package_selection.py](/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/package_selection.py)：优先清单、视频与关键帧生成脚本

运行顺序为上述四个 Python 脚本，使用系统 `python3`。各脚本依赖本机数据路径、仓库 G1 模型及已有渲染辅助代码。
