# 四套 Unitree H2 模型完整差异审计

生成日期：2026-07-15；最近复核：2026-07-19

工程目录：`/home/qingyul/workspace/SONIC-H2`

审计目的：为 H2LoRA sim2sim/sim2real 模型选择、站立前倾问题定位和后续真机参数辨识建立可重复核对的模型基线。

## 对比模型定义

本文从此处开始固定使用以下名称和代词。后文所有表格、结论和附录均遵循该定义，不再用临时缩写替代。

| 正式名称 | 模型代词 | 完整定义 | 主要文件 |
|---|---|---|---|
| H2LoRA 训练模型 | **模型 A** | 本工程 H2LoRA 在 IsaacLab/PhysX 训练时实际使用的 H2 robot asset，包括源 URDF 和 `H2_CFG` 对执行器、初始姿态、碰撞导入和求解参数的覆盖 | `gear_sonic/data/assets/robot_description/urdf/h2/h2.urdf`；`gear_sonic/envs/manager_env/robots/h2.py` |
| Unitree MuJoCo H2 | **模型 B** | 从 Unitree `unitree_mujoco` 仓库导入的官方 H2 MJCF 源模型；讨论本工程 sim2sim 时，再单独说明运行时 WBC 覆盖 | `gear_sonic/data/assets/robot_description/unitree_mujoco/h2/h2_mujoco.xml` |
| Unitree ROS H2 | **模型 C** | Unitree `unitree_ros` 中的 H2 URDF 家族。正文 A/B/C/D 统一数值表中的 C 特指基础 STL 版本 `H2.urdf`；`H2_dae.urdf` 和 `H2_Plus.urdf` 作为 C 的子版本在第 17 节单独审计 | `gear_sonic/data/assets/robot_description/unitree_ros/h2_description/H2.urdf`；上游 `robots/h2_plus/H2_Plus.urdf` |
| Unitree RL MjLab H2 训练模型 | **模型 D** | `/home/qingyul/workspace/unitree_rl_mjlab` 的 velocity/tracking 训练任务通过 `get_h2_robot_cfg()` 实际加载的 H2 MJCF，加上 MjLab position actuator、collision 和 initial-state 配置 | `/home/qingyul/workspace/unitree_rl_mjlab/src/assets/robots/unitree_h2/xmls/h2.xml`；同目录上级 `h2_constants.py` |

需要特别注意：

- 模型 A/C 的源格式是 URDF，模型 B/D 的源格式是 MJCF。
- 模型 A/B/C 的基础 H2 表示 31 个活动关节；模型 C 的 Plus 子版本另有 44 个手指关节，共 75 个活动关节；模型 D 把两个 head link 固定，只保留 29 个活动关节。
- “模型 B”只表示 Unitree MuJoCo 官方源 robot XML，不等于本工程覆盖 torque limit/armature/PD 后的完整 sim2sim 环境。
- “模型 D”表示 MjLab 训练的完整 robot 配置，不只表示 `h2.xml` 文件；源 XML 本身没有 actuator，29 个 position actuator 由 `h2_constants.py` 添加。
- 本文不把实验派生模型 `h2_lora_unitree_official_ros_mass`、`h2_lora_unitree_official_ros_mass_debug` 另立为第五、第六个源模型。

模型 C 家族在本文中进一步固定为：

- **C-STL**：`h2_description/H2.urdf`，34 个 visual 和全部 mesh collision 引用 STL；原四模型表中的“模型 C”默认就是它。
- **C-DAE**：`h2_description/H2_dae.urdf`，只把 34 个 visual 换成 DAE，动力学和 collision 与 C-STL 相同。
- **C-Plus**：`h2_plus/H2_Plus.urdf`，基础 H2 机身加双侧 Sharpa 灵巧手；它不是 C-STL 的纯 visual 变体。

### 四模型一页速览

下表先给出四套模型的关系主线，后续章节再逐项给出原始数值和复核依据。

| 维度 | 模型 A | 模型 B | 模型 C | 模型 D |
|---|---|---|---|---|
| 主要用途 | H2LoRA IsaacLab/PhysX 训练 | Unitree MuJoCo 官方仿真资产；本工程也用于 sim2sim | Unitree ROS/Gazebo 描述资产 | Unitree RL MjLab velocity/tracking 训练 |
| 源 robot 名称 | `H1031` | `H2` | `H2` | `h2` |
| 源格式 | URDF + IsaacLab config | MJCF | URDF | MJCF + MjLab Entity config |
| 活动关节 | 31；H2LoRA policy 使用其中 29 维 | 31 | C-STL/C-DAE 为 31；C-Plus 为 75 | 29；head pitch/yaw 在 robot 层固定 |
| 总质量 | 72.424532 kg | 71.287850 kg | C-STL/C-DAE 为 75.587456 kg；C-Plus 为 77.238272 kg | 72.084580 kg |
| 惯性主线 | `H1031` 参数；腿/torso 比 C 轻，手臂比 C 重 | 除 pelvis 和 wrist-yaw 表示外基本接近 C | ROS 中显式给出的 H2 参数 | 主体几乎继承 A；清理部分 COM，并去掉 A 的两只 0.17 kg fixed hand |
| 运动学主线 | 与 B/C 存在毫米到厘米级局部差异 | 与 C 基本一致 | 与 B 基本一致 | joint range/axis 接近 A，累计足端和 torso 位置更接近 C，局部 frame 为独立重排 |
| 每脚地面接触 | 训练原始为完整 ankle-pitch mesh；A 派生 sim2sim 会换成 B 的 primitives | 6 sphere + 1 box | 完整 ankle-pitch mesh | 7 capsule |
| 源/训练执行器 | IsaacLab 31 个 implicit actuator | 源 MJCF 31 个 torque motor；本工程运行时覆盖 PD/limit | 源 URDF 无 actuator | MjLab 添加 29 个 builtin position actuator |
| 最关键风险 | 训练模型的质量分布、ankle 几何和训练/仿真接触不统一 | pelvis 惯性由 mesh 自动推导，明显偏轻 | 不能仅凭“官方 ROS”认定等于当前真机 | 混合模型稳定不代表任一单项参数更接近真机 |

可以把四者的关系简化为：**B 与 C 主要共享运动学；D 主要共享 A 的惯性和关节范围、共享 C 风格的累计骨架位置，但使用独立的 capsule 接触和 MjLab 执行器。**

## 1. 审计结论

本次对比得到以下主要结论。

1. **模型 A 不是模型 B/C 的简单副本。**模型 A 的 URDF 名为 `H1031`；模型 B/C 名为 `H2`。三者虽然都有 31 个活动关节，但质量、惯量、关节原点、碰撞体和 mesh 文件均存在实质差异。
2. **模型 B 实际来自模型 C 的历史快照，而不是独立的一套 H2 骨架。**B 与 ROS 2026-04-24 H2 快照的 32/32 个 STL 字节完全相同，31 个活动关节运动学完全相同；除 pelvis 外，31 个 body inertial 也只存在 MJCF 小数截断。
3. **B/C 真正显著的刚体差异集中在 pelvis。**模型 B 的 pelvis 没有显式 inertial，由 MuJoCo 从 mesh 推导为 `4.248093 kg`；模型 C 明确给出 `8.547700 kg`，相差 `4.299607 kg`。其他 Unitree MuJoCo 人形/四足模型的 root body 均有显式 inertial，因此这更像 H2 MJCF 转换遗漏，而不是一套有意发布的第二组 pelvis 标定；上游没有文字说明，仍应标记为高置信推断而非官方结论。
4. **模型 D 是一套混合参数模型，不等于模型 A、B 或 C。**其绝大多数 body mass、principal inertia 和 joint range 取自或数值上等同模型 A；累计足端、腰部和手臂运动学又更接近模型 C；碰撞则重新设计为 31 个 capsule + 2 个 sphere。
5. **模型 D 只有 29 个活动关节。**`head_pitch_link` 和 `head_yaw_link` 保留为固定 body，但没有 head joint/actuator；模型 A/B/C-STL/C-DAE 均为 31 个活动关节，C-Plus 则为 75 个。
6. **模型 D 没有模型 A 的两个 0.17 kg rubber hand inertial。**所以模型 D 总质量 `72.084580 kg`，比模型 A 少约 `0.339952 kg`；其 wrist-yaw body 每侧约 `0.605752 kg`，而模型 A 固定手掌折叠后为 `0.775752 kg`。
7. **模型 A 的质量误差不是统一缩放。**相对模型 C，模型 A 双腿轻 `2.134315 kg`、腰和躯干轻 `2.767480 kg`，双臂和手反而重 `2.681889 kg`。模型 D 基本继承这一质量分布，只去掉两只 fixed rubber hand。
8. **四者脚底接触模型分成三类。**模型 A 训练和模型 C 原始 URDF 使用完整 ankle-pitch mesh；模型 B 每脚使用 6 sphere + 1 box；模型 D 每脚使用 7 条 capsule。接触实现足以显著改变站立稳定性和小碎步触发条件。
9. **模型 A 的训练接触和当前 A 派生 sim2sim 接触并不相同。**`h2_lora_sonic.yaml` 会在运行时禁用 ankle mesh，并补模型 B 的 6 sphere + 1 box。因此 A 派生 sim2sim 正常不能证明训练 PhysX 接触与真机一致。
10. **当前模型 B sim2sim 会覆盖低层参数，但不会覆盖 robot XML 的刚体参数。**torque limit、armature、PD 和求解器可对齐模型 A；质量、COM、惯量、运动学和碰撞仍由模型 B 决定。
11. 在模型 A 默认站姿和统一关节角下，COM 相对左右 ankle-pitch 原点中点的前向距离分别为：模型 A `87.953 mm`、B `81.096 mm`、C `79.488 mm`、D `79.144 mm`。模型 D 的该项几乎与模型 C 相同，但它们的质量分布和 COM 高度仍不同。
12. **C-STL 与 C-DAE 是同一个物理模型。**两者 37 links、36 joints、全部 inertial、joint origin/axis/limit 和 23 个 collision 逐项完全相同；区别仅在 visual mesh 和材质标签。
13. **C-Plus 保留基础 H2 的 31 个机身关节和 32 个承载 link 参数，但双手是实质动力学扩展。**它新增 66 个手部 links、移除普通版 5 个 fixed links，净增 61 links；同时增加 44 个 revolute joint 和 `1.65081554 kg` 净质量，并使 H2LoRA 默认姿态下的整机 COM 相对 ankle reference 再向前移动约 `4.091 mm`。
14. **C-DAE 和 C-Plus 不能原样交给当前 MuJoCo 3.10 URDF importer。**源文件同一 `<visual>` 下存在多个 `<material>`，首先触发 `repeated element: material`；移除这些标签后，当前 MuJoCo 构建仍不支持 `.dae` decoder。该结论是 MuJoCo 导入兼容性结论，不等于已经证明 Gazebo 也无法加载。
15. **此前 B/C wrist-yaw 的 50-60 mm COM 差异是转换器伪差异。**当前 MuJoCo 3.10 在融合带惯性的 fixed child 时命中上游 `mjCBody::AccumulateInertia` frame composition bug；按 URDF 刚体公式手工合并后，当前 C 与 B 的左右 wrist-yaw COM 只差 `0.172/0.527 mm`。模型 A 的 fixed rubber hand 也会触发同一路径，但仓库中已有的 A MJCF 已保存正确复合惯性；风险出现在用当前 importer 重新生成时。
16. **此前 MuJoCo bridge 把 pelvis 惯性主轴角误当成了 IMU 坐标变换。**`mj_objectVelocity(..., mjOBJ_BODY, local=1)` 返回惯性主轴 frame 的速度，不是 link frame；模型 C 的 pelvis `iquat` 近似绕 X 轴 90 度，因此 policy 收到的 gyro 近似为 `[omega_x, omega_z, -omega_y]`。模型 A、B 和 `ros_mass_debug` 的 pelvis `iquat` 接近单位四元数，错误被偶然隐藏。2026-07-19 已改成取 world velocity 后显式用 `xmat.T` 转到 link frame。

因此，不能把当前问题归结为单一 `action scale`、单一 ankle pitch 零点或单一 PD 静差。模型侧至少同时存在质量分布、COM/惯量、ankle 几何、腰部几何、足底接触和低层执行器模型六类差异。

## 2. 比较口径与方法

### 2.1 为什么 A 必须包含 `H2_CFG`

训练阶段不是只读取 URDF。`H2_CFG` 还会覆盖或新增：

- floating base；
- PhysX 刚体和 articulation 求解参数；
- cylinder 到 capsule 的替换；
- 自碰撞；
- 初始姿态；
- 各关节 armature、Kp、Kd、effort limit 和 velocity limit；
- action scale；
- contact sensor；
- 训练时物理材质随机化。

所以本文把“训练模型”定义为“训练 URDF + 实际训练配置”，而不是把 URDF 的 `<limit effort>` 当成最终训练扭矩上限。

### 2.2 为什么 B 和 C 需要编译到统一表示再比较

B 是 MJCF；A/C 是 URDF。直接比较 XML 标签会产生以下误判：

- URDF 的固定 hand、camera、IMU link 会在 MuJoCo 导入时折叠到活动父 body；
- URDF 用完整惯量矩阵表示惯性，MJCF 常用 `diaginertia + iquat` 表示同一物理量；
- mesh 导入后，MuJoCo 会把 mesh reference frame 规范化，`geom pos` 不再等于原始 URDF collision origin；
- A/C 源 URDF 中 floating-base joint 被注释，训练器或生成脚本会在运行时补回。

本次审计最初使用 `scripts/build_h2_unitree_mujoco_ros_mass.py::compile_ros_urdf()` 给 A/C 临时增加 floating base，然后通过 MuJoCo 编译结果比较 body、joint、mass、COM、principal inertia、`iquat` 和 collision geom。进一步复核发现，MuJoCo 3.10 的 fixed-body inertia fusion 命中上游未合并修复 [PR #3250](https://github.com/google-deepmind/mujoco/pull/3250)：fixed child 的 body pose 和 inertial offset 组合顺序错误。A 的 rubber hand、C 的普通 fixed hand 以及 C-Plus 的固定手部结构都会触发该问题。

因此本文修正后的比较口径为：

- 活动 body、joint 和 collision 仍使用统一 MuJoCo 编译结果；
- A/C 的 fixed-hand 质量、COM 和完整惯量矩阵直接按 URDF 刚体变换和平行轴定理合并，不采用 MuJoCo 3.10 的错误融合结果；
- A/C/C-Plus 的整机 COM 直接按 URDF 关节树前向运动学和 link mass 加权计算；
- 文中若特指本工程当前生成文件的错误数值，会明确写成“MuJoCo 3.10 importer output”。

临时文件不会改变源资产。仓库内已有的 `mjcf/h2.xml` 对模型 A 的 fixed rubber hand 合成结果正确。2026-07-19 起，C 的直接 MJCF 和两个 ROS-mass 派生 MJCF 已改为按 URDF 刚体公式自行合成 fixed-link 惯量，并通过源 COM/完整惯量矩阵回归校验；不再采用 MuJoCo 3.10 importer 的错误 wrist 输出。

### 2.3 模型 D 的两层编译口径

模型 D 必须区分两层：

1. `h2.xml` 源 MJCF：包含 body、joint、inertial、visual、collision、site 和 4 个基础 sensor，但不包含 actuator。
2. MjLab 实际训练 robot：`get_h2_robot_cfg()` 通过 `Entity` 添加 29 个 `BuiltinPositionActuator`，应用 `FULL_COLLISION`、默认姿态、soft joint limit 和 action scale，再编译为训练模型。

本文的质量、运动学和源 collision XML 比较使用模型 D 源 MJCF；执行器、有效 collision 参数和 action scale 使用 MjLab `Entity(get_h2_robot_cfg()).spec.compile()` 的结果。四模型统一数值表使用 MuJoCo `3.10.0` 编译；模型 D 的 MjLab `.venv` 当前使用 MuJoCo `3.5.0`，两版编译所得总质量和拓扑一致。

`unitree_rl_mjlab` 当前 HEAD 为 `1425b15f73bd4095f0df53709d7c389c3eb9e790`。模型 D 的 asset 和 `h2_constants.py` 本身工作树干净；当前仓库另有用户未提交的 H2 tracking 任务文件，它们复用同一个 `get_h2_robot_cfg()`，本文不会把这些任务代码改动误写成 robot asset 改动。

## 3. 来源、版本和可复现标识

| 项目 | 模型 A | 模型 B | 模型 C | 模型 D |
|---|---|---|---|---|
| XML robot/model name | `H1031` | `H2` | `H2` | `h2` |
| 源格式 | URDF | MJCF | URDF | MJCF + MjLab Entity config |
| 本地源文件 SHA-256 | `be350690...958e3a0f` | `2ac63ffc...b69c47f` | `32866f6f...c741401` | `17b4e16a...602273` |
| 完整 SHA-256 | `be3506900e3eab3a1bc75e079881faa3c7991496cad8a8457b5b90ba958e3a0f` | `2ac63ffc436fd4a368a448acacfd4f0915e7f986596b254fa681b6a95b69c47f` | `32866f6f2dae8ed79dc622b015160f4f8fc018ce266ce4f84322084ddc741401` | `17b4e16ad12cbda1ac97516f7242f2986d0fdd5dbd66bb830da27b7a37602273` |
| 上游来源 | 当前 SONIC-H2 仓库资产 | `unitreerobotics/unitree_mujoco` | `unitreerobotics/unitree_ros` | `unitreerobotics/unitree_rl_mjlab` |
| 上游/本仓库记录提交 | `daf389964fa4a4545218e8405f24eb55f4912453` | `ae6a8403e272733e9996ef59990880330496177f` | `d96d8f63ae17a7108d4f7229c00ef875ba7129c9` | asset 首次提交 `7e09b5031a2efe1086c6d9d8d8af8204722ada2c`；当前仓库 HEAD `1425b15f...` |
| mesh 文件 | 37 STL | 32 STL | 34 STL + 34 DAE | 32 STL |
| 本地目录 | 本工程内 | 本工程内 | 本工程内 | `/home/qingyul/workspace/unitree_rl_mjlab/src/assets/robots/unitree_h2` |

说明：模型 A 表中的提交是该资产进入当前仓库的提交，不代表已经证明它来自某个 Unitree 上游版本。模型 B/C 的来源和提交记录在各自 `SOURCE.txt` 中。模型 D 的 asset 由 `unitree_rl_mjlab` 提交 `7e09b503` 加入，当前 asset 没有本地修改。

本次数值编译环境：Python `3.12.3`、MuJoCo `3.10.0`、NumPy `1.26.4`。MuJoCo 版本变化可能造成 mesh 自动惯性和主轴特征分解末位变化，复核时应同时记录版本。

## 4. 总体模型规模

把模型 A/C 编译成带 floating base 的 MuJoCo 模型，并把模型 D 通过 MjLab robot config 编译后：

| 指标 | 模型 A | 模型 B | 模型 C | 模型 D |
|---|---:|---:|---:|---:|
| source link/body 数 | 34 links | 32 robot bodies + world | 37 links | 32 robot bodies + world |
| source joint 数 | 33 | 32，含 free root | 36 | 30，含 free root |
| 活动 hinge/revolute joint | 31 | 31 | 31 | 29 |
| 固定/无 joint body | 2 fixed hands | 固定结构已折叠/直接建模 | 5 fixed links | head pitch/yaw 两个 body 固定在 torso 链上 |
| 编译后 `nbody` | 33 | 33 | 33 | 33 |
| 编译后 `njnt` | 32 | 32 | 32 | 30 |
| 编译后 `nq` | 38 | 38 | 38 | 36 |
| 编译后 `nv` | 37 | 37 | 37 | 35 |
| 编译后 `ngeom` | 57 | 65 | 57 | 65 |
| 编译后 `nmesh` | 34 | 32 | 34 | 32 |
| 有效 collision geom | 23 | 33 | 23 | 33 |
| actuator 数 | IsaacLab 添加 31 个 implicit actuator | 源 MJCF 31 个 torque motor | 源 URDF 0 | 源 MJCF 0；MjLab 添加 29 个 position actuator |
| 源/内建 sensor | URDF 0；训练启用 contact sensor | 101 | URDF 0 | 4，合计 12 维；任务另加 contact sensor |
| 总质量 | 72.42453191 kg | 71.28785032 kg | 75.58745630 kg | 72.08458040 kg |

模型 A/B/C 的物理资产都保留 31 个活动关节。H2LoRA policy 对模型 A 使用 29 维接口时，两个 head joint 仍由 policy 外默认角控制。模型 D 则是在 robot MJCF 层面直接取消两个 head joint，因此其 29 维不是接口裁剪，而是物理 articulation 本身只有 29 个 hinge joint。

## 5. 拓扑和固定 link

### 5.1 活动关节父子拓扑

四者的腿、腰和手臂活动链语义一致：

- pelvis 分出左腿、右腿和 waist-yaw 三条链；
- 每条腿为 hip-pitch -> hip-roll -> hip-yaw -> knee -> ankle-roll -> ankle-pitch；
- 腰部为 waist-yaw -> waist-roll -> waist-pitch/torso；
- torso 分出 head、左臂和右臂；
- 每条手臂为 shoulder-pitch -> shoulder-roll -> shoulder-yaw -> elbow -> wrist-roll -> wrist-pitch -> wrist-yaw。

模型 A/B/C 的 head 链为 `head-pitch joint -> head-yaw joint`，两关节均活动。模型 D 保留 `head_pitch_link -> head_yaw_link` 两个 body 和对应 visual/inertial，但没有 `<joint>`，所以二者相对父 body 固定。

### 5.2 固定 link 差异

| 模型 | fixed joint | parent -> child | child mass | 导入后的处理 |
|---|---|---|---:|---|
| A | `left_hand_palm_joint` | `left_wrist_yaw_link -> left_rubber_hand` | 0.170 kg | 折叠到左 wrist-yaw body |
| A | `right_hand_palm_joint` | `right_wrist_yaw_link -> right_rubber_hand` | 0.170 kg | 折叠到右 wrist-yaw body |
| C | `camera` | `head_yaw_link -> camera` | 无 inertial | 作为零质量固定结构折叠 |
| C | `imu_in_torso_joint` | `torso_link -> imu_in_torso` | 无 inertial | 作为 site/固定坐标语义使用 |
| C | `imu_in_pelvis_joint` | `pelvis -> imu_in_pelvis` | 无 inertial | 作为 site/固定坐标语义使用 |
| C | `left_hand_joint` | `left_wrist_yaw_link -> left_hand_link` | 0.41885723 kg | 应按刚体公式折叠到左 wrist-yaw；MuJoCo 3.10 importer 当前算错 COM frame |
| C | `right_hand_joint` | `right_wrist_yaw_link -> right_hand_link` | 0.41885723 kg | 应按刚体公式折叠到右 wrist-yaw；MuJoCo 3.10 importer 当前算错 COM frame |
| D | 无独立 fixed joint 标签 | `torso -> head_pitch_link -> head_yaw_link` | 1.82 + 1.30 kg | 两个 head body 直接固定嵌套，无 head actuator |
| D | 无 hand body | wrist-yaw body 后只有 palm site 和 hand collision capsule | 0 kg 独立 hand mass | wrist-yaw 仅保留自身 0.605752 kg |

A 的 wrist-yaw 源 link 质量为 `0.60575230 kg`，加固定 rubber hand 后为 `0.77575230 kg`。C 的 wrist-yaw 源 link 质量为 `0.19722487 kg`，加固定 hand 后为 `0.61608210 kg`。

C 在 fixed hand joint 上写了非标准属性 `dont_collapse="true"`，MuJoCo URDF importer 不保留该 fixed body，仍会将其惯性合并到 wrist-yaw。移除该属性或设成 `false` 均不会改变结果。需要注意，当前 MuJoCo 3.10 合并后的 wrist COM/inertia 是错误的；比较必须使用源 URDF 手工合成值，而不是直接读取 `compile_ros_urdf()` 的 wrist `body_ipos/body_inertia/body_iquat`。

模型 D 没有模型 A 的 `left/right_rubber_hand` body，也没有模型 C 的 `left/right_hand_link` body。它用腕部末端 capsule 提供手部碰撞范围，但该 capsule 没有质量，不能等价替代 fixed hand inertial。

### 5.3 joint/qpos 顺序

模型 A/B/C 编译后的 MuJoCo joint/qpos 顺序一致：

```text
floating_base,
left leg 6, right leg 6,
waist yaw/roll/pitch,
head pitch/yaw,
left arm 7, right arm 7
```

B 的 actuator 顺序不同：

```text
left leg 6, right leg 6,
waist yaw/roll/pitch,
left arm 7, right arm 7,
head pitch/yaw
```

也就是说 B 的 qpos 中 head 位于索引 15/16，而 motor actuator 中 head 位于最后 29/30。当前 `base_sim.py` 按 joint name 构建 joint -> actuator 映射，不能简单假设 qpos 和 actuator 是逐位置 identity。

IsaacLab/PhysX 的 articulation joint 顺序又与 MuJoCo DFS 顺序不同，工程通过 `H2_ISAACLAB_TO_MUJOCO_DOF` 和逆映射显式处理。这个顺序差异是接口问题，不是运动学差异。

模型 D 的 qpos 顺序为：

```text
floating_base,
left leg 6, right leg 6,
waist yaw/roll/pitch,
left arm 7, right arm 7
```

模型 D 没有 head qpos；MjLab 添加的 actuator 按配置匹配后共 29 个。其 actuator 编译顺序不是纯 body DFS：腿 8 个 hip/knee、ankle roll 2、ankle pitch 2、waist 3、两侧主手臂 10、两侧 wrist pitch/yaw 4。依赖数组索引的部署代码必须使用 joint name 映射，不能直接复用模型 A/B/C 的 31-DoF 顺序。

## 6. 质量分布

### 6.1 区域汇总

| 区域 | 模型 A | 模型 B | 模型 C | 模型 D | 关系分析 |
|---|---:|---:|---:|---:|---|
| pelvis | 7.604681 | 4.248093 | 8.547700 | 7.604680 | D≈A；B 比 C 少 4.299607 kg |
| 双腿 | 30.067487 | 32.201797 | 32.201802 | 30.067490 | D≈A；A/D 比 C 轻约 2.134 kg |
| waist + torso | 16.215876 | 18.983355 | 18.983355 | 16.215912 | D≈A；A/D 比 C 轻约 2.767 kg |
| head | 3.120000 | 3.120000 | 3.120000 | 3.120000 | 质量相同；D 的 head 固定 |
| 双臂 + hand 表示 | 15.416489 | 12.734605 | 12.734600 | 15.076498 | D 接近 A，但少两只各 0.17 kg 的 rubber hand |
| 总计 | 72.424532 | 71.287850 | 75.587456 | 72.084580 | D 比 A 少 0.339952 kg；C 最重 |

### 6.2 逐活动 body 质量

单位均为 kg。L/R 合并行表示左右相同或仅存在末位舍入；明确不对称的 body 分开记录。

| Body | 模型 A | 模型 B | 模型 C | 模型 D | 主要差异 |
|---|---:|---:|---:|---:|---|
| pelvis | 7.604681 | 4.248093 | 8.547700 | 7.604680 | D≈A；B 比 C 少 4.299607 |
| hip-pitch L/R，各 | 2.797525 | 2.873700 | 2.873700 | 2.797530 | D≈A；A/D 每侧比 C 少约 0.07617 |
| left hip-roll | 0.627146 | 0.785079 | 0.785079 | 0.627146 | D=A；A/D 比 C 少 0.157933 |
| right hip-roll | 0.627146 | 0.751220 | 0.751220 | 0.627146 | D=A；A/D 把左右质量设成相同 |
| hip-yaw L/R，各 | 7.087520 | 7.087500 | 7.087500 | 7.087520 | 四者仅末位舍入 |
| knee L/R，各 | 3.553402 | 4.353400 | 4.353402 | 3.553400 | D≈A；每侧比 C 少约 0.800000 |
| ankle-roll L/R，各 | 0.123182 | 0.123182 | 0.123182 | 0.123182 | 一致 |
| ankle-pitch L/R，各 | 0.844967 | 0.894967 | 0.894967 | 0.844967 | D=A；每侧比 C 少 0.050000 |
| waist-yaw | 0.279762 | 0.312936 | 0.312936 | 0.279762 | D=A |
| waist-roll | 0.065850 | 0.138419 | 0.138419 | 0.065850 | D=A，约为 C 的 47.6% |
| torso | 15.870263 | 18.532000 | 18.532000 | 15.870300 | D≈A；比 C 少约 2.662 kg |
| head-pitch | 1.820000 | 1.820000 | 1.820000 | 1.820000 | 质量一致；D 固定 |
| head-yaw | 1.300000 | 1.300000 | 1.300000 | 1.300000 | 质量一致；D 固定 |
| shoulder-pitch L/R，各 | 1.060556 | 1.192700 | 1.192700 | 1.060560 | D≈A |
| shoulder-roll L/R，各 | 1.345111 | 约 1.109825 | 约 1.109824 | 1.345110 | D≈A；每侧比 C 多约 0.235286 |
| shoulder-yaw L/R，各 | 0.915717 | 约 0.352741 | 约 0.352740 | 0.915717 | D=A，约为 C 的 2.60 倍 |
| elbow L/R，各 | 1.694319 | 约 1.706145 | 约 1.706144 | 1.694320 | D≈A |
| wrist-roll L/R，各 | 0.969837 | 约 0.774263 | 约 0.774262 | 0.969837 | D=A |
| wrist-pitch L/R，各 | 0.946953 | 0.615258 / 0.615837 | 0.615258 / 0.615837 | 0.946953 | D=A；每侧比 C 多约 0.331 kg |
| wrist-yaw + hand 表示 L/R，各 | 0.775752 | 0.616082 | 0.616082 | 0.605752 | D 比 A 每侧少 0.17；D 比 C 每侧少 0.01033 |

### 6.3 质量差异的动力学含义

- A 的 torso 更轻，意味着在相同腰/髋姿态下，训练中的 hip-pitch、ankle-pitch 和 waist-pitch 抵抗上身重力的静态力矩偏小。
- A 的手臂总质量更大，但质量被分配到与 C 不同的 shoulder/wrist body 上，因此手臂动作产生的反作用力矩和惯性耦合也不同。
- A 的双膝合计少 `1.6 kg`，会降低膝关节附近的转动惯量和腿部摆动负担。
- B 的 pelvis 仅 `4.248 kg`，相对 C 少 `50.3%`。该值来自 mesh 自动推导，不应直接视作 Unitree 对真实 pelvis 的标定值。
- 仅把 B 的各 body mass 换成 C 的质量仍然不够；COM、惯量矩阵和固定 hand 折叠后的惯量也要一起处理，否则动力学不自洽。
- D 的主体质量分布几乎完整继承 A，所以 D 在 MjLab 中表现稳定不能证明 C/真机的质量模型已经被验证；D 的运动学和 capsule 接触可以抵消部分 A 惯性偏差。
- D 使用 hand collision capsule 表示末端外形，却没有对应 hand mass。对于站立静态影响较小，但对快速摆臂、全身 tracking 和腕部反作用力矩有直接影响。

## 7. COM、主惯量和 `iquat`

### 7.1 表示方法

MuJoCo 对每个 body 使用：

- `body_ipos`：body 坐标系下的质心位置；
- `body_inertia`：惯性主轴坐标系下的三个主惯量；
- `body_iquat`：从 body 坐标系到惯性主轴坐标系的四元数，顺序为 `w x y z`。

等价的 body 坐标系完整惯量矩阵为：

```text
I_body = R(iquat) * diag(principal_inertia) * R(iquat)^T
```

`iquat` 本身不是额外的物理量。若两个主惯量接近，特征向量可以交换、翻转或绕近简并轴旋转，所以仅比较四元数角度可能夸大差异。判断是否等价应同时比较 COM 和重建后的完整惯量矩阵。附录 3/5 保留模型 A/B/C 的 principal inertia 和 `iquat`；其中 C 的 wrist-yaw 使用源 URDF 手工合成结果，附录 4 记录模型 D，以便逐值复核。

### 7.2 A 相对 C 的主要惯性差异

| Body | COM 距离差 | 最大主惯量绝对差 | 主轴四元数表观夹角 | 说明 |
|---|---:|---:|---:|---|
| pelvis | 0.284 mm | 0.000297 kg·m² | 2.079° | 质量差更显著 |
| hip-pitch L/R | 1.807 mm | 0.000271 | 2.794° | 下肢近端差异 |
| hip-roll L/R | 13.150 mm | 0.000375 | 9.370° | 对髋部耦合较敏感 |
| knee L/R | 1.983 mm | 约 0 | 约 0° | 主要是质量和 COM 差异 |
| ankle-pitch L/R | 约 0 | 约 0 | 约 0° | 惯量形状相同，但 A 少 0.05 kg 且 joint origin 不同 |
| waist-yaw | 0.672 mm | 0.000037 | 0.388° | 小但非零 |
| waist-roll | 0.674 mm | 0.000007 | 表观 90° | 主惯量接近，90° 主要是主轴置换，不应单独解读 |
| torso | 7.915 mm | 0.018397 | 1.959° | 上身动力学的重要差异 |
| head-pitch | 20.702 mm | 约 0 | 约 0° | 同质量/惯量、不同 COM 坐标 |
| head-yaw | 23.691 mm | 约 0 | 约 0° | 同质量/惯量、不同 COM 坐标 |
| shoulder-pitch | 6.077 mm | 0.000527 | 约 6.38° | 质量和惯量均不同 |
| shoulder-roll | 19.672 mm | 0.000220 | 约 15.78° | 明显不同 |
| shoulder-yaw | 7.411 mm | 0.002165 | 约 88-98° | A 质量约为 C 的 2.6 倍 |
| elbow | 约 8.18 mm | 0.000424 | 约 64-116° | 左右四元数符号/轴排列也影响表观角 |
| wrist-roll | 约 4.50 mm | 0.000243 | 约 75-105° | 惯量与几何链不同 |
| wrist-pitch | 约 79.21 mm | 0.000495 | 约 134-140° | 局部 link 定义发生了大幅变化 |
| wrist-yaw L/R | 74.866 / 75.199 mm | 0.000306 / 0.000295 | 不单独解读 | A/C 均使用源 URDF fixed-hand 刚体合成；旧值来自 MuJoCo importer bug |

### 7.3 B 相对 C 的例外

B 与当前 C 源参数的实质惯性差异如下：

| Body | mass 差 | COM 距离 | principal inertia 最大差 | 判断 |
|---|---:|---:|---:|---|
| `pelvis` | 4.299607 kg | 11.390 mm | 0.018922 kg·m² | B 缺少显式 inertial，改由同一 pelvis mesh 自动推导 |
| `left_wrist_yaw_link + hand` | 0.0000001 kg | 0.172 mm | 0.00001086 kg·m² | C 在 2026-05-19 拆分手部后有很小参数更新 |
| `right_wrist_yaw_link + hand` | 0.0000001 kg | 0.527 mm | 0.00000017 kg·m² | 同上 |

除 pelvis 和上述拆手后的亚毫米更新外，B 的其他 29 个活动 child body 与 C 2026-04-24 快照只存在 XML 舍入。以全部 31 个非 pelvis body 统计，最大 mass 截断为 `4.44e-6 kg`、最大 COM 截断为 `0.441 µm`、完整惯量矩阵最大元素差为 `4.31e-7 kg·m²`。

MuJoCo 3.10 importer 直接编译当前 C 时，会把左右 wrist-yaw COM 错算到距 B `50.205/60.510 mm` 的位置。这不是 Unitree B/C 源资产差异，而是 fixed-link inertia fusion bug；对应最小复现和上游修复见第 18 节。

### 7.4 模型 D 的惯性来源和例外

模型 D 的 inertial 与模型 A 高度一致。除 XML 小数截断外，以下 body 的 mass、COM、principal inertia 和 `iquat` 均可视为直接等同模型 A：pelvis、hip-pitch/roll/yaw、ankle-roll/pitch、waist-yaw/roll、torso、head、shoulder-pitch/yaw、elbow、wrist-roll/pitch。

实质例外为：

| Body | 模型 A | 模型 D | 差异与含义 |
|---|---|---|---|
| left knee COM | `[0.004613,-0.024532,-0.167155]` | `[0.004613,0,-0.167155]` | D 将 lateral COM 分量清零，COM 相差 24.532 mm；质量和 principal inertia 保持 A |
| right knee COM | `[0.004613,0.024532,-0.167155]` | `[0.004613,0,-0.167155]` | 左右同理，D 将两侧膝 COM 对称到各自 body 的 Y=0 |
| left/right shoulder-roll COM | X=`-0.020610` m | X=`0` | D 将 X 分量清零，COM 相差 20.610 mm；质量、principal inertia、`iquat` 保持 A |
| left wrist-yaw | A 合并 hand 后 0.775752 kg | D 0.605752 kg | D 少 0.17 kg；COM 相对 A 差 19.641 mm，principal inertia 最大差约 0.001346 kg·m² |
| right wrist-yaw | A 合并 hand 后 0.775752 kg | D 0.605752 kg | D 少 0.17 kg；COM 相对 A 差 19.641 mm，principal inertia 最大差约 0.001346 kg·m² |

所以模型 D 不是采用模型 C 的官方 ROS 惯性。更准确的描述是：**模型 D 使用模型 A/H1031 风格的主体惯性，局部清理 knee/shoulder-roll COM，并移除 fixed rubber-hand 惯性。**

## 8. 运动学差异概览

### 8.1 B 与 C

B 和 C 的 31 个活动关节具有相同的：

- parent/child；
- local joint axis；
- body/joint origin；
- hip-pitch joint frame 的 `±0.5236 rad` 绕 X 倾角；
- shoulder-pitch joint frame 的 `±0.26914 rad` 绕 X 倾角。

差异主要是官方 MJCF 把数值截断到较少小数位。编译后 joint range 最大差约 `4.3e-6 rad`，不构成结构变化。

### 8.2 A 与 C/B

模型 A/B/C 的逐关节原点见附录 1。主要局部平移差为：

| Joint/body origin | A - C/B 的距离 |
|---|---:|
| hip-pitch L/R | 4.034 mm |
| hip-roll L/R | 0.077 mm |
| hip-yaw L/R | 0.700 mm |
| knee L/R | 4.090 mm |
| ankle-roll L/R | 0.001 mm |
| ankle-pitch L/R | 7.984 mm，主要是 X 方向 `-22.5 -> -14.516 mm` |
| waist-yaw | 0.700 mm |
| waist-roll | 32.500 mm，A 中 waist-roll 与 waist-yaw 同点，C/B 中沿 Z 分开 32.5 mm |
| head-pitch | 18.927 mm |
| head-yaw | 18.627 mm |
| shoulder-pitch | 约 3.55-3.58 mm |
| shoulder-roll | 30.936 mm |
| shoulder-yaw | 28.072 mm |
| elbow | 8.931 mm |
| wrist-roll | 9.779 mm |
| wrist-pitch | 76.152 mm |
| wrist-yaw | 80.513 mm |

这些是局部 parent frame 中的 origin 差，不应直接累加。部分腕部大差异是 link 分段方式重新分配，末端累计位置差小于单个局部 origin 差。

### 8.3 模型 D 的运动学来源

模型 D 的 joint axis 和 29 个活动 joint range 与模型 A 一致到 `4.3e-6 rad` 舍入，但 joint/body origin 不是模型 A 的复制。其局部 origin 混合了重新布置的左右腿、模型 C 风格的 32.5 mm waist 间距，以及重新切分的手臂/腕部 frame。

关键局部差异如下；完整 31 个语义 body 表见附录 2。

| Joint/body | D-A 局部 origin 差 | D-C 局部 origin 差 | 观察 |
|---|---:|---:|---|
| left/right hip-pitch | 5.088 / 4.035 mm | 3.098 / 0.004 mm | 右侧几乎等于 C，左侧另有约 3.1 mm 偏移 |
| left/right hip-roll | 23.633 / 4.077 mm | 23.556 / 4.000 mm | 明显左右不对称 |
| left/right hip-yaw | 23.918 / 6.251 mm | 23.844 / 5.961 mm | 局部 frame 重排 |
| knee L/R | 22.769 mm | 26.855 mm | D 将 knee body 的 lateral origin 改到另一侧 |
| ankle-roll L/R | 27.104 mm | 27.104 mm | D 的 parent-local Y 分量为 0 |
| ankle-pitch L/R | 22.506 / 22.500 mm | 14.525 / 14.516 mm | D 的 X origin 为 0 |
| waist-roll | 32.535 mm | 1.510 mm | D 基本采用 C 的 32.5 mm Z 间距，附加 1.51 mm Y 偏移 |
| shoulder-roll L/R | 约 28.53 mm | 约 2.44 mm | 更接近 C |
| shoulder-yaw L/R | 约 29.15 mm | 约 7.98 mm | 更接近 C，但并不相同 |
| wrist-roll L/R | 约 10.0-10.3 mm | 0.515 / 0.653 mm | 很接近 C |
| wrist-pitch L/R | 约 31-32 mm | 52.778 / 49.314 mm | 采用第三种局部 frame 切分 |
| wrist-yaw L/R | 约 40-42 mm | 51.784 / 51.466 mm | 同样不是 A/C 的直接复制 |

局部 origin 看起来差异很大，但累计运动学更有解释力：

- 零姿态下 D 的左右 ankle-pitch world position 与 C 仅差 `0.539/0.178 mm`；
- D 的 torso origin 与 C 完全一致；
- D 的 head-yaw world position 与 C 相差约 `2.062 mm`；
- D 的左右 wrist-yaw world position 与 C 相差约 `5.620/2.554 mm`。

因此模型 D 的准确描述是：**joint range/inertial 主体接近模型 A，累计骨架位置整体更接近模型 C，但局部 link frame 和左右腿并非严格对称复制。**

### 8.4 零姿态下的累计关键 body 位置

以下坐标均以 pelvis 原点为世界原点，关节角全零，单位 m。右侧基本镜像，因此列左侧代表项。

| Body | 模型 A | 模型 B/C | 模型 D | D-C norm |
|---|---|---|---|---:|
| left hip-pitch | `[0,0.083873,-0.048424]` | `[0,0.080380,-0.046407]` | `[0,0.081928,-0.043723]` | 3.098 mm |
| left knee | `[0.003323,0.138355,-0.488729]` | `[0.003279,0.138922,-0.487902]` | `[0.003293,0.115922,-0.487903]` | 23.000 mm |
| left ankle-roll | `[0.036043,0.115353,-0.985349]` | `[0.035998,0.115920,-0.984522]` | `[0.021676,0.115922,-0.984523]` | 14.322 mm |
| left ankle-pitch | `[0.013543,0.078053,-0.985349]` | `[0.021482,0.078620,-0.984522]` | `[0.021676,0.079122,-0.984523]` | 0.539 mm |
| waist-roll/torso origin | `[0,0,0.089837]` | `[0,0,0.123037]` | `[0,0,0.123037]` | 0 mm |
| head-yaw | `[-0.000228,0.000327,0.644147]` | `[-0.000115,0,0.678337]` | `[-0.000115,0.001809,0.677347]` | 2.062 mm |
| left shoulder-pitch | `[0.001500,0.112540,0.447957]` | `[0.003085,0.110000,0.483117]` | `[0.001978,0.115530,0.480957]` | 6.039 mm |
| left elbow | `[0.015505,0.217358,0.194464]` | `[0.013972,0.204166,0.228495]` | `[0.013986,0.208857,0.228549]` | 4.691 mm |
| left wrist-yaw | `[0.251793,0.186536,0.208354]` | `[0.252439,0.182768,0.214552]` | `[0.253990,0.188169,0.214458]` | 5.620 mm |

注意 wrist-pitch 局部/累计位置差很大，但 wrist-yaw 末端位置重新接近，说明两套资产对 wrist link frame 的切分方式不同，而不只是整条手臂简单变长。

### 8.5 joint axis 与 H2 hip coupling

四套 H2 模型中所有同名活动 joint 的 local axis 语义一致：pitch 为 Y、roll 为 X、yaw 为 Z。H2 hip-pitch body frame 相对 pelvis 绕 X 倾斜约 30°，所以 hip pitch/roll 在更上层坐标中不是 G1 式简单正交语义。该特征四者都有，不是四者之间的差异。

H2LoRA 的 G1 <-> H2 hip 语义映射仍然必须保留；更换 B/C 资产不会自动取消该映射。

## 9. 关节范围、源 effort/velocity 与实际训练限制

### 9.1 关节角范围

下肢、腰部和头部的 A/C 关节范围一致。B 与 C 仅有小数位截断。A/C 的实质范围差异集中在手臂：

| Joint | A | C/B | 差异 |
|---|---|---|---|
| left shoulder-roll | `[-0.516617, 2.635447]` | `[-0.517, 2.494]` | A 正向多 0.141447 rad，即 8.104° |
| right shoulder-roll | `[-2.635447, 0.516617]` | `[-2.494, 0.517]` | 左右镜像，A 负向多 8.104° |
| wrist-pitch L/R | `[-0.436332, 0.436332]` | `[-0.576, 0.576]` | A 每侧范围窄约 0.139668 rad，即 8.002° |
| wrist-yaw L/R | `[-1.221731, 1.221731]` | `[-1.22, 1.22]` | 约 0.099° 的小差异 |
| shoulder-pitch upper | `1.832596` | `1.833` | 约 0.023° |
| elbow upper | `3.071779` | `3.071` | 约 0.045° |

模型 D 的 29 个活动 joint range 与模型 A 对应关节一致，最大差 `4.3e-6 rad`，只是 MJCF 小数截断。模型 D 不存在 head joint range，因为 head 两 body 固定。

### 9.2 源资产中的 effort/velocity

A/C URDF 的下肢、腰和头部 effort/velocity 元数据一致，手臂不同：

| Joint class | A effort / velocity | C effort / velocity | B 源 MJCF |
|---|---|---|---|
| hip pitch/roll/yaw, knee | 360 Nm / 20 rad/s | 360 / 20 | torque ctrl range ±360；没有 velocity limit |
| ankle-roll | 19 / 100.7 | 19 / 100.7 | ±19；没有 velocity limit |
| ankle-pitch | 66.88 / 28.61 | 66.88 / 28.61 | ±66.88；没有 velocity limit |
| waist-yaw | 120 / 28.375 | 120 / 28.375 | ±120；没有 velocity limit |
| waist-roll/pitch | 180 / 28.375 | 180 / 28.375 | ±180；没有 velocity limit |
| head pitch/yaw | 50 / 10 | 50 / 10 | ±50；没有 velocity limit |
| shoulder-pitch | 120 / 28.375 | 130 / 21.9 | ±130；没有 velocity limit |
| shoulder-roll/yaw | 54 / 34.3 | 60 / 18.7 | ±60；没有 velocity limit |
| elbow | 54 / 34.3 | 60 / 18.7 | ±60；没有 velocity limit |
| wrist-roll | 54 / 34.3 | 60 / 18.7 | ±60；没有 velocity limit |
| wrist-pitch/yaw | 25 / 50.125 | 10 / 37.7 | ±10；没有 velocity limit |

这里的 A URDF effort 不是训练最终限制。

模型 D 源 MJCF 的 joint 只给 range，没有 effort/velocity。effort、Kp、Kd 和 armature 全部由 `h2_constants.py` 的 MjLab actuator config 添加。

### 9.3 H2LoRA 训练实际 actuator 参数

`H2_CFG` 以 20 Hz natural frequency、damping ratio 2.0 计算 implicit actuator 参数：

| Joint class | armature | Kp | Kd | effective effort | effective velocity | action scale |
|---|---:|---:|---:|---:|---:|---:|
| hip-pitch/roll/knee | 0.025101925 | 396.393711 | 12.617604 | 417 Nm | 20 rad/s | 0.262996 |
| hip-yaw、waist-yaw | 0.010177520 | 160.716954 | 5.115780 | 264 Nm | 32 rad/s | 0.410660 |
| ankle-pitch/roll | 0.007219450 | 114.004985 | 3.628891 | 150 Nm | 37 rad/s | 0.328933 |
| waist-roll/pitch | 0.007219450 | 114.004985 | 3.628891 | 150 Nm | 37 rad/s | 0.328933 |
| head pitch/yaw | 0.007219450 | 114.004985 | 3.628891 | 150 Nm | 37 rad/s | 0.328933；policy 外默认角 |
| shoulder、elbow、wrist-roll | 0.003609725 | 57.002492 | 1.814446 | 75 Nm | 37 rad/s | 0.328933 |
| wrist-pitch/yaw | 0.004250000 | 67.113310 | 2.136283 | 15 Nm | 22 rad/s | 0.055876 |

action scale 由 `0.25 * effort_limit / stiffness` 计算。它是 policy normalized action 到 joint position delta 的缩放，不是质量或零点补偿。

训练限制与官方 B/C 源标称值存在很大差异，例如：

- ankle-roll：训练 150 Nm，B/C 源值 19 Nm；
- ankle-pitch：训练 150 Nm，B/C 源值 66.88 Nm；
- hip-pitch/roll/knee：训练 417 Nm，B/C 源值 360 Nm；
- hip-yaw：训练 264 Nm，B/C 源值 360 Nm；
- waist-yaw：训练 264 Nm，B/C 源值 120 Nm；
- head：训练 150 Nm，B/C 源值 50 Nm。

这说明“换 robot XML”之后若不覆盖 actuator，sim2sim 不是公平模型对比；当前工程已经在官方模型实验配置中覆盖这些限制。

### 9.4 模型 D 的 MjLab 训练 actuator

模型 D 使用 MuJoCo builtin position actuator，编译后的控制律等价于 position servo：

```text
tau = Kp * (q_des - q) - Kd * dq
```

| Joint class | armature | Kp | Kd | effort limit | velocity limit | action scale |
|---|---:|---:|---:|---:|---|---:|
| hip-pitch/roll/yaw、knee | 0.01 | 200 | 4 | 360 Nm | 未设置 | 0.450000 |
| ankle-roll | 0.01 | 40 | 2 | 19 Nm | 未设置 | 0.118750 |
| ankle-pitch | 0.01 | 40 | 2 | 66 Nm | 未设置 | 0.412500 |
| waist yaw/roll/pitch | 0.01 | 150 | 3 | 120 Nm | 未设置 | 0.200000 |
| shoulder、elbow、wrist-roll | 0.01 | 40 | 2 | 54 Nm | 未设置 | 0.337500 |
| wrist-pitch/yaw | 0.01 | 20 | 1 | 25 Nm | 未设置 | 0.312500 |
| head | 无 actuator | - | - | - | - | - |

模型 D 同样按 `0.25 * effort / Kp` 计算 action scale，但其 Kp/effort 与模型 A 完全不同。因此二者即使 policy 都输出 29 维 normalized action，也不能直接共享同一 action-to-joint-delta 映射。

编译后的模型 D 关节 passive damping 和 frictionloss 均为 0，29 个 hinge DoF 的 armature 均为 0.01。阻尼来自 position actuator 的 `Kd`，不是 joint passive damping。其 actuator position `ctrlrange` 会被 MjLab 扩展到 joint range 外约 `effort/Kp`，但训练 action 仍以 default joint position 加上述 action scale 生成 target；不能把扩展后的 ctrlrange 当作允许的物理 joint range。

## 10. 碰撞几何

### 10.1 编译后统计

| geom 类型 | 模型 A 原始 URDF 编译 | 模型 B 官方 MJCF | 模型 C 原始 URDF 编译 | 模型 D MjLab 编译 |
|---|---:|---:|---:|---:|
| mesh | 51 | 47 | 53 | 32，全部 visual/non-contact |
| sphere | 2 | 14 | 2 | 2，pelvis + head collision |
| cylinder | 4 | 2 | 2 | 0 |
| capsule | 0 | 0 | 0 | 31 |
| box | 0 | 2 | 0 | 0 |
| 总 geom | 57 | 65 | 57 | 65 |
| 有效 collision geom | 23 | 33 | 23 | 33 |

上表是统一编译统计。模型 A 在 IsaacLab 训练中还设置 `replace_cylinders_with_capsules=True`，所以 PhysX 实际使用 capsule 替换 URDF cylinder。模型 D 则从源 MJCF 开始就使用显式 capsule collision，不依赖导入器替换。

### 10.2 关键 link 的 collision 差异

| Link | 模型 A | 模型 B | 模型 C | 模型 D | 动力学含义 |
|---|---|---|---|---|---|
| pelvis | 专用 `pelvis_collision.stl` | 完整 `pelvis.stl` | 完整 `pelvis.stl` | sphere r=0.08 | D 使用最简 pelvis 包络 |
| hip-pitch L/R | 完整 hip-pitch mesh | 无 | collision 被注释 | 无 | A 独有较大 hip-pitch mesh collision |
| hip-roll L/R | 完整 mesh | sphere r=0.05 | 同 B | 短 capsule r=0.06 | 四者髋部自碰撞边界不同 |
| hip-yaw/thigh | 完整 mesh | 完整 mesh | 完整 mesh | capsule r=0.05，half≈0.1003 | D 用规则大腿 capsule 代替 mesh |
| knee/shin | cylinder r=0.04, half=0.175 | cylinder r=0.025, half=0.175 | 同 B | 每侧 shin capsule r=0.03 + linkage capsule r=0.02 | D 显式覆盖小腿和连杆两段 |
| ankle-roll | 无 | 无 | 无 | 无 | 无直接 collision geom |
| ankle-pitch/foot | 完整 ankle-pitch mesh | 每脚 6 sphere + 1 box | 完整 ankle-pitch mesh | 每脚 7 capsule，r=0.01 | 三种完全不同的足底接触表示 |
| torso | 专用 `torso_collision.stl` | 完整 `torso_link.stl` | 完整 `torso_link.stl` | capsule r=0.09, half=0.09 | D 用竖直 torso capsule |
| head-pitch | 无 | 无 | 无 | 无 | 无直接 collision |
| head-yaw | head-yaw mesh | head-yaw mesh | head-yaw mesh | sphere r=0.065 | D 用 sphere 替代 mesh |
| shoulder-yaw | 多数对应 link mesh | 对应 link mesh | 同 B | capsule r=0.035, half=0.075 | D 规则化上臂碰撞 |
| elbow/forearm | elbow mesh | elbow mesh | 同 B | capsule r=0.035, half=0.05 | D 规则化前臂碰撞 |
| wrist | wrist-yaw sphere；roll/pitch 无 | wrist-yaw mesh | wrist-yaw + hand mesh | wrist-pitch capsule r=0.035, half=0.04 | D 增加 wrist capsule |
| hand 末端 | fixed rubber hand 质量 + sphere collision | wrist-yaw mesh 内表示 | fixed hand mesh | wrist-yaw 上 hand capsule r=0.035, half=0.04 | D 有碰撞长度但无独立 hand mass |

模型 A/B/C 的完整逐 body collision 表见附录 6，模型 D 的 33 个 collision geom 见附录 7。

### 10.3 脚底接触的具体数值

B 每只脚的 6 个球半径均为 `0.008 m`。左脚局部坐标为：

```text
(-0.075,  0.066, -0.050)
(-0.075,  0.007, -0.050)
( 0.100,  0.080, -0.048)
( 0.100, -0.007, -0.048)
( 0.150,  0.066, -0.050)
( 0.150,  0.007, -0.050)
```

右脚 Y 坐标镜像。每只脚另有一个 half-size 为 `[0.115, 0.040, 0.004] m` 的 box，中心 X 为 `0.035 m`，Z 为 `-0.040 m`。

模型 D 每只脚使用 7 条半径 `0.01 m` 的 capsule，全部位于 Z=`-0.038 m`。左脚 `fromto` 为：

```text
( 0.080, 0.008, -0.038) -> (0.130, 0.008, -0.038)
(-0.065, 0.018, -0.038) -> (0.140, 0.018, -0.038)
(-0.068, 0.027, -0.038) -> (0.145, 0.027, -0.038)
(-0.070, 0.037, -0.038) -> (0.150, 0.037, -0.038)
(-0.068, 0.047, -0.038) -> (0.145, 0.047, -0.038)
(-0.065, 0.055, -0.038) -> (0.140, 0.055, -0.038)
( 0.130, 0.065, -0.038) -> (0.080, 0.065, -0.038)
```

右脚 Y 坐标镜像。前六条主要覆盖鞋底纵向条带，第七条在外侧形成横向边界。

按各 collision geom 在 ankle-pitch body 坐标系中的整体 AABB 计算：

| 模型 | 左脚 collision AABB min | max | extent |
|---|---|---|---|
| A mesh | `[-0.079995, -0.005493, -0.054300]` | `[0.160007, 0.080065, 0.019000]` | `[0.240002, 0.085557, 0.073300]` |
| B primitives | `[-0.083000, -0.015000, -0.058000]` | `[0.158000, 0.088000, -0.036000]` | `[0.241000, 0.103000, 0.022000]` |
| C mesh | `[-0.080000, -0.005179, -0.054300]` | `[0.160000, 0.080379, 0.019000]` | `[0.240000, 0.085557, 0.073300]` |
| D capsules | `[-0.080000, -0.002000, -0.048000]` | `[0.160000, 0.075000, -0.028000]` | `[0.240000, 0.077000, 0.020000]` |

A/C mesh 的总体尺寸接近，但 MuJoCo mesh collision 会使用凸包。该凸包包含鞋底以上的整个 ankle 外壳，高度约 `73.3 mm`；B 和 D 的专用 contact primitives 分别只形成约 `22 mm` 和 `20 mm` 高的薄接触层。B 的离散球/box 与 D 的平行 capsule 条带也不会产生相同的接触点数量、滚动边界和压力中心轨迹。

在不施加训练/运行时覆盖的纯 MuJoCo 编译结果中，模型 A/B/C 的可碰撞 robot geom 都继承相同默认值：

```text
friction = [1.0, 0.005, 0.0001]
solref   = [0.02, 1.0]
solimp   = [0.9, 0.95, 0.001, 0.5, 2.0]
```

这只说明对应 MuJoCo XML 没有给模型 A/B/C 设置不同的 geom material。A 的实际 IsaacLab 训练随后使用 PhysX material 和启动随机化，不能拿上述 MuJoCo 默认值代表训练摩擦。

模型 D 的 `FULL_COLLISION` 在训练编译时进一步规定：14 个 foot capsule 使用 `condim=3`、friction=`[0.6,0.005,0.0001]`、priority=1；其余 19 个 collision geom 使用 `condim=1`、friction=`[1,0.005,0.0001]`。tracking 任务启动时再把足底滑动摩擦随机化到 `[0.3,1.2]`，velocity 任务随机化到 `[0.3,1.6]`，每个环境内 14 个 foot capsule 共享同一个采样值。

### 10.4 训练、原始 sim2sim 与官方 sim2sim 的实际接触链路

| 运行链路 | robot asset | 脚底实际接触 |
|---|---|---|
| IsaacLab H2LoRA 训练 | A URDF | PhysX ankle-pitch mesh；cylinder 转 capsule |
| `h2_lora` sim2sim | A 派生 `mjcf/h2.xml` | 运行时禁用 ankle mesh，并补 B 的 6 sphere + 1 box |
| `h2_lora_unitree_official` sim2sim | B | B 原生 6 sphere + 1 box |
| `h2_lora_unitree_ros` sim2sim | C 生成 MJCF | 保留 C ankle-pitch mesh，不补官方 primitives |
| `unitree_rl_mjlab` H2 tracking/velocity 训练 | D MJCF + `FULL_COLLISION` | 每脚 7 capsule；foot condim=3；friction 启动随机化 |

因此，当前 `h2_lora` sim2sim 与模型 A 的训练并不共享完全相同的接触模型；它与模型 B 的脚底接触反而更接近。模型 D 的稳定性也不能直接归因于其惯性更真实，因为它使用了第四套、更规则的 capsule 接触。

## 11. Visual 和 mesh 文件差异

### 11.1 文件数量和额外文件

- A：37 个 STL；包含 `pelvis_collision.stl`、`torso_collision.stl`、`left/right_rubber_hand.stl`、`simple_torso_link.stl` 和旧版 small arm/roll mesh。
- B：32 个 STL；没有独立 fixed hand mesh，wrist-yaw mesh 表示其末端结构。
- C：34 个 STL + 34 个 DAE；比 B 多 `left_hand_link.stl`、`right_hand_link.stl`，并提供 DAE visual 版本 `H2_dae.urdf`。
- D：32 个 STL；basename 集合与 B 完全一致，但绝大多数文件并非字节相同。D 没有 rubber-hand/hand-link 独立 mesh。

### 11.2 SHA-256 字节级交集

| 对比 | 同名 STL 数 | 字节完全相同 | 字节不同 |
|---|---:|---:|---:|
| A vs B | 30 | 1 | 29 |
| A vs C | 30 | 0 | 30 |
| B vs C | 32 | 27 | 5 |
| D vs A | 30 | 3 | 27 |
| D vs B | 32 | 3 | 29 |
| D vs C | 32 | 1 | 31 |

A/B 唯一字节相同的同名文件是 `head_yaw_link.stl`。B/C 不同的 5 个文件为：

- `head_yaw_link.stl`
- `left_wrist_pitch_link.stl`
- `right_wrist_pitch_link.stl`
- `left_wrist_yaw_link.stl`
- `right_wrist_yaw_link.stl`

hash 不同只能证明不是同一字节文件，可能还包含 STL 编码、三角形排序或浮点精度变化；不能单凭 hash 断言外形差了多少。但结合 joint origin、COM、惯量和 collision 定义，可以确定 A 与 B/C 不是同一套机械资产。

模型 D 的字节相同项为：

- D/A：`head_yaw_link.stl`、`left_ankle_roll_link.stl`、`right_ankle_roll_link.stl`；
- D/B：`head_yaw_link.stl`、`left_elbow_link.stl`、`right_wrist_yaw_link.stl`；
- D/C：仅 `left_elbow_link.stl`。

该混合结果再次说明 D 不是从 B 或 C 整目录复制得到的。其 mesh basename 与 B/C 对齐，但文件内容来自混合来源或经过重新导出。完整文件级状态见附录 9。

## 12. 仿真器、执行器和传感器差异

### 12.1 A 的训练配置

`H2_CFG` 的关键配置：

- `fix_base=False`；
- root 初始位置 `[0, 0, 1.04]`；
- 重力开启；
- rigid body linear/angular damping 均为 0；
- max depenetration velocity = 1.0；
- self collision 开启；
- solver position iterations = 8；
- solver velocity iterations = 4；
- URDF drive 原始 stiffness/damping 先设为 0，再由 implicit actuator 设置；
- soft joint position limit factor = 0.9；
- contact sensor 开启；
- cylinder collision 转 capsule。

默认关节姿态：

```text
hip-pitch L/R   = -0.312 rad
knee L/R        =  0.669 rad
ankle-pitch L/R = -0.363 rad
elbow L/R       =  0.600 rad
shoulder-pitch  =  0.200 rad
shoulder-roll   = +0.200 rad(left), -0.200 rad(right)
其他关节        =  0
```

训练 event 还在启动时随机化 robot material：

- static friction `[0.3, 1.6]`；
- dynamic friction `[0.3, 1.2]`；
- restitution `[0.0, 0.5]`；
- 64 个离散 bucket。

### 12.2 B 源 MJCF 自带配置

B 源文件包含：

- timestep `0.001 s`；
- 全关节默认 passive damping `0.05`；
- 全关节默认 armature `0.01`；
- 全关节默认 frictionloss `0.2`；
- pelvis 初始 body position `[0, 0, 1.2]`；
- free floating-base joint；
- 31 个 torque motor；
- 101 个 sensor。

B 中只有 pelvis body 缺少显式 `<inertial>`；其他 31 个活动 child body 均有显式 inertial。pelvis 的可碰撞 mesh 没有覆盖 density，因此 MuJoCo 按默认 mesh density 自动推导其 mass、COM 和 inertia。

101 个 sensor 由以下部分组成：

- 31 joint position；
- 31 joint velocity；
- 31 joint actuator force；
- pelvis IMU：quaternion、gyro、accelerometer、frame position、frame linear velocity，共 5；
- torso secondary IMU：quaternion、gyro、accelerometer，共 3。

C 源 URDF 没有 actuator、sensor、timestep、joint damping、armature 或 frictionloss；这些必须由 Gazebo/MuJoCo/控制器插件补充。URDF 中 camera/IMU fixed link 只定义坐标位置，并不自动产生可读取的仿真 sensor。

四者 floating base 和初始高度还存在以下区别：

| 对象 | 源资产中的 root | 实际运行初始高度 |
|---|---|---|
| A 训练 URDF | floating joint 在 URDF 中被注释 | `H2_CFG.fix_base=False` 补 floating root，pelvis Z=1.04 m |
| A 派生 sim2sim MJCF | 显式 free root | WBC 配置重置到 Z=1.04 m，再按鞋底最低点校准 |
| B 官方 MJCF | 显式 free root，pelvis body Z=1.2 m | 官方模型实验配置先用 Z=1.2 m，再按鞋底最低点校准 |
| C 源 URDF | floating joint 被注释，单独加载时不是完整自由落体场景 | 生成脚本补 free root；当前 WBC 配置重置到 Z=1.04 m，再按 mesh 最低点校准 |
| D 源 MJCF | 显式 free root，pelvis body Z=1.03 m | MjLab `HOME_KEYFRAME.pos=(0,0,1.03)`；task reset 在该高度基础上初始化 |

### 12.3 当前 B/C sim2sim 运行时覆盖

当前 `h2_lora_unitree_official` 和 `h2_lora_unitree_ros` WBC 配置会：

- 使用外部 Unitree LowCmd 显式 PD：`tau + kp*(q_des-q) + kd*(dq_des-dq)`；
- 将 motor torque clamp 覆盖为训练的 `motor_effort_limit_list`；
- 将每个关节 armature 覆盖为 H2_CFG 对应值；
- 设置 damping=0.05、frictionloss=0.2；
- 使用 `implicitfast` integrator、Newton solver、50 iterations、10 line-search iterations；
- MuJoCo 低层积分 1000 Hz，policy/deploy 周期 50 Hz；
- 使用同一套 MOTOR_KP/MOTOR_KD；
- 按 joint name 处理 qpos 与 actuator 顺序差异；
- 默认启用上方弹力带支撑机制；
- 根据鞋底最低点校准初始 base Z，使接触 clearance 为 2 mm。

因此，在当前 sim2sim 对比中，执行器 torque limit、PD 和 armature 已尽量一致；剩余主要差异就是 robot XML 的运动学、质量/COM/惯量、collision/mesh 和初始 base 高度来源。求解器与 IsaacLab PhysX 仍然不是同一个物理引擎，不能宣称数值行为完全相同。

### 12.4 模型 D 的 MjLab 训练配置

模型 D 的 `HOME_KEYFRAME` 为：

```text
pelvis Z           = 1.03 m
hip-pitch L/R      = -0.25 rad
knee L/R           =  0.50 rad
ankle-pitch L/R    = -0.25 rad
shoulder-pitch L/R =  0.35 rad
elbow L/R          =  0.87 rad
shoulder-roll      = +0.18 rad(left), -0.18 rad(right)
其他 29D joint     = 0
head               = 固定，无 joint state
```

robot 公共配置：

- 29 个 builtin position actuator，参数见 9.4；
- soft joint position limit factor=0.9；
- `FULL_COLLISION` 开启全部 33 个 collision geom 和自碰撞；
- 4 对显式 contact exclude：左右 `elbow_link <-> wrist_pitch_link`，以及 pelvis 与左右 hip-roll；
- 非足 collision `condim=1`，足底 `condim=3`；
- 源 MJCF 内建 4 个 3D sensor：pelvis gyro、velocimeter、accelerometer，以及 torso subtree angular momentum，共 12 维；
- tracking/velocity task 再通过 MjLab contact sensor 记录足底接触和自碰撞。

公共仿真周期：

| 参数 | tracking | velocity base | H2 rough override | H2 flat override |
|---|---:|---:|---:|---:|
| MuJoCo timestep | 0.005 s | 0.005 s | 0.005 s | 0.005 s |
| decimation | 4 | 4 | 4 | 4 |
| policy period | 0.020 s / 50 Hz | 0.020 s / 50 Hz | 同 | 同 |
| solver iterations | 10 | 10 | 10 | 10 |
| line-search iterations | 20 | 20 | 20 | 20 |
| `ccd_iterations` | H2 config 50 | base 默认 | 500 | 50 |
| `nconmax` | H2 config 64 | base 35 | 48 | unlimited/None |
| `njmax` | H2 config 300 | 1500 | 1500 | 300 |

训练 domain randomization：

| 随机项 | D tracking | D velocity |
|---|---|---|
| foot friction | startup `[0.3,1.2]`，14 个 foot capsule 共享采样 | startup `[0.3,1.6]`，共享采样 |
| torso COM offset | X/Y/Z 各 `[-0.05,0.05] m` | 相同 |
| encoder bias | `[-0.01,0.01] rad` | `[-0.015,0.015] rad` |
| push interval | 1-3 s | 5-6 s |
| mass randomization | 未配置 | 未配置 |

模型 D 与模型 A 都做摩擦、COM、编码器和 push 类随机化，但 nominal actuator、碰撞 primitives、默认姿态、物理引擎和 nominal inertial 不相同，不能把“都做了 domain randomization”理解为训练分布已经一致。

## 13. 统一姿态下的整机质心

### 13.1 计算方法

- 把四者 floating base 平移统一设为原点、姿态统一设为单位四元数；
- 使用相同关节角；
- B/D 用 MuJoCo `subtree_com[pelvis]` 计算整机 COM；A/C/C-Plus 为避开 MuJoCo 3.10 fixed-link fusion bug，直接按 URDF 关节树做前向运动学并对所有 link COM 质量加权；
- 用左右 `ankle_pitch_link` body origin 的中点作为统一几何参考点；
- 未使用 foot pressure、地面接触或 policy 输出，因此结果只反映模型参数和运动学。

### 13.2 全零关节姿态

| 模型 | 整机 COM `[x,y,z]` m | ankle-pitch 中点 | COM - ankle 中点 |
|---|---|---|---|
| A | `[0.028656, 0.001198, -0.009908]` | `[0.013543, 0, -0.985349]` | `[0.015113, 0.001198, 0.975440]` |
| B | `[0.025382, 0.001764, -0.004331]` | `[0.021482, 0, -0.984522]` | `[0.003900, 0.001764, 0.980190]` |
| C | `[0.023975, 0.001575, -0.005546]` | `[0.021482, 0, -0.984522]` | `[0.002493, 0.001575, 0.978976]` |
| D | `[0.026802, 0.002211, 0.004913]` | `[0.021668, 0.000250, -0.984523]` | `[0.005134, 0.001960, 0.989436]` |

零姿态下，A 的 COM 相对 ankle reference 前移 `15.113 mm`，C 为 `2.493 mm`，两者相差约 `12.621 mm`。B 因 pelvis 更轻，前移量比 C 多约 `1.408 mm`；D 为 `5.134 mm`。D 的 COM 高度相对 ankle reference 最高，来自其混合骨架位置和 A 风格质量分布。

### 13.3 H2LoRA 训练默认站姿

| 模型 | 整机 COM `[x,y,z]` m | ankle-pitch 中点 | COM - ankle 中点 |
|---|---|---|---|
| A | `[0.028801, 0.001198, -0.014104]` | `[-0.059152, 0, -0.933101]` | `[0.087953, 0.001198, 0.918997]` |
| B | `[0.028871, 0.001765, -0.006413]` | `[-0.052225, 0, -0.935435]` | `[0.081096, 0.001765, 0.929021]` |
| C | `[0.027263, 0.001576, -0.007513]` | `[-0.052225, 0, -0.935435]` | `[0.079488, 0.001576, 0.927922]` |
| D | `[0.027527, 0.002212, 0.001805]` | `[-0.051618, 0.000286, -0.935445]` | `[0.079144, 0.001926, 0.937250]` |

在默认弯腿站姿下：

- A 相对 C 的前向 COM/ankle 偏置多 `8.465 mm`；
- B 相对 C 多 `1.608 mm`，主要受 pelvis 自动推导惯性影响；
- A/C 的 COM 高度相对 ankle reference 相差约 `8.925 mm`；
- D 的前向偏置比 C 少约 `0.344 mm`，但 COM 高度相对 ankle reference 比 C 高约 `9.328 mm`。

这不是“真实机器人一定有 8.5 mm 零点误差”的证据。它证明的是：即使给完全相同的关节角，四套模型也不代表相同的整机质量位置和 ankle reference 几何，policy 所需的静态平衡输出自然可能不同。D 在该姿态下前向 COM 与 C 接近，是 D 的 C-like 累计足端运动学和 A-like 惯性共同作用的结果，不能据此说 D 的每个参数都更接近 C。

### 13.4 模型 D 自身 HOME_KEYFRAME 姿态

将 D 的 HOME_KEYFRAME 关节角同时施加到四模型：

| 模型 | COM - ankle-pitch 中点 `[x,y,z]` m | 前向 X |
|---|---|---:|
| A | `[0.043602,0.001198,0.938177]` | 43.602 mm |
| B | `[0.038211,0.001765,0.947399]` | 38.211 mm |
| C | `[0.037320,0.001575,0.946574]` | 37.320 mm |
| D | `[0.035033,0.001939,0.956012]` | 35.033 mm |

在 D 自己的较浅弯腿站姿下，D 的前向 COM/ankle 偏置反而比 C 少约 `2.287 mm`。这进一步说明“哪个模型更前倾”依赖具体姿态；比较模型时必须固定同一组关节角和同一参考点。

## 14. 与真机前倾、小碎步现象的关系

### 14.1 高优先级模型因素

1. **训练 torso/腰/腿偏轻。**相同姿态下，训练所需 ankle/hip 抗重力矩偏小。部署到更重的 C/真机质量分布时，脚踝 pitch PD 静差和 policy 补偿都可能增大。
2. **ankle-pitch 原点前后相差 7.984 mm。**这是接近此前 zero offset 实验改善量级的纯几何差异，会改变同一 ankle angle 对足底/小腿相对姿态的意义。
3. **默认姿态的整机 COM 相对 ankle reference 前向位置相差约 8.5 mm。**这足以改变静态 ankle torque 和接触压力中心。
4. **训练与 sim2sim 的脚底接触不一致。**当前 A 派生 sim2sim 和 B 都用友好的官方 primitives；训练 PhysX 和 C 原始模型使用完整 ankle mesh。sim2sim 更稳可能主要来自接触模型，而不是 policy 已适应真实 H2。
5. **B 的 pelvis 因缺失显式 inertial 而自动偏轻。**更轻的 pelvis 会降低髋/脚踝需要控制的近端负担，使 `h2_lora_unitree_official` 看起来更稳；这不能证明 B 更接近真机。
6. **waist/torso 几何和惯量不同。**A 的 torso origin 比 C/B 低约 33.2 mm，torso COM、质量和惯量也不同，policy 学到的腰部姿态与真实上身重力矩不完全匹配。
7. **D 的稳定性主要不能归因于“使用了官方 C 惯性”。**D 实际继承 A 风格的轻 torso/腿和重手臂分布，但采用更接近 C 的累计足端位置与规则 capsule 接触。这两项可能掩盖 nominal inertia mismatch。
8. **D 的 ankle actuator 比 A 弱很多。**D ankle-pitch 为 Kp=40、Kd=2、66 Nm；A 为 Kp≈114、Kd≈3.63、150 Nm。比较 policy 稳定性时必须连 actuator 一起固定，否则无法分离 robot asset 与低层控制影响。

### 14.2 中优先级因素

- hip-pitch/knee 数毫米运动学差异；
- hip-roll COM 和惯量差异；
- arm mass 重新分配导致动作时反作用力矩不匹配；
- D 移除 0.34 kg fixed hand mass、固定 head 后造成的上身响应差异；
- PhysX 与 MuJoCo 的接触/求解器差异；
- 真机电机、减速器、结构柔顺、通信延迟、速度估计滤波与模型中的理想 PD 差异。

### 14.3 不能从本审计直接推出的结论

- 不能仅凭 C 是官方 ROS 资产就认定其质量/惯量一定等于当前这台真机；仍需称重、CAD/BOM 或系统辨识确认。
- 不能仅凭 B sim2sim 更稳就认定 B 更真实；B 的脚底和 pelvis 都有让仿真更易稳定的因素。
- 不能仅凭 D 在 MjLab 能稳定训练就认定 D 更接近真机；D 同时更换了惯性组合、29-DoF 拓扑、actuator、足底 capsule 和 domain randomization。
- 不能从 COM 表直接计算 ankle zero offset；还需要真机绝对编码器零位、机械安装零位、鞋底平面和关节正方向标定。
- 不能把 policy target-reference 误差与低层 PD target-measured 误差混为同一误差。模型不匹配可能让 policy 主动输出补偿，而低层 PD 又因负载/柔顺产生另一个方向和量级的误差。

## 15. 后续模型基线建议

建议后续训练/仿真建立以下可控基线，而不是继续隐式混用四个源模型的部分参数：

1. 以 C 的 31-DoF 运动学、显式 mass、COM 和完整惯量矩阵为动力学基准。
2. 把 B 的 6 sphere + 1 box 与 D 的 7 capsule 作为两套显式、可切换的足底候选，在相同惯性和 actuator 下做严格对照实验，而不是默认其中一套更真实。
3. IsaacLab/PhysX 中建立与选定 MuJoCo contact primitives 同坐标的显式脚底，关闭 ankle mesh 与地面接触，使训练和 sim2sim 接触语义一致。
4. actuator 使用同一份按 joint name 管理的 armature/Kp/Kd/torque limit，不依赖 URDF/MJCF joint 顺序。
5. 将 pelvis、torso、hip、knee、ankle 的 mass/COM/inertia 设为可辨识参数，并围绕 C 数值做 domain randomization，而不是围绕 A 的旧 `H1031` 数值随机化。
6. 单独标定左右 ankle pitch/roll 的机械零位；零位修正应作用于 measured joint coordinate 的坐标变换，而不是只给 policy target 增加常量。
7. 在相同 reference、相同 PD、相同接触 primitives 下做 A inertial、C inertial、D hybrid inertial 和真机四方 log 对比，重点记录 base pitch、COM proxy、ankle torque、ankle target/measured、hip/knee/waist pitch 误差和接触状态。
8. 对 D 单独做消融：保持 D kinematics 不变，把惯性从 A-like 换成 C；保持惯性不变，把 7 capsule 换成 B foot；由此判断 D 的稳定主要来自运动学、惯性还是接触。
9. 已绕过 MuJoCo 3.10 fixed-link inertia fusion bug：C 和 ROS-mass 派生模型由生成器按 fixed-joint transform 与平行轴定理重新合成惯量，并自动断言左右 wrist composite COM 与源 URDF 一致到 `1e-9 m`。

推荐的最终目标不是“让某个 MuJoCo XML 看起来站得最稳”，而是让训练、sim2sim 和 real 对同一低层命令产生可解释、可量化接近的状态和扭矩响应。

## 16. 审计范围和限制

- 本文比较的是仓库当前文件，结果由开头 SHA-256 锁定；上游更新后必须重新审计。
- mesh hash 是文件级比较，不是几何等价证明。
- MuJoCo 编译 mesh 时会规范化 mesh reference frame，附录 6 的 mesh geom position 是编译后坐标，不等同于 URDF `<collision><origin>` 原文。
- principal inertia/`iquat` 由特征值分解得到，近简并主轴可能发生合法置换或符号翻转。
- 整机 COM 表使用理想刚体模型和统一关节角，没有电缆、外接设备、弹力绳、鞋底形变或实机载荷。
- C 的 fixed camera/IMU link 没有 inertial，因此不增加质量；若真机安装的传感器质量未包含在 torso/pelvis 的已有 inertial 中，C 仍可能漏算。
- 当前 B/C sim2sim 配置开启弹力带；评价自由站立时必须确认弹力带张力阶段和 log 区间，否则会把外力支撑误认为模型稳定性。
- D 的统一数值比较由 MuJoCo 3.10.0 完成，实际 MjLab `.venv` 使用 3.5.0；更新 MjLab/MuJoCo 后应复核 contact 和 actuator 编译结果。
- D 的 H2 robot asset 当前工作树干净，但 H2 tracking task 是当前 `unitree_rl_mjlab` 工作树中的未跟踪扩展；本文记录的是当前本地任务配置，不宣称它已包含在上游 HEAD。
- 真实部署中的 PR/AB 控制模式、编码器零点和电机方向属于低层接口审计，不由 robot XML 单独决定。
- 当前 MuJoCo 3.10 的 fixed-link inertia fusion 已由本机最小模型复现，并有上游未合并 PR #3250；升级 MuJoCo 后必须重新运行最小复现，不能假设版本号增加就已包含修复。

## 17. Unitree ROS H2 家族内部审计

本节只比较模型 C 家族内部的三份官方 URDF。它补充前文，不改变 A/B/C/D 四个模型代词：前文未特别标注时，“模型 C”仍表示 C-STL。

### 17.1 上游范围、版本和文件锁定

审计上游为 Unitree 官方 [`unitreerobotics/unitree_ros`](https://github.com/unitreerobotics/unitree_ros)，固定提交：

```text
d96d8f63ae17a7108d4f7229c00ef875ba7129c9
```

该提交的 `robots/` 树中只有以下三份 H2 robot URDF，没有发现其他 H2 URDF：

| 子版本 | 官方文件 | SHA-256 | 行数 | 字节数 |
|---|---|---|---:|---:|
| C-STL | [`robots/h2_description/H2.urdf`](https://github.com/unitreerobotics/unitree_ros/blob/d96d8f63ae17a7108d4f7229c00ef875ba7129c9/robots/h2_description/H2.urdf) | `32866f6f2dae8ed79dc622b015160f4f8fc018ce266ce4f84322084ddc741401` | 960 | 33,905 |
| C-DAE | [`robots/h2_description/H2_dae.urdf`](https://github.com/unitreerobotics/unitree_ros/blob/d96d8f63ae17a7108d4f7229c00ef875ba7129c9/robots/h2_description/H2_dae.urdf) | `9c8b06ba98a240c753b5fd6db21b1aaff30a2d4e7ec081bb6c6c242543265d54` | 971 | 34,213 |
| C-Plus | [`robots/h2_plus/H2_Plus.urdf`](https://github.com/unitreerobotics/unitree_ros/blob/d96d8f63ae17a7108d4f7229c00ef875ba7129c9/robots/h2_plus/H2_Plus.urdf) | `212533d728ca52c948df44d30393291137b69f4371cdb3a2bce703f3c4ad042a` | 3,819 | 103,133 |

本工程已导入的 C-STL 和 C-DAE 与该上游提交逐字节一致。C-Plus 当前只从上游审计，尚未复制到本工程 robot asset 目录。三份文件的 `<robot name>` 都是 `H2`；`H2_Plus.urdf` 的文件名带 Plus，但 XML robot name 并不带 Plus。

### 17.2 源 URDF 总体规模

| 指标 | C-STL | C-DAE | C-Plus |
|---|---:|---:|---:|
| links | 37 | 37 | 98 |
| 带 inertial 的 links | 34 | 34 | 98 |
| revolute joints | 31 | 31 | 75 |
| fixed joints | 5 | 5 | 22 |
| visual geoms | 34 mesh | 34 mesh | 88 mesh |
| collision geoms | 19 mesh + 2 sphere + 2 cylinder = 23 | 同 C-STL | 69 mesh + 2 sphere + 2 cylinder = 73 |
| 原始 link mass 合计 | 75.58745630 kg | 75.58745630 kg | 77.23827184 kg |
| transmission | 0 | 0 | 0 |
| `<gazebo>` 扩展 | 0 | 0 | 0 |
| mimic joints | 0 | 0 | 0 |
| URDF 内 actuator/PID | 无 | 无 | 无 |

`h2_description/meshes` 共 68 个文件，即 34 STL + 34 DAE，合计 43,765,606 bytes。`h2_plus/meshes` 共 148 个文件，即 114 STL + 34 DAE，合计 75,043,018 bytes；新增的 80 个 STL 位于左右 `sharpa/` 子目录。

三份 URDF 都包含一个 `<mujoco><compiler discardvisual="false">` 标签，并把 floating-base link/joint 放在注释中。该 compiler 标签不提供 actuator、接触材料或仿真器求解参数，不能把它理解为一份完整 MuJoCo robot model。

### 17.3 C-STL 与 C-DAE：物理模型完全相同

结构化 XML 比较结果如下：

- 37 个 link 名称和层级完全相同；
- 34 组 mass、inertial origin 和六项惯量矩阵完全相同；
- 31 个 revolute joint 和 5 个 fixed joint 的 parent/child、origin、axis、limit、effort、velocity 完全相同；
- 23 个 collision 的类型、origin、mesh 路径或 primitive 尺寸完全相同；
- 34 个 visual origin 相同，visual 文件扩展名从 `.stl` 换成 `.dae`；
- 因 collision 继续引用同一目录中的 STL，左右 ankle-pitch 的完整鞋脚 collision 也完全相同。

因此，**在相同仿真器和相同外部配置下，C-DAE 不会改变质量、惯量、关节运动学、关节限位或地面接触。**它只能改变渲染几何和材质。用 C-DAE 替换 C-STL 不能解释站立稳定性改善，也不能修复 sim2real 前倾。

C-DAE 的材质写法存在两个兼容性问题：

1. 11 个 visual 在同一个 `<visual>` 中放了两个 `<material>`，例如 pelvis 同时引用 `ID2` 和 `ID4`。
2. 共 25 个 `ID2`/`ID4` material reference 没有对应的 URDF 根级 material 定义；这些名字实际来自 DAE 内部材质，而不是标准的单一 URDF visual material 映射。

文件本身是 well-formed XML，但当前 MuJoCo 3.10 URDF importer 原样加载时首先报：

```text
XML Error: repeated element: 'material'
```

移除 URDF material 标签后还会报：

```text
Error: no decoder found for mesh file '...dae'
```

所以本工程的 MuJoCo sim2sim 应继续使用 C-STL，或先把 visual 做显式格式转换。以上只证明当前 MuJoCo 导入链路不兼容；本文没有本地 Gazebo 环境，不能据此断言 Gazebo/urdfdom 一定采用相同行为。

### 17.4 C-Plus 的基础机身是否改变

C-Plus 与 C-STL/C-DAE 共有 32 个承载 link：pelvis、双腿、三轴腰、torso、两轴 head 和双臂。对这些共同 link 的源参数逐项比较：

| 项目 | 结果 |
|---|---|
| 32 个 link 的 mass | 完全相同 |
| 32 个 link 的 inertial origin/COM | 完全相同 |
| 32 个 link 的六项 inertia matrix | 完全相同 |
| 31 个基础 revolute joint 的 parent/child | 完全相同 |
| 31 个基础 revolute joint 的 origin、axis | 完全相同 |
| 31 个基础 revolute joint 的 lower/upper、effort、velocity | 完全相同 |
| 基础 collision XML | 同名 link 上的标签和路径相同；`head_yaw_link.stl` 文件内容例外，见 17.6 |

C-Plus 从基础版本移除了以下 5 个 link：

```text
camera
imu_in_torso
imu_in_pelvis
left_hand_link
right_hand_link
```

前三个是零质量固定坐标 link，因此移除它们不改变总质量，但会让依赖这些 TF frame 的相机/IMU节点失效。后两个是每侧 `0.41885723 kg` 的简单固定手；C-Plus 用详细 Sharpa 手树替代。

两侧 `hand_joint` 的平移保持不变，但旋转和 child 改变：

| joint | C-STL/C-DAE | C-Plus |
|---|---|---|
| `left_hand_joint` | xyz `[0.05771524,0,-0.00014865]`，rpy `[0,0,0]`，child `left_hand_link` | xyz 不变，rpy `[1.57,3.1416,1.57]`，child `left_hand_C_MC` |
| `right_hand_joint` | xyz `[0.05824125,0,-0.00017221]`，rpy `[0,0,0]`，child `right_hand_link` | xyz 不变，rpy `[1.57,0,1.57]`，child `right_hand_C_MC` |

也就是说，C-Plus 不是“普通 H2 加两个同质量的手掌 visual”；它替换了手部坐标系、质量分布、关节树和接触几何。

### 17.5 Sharpa 双手的拓扑和质量增量

每只 Plus 手从对应 wrist-yaw 后增加 33 个 link：1 个掌座和 5 根手指。每手包含 22 个 revolute joint、10 个内部 fixed joint，再加 wrist 到掌座的 1 个 fixed `hand_joint`。

| 每手区域 | links | revolute | 内部 fixed | 左质量 kg | 右质量 kg | visual | collision |
|---|---:|---:|---:|---:|---:|---:|---:|
| 掌座 | 1 | 0 | 0 | 0.650600 | 0.650600 | 1 | 1 |
| thumb | 7 | 5 | 2 | 0.179211 | 0.177211 | 6 | 4 |
| index | 6 | 4 | 2 | 0.072511 | 0.072511 | 5 | 5 |
| middle | 6 | 4 | 2 | 0.072511 | 0.072511 | 5 | 5 |
| ring | 6 | 4 | 2 | 0.072511 | 0.072511 | 5 | 5 |
| pinky | 7 | 5 | 2 | 0.197921 | 0.197921 | 6 | 6 |
| **每手合计** | **33** | **22** | **10** | **1.245265** | **1.243265** | **28** | **26** |

左右手相差 `0.002 kg`，唯一质量差异来自 `left_thumb_DP=0.00988 kg`、`right_thumb_DP=0.00788 kg`。这可能是设计非对称，也可能是资产录入差异；仅凭 URDF 无法判断，应对照手部 CAD/BOM。

整机质量变化为：

```text
基础 H2 两个简单手：2 * 0.41885723 = 0.83771446 kg
Plus 两个详细手：   1.245265 + 1.243265 = 2.48853000 kg
Plus 净增：         2.48853000 - 0.83771446 = 1.65081554 kg
整机：              75.58745630 -> 77.23827184 kg
```

MuJoCo 导入固定结构后，掌座质量会折叠到 wrist-yaw body：

| body | C-STL 编译质量 | C-Plus 编译质量 | 原因 |
|---|---:|---:|---|
| `left_wrist_yaw_link` | 0.61608210 kg | 0.84782487 kg | `0.19722487 + 0.41885723` 变为 `0.19722487 + 0.6506` |
| `right_wrist_yaw_link` | 0.61608210 kg | 0.84782487 kg | 同上 |

其余手指质量保留在 44 个独立活动 body 中。C-Plus 没有 transmission、mimic 或 PID，因此这 44 个手指关节在 URDF 语义上全部独立；控制器若只提供基础 H2 的 31 个输出，并不会自动得到合理的手指保持力。

### 17.6 Visual 和 collision 的文件级差异

#### 基础机身 visual

C-Plus 的 32 个基础机身 visual 都使用同名 DAE。排除材质、颜色和 surface binding 后比较几何/scene 数据：

- 27 个基础 visual 的几何完全相同，只改变 DAE 内颜色/高光等材质；
- 5 个 visual 的几何确实改变：`head_pitch_link.dae`、`head_yaw_link.dae`、左右 `ankle_pitch_link.dae`、`torso_link.dae`。

| DAE visual | C-DAE polylist 面数 | C-Plus polylist 面数 | 结论 |
|---|---|---|---|
| `head_pitch_link.dae` | 788 | 1,144 + 1,020 + 1,196 | Plus 改为多材质、更详细几何 |
| `head_yaw_link.dae` | 20,533 + 1,925 | 9,677 + 7,838 + 1,837 | 几何重新组织并简化总面数 |
| `left_ankle_pitch_link.dae` | 5,303 + 1,078 | 16,086 + 28,988 + 1,256 | visual 大幅细化 |
| `right_ankle_pitch_link.dae` | 5,265 + 1,088 | 16,086 + 28,988 + 1,256 | visual 大幅细化 |
| `torso_link.dae` | 9,200 + 1,062 | 24 + 4,112 + 17,481 | visual 几何重新拆分并增加细节 |

注意：ankle-pitch 和 torso 的上述变化只发生在 DAE visual，不发生在对应 STL collision。

#### Collision

C-STL/C-DAE 有 23 个 collision，其中两个来自简单手；去掉简单手后，C-Plus 与基础机身共有 21 个 collision。C-Plus 的双手各增加 26 个 mesh collision，得到 `21 + 52 = 73` 个 collision。C-Plus 的 69 个 mesh collision 实例引用 39 条唯一文件路径，其中多根手指会复用同一个 STL。

基础机身 34 个同名 STL 文件中有 33 个 SHA-256 完全相同，唯一不同的是 `head_yaw_link.stl`。该文件同时被用作 head-yaw collision：

| 文件 | 三角面数 | bounding box min m | bounding box max m |
|---|---:|---|---|
| C-STL/C-DAE `head_yaw_link.stl` | 29,624 | `[-0.067922,-0.067480,-0.086595]` | `[0.097778,0.067482,0.116131]` |
| C-Plus `head_yaw_link.stl` | 22,458 | `[-0.067922,-0.067480,-0.085595]` | `[0.097778,0.067482,0.117131]` |

因此 Plus 的头部接触形状有小幅变化，脚底、腿、pelvis、torso 和手臂的基础 collision 文件则保持不变。特别是左右 ankle-pitch collision 与 C-STL 完全相同；C-Plus 没有采用模型 B 的 6 sphere + 1 box，也没有采用模型 D 的 capsule foot。

### 17.7 MuJoCo 编译核验

原始 C-DAE/C-Plus 因 17.3 的 material/DAE decoder 问题不能直接编译。为只审计物理模型，临时副本移除了全部 `<visual>`，保留 inertial、joint 和 collision 原文，并统一补 floating base。MuJoCo 3.10 编译结果：

C-Plus 的兼容性问题比 C-DAE 更广：23 个基础 visual 含多个 `<material>`，共 57 个空 material reference 没有根级定义；它的基础机身 visual 是 32 个 DAE，手部 visual 才是 56 个 STL。因此只处理重复 material 仍不足以让当前 MuJoCo 完整导入其 visual。

| 指标 | C-STL | C-DAE | C-Plus |
|---|---:|---:|---:|
| `nbody` | 33 | 33 | 77 |
| `njnt` | 32 | 32 | 76 |
| `nq` | 38 | 38 | 82 |
| `nv` | 37 | 37 | 81 |
| collision `ngeom` | 23 | 23 | 73 |
| collision `nmesh` | 19 | 19 | 34 |
| 总质量 kg | 75.58745630 | 75.58745630 | 77.23827184 |

这里 `ngeom` 是 collision 实例数，`nmesh` 是 MuJoCo importer 最终建立的 mesh asset 数，两者不能混用。C-Plus 源 URDF 有 69 个 mesh collision geom、39 条唯一 mesh 路径，编译后由 34 个 mesh asset 支撑这些 geom。

C-STL 与 C-DAE 的以下编译数组不仅数值近似，而是逐元素完全相同：

```text
body_mass, body_ipos, body_iquat, body_inertia
jnt_pos, jnt_axis, jnt_range
geom_pos, geom_quat, geom_size, geom_type, geom_bodyid
```

C-Plus 与 C-STL 的 31 个基础 hinge joint 在编译后仍保持相同 `jnt_pos/axis/range`。共同 body 中只有左右 `wrist_yaw_link` 的 importer 输出惯性不同，原因是不同的固定手掌质量被折叠到腕部；其他 30 个承载 body 的 importer 输出 mass/COM/principal inertia/iquat 相同。这里的 wrist importer 数值同样受 MuJoCo 3.10 fixed-link bug 影响，不能用于源物理参数比较。

还要注意 joint 顺序：C-Plus 的 22 个左手关节插在左腕和右肩之间，22 个右手关节位于右腕之后。任何硬编码“前 31 个 qpos/dof 就是基础 H2”的程序都会错位；必须按 joint name 建立映射。

### 17.8 双手对整机 COM 的影响

使用相同 floating-base pose、相同 31 个基础关节角，并把 C-Plus 的 44 个手指角设为零，得到：

| 姿态 | 模型 | COM - 左右 ankle-pitch 原点中点 `[x,y,z]` m | 前向 X |
|---|---|---|---:|
| 全零 | C-STL/C-DAE | `[0.002493,0.001575,0.978976]` | 2.493 mm |
| 全零 | C-Plus | `[0.010275,0.001544,0.983925]` | 10.275 mm |
| H2LoRA 默认站姿 | C-STL/C-DAE | `[0.079488,0.001576,0.927922]` | 79.488 mm |
| H2LoRA 默认站姿 | C-Plus | `[0.083580,0.001563,0.927522]` | 83.580 mm |

在 H2LoRA 默认站姿下，Plus 比基础 H2 的整机 COM 相对 ankle reference 向前 `4.091 mm`。全零姿态差值为 `7.782 mm`。这些数值直接按两份 URDF 的 link tree 计算，没有经过 MuJoCo fixed-link fusion。差值会随肩、肘、腕和 44 个手指关节姿态变化，不能用一个固定 ankle offset 代替。

这组结果只反映刚体参数和运动学，不包含 policy、PD、地面接触或执行器。如果拿 C-Plus 做站立 sim2sim，却仍使用针对普通手质量训练的 policy，那么 distal arm inertia 和整机 COM 都已经变了；仿真稳定性变化不能归因于 visual。

### 17.9 选型结论

1. **普通 H2 locomotion/sim2sim 基线使用 C-STL。**它与 C-DAE 物理等价，且当前 MuJoCo 可直接读取 STL。
2. **C-DAE 只适合需要 DAE 材质/渲染的链路。**使用前必须确认目标 parser 对多 material 的处理；它不提供任何动力学收益。
3. **只有实际机器人安装对应 Sharpa 双手时才使用 C-Plus。**还要给 44 个手指关节补 actuator/hold controller，并同步 policy observation/action mapping。
4. 若 locomotion policy 不控制手指，可以按 joint name 锁定手指，但不能简单删除手部质量；应把每只手在选定姿态下的复合惯性正确折叠到腕部，并明确处理 52 个手部 collision。
5. C-Plus 不会解决本项目的 ankle 接触差异：它的脚底 collision 仍是 C-STL 的完整 ankle-pitch STL。若要验证前倾问题，应该继续把足底 contact、普通/Plus 手质量和 actuator 作为三个独立变量。
6. 当前 `h2_lora_unitree_official_ros_mass(_debug)` 是普通 H2 31-DoF 实验，不应在未重新生成质量映射和 joint-name mapping 的情况下把 C-Plus 当作其替代源。

## 18. 模型 B 与模型 C 的版本血缘和一致性复核

### 18.1 先给结论

模型 B 和模型 C **不是两套从零独立建立、彼此冲突的 H2 标定**。模型 B 可以被精确追溯为模型 C 的 2026-04-24 历史快照，再进行 MuJoCo 专用改造。当前看起来“差很多”，由三类因素叠加：

1. B 从 ROS 历史快照分叉后没有继续同步，C 后续拆出了 hand 并更新了 head/camera；
2. B 有意把每脚完整 mesh collision 换成 6 sphere + 1 box；
3. B 的 pelvis 漏掉显式 inertial，而本工程将当前 C 直接导入 MuJoCo 时又命中 fixed-link inertia fusion bug。

把版本漂移和转换器错误剥离后，B/C 共同的 31-DoF 机身参数高度一致。真正值得作为 sim2real 不确定参数处理的是 pelvis 和 foot contact，而不是整套骨架。

### 18.2 可复现的版本时间线

| 日期 | 仓库/提交 | H2 变化 | 与 B 的关系 |
|---|---|---|---|
| 2026-02-25 | ROS [`ad5c7db5`](https://github.com/unitreerobotics/unitree_ros/commit/ad5c7db54c75c754dbbcef3e72369508c1575552) | 首次加入 `h2_description` | 早期版本，mesh 尚未对齐 B |
| 2026-04-24 16:57 | ROS [`31ab1062`](https://github.com/unitreerobotics/unitree_ros/commit/31ab1062ab1607f442451a179641aceefda3379c) | 更新 H2 URDF 和 mesh | B 的 32 个 mesh 全部来自这一版 |
| 2026-04-24 17:07 | ROS [`202c5987`](https://github.com/unitreerobotics/unitree_ros/commit/202c59879872d0c8d61ab20b93eaedd9ec04b144) | wrist-yaw effort 由 60 修正为 10 Nm | B 也使用修正后的 10 Nm；这是最精确的 B 源快照 |
| 2026-05-19 | ROS [`fceefdfd`](https://github.com/unitreerobotics/unitree_ros/commit/fceefdfd21389778ebb42419d503c7e5eee4a1be) | 将左右手从 wrist-yaw 整体 mesh/inertial 拆成 wrist + fixed hand | B 没有同步，仍保留拆分前表示 |
| 2026-06-04 | MuJoCo [`c1ccb767`](https://github.com/unitreerobotics/unitree_mujoco/commit/c1ccb767326e74bf659a787815cea7205b75a2d1) | 一次性加入 H2 MJCF | 文件内容对应 4 月 24 日 ROS 快照；之后 H2 路径无提交 |
| 2026-07-08 | ROS [`1cb2d0b6`](https://github.com/unitreerobotics/unitree_ros/commit/1cb2d0b66946460435f88458d29584e02f3ff92f) | 增加 camera/双 IMU fixed frame，并更新 `head_yaw_link.stl` | B 没有同步 head/camera 更新 |

B 虽然在 6 月 4 日才提交，但内容位于 ROS 的 4 月 24 日和 5 月 19 日两版之间。这说明 `feat/add_h2` 开发分支先复制了旧 H2 snapshot，随后独立转换和测试，最后再合入 `unitree_mujoco`；两个仓库没有用生成脚本或 source commit manifest 保持同步。

### 18.3 B 与 2026-04-24 C 快照到底有多一致

逐文件和逐参数核验结果：

| 项目 | 结果 |
|---|---|
| 32 个同名 STL | **32/32 SHA-256 完全相同** |
| 31 个 revolute joint parent/child | 完全相同 |
| 31 个 joint origin/rpy/axis/range | 完全相同到 XML 舍入 |
| effort/actuator force range | 与 4 月 24 日修正版一致 |
| 除 pelvis 外的 31 个 body mass | 最大差 `4.44e-6 kg`，为 MJCF 小数截断 |
| 除 pelvis 外的 31 个 body COM | 最大距离 `0.441 µm` |
| 除 pelvis 外的 31 个完整惯量矩阵 | 最大元素差 `4.31e-7 kg·m²` |
| collision | 除左右 ankle-pitch 外语义相同 |

4 月 ROS snapshot 有 21 个 collision。B 保留其中 19 个非足底 collision，只把每脚 1 个 ankle-pitch mesh 替换为 6 个 sphere + 1 个 box，所以 collision 数由 `21 - 2 + 14 = 33`。这不是随机换了一套 collision，而是明确的足底数值稳定化改造。

因此，B 与其源快照之间不存在 torso、腿、腰或手臂 mass/inertia 的系统性冲突。前文当前 B/C 的 27/32 mesh hash 一致，是因为当前 C 在分叉后改了 4 个 wrist mesh 和 1 个 head-yaw mesh；回到正确历史版本后是 32/32。

### 18.4 两项真正重要的 B/C 物理差异

#### Pelvis inertial

ROS 4 月快照已经明确写有：

```text
mass = 8.547700 kg
COM  = [0.00027091, -0.00003310, -0.01452551] m
```

B 的 root pelvis 是唯一没有 `<inertial>` 的 robot body。其 visual geom 为 `density=0`，collision pelvis mesh 没写 density，MuJoCo 因而按默认 mesh density 自动得到：

```text
mass = 4.248093 kg
COM  = [0.00000084, 0.00150909, -0.00324389] m
```

结果是 pelvis 少 `4.299607 kg`、少 `50.3%`，整机从 `75.587456 kg` 变成 `71.287850 kg`，少 `5.69%`。同仓库抽查的 G1、H1、H1-2、R1、Go2 和 B2 都给 root body 写了显式 inertial，只有 H2 缺失。上游提交没有解释这一选择，所以最符合现有证据的判断是：**H2 从 URDF 转 MJCF 时遗漏了 root inertial**，不是 Unitree 有意发布了第二套 4.248 kg pelvis 标定。

#### Foot contact

B 的 6 sphere + 1 box 是仿真接触代理，C 的 ankle-pitch STL 是几何 collision。两者用途不同：

- primitives 接触点数量和法向更可控，不容易受三角网格边缘、凸包化和接触切换影响；
- 完整 mesh 更接近外形，但不自动等于真实鞋底柔顺、摩擦和压力分布；
- B 更稳只能证明这组 primitives 对当前 MuJoCo/policy 更友好，不能证明它更接近真机。

### 18.5 当前 C 的 wrist“大差异”为何是假的

ROS 2026-05-19 将旧 wrist-yaw 分成 `wrist_yaw_link + fixed hand_link`。按 URDF joint transform 和平行轴定理正确合并后：

| 侧别 | B 旧版合并 COM m | 当前 C 源参数正确合并 COM m | 距离 |
|---|---|---|---:|
| left | `[0.08002670,-0.00476807,0.00340465]` | `[0.08019497,-0.00480351,0.00339767]` | 0.172 mm |
| right | `[0.08002670,0.00476817,0.00340501]` | `[0.08055259,0.00479790,0.00338165]` | 0.527 mm |

质量仍为每侧 `0.6160821 kg`；principal inertia 最大差分别只有 `1.086e-5` 和 `1.716e-7 kg·m²`。这说明拆手前后物理参数基本连续。

但当前 MuJoCo 3.10 importer 会把同一 C 错误融合为：

| 侧别 | importer output COM m | 相对 B 的假差异 |
|---|---|---:|
| left | `[0.04793240,0.03224722,0.01437451]` | 50.205 mm |
| right | `[0.03370130,0.04214319,-0.00748330]` | 60.510 mm |

最小双 link URDF 已在本机稳定复现同一错误；删除 `dont_collapse` 或将其设为 `false` 都不改变结果。MuJoCo 上游 [PR #3250](https://github.com/google-deepmind/mujoco/pull/3250) 直接描述了根因：`mjCBody::AccumulateInertia` 把 body pose 与 inertial offset 的 frame composition 顺序交换，导致 fixed child 的合并 COM 错误。当前该 PR 仍未合并，本机 MuJoCo `3.10.0` 命中该路径。

修复前，这会影响本工程以下产物：

- `gear_sonic/data/assets/robot_description/unitree_ros/h2_description/h2_mujoco.xml`；
- `unitree_mujoco/h2_ros_mass_aligned/h2_mujoco.xml`；
- `unitree_mujoco/h2_ros_mass_debug/h2_mujoco.xml`。

2026-07-19 已修改两个生成脚本，不再把 importer 输出作为 fixed-link 惯性的权威值，并重新生成上述三个 MJCF。当前三个产物左右 wrist 的复合质量、COM 和完整惯量都与源 URDF 一致；`ros_mass` 与 `ros_mass_debug` 现在仍只在 pelvis `iquat` 上不同。此前采集的两组旧日志依然来自错误 wrist 产物，解释旧实验时不能追溯性地当作已修复模型。

### 18.6 为什么两个“官方模型”仍会出现这些问题

“官方发布”在这里表示仓库和接口由 Unitree 提供，不表示存在单一 CAD source of truth、跨仓库自动生成、质量守恒 CI 或真机系统辨识认证。现有证据显示：

1. **仓库用途不同。**`unitree_ros` README 将其定位为 ROS/Gazebo robot description，并明确说明 Gazebo 不能做 walking high-level control；`unitree_mujoco` 则面向 LowCmd/LowState 低层控制器 sim2real，必须额外提供 actuator、sensor 和更稳定的 contact。
2. **发布流程是快照分叉。**B 的 32 个 mesh 精确锁定到 C 的 4 月 24 日状态，但 B 文件中没有记录该 source commit，ROS 后续更新也没有回灌。
3. **没有统一版本号或一致性测试。**B 的 H2 路径只有一次 `add h2` 提交；C 在此后继续修改 hands、camera 和 head mesh。两个仓库当前都没有检查 total mass、link hash 或 fixed-link composite inertia 的跨仓库 CI。
4. **接口文档本身也经历并行修正。**H2 加入 MuJoCo 后的 [issue #123](https://github.com/unitreerobotics/unitree_mujoco/issues/123) 曾发现 SDK ankle 顺序与文档/MuJoCo 不一致；维护者最终确认文档和 MuJoCo 顺序正确，并修正 SDK。这说明各发布面之间确实可能短期漂移，不能只凭“官方”跳过逐项验证。

把当前文件直接对比时看到的差异按成因归类如下：

| 表面差异 | 源物理差异大小 | 成因 | 是否应作为 sim2real 参数 |
|---|---:|---|---|
| 31 个活动关节 parent/origin/axis/range | 无，仅 XML 舍入 | B 直接继承 C 的 4 月快照 | 否 |
| 当前同名 STL 只有 27/32 hash 相同 | 5 个文件不同 | C 分叉后更新 4 个 wrist mesh 和 1 个 head mesh | visual 本身否；若用作 collision 则需单独检查 |
| 除 pelvis 外的 mass/COM/inertia | 可忽略，最大为微米/`1e-7` 量级 | 同一源参数转 MJCF 后的小数截断 | 否 |
| pelvis `8.547700 -> 4.248093 kg` | **很大** | B root 漏写显式 inertial，MuJoCo 从 mesh/density 自动推导 | **是，高优先级** |
| wrist-yaw 当前文件结构不同 | 源物理差异很小 | C 后续把旧合并 hand 拆成 fixed hand；复合质量保持不变 | 源参数通常否；必须正确做 fixed-link 合成 |
| wrist-yaw importer COM 相差 50-60 mm | **伪差异** | MuJoCo 3.10 fixed-child inertia fusion bug | 否；应先修生成器 |
| 每脚 mesh 与 6 sphere + 1 box | **很大** | B 为动态仿真主动设计 contact proxy | **是，高优先级** |
| actuator/sensor/default damping | 表示层差异，可影响闭环 | ROS URDF 不负责低层执行器；MJCF 必须可直接仿真 | 是，但本工程 wrapper 覆盖后要按有效值比较 |
| camera/IMU fixed frame | 对刚体动力学为零 | C 7 月新增 ROS frame，B 未同步 | 否；只影响 TF/sensor 接口 |

因此更准确的说法不是“Unitree 发布了两套彼此不一致的 H2 标定”，而是：**Unitree MuJoCo 发布了一份从 Unitree ROS 历史版本分叉的仿真产物，其中包含一项明确的接触建模改造、一项很可能的 pelvis 转换遗漏，以及未持续同步的版本差异。**本工程后续再次转换当前 ROS URDF 时，又叠加了第三方 MuJoCo importer 的 fixed-link bug。

### 18.7 对当前站立实验的直接解释

当前 `h2_lora_unitree_official` 比 ROS-mass 版本更容易站住，并不说明整套 B 比 C 更真实。B 相对正确 C 源物理模型，最有能力改变站立行为的只有：

1. **pelvis 自动惯性**：少 4.30 kg，COM 和 pitch/roll 惯量也改变；
2. **足底 primitives**：接触更规则、更宽容；
3. **仿真默认 joint damping/armature/frictionloss**：若运行时未完全覆盖，会进一步改变闭环响应。

当前本工程的 B/C sim2sim wrapper 会覆盖第 3 项，并可让不同模型共享 B 的 foot primitives。但 2026-07-19 复核发现，旧 bridge 还把 `mj_objectVelocity` 的惯性主轴 local frame 当成了 pelvis link frame。于是旧实验中 `iquat` 同时改变了真实 inertia tensor 和送给 policy 的 gyro 坐标轴，不能再把旧 B/C/debug 稳定性差异解释成单纯的 pelvis 完整惯量效应。

将 principal inertia 重建到 pelvis body frame 后：

| 配置 | pelvis mass kg | body-frame `Iyy` kg·m² | 说明 |
|---|---:|---:|---|
| B 官方 | 4.248093 | 0.01314690 | mesh 自动惯性，质量也更小 |
| `ros_mass` / C 正确惯性 | 8.547700 | 0.02269029 | 与 C URDF 完整矩阵一致 |
| `ros_mass_debug` | 8.547700 | 0.02723494 | 保留 C principal values，但套用 B 的近单位 `iquat` |

H2 body-frame pitch 轴是 Y，所以 debug 相对 C 确实把 pelvis pitch inertia 提高约 `20.03%`，同时把 `Ixy/Ixz` 交叉项压到接近零；这是客观存在的物理改动。但 debug 的 `iquat` 也接近单位四元数，恰好让旧 bridge 的错误 gyro frame 近似退化为正确 link frame。后者是旧实验中更直接、量级更大的混杂变量。修复 bridge 并重跑严格对照前，不能根据旧日志判断 20% pitch inertia 改动是否足以解释站立差异。

后续推荐基线：

1. 所有模型统一使用 link-frame gyro，先消除 `iquat -> observation frame` 的非物理耦合；
2. 以当前 C 的 31-DoF kinematics 和显式 inertial 为源参数，但手工正确合并 fixed hands；
3. 将 B foot primitives 与完整 foot mesh 作为独立可切换变量；
4. pelvis 同时保留 C 显式值、B mesh 自动值和围绕实机辨识值的随机化，不先假定任一官方值正确；
5. 保留当前 fixed-link 手工合成和源矩阵回归测试；在上游修复合入并完成同值验证前，不恢复为直接读取 importer wrist inertia；
6. 每个派生模型记录 source commit、总质量、整机 COM、32 个 mesh hash 和 31 个 joint-name mapping，避免再次发生静默版本漂移。

### 18.8 2026-07-19 ROS URDF 到 MuJoCo 实现复核

针对“同一 ROS URDF 在 IsaacLab 可训练，但当前 MuJoCo 产物完全站不住”的现象，进一步检查的不只是 visual，而是生成后的有效 inertial 和碰撞坐标。确认了两项本工程实现错误，并已修复。

#### Fixed-link 惯量不能直接采用 importer 输出

当前 MuJoCo 3.10 importer 对左右 fixed hand 的错误见 18.5。修复后的生成器会遍历所有 URDF inertial link，将 fixed child 的 COM 和完整惯量矩阵变换到最近的活动父 body，再按平行轴定理合成。重新生成后的关键结果为：

| 项目 | 修复后数值 |
|---|---:|
| 总质量 | `75.587456300 kg` |
| left wrist-yaw 复合 COM | `[0.080194973,-0.004803506,0.003397670] m` |
| right wrist-yaw 复合 COM | `[0.080552593,0.004797897,0.003381652] m` |
| 直接 C MJCF 与源复合惯量误差 | mass/COM `<1e-9`，完整矩阵 `<1e-8 kg·m²` |

同一逻辑也已用于 `h2_ros_mass_aligned` 和 `h2_ros_mass_debug`。当前两者质量、COM、principal inertia 和非 pelvis `iquat` 相同，只保留实验定义要求的 pelvis `iquat` 差异。

#### Mesh 足底高度曾被重复变换

MuJoCo 的 `mesh_vert` 已处于编译后的 mesh frame，而 `geom_xmat/geom_xpos` 已包含 mesh asset reference transform。旧 `_geom_bottom_z()` 又额外应用一次 `mesh_quat/mesh_pos`，相当于重复变换。

H2LoRA 默认站姿下的实测结果：

| 项目 | 修复前 | 修复后 |
|---|---:|---:|
| reset pelvis Z | `1.038632793 m` | `0.998292379 m` |
| 代码认为的足底 clearance | `2.000 mm` | `2.000 mm` |
| MuJoCo 实际 collision clearance | `42.340 mm` | `2.000 mm` |
| 再下沉 3 mm 后地面接触数 | `0` | `2` |

因此修复前机器人实际比预期多悬空 `40.340 mm`。若没有弹力带重新约束高度，释放后约经历 `0.9 m/s` 量级的自由落地速度；即使有弹力带，这也会让初始几何高度、释放过程和不同 contact 模型的比较失去一致性。

#### 可视化正确不等于接触等价

IsaacLab 和 MuJoCo 都能把同一 STL 画在正确位置，但渲染 mesh 不参与判断以下事项：fixed-link 复合惯量是否正确、碰撞近似是 convex hull 还是 decomposition、接触 manifold 如何生成，以及 solver/material 参数是否一致。因此 IsaacLab rollout 外观正常只能排除 visual/frame 的显著错误，不能证明导出的 MuJoCo 动力学等价。

修复后仍应把两种足底作为独立变量：

- 原始 ROS ankle mesh：`GEAR_SONIC_MUJOCO_H2_OFFICIAL_FOOT_CONTACTS=0`；
- Unitree MuJoCo 6 sphere + 1 box：`GEAR_SONIC_MUJOCO_H2_OFFICIAL_FOOT_CONTACTS=1`。

两者现在共享同一套 C 运动学、正确 fixed-hand 惯量、执行器覆盖和 reset clearance。实际复测中两者都在卸载弹力带后失稳：mesh 约在控制开始后 1-2 秒明显发散，primitives 约在 2-3 秒发散。primitives 只延迟失稳，说明 collision 不是共同根因。随后发现的 gyro frame 错误见 18.9。

### 18.9 Pelvis gyro 曾错误使用惯性主轴坐标系

H2LoRA 训练的 `base_ang_vel` 是 pelvis/root **link frame** 角速度。旧 MuJoCo bridge 为获得 local gyro，调用：

```python
mujoco.mj_objectVelocity(model, data, mjOBJ_BODY, pelvis_id, velocity, 1)
```

问题在于，`mjOBJ_BODY + local=1` 使用 `ximat`，即 body inertial principal-axis frame；它不是由 `xmat` 表示的 link frame。用模型 C 构造已知 free-joint 角速度 `[0.4, 0.5, 0.6] rad/s` 复核时：

| 表示 | 得到的角速度 rad/s |
|---|---|
| pelvis link frame，应提供给 policy | `[0.400000, 0.500000, 0.600000]` |
| 旧 `local=1` 输出 | `[0.402847, 0.595259, -0.503370]` |
| `ximat.T @ omega_world` | 与旧输出逐值相同 |
| `xmat.T @ omega_world` | 与正确 link-frame 值逐值相同 |

模型 C 编译后的 pelvis `iquat` 为 `[0.707868, 0.706283, -0.009313, 0.001018]`，近似绕 X 轴 90 度。因此忽略小角误差后，旧输出相当于：

```text
gyro_x ~=  omega_x
gyro_y ~=  omega_z
gyro_z ~= -omega_y
```

两组失败日志也直接支持该结论。在失稳前 0.1-1.0 秒，把 pelvis quaternion 差分得到的 link-frame 角速度拟合到日志 gyro：

| 日志 | `omega_x -> gyro_x` 相关系数 | `omega_z -> gyro_y` 相关系数 | `omega_y -> gyro_z` 相关系数 |
|---|---:|---:|---:|
| C mesh | `0.997` | `0.995` | `-0.983` |
| C + B foot primitives | `0.994` | `0.996` | `-0.989` |
| 同一 policy 的实机日志 | 三个同轴相关系数约 `0.956/0.957/0.985` | 不存在 Y/Z 交换 | 不存在符号翻转 |

这也统一解释了两个此前看似独立的现象：

1. 模型 A 的 pelvis `iquat=[1,0,0,0]`、模型 B 和 `ros_mass_debug` 的 pelvis `iquat` 均接近单位四元数，所以旧 bridge 的错误在这些配置上几乎不可见；
2. `ros_mass_debug` 只改 pelvis `iquat` 后明显更稳，不仅改变了物理 inertia tensor，也意外把 policy gyro 恢复到了近似正确的轴序。旧实验不能用于单独证明 pelvis pitch inertia 更大更稳定。

当前实现改为先取 world-frame object velocity，再显式计算 `xmat.T @ velocity_world`，并对 pelvis 和 torso 都输出 `[linear, angular]` link-frame velocity。回归测试特意使用模型 C 的非单位 `iquat`，确保以后修改 inertia principal-axis 表示不会再次改变 IMU observation frame。

修复后又用同一 decoder、同一模型 C inertial 和无弹力带 headless 仿真复测：

| 脚底配置 | 自由闭环时长 | 结果 | 末段姿态 |
|---|---:|---|---|
| C 原始 ankle mesh | `8.76 s` | 持续站立，无跌倒重置 | 末帧 roll `0.003 deg`、pitch `0.251 deg` |
| B 6 sphere + 1 box | `22.24 s` | 持续站立，无跌倒重置 | 最后 17 s roll 约 `-0.84 deg`、pitch `0.55-0.60 deg` |

第二组为了自动化验证，在 deploy 完成初始化前没有弹力带支撑，policy 接管首帧 pelvis pitch 已约 `70 deg`；其在约 `0.5 s` 内恢复到接近直立，随后保持稳定。因此这不是弹力带把机器人托住的假阳性。修复后日志的 quaternion 差分角速度与 gyro 同轴相关恢复为 mesh `0.84/0.96/0.98`、primitives `0.98/0.91/0.99`，原来的 Y/Z 交换消失。

结论是：两种 collision 仍会改变稳态姿态和恢复过程，但先前“两者都站不住”的共同根因是 observation frame 错误。修复后的结果才可以继续用于分离 mesh/primitives、pelvis inertia 和 PhysX/MuJoCo 接触差异。

## 附录 1：模型 A/B/C 的 31 个活动关节逐项对比

说明：

- `C/B origin` 表示 C 与 B 的运动学原点一致到文件舍入精度；
- `rpy` 和 `axis` 三者相同；
- range 列为 `A / C`，B 与 C 相同但小数位较少；
- effort/velocity 是源 URDF 元数据，不是 H2_CFG 的实际训练限制；
- B 只有 torque ctrl range，没有 velocity limit。

| Joint | A origin xyz (m) | C/B origin xyz (m) | Δxyz mm | origin rpy | axis | range A / C (rad) | effort A/C Nm | velocity A/C rad/s |
|---|---|---|---:|---|---|---|---:|---:|
| `left_hip_pitch_joint` | `0 0.083873 -0.048424` | `0 0.080380 -0.046407` | 4.034 | `-0.5236 0 0` | `0 1 0` | `[-2.452597,2.775422] / [-2.452597,2.775422]` | 360/360 | 20/20 |
| `left_hip_roll_joint` | `0.041851 0.063500 -0.019000` | `0.041774 0.063500 -0.019000` | 0.077 | `0.5236 0 0` | `1 0 0` | `[-0.467441,2.168861] / [-0.467441,2.168861]` | 360/360 | 20/20 |
| `left_hip_yaw_joint` | `-0.028187 -0.009500 -0.073000` | `-0.028187 -0.009500 -0.073700` | 0.700 | `0 0 0` | `0 0 1` | `[-2.827,2.827] / [-2.827,2.827]` | 360/360 | 20/20 |
| `left_knee_joint` | `-0.010341 0.018489 -0.319100` | `-0.010308 0.022549 -0.319590` | 4.090 | `0 0 0` | `0 1 0` | `[-0.08725,2.53025] / [-0.08725,2.53025]` | 360/360 | 20/20 |
| `left_ankle_roll_joint` | `0.032720 -0.023002 -0.496620` | `0.032719 -0.023002 -0.496620` | 0.001 | `0 0 0` | `1 0 0` | `[-0.349066,0.296706] / [-0.349066,0.296706]` | 19/19 | 100.7/100.7 |
| `left_ankle_pitch_joint` | `-0.022500 -0.037300 0` | `-0.014516 -0.037300 0` | 7.984 | `0 0 0` | `0 1 0` | `[-1.134464,0.610865] / [-1.134464,0.610865]` | 66.88/66.88 | 28.61/28.61 |
| `right_hip_pitch_joint` | `0 -0.083873 -0.048424` | `0 -0.080380 -0.046407` | 4.034 | `0.5236 0 0` | `0 1 0` | `[-2.452597,2.775422] / [-2.452597,2.775422]` | 360/360 | 20/20 |
| `right_hip_roll_joint` | `0.041851 -0.063500 -0.019000` | `0.041774 -0.063500 -0.019000` | 0.077 | `-0.5236 0 0` | `1 0 0` | `[-2.168861,0.467441] / [-2.168861,0.467441]` | 360/360 | 20/20 |
| `right_hip_yaw_joint` | `-0.028187 0.009500 -0.073000` | `-0.028187 0.009500 -0.073700` | 0.700 | `0 0 0` | `0 0 1` | `[-2.827,2.827] / [-2.827,2.827]` | 360/360 | 20/20 |
| `right_knee_joint` | `-0.010341 -0.018489 -0.319100` | `-0.010308 -0.022549 -0.319590` | 4.090 | `0 0 0` | `0 1 0` | `[-0.08725,2.53025] / [-0.08725,2.53025]` | 360/360 | 20/20 |
| `right_ankle_roll_joint` | `0.032720 0.023002 -0.496620` | `0.032719 0.023002 -0.496620` | 0.001 | `0 0 0` | `1 0 0` | `[-0.296706,0.349066] / [-0.296706,0.349066]` | 19/19 | 100.7/100.7 |
| `right_ankle_pitch_joint` | `-0.022500 0.037300 0` | `-0.014516 0.037300 0` | 7.984 | `0 0 0` | `0 1 0` | `[-1.134464,0.610865] / [-1.134464,0.610865]` | 66.88/66.88 | 28.61/28.61 |
| `waist_yaw_joint` | `0 0 0.089837` | `0 0 0.090537` | 0.700 | `0 0 0` | `0 0 1` | `[-1.7453,1.7453] / [-1.7453,1.7453]` | 120/120 | 28.375/28.375 |
| `waist_roll_joint` | `0 0 0` | `0 0 0.032500` | 32.500 | `0 0 0` | `1 0 0` | `[-0.5236,0.5236] / [-0.5236,0.5236]` | 180/180 | 28.375/28.375 |
| `waist_pitch_joint` | `0 0 0` | `0 0 0` | 0 | `0 0 0` | `0 1 0` | `[-0.43633,0.5236] / [-0.43633,0.5236]` | 180/180 | 28.375/28.375 |
| `head_pitch_joint` | `-0.000228 0.000027 0.421810` | `-0.000115 -0.018900 0.421800` | 18.927 | `0 0 0` | `0 1 0` | `[-0.5236,0.83775] / [-0.5236,0.83775]` | 50/50 | 10/10 |
| `head_yaw_joint` | `0 0.000300 0.132500` | `0 0.018900 0.133500` | 18.627 | `0 0 0` | `0 0 1` | `[-1.7453,1.7453] / [-1.7453,1.7453]` | 50/50 | 10/10 |
| `left_shoulder_pitch_joint` | `0.001500 0.112540 0.358120` | `0.003085 0.110000 0.360080` | 3.578 | `0.26914 0 0` | `0 1 0` | `[-2.617994,1.832596] / [-2.618,1.833]` | 120/130 | 28.375/21.9 |
| `left_shoulder_roll_joint` | `0.027682 0.061638 -0.039909` | `-0.003207 0.059938 -0.039865` | 30.936 | `-0.26914 0 0` | `1 0 0` | `[-0.516617,2.635447] / [-0.517,2.494]` | 54/60 | 34.3/18.7 |
| `left_shoulder_yaw_joint` | `-0.030632 0.003781 -0.104220` | `-0.002560 0.003675 -0.104210` | 28.072 | `0 0 0` | `0 0 1` | `[-2.617994,2.617994] / [-2.618,2.618]` | 54/60 | 34.3/18.7 |
| `left_elbow_joint` | `0.016955 0.031006 -0.127190` | `0.016655 0.022110 -0.127920` | 8.931 | `0 0 0` | `0 1 0` | `[-0.986111,3.071779] / [-0.986,3.071]` | 54/60 | 34.3/18.7 |
| `left_wrist_roll_joint` | `0.128260 -0.030815 -0.013299` | `0.131190 -0.021497 -0.013769` | 9.779 | `0 0 0` | `1 0 0` | `[-2.617994,2.617994] / [-2.618,2.618]` | 54/60 | 34.3/18.7 |
| `left_wrist_pitch_joint` | `0.082003 -0.031003 0.000208` | `0.037994 0.031144 0.000451` | 76.152 | `0 0 0` | `0 1 0` | `[-0.436332,0.436332] / [-0.576,0.576]` | 25/10 | 50.125/37.7 |
| `left_wrist_yaw_joint` | `0.026025 0.030996 0.026981` | `0.069283 -0.031045 -0.000625` | 80.513 | `0 0 0` | `0 0 1` | `[-1.221731,1.221731] / [-1.22,1.22]` | 25/10 | 50.125/37.7 |
| `right_shoulder_pitch_joint` | `0.001500 -0.112500 0.358120` | `0.003085 -0.110000 0.360080` | 3.550 | `-0.26914 0 0` | `0 1 0` | `[-2.617994,1.832596] / [-2.618,1.833]` | 120/130 | 28.375/21.9 |
| `right_shoulder_roll_joint` | `0.027682 -0.061638 -0.039909` | `-0.003207 -0.059938 -0.039865` | 30.936 | `0.26914 0 0` | `1 0 0` | `[-2.635447,0.516617] / [-2.494,0.517]` | 54/60 | 34.3/18.7 |
| `right_shoulder_yaw_joint` | `-0.030632 -0.003781 -0.104220` | `-0.002560 -0.003675 -0.104210` | 28.072 | `0 0 0` | `0 0 1` | `[-2.617994,2.617994] / [-2.618,2.618]` | 54/60 | 34.3/18.7 |
| `right_elbow_joint` | `0.016955 -0.031006 -0.127190` | `0.016655 -0.022110 -0.127920` | 8.931 | `0 0 0` | `0 1 0` | `[-0.986111,3.071779] / [-0.986,3.071]` | 54/60 | 34.3/18.7 |
| `right_wrist_roll_joint` | `0.128260 0.030815 -0.013299` | `0.131190 0.021497 -0.013769` | 9.779 | `0 0 0` | `1 0 0` | `[-2.617994,2.617994] / [-2.618,2.618]` | 54/60 | 34.3/18.7 |
| `right_wrist_pitch_joint` | `0.082003 0.031003 0.000208` | `0.037994 -0.031144 0.000451` | 76.152 | `0 0 0` | `0 1 0` | `[-0.436332,0.436332] / [-0.576,0.576]` | 25/10 | 50.125/37.7 |
| `right_wrist_yaw_joint` | `0.026025 -0.030996 0.026981` | `0.069283 0.031045 -0.000625` | 80.513 | `0 0 0` | `0 0 1` | `[-1.221731,1.221731] / [-1.22,1.22]` | 25/10 | 50.125/37.7 |

## 附录 2：模型 D 逐关节/body origin 对比

### 2.1 模型 D 与 A/B/C 的局部 origin

模型 B/C 的活动 joint origin 一致到舍入精度。D 的 head 两行列出固定 body origin，不表示存在 head joint。距离单位为 mm。

| Joint/body | 模型 A origin | 模型 B/C origin | 模型 D origin | D-A | D-C | D 状态 |
|---|---|---|---|---:|---:|---|
| `left_hip_pitch_joint` | `0 0.083873 -0.048424` | `0 0.080380 -0.046407` | `0 0.081928 -0.043723` | 5.088 | 3.098 | 活动 |
| `left_hip_roll_joint` | `0.041851 0.063500 -0.019000` | `0.041774 0.063500 -0.019000` | `0.018422 0.063500 -0.022095` | 23.633 | 23.556 | 活动 |
| `left_hip_yaw_joint` | `-0.028187 -0.009500 -0.073000` | `-0.028187 -0.009500 -0.073700` | `-0.004743 -0.005733 -0.075875` | 23.918 | 23.844 | 活动 |
| `left_knee_joint` | `-0.010341 0.018489 -0.319100` | `-0.010308 0.022549 -0.319590` | `-0.010386 -0.004218 -0.317420` | 22.769 | 26.855 | 活动 |
| `left_ankle_roll_joint` | `0.032720 -0.023002 -0.496620` | `0.032719 -0.023002 -0.496620` | `0.018383 0 -0.496620` | 27.104 | 27.104 | 活动 |
| `left_ankle_pitch_joint` | `-0.022500 -0.037300 0` | `-0.014516 -0.037300 0` | `0 -0.036800 0` | 22.506 | 14.525 | 活动 |
| `right_hip_pitch_joint` | `0 -0.083873 -0.048424` | `0 -0.080380 -0.046407` | `0 -0.080380 -0.046403` | 4.035 | 0.004 | 活动 |
| `right_hip_roll_joint` | `0.041851 -0.063500 -0.019000` | `0.041774 -0.063500 -0.019000` | `0.037774 -0.063500 -0.019000` | 4.077 | 4.000 | 活动 |
| `right_hip_yaw_joint` | `-0.028187 0.009500 -0.073000` | `-0.028187 0.009500 -0.073700` | `-0.024111 0.005733 -0.075875` | 6.251 | 5.961 | 活动 |
| `right_knee_joint` | `-0.010341 -0.018489 -0.319100` | `-0.010308 -0.022549 -0.319590` | `-0.010386 0.004218 -0.317420` | 22.769 | 26.855 | 活动 |
| `right_ankle_roll_joint` | `0.032720 0.023002 -0.496620` | `0.032719 0.023002 -0.496620` | `0.018383 0 -0.496620` | 27.104 | 27.104 | 活动 |
| `right_ankle_pitch_joint` | `-0.022500 0.037300 0` | `-0.014516 0.037300 0` | `0 0.037300 0` | 22.500 | 14.516 | 活动 |
| `waist_yaw_joint` | `0 0 0.089837` | `0 0 0.090537` | `0 0.001510 0.090537` | 1.664 | 1.510 | 活动 |
| `waist_roll_joint` | `0 0 0` | `0 0 0.032500` | `0 -0.001510 0.032500` | 32.535 | 1.510 | 活动 |
| `waist_pitch_joint` | `0 0 0` | `0 0 0` | `0 0 0` | 0 | 0 | 活动 |
| `head_pitch_link` | `-0.000228 0.000027 0.421810` | `-0.000115 -0.018900 0.421800` | `-0.000115 0.020709 0.421810` | 20.682 | 39.609 | 固定 body |
| `head_yaw_link` | `0 0.000300 0.132500` | `0 0.018900 0.133500` | `0 -0.018900 0.132500` | 19.200 | 37.813 | 固定 body |
| `left_shoulder_pitch_joint` | `0.001500 0.112540 0.358120` | `0.003085 0.110000 0.360080` | `0.001978 0.115530 0.357920` | 3.035 | 6.039 | 活动 |
| `left_shoulder_roll_joint` | `0.027682 0.061638 -0.039909` | `-0.003207 0.059938 -0.039865` | `-0.000822 0.060413 -0.040043` | 28.530 | 2.439 | 活动 |
| `left_shoulder_yaw_joint` | `-0.030632 0.003781 -0.104220` | `-0.002560 0.003675 -0.104210` | `-0.002595 0.003806 -0.096230` | 29.154 | 7.981 | 活动 |
| `left_elbow_joint` | `0.016955 0.031006 -0.127190` | `0.016655 0.022110 -0.127920` | `0.015424 0.020635 -0.133640` | 12.309 | 6.034 | 活动 |
| `left_wrist_roll_joint` | `0.128260 -0.030815 -0.013299` | `0.131190 -0.021497 -0.013769` | `0.131190 -0.021207 -0.013343` | 10.045 | 0.515 | 活动 |
| `left_wrist_pitch_joint` | `0.082003 -0.031003 0.000208` | `0.037994 0.031144 0.000451` | `0.080589 0.000030 -0.001332` | 31.104 | 52.778 | 活动 |
| `left_wrist_yaw_joint` | `0.026025 0.030996 0.026981` | `0.069283 -0.031045 -0.000625` | `0.028225 0.000489 0.000583` | 40.402 | 51.784 | 活动 |
| `right_shoulder_pitch_joint` | `0.001500 -0.112500 0.358120` | `0.003085 -0.110000 0.360080` | `0.001971 -0.111850 0.357890` | 0.835 | 3.076 | 活动 |
| `right_shoulder_roll_joint` | `0.027682 -0.061638 -0.039909` | `-0.003207 -0.059938 -0.039865` | `-0.000826 -0.060403 -0.040059` | 28.535 | 2.434 | 活动 |
| `right_shoulder_yaw_joint` | `-0.030632 -0.003781 -0.104220` | `-0.002560 -0.003675 -0.104210` | `-0.002595 -0.003781 -0.096231` | 29.153 | 7.980 | 活动 |
| `right_elbow_joint` | `0.016955 -0.031006 -0.127190` | `0.016655 -0.022110 -0.127920` | `0.015346 -0.020599 -0.133630` | 12.344 | 6.050 | 活动 |
| `right_wrist_roll_joint` | `0.128260 0.030815 -0.013299` | `0.131190 0.021497 -0.013769` | `0.131770 0.021197 -0.013782` | 10.250 | 0.653 | 活动 |
| `right_wrist_pitch_joint` | `0.082003 0.031003 0.000208` | `0.037994 -0.031144 0.000451` | `0.076351 -0.000159 0.001146` | 31.684 | 49.314 | 活动 |
| `right_wrist_yaw_joint` | `0.026025 -0.030996 0.026981` | `0.069283 0.031045 -0.000625` | `0.028195 0.000058 -0.001218` | 42.003 | 51.466 | 活动 |

### 2.2 模型 D 的 29 个活动关节与 actuator 完整映射

下表来自 MjLab `Entity(get_h2_robot_cfg()).spec.compile()` 的实际编译结果。模型 D 的 position actuator 与同名 joint 一一映射；action scale 单位为 rad/normalized-action。固定的 head pitch/yaw 不在表内。

| D joint | parent -> child | axis | range (rad) | actuator | Kp | Kd | effort (Nm) | action scale |
|---|---|---|---|---|---:|---:|---:|---:|
| `left_hip_pitch_joint` | `pelvis -> left_hip_pitch_link` | `0 1 0` | `[-2.452600,2.775420]` | `left_hip_pitch_joint` | 200 | 4 | 360 | 0.450000 |
| `left_hip_roll_joint` | `left_hip_pitch_link -> left_hip_roll_link` | `1 0 0` | `[-0.467441,2.168860]` | `left_hip_roll_joint` | 200 | 4 | 360 | 0.450000 |
| `left_hip_yaw_joint` | `left_hip_roll_link -> left_hip_yaw_link` | `0 0 1` | `[-2.827000,2.827000]` | `left_hip_yaw_joint` | 200 | 4 | 360 | 0.450000 |
| `left_knee_joint` | `left_hip_yaw_link -> left_knee_link` | `0 1 0` | `[-0.087250,2.530250]` | `left_knee_joint` | 200 | 4 | 360 | 0.450000 |
| `left_ankle_roll_joint` | `left_knee_link -> left_ankle_roll_link` | `1 0 0` | `[-0.349066,0.296706]` | `left_ankle_roll_joint` | 40 | 2 | 19 | 0.118750 |
| `left_ankle_pitch_joint` | `left_ankle_roll_link -> left_ankle_pitch_link` | `0 1 0` | `[-1.134460,0.610865]` | `left_ankle_pitch_joint` | 40 | 2 | 66 | 0.412500 |
| `right_hip_pitch_joint` | `pelvis -> right_hip_pitch_link` | `0 1 0` | `[-2.452600,2.775420]` | `right_hip_pitch_joint` | 200 | 4 | 360 | 0.450000 |
| `right_hip_roll_joint` | `right_hip_pitch_link -> right_hip_roll_link` | `1 0 0` | `[-2.168860,0.467441]` | `right_hip_roll_joint` | 200 | 4 | 360 | 0.450000 |
| `right_hip_yaw_joint` | `right_hip_roll_link -> right_hip_yaw_link` | `0 0 1` | `[-2.827000,2.827000]` | `right_hip_yaw_joint` | 200 | 4 | 360 | 0.450000 |
| `right_knee_joint` | `right_hip_yaw_link -> right_knee_link` | `0 1 0` | `[-0.087250,2.530250]` | `right_knee_joint` | 200 | 4 | 360 | 0.450000 |
| `right_ankle_roll_joint` | `right_knee_link -> right_ankle_roll_link` | `1 0 0` | `[-0.296706,0.349066]` | `right_ankle_roll_joint` | 40 | 2 | 19 | 0.118750 |
| `right_ankle_pitch_joint` | `right_ankle_roll_link -> right_ankle_pitch_link` | `0 1 0` | `[-1.134460,0.610865]` | `right_ankle_pitch_joint` | 40 | 2 | 66 | 0.412500 |
| `waist_yaw_joint` | `pelvis -> waist_yaw_link` | `0 0 1` | `[-1.745300,1.745300]` | `waist_yaw_joint` | 150 | 3 | 120 | 0.200000 |
| `waist_roll_joint` | `waist_yaw_link -> waist_roll_link` | `1 0 0` | `[-0.523600,0.523600]` | `waist_roll_joint` | 150 | 3 | 120 | 0.200000 |
| `waist_pitch_joint` | `waist_roll_link -> torso_link` | `0 1 0` | `[-0.436330,0.523600]` | `waist_pitch_joint` | 150 | 3 | 120 | 0.200000 |
| `left_shoulder_pitch_joint` | `torso_link -> left_shoulder_pitch_link` | `0 1 0` | `[-2.617990,1.832600]` | `left_shoulder_pitch_joint` | 40 | 2 | 54 | 0.337500 |
| `left_shoulder_roll_joint` | `left_shoulder_pitch_link -> left_shoulder_roll_link` | `1 0 0` | `[-0.516617,2.635450]` | `left_shoulder_roll_joint` | 40 | 2 | 54 | 0.337500 |
| `left_shoulder_yaw_joint` | `left_shoulder_roll_link -> left_shoulder_yaw_link` | `0 0 1` | `[-2.617990,2.617990]` | `left_shoulder_yaw_joint` | 40 | 2 | 54 | 0.337500 |
| `left_elbow_joint` | `left_shoulder_yaw_link -> left_elbow_link` | `0 1 0` | `[-0.986111,3.071780]` | `left_elbow_joint` | 40 | 2 | 54 | 0.337500 |
| `left_wrist_roll_joint` | `left_elbow_link -> left_wrist_roll_link` | `1 0 0` | `[-2.617990,2.617990]` | `left_wrist_roll_joint` | 40 | 2 | 54 | 0.337500 |
| `left_wrist_pitch_joint` | `left_wrist_roll_link -> left_wrist_pitch_link` | `0 1 0` | `[-0.436332,0.436332]` | `left_wrist_pitch_joint` | 20 | 1 | 25 | 0.312500 |
| `left_wrist_yaw_joint` | `left_wrist_pitch_link -> left_wrist_yaw_link` | `0 0 1` | `[-1.221730,1.221730]` | `left_wrist_yaw_joint` | 20 | 1 | 25 | 0.312500 |
| `right_shoulder_pitch_joint` | `torso_link -> right_shoulder_pitch_link` | `0 1 0` | `[-2.617990,1.832600]` | `right_shoulder_pitch_joint` | 40 | 2 | 54 | 0.337500 |
| `right_shoulder_roll_joint` | `right_shoulder_pitch_link -> right_shoulder_roll_link` | `1 0 0` | `[-2.635450,0.516617]` | `right_shoulder_roll_joint` | 40 | 2 | 54 | 0.337500 |
| `right_shoulder_yaw_joint` | `right_shoulder_roll_link -> right_shoulder_yaw_link` | `0 0 1` | `[-2.617990,2.617990]` | `right_shoulder_yaw_joint` | 40 | 2 | 54 | 0.337500 |
| `right_elbow_joint` | `right_shoulder_yaw_link -> right_elbow_link` | `0 1 0` | `[-0.986111,3.071780]` | `right_elbow_joint` | 40 | 2 | 54 | 0.337500 |
| `right_wrist_roll_joint` | `right_elbow_link -> right_wrist_roll_link` | `1 0 0` | `[-2.617990,2.617990]` | `right_wrist_roll_joint` | 40 | 2 | 54 | 0.337500 |
| `right_wrist_pitch_joint` | `right_wrist_roll_link -> right_wrist_pitch_link` | `0 1 0` | `[-0.436332,0.436332]` | `right_wrist_pitch_joint` | 20 | 1 | 25 | 0.312500 |
| `right_wrist_yaw_joint` | `right_wrist_pitch_link -> right_wrist_yaw_link` | `0 0 1` | `[-1.221730,1.221730]` | `right_wrist_yaw_joint` | 20 | 1 | 25 | 0.312500 |

## 附录 3：模型 A/B/C 逐 body 质量、COM 和主惯量

约定：

- 每行 mass 顺序为 `A / B / C`，单位 kg；
- COM 为各自 body frame 下的 `[x y z]`，单位 m；
- principal inertia 为 `[I1 I2 I3]`，单位 kg·m²；
- 除 A/C 的 fixed-hand 复合 wrist-yaw 外，数字来自统一 MuJoCo 编译结果；A/C wrist-yaw 使用源 URDF 完整惯量矩阵和 fixed-joint transform 按平行轴定理手工合成，避免 MuJoCo 3.10 importer bug。

### 3.1 pelvis、双腿、腰、躯干和头部

| Body | mass A/B/C | COM A | COM B | COM C | principal A | principal B | principal C |
|---|---|---|---|---|---|---|---|
| `pelvis` | 7.604681 / 4.248093 / 8.547700 | `0.00046244 -0.00004214 -0.01473435` | `0.00000084 0.00150909 -0.00324389` | `0.00027091 -0.00003310 -0.01452551` | `0.03551368 0.02753239 0.02287793` | `0.01635533 0.01314690 0.01283785` | `0.03527720 0.02723494 0.02268854` |
| `left_hip_pitch_link` | 2.797525 / 2.873700 / 2.873700 | `0.00300595 0.05931469 -0.01709150` | `0.00294368 0.06110200 -0.01735100` | `0.00294368 0.06110195 -0.01735095` | `0.005102974 0.004100852 0.003459304` | `0.005374080 0.004211630 0.003687800` | `0.005374080 0.004211634 0.003687795` |
| `left_hip_roll_link` | 0.627146 / 0.785079 / 0.785079 | `-0.03026193 -0.00468930 -0.04449852` | `-0.01757890 -0.00607425 -0.04131500` | `-0.01757892 -0.00607425 -0.04131500` | `0.002002522 0.001675094 0.001280234` | `0.002377800 0.002004430 0.001515300` | `0.002377798 0.002004433 0.001515299` |
| `left_hip_yaw_link` | 7.087520 / 7.087500 / 7.087500 | `0.00069408 -0.01086587 -0.11301460` | `0.00069408 -0.01086590 -0.11301500` | `0.00069408 -0.01086587 -0.11301460` | `0.05140120 0.05059600 0.01300672` | `0.05140130 0.05059640 0.01300630` | `0.05140131 0.05059642 0.01300626` |
| `left_knee_link` | 3.553402 / 4.353400 / 4.353402 | `0.00461331 -0.02453217 -0.16715460` | `0.00461331 -0.02254900 -0.16715500` | `0.00461331 -0.02254900 -0.16715460` | `0.05123878 0.05108841 0.003339976` | `0.05123880 0.05108840 0.003339980` | `0.05123878 0.05108841 0.003339976` |
| `left_ankle_roll_link` | 0.123182 / 0.123182 / 0.123182 | `-0.01740632 0.00006424 -0.00021460` | `-0.01740630 0.00006424 -0.00021460` | `-0.01740632 0.00006424 -0.00021460` | `0.0000400501 0.0000343207 0.0000144993` | `0.0000400501 0.0000343207 0.0000144993` | `0.0000400501 0.0000343207 0.0000144993` |
| `left_ankle_pitch_link` | 0.844967 / 0.894967 / 0.894967 | `0.03205900 0.03700531 -0.02882818` | `0.03205900 0.03700530 -0.02882820` | `0.03205900 0.03700531 -0.02882818` | `0.004383172 0.004109640 0.000599978` | `0.004383170 0.004109640 0.000599978` | `0.004383172 0.004109640 0.000599978` |
| `right_hip_pitch_link` | 2.797525 / 2.873700 / 2.873700 | `0.00300595 -0.05931469 -0.01709150` | `0.00294368 -0.06110200 -0.01735100` | `0.00294368 -0.06110195 -0.01735095` | `0.005102974 0.004100852 0.003459304` | `0.005374080 0.004211630 0.003687800` | `0.005374080 0.004211634 0.003687795` |
| `right_hip_roll_link` | 0.627146 / 0.751220 / 0.751220 | `-0.03026193 0.00468930 -0.04449852` | `-0.01757890 0.00607425 -0.04131500` | `-0.01757892 0.00607425 -0.04131500` | `0.002002522 0.001675094 0.001280234` | `0.002377800 0.002004430 0.001515300` | `0.002377798 0.002004433 0.001515299` |
| `right_hip_yaw_link` | 7.087520 / 7.087500 / 7.087500 | `0.00069408 0.01086587 -0.11301460` | `0.00069408 0.01086590 -0.11301500` | `0.00069408 0.01086587 -0.11301460` | `0.05140120 0.05059600 0.01300672` | `0.05140130 0.05059640 0.01300630` | `0.05140131 0.05059642 0.01300626` |
| `right_knee_link` | 3.553402 / 4.353400 / 4.353402 | `0.00461331 0.02453217 -0.16715460` | `0.00461331 0.02254900 -0.16715500` | `0.00461331 0.02254900 -0.16715460` | `0.05123878 0.05108841 0.003339976` | `0.05123880 0.05108840 0.003339980` | `0.05123878 0.05108841 0.003339976` |
| `right_ankle_roll_link` | 0.123182 / 0.123182 / 0.123182 | `-0.01740632 -0.00006424 -0.00021460` | `-0.01740630 -0.00006424 -0.00021460` | `-0.01740632 -0.00006424 -0.00021460` | `0.0000400501 0.0000343207 0.0000144993` | `0.0000400501 0.0000343207 0.0000144993` | `0.0000400501 0.0000343207 0.0000144993` |
| `right_ankle_pitch_link` | 0.844967 / 0.894967 / 0.894967 | `0.03205900 -0.03700531 -0.02882818` | `0.03205900 -0.03700530 -0.02882820` | `0.03205900 -0.03700531 -0.02882818` | `0.004383172 0.004109640 0.000599978` | `0.004383170 0.004109640 0.000599978` | `0.004383172 0.004109640 0.000599978` |
| `waist_yaw_link` | 0.279762 / 0.312936 / 0.312936 | `0.00626994 0 0.01696958` | `0.00648295 0.00001018 0.01760630` | `0.00648295 0.00001018 0.01760634` | `0.000311422 0.000247300 0.000123348` | `0.000338910 0.000284360 0.000120180` | `0.000338910 0.000284360 0.000120180` |
| `waist_roll_link` | 0.065850 / 0.138419 / 0.138419 | `0 0.00128474 0` | `0 0.00061119 0` | `0 0.00061119 0` | `0.000015800 0.000010410 0.000008090` | `0.000023040 0.000016680 0.000015090` | `0.000023040 0.000016680 0.000015090` |
| `torso_link` | 15.870263 / 18.532000 / 18.532000 | `0.01342470 0.00547190 0.21050130` | `0.00740421 0.00628045 0.21557500` | `0.00740421 0.00628045 0.21557480` | `0.2730635 0.2443831 0.08408583` | `0.2873210 0.2627800 0.09345250` | `0.2873214 0.2627797 0.09345251` |
| `head_pitch_link` | 1.820000 / 1.820000 / 1.820000 | `0.00000005 -0.00035318 0.04512500` | `0.00012600 0.01881700 0.05293900` | `0.00012600 0.01881700 0.05293900` | `0.005386968 0.005324420 0.001409818` | `0.005386970 0.005324420 0.001409820` | `0.005386968 0.005324420 0.001409818` |
| `head_yaw_link` | 1.300000 / 1.300000 / 1.300000 | `0.02190000 0 0.01870000` | `0.02365400 -0.00073400 0.04231500` | `0.02365400 -0.00073400 0.04231500` | `0.005352739 0.004723356 0.003780905` | `0.005352740 0.004723360 0.003780910` | `0.005352739 0.004723356 0.003780905` |

### 3.2 双臂和手

| Body | mass A/B/C | COM A | COM B | COM C | principal A | principal B | principal C |
|---|---|---|---|---|---|---|---|
| `left_shoulder_pitch_link` | 1.060556 / 1.192700 / 1.192700 | `0.00296153 0.05202683 -0.02504502` | `-0.00278170 0.05039300 -0.02391400` | `-0.00278170 0.05039300 -0.02391400` | `0.001221923 0.001105812 0.000458834` | `0.001749250 0.001629020 0.000852428` | `0.001749251 0.001629020 0.000852428` |
| `left_shoulder_roll_link` | 1.345111 / 1.109830 / 1.109826 | `-0.02061003 0.00415443 -0.06294221` | `-0.00126759 0.00399799 -0.06652670` | `-0.00126759 0.00399799 -0.06652669` | `0.002102254 0.001985133 0.001056663` | `0.001938360 0.001764890 0.001063780` | `0.001938363 0.001764893 0.001063784` |
| `left_shoulder_yaw_link` | 0.915717 / 0.352739 / 0.352739 | `0.00364187 0.01865147 -0.05585548` | `0.00265957 0.01234540 -0.05962200` | `0.00265957 0.01234537 -0.05962197` | `0.003062671 0.002573488 0.001430422` | `0.000897689 0.000854204 0.000287447` | `0.000897689 0.000854204 0.000287447` |
| `left_elbow_link` | 1.694319 / 1.706120 / 1.706116 | `0.04599687 -0.02832961 -0.00661325` | `0.05067670 -0.02161730 -0.00672238` | `0.05067666 -0.02161732 -0.00672238` | `0.005487509 0.005338359 0.000695052` | `0.005227430 0.005214280 0.001118970` | `0.005227428 0.005214278 0.001118974` |
| `left_wrist_roll_link` | 0.969837 / 0.774260 / 0.774260 | `0.03371947 0.00237482 0.00031544` | `0.03591500 -0.00154506 0.00012879` | `0.03591503 -0.00154506 0.00012879` | `0.000926999 0.000903832 0.000852829` | `0.000715547 0.000661189 0.000632495` | `0.000715547 0.000661189 0.000632495` |
| `left_wrist_pitch_link` | 0.946953 / 0.615258 / 0.615258 | `0.01820997 0.03140354 0.01156873` | `0.06510930 -0.03176070 0.00235463` | `0.06510927 -0.03176065 0.00235463` | `0.000918192 0.000821040 0.000648419` | `0.000423535 0.000417983 0.000396162` | `0.000423535 0.000417983 0.000396162` |
| `left_wrist_yaw_link` | 0.775752 / 0.616082 / 0.616082 | `0.00932726 0.00247855 -0.01961661` | `0.08002670 -0.00476807 0.00340465` | `0.08019497 -0.00480351 0.00339767` | `0.002124703 0.001960977 0.000336621` | `0.001899120 0.001665830 0.000524241` | `0.001888332 0.001654966 0.000524142` |
| `right_shoulder_pitch_link` | 1.060556 / 1.192700 / 1.192700 | `0.00296153 -0.05202683 -0.02504502` | `-0.00278140 -0.05039300 -0.02391400` | `-0.00278140 -0.05039300 -0.02391400` | `0.001221923 0.001105812 0.000458835` | `0.001749250 0.001629020 0.000852428` | `0.001749254 0.001629018 0.000852428` |
| `right_shoulder_roll_link` | 1.345111 / 1.109820 / 1.109821 | `-0.02061003 -0.00415443 -0.06294221` | `-0.00126782 -0.00399810 -0.06652610` | `-0.00126782 -0.00399810 -0.06652608` | `0.002102254 0.001985133 0.001056663` | `0.001938380 0.001764960 0.001063790` | `0.001938384 0.001764961 0.001063785` |
| `right_shoulder_yaw_link` | 0.915717 / 0.352742 / 0.352742 | `0.00364187 -0.01865147 -0.05585548` | `0.00266059 -0.01234510 -0.05962340` | `0.00266059 -0.01234515 -0.05962337` | `0.003062671 0.002573488 0.001430422` | `0.000897692 0.000854184 0.000287444` | `0.000897692 0.000854184 0.000287444` |
| `right_elbow_link` | 1.694319 / 1.706170 / 1.706172 | `0.04599687 0.02832961 -0.00661325` | `0.05067840 0.02161710 -0.00672338` | `0.05067838 0.02161708 -0.00672338` | `0.005487509 0.005338359 0.000695052` | `0.005227670 0.005214500 0.001119050` | `0.005227666 0.005214504 0.001119051` |
| `right_wrist_roll_link` | 0.969837 / 0.774265 / 0.774265 | `0.03371947 -0.00237482 0.00031544` | `0.03591370 0.00155865 0.00014187` | `0.03591366 0.00155865 0.00014187` | `0.000926999 0.000903832 0.000852829` | `0.000715950 0.000661595 0.000632525` | `0.000715950 0.000661595 0.000632525` |
| `right_wrist_pitch_link` | 0.946953 / 0.615837 / 0.615837 | `0.01820997 -0.03140354 0.01156873` | `0.06511310 0.03175230 0.00231986` | `0.06511306 0.03175232 0.00231986` | `0.000918192 0.000821040 0.000648419` | `0.000424256 0.000418373 0.000396861` | `0.000424256 0.000418373 0.000396861` |
| `right_wrist_yaw_link` | 0.775752 / 0.616082 / 0.616082 | `0.00932726 -0.00247855 -0.01961661` | `0.08002670 0.00476817 0.00340501` | `0.08055259 0.00479790 0.00338165` | `0.002124703 0.001960977 0.000336621` | `0.001899150 0.001665850 0.000524230` | `0.001899322 0.001665981 0.000524178` |

## 附录 4：模型 D 逐 body inertial 原始值

该表完整记录模型 D 的 mass、COM、principal inertia 和 `iquat=[w x y z]`。单位分别为 kg、m、kg·m²。

| Body | mass D | COM D | principal inertia D | iquat D |
|---|---:|---|---|---|
| `pelvis` | 7.604680 | `0.00046244 -0.00004214 -0.0147344` | `0.0355137 0.0275324 0.0228779` | `0.7075462 0.7062012 -0.0235095 -0.0102787` |
| `left_hip_pitch_link` | 2.797530 | `0.00300595 0.0593147 -0.0170915` | `0.00510297 0.00410085 0.00345930` | `0.8994849 0.4359340 -0.0222136 -0.0198741` |
| `left_hip_roll_link` | 0.627146 | `-0.0302619 -0.0046893 -0.0444985` | `0.00200252 0.00167509 0.00128023` | `-0.0935812 0.6334723 -0.0477755 0.7665983` |
| `left_hip_yaw_link` | 7.087520 | `0.00069408 -0.0108659 -0.113015` | `0.0514012 0.0505960 0.0130067` | `0.6219932 -0.0405952 0.0198637 0.7817173` |
| `left_knee_link` | 3.553400 | `0.00461331 0 -0.167155` | `0.0512388 0.0510884 0.00333998` | `0.6896621 0.0032528 0.0029059 0.7241182` |
| `left_ankle_roll_link` | 0.123182 | `-0.0174063 0.00006424 -0.0002146` | `0.0000400501 0.0000343207 0.0000144993` | `0.4991569 0.5025729 -0.4978089 0.5004489` |
| `left_ankle_pitch_link` | 0.844967 | `0.0320590 0.0370053 -0.0288282` | `0.00438317 0.00410964 0.000599978` | `0.0000619 0.7234499 -0.0000300 0.6903769` |
| `right_hip_pitch_link` | 2.797530 | `0.00300595 -0.0593147 -0.0170915` | `0.00510297 0.00410085 0.00345930` | `0.4359340 0.8994849 0.0198741 0.0222136` |
| `right_hip_roll_link` | 0.627146 | `-0.0302619 0.0046893 -0.0444985` | `0.00200252 0.00167509 0.00128023` | `0.0935812 0.6334723 0.0477755 0.7665983` |
| `right_hip_yaw_link` | 7.087520 | `0.00069408 0.0108659 -0.113015` | `0.0514012 0.0505960 0.0130067` | `0.7817173 0.0198637 -0.0405952 0.6219932` |
| `right_knee_link` | 3.553400 | `0.00461331 0 -0.167155` | `0.0512388 0.0510884 0.00333998` | `0.7241182 0.0029059 0.0032528 0.6896621` |
| `right_ankle_roll_link` | 0.123182 | `-0.0174063 -0.00006424 -0.0002146` | `0.0000400501 0.0000343207 0.0000144993` | `0.5025729 0.4991569 -0.5004489 0.4978089` |
| `right_ankle_pitch_link` | 0.844967 | `0.0320590 -0.0370053 -0.0288282` | `0.00438317 0.00410964 0.000599978` | `-0.0000619 0.7234499 0.0000300 0.6903769` |
| `waist_yaw_link` | 0.279762 | `0.00626994 0 0.0169696` | `0.000311422 0.000247300 0.000123348` | `0 0.7177283 0 0.6963233` |
| `waist_roll_link` | 0.065850 | `0 0.00128474 0` | `0.00001580 0.00001041 0.00000809` | `0 0.7071068 0 0.7071068` |
| `torso_link` | 15.870300 | `0.0134247 0.0054719 0.210501` | `0.273064 0.244383 0.0840858` | `0.9993993 -0.0341887 0.0039983 0.0040318` |
| `head_pitch_link` | 1.820000 | `0.00000005 -0.00035318 0.045125` | `0.00538697 0.00532442 0.00140982` | `0.9999959 -0.0009552 -0.0026240 -0.0005805` |
| `head_yaw_link` | 1.300000 | `0.0219 0 0.0187` | `0.00535274 0.00472336 0.00378091` | `0.6673112 -0.2277671 -0.2317511 0.6701562` |
| `left_shoulder_pitch_link` | 1.060560 | `0.00296153 0.0520268 -0.025045` | `0.00122192 0.00110581 0.000458834` | `0.9558515 0.2913799 -0.0379121 0.0028829` |
| `left_shoulder_roll_link` | 1.345110 | `0 0.00415443 -0.0629422` | `0.00210225 0.00198513 0.00105666` | `0.6783661 0.1108280 0.0778018 0.7221381` |
| `left_shoulder_yaw_link` | 0.915717 | `0.00364187 0.0186515 -0.0558555` | `0.00306267 0.00257349 0.00143042` | `0.9956171 -0.0729116 -0.0584518 -0.0037155` |
| `left_elbow_link` | 1.694320 | `0.0459969 -0.0283296 -0.00661325` | `0.00548751 0.00533836 0.000695052` | `0.5201772 0.5072292 0.5076382 0.4630742` |
| `left_wrist_roll_link` | 0.969837 | `0.0337195 0.00237482 0.00031544` | `0.000926999 0.000903832 0.000852829` | `-0.2429451 0.6385991 0.2868061 0.6714992` |
| `left_wrist_pitch_link` | 0.946953 | `0.0182100 0.0314035 0.0115687` | `0.000918192 0.000821040 0.000648419` | `0.9776985 -0.0379120 -0.2038071 0.0336290` |
| `left_wrist_yaw_link` | 0.605752 | `-0.0101173 0.00400457 -0.0173071` | `0.000778685 0.000671640 0.000238076` | `0.5547068 0.4509329 0.5850648 0.3829609` |
| `right_shoulder_pitch_link` | 1.060560 | `0.00296153 -0.0520268 -0.025045` | `0.00122192 0.00110581 0.000458835` | `0.9558517 -0.2913789 -0.0379149 -0.0028970` |
| `right_shoulder_roll_link` | 1.345110 | `0 -0.00415443 -0.0629422` | `0.00210225 0.00198513 0.00105666` | `0.7221381 0.0778018 0.1108280 0.6783661` |
| `right_shoulder_yaw_link` | 0.915717 | `0.00364187 -0.0186515 -0.0558555` | `0.00306267 0.00257349 0.00143042` | `0.9956171 0.0729116 -0.0584518 0.0037155` |
| `right_elbow_link` | 1.694320 | `0.0459969 0.0283296 -0.00661325` | `0.00548751 0.00533836 0.000695052` | `0.4630742 0.5076382 0.5072292 0.5201772` |
| `right_wrist_roll_link` | 0.969837 | `0.0337195 -0.00237482 0.00031544` | `0.000926999 0.000903832 0.000852829` | `0.2429451 0.6385991 -0.2868061 0.6714992` |
| `right_wrist_pitch_link` | 0.946953 | `0.0182100 -0.0314035 0.0115687` | `0.000918192 0.000821040 0.000648419` | `0.9776985 0.0379120 -0.2038071 -0.0336290` |
| `right_wrist_yaw_link` | 0.605752 | `-0.0101173 -0.00400457 -0.0173071` | `0.000778685 0.000671640 0.000238076` | `0.3829609 0.5850648 0.4509329 0.5547068` |

## 附录 5：模型 A/B/C 逐 body `iquat`

四元数顺序为 `[w x y z]`。`q` 与 `-q` 表示相同旋转；主惯量接近时还可能出现主轴置换，因此本表用于复现，不应脱离附录 3 的 principal inertia 单独判断模型差异。

### 5.1 pelvis、双腿、腰、躯干和头部

| Body | iquat A | iquat B | iquat C |
|---|---|---|---|
| `pelvis` | `0.7075465 0.7062009 -0.0235095 -0.0102787` | `0.9999999 0.0004150 0.0000100 -0.0000476` | `0.7078676 0.7062830 -0.0093132 0.0010183` |
| `left_hip_pitch_link` | `0.8994849 0.4359340 -0.0222136 -0.0198741` | `0.8904398 0.4538829 -0.0133707 -0.0304704` | `0.8904395 0.4538835 -0.0133707 -0.0304704` |
| `left_hip_roll_link` | `-0.0935812 0.6334725 -0.0477755 0.7665981` | `-0.1655150 0.5982510 -0.0524355 0.7822730` | `-0.1655153 0.5982510 -0.0524356 0.7822729` |
| `left_hip_yaw_link` | `0.6219933 -0.0405952 0.0198638 0.7817172` | `0.6219560 -0.0405938 0.0198649 0.7817470` | `0.6219557 -0.0405938 0.0198649 0.7817472` |
| `left_knee_link` | `0.6896622 0.0032528 0.0029059 0.7241181` | `0.6896621 0.0032528 0.0029059 0.7241182` | `0.6896622 0.0032528 0.0029059 0.7241181` |
| `left_ankle_roll_link` | `0.4991568 0.5025730 -0.4978093 0.5004486` | `0.4991569 0.5025729 -0.4978089 0.5004489` | `0.4991568 0.5025730 -0.4978093 0.5004486` |
| `left_ankle_pitch_link` | `0.0000619 0.7234502 -0.0000300 0.6903766` | `0.0000619 0.7234499 -0.0000300 0.6903769` | `0.0000619 0.7234502 -0.0000300 0.6903766` |
| `right_hip_pitch_link` | `0.4359340 0.8994849 0.0198741 0.0222136` | `0.4538829 0.8904398 0.0304704 0.0133707` | `0.4538835 0.8904395 0.0304704 0.0133707` |
| `right_hip_roll_link` | `0.0935812 0.6334725 0.0477755 0.7665981` | `0.1655150 0.5982510 0.0524355 0.7822730` | `0.1655153 0.5982510 0.0524356 0.7822729` |
| `right_hip_yaw_link` | `0.7817172 0.0198638 -0.0405952 0.6219933` | `0.7817470 0.0198649 -0.0405938 0.6219560` | `0.7817472 0.0198649 -0.0405938 0.6219557` |
| `right_knee_link` | `0.7241181 0.0029059 0.0032528 0.6896622` | `0.7241182 0.0029059 0.0032528 0.6896621` | `0.7241181 0.0029059 0.0032528 0.6896622` |
| `right_ankle_roll_link` | `0.5025730 0.4991568 -0.5004486 0.4978093` | `0.5025729 0.4991569 -0.5004489 0.4978089` | `0.5025730 0.4991568 -0.5004486 0.4978093` |
| `right_ankle_pitch_link` | `-0.0000619 0.7234502 0.0000300 0.6903766` | `-0.0000619 0.7234499 0.0000300 0.6903769` | `-0.0000619 0.7234502 0.0000300 0.6903766` |
| `waist_yaw_link` | `0 0.7177283 0 0.6963233` | `0.0001154 0.7200731 0.0002562 0.6938981` | `0.0001154 0.7200735 0.0002562 0.6938978` |
| `waist_roll_link` | `0 0.7071068 0 0.7071068` | `0.5 0.5 -0.5 0.5` | `0.5 0.5 -0.5 0.5` |
| `torso_link` | `0.9993993 -0.0341887 0.0039983 0.0040318` | `0.9993198 -0.0321662 -0.0077523 0.0162839` | `0.9993198 -0.0321662 -0.0077523 0.0162839` |
| `head_pitch_link` | `0.9999959 -0.0009552 -0.0026240 -0.0005805` | `0.9999959 -0.0009552 -0.0026240 -0.0005805` | `0.9999959 -0.0009552 -0.0026240 -0.0005805` |
| `head_yaw_link` | `0.6673111 -0.2277674 -0.2317509 0.6701562` | `0.6673112 -0.2277671 -0.2317511 0.6701562` | `0.6673111 -0.2277674 -0.2317509 0.6701562` |

### 5.2 双臂和手

| Body | iquat A | iquat B | iquat C |
|---|---|---|---|
| `left_shoulder_pitch_link` | `0.9558516 0.2913795 -0.0379121 0.0028829` | `0.9594274 0.2798031 0.0064496 0.0341716` | `0.9594273 0.2798033 0.0064496 0.0341716` |
| `left_shoulder_roll_link` | `0.6783661 0.1108280 0.0778018 0.7221381` | `0.7192258 0.0130341 -0.0054394 0.6946328` | `0.7192258 0.0130341 -0.0054394 0.6946329` |
| `left_shoulder_yaw_link` | `0.9956171 -0.0729116 -0.0584518 -0.0037155` | `0.6555768 0.0514587 -0.1549289 0.7372707` | `0.6555768 0.0514587 -0.1549286 0.7372708` |
| `left_elbow_link` | `0.5201766 0.5072294 0.5076384 0.4630744` | `0.1623519 0.7313286 0.1928329 0.6337317` | `0.1623525 0.7313285 0.1928330 0.6337316` |
| `left_wrist_roll_link` | `-0.2429453 0.6385993 0.2868064 0.6714988` | `0.3494290 0.6163149 -0.3655999 0.6036489` | `0.3494288 0.6163147 -0.3655998 0.6036493` |
| `left_wrist_pitch_link` | `0.9776985 -0.0379121 -0.2038074 0.0336290` | `0.4916820 0.2287540 0.7186530 0.4352680` | `0.4916822 0.2287538 0.7186529 0.4352681` |
| `left_wrist_yaw_link` | `0.5282247 0.5035787 0.5452041 0.4124799` | `0.4881121 0.5674401 0.4126411 0.5191201` | `0.4884779 0.5675614 0.4122871 0.5189247` |
| `right_shoulder_pitch_link` | `0.9558517 -0.2913788 -0.0379149 -0.0028970` | `0.9594248 -0.2798039 0.0064359 -0.0342398` | `0.9594247 -0.2798043 0.0064359 -0.0342398` |
| `right_shoulder_roll_link` | `0.7221381 0.0778018 0.1108280 0.6783661` | `0.6946070 -0.0054503 0.0130148 0.7192510` | `0.6946067 -0.0054503 0.0130148 0.7192513` |
| `right_shoulder_yaw_link` | `0.9956171 0.0729116 -0.0584518 0.0037155` | `0.7372615 -0.1549209 0.0514524 0.6555896` | `0.7372615 -0.1549206 0.0514524 0.6555896` |
| `right_elbow_link` | `0.4630744 0.5076384 0.5072294 0.5201766` | `-0.1624000 0.7313180 -0.1928940 0.6337130` | `-0.1623997 0.7313181 -0.1928944 0.6337128` |
| `right_wrist_roll_link` | `0.2429453 0.6385993 -0.2868064 0.6714988` | `0.6158699 0.3489819 -0.6023869 0.3688439` | `0.6158702 0.3489818 -0.6023870 0.3688435` |
| `right_wrist_pitch_link` | `0.9776985 0.0379121 -0.2038074 -0.0336290` | `0.4404260 0.7167619 0.2425070 0.4831849` | `0.4404262 0.7167619 0.2425065 0.4831850` |
| `right_wrist_yaw_link` | `0.4124799 0.5452041 0.5035787 0.5282247` | `0.5191168 0.4126399 0.5674528 0.4881018` | `0.5189821 0.4124969 0.5675461 0.4882574` |

## 附录 6：模型 A/B/C 逐 body 可碰撞 geom

本表仅列 `contype != 0` 的 geom。`mesh(name)@[x y z]` 中的位置是 MuJoCo 编译和 mesh reference-frame 规范化后的 body-local `geom_pos`；primitive 的 `size` 使用 MuJoCo half-size 约定。B 的脚底完整 primitive 坐标见 10.3 节。

| Body | A | B | C |
|---|---|---|---|
| `pelvis` | `mesh(pelvis_collision)@[0.000002,-0.000245,-0.009665]` | `mesh(pelvis)@[0.000001,0.001509,-0.003244]` | 同 B |
| `left_hip_pitch_link` | `mesh(left_hip_pitch_link)@[0.002029,0.055677,-0.015882]` | 无 | 无 |
| `left_hip_roll_link` | `mesh(left_hip_roll_link)@[-0.034664,-0.001260,-0.025948]` | `sphere(r=0.05)@[-0.02,0,0]` | 同 B |
| `left_hip_yaw_link` | `mesh(left_hip_yaw_link)@[0.002558,-0.003344,-0.124935]` | `mesh(left_hip_yaw_link)@[0.002948,-0.000276,-0.148880]` | 同 B |
| `left_knee_link` | `cylinder(r=0.04,half=0.175)@[0.01,-0.025,-0.25]` | `cylinder(r=0.025,half=0.175)@[0.01,-0.022549,-0.25]` | 同 B |
| `left_ankle_roll_link` | 无 | 无 | 无 |
| `left_ankle_pitch_link` | `mesh(left_ankle_pitch_link)@[0.031261,0.037293,-0.023974]` | 6 个 r=0.008 sphere + 1 个 half-size `[0.115,0.04,0.004]` box | `mesh(left_ankle_pitch_link)@[0.031403,0.037342,-0.023706]` |
| `right_hip_pitch_link` | `mesh(right_hip_pitch_link)@[0.002030,-0.055677,-0.015880]` | 无 | 无 |
| `right_hip_roll_link` | `mesh(right_hip_roll_link)@[-0.034666,0.001265,-0.025948]` | `sphere(r=0.05)@[-0.038,0,0]` | 同 B |
| `right_hip_yaw_link` | `mesh(right_hip_yaw_link)@[0.002558,0.003340,-0.124938]` | `mesh(right_hip_yaw_link)@[0.002947,0.000271,-0.148877]` | 同 B |
| `right_knee_link` | `cylinder(r=0.04,half=0.175)@[0.01,0.025,-0.25]` | `cylinder(r=0.025,half=0.175)@[0.01,0.022549,-0.25]` | 同 B |
| `right_ankle_roll_link` | 无 | 无 | 无 |
| `right_ankle_pitch_link` | `mesh(right_ankle_pitch_link)@[0.031256,-0.037298,-0.023974]` | 6 个 r=0.008 sphere + 1 个 half-size `[0.115,0.04,0.004]` box | `mesh(right_ankle_pitch_link)@[0.031406,-0.037343,-0.023706]` |
| `waist_yaw_link` | 无 | 无 | 无 |
| `waist_roll_link` | 无 | 无 | 无 |
| `torso_link` | `mesh(torso_collision)@[0.008689,-0.001885,0.241614]` | `mesh(torso_link)@[-0.013269,-0.000096,0.231567]` | 同 B |
| `head_pitch_link` | 无 | 无 | 无 |
| `head_yaw_link` | `mesh(head_yaw_link)@[0.014797,0.000003,0.034399]` | 与 A 编译位置相同 | `mesh(head_yaw_link)@[0.015573,0.000003,0.033079]` |
| `left_shoulder_pitch_link` | `mesh(left_shoulder_pitch_link)@[0.000596,0.050903,-0.028470]` | `mesh(left_shoulder_pitch_link)@[-0.002648,0.044333,-0.021817]` | 同 B |
| `left_shoulder_roll_link` | `mesh(left_shoulder_roll_link)@[-0.029132,0.002642,-0.032656]` | `mesh(left_shoulder_roll_link)@[-0.001542,0.003326,-0.047795]` | 同 B |
| `left_shoulder_yaw_link` | `mesh(left_shoulder_yaw_link)@[0.003523,0.009013,-0.057715]` | `mesh(left_shoulder_yaw_link)@[0.001646,0.006903,-0.053329]` | 同 B |
| `left_elbow_link` | `mesh(left_elbow_link)@[0.060409,-0.030814,-0.008560]` | `mesh(left_elbow_link)@[0.052128,-0.024295,-0.008305]` | 同 B |
| `left_wrist_roll_link` | 无 | 无 | 无 |
| `left_wrist_pitch_link` | 无 | 无 | 无 |
| `left_wrist_yaw_link` | `sphere(r=0.04)@[0,0,-0.029]` | `mesh(left_wrist_yaw_link)@[0.091293,-0.005780,0.002645]` | `mesh(left_wrist_yaw_link)@[0.019253,0.000017,0.000032]` + `mesh(left_hand_link)@[0.128150,-0.007200,0.005896]` |
| `right_shoulder_pitch_link` | `mesh(right_shoulder_pitch_link)@[-0.000554,-0.040390,-0.016181]` | `mesh(right_shoulder_pitch_link)@[-0.002648,-0.044333,-0.021820]` | 同 B |
| `right_shoulder_roll_link` | `mesh(right_shoulder_roll_link)@[-0.029133,-0.002641,-0.032657]` | `mesh(right_shoulder_roll_link)@[-0.001542,-0.003326,-0.047795]` | 同 B |
| `right_shoulder_yaw_link` | `mesh(right_shoulder_yaw_link)@[0.003533,-0.009016,-0.057741]` | `mesh(right_shoulder_yaw_link)@[0.001646,-0.006902,-0.053329]` | 同 B |
| `right_elbow_link` | `mesh(right_elbow_link)@[0.060411,0.030813,-0.008560]` | `mesh(right_elbow_link)@[0.052278,0.024589,-0.008152]` | 同 B |
| `right_wrist_roll_link` | 无 | 无 | 无 |
| `right_wrist_pitch_link` | 无 | 无 | 无 |
| `right_wrist_yaw_link` | `sphere(r=0.04)@[0,0,-0.029]` | `mesh(right_wrist_yaw_link)@[0.093527,0.005774,0.002546]` | `mesh(right_wrist_yaw_link)@[0.015963,-0.000013,0.000200]` + `mesh(right_hand_link)@[0.128671,0.007196,0.005872]` |

## 附录 7：模型 D 的 33 个有效 collision geom

模型 D 的 visual mesh 全部 `density=0, contype=0, conaffinity=0`。下表只列 MjLab `FULL_COLLISION` 实际启用的 geom；左右合并行各代表两个 geom。`fromto` 和 size 单位为 m。

| Geom | Body | 类型与尺寸 | source fromto/pos | 数量 | condim | friction |
|---|---|---|---|---:|---:|---|
| `pelvis_collision` | pelvis | sphere r=0.08 | pos `[0,0,0]` | 1 | 1 | `[1,0.005,0.0001]` |
| `left_hip_collision` | left hip-roll | capsule r=0.06 | `[-0.013,0,0] -> [-0.013,0,-0.025]` | 1 | 1 | default 1 |
| `right_hip_collision` | right hip-roll | capsule r=0.06 | `[-0.033,0,0] -> [-0.033,0,-0.025]` | 1 | 1 | default 1 |
| `left/right_thigh_collision` | hip-yaw | capsule r=0.05 | `[0,0,-0.05] -> [0.015,0,-0.25]` | 2 | 1 | default 1 |
| `left/right_shin_collision` | knee | capsule r=0.03 | `[0.03,0,-0.05] -> [0,0,-0.20]` | 2 | 1 | default 1 |
| `left/right_linkage_brace_collision` | knee | capsule r=0.02 | `[0,0,-0.28] -> [0.02,0,-0.45]` | 2 | 1 | default 1 |
| `left_foot1_collision` | left ankle-pitch | capsule r=0.01 | `[0.080,0.008,-0.038] -> [0.130,0.008,-0.038]` | 1 | 3 | foot 0.6 |
| `left_foot2_collision` | left ankle-pitch | capsule r=0.01 | `[-0.065,0.018,-0.038] -> [0.140,0.018,-0.038]` | 1 | 3 | foot 0.6 |
| `left_foot3_collision` | left ankle-pitch | capsule r=0.01 | `[-0.068,0.027,-0.038] -> [0.145,0.027,-0.038]` | 1 | 3 | foot 0.6 |
| `left_foot4_collision` | left ankle-pitch | capsule r=0.01 | `[-0.070,0.037,-0.038] -> [0.150,0.037,-0.038]` | 1 | 3 | foot 0.6 |
| `left_foot5_collision` | left ankle-pitch | capsule r=0.01 | `[-0.068,0.047,-0.038] -> [0.145,0.047,-0.038]` | 1 | 3 | foot 0.6 |
| `left_foot6_collision` | left ankle-pitch | capsule r=0.01 | `[-0.065,0.055,-0.038] -> [0.140,0.055,-0.038]` | 1 | 3 | foot 0.6 |
| `left_foot7_collision` | left ankle-pitch | capsule r=0.01 | `[0.130,0.065,-0.038] -> [0.080,0.065,-0.038]` | 1 | 3 | foot 0.6 |
| `right_foot1-7_collision` | right ankle-pitch | capsule r=0.01 | 左脚七条的 Y 镜像 | 7 | 3 | foot 0.6 |
| `torso_collision` | torso | capsule r=0.09 | `[0,0,0.12] -> [0,0,0.30]` | 1 | 1 | default 1 |
| `head_collision` | head-yaw | sphere r=0.065 | pos `[0.02,0,0.02]` | 1 | 1 | default 1 |
| `left/right_shoulder_yaw_collision` | shoulder-yaw | capsule r=0.035 | `[0,0,-0.06] -> [0,0,0.09]` | 2 | 1 | default 1 |
| `left/right_elbow_yaw_collision` | elbow | capsule r=0.035 | `[0,∓0.025,-0.01] -> [0.10,∓0.025,-0.01]` | 2 | 1 | default 1 |
| `left/right_wrist_collision` | wrist-pitch | capsule r=0.035 | `[-0.03,0,0] -> [0.05,0,0]` | 2 | 1 | default 1 |
| `left/right_hand_collision` | wrist-yaw | capsule r=0.035 | `[0.10,∓0.01,0] -> [0.18,∓0.01,0]` | 2 | 1 | default 1 |

数量合计：2 sphere + 31 capsule = 33。14 个 foot capsule 的 `friction[0]` 在 nominal config 为 0.6，并在训练 startup event 中被随机化；表中 `default 1` 表示 nominal sliding friction 1.0。

## 附录 8：关键文件索引

| 内容 | 路径 |
|---|---|
| A 源 URDF | `gear_sonic/data/assets/robot_description/urdf/h2/h2.urdf` |
| A 训练 actuator/初始姿态 | `gear_sonic/envs/manager_env/robots/h2.py` |
| H2LoRA 29D policy 配置 | `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2lora.yaml` |
| 训练物理材质随机化 | `gear_sonic/config/manager_env/events/terms/physics_material.yaml` |
| A 派生 sim2sim MJCF | `gear_sonic/data/assets/robot_description/mjcf/h2.xml` |
| B 官方 MJCF | `gear_sonic/data/assets/robot_description/unitree_mujoco/h2/h2_mujoco.xml` |
| B 来源记录 | `gear_sonic/data/assets/robot_description/unitree_mujoco/SOURCE.txt` |
| C 官方 URDF | `gear_sonic/data/assets/robot_description/unitree_ros/h2_description/H2.urdf` |
| C DAE visual URDF | `gear_sonic/data/assets/robot_description/unitree_ros/h2_description/H2_dae.urdf` |
| C-Plus 官方 URDF | 上游 [`robots/h2_plus/H2_Plus.urdf`](https://github.com/unitreerobotics/unitree_ros/blob/d96d8f63ae17a7108d4f7229c00ef875ba7129c9/robots/h2_plus/H2_Plus.urdf)；当前未导入本工程 |
| C 来源记录 | `gear_sonic/data/assets/robot_description/unitree_ros/SOURCE.txt` |
| C -> MuJoCo 生成脚本 | `scripts/build_h2_mujoco_from_unitree_ros.py` |
| B 几何 + C 惯性生成脚本 | `scripts/build_h2_unitree_mujoco_ros_mass.py` |
| B 的 H2 首次提交 | [`unitree_mujoco c1ccb767`](https://github.com/unitreerobotics/unitree_mujoco/commit/c1ccb767326e74bf659a787815cea7205b75a2d1) |
| B 对应的 C 历史快照 | [`unitree_ros 202c5987`](https://github.com/unitreerobotics/unitree_ros/commit/202c59879872d0c8d61ab20b93eaedd9ec04b144) |
| C 拆分 fixed hands | [`unitree_ros fceefdfd`](https://github.com/unitreerobotics/unitree_ros/commit/fceefdfd21389778ebb42419d503c7e5eee4a1be) |
| MuJoCo fixed inertia 修复 | [MuJoCo PR #3250](https://github.com/google-deepmind/mujoco/pull/3250) |
| B sim2sim 配置 | `gear_sonic/utils/mujoco_sim/wbc_configs/h2_lora_sonic_unitree_official.yaml` |
| C sim2sim 配置 | `gear_sonic/utils/mujoco_sim/wbc_configs/h2_lora_sonic_unitree_ros.yaml` |
| A 派生 sim2sim 配置 | `gear_sonic/utils/mujoco_sim/wbc_configs/h2_lora_sonic.yaml` |
| 运行时 MJCF/接触/低层覆盖 | `gear_sonic/utils/mujoco_sim/base_sim.py` |
| D 仓库 | `/home/qingyul/workspace/unitree_rl_mjlab` |
| D 源 MJCF | `/home/qingyul/workspace/unitree_rl_mjlab/src/assets/robots/unitree_h2/xmls/h2.xml` |
| D actuator/collision/home config | `/home/qingyul/workspace/unitree_rl_mjlab/src/assets/robots/unitree_h2/h2_constants.py` |
| D velocity H2 task config | `/home/qingyul/workspace/unitree_rl_mjlab/src/tasks/velocity/config/h2/env_cfgs.py` |
| D tracking H2 task config | `/home/qingyul/workspace/unitree_rl_mjlab/src/tasks/tracking/config/h2/env_cfgs.py` |
| D tracking/velocity base physics/events | `/home/qingyul/workspace/unitree_rl_mjlab/src/tasks/tracking/tracking_env_cfg.py`；`src/tasks/velocity/velocity_env_cfg.py` |

## 附录 9：STL 文件级对应关系

`相同`表示 SHA-256 完全相同；`不同`表示同名但字节不同；`无`表示该目录没有同名文件。

| STL basename | A-B | A-C | B-C | D-A | D-B | D-C |
|---|---|---|---|---|---|---|
| `pelvis.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `left_hip_pitch_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `left_hip_roll_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `left_hip_yaw_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `left_knee_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `left_ankle_roll_link.stl` | 不同 | 不同 | 相同 | 相同 | 不同 | 不同 |
| `left_ankle_pitch_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `right_hip_pitch_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `right_hip_roll_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `right_hip_yaw_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `right_knee_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `right_ankle_roll_link.stl` | 不同 | 不同 | 相同 | 相同 | 不同 | 不同 |
| `right_ankle_pitch_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `waist_yaw_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `waist_roll_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `torso_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `head_pitch_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `head_yaw_link.stl` | 相同 | 不同 | 不同 | 相同 | 相同 | 不同 |
| `left_shoulder_pitch_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `left_shoulder_roll_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `left_shoulder_yaw_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `left_elbow_link.stl` | 不同 | 不同 | 相同 | 不同 | 相同 | 相同 |
| `left_wrist_roll_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `left_wrist_pitch_link.stl` | 不同 | 不同 | 不同 | 不同 | 不同 | 不同 |
| `left_wrist_yaw_link.stl` | A 无 | A 无 | 不同 | A 无 | 不同 | 不同 |
| `right_shoulder_pitch_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `right_shoulder_roll_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `right_shoulder_yaw_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `right_elbow_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `right_wrist_roll_link.stl` | 不同 | 不同 | 相同 | 不同 | 不同 | 不同 |
| `right_wrist_pitch_link.stl` | 不同 | 不同 | 不同 | 不同 | 不同 | 不同 |
| `right_wrist_yaw_link.stl` | A 无 | A 无 | 不同 | A 无 | 相同 | 不同 |

A 独有 basename：

```text
left_rubber_hand.stl
right_rubber_hand.stl
left_small_roll_link.stl
right_small_arm_roll_link.stl
pelvis_collision.stl
torso_collision.stl
simple_torso_link.stl
```

C 相对 B 额外具有：

```text
left_hand_link.stl
right_hand_link.stl
```

C 还对 34 个 visual mesh 提供对应 DAE 文件；前文 A/B/C/D 统一动态参数表使用 C-STL。C-DAE 和 C-Plus 的完整内部关系见第 17 节及附录 10、11。

## 附录 10：C-Plus 的 44 个手指活动关节

左右手各使用下表 22 个 joint suffix，因此总计 44 个 revolute joint。所有 joint 的 local axis 都是 `[0,0,1]`，左右同名 joint 的 range、effort 和 velocity 完全相同；左右差异由 attachment 和 local origin/rpy 的镜像关系表达。

### 10.1 拓扑、限位和执行器元数据

| joint suffix | 每侧 parent -> child | range rad | effort Nm | velocity rad/s |
|---|---|---|---:|---:|
| `thumb_CMC_FE` | `hand_C_MC -> thumb_CMC_VL` | `[-0.1745,1.9199]` | 3.3 | 11.84076833 |
| `thumb_CMC_AA` | `thumb_CMC_VL -> thumb_MC` | `[-0.3491,0.3491]` | 3.3 | 11.84076833 |
| `thumb_MCP_FE` | `thumb_MC -> thumb_MCP_VL` | `[-0.5236,1.3963]` | 1.864 | 16.07692878 |
| `thumb_MCP_AA` | `thumb_MCP_VL -> thumb_PP` | `[-0.3491,0.3491]` | 1.864 | 16.07692878 |
| `thumb_IP` | `thumb_PP -> thumb_DP` | `[0,1.7453]` | 0.638 | 11.61831603 |
| `index_MCP_FE` | `hand_C_MC -> index_MCP_VL` | `[-0.17453293,1.5708]` | 1.864 | 16.07692878 |
| `index_MCP_AA` | `index_MCP_VL -> index_PP` | `[-0.3491,0.3491]` | 1.864 | 16.07692878 |
| `index_PIP` | `index_PP -> index_MP` | `[0,1.7453]` | 0.638 | 11.61831603 |
| `index_DIP` | `index_MP -> index_DP` | `[0,1.3963]` | 0.189369 | 14.66594768 |
| `middle_MCP_FE` | `hand_C_MC -> middle_MCP_VL` | `[-0.17453293,1.5708]` | 1.864 | 16.07692878 |
| `middle_MCP_AA` | `middle_MCP_VL -> middle_PP` | `[-0.3491,0.3491]` | 1.864 | 16.07692878 |
| `middle_PIP` | `middle_PP -> middle_MP` | `[0,1.7453]` | 0.638 | 11.61831603 |
| `middle_DIP` | `middle_MP -> middle_DP` | `[0,1.3963]` | 0.189369 | 14.66594768 |
| `ring_MCP_FE` | `hand_C_MC -> ring_MCP_VL` | `[-0.17453293,1.5708]` | 1.864 | 16.07692878 |
| `ring_MCP_AA` | `ring_MCP_VL -> ring_PP` | `[-0.3491,0.3491]` | 1.864 | 16.07692878 |
| `ring_PIP` | `ring_PP -> ring_MP` | `[0,1.7453]` | 0.638 | 11.61831603 |
| `ring_DIP` | `ring_MP -> ring_DP` | `[0,1.3963]` | 0.189369 | 14.66594768 |
| `pinky_CMC` | `hand_C_MC -> pinky_MC` | `[0,0.2618]` | 0.5285 | 35.07416159 |
| `pinky_MCP_FE` | `pinky_MC -> pinky_MCP_VL` | `[-0.17453293,1.5708]` | 1.864 | 16.07692878 |
| `pinky_MCP_AA` | `pinky_MCP_VL -> pinky_PP` | `[-0.3491,0.3491]` | 1.864 | 16.07692878 |
| `pinky_PIP` | `pinky_PP -> pinky_MP` | `[0,1.7453]` | 0.638 | 11.61831603 |
| `pinky_DIP` | `pinky_MP -> pinky_DP` | `[0,1.3963]` | 0.189369 | 14.66594768 |

每根手指还包含 `elastomer_fix_joint` 和 `fingertip_fix_joint` 两个 fixed joint，共每手 10 个。URDF 中没有 mimic，因此 DIP/PIP 等关节不会自动耦合。

### 10.2 左右 joint origin

表中格式为 `xyz / rpy`，单位分别为 m 和 rad；joint name 需要在 suffix 前加 `left_` 或 `right_`。

| joint suffix | left origin xyz / rpy | right origin xyz / rpy |
|---|---|---|
| `thumb_CMC_FE` | `0.01 -0.026 0.0214` / `0 0 -1.5708` | `0.01 0.026 0.0214` / `3.1416 0 1.5708` |
| `thumb_CMC_AA` | `0 -0.005 0` / `1.5708 -0.7854 0` | `0 -0.005 0` / `-1.5708 0.7854 0` |
| `thumb_MCP_FE` | `0.065 -0.006 -0.010392` / `-0.5236 0 0` | `0.065 -0.006 0.010392` / `0.5236 0 0` |
| `thumb_MCP_AA` | `0 0 0` / `1.5708 0 0` | `0 0 0` / `-1.5708 0 0` |
| `thumb_IP` | `0.039 0 0` / `-1.5708 0 0` | `0.039 0 0` / `1.5708 0 0` |
| `index_MCP_FE` | `0.001 -0.0303 0.0959` / `0 -1.5708 -1.5708` | `0.001 0.0303 0.0959` / `0 -1.5708 -1.5708` |
| `index_MCP_AA` | `0 0 0` / `1.5708 0 0` | `0 0 0` / `-1.5708 0 0` |
| `index_PIP` | `0.047 0 0` / `-1.5708 0 0` | `0.047 0 0` / `1.5708 0 0` |
| `index_DIP` | `0.0315 0 0` / `0 0 0` | `0.0315 0 0` / `0 0 0` |
| `middle_MCP_FE` | `0 -0.01 0.0989` / `0 -1.5708 -1.5708` | `0 0.01 0.0989` / `0 -1.5708 -1.5708` |
| `middle_MCP_AA` | `0 0 0` / `1.5708 0 0` | `0 0 0` / `-1.5708 0 0` |
| `middle_PIP` | `0.047 0 0` / `-1.5708 0 0` | `0.047 0 0` / `1.5708 0 0` |
| `middle_DIP` | `0.0315 0 0` / `0 0 0` | `0.0315 0 0` / `0 0 0` |
| `ring_MCP_FE` | `0.0015 0.0103 0.0929` / `0 -1.5708 -1.5708` | `0.0015 -0.0103 0.0929` / `0 -1.5708 -1.5708` |
| `ring_MCP_AA` | `0 0 0` / `1.5708 0 0` | `0 0 0` / `-1.5708 0 0` |
| `ring_PIP` | `0.047 0 0` / `-1.5708 0 0` | `0.047 0 0` / `1.5708 0 0` |
| `ring_DIP` | `0.0315 0 0` / `0 0 0` | `0.0315 0 0` / `0 0 0` |
| `pinky_CMC` | `0.01136 0.0263 0.0869` / `-3.1416 0 3.1416` | `0.01136 -0.0263 0.0869` / `0 0 3.1416` |
| `pinky_MCP_FE` | `0.00836 0.0048 0` / `-1.5708 1.5708 0` | `0.00836 0.0048 0` / `1.5708 -1.5708 0` |
| `pinky_MCP_AA` | `0 0 0` / `1.5708 0 0` | `0 0 0` / `-1.5708 0 0` |
| `pinky_PIP` | `0.047 0 0` / `-1.5708 0 0` | `0.047 0 0` / `1.5708 0 0` |
| `pinky_DIP` | `0.0315 0 0` / `0 0 0` | `0.0315 0 0` / `0 0 0` |

## 附录 11：C-Plus 双手逐 link 质量

下表列出每侧 33 个手部 link 的全部质量。index/middle/ring 三组数值相同，但仍逐 link 列出，便于核对总质量。所有这些 link 在 C-Plus 中均有显式 inertial；C-STL/C-DAE 只有每侧一个 `0.41885723 kg` 的简单 `hand_link`，没有逐指对应项。

| link suffix | left mass kg | right mass kg |
|---|---:|---:|
| `hand_C_MC` | 0.650600 | 0.650600 |
| `thumb_CMC_VL` | 0.002500 | 0.002500 |
| `thumb_MC` | 0.135870 | 0.135870 |
| `thumb_MCP_VL` | 0.000080 | 0.000080 |
| `thumb_PP` | 0.030780 | 0.030780 |
| `thumb_DP` | 0.009880 | 0.007880 |
| `thumb_elastomer` | 0.000100 | 0.000100 |
| `thumb_fingertip` | 0.000001 | 0.000001 |
| `index_MCP_VL` | 0.002210 | 0.002210 |
| `index_PP` | 0.047410 | 0.047410 |
| `index_MP` | 0.018240 | 0.018240 |
| `index_DP` | 0.004550 | 0.004550 |
| `index_elastomer` | 0.000100 | 0.000100 |
| `index_fingertip` | 0.000001 | 0.000001 |
| `middle_MCP_VL` | 0.002210 | 0.002210 |
| `middle_PP` | 0.047410 | 0.047410 |
| `middle_MP` | 0.018240 | 0.018240 |
| `middle_DP` | 0.004550 | 0.004550 |
| `middle_elastomer` | 0.000100 | 0.000100 |
| `middle_fingertip` | 0.000001 | 0.000001 |
| `ring_MCP_VL` | 0.002210 | 0.002210 |
| `ring_PP` | 0.047410 | 0.047410 |
| `ring_MP` | 0.018240 | 0.018240 |
| `ring_DP` | 0.004550 | 0.004550 |
| `ring_elastomer` | 0.000100 | 0.000100 |
| `ring_fingertip` | 0.000001 | 0.000001 |
| `pinky_MC` | 0.125410 | 0.125410 |
| `pinky_MCP_VL` | 0.002210 | 0.002210 |
| `pinky_PP` | 0.047410 | 0.047410 |
| `pinky_MP` | 0.018240 | 0.018240 |
| `pinky_DP` | 0.004550 | 0.004550 |
| `pinky_elastomer` | 0.000100 | 0.000100 |
| `pinky_fingertip` | 0.000001 | 0.000001 |
| **合计** | **1.245265** | **1.243265** |

这里的微小 `0.000001 kg` fingertip 和 `0.0001 kg` elastomer 也会进入编译模型；不要在质量汇总时因数值小而漏掉。完整 COM 和六项惯量矩阵以固定 SHA-256 的官方 `H2_Plus.urdf` 为准。
