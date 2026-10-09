# XGO 机器狗 Isaac Lab 配置文件指南

> 基于 Isaac Lab 2.2.1，针对 `assets/xgo.py` 配置文件的详细说明与学习指南。

---

## 一、配置文件整体结构

`xgo.py` 本质上是一份"给仿真器描述机器人的说明书"，核心是 `ArticulationCfg`，包含四大区块：

```
ArticulationCfg
├── spawn          → 怎么把机器人"放"进仿真（URDF路径、物理属性）
├── init_state     → 机器人初始姿态（站多高、各关节初始角度）
├── 全局参数        → soft_joint_pos_limit_factor、fix_base_link 等
└── actuators      → 每个关节用什么执行器、力矩/速度上限、PD增益
```

---

## 二、原始配置存在的问题与修复

### 问题清单

| 优先级 | 问题 | 影响 |
|--------|------|------|
| 🔴 高 | 路径硬编码为 Linux 绝对路径 | 换环境直接报错 |
| 🔴 高 | 执行器力矩(23.5)远大于URDF定义(0.45) | 训练结果真机不可用 |
| 🔴 高 | 移动关节(手爪)被当成旋转关节 | 物理仿真异常 |
| 🔴 高 | 用 `".*"` 匹配所有关节到同一组 | 腿/臂/手混在一起，控制混乱 |
| 🟠 中 | floating base 可能被固定 | 机器人无法移动 |
| 🟠 中 | URDF 路径目录名大小写错误 | 找不到文件 |
| 🟡 低 | 前后腿初始姿态不对称 | 可能影响初始平衡 |
| 🟡 低 | 重复导入 ImplicitActuatorCfg | 不报错但不规范 |
| 🟡 低 | 缺少 collision_group 设置 | 碰撞检测可能不准确 |

### 修复后完整代码

```python
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

# 基于当前文件位置动态获取资产目录，避免硬编码绝对路径
# 当前文件: assets/xgo.py
# URDF 路径:  assets/robots/xgo/urdf/xgo.urdf
XGO_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robots", "xgo")

XGO_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{XGO_MODEL_DIR}/urdf/xgo.urdf",
        usd_dir=f"{XGO_MODEL_DIR}/usd",
        usd_file_name="xgo.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            collision_group=-1,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.18),  # 初始高度 18cm，需根据实际腿长验证脚是否刚好触地
        joint_pos={
            # ===== 四条腿（12 个关节）=====
            # 前左腿 FL
            "fl_hip_joint": 0.0,   # 髋侧摆
            "fl_thigh_joint": 0.8,  # 髋前后
            "fl_calf_joint": -1.2,  # 膝盖
            # 前右腿 FR
            "fr_hip_joint": 0.0,
            "fr_thigh_joint": 0.8,
            "fr_calf_joint": -1.2,
            # 后左腿 BL
            "bl_hip_joint": 0.0,
            "bl_thigh_joint": 0.8,
            "bl_calf_joint": -1.2,
            # 后右腿 BR
            "br_hip_joint": 0.0,
            "br_thigh_joint": 0.8,
            "br_calf_joint": -1.2,
            # ===== 机械臂（收起姿态，不影响走路）=====
            "arm_thigh_joint": 0.0,  # 大臂
            "arm_calf_joint": 0.0,   # 小臂
            # ===== 手爪（闭合）=====
            "hand_left_joint": 0.0,
            "hand_right_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    # 不固定基座，让机器人可以自由移动（floating base）
    fix_base_link=False,
    actuators={
        # --- 腿部 12 个旋转关节 ---
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "fl_hip_joint", "fl_thigh_joint", "fl_calf_joint",
                "fr_hip_joint", "fr_thigh_joint", "fr_calf_joint",
                "bl_hip_joint", "bl_thigh_joint", "bl_calf_joint",
                "br_hip_joint", "br_thigh_joint", "br_calf_joint",
            ],
            effort_limit=0.45,       # 与 URDF 一致：0.45 N·m
            velocity_limit=10.0,     # 与 URDF 一致：10 rad/s
            stiffness=25.0,          # 位置 P 增益 (N·m/rad)
            damping=0.5,             # 速度 D 增益 (N·m·s/rad)
            friction=0.0,
        ),
        # --- 机械臂 2 个旋转关节 ---
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["arm_thigh_joint", "arm_calf_joint"],
            effort_limit=0.45,
            velocity_limit=10.0,
            stiffness=50.0,          # 更高刚度，保持收起姿态不动
            damping=1.0,
            friction=0.0,
        ),
        # --- 手爪 2 个移动关节 (prismatic) ---
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["hand_left_joint", "hand_right_joint"],
            effort_limit=0.45,       # 对于 prismatic 是力 (N)
            velocity_limit=0.1,      # 移动速度调低 (m/s)
            stiffness=100.0,         # 保持闭合的刚度 (N/m)
            damping=5.0,             # 阻尼 (N·s/m)
            friction=0.0,
        ),
    },
)
```

### 逐项修改说明

| # | 问题 | 修改方法 |
|---|------|----------|
| 1 | 重复导入 | 去掉重复，统一用 `ImplicitActuatorCfg` |
| 2 | 路径硬编码 | 用 `os.path.dirname(__file__)` 动态获取 |
| 3 | URDF 路径大小写错误 | `Robots` → `robots`，匹配实际目录 |
| 4 | 力矩参数错误 | `effort_limit=23.5` → `0.45`，与 URDF 一致 |
| 5 | 速度参数错误 | `velocity_limit=30.0` → `10.0`，与 URDF 一致 |
| 6 | 关节分组混乱 | 拆分为 legs / arm / hand 三组 |
| 7 | 移动关节单位错误 | hand 组单独配置，速度调低到 0.1 m/s |
| 8 | floating base 可能固定 | 显式设置 `fix_base_link=False` |
| 9 | 前后腿不对称 | 统一为相同角度（需验证URDF中关节轴方向） |
| 10 | 缺少 collision_group | 加上 `collision_group=-1` |
| 11 | soft limit 略低 | 0.9 → 0.95，给训练更多可用空间 |

---

## 三、每个参数从哪来？怎么确定？

### 3.1 `spawn` 区块 — URDF + 物理引擎参数

| 参数 | 含义 | 来源 | 调参建议 |
|------|------|------|----------|
| `asset_path` | URDF 文件路径 | 你的项目目录 | 用 `os.path` 动态获取，别写死 |
| `usd_dir` / `usd_file_name` | 转换后的 USD 缓存路径 | 你决定 | 首次导入自动生成，后续加速加载 |
| `disable_gravity` | 是否禁用重力 | 常识 | 机器人必须受重力，`False` |
| `linear_damping` / `angular_damping` | 线速度/角速度阻尼 | 仿真参数 | 一般设 0，空气阻力可忽略 |
| `max_linear_velocity` / `max_angular_velocity` | 最大速度截断 | 仿真稳定 | 设大一点（1000），避免意外截断 |
| `max_depenetration_velocity` | 最大穿透修正速度 | 仿真稳定 | 默认 1.0，穿模严重就调大 |
| `collision_group` | 碰撞组 | 仿真配置 | `-1` 表示与所有组碰撞 |
| `enabled_self_collisions` | 自碰撞检测 | 看需求 | 训练初期关（省算力），后期再开 |
| `solver_position_iteration_count` | 物理求解器迭代次数 | 仿真精度 | 默认 4，不稳定就加到 8 |
| `solver_velocity_iteration_count` | 速度迭代次数 | 仿真精度 | 默认 0 够用 |
| `sleep_threshold` | 休眠阈值 | 仿真优化 | 物体不动了就休眠省算力 |
| `stabilization_threshold` | 稳定阈值 | 仿真优化 | 太小容易误判休眠 |

> **质量和惯量在哪？** 在 URDF 的 `<inertial>` 标签里，SolidWorks 导出时自动计算，一般不用改。如果要改，直接改 URDF 文件。

---

### 3.2 `init_state` 区块 — 初始姿态

| 参数 | 含义 | 来源 | 怎么确定 |
|------|------|------|----------|
| `pos=(x, y, z)` | 基座初始位置 | 你决定 | z 是初始高度，让脚刚好碰到地面 |
| `joint_pos` | 各关节初始角度 | 你决定 | 必须在 URDF `<limit>` 范围内 |
| `joint_vel` | 各关节初始速度 | 一般都是 0 | 初始静止 |

**怎么确定初始高度 z：**

1. **估算法：** 看 URDF 中从 body 到 foot 的连杆长度，加起来估个值
2. **试错法：** 先随便填一个，跑仿真看效果：
   - 脚穿透地面 → 高度调大
   - 脚悬空掉落 → 高度调小
   - 刚好触地且身体不晃 → 合适

**⚠️ 注意后腿关节轴方向：**
URDF 中前后腿 thigh 关节轴都是 `axis="0 1 0"`（Y轴），但安装方向可能相反。如果仿真中后腿朝前弯了，把后腿的 thigh 和 calf 角度都取反：

```python
"bl_thigh_joint": -0.8,   # 取反
"bl_calf_joint": 1.2,     # 取反
```

---

### 3.3 `actuators` 区块 — 最关键也最容易错

执行器配置是整个文件的核心，直接决定训练出来的策略能不能用在真机上。

#### A. 物理极限参数 — 必须跟真机一致

| 参数 | 含义 | 单位（旋转关节） | 单位（移动关节） | 来源 |
|------|------|-----------------|-----------------|------|
| `effort_limit` | 最大输出 | N·m（牛·米） | N（牛） | 电机手册 / URDF |
| `velocity_limit` | 最大速度 | rad/s（弧度/秒） | m/s（米/秒） | 电机手册 / URDF |

> ✅ **铁律：这两个值必须跟真机一致！差得太远训练出来的策略没用。**

**怎么获取真实参数：**
- 查 XGO 官方手册 / 电机型号 datasheet
- 如果 URDF 是 SolidWorks 导出的，里面的 `effort` 和 `velocity` 可能是随便填的，要核实

#### B. 控制增益参数 — PD 控制器

| 参数 | 含义 | 调参口诀 |
|------|------|----------|
| `stiffness` | 位置刚度（P增益） | 太小→软/抖，太大→震荡 |
| `damping` | 阻尼（D增益） | 太小→震荡，太大→响应慢 |
| `friction` | 摩擦力 | 仿真先设0，要更真实再加 |

**调参步骤：**
1. 固定 damping=0，慢慢增大 stiffness 直到关节开始震荡
2. 把 stiffness 回调 30%
3. 慢慢增大 damping 把剩余震荡消掉
4. 经验值：damping ≈ stiffness 的 1/10 ~ 1/50

#### C. 执行器类型选择

| 类型 | 适用场景 | 特点 |
|------|----------|------|
| `ImplicitActuatorCfg` | 大多数 RL 训练 | 简单、稳定、快，PD 控制直接给力 |
| `DCMotorCfg` | 模拟真实电机特性 | 有电枢电感、反电动势，更真实但慢 |
| `ActuatorNetLSTMCfg` 等 | 神经网络拟合电机 | 最真实，但需要真机数据训练 |

> **入门建议：** 先用 `ImplicitActuatorCfg` 跑通，等策略训练好了再换更真实的模型。

#### D. 为什么要分组？

不同类型的关节混在一起会出问题：

- **腿部关节**：旋转关节，单位 rad，需要快速响应
- **机械臂关节**：旋转关节，但要保持固定姿态（刚度要高）
- **手爪关节**：移动关节，单位 m，速度慢、行程小

分组后可以给每组配置不同的参数，更精准也更安全。

---

### 3.4 全局参数

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `soft_joint_pos_limit_factor` | 软限位系数 | 0.95（硬限位的 95% 处开始限制） |
| `fix_base_link` | 是否固定基座 | `False`（四足 locomotion 必须浮空） |

---

## 四、学习路径建议

### 第 1 周：跑通官方示例，学会"抄"

1. **看官方自带的四足配置**，对比着学：
   - `isaaclab_assets/robots/anymal_d/anymal_d.py`
   - `isaaclab_assets/robots/spot/spot.py`
   - 人家怎么写的，你就怎么写

2. **跑通一个官方 locomotion 环境**，比如 `anymal_c_velocity`，先看效果，再慢慢改参数。

3. **对照法学习**（最推荐）：
   - 左边打开官方四足配置
   - 右边打开你的 `xgo.py`
   - 逐行对比，不懂的参数记下来去查文档

### 第 2 周：理解每个参数的物理意义

这些参数本质上都是**机器人学**基础知识，不用学太深，知道概念就行：

- **关节空间 vs 任务空间** → 为什么用关节角度控制
- **PID 控制** → stiffness/damping 就是 PD 控制
- **正运动学** → 怎么从关节角算脚的位置
- **刚体动力学** → 质量、惯量、摩擦力矩

### 第 3 周：对着自己的机器人调参

按这个顺序调，**每次只改一个参数**：

```
1. 确认 URDF 正确（质量、惯量、关节限位、关节名称）
   ↓
2. 调整初始高度 z，让机器人平稳站立
   ↓
3. 调腿部 stiffness/damping，让机器人站着不抖
   ↓
4. 用固定周期步态（正弦信号）测试能不能走
   ↓
5. 再上 RL 训练
```

---

## 五、推荐学习资源

| 资源 | 说明 |
|------|------|
| **Isaac Lab 官方文档** | https://isaac-sim.github.io/IsaacLab/ （查配置类必看） |
| **Isaac Lab 源码** | 直接看 `ArticulationCfg`、`ImplicitActuatorCfg` 的类定义和注释 |
| **《机器人学导论》(Craig)** | 经典教材，前 6 章够了 |
| **B站 机器人学课程** | 上交/哈工大的都可以，补基础概念 |
| **XGO 官方手册** | 找电机参数、关节限位、重量等真实数据 |
| **Unitree Robotics 论文** | 四足控制论文里有很多参数参考 |

---

## 六、常见问题排查

### Q1: 机器人一启动就散架/乱飞？
- 检查 `fix_base_link` 是否为 `False`
- 检查 URDF 中质量/惯量是否合理（太小会不稳定）
- 调大 `solver_position_iteration_count`

### Q2: 机器人站不稳，一直抖？
- 增大 `damping`
- 减小 `stiffness`
- 检查初始高度是否合适

### Q3: 训练时关节经常触发限位？
- 调大 `soft_joint_pos_limit_factor`（比如 0.98）
- 或者检查初始姿态和目标步态是否合理

### Q4: 脚穿进地面里？
- 调大初始高度 `pos[2]`
- 调大 `max_depenetration_velocity`
- 检查地面碰撞体是否正确

### Q5: 训练出来的策略真机用不了？
- 首先检查 `effort_limit` 和 `velocity_limit` 是否跟真机一致
- 检查 `friction` 是否设置了合理的摩擦力
- 考虑从 `ImplicitActuatorCfg` 换成更真实的电机模型

---

## 七、验证清单

配置写完后，按这个清单逐项检查：

- [ ] 路径是动态获取的，没有硬编码绝对路径
- [ ] URDF 文件路径正确，能找到文件
- [ ] `effort_limit` 与 URDF / 真机一致
- [ ] `velocity_limit` 与 URDF / 真机一致
- [ ] 旋转关节和移动关节分开配置
- [ ] 腿部关节、机械臂关节、手爪关节分了不同的组
- [ ] `fix_base_link=False`（locomotion 任务）
- [ ] 初始关节角度都在 URDF 限位范围内
- [ ] 初始高度合适，脚刚好触地
- [ ] 前后腿姿态对称（或有明确的不对称理由）
- [ ] 关节名称与 URDF 完全一致（拼写、大小写）
