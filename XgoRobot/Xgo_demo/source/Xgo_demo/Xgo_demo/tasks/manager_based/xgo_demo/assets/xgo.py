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
            # 后左腿 BL - 注意：后腿关节轴与前腿可能反向，
            # 如果姿态不对，把 thigh 和 calf 的符号都取反试试
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
        # 匹配 fl_/fr_/bl_/br_ 开头的 hip/thigh/calf 关节
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
        # 单独一组，用更高刚度保持收起姿态不动
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["arm_thigh_joint", "arm_calf_joint"],
            effort_limit=0.45,
            velocity_limit=10.0,
            stiffness=50.0,          # 更高刚度，保持固定姿态
            damping=1.0,
            friction=0.0,
        ),
        # --- 手爪 2 个移动关节 (prismatic) ---
        # 单独一组，单位是米不是弧度，速度/增益都要调整
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
