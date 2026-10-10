import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

# 基于当前文件位置动态获取资产目录，避免硬编码绝对路径
# 当前文件: assets/xgo.py
# 模型根目录: assets/robots/xgo/
#   - USD 文件: assets/robots/xgo/usd/xgo.usd     （方式二：直接加载）
#   - URDF 文件: assets/robots/xgo/urdf/xgo.urdf  （方式一：自动转 USD）
XGO_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robots", "xgo")


# =============================================================================
# 方式一：加载 URDF（Isaac Lab 自动转换为 USD）
#   asset_path 指向 urdf 文件；usd_dir/usd_file_name 是转换后 USD 的缓存路径
# =============================================================================
XGO_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{XGO_MODEL_DIR}/urdf/xgo.urdf",
        usd_dir=f"{XGO_MODEL_DIR}/usd",
        usd_file_name="xgo.usd",
        fix_base=False,                            # 不固定基座，自由移动（floating base）
        activate_contact_sensors=True,             # 启用接触传感器（illegal_contact 等终止项需要）
        # joint_drive：URDF 独有，在 URDF→USD 转换时生成 PhysX 驱动
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness=25.0,
                damping=0.5,
            ),
        ),
        # 刚体物理属性
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        # 关节链物理属性
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
    # IsaacLab 2.x 中基座是否浮动由 URDF 决定（root link 无父关节即为 floating base）
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


# =============================================================================
# 方式二：直接加载 USD（模板，推荐用于正式训练）
#   usd_path 直接指向 usd 文件，省去每次 URDF→USD 转换，加载更快、更可控。
#   与方式一唯一的区别是 spawn 参数（init_state / actuators 完全一致），
#   所以用 .replace() 复用方式一里已经写好的其余配置。
#
#   使用方式：把 xgo_demo_dog.py 里的导入改成
#       from Xgo_demo.assets.xgo import XGO_USD_CFG
#   并把场景里的
#       robot: ArticulationCfg = XGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
#   改成
#       robot: ArticulationCfg = XGO_USD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
# =============================================================================
XGO_USD_CFG = XGO_CFG.replace(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{XGO_MODEL_DIR}/usd/xgo.usd",  # USD 文件路径（由方式一首次运行时自动生成）
        fix_base=False,                            # 不固定基座，自由移动（floating base）
        activate_contact_sensors=True,             # 启用接触传感器（illegal_contact 等终止项需要）
        # 刚体物理属性
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        # 关节链物理属性
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        # 注意：UsdFileCfg 没有 joint_drive 参数（它是 UrdfFileCfg 独有的，
        # 用于 URDF→USD 转换时生成 PhysX 驱动）。USD 文件里已内置关节驱动，
        # 实际控制由上面的 actuators (ImplicitActuatorCfg) 完成。
    ),
)