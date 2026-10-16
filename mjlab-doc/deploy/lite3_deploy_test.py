"""lite3_deploy_test.py — luwu_mjlab Lite3-Walk-Flat 策略真机测试（实验级）.

在机器狗板载电脑（或经 USB-TTL 接机器狗的 PC）上运行：

  标定三步（按顺序，结果自动写入 calib_result.json，无需手填代码）：
    python3 lite3_deploy_test.py --stage calib_map    # ① P+SIGN：卸力后逐个扳关节，程序自动识别
    python3 lite3_deploy_test.py --stage calib_zero   # ② M0：上电 reset() 站姿自动读数（不卸力！）
    python3 lite3_deploy_test.py --stage calib_imu    # ③ G_SIGN：做三个动作，自动识别陀螺仪方向
    python3 lite3_deploy_test.py --stage calib_all    # 或：三步连跑

  验证与闭环：
    python3 lite3_deploy_test.py --stage obs      # ④ 观测检查：只读 + 推理，不发舵机指令
    python3 lite3_deploy_test.py --stage hold     # ⑤ 保持默认站姿（验证执行链路 P/M0）
    python3 lite3_deploy_test.py --stage policy   # ⑥ 策略闭环（务必先架空！）
    python3 lite3_deploy_test.py --stage policy --vx 0.2   # ⑦ 给一点前进指令

依赖：pip install onnxruntime numpy pyserial xgolib
策略：拷贝 logs/rsl_rl/lite3_velocity_walk/2026-10-14_16-06-38/policy.onnx 到本目录
"""

import argparse
import json
import math
import os
import time

import numpy as np
import onnxruntime as ort

from xgolib.xgolib_dog import XGO_DOG

# =====================================================================
# ① 策略常量 —— 来源：policy.onnx 元数据（怎么查见教程第 4 篇 §1.1）
# =====================================================================
ONNX_PATH = "policy.onnx"

# 仿真关节顺序（元数据 joint_names 字段原文）
SIM_JOINTS = [
    "FL_hip_joint", "FL_thigh_joint", "FL_shank_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_shank_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_shank_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_shank_joint",
    "mainarm_joint", "forearm_joint"
]
# 默认位型（rad，元数据 default_joint_pos 字段：hip 0 / thigh 0.95 / shank 0 ×4 腿 + 双臂）
DEFAULT_POS = np.array(
    [0.0, 0.95, 0.0] * 4 + [-1.55, 1.3]
)
ACTION_SCALE = 0.25          # 元数据 action_scale
OBS_DIM, ACT_DIM = 53, 14    # onnx 输入/输出维度
PHASE_PERIOD = 0.4           # env.yaml: observations.actor.terms.phase.params.period

# =====================================================================
# ② 标定结果 —— 由 calib_* 阶段自动写入 calib_result.json，启动时自动加载
#    下面的常量只是【占位/兜底】，标定后一切以 calib_result.json 为准
# =====================================================================
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_result.json")

# P[i]   : 第 i 个仿真关节对应的 XGO 舵机 ID（腿 11..43，臂 51..53）
P = [11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43, 52, 53]
# SIGN[i]: 仿真关节正方向 与 舵机角度正方向 是否一致（+1 / -1）
SIGN = np.array([1] * 14, dtype=float)
# M0[i] : 默认站姿下 read_motor 读到的舵机角度（度）
M0 = np.zeros(14)
# G_SIGN : 陀螺仪三轴方向与仿真是否一致（+1 / -1）
G_SIGN = np.array([1.0, 1.0, 1.0])


def load_calib():
    """启动时加载标定结果，覆盖顶部占位常量。"""
    global P, SIGN, M0, G_SIGN
    if not os.path.exists(CALIB_FILE):
        return
    with open(CALIB_FILE, "r") as f:
        d = json.load(f)
    if "P" in d:
        P = list(d["P"])
    if "SIGN" in d:
        SIGN = np.array(d["SIGN"], dtype=float)
    if "M0" in d:
        M0 = np.array(d["M0"], dtype=float)
    if "G_SIGN" in d:
        G_SIGN = np.array(d["G_SIGN"], dtype=float)
    print(f"已加载标定文件 {CALIB_FILE}（含字段: {list(d.keys())}）")


def save_calib(**kwargs):
    """把某个阶段的结果合并写回 calib_result.json。"""
    d = {}
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, "r") as f:
            d = json.load(f)
    d.update(kwargs)
    with open(CALIB_FILE, "w") as f:
        json.dump(d, f, indent=2)
    print(f"已写入 {CALIB_FILE}：{list(kwargs.keys())}")


# =====================================================================
# ③ 安全参数
# =====================================================================
TARGET_DT = 1.0 / 50.0       # 训练控制频率 50 Hz（0.004s × decimation 5）
SAFETY_MAX_DELTA_DEG = 10.0  # 每步关节目标变化限幅（度），标定不稳时先调小
MOTOR_IDS_ALL = [11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43, 51, 52, 53]

# =====================================================================
# ④ 仿真正方向对照表 —— 由 lite3.xml 各关节 axis 按右手定则推出
#    （hip: axis=-1 0 0 → 足端向狗右侧摆；thigh: axis=0 1 0 → 大腿后摆；
#      shank: axis=0 -1 0 → 小腿前摆；双臂: axis=0 1 0 → 向前摆）
#    calib_map 会按这张表引导你扳动，方向对错直接决定 SIGN 对错！
# =====================================================================
_LEG_CN = {"FL": "左前腿", "FR": "右前腿", "RR": "右后腿", "RL": "左后腿"}
_JOINT_CN = {
    "hip": "【髋】把足端向狗的右侧摆（左腿=向中线内收，右腿=向外展）",
    "thigh": "【大腿】把大腿向后摆（足端向后）",
    "shank": "【小腿】把小腿向前摆（膝盖弯曲）",
}


def joint_prompt(joint_name: str) -> str:
    """生成单个关节的扳动指引。"""
    if joint_name == "mainarm_joint":
        return "机械臂【大臂】向前摆"
    if joint_name == "forearm_joint":
        return "机械臂【小臂】向前摆"
    leg, kind = joint_name.split("_")[:2]
    return _LEG_CN[leg] + _JOINT_CN[kind]


def read_servo_deg(dog: XGO_DOG) -> dict:
    """读 15 个舵机角度（度），按舵机 ID 存字典。"""
    angles = dog.read_motor()
    return {mid: a for mid, a in zip(MOTOR_IDS_ALL, angles)}


def read_servo_deg_avg(dog: XGO_DOG, n: int = 5, interval: float = 0.05) -> dict:
    """连续读 n 次取平均，降低舵机读数噪声。"""
    acc = {mid: 0.0 for mid in MOTOR_IDS_ALL}
    for _ in range(n):
        deg = read_servo_deg(dog)
        for mid in MOTOR_IDS_ALL:
            acc[mid] += deg[mid]
        time.sleep(interval)
    return {mid: v / n for mid, v in acc.items()}


def build_obs(dog, imu, cmd, joint_pos_rel, joint_vel_rel, prev_action, t):
    """按 policy.onnx 的 53 维布局组装观测（顺序见 observation_names 元数据）：
    base_ang_vel(3) + projected_gravity(3) + command(3) + phase(2)
    + joint_pos(14) + joint_vel(14) + actions(14) = 53
    """
    # 1) base_ang_vel：陀螺仪 deg/s → rad/s（read_imu 返回 [ax,ay,az,gx,gy,gz,r,p,yw]）
    ang_vel = G_SIGN * np.array(imu[3:6]) * math.pi / 180.0
    # 2) projected_gravity：由 roll/pitch 重构机体系重力方向（单位向量）
    #    read_imu 的欧拉角已被库转成近似弧度（用的是 3.14，误差可忽略）
    roll, pitch = imu[6], imu[7]
    proj_g = np.array([
        math.sin(pitch),
        -math.sin(roll) * math.cos(pitch),
        -math.cos(roll) * math.cos(pitch),
    ])
    # 3) command：[vx, vy, wz]，机体系（vel_command_b 的排布）
    # 4) phase：sin/cos 时钟，period=0.4s；指令接近 0 时置零（stand_mask）
    if np.linalg.norm(cmd) < 0.1:
        phase = np.zeros(2)
    else:
        ph = (t % PHASE_PERIOD) / PHASE_PERIOD * 2.0 * math.pi
        phase = np.array([math.sin(ph), math.cos(ph)])
    obs = np.concatenate([
        ang_vel, proj_g, cmd, phase,
        joint_pos_rel, joint_vel_rel, prev_action,
    ])
    assert obs.shape == (OBS_DIM,), f"obs 维度错误: {obs.shape}"
    return obs.astype(np.float32)


# =====================================================================
# 标定阶段一：P + SIGN（卸力手扳，程序自动识别）
# =====================================================================
def stage_calib_map(dog: XGO_DOG):
    """标定 P（关节→舵机ID 映射）和 SIGN（方向符号）。

    原理：卸力后关节可徒手扳动且编码器仍在计数。每个关节做两轮采样：
      第 1 轮（基准）：狗摆回自然站姿，记录 15 个舵机读数 baseline；
      第 2 轮（测量）：按【仿真正方向对照表】把指定关节扳动 >10° 并保持，
                       再记录一轮。两轮之差中绝对值最大的那个舵机 ID
                       就是这个关节 → P[i]；差值的正负 → SIGN[i]。
    每轮取 5 次平均降噪；变化量 <4° 认为没扳动，要求重扳。
    """
    print("=" * 64)
    print("标定 P + SIGN：即将卸力，之后可徒手扳动关节。")
    print("每个关节两步：① 摆回站姿按回车（记基准） ② 按指引扳住不动按回车（测量）")
    print("扳动方向必须严格按指引（= 仿真正方向），否则 SIGN 会标反！")
    print("=" * 64)
    input("确认狗已架空或放稳，按回车开始卸力...")
    dog.unload_allmotor()
    print("已卸力。\n")

    P_new, SIGN_new = [0] * 14, [0] * 14
    for i, jn in enumerate(SIM_JOINTS):
        while True:
            input(f"[{i+1:2d}/14] {jn}\n"
                  f"  ① 把狗摆回自然站姿，按回车记录基准...")
            base = read_servo_deg_avg(dog)
            input(f"  ② {joint_prompt(jn)}，幅度大于 10°，扳住不动按回车...")
            meas = read_servo_deg_avg(dog)

            delta = {mid: meas[mid] - base[mid] for mid in MOTOR_IDS_ALL}
            mid = max(delta, key=lambda k: abs(delta[k]))
            if abs(delta[mid]) < 4.0:
                print(f"  ✗ 最大变化量仅 {abs(delta[mid]):.1f}°（{mid} 号），"
                      "没检测到有效扳动，请重试（扳大一些、扳住别松）\n")
                continue
            if mid in P_new:
                dup = SIM_JOINTS[P_new.index(mid)]
                print(f"  ✗ 检测到 {mid} 号变化最大，但它已分配给 {dup}！"
                      "你可能扳错了关节，请重试\n")
                continue
            P_new[i] = mid
            SIGN_new[i] = 1 if delta[mid] > 0 else -1
            print(f"  ✓ {jn} → 舵机 {mid} 号，变化 {delta[mid]:+.1f}°，"
                  f"SIGN = {SIGN_new[i]:+d}\n")
            break

    save_calib(P=P_new, SIGN=[int(s) for s in SIGN_new])
    print("\nP    =", P_new)
    print("SIGN =", SIGN_new)
    print("下一步：python3 lite3_deploy_test.py --stage calib_zero")


# =====================================================================
# 标定阶段二：M0（上电站姿自动读数，禁止卸力）
# =====================================================================
def stage_calib_zero(dog: XGO_DOG):
    """标定 M0（默认站姿下各舵机的角度零点）。

    原理：仿真观测/动作都是『相对默认位型的偏移』，而舵机读数是其自身坐标系
    的绝对角度，两者零点不同。M0 就是默认站姿下舵机的读数，起到零点对齐作用：
        pos_rel = SIGN × radians(读数[P] − M0)
    做法：让固件自己站成初始姿势（reset()），连读 20 次取平均。
    注意：全程【不能卸力】——卸力后狗瘫掉，读数就不是站姿了。

    前提假设：固件 reset 站姿 == 仿真默认位型。可用仿真几何指纹核验：
        站高（机身到地面）≈ 8.7 cm；前后足落点间距 ≈ 13.4 cm；左右 ≈ 10.0 cm
    若不符，说明两边站姿不一致，M0 需按 obs 阶段 pos_rel 的残差修正。
    """
    print("=" * 64)
    print("标定 M0：需要固件保持站姿，本阶段【不卸力】。")
    print("开始前请先用尺子核对（仿真默认位型的几何指纹）：")
    print("  站高 ≈ 8.7 cm | 前后足间距 ≈ 13.4 cm | 左右足间距 ≈ 10.0 cm")
    print("若差距明显（>1cm），说明固件站姿 ≠ 仿真默认位型，先不要继续！")
    print("=" * 64)
    input("狗已放在平地、上电，按回车执行 reset() 并读数...")

    dog.reset()
    time.sleep(2.0)  # 等站姿稳定

    deg = read_servo_deg_avg(dog, n=20, interval=0.05)
    M0_new = [round(deg[p], 2) for p in P]  # 按 P 的顺序重排成仿真关节顺序

    print("\n各舵机原始读数（度）：")
    print({mid: round(v, 1) for mid, v in deg.items()})
    save_calib(M0=M0_new)
    print("\nM0 =", M0_new)
    print("自检：M0 不应全为 0 或全相同，否则 read_motor 可能没读到数。")
    print("下一步：python3 lite3_deploy_test.py --stage calib_imu")


# =====================================================================
# 标定阶段三：G_SIGN（做动作，自动识别陀螺仪方向）
# =====================================================================
# 动作 → （期望的仿真角速度轴, 仿真中的符号）
# 仿真机体系 x 前 / y 左 / z 上，右手定则：
#   向右倾  → 绕 +x 正转 → sim ωx > 0
#   抬鼻子  → 绕 +y 反转 → sim ωy < 0（注意：+ωy 是低头！）
#   向左转头 → 绕 +z 正转 → sim ωz > 0
_IMU_TESTS = [
    ("把狗【向右倾斜】（右侧下沉）", 0, +1.0),
    ("把狗头【向上抬】（nose up）", 1, -1.0),
    ("把狗头【向左转】（yaw left）", 2, +1.0),
]


def _detect_gyro_peak(dog: XGO_DOG, duration: float = 4.0) -> tuple:
    """在 duration 秒内高频读陀螺仪，返回峰值时刻的 (轴序号, 原始值)。"""
    best_axis, best_val = -1, 0.0
    t_end = time.perf_counter() + duration
    while time.perf_counter() < t_end:
        gyro = dog.read_imu()[3:6]
        for k in range(3):
            if abs(gyro[k]) > abs(best_val):
                best_axis, best_val = k, gyro[k]
        time.sleep(0.02)
    return best_axis, best_val


def stage_calib_imu(dog: XGO_DOG):
    """标定 G_SIGN（陀螺仪三轴方向符号）。

    原理：build_obs 中 base_ang_vel = G_SIGN × gyro_raw，要求变换后的符号
    与仿真机体系一致（x 前 / y 左 / z 上，右手定则）。做法：让你做一个已知
    方向的单轴动作，程序在 4 秒窗口内找角速度峰值：
      - 峰值出现在哪个轴 → 物理上该动作激励了哪根轴（应等于期望轴，
        不等说明 IMU 轴向有置换，G_SIGN 解决不了，需要改 build_obs 的轴序）
      - 峰值的正负 与 该动作在仿真中的理论符号 比较 → G_SIGN
    """
    print("=" * 64)
    print("标定 G_SIGN：依次做 3 个单轴动作，每个动作有 4 秒窗口。")
    print("动作要干脆（快做快停），幅度大没关系，程序只看峰值方向。")
    print("=" * 64)

    g_sign_new = [1.0, 1.0, 1.0]
    axis_warn = False
    for prompt, expect_axis, sim_sign in _IMU_TESTS:
        input(f"\n准备：{prompt}。按回车后开始 4 秒采样，请在此期间完成动作...")
        axis, val = _detect_gyro_peak(dog)
        raw_sign = 1.0 if val > 0 else -1.0
        g_sign_new[expect_axis] = sim_sign * raw_sign
        status = "✓" if axis == expect_axis else "✗ 轴向不符！"
        print(f"  峰值出现在 gyro[{axis}] = {val:+.1f} deg/s（期望轴 gyro[{expect_axis}]）{status}")
        print(f"  → G_SIGN[{expect_axis}] = {g_sign_new[expect_axis]:+.0f}")
        if axis != expect_axis:
            axis_warn = True
        if abs(val) < 20.0:
            print("  ⚠ 峰值角速度偏小（<20 deg/s），动作可能太轻，结果可信度低，建议重测本动作")

    save_calib(G_SIGN=g_sign_new)
    print("\nG_SIGN =", g_sign_new)
    if axis_warn:
        print("⚠ 存在轴向置换：G_SIGN 只能翻转符号，无法交换轴顺序，"
              "请检查 IMU 安装方向或修改 build_obs 中 gyro 的取轴顺序！")
    print("\n标定全部完成！接下来按顺序验证：")
    print("  python3 lite3_deploy_test.py --stage obs")
    print("  python3 lite3_deploy_test.py --stage hold")
    print("  python3 lite3_deploy_test.py --stage policy   （先架空！）")


def stage_calib_all(dog: XGO_DOG):
    """三步标定连跑。calib_zero 需要固件站姿，故安排在 map（卸力）之后，
    且内部会先让固件重新接管（reset 自带上力）。"""
    stage_calib_map(dog)
    print("\n>>> 进入第二步前，请确保狗已放回平地，固件将重新上力 <<<\n")
    load_calib()  # 把刚写盘的 P 加载进来，calib_zero 读数要按 P 重排
    stage_calib_zero(dog)
    stage_calib_imu(dog)


def stage_obs(dog: XGO_DOG, session, cmd):
    """只读 + 推理：验证观测组装和推理链路，不发任何舵机指令。"""
    print("观测检查：让狗保持站姿（固件控制），观察 obs 数值是否合理、推理是否出数。Ctrl+C 退出\n")
    prev_action = np.zeros(ACT_DIM)
    prev_pos = None
    t0 = time.perf_counter()
    try:
        while True:
            t = time.perf_counter() - t0
            deg = read_servo_deg(dog)
            pos_rel = SIGN * np.radians(
                np.array([deg[p] for p in P]) - M0
            )
            vel_rel = (
                (pos_rel - prev_pos) / TARGET_DT if prev_pos is not None
                else np.zeros(ACT_DIM)
            )
            prev_pos = pos_rel
            obs = build_obs(dog, dog.read_imu(), cmd, pos_rel, vel_rel, prev_action, t)
            action = session.run(["actions"], {"obs": obs[None]})[0][0]
            prev_action = action
            print(
                "obs[:9]:", np.round(obs[:9], 3),
                "| pos_rel[:3]:", np.round(pos_rel[:3], 3),
                "| action[:3]:", np.round(action[:3], 3),
            )
            time.sleep(TARGET_DT)
    except KeyboardInterrupt:
        print("\n检查点：静止时 proj_g 应≈[0,0,-1]，pos_rel 应≈0，action 应有小幅输出。")
        print("若 pos_rel 某关节明显不为 0（>0.1 rad），说明 M0 有偏差，")
        print("可按 M0[i] += degrees(pos_rel[i]) × SIGN[i] 修正后重测。")


def stage_hold(dog: XGO_DOG):
    """保持默认站姿：把 M0 原样发回去。狗的姿势不变 → 执行链路 P/M0 正确。"""
    print("保持默认站姿 10 秒。若狗姿势突变/腿乱动 → P 或 M0 有错，立即 Ctrl+C！\n")
    try:
        t_end = time.perf_counter() + 10.0
        while time.perf_counter() < t_end:
            dog.motor(list(P), [float(m) for m in M0])
            time.sleep(TARGET_DT)
        dog.stop()
        dog.reset()
    except KeyboardInterrupt:
        dog.stop()
        dog.reset()


def stage_policy(dog: XGO_DOG, session, cmd):
    """策略闭环：obs → onnx → 目标角度 = M0 + SIGN×deg(0.25×action)。先架空！"""
    print(
        f"策略闭环，指令 vx={cmd[0]} vy={cmd[1]} wz={cmd[2]}。"
        "务必架空！Ctrl+C 急停（停机+回正）。\n"
    )
    prev_action = np.zeros(ACT_DIM)
    prev_pos = None
    t0 = time.perf_counter()
    steps, t_mark = 0, t0
    try:
        while True:
            loop_t = time.perf_counter()
            t = loop_t - t0
            # --- 读传感 ---
            deg = read_servo_deg(dog)
            pos_rel = SIGN * np.radians(np.array([deg[p] for p in P]) - M0)
            vel_rel = (
                (pos_rel - prev_pos) / TARGET_DT if prev_pos is not None
                else np.zeros(ACT_DIM)
            )
            prev_pos = pos_rel
            imu = dog.read_imu()
            # --- 推理 ---
            obs = build_obs(dog, imu, cmd, pos_rel, vel_rel, prev_action, t)
            action = session.run(["actions"], {"obs": obs[None]})[0][0]
            prev_action = action
            # --- 动作 → 舵机角度目标（相对默认位型的偏移，限幅保平安） ---
            delta_deg = SIGN * np.degrees(ACTION_SCALE * action)
            delta_deg = np.clip(delta_deg, -SAFETY_MAX_DELTA_DEG, SAFETY_MAX_DELTA_DEG)
            target_deg = M0 + delta_deg
            dog.motor(list(P), [float(x) for x in target_deg])
            # --- 频率统计 ---
            steps += 1
            if steps % 100 == 0:
                hz = 100.0 / (loop_t - t_mark)
                t_mark = loop_t
                print(f"loop: {hz:.1f} Hz（目标 50；低于 30 会明显劣化）")
            elapsed = time.perf_counter() - loop_t
            if elapsed < TARGET_DT:
                time.sleep(TARGET_DT - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        dog.stop()
        dog.reset()
        print("已急停并回正。")


_HELP_DESC = """\
luwu_mjlab Lite3-Walk-Flat 策略真机部署测试脚本（实验级）

整体流程：标定（calib_*）→ 观测检查（obs）→ 站姿保持（hold）→ 策略闭环（policy）
标定结果自动保存到 calib_result.json，后续阶段启动时自动加载，无需手改代码。
"""

_HELP_EPILOG = """\
阶段（--stage）说明：
  标定三步（按顺序各跑一次，或 calib_all 连跑）：
    calib_map   标定 P（仿真关节→舵机ID 映射）和 SIGN（方向符号）。
                卸力后按屏幕指引逐个扳动关节，程序自动识别 ID 和方向。
                扳动方向必须严格按指引（= 仿真正方向），否则 SIGN 会标反！
    calib_zero  标定 M0（默认站姿下各舵机的零点角度）。
                由固件 reset() 自动站立后读数，全程【不卸力】。
                前提：固件站姿 == 仿真默认位型（先核对站高 ≈8.7cm、
                前后足距 ≈13.4cm、左右足距 ≈10.0cm）。
    calib_imu   标定 G_SIGN（陀螺仪三轴方向符号）。
                按提示做右倾/抬头/左转三个动作，程序自动识别符号。
    calib_all   以上三步连跑。

  验证与闭环（必须先完成标定）：
    obs         只读传感器 + onnx 推理，不发舵机指令。
                检查点：静止时 proj_g≈[0,0,-1]、pos_rel≈0、action 有小幅输出。
    hold        把 M0 原样发回电机保持站姿 10 秒。
                狗姿势突变/腿乱动 = P 或 M0 标错，立即 Ctrl+C！
    policy      策略闭环：obs → onnx → 舵机目标角。
                务必先把狗架空！Ctrl+C 急停（停机+回正）。

常用示例：
  python3 lite3_deploy_test.py --stage calib_all
  python3 lite3_deploy_test.py --stage obs
  python3 lite3_deploy_test.py --stage policy --vx 0.2

速度指令（仅 policy/obs 阶段有效，训练时的指令范围）：
  --vx    前进速度 m/s，训练范围 [-0.2, 0.4]，建议从 0.1~0.2 试起
  --vy    横移速度 m/s，训练范围 [-0.15, 0.15]
  --vyaw  转向速度 rad/s，训练范围 [-0.8, 0.8]，正值向左转
"""


def main():
    parser = argparse.ArgumentParser(
        prog="lite3_deploy_test.py",
        description=_HELP_DESC,
        epilog=_HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,  # 保留换行排版
    )
    parser.add_argument("--stage", required=True,
                        choices=["calib_map", "calib_zero", "calib_imu",
                                 "calib_all", "obs", "hold", "policy"],
                        metavar="STAGE",
                        help="运行阶段（必选），各阶段含义见下方详细说明")
    parser.add_argument("--port", default="/dev/ttyAMA0",
                        help="串口：板载串口默认 /dev/ttyAMA0；"
                             "PC 经 USB-TTL 连接则填 COM5 之类（默认: %(default)s）")
    parser.add_argument("--vx", type=float, default=0.0,
                        help="前进速度指令 m/s，范围 [-0.2, 0.4]（默认: %(default)s）")
    parser.add_argument("--vy", type=float, default=0.0,
                        help="横移速度指令 m/s，范围 [-0.15, 0.15]（默认: %(default)s）")
    parser.add_argument("--vyaw", type=float, default=0.0,
                        help="转向速度指令 rad/s，范围 [-0.8, 0.8]，正值左转（默认: %(default)s）")
    args = parser.parse_args()

    load_calib()  # 标定结果存在则以文件为准（calib_map 除外，它重新生成 P/SIGN）

    dog = XGO_DOG(port=args.port)
    session = None
    if args.stage in ("obs", "policy"):
        session = ort.InferenceSession(
            ONNX_PATH, providers=["CPUExecutionProvider"]
        )
        in_dim = session.get_inputs()[0].shape
        print(f"已加载 {ONNX_PATH}，输入 {in_dim}，输出 {session.get_outputs()[0].shape}")
    # hold/policy 前置检查：M0 是否还是全 0（忘了标定）
    if args.stage in ("hold", "policy") and not np.any(M0):
        raise SystemExit("M0 全 0：请先跑 --stage calib_zero 完成标定！")
    # calib_zero 需要 P（按 P 重排读数）
    if args.stage == "calib_zero" and not os.path.exists(CALIB_FILE):
        print("警告：未找到 calib_result.json，将使用脚本顶部占位 P，"
              "若占位 P 与实际不符，M0 的顺序也会跟着错！建议先跑 calib_map。")

    cmd = np.array([args.vx, args.vy, args.vyaw])
    if args.stage == "calib_map":
        stage_calib_map(dog)
    elif args.stage == "calib_zero":
        stage_calib_zero(dog)
    elif args.stage == "calib_imu":
        stage_calib_imu(dog)
    elif args.stage == "calib_all":
        stage_calib_all(dog)
    elif args.stage == "obs":
        stage_obs(dog, session, cmd)
    elif args.stage == "hold":
        stage_hold(dog)
    elif args.stage == "policy":
        stage_policy(dog, session, cmd)


if __name__ == "__main__":
    main()
