# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Xgo-Demo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.xgo_demo_env_cfg:XgoDemoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Xgo-Dog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.xgo_demo_dog:XgoDemoDogEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Xgo-Dog-Distill-v0",                                        # 对应 Go2-velocity-Distill-v0
    entry_point=f"{__name__}.aer_env:AERManagerBasedRLEnv",          # 如果你的 aer_env.py 里有这个类
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.xgo_demo_dog:XgoDemoDogEnvCfg",   # 对应 GO2RobotDemoEnv
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DistillationRunnerCfg",
    },
)
