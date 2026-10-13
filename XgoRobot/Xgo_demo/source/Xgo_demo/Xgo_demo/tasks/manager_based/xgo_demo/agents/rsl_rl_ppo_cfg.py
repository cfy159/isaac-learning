# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl import RslRlDistillationRunnerCfg, RslRlDistillationStudentTeacherRecurrentCfg, RslRlDistillationAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "xgo_dog"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

# 请注意继承的这个类
# 命令是 python scripts\rsl_rl\train.py --task Go2-velocity-Distill-v0 --num_envs 400 --load_run 文件夹 --checkpoint model_xx.pt

@configclass
class DistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distill the full-observation teacher into a recurrent deployable student."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "xgo_dog_distill"
    obs_groups = {
        # 这个代表我们的学生网络是student 之前的policy变成教师网络
        "policy": ["student"],
        "teacher": ["policy"],
    }
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        init_noise_std=0.1,
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
        # GRU读取一段历史观测
        rnn_type="gru",
        # 这个hidden_dim一[定要]和教师网络对应上去   ← "定要" 被贴纸遮挡，为推测内容
        rnn_hidden_dim=247,
        rnn_num_layers=1,
        teacher_recurrent=False,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=24,
        max_grad_norm=1.0,
        loss_type="huber",
    )
