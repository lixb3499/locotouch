from __future__ import annotations

import argparse
import copy
import os

from isaaclab_tasks.utils import get_checkpoint_path

import cli_args
from locotouch.distill.student import Student


def _patched_distillation_args(args_cli: argparse.Namespace) -> argparse.Namespace:
    patched = copy.deepcopy(args_cli)
    if not hasattr(patched, "logger"):
        patched.logger = None
    if not hasattr(patched, "training"):
        patched.training = False
    return patched


def get_student_checkpoint_path(student_task: str, args_cli: argparse.Namespace) -> tuple[str, object]:
    patched_args = _patched_distillation_args(args_cli)
    distillation_cfg = cli_args.parse_distillation_cfg(student_task, patched_args)
    distillation_log_root = os.path.abspath(
        os.path.join(distillation_cfg.log_root_path, distillation_cfg.experiment_name)
    )
    resume_path = get_checkpoint_path(
        distillation_log_root, distillation_cfg.log_dir_distill, distillation_cfg.checkpoint_distill
    )
    return resume_path, distillation_cfg


def make_student_policy(
    student_task: str,
    args_cli: argparse.Namespace,
    proprioception_dim: int,
    tactile_signal_dim,
    action_dim: int,
) -> tuple[Student, str, object]:
    resume_path, distillation_cfg = get_student_checkpoint_path(student_task, args_cli)
    student = Student(
        distillation_cfg,
        proprioception_dim=proprioception_dim,
        tactile_signal_dim=tactile_signal_dim,
        action_dim=action_dim,
    )
    student.load_checkpoint(resume_path)
    student.eval()
    return student, resume_path, distillation_cfg


def get_side_observations(env_obs: dict, side: str) -> dict:
    return {
        "policy": env_obs[f"{side}_policy"],
        "tactile": env_obs[f"{side}_tactile"],
    }
