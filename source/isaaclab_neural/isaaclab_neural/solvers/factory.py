from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import newton
import torch

from .neural_solver import NeuralSolver
from .rnn_neural_solver import RNNNeuralSolver
from .transformer_neural_solver import TransformerNeuralSolver


_NEURAL_SOLVER_CLASSES = {
    "NeuralSolver": NeuralSolver,
    "RNNNeuralSolver": RNNNeuralSolver,
    "TransformerNeuralSolver": TransformerNeuralSolver,
}


def _valid_solver_args(solver_cls: type[NeuralSolver]) -> set[str]:
    """Return constructor kwargs accepted by a solver class and its NeuralSolver base."""
    solver_sig = inspect.signature(solver_cls.__init__)
    valid_args = {
        name
        for name, parameter in solver_sig.parameters.items()
        if name != "self" and parameter.kind != inspect.Parameter.VAR_KEYWORD
    }

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in solver_sig.parameters.values()
    )
    if accepts_kwargs:
        base_sig = inspect.signature(NeuralSolver.__init__)
        valid_args.update(
            name
            for name, parameter in base_sig.parameters.items()
            if name != "self" and parameter.kind != inspect.Parameter.VAR_KEYWORD
        )

    return valid_args - {"model", "contacts", "neural_model", "num_contacts_per_env"}


def create_neural_solver(
    solver_cfg: Mapping[str, Any] | Any,
    model: newton.Model,
    contacts: newton.Contacts | None,
    neural_model: torch.nn.Module | None,
    num_contacts_per_env: int,
    **overrides,
) -> NeuralSolver:
    """Create a NeRD neural solver from config-like data.

    Args:
        solver_cfg: Mapping, configclass, or object with ``to_dict()``.
        model: Newton model.
        contacts: Fixed-ground contacts or native Newton contacts.
        neural_model: Optional neural model used for prediction.
        num_contacts_per_env: Fixed contact width per environment.
        **overrides: Runtime-only values such as ``contact_mode`` and
            ``contact_adapter``.
    """
    if hasattr(solver_cfg, "to_dict"):
        cfg_dict = solver_cfg.to_dict()
    elif isinstance(solver_cfg, Mapping):
        cfg_dict = dict(solver_cfg)
    else:
        cfg_dict = dict(vars(solver_cfg))

    cfg_dict.pop("solver_type", None)
    cfg_dict.pop("neural_model_path", None)
    cfg_dict.pop("neural_model_cfg", None)
    cfg_dict.pop("contact_fingerprint", None)
    cfg_dict.pop("validate_contact_fingerprint", None)
    cfg_dict.pop("contact_packing_policy", None)
    cfg_dict.pop("use_cuda_graph", None)
    cfg_dict.pop("num_contacts_per_env", None)
    solver_cls_name = cfg_dict.pop("name", cfg_dict.pop("neural_solver_name", "NeuralSolver"))
    solver_cls = _NEURAL_SOLVER_CLASSES.get(solver_cls_name)
    if solver_cls is None:
        raise NotImplementedError(
            f"Unknown neural solver type: {solver_cls_name}. "
            f"Available: {list(_NEURAL_SOLVER_CLASSES.keys())}"
        )

    cfg_dict.update(overrides)
    valid_solver_args = _valid_solver_args(solver_cls)
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_solver_args}
    return solver_cls(
        model=model,
        contacts=contacts,
        neural_model=neural_model,
        num_contacts_per_env=num_contacts_per_env,
        **cfg_dict,
    )
