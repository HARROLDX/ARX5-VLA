from openpi.model_dp.diffusion_policy import DiffusionPolicyModel


class DiffusionPolicyLeftModel(DiffusionPolicyModel):
    """Left-arm DP model.

    The parent implementation is already dimension-agnostic through the config;
    this subclass gives the single-arm model a distinct import path/checkpoint
    identity without forking the proven DP internals.
    """

    pass
