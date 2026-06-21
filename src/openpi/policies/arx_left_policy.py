"""Policy transforms for the ARX left-arm-only dataset."""

import dataclasses
from typing import ClassVar

import numpy as np
import torch

import openpi.models.model as _model
import openpi.transforms as transforms


@dataclasses.dataclass(frozen=True)
class ARXLeftInputs(transforms.DataTransformFn):
    """Inputs for left-arm DP.

    Expected dataset inputs:
    - images: top_head and hand_left, optionally camera_third for 3cam DP
    - state: [7]
    - actions: [action_horizon, 7]
    """

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.DP

    required_rename_map: ClassVar[dict[str, str]] = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
    }
    third_camera_rename_map: ClassVar[dict[str, str]] = {
        "camera_third": "third_0_rgb",
    }

    mask_state: bool = False
    include_third_camera: bool = False

    @property
    def camera_rename_map(self) -> dict[str, str]:
        if self.include_third_camera:
            return {**self.required_rename_map, **self.third_camera_rename_map}
        return dict(self.required_rename_map)

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        camera_rename_map = self.camera_rename_map
        expected_cameras = tuple(camera_rename_map.keys())
        if set(in_images) - set(expected_cameras):
            raise ValueError(f"Expected images to contain {expected_cameras}, got {tuple(in_images)}")

        state = transforms.pad_to_dim(data["state"], self.action_dim).squeeze()

        images = {}
        image_masks = {}
        for camera in expected_cameras:
            if camera not in in_images:
                raise ValueError(f"Camera {camera} not found in data")
            img = in_images[camera]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            target_key = camera_rename_map[camera]
            images[target_key] = img
            image_masks[target_key] = np.True_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": np.zeros_like(state) if self.mask_state else state,
        }

        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            joint_mask = np.ones(actions.shape[-1], dtype=bool)
            joint_mask[6] = False
            actions = np.where((actions > np.pi) & joint_mask, 0, actions)
            actions = np.where((actions < -np.pi) & joint_mask, 0, actions)
            inputs["actions"] = actions.squeeze()

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ARXLeftOutputs(transforms.DataTransformFn):
    """Outputs for the ARX left-arm-only policy."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
