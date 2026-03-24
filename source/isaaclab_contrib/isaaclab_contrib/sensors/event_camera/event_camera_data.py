# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

import torch

@dataclass
class EventCameraData:
    """Data container for the event camera sensor."""

    ##
    # Frame state.
    ##

    pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame, following ROS convention.

    Shape is (N, 3) where N is the number of sensors.
    """

    quat_w_world: torch.Tensor = None
    """Quaternion orientation `(w, x, y, z)` of the sensor origin in world frame, following the world coordinate frame

    .. note::
        World frame convention follows the camera aligned with forward axis +X and up axis +Z.

    Shape is (N, 4) where N is the number of sensors.
    """

    ##
    # Camera data
    ##

    image_shape: tuple[int, int] = None
    """A tuple containing (height, width) of the camera sensor."""

    intrinsic_matrices: torch.Tensor = None
    """The intrinsic matrices for the camera.

    Shape is (N, 3, 3) where N is the number of sensors.
    """

    output: dict[str, torch.Tensor] = None
    """
    The retrieved event data with:
    
    - ``"event_frame"``: A tensor of shape (N, H, W, 2) where N is the number of sensors, H and W are the height and width of the camera sensor,
    and the last dimension corresponds to the number of positive and negative events, respectively.
    
    Note: ``"raw_events"`` is a feature to be implemented in the future, and is not currently available.
    """