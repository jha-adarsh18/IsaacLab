# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sim import PinholeCameraCfg

from .event_camera import EventCamera

@configclass
class EventCameraCfg(SensorBaseCfg):
    """Configuration for an event camera sensor."""
    
    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""

        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

        convention: Literal["opengl", "ros", "world"] = "ros"
        """The convention in which the frame offset is applied. Defaults to "ros".

        - ``"opengl"`` - forward axis: ``-Z`` - up axis: ``+Y`` - Offset is applied in the OpenGL (Usd.Camera)
          convention.
        - ``"ros"``    - forward axis: ``+Z`` - up axis: ``-Y`` - Offset is applied in the ROS convention.
        - ``"world"``  - forward axis: ``+X`` - up axis: ``+Z`` - Offset is applied in the World Frame convention.

        """

    class_type: type = EventCamera

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity.
    
    Note:
        The parent frame is the frame the sensor attaches to. For example, the parent frame of a
        camera at path ``/World/envs/env_0/Robot/Camera`` is ``/World/envs/env_0/Robot``.
    """

    spawn: PinholeCameraCfg | None = MISSING
    """Spawn configuration for the asset.

    If None, then the prim is not spawned by the asset. Instead, it is assumed that the
    asset is already present in the scene.
    """

    width: int = MISSING
    """width of the event camera sensor in pixels.
    
    Note: This is also the width of the event frame retrieved from the sensor.
    """

    height: int = MISSING
    """height of the event camera sensor in pixels.

    Note: This is also the height of the event frame retrieved from the sensor.
    """

    update_latest_camera_pose: bool = False
    """Whether to update the latest camera pose when fetching the camera's data. Defaults to False.

    If True, the latest camera pose is updated in the camera's data which will slow down performance
    due to the use of :class:`XformPrimView`.
    If False, the pose of the camera during initialization is returned.
    """

    # Event geenration parameters
    # Two different threholds represent the true nature of how the silicon circuit detects increase and decrease in log intensity separately.
    # Can be tuned separately to achieve a more realistic event generation and thus, better sim-to-real transfer performance.
    
    pos_threshold: float = 0.1
    """The threshold for positive event generation. Defaults to 0.1.
    
    It is defined as the minimum change in log intensity required to trigger a positive event.
    A lower threshold will result in more positive events being generated, while a higher threshold will result in fewer positive events being generated.
    """

    neg_threshold: float = 0.1
    """The threshold for negative event generation. Defaults to 0.1.

    It is defined as the minimum change in log intensity required to trigger a negative event.
    A lower threshold will result in more negative events being generated, while a higher threshold will result in fewer negative events being generated.
    """