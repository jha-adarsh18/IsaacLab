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

    event_frame: torch.Tensor = None
    """The retrieved event data with shape (N, H, W, 2) where
    N is the number of sensors, H and W are the height and width of the camera sensor, 
    and the last dimension corresponds to the number of positive and negative events, respectively. 

    Accumulated over one simulation step, which can be interpreted as a discrete time bin for event generation.

    (dtype is int32, as the number of events are discrete counts)
    """

    raw_events: torch.Tensor = None
    """The retrieved raw event data with shape (N, E, 4) where N is the number of sensors, E is the maximum number of events that can be stored,
    and the last dimension corresponds to a tuple of (x, y, polarity, timestamp) for each event.
    
    (dtype is int32 for x, y, and polarity, and float32 for timestamp)
    
    Note: This is reserved for future implementation of true asynchronus event generation.
    Currently, only the accumulated event frame (for the duration of the simulation time step) is implemented and returned in `event_frame`.
    """
    
    log_intensity: torch.Tensor = None
    """The retrieved log intensity data with shape (N, H, W) where N is the number of sensors, H and W are the height and width of the camera sensor.
    
    (dtype is float32, representing the log intensity of the image)"""

    motion_vectors: torch.Tensor = None
    """The retrieved motion vector data with shape (N, H, W, 2) where N is the number of sensors, H and W are the height and width of the camera sensor,
    and the last dimension corresponds to the motion vector in x and y directions, respectively.
    
    (dtype is float32, representing the motion vectors of the image)"""