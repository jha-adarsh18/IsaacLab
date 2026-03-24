# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.sensors import Camera, CameraCfg, SensorBase

from .event_camera_data import EventCameraData

if TYPE_CHECKING:
    from .event_camera_cfg import EventCameraCfg

# import logger
logger = logging.getLogger(__name__)

@wp.func
def _rgb_to_log_intensity(r: wp.uint8, g: wp.uint8, b: wp.uint8) -> wp.float32:
    """Converts RGB values to log intensity using the formula: log(0.299 * R + 0.587 * G + 0.114 * B)."""
    rf = wp.float32(r) / wp.float32(255.0)
    gf = wp.float32(g) / wp.float32(255.0)
    bf = wp.float32(b) / wp.float32(255.0)
    grey = wp.float32(0.299) * rf + wp.float32(0.587) * gf + wp.float32(0.114) * bf

    log_intensity = wp.log(grey + wp.float32(1e-6)) # add small value to avoid log(0)

    return log_intensity

@wp.kernel
def _event_kernel(
    rgb: wp.array(dtype=wp.uint8, ndim=4),
    motion_vectors: wp.array(dtype=wp.float32, ndim=4),
    env_indices: wp.array(dtype=wp.int32, ndim=1),
    threshold_pos: float,
    threshold_neg: float,
    height: int,
    width: int,
    events_out: wp.array(dtype=wp.int32, ndim=4),
):
    """Warp kernel to compute event camera output given the current RGB frame and motion vectors."""
    n, y, x = wp.tid()
    env_idx = env_indices[n]
       
    # check for out of bounds
    if y == 0 or y >= height -1 or x == 0 or x >= width - 1:
        return
          
    # compute log intensity gradients using central difference
    log_r = _rgb_to_log_intensity(rgb[env_idx, y, x+1, 0], rgb[env_idx, y, x+1, 1], rgb[env_idx, y, x+1, 2])
    log_l = _rgb_to_log_intensity(rgb[env_idx, y, x-1, 0], rgb[env_idx, y, x-1, 1], rgb[env_idx, y, x-1, 2])
    log_u = _rgb_to_log_intensity(rgb[env_idx, y+1, x, 0], rgb[env_idx, y+1, x, 1], rgb[env_idx, y+1, x, 2])
    log_d = _rgb_to_log_intensity(rgb[env_idx, y-1, x, 0], rgb[env_idx, y-1, x, 1], rgb[env_idx, y-1, x, 2])

    # compute gradients
    grad_x = (log_r - log_l) * wp.float32(0.5)
    grad_y = (log_u - log_d) * wp.float32(0.5)
    
    flow_x = motion_vectors[env_idx, y, x, 0]
    flow_y = motion_vectors[env_idx, y, x, 1]

    delta_L = - (grad_x * flow_x + grad_y * flow_y)

    ne_pos = wp.int32(wp.floor(delta_L / threshold_pos))
    ne_neg = wp.int32(wp.floor(delta_L / threshold_neg))

    events_out[env_idx, y, x, 0] = wp.int32(wp.max(ne_pos, 0)) # if ne_pos is positive, we have that many positive events, otherwise 0
    events_out[env_idx, y, x, 1] = wp.int32(wp.max(-ne_neg, 0)) # if ne_neg is negative, we have that many negative events, otherwise 0

class EventCamera(SensorBase):
    """Event camera sensor implementation for Isaac Lab."""

    cfg: EventCameraCfg

    def __init__(self, cfg: EventCameraCfg):
        """Initializes the event camera sensor.
        
        Args:
            cfg: The configuration for the event camera sensor.
        """

        self._data: EventCameraData = EventCameraData()

        self._camera: Camera | None = None

        # super init parses config and handles USD prim spawning
        super().__init__(cfg)

    def __del__(self):
        """Unsubscribes from callbacks and detach from the replicator registry."""
        if self._camera is not None:
            self._camera.__del__()
        # unsubscribe from callbacks
        super().__del__()  

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return(
            f"EventCamera @ '{self.cfg.prim_path}': \n"
            f"\timage resolution : {self.cfg.width} x {self.cfg.height}\n"
            f"\tpositive threshold : {self.cfg.pos_threshold}\n"
            f"\tnegative threshold : {self.cfg.neg_threshold}\n"
            f"\tupdate period (s): {self.cfg.update_period}\n"
            f"\tnumber of sensors: {self.num_instances}\n"
        )
    
    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._camera.num_instances if self._camera else 0
    
    
    @property
    def data(self) -> EventCameraData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data
    
    @property
    def image_shape(self) -> tuple[int, int]:
        """Returns: A tuple containing the height and width of the event camera frame."""
        return (self.cfg.height, self.cfg.width)
    
    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the sensor for the specified environment ids.

        Args:
            env_ids: The environment ids to reset.
        """
        if not self._is_initialized:
            raise RuntimeError(
                "Event camera could not be initialized. Please ensure --enable_cameras is used to enable rendering."
            )
        
        # reset the timestamps
        super().reset(env_ids)

        # reset camera sensor if enabled
        if self._camera:
            self._camera.reset(env_ids)

    """
    Implementation
    """

    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers for the event camera."""
        super()._initialize_impl()

        camera_cfg = CameraCfg(
            prim_path=self.cfg.prim_path,
            width=self.cfg.width,
            height=self.cfg.height,
            offset=self.cfg.offset,
            update_period=0.0, # hard coded to ensure there is a sync between the inner camera and the event camera.
            update_latest_camera_pose=self.cfg.update_latest_camera_pose,
            spawn=None,
            data_types=["rgb", "motion_vectors"],
        )

        self._camera = Camera(camera_cfg)

        if self._camera is not None and not self._camera.is_initialized:
            self._camera._initialize_impl()
            self._camera._is_initialized = True

        # create internal buffers
        self._create_buffers()
        logger.info(f"Initialized EventCamera with {self.num_instances} sensors.")

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Updates the internal buffers for the event camera given the current RGB frames and motion vectors from the camera sensor."""
        # update camera sensor
        self._camera.update(self._sim_physics_dt)

        # update camera pose
        if self.cfg.update_latest_camera_pose:
            self._data.pos_w[env_ids] = self._camera.data.pos_w[env_ids]
            self._data.quat_w_world[env_ids] = self._camera.data.quat_w_world[env_ids]
            self._data.intrinsic_matrices[env_ids] = self._camera.data.intrinsic_matrices[env_ids]

        rgb = self._camera.data.output["rgb"] # (N, H, W, 3), uint8
        motion_vectors = self._camera.data.output["motion_vectors"] # (N, H, W, 2), float32

        self._data.output["event_frame"][env_ids] = 0 # zero only the active environments before computing new events
        self._compute_events(rgb, motion_vectors, env_ids)

        self._data.image_shape = (self.cfg.height, self.cfg.width)

    """
    Private Helpers
    """

    def _create_buffers(self):
        """Creates the internal buffers for the event camera data."""
        self._data.output = {
            "event_frame": torch.zeros((self.num_instances, self.cfg.height, self.cfg.width, 2), dtype=torch.int32, device=self._device),
            # "raw_events": None, # to be implemented in the future
        }

        self._events_out_wp = wp.from_torch(self._data.output["event_frame"])

        self._data.pos_w = torch.zeros((self.num_instances, 3), dtype=torch.float32, device=self._device)
        self._data.quat_w_world = torch.zeros((self.num_instances, 4), dtype=torch.float32, device=self._device)
        self._data.intrinsic_matrices = torch.zeros((self.num_instances, 3, 3), dtype=torch.float32, device=self._device)
        self._data.image_shape = self.image_shape
        
    def _compute_events(self, rgb: torch.Tensor, motion_vectors: torch.Tensor, env_ids: Sequence[int]):
        """Computes the event camera output given the current RGB frames and motion vectors from the camera sensor."""
        rgb_wp = wp.from_torch(rgb)
        motion_vectors_wp = wp.from_torch(motion_vectors)
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.int32, device=self._device)
        env_ids_wp = wp.from_torch(env_ids_tensor)

        wp.launch(
            _event_kernel,
            dim=(len(env_ids), self.cfg.height, self.cfg.width),
            inputs=[
                rgb_wp,
                motion_vectors_wp,
                env_ids_wp,
                self.cfg.pos_threshold,
                self.cfg.neg_threshold,
                self.cfg.height,
                self.cfg.width,
                self._events_out_wp,
            ]
        )