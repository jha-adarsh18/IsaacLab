from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="This script launches the app to read data for WarpEventCamera in IsaacLab")
parser.add_argument("--z_value", type=float, default=10.0, help="Z value for the cube")
parser.add_argument("--steps", type=int, default=1000, help="Number of steps to run the app")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg

N = 64   # batch size
H = 64  # image height
W = 64  # image width

C_THRESHOLD = 0.1     # log-intensity threshold for event firing
DEVICE_TORCH = "cuda" if torch.cuda.is_available() else "cpu"    # wp.device string
DEVICE_WARP  = "cuda" if torch.cuda.is_available() else "cpu"    # wp.device string

@wp.kernel
def event_kernel(
    log_intensity:  wp.array(dtype=wp.float32, ndim=3),
    motion_vectors: wp.array(dtype=wp.float32, ndim=4),
    threshold:      float,
    height:         int,
    width:          int,
    events_out:     wp.array(dtype=wp.int32, ndim=4),
):
    n, y, x = wp.tid()

    if y == 0 or y >= height - 1 or x == 0 or x >= width - 1:
        return # Out of bounds check
    
    # Compute spatial gradients using central differences
    grad_x = (log_intensity[n, y, x + 1] - log_intensity[n, y, x - 1]) * 0.5
    grad_y = (log_intensity[n, y + 1, x] - log_intensity[n, y - 1, x]) * 0.5
    flow_x = motion_vectors[n , y, x, 0]
    flow_y = motion_vectors[n, y, x, 1]
    delta_L = - (grad_x * flow_x + grad_y * flow_y)
    ne = int(wp.floor(delta_L / threshold))
    events_out[n, y, x, 0] = wp.int32(wp.max(ne, 0))
    events_out[n, y, x, 1] = wp.int32(wp.max(-ne, 0))

def rgb_to_log_intensity(rgb: torch.Tensor) -> torch.Tensor:
    # input:  (N, H, W, 3) uint8
    # output: (N, H, W)    float32
    # steps: cast → normalize → grayscale → log
    rgb_float = rgb.float() / 255.0
    grayscale = 0.299 * rgb_float[..., 0] + 0.587 * rgb_float[..., 1] + 0.114 * rgb_float[..., 2]
    log_intensity = torch.log(grayscale + 1e-6)  # add epsilon to avoid log(0)
    return log_intensity

def compute_events(log_intensity: torch.Tensor, motion_vectors: torch.Tensor, events_out: torch.Tensor) -> torch.Tensor:
    events_out.zero_()

    log_intensity_wp = wp.from_torch(log_intensity)
    motion_vectors_wp = wp.from_torch(motion_vectors)
    events_out_wp = wp.from_torch(events_out)

    wp.launch(event_kernel, dim=(N, H, W), inputs=[log_intensity_wp, motion_vectors_wp, C_THRESHOLD, H, W, events_out_wp])

    return wp.to_torch(events_out_wp)

@configclass
class CubeSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg()
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1), roughness=0.2, metallic=0.5)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 10.0), rot=(1.0, 0.0, 0.0, 0.0))
    )

    camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/camera",
        offset=CameraCfg.OffsetCfg(pos=(3.0, 0.0, 5.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=PinholeCameraCfg(),
        depth_clipping_behavior="none",
        width=64,
        height=64,
        data_types=["rgb", "motion_vectors"]
    )

@configclass
class CubeEnvCfg(DirectRLEnvCfg):
    """Configuration for the CubeEnv environment."""

    # Define any additional configuration parameters here if needed
    decimation = 2
    episode_length_s: float = 10.0
    action_space: int = 0
    observation_space: int = 0
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(dt = 1 / 120, render_interval = decimation)

    scene: CubeSceneCfg = CubeSceneCfg(num_envs=64, env_spacing=4.0)

class CubeEnv(DirectRLEnv):
    cfg: CubeEnvCfg

    def __init__(self, cfg: CubeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
    # Dummy returns for abstract methods
    def _pre_physics_step(self, actions):
        pass

    def _apply_action(self):
        pass

    def _get_observations(self):
        return {}
        
    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)
        
    def _get_dones(self):
        return (
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
    

def main(args_cli):
    env_cfg = CubeEnvCfg()
    env_cfg.scene.num_envs = 64
    env = CubeEnv(env_cfg)

    events_out = torch.zeros((N, H, W, 2), dtype=torch.int32, device=DEVICE_TORCH)

    for steps in range(args_cli.steps):
        obs, reward, terminated, timeout, info = env.step(torch.zeros(env.num_envs, device=env.device))

        rgb = env.scene.sensors["camera"].data.output["rgb"]
        motion_vectors = env.scene.sensors["camera"].data.output["motion_vectors"]

        log_intensity = rgb_to_log_intensity(rgb)
        events = compute_events(log_intensity, motion_vectors, events_out)

        print(f"Step {steps}: rgb shape {rgb.shape}, motion_vectors shape {motion_vectors.shape}")
        print(f"Output events shape: {events.shape}, dtype: {events.dtype}")
        print(f"Positive events: {events[..., 0].sum().item()}")
        print(f"Negative events: {events[..., 1].sum().item()}")

    env.close()

if __name__ == "__main__":
    wp.init()

    main(args_cli)

    simulation_app.close()