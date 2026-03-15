from isaaclab.apps import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="This script launches the app to read data for WarpEventCamera in IsaacLab")
parser.add_argument("--z_value", type=float, default=10.0, help="Z value for the cube")
parser.add_argument("--steps", type=int, default=1000, help="Number of steps to run the app")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.sim import PinholeCameraCfg


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

    cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1), roughness=0.2, metallic=0.5)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 10.0), rot=(1.0, 0.0, 0.0, 0.0))
    )

    camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/camera",
        offset=CameraCfg.OffsetCfg(pos=(3.0, 0.0, 5.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=PinholeCameraCfg(),
        depth_clipping_behavior="none",
        width=64,
        height=64,
        data_types=["rgb", "motion_vectors"]
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs = 64, env_spacing = 4.0, replicate_physics = True, clone_in_fabric = True
    )

class CubeEnv(DirectRLEnv):
    cfg: CubeEnvCfg

    def __init__(self, cfg: CubeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        # spawn ground plane

        self.cube = RigidObject(self.cfg.cube_cfg)
        self.scene.rigid_objects["cube"] = self.cube

        self.camera = Camera(self.cfg.camera_cfg)
        self.scene.sensors["camera"] = self.camera

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/light", light_cfg)
        
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

    for steps in range(args_cli.steps):
        obs, reward, terminated, timeout, info = env.step(torch.zeros(env.num_envs, device=env.device))

        rgb = env.camera.data.output["rgb"]
        motion_vectors = env.camera.data.output["motion_vectors"]

        print(f"Step {steps}: rgb shape {rgb.shape}, motion_vectors shape {motion_vectors.shape}")

    env.close()

if __name__ == "__main__":
    main(args_cli)

    simulation_app.close()