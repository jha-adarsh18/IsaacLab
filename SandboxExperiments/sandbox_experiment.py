# sandbox_event.py
# Phase 1: Standalone validation of event camera math
# No Isaac Lab imports — only PyTorch and Warp

import torch
import warp as wp

# ──────────────────────────────────────────────
# CONSTANTS — lock your contracts here
# ──────────────────────────────────────────────

N = 64   # batch size
H = 64  # image height
W = 64  # image width

C_THRESHOLD = 0.1     # log-intensity threshold for event firing
DEVICE_TORCH = "cuda" if torch.cuda.is_available() else "cpu"    # wp.device string
DEVICE_WARP  = "cuda" if torch.cuda.is_available() else "cpu"    # wp.device string

# ──────────────────────────────────────────────
# SECTION 1: Synthetic Data Generators
# Mimic exactly what Isaac Lab will hand you
# ──────────────────────────────────────────────

def generate_motion_vectors() -> torch.Tensor:
    # shape: (N, H, W, 2), dtype: float32
    # channel 0 = x-flow, channel 1 = y-flow
    return torch.randn(N, H, W, 2, dtype=torch.float32, device=DEVICE_TORCH) * 5.0

def generate_rgb() -> torch.Tensor:
    # shape: (N, H, W, 3), dtype: uint8
    # raw camera output before any processing
    return torch.randint(0, 256, (N, H, W, 3), dtype=torch.uint8, device=DEVICE_TORCH)

def rgb_to_log_intensity(rgb: torch.Tensor) -> torch.Tensor:
    # input:  (N, H, W, 3) uint8
    # output: (N, H, W)    float32
    # steps: cast → normalize → grayscale → log
    rgb_float = rgb.float() / 255.0
    grayscale = 0.299 * rgb_float[..., 0] + 0.587 * rgb_float[..., 1] + 0.114 * rgb_float[..., 2]
    log_intensity = torch.log(grayscale + 1e-6)  # add epsilon to avoid log(0)
    return log_intensity

# ──────────────────────────────────────────────
# SECTION 2: Warp Kernel
# Only sees wp.array typed inputs — no Torch here
# ──────────────────────────────────────────────

@wp.kernel
def event_kernel(
    log_intensity:  wp.array(dtype=wp.float32, ndim=3),   # fill dtype + ndim
    motion_vectors: wp.array(dtype=wp.float32, ndim=4),   # fill dtype + ndim
    threshold:      float,
    height:         int,
    width:          int,
    events_out:     wp.array(dtype=wp.int32, ndim=4),   # fill dtype + ndim
):
    # 1. get pixel index from wp.tid()
    # 2. compute spatial gradient of log_intensity (finite difference)
    # 3. dot product: grad · flow vector
    # 4. threshold check → accumulate into events_out
    n, y, x = wp.tid()  # unpack thread indices
    if y == 0 or y >= height - 1 or x == 0 or x >= width - 1:
        return  # out of bounds check
    
    # Compute spatial gradients using central differences
    grad_x = (log_intensity[n, y, x + 1] - log_intensity[n, y, x - 1]) * 0.5
    grad_y = (log_intensity[n, y + 1, x] - log_intensity[n, y - 1, x]) * 0.5
    flow_x = motion_vectors[n, y, x, 0]
    flow_y = motion_vectors[n, y, x, 1]
    delta_L = - (grad_x * flow_x + grad_y * flow_y)
    ne = int(wp.floor(delta_L / threshold))
    events_out[n, y, x, 0] = wp.int32(wp.max(ne, 0))
    events_out[n, y, x, 1] = wp.int32(wp.max(-ne, 0))


# ──────────────────────────────────────────────
# SECTION 3: Launch Wrapper
# Mirrors ops.py:74 — Torch in, wp.from_torch bridge, kernel out
# ──────────────────────────────────────────────

def compute_events(
    log_intensity:  torch.Tensor,   # (N, H, W)    float32
    motion_vectors: torch.Tensor,   # (N, H, W, 2) float32
    events_out:     torch.Tensor,   # (N, H, W, 2) int32
) -> torch.Tensor:                  # (N, H, W, 2) int32

    # 1. allocate output as native wp.array (fresh buffer, not from Torch)
    # 2. wrap inputs with wp.from_torch()
    # 3. wp.launch() the kernel
    # 4. return output as torch tensor via wp.to_torch()

    events_out.zero_()  # reset output tensor to zero before launch

    log_intensity_wp = wp.from_torch(log_intensity)
    motion_vectors_wp = wp.from_torch(motion_vectors)
    events_out_wp = wp.from_torch(events_out)

    wp.launch(event_kernel, dim=(N, H, W), inputs=[log_intensity_wp, motion_vectors_wp, C_THRESHOLD, H, W, events_out_wp])

    return wp.to_torch(events_out_wp)

# ──────────────────────────────────────────────
# SECTION 4: Profiling
# ──────────────────────────────────────────────

def profile():
    # 1. generate synthetic data
    # 2. warm-up launch (1 pass, not timed — GPU needs to JIT compile)
    # 3. timed launch with wp.ScopedTimer or CUDA events
    # 4. print shape and timing
    motion_vectors = generate_motion_vectors()
    rgb = generate_rgb()
    log_intensity = rgb_to_log_intensity(rgb)
    events_out = torch.zeros((N, H, W, 2), dtype=torch.int32, device=DEVICE_TORCH)  # pre-allocate output tensor
    compute_events(log_intensity, motion_vectors, events_out)  # warm-up
    with wp.ScopedTimer("Event Kernel", synchronize=True):
        events = compute_events(log_intensity, motion_vectors, events_out)
    print(f"Output events shape: {events.shape}, dtype: {events.dtype}")
    print(f"Positive events: {events[..., 0].sum().item()}")
    print(f"Negative events: {events[..., 1].sum().item()}")

# ──────────────────────────────────────────────

if __name__ == "__main__":
    wp.init()
    profile()