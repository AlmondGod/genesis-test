import argparse
import numpy as np
import genesis as gs
from time import time, sleep

def get_circular_trajectory(t, center, radius, n_points):
    theta = np.linspace(0, 2*np.pi, n_points)
    x = center[0] + radius * np.cos(theta + t)
    y = center[1] + radius * np.sin(theta + t)
    z = center[2] * np.ones_like(theta)
    return np.column_stack([x, y, z])

def run_sim(scene, enable_vis):
    n_particles = 100
    n_steps = 1000
    diffusion_rate = 0.01  # Controls how quickly particles spread apart
    base_noise = 0.05      # Base noise level
    
    t_prev = time()
    for t in range(n_steps):
        # Calculate increasing noise scale
        current_noise_scale = base_noise * (1 + diffusion_rate * t)
        
        # Get positions
        target_pos = get_circular_trajectory(t * 0.05, [0.0, 0.0, 0.5], 0.3, n_particles)
        noise = np.random.normal(0, current_noise_scale, target_pos.shape)
        noise_pos = target_pos + noise
        
        # Update particle positions
        scene.step()
        
        t_now = time()
        print(1 / (t_now - t_prev), "FPS")
        t_prev = t_now
        sleep(0.0005)  # Small delay to control simulation speed

    if enable_vis:
        scene.viewer.stop()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    # Initialize Genesis
    gs.init(backend=gs.cpu)  # Using CPU backend like render_on_macos.py

    n_particles = 100

    # Create scene with appropriate camera settings
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-1.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis
    )

    # Target particles (red)
    target_particles = scene.add_entity(
        material=gs.materials.MPM.Liquid(),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.5),
            size=(0.1, 0.1, 0.1)
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.4, 0.4),
            vis_mode="particle"
        )
    )

    # Noise particles (blue)
    noise_particles = scene.add_entity(
        material=gs.materials.MPM.Liquid(),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.5),
            size=(0.1, 0.1, 0.1)
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 1.0),
            vis_mode="particle"
        )
    )

    scene.build()

    # Run simulation in another thread
    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, args.vis))
    if args.vis:
        scene.viewer.start()

if __name__ == "__main__":
    main()