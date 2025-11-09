"""
N-Body Gravitational Simulation in Pure Python
By: Andrey Maltsev

This simulation models gravitational interactions between particles.
Start with CPU version, then I'll accelerate with CUDA.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Check if user root or not
if os.getuid() == 0:
    print("\n"+"-"*60 )
    print("ERROR: Don't run GPU as root! Use: su - ansible")
    print("-"*60 + "\n")
    sys.exit(1)
    
# Patch NumPy for scikit-cuda compatibility
#if not hasattr(np, 'typeDict'):
#    np.typeDict = np.sctypeDict
#if not hasattr(np, 'float'):
#    np.float = np.float64
#if not hasattr(np, 'int'):
#    np.int = np.int_
#if not hasattr(np, 'complex'):
#    np.complex = np.complex128

class NBodySimulation:
    """N-body gravitational simulation using direct summation (O(N²))"""

    def __init__(self, n_particles=100, G=1.0, softening=0.01):
        """
        Initialize the simulation
        Args:
           n_particles: Number of particles
           G: Gravitational constant (scaled units)
           softening: Softening parameter to prevent singularities
        """
        self.n = n_particles
        self.G = G
        self.softening = softening

        # Initialize particle data
        # Shape: (n_particles, 3) for x, y, z coordinates
        self.positions = np.zeros((n_particles, 3), dtype = np.float32)
        self.velocities = np.zeros((n_particles, 3), dtype = np.float32)
        self.masses = np.ones(n_particles, dtype = np.float32)
        self.forces = np.zeros((n_particles, 3), dtype = np.float32)

        self._initialize_particles()

    def _initialize_particles(self):
        """Initialize particles in a random sphere distribution"""
        print(f"Initializing {self.n} partticles...")
    
        # Random positions in the sphere (radius = 10)
        for i in range(self.n):
            # Uniform distribution in a sphere using spherical coordinates
            theta = 2 * np.pi * np.random.random()
            phi = np.arccos(2 * np.random.random() - 1)
            r = 10.0 * np.random.random() ** (1/3) # Uniform in volume
    
            self.positions[i, 0] = r * np.sin(phi) * np.cos(theta)
            self.positions[i, 1] = r * np.sin(phi) * np.sin(theta)
            self.positions[i, 2] = r * np.cos(phi)

        # Small random velocities
        self.velocities = (np.random.random((self.n, 3)) - 0.5) * 0.5

        # Random masses between 0.5 and 2.0
        self.masses = 0.5 + 1.5 * np.random.random(self.n)

        print(f"  Particles initialized in sphere")
        print(f"  Position range: [{self.positions.min():.2f}, {self.positions.max():.2f}]")
        print(f"  Velocity range: [{self.velocities.min():.2f}, {self.velocities.max():.2f}]")
        print(f"  Mass range: [{self.masses.min():.2f}, {self.masses.max():.2f}]")

    def compute_forces(self):
        """
        Compute gravitational forces on all particles (O(N²) algorithm)
        
        This is the expensive part that we'll accelerate with CUDA later!
        """

        # Reset forces
        self.forces.fill(0)

        # Compute pairwise forces
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue

                # Vector from particle i to particle j
                r_vec = self.positions[j] - self.positions[i]

                # Distance squared with softening
                r_squared = np.sum(r_vec ** 2) + self.softening ** 2

                # Distance
                r = np.sqrt(r_squared)

                # Gravitational force: F = G * m1 * m2 / r2 * (unit vector)
                force_magnitude = self.G * self.masses[i] * self.masses[j] / r_squared
                force_direction = r_vec / r # Unit vector

                # Add force to particle i
                self.forces[i] += force_magnitude * force_direction

    def update(self, dt=0.01):
        """
        Update particle positions and velocities using Euler integration
        
        Args:
            dt: Time step
        """
        # Compute forces
        self.compute_forces()

        # F = ma, so a = F/m
        accelerations = self.forces / self.masses[:, np.newaxis]

        # Update velocities: v = v + a*dt
        self.velocities += accelerations * dt

        # Update positions: x = x + v*dt
        self.positions += self.velocities * dt

    def calculate_energy(self):
        """Calculate total energy (kinetic + potential)"""
        # Kinetic energy: 0.5 * m * v**2
        kinetic = 0.5 * np.sum(self.masses * np.sum(self.velocities ** 2, axis=1))

        # Potential energy: -G * m1 * m2 / r (sum over all pairs)
        potential = 0.0
        for i in range(self.n):
            for j in range(i+1, self.n):
                r_vec = self.positions[j] - self.positions[i]
                r = np.linalg.norm(r_vec)
                if r > 0:
                    potential -= self.G * self.masses[i] * self.masses[j] / r

        return kinetic + potential

    def simulate(self, n_steps=100, dt=0.01):
        """
        Run simulation for n_steps
        
        Args:
            n_steps: Number of time steps
            dt: Time step size
        """
        print(f"\n{'='*50}")
        print(f"Running simulation: {n_steps} steps")
        print(f"{'='*50}")

        initial_energy = self.calculate_energy()
        print(f"Initial energy: {initial_energy:.2f}")

        times = []
        for step in range(n_steps):
            start_time = time.time()
            self.update(dt)
            step_time = (time.time() - start_time) * 1000 # convert to ms
            times.append(step_time)

            # print progress
            if step % 20 == 0 or step == n_steps - 1:
                energy = self.calculate_energy()
                energy_change = abs((energy - initial_energy) / initial_energy) * 100
                print(f"Step {step:3d}: Energy = {energy:8.2f}"
                      f"(Δ = {energy_change:5.2f}%), Time = {step_time:5.2f} ms")

        avg_time = np.mean(times)
        total_time = np.sum(times)

        print(f"\n{'='*50}")
        print(f"  Simulation complete!")
        print(f"  Total time: {total_time:.2f} ms")
        print(f"  Average time per step: {avg_time:.2f} ms")
        print(f"  Complexity: O(N²) = {self.n}² = {self.n**2} force calculations/step")
        print(f"{'='*50}\n")
        
        return avg_time


def visualize_2d(sim, n_steps=500, dt=0.01):
    """
    Create 2D visualization of the simulation (XY plane projection)
    
    Args:
        sim: NBodySimulation instance
        n_steps: Number of steps to animate
        dt: Time step
    """
    print("Creating visualization...")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.set_title('N-Body Gravitational Simulation', color='white', fontsize=16)
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.tick_params(colors='white')

    # Color particles by mass
    colors = plt.cm.plasma(sim.masses / sim.masses.max())

    # Create scatter plot
    scatter = ax.scatter(sim.positions[:, 0], sim.positions[:, 1],
                        c=colors, s=sim.masses*20, alpha=0.8, edgecolor='white', linewidths=0.5)

    # Text for step counter
    step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       color='white', fontsize=12, verticalalignment='top')

    def animate(frame):
        sim.update(dt)

        # Update positions
        scatter.set_offsets(sim.positions[:, :2]) # XY projection

        # Update step counter
        step_text.set_text(f'Step: {frame}')

        return scatter, step_text
    print(f"Animating {n_steps} steps...")
    anim = FuncAnimation(fig, animate, frames=n_steps, interval=20, blit=True)

    plt.show()

def visualize_2d(sim, n_steps=500, dt=0.01, save_file=None, show=True):
    """
    Create 2D visualization of the simulation (XY plane projection)
    
    Args:
        sim: NBodySimulation instance
        n_steps: Number of steps to animate
        dt: Time step
        save_file: If provided, save animation to this file (.mp4, .gif, .mpeg)
        show: Whether to display the animation window
    """
    print("Creating visualization...")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.set_title('N-Body Gravitational Simulation', color='white', fontsize=16)
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.tick_params(colors='white')

    # Color particles by mass
    colors = plt.cm.plasma(sim.masses / sim.masses.max())

    # Create scatter plot
    scatter = ax.scatter(sim.positions[:, 0], sim.positions[:, 1],
                        c=colors, s=sim.masses*20, alpha=0.8, 
                        edgecolor='white', linewidths=0.5)

    # Text for step counter
    step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       color='white', fontsize=12, verticalalignment='top')

    def animate(frame):
        sim.update(dt)
        scatter.set_offsets(sim.positions[:, :2])
        step_text.set_text(f'Step: {frame}')
        return scatter, step_text

    print(f"Animating {n_steps} steps...")
    anim = FuncAnimation(fig, animate, frames=n_steps, interval=20, blit=True)
    
    if save_file:
        print(f"Saving animation to {save_file}...")
        
        # Determine writer and settings based on file extension
        if save_file.endswith('.gif'):
            # GIF format using Pillow
            anim.save(save_file, writer='pillow', fps=30, dpi=100)
        elif save_file.endswith(('.mp4', '.mpeg', '.avi', '.mov')):
            # Video format using ffmpeg
            # MP4 with H.264 codec for best compatibility
            anim.save(save_file, writer='ffmpeg', fps=30, dpi=100, 
                     bitrate=2000, codec='libx264', 
                     extra_args=['-pix_fmt', 'yuv420p'])
        else:
            print(f"Unknown format, defaulting to MP4...")
            mp4_file = save_file + '.mp4'
            anim.save(mp4_file, writer='ffmpeg', fps=30, dpi=100, 
                     bitrate=2000, codec='libx264',
                     extra_args=['-pix_fmt', 'yuv420p'])
            save_file = mp4_file
        
        print(f" Animation saved to: {save_file}")
    
    if show:
        plt.show()

def main():
    """Main function to run the simulation"""
    print("=" * 60)
    print("N-Body Gravitational Simulation")
    print("=" * 60)
    print("\nThis simulation uses Python + NumPy")
    print("Next step: Accelerate with CUDA for 100x+ speedup!\n")
    
    # Test different sizes
    sizes = [50, 100, 200, 500]
    results = []
    
    for n in sizes:
        print(f"\n{'#'*60}")
        print(f"Testing with {n} particles")
        print(f"{'#'*60}")
        
        sim = NBodySimulation(n_particles=n)
        avg_time = sim.simulate(n_steps=50, dt=0.01)
        results.append((n, avg_time))

    # Print comparison
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    print(f"{'Particles':<15} {'Time/Step (ms)':<20} {'Speedup Needed':<15}")
    print("=" * 60)

    baseline_time = results[0][1]
    for n, t in results:
        speedup_needed = t/baseline_time
        print(f"{n:<15} {t:<20.2f} {speedup_needed:<15.1f}x slower")

    print("\n With CUDA, we can get 100-1000x speedup!")
    print(" -> 500 particles: ~0.5 ms/step instead of current time\n")

    # Visualize smallest system
    # print("\nWould you like to see visualisation? (Close windows to continue)")
    # print("Creating animation for 50 particles...")

    # sim_vis = NBodySimulation(n_particles=50)
    # visualize_2d(sim_vis, n_steps=300, dt=0.01)
    print("\nVisualization options:")
    print("1. Show animation window (live)")
    print("2. Save to MP4 file")
    print("3. Save to GIF file")
    print("4. Show and save to MP4")
    choice = input("Choose (1/2/3/4) or press Enter for option 1: ").strip() or "1"

    sim_vis = NBodySimulation(n_particles=50)
    output_dir = "/home/ansible/gpu-projects/n-body-sumulation/demo"
    
    if choice == "2":
        # Save MP4 only
        output_file = f"{output_dir}/nbody_animation.mp4"
        visualize_2d(sim_vis, n_steps=600, dt=0.01, save_file=output_file, show=False)
        print(f"\n Done! Play with: vlc {output_file}")
    elif choice == "3":
        # Save GIF only
        output_file = f"{output_dir}/nbody_animation.gif"
        visualize_2d(sim_vis, n_steps=600, dt=0.01, save_file=output_file, show=False)
        print(f"\n Done! Open with: eog {output_file}")
    elif choice == "4":
        # Show and save MP4
        output_file = f"{output_dir}/nbody_animation.mp4"
        visualize_2d(sim_vis, n_steps=1200, dt=0.01, save_file=output_file, show=True)
        print(f"\n Done! Animation also saved to: {output_file}")
    else:
        # Show only
        visualize_2d(sim_vis, n_steps=600, dt=0.01, show=True)


if __name__ == "__main__":
    main()