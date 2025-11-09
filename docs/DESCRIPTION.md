# N-Body Gravitational Simulation - Technical Description

## Executive Summary

This project uses a gravitational N-body simulation to show how celestial bodies move when they are attracted to each other by gravity. It is the basis for learning how to program GPUs and optimize parallel computing, and it will eventually use CUDA to speed things up by 100 to 1000 times.

## Scientific Background

### The N-Body Problem

The N-body problem is a classical challenge in physics and computational science: given N particles with known positions, velocities, and masses, predict their motion under mutual gravitational forces. This problem has applications in:

- **Astrophysics**: Galaxy formation, stellar clusters, planetary systems
- **Molecular dynamics**: Protein folding, chemical reactions
- **Cosmology**: Dark matter distribution, universe evolution

### Gravitational Physics

The simulation implements Newton's law of universal gravitation:

```
F = G * (m₁ * m₂) / r**2
```

Where:
- `F` = Gravitational force between two bodies
- `G` = Gravitational constant (1.0 in scaled units)
- `m₁, m₂` = Masses of the two bodies
- `r` = Distance between centers of mass

## Algorithm Implementation

### Direct Summation Method

The code uses the **brute-force O(N**2) algorithm**, which calculates every pairwise interaction:

**Pseudocode:**
```
for each particle i from 1 to N:
    force[i] = 0
    for each particle j from 1 to N:
        if i ≠ j:
            r_vec = position[j] - position[i]
            r = ||r_vec|| + ε  // softening parameter
            F_mag = G * mass[i] * mass[j] / r**2
            F_dir = r_vec / ||r_vec||
            force[i] += F_mag * F_dir
```

**Time Complexity:** O(N**2) per timestep
- 50 particles: 2,500 force calculations
- 100 particles: 10,000 force calculations
- 1000 particles: 1,000,000 force calculations

### Numerical Integration

The simulation uses **Euler's method** for time integration:

```
acceleration[i] = force[i] / mass[i]
velocity[i] = velocity[i] + acceleration[i] * dt
position[i] = position[i] + velocity[i] * dt
```

**Properties:**
- Simple, fast, easy to implement
- First-order accuracy: O(dt)
- Energy drift over long simulations

### Softening Parameter

To prevent numerical instabilities when particles get very close:

```
r_effective = sqrt(r**2 + ε**2)
```

Where `ε = 0.01` is the softening length. This prevents:
- Division by zero
- Infinite forces
- Numerical overflow

## Performance Analysis

### Target Hardware: NVIDIA Tesla P100

**GPU Specifications:**
- **CUDA Cores:** 3584
- **Memory:** 16GB HBM2
- **Memory Bandwidth:** 732 GB/s
- **FP32 Performance:** 9.3 TFLOPS
- **FP64 Performance:** 4.7 TFLOPS
- **Architecture:** Pascal (GP100)

The Tesla P100 is a professional-grade datacenter GPU designed for HPC and scientific computing workloads, making it ideal for N-body simulations.

### CPU Baseline (NumPy)

Current performance on standard CPU:

| Particles (N) | Operations per step | Time per step | Throughput |
|--------------|---------------------|---------------|------------|
| 50           | 2,500               | ~5 ms         | 500K ops/s |
| 100          | 10,000              | ~20 ms        | 500K ops/s |
| 200          | 40,000              | ~80 ms        | 500K ops/s |
| 500          | 250,000             | ~500 ms       | 500K ops/s |

**Observation:** Linear scaling with O(N**2), consistent throughput

### GPU Acceleration Potential

**Why this algorithm is perfect for GPU:**

1. **Embarrassingly parallel**: Each force calculation is independent
2. **Uniform computation**: Same operation for all particle pairs
3. **High arithmetic intensity**: Many FLOPs per memory access
4. **Regular memory access**: Predictable patterns

**Expected CUDA performance:**
- Tesla P100 GPU: 3584 CUDA cores, 16GB HBM2 memory
- Peak performance: 4.7 TFLOPS (FP64), 9.3 TFLOPS (FP32)
- Memory bandwidth: 732 GB/s
- Theoretical speedup: 100-1000x over CPU baseline
- Target: 1000 particles in <1ms per step, 10,000+ particles real-time capable

## Visualization

### 2D Projection

The visualization shows the XY plane projection of the 3D simulation:

**Visual Elements:**
- **Particle color**: Mass mapping (plasma colormap)
  - Yellow/white = Heavy particles
  - Purple/dark = Light particles
- **Particle size**: Proportional to mass (s = mass × 20)
- **Background**: Black (space-like)
- **Edges**: White outlines for contrast

### Animation Parameters

- **Frame rate**: 30 FPS
- **Resolution**: 1000×1000 pixels (dpi=100, 10" figure)
- **Duration**: 300 timesteps ≈ 10 seconds at 30 FPS
- **Format**: MP4 (H.264 codec) or GIF (Pillow)

## Code Architecture

### Class Structure

```python
class NBodySimulation:
    # Data arrays (all numpy.float32)
    positions: ndarray[N, 3]    # x, y, z coordinates
    velocities: ndarray[N, 3]   # vx, vy, vz components
    masses: ndarray[N]          # particle masses
    forces: ndarray[N, 3]       # fx, fy, fz components
    
    # Methods
    __init__()              # Initialize particles
    _initialize_particles() # Random sphere distribution
    compute_forces()        # O(N**2) force calculation
    update()                # Euler integration step
    calculate_energy()      # Energy conservation check
    simulate()              # Run N timesteps
```

### Key Design Decisions

1. **NumPy arrays**: Efficient vectorized operations
2. **float32**: Balance precision vs. memory/speed
3. **Modular methods**: Easy to swap algorithms
4. **Energy tracking**: Verify physical correctness

## Energy Conservation

The simulation tracks total energy:

```
E_total = E_kinetic + E_potential

E_kinetic = Σ (½ * m * v**2)
E_potential = Σ Σ (-G * m_i * m_j / r_ij)  for i < j
```

**Expected behavior:**
- Ideal: E_total = constant
- Reality: Small drift due to Euler integration
- Typical: <5% drift over 300 steps

## Initial Conditions

### Particle Distribution

Particles are initialized uniformly in a sphere:

**Spherical coordinates:**
```python
θ = random(0, 2π)           # azimuthal angle
φ = arccos(random(-1, 1))   # polar angle
r = R * random(0, 1)^(1/3)  # radius (uniform in volume)

x = r * sin(φ) * cos(θ)
y = r * sin(φ) * sin(θ)
z = r * cos(φ)
```

**Parameters:**
- Sphere radius: R = 10.0
- Velocity range: [-0.25, 0.25] in each dimension
- Mass range: [0.5, 2.0]

## Future Enhancements

### CUDA Implementation

**Planned optimizations:**
1. **GPU kernels**: Parallel force computation
2. **Shared memory**: Cache particle data
3. **Tiling**: Process in blocks for cache efficiency
4. **Async execution**: Overlap compute and copy

### Advanced Algorithms

**Barnes-Hut (O(N log N)):**
- Octree spatial decomposition
- Far-field approximation
- Better scaling for large N

**Fast Multipole Method (O(N)):**
- Multipole expansions
- Best asymptotic complexity
- More complex implementation

### Enhanced Visualization

- 3D interactive rendering (Plotly)
- Velocity vectors
- Trajectory trails
- Real-time parameter adjustment

## References

### Algorithms
- Aarseth, S. J. (2003). *Gravitational N-Body Simulations*
- Hockney, R. W., & Eastwood, J. W. (1988). *Computer Simulation Using Particles*

### GPU Programming
- Sanders, J., & Kandrot, E. (2010). *CUDA by Example*
- Kirk, D. B., & Hwu, W. W. (2016). *Programming Massively Parallel Processors*

## Performance Benchmarks

### Test System
- OS: RHEL 9.5 (Plow)
- Python: 3.9.21
- NumPy: 1.26.3
- CPU: Xeon CPU E5-2650 v4 @ 2.20GHz

### Metrics Collected
1. **Time per timestep**: Total wall-clock time
2. **Force calculations**: N × (N-1) per step
3. **Energy conservation**: Δ(E_total) / E_initial
4. **Memory usage**: O(N) arrays

---

*This document provides the technical foundation for understanding the simulation. For implementation details, see the source code with inline comments.*
