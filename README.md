# N-Body Gravitational Simulation

A high-performance gravitational N-body simulation.

![N-Body Simulation](demo/nbody_animation.mp4)

## Project Overview

This project simulates gravitational interactions between celestial bodies using Newton's law of gravitation. It demonstrates the transition from CPU-based computation to GPU-accelerated parallel processing, achieving **100-1000x performance improvements**.

### What It Does

The simulation calculates gravitational forces between N particles in 3D space, where each particle attracts every other particle. The system uses:
- **Direct N-body summation** (O(N**2) algorithm)
- **Euler integration** for position/velocity updates
- **Real-time visualization** of particle dynamics

### Features

- Python/NumPy CPU implementation (baseline)
- Real-time 2D visualization with matplotlib
- Energy conservation tracking
- Video export (MP4/GIF)
- Configurable particle counts (50-1000+)
- **Coming soon:** CUDA GPU acceleration

## Performance Metrics

Current CPU baseline performance:

| Particles | Time/Step | Force Calculations |
|-----------|-----------|--------------------|
| 50        | ~5 ms     | 2,500              |
| 100       | ~20 ms    | 10,000             |
| 200       | ~80 ms    | 40,000             |
| 500       | ~500 ms   | 250,000            |

**Target with CUDA:** Sub-millisecond performance for 1000+ particles

## Demo

[**Watch the simulation in action (MP4)**](demo/nbody_animation.mp4)

The visualization shows:
- 50 particles colored by mass (lighter = heavier)
- Gravitational clustering and orbital patterns_
- Real-time physics over 1200 timesteps

## Start

### Prerequisites

```bash
# RHEL 9.5
sudo dnf install python3 python3-tkinter ffmpeg

# Install Python dependencies
pip3 install numpy matplotlib pillow
```

### Running the Simulation

```bash
cd src
python3 nbody_python.py
```

Choose visualization option:
1. **Show animation** - Live window display
2. **Save to MP4** - Export video file
3. **Save to GIF** - Export animated GIF
4. **Both** - Show and save

## Project Structure

```
n-body-simulation/
├── README.md                # Main documentation
├── src/
│   └── nbody_python.py      # Main simulation script
├── demo/
│   ├── nbody_animation.mp4  # Demo video
│   └── nbody_preview.gif  # Preview GIF
└── docs/
    └── DESCRIPTION.md       # Detailed description
```

## Technical Details

### Algorithm

The simulation uses the **direct summation method**:

```
For each particle i:
    For each particle j (where j ≠ i):
        Calculate distance: r = ||pos_j - pos_i||
        Calculate force: F = G * m_i * m_j / r²
        Update acceleration: a_i += F / m_i
    Update velocity: v_i += a_i * dt
    Update position: pos_i += v_i * dt
```

**Complexity:** O(N²) per timestep
- **Perfect for GPU parallelization**
- Each force calculation is independent
- Massively parallel workload ideal for CUDA

### Physics Parameters

- **Gravitational constant (G):** 1.0 (scaled units)
- **Softening parameter (ε):** 0.01 (prevents singularities)
- **Time step (dt):** 0.01
- **Integration method:** Euler (simple, fast)

## Learning Objectives

1. **Scientific Computing**
   - N-body physics simulation
   - Numerical integration methods
   - Energy conservation analysis

2. **Performance Optimization**
   - Algorithm complexity analysis (O(N²))
   - Baseline CPU performance measurement
   - Preparation for GPU acceleration

3. **Software Engineering**
   - Documented code
   - Modular design (class-based)
   - Visualization

4. **GPU Programming** (upcoming)
   - CUDA kernel development
   - Parallel algorithm design
   - Performance benchmarking

## Technologies Used

- **Python 3.x** - programming language
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization and animation
- **FFmpeg** - Video encoding
- **TkAgg** - GUI backend for X11

## Next Steps

- [ ] Implement CUDA GPU acceleration with Numba
- [ ] Add Barnes-Hut algorithm (O(N log N))
- [ ] 3D visualization with Plotly
- [ ] Interactive parameter controls
- [ ] Performance comparison dashboard

## Author

**Andrey Maltsev**

This project is part of my study GPU programming and scientific computing skills.

## License

This project is open source and available for educational purposes.

## Contributing

Suggestions and improvements are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

---

**⭐ If you find this project interesting, please star the repository!**
