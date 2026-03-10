# RL-Inspired Brain Tractography

A novel framework combining reinforcement learning with diffusion MRI tractography to produce anatomically superior white matter reconstructions.

## 🏆 Key Results: Final Agent vs Traditional EuDX

| Metric | EuDX Tracker | **Our Final Agent** | **Improvement** | Interpretation |
|--------|--------------|---------------------|-----------------|----------------|
| **Streamline Count** | 4,536 | **2,252** | **-50%** | Better selectivity, fewer false positives |
| **Mean Length (mm)** | 57.9 | **40.4** | **-30%** | More anatomically plausible lengths |
| **Max Length (mm)** | 307.5 | **134.4** | **-56%** | Eliminates biologically impossible connections |
| **Mean Tortuosity** | 2.32 | **1.45** | **-38%** | **Much closer to real axon tortuosity (~1.1-1.3)** |
| **Mean Max Curvature (°)** | 13.0 | 25.1 | +93% | Allows anatomically justified turns |
| **Mean GFA Support** | 0.426 | **0.521** | **+22%** | **Better white matter specificity** |

## 🎯 Core Innovation

**From Discrete Grid to Continuous Anatomy**: Traditional tractography suffers from "grid-snapping" artifacts due to 26 discrete directions. Our RL agents navigate in **continuous 3D space** with infinite angular precision, producing biologically plausible pathways with **38% lower tortuosity** than standard methods.

## 🌍 The Environment

`UnifiedTractographyEnv` – A Gymnasium-compatible environment for neuroimaging:
- **Continuous 3D Action Space**: Move in any direction, not just 26 neighbors
- **Trilinear Interpolation**: Smooth sampling of diffusion signals between voxels
- **Multi-Objective Rewards**: Combine white matter integrity (GFA), anatomical continuity, and target guidance
- **Multiple Integration Methods**: Euler, RK2, and RK4 for precision control
- **Built-in Visualization**: 3D rendering of streamlines and anatomy

## 🤖 Agent Evolution

We developed a progression of 11 agents, each adding sophistication:

### **Phase 1: Deterministic Foundations**
- `EuDXAgent`: Classic peak follower (26 discrete directions)
- `EnhancedEuDXAgent`: Continuous peak alignment

### **Phase 2: Reward-Driven Exploration**
- `RewardDrivenAgent`: Greedy reward optimization
- `ProbabilisticRewardDrivenAgent`: Adds epsilon-greedy exploration
- `ContinuousRewardAgent`: Spherical sampling in continuous space

### **Phase 3: Advanced Planning**
- `LookAheadContinuousAgent`: Multi-step planning with discounting
- `MemoryAwareLookAheadAgent`: 4-step momentum tracking
- `UltimateUnifiedAgent`: Combines look-ahead with signal adaptation

### **Phase 4: Complete Systems**
- `FinalUnifiedAgent`: Adaptive step sizing (0.5mm → 0.2mm in low-GFA)
- `BranchingStreamlineAgent`: Multi-path tracking for fiber fanning
- **`UnifiedBranchingAgent`**: **Our final agent** – full feature set with controlled branching

## 📈 Design Philosophy: Five Transformations

Our agents evolve along five key dimensions:
1. **Deterministic → Probabilistic** – Adds intelligent exploration
2. **Discrete → Continuous** – Eliminates grid artifacts
3. **Greedy → Look-ahead** – Enables strategic planning
4. **Static → Adaptive** – Responds to local signal quality
5. **Single-path → Branching** – Captures anatomical complexity

## 🔬 Scientific Contribution

Our results demonstrate that **RL-inspired anatomical constraints** produce superior tractography:

### **1. Biological Plausibility**
- **Tortuosity of 1.45** vs EuDX's 2.32 – closely matches real axon structure
- **Conservative length distribution** – avoids false long-range connections
- **Anatomically justified curvature** – allows turns where biology permits

### **2. White Matter Specificity**
- **22% higher GFA support** – better tracking within valid white matter
- **Fewer gray matter violations** – respects tissue boundaries

### **3. Quality over Quantity**
- **Half as many streamlines** but anatomically superior
- **Reduced false positives** – each streamline has higher confidence

## 📊 Performance Spectrum

| Agent Type | Speed | Precision | Best For |
|------------|-------|-----------|----------|
| Deterministic | ⚡ Fast | ⭐⭐ Medium | Quick reconstructions |
| Reward-Driven | ⚡⚡ Medium | ⭐⭐⭐ High | Goal-directed tracking |
| Look-Ahead | ⚡ Slow | ⭐⭐⭐⭐ Very High | Complex navigation |
| **UnifiedBranchingAgent** | ⚡ Very Slow | ⭐⭐⭐⭐⭐ **Extreme** | **Complete anatomical reconstruction** |

## 🚀 Quick Start

```python
import numpy as np
from agents import UnifiedBranchingAgent
from environment import UnifiedTractographyEnv

# 1. Initialize environment with your data
env = UnifiedTractographyEnv(
    data=your_dmri_data,
    affine=your_affine_matrix,
    labels=your_tissue_labels,
    bvals=b_values,
    bvecs=b_vectors,
    vox_size=1.0  # mm
)

# 2. Create our best agent
agent = UnifiedBranchingAgent(env)

# 3. Track from seed points
seeds = np.load('seed_points.npy')  # (N, 3) world coordinates
streamlines = agent.track_all(seeds)

# 4. Visualize
scene = window.Scene()
scene = env.render_wm_surface(scene, opacity=0.2)
scene = env.render_streamlines(scene, streamlines, colors=(0, 1, 1))
window.show(scene, size=(1200, 800))
```

## 🧠 Key Concepts

- **GFA (Generalized Fractional Anisotropy)**: White matter integrity measure guiding exploration
- **Curvature Constraint**: Limits angular changes (typically 20-25°) for biological realism
- **Reward Shaping**: Combines local signal quality with global anatomical goals
- **Frontier Management**: Intelligent queue system for controlled branching
- **Adaptive Step Sizing**: Smaller steps (0.2mm) in low-signal regions for precision

## 📋 Requirements

```txt
gymnasium>=0.29.1
numpy>=1.24.0
dipy>=1.8.0
fury>=0.10.0
scipy>=1.11.0
```

## 📚 Citation

If this work contributes to your research, please cite:

```bibtex
@software{rl_tractography_2025,
  title = {RL-Inspired Brain Tractography: Continuous Navigation in Diffusion MRI},
  author = {Ashray},
  year = {2025},
  url = {https://github.com/Ashray0013/rl-tractography},
  note = {Framework for reinforcement learning-based white matter reconstruction}
}
```

## 🔍 Future Directions

- Integration with deep RL libraries (Stable-Baselines3, RLlib)
- GPU acceleration for real-time tracking
- Multi-modal constraints (T1 anatomy, fMRI connectivity)
- Clinical validation on patient cohorts
- Connectome analysis applications

---

**Status**: Research-ready framework with demonstrated superiority over traditional methods in anatomical plausibility. The final agent produces streamlines with **38% lower tortuosity** and **22% better white matter specificity** than standard EuDX tractography.

**Contributions welcome** for validation metrics, performance optimization, and integration with clinical pipelines.
