import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dipy.viz import actor,colormap
from dipy.reconst.shm import CsaOdfModel
from dipy.direction import peaks_from_model
from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.tracking import utils
from fury import window, actor
from matplotlib import cm

class UnifiedTractographyEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data, affine, labels, bvals, bvecs, vox_size=[1.0,1.0,1.0], 
                 step_size=0.5, max_steps=500, fa_threshold=0.25, 
                 max_curvature_deg=45, sh_order=6, target_label=None):
        super().__init__()
        self.data = data
        self.affine = affine
        self.inv_affine = np.linalg.inv(affine)
        self.vox_size = vox_size
        self.labels = labels
        self.volume_shape = data.shape[:3]
        
        self.step_size = step_size
        self.max_steps = max_steps
        self.fa_threshold = fa_threshold
        self.max_curvature_cos = np.cos(np.deg2rad(max_curvature_deg))
        self.target_label=target_label
        if target_label is not None:
            self.target_center = np.mean(np.argwhere(self.labels == target_label), axis=0)

        self._init_diffusion_model(bvals, bvecs, sh_order)

        # Actions are 26 directions, but for a Peak Agent, 
        # we will map peak indices to these vectors.
        self.actions = self._create_action_vectors()
        self.action_space = spaces.Discrete(len(self.actions))
        
        # Obs: [x, y, z, prev_dx, prev_dy, prev_dz, gfa, step_fraction]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        self.streamline = [] # Stored in Voxel Space internally
        self.pos = None
        self.prev_dir = None
        self.steps = 0

    def _init_diffusion_model(self, bvals, bvecs, sh_order):
        gtab = gradient_table(bvals=bvals, bvecs=bvecs)
        model = CsaOdfModel(gtab, sh_order_max=sh_order)
        mask = (self.labels > 0)
        self.peaks = peaks_from_model(
            model, self.data, sphere=default_sphere,
            relative_peak_threshold=0.8, min_separation_angle=45,
            mask=mask, return_sh=False
        )
        self.gfa_map = self.peaks.gfa

    def _get_interpolated_val(self, volume, coords):
        """
        Performs trilinear interpolation on a 3D volume.
        Guaranteed to return a float.
        """
        # 1. Coordinate Extraction & Clamping
        # We clamp to (dim - 1.001) to ensure the 'upper' neighbor (i+1) is always in bounds
        x, y, z = coords
        max_x, max_y, max_z = np.array(volume.shape) - 1.001
        
        x = np.clip(x, 0, max_x)
        y = np.clip(y, 0, max_y)
        z = np.clip(z, 0, max_z)

        # 2. Identify the 8 neighboring voxel indices (the 'cube' corners)
        x0, y0, z0 = np.floor([x, y, z]).astype(int)
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

        # 3. Calculate weights (distance from the 'lower' corner)
        xd, yd, zd = x - x0, y - y0, z - z0

        # 4. Extract values from the 8 corners
        # Using try-except or .get logic isn't needed due to clamping above
        c000 = volume[x0, y0, z0]
        c100 = volume[x1, y0, z0]
        c010 = volume[x0, y1, z0]
        c110 = volume[x1, y1, z0]
        c001 = volume[x0, y0, z1]
        c101 = volume[x1, y0, z1]
        c011 = volume[x0, y1, z1]
        c111 = volume[x1, y1, z1]

        # 5. Interpolate along X
        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        # 6. Interpolate along Y
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        # 7. Interpolate along Z
        result = c0 * (1 - zd) + c1 * zd

        # 8. Final Safety Check
        # Returns 0.0 if the result is NaN or Inf (rare with clamping, but good for RL)
        if not np.isfinite(result):
            return 0.0
            
        return float(result)

    def _get_interpolated_vec(self, volume, coords):
        """
        Interpolates a 3D vector field at a continuous point.
        volume: Shape (X, Y, Z, 3) e.g., self.peaks.peak_dirs[:,:,:,0,:]
        """
        x, y, z = coords
        max_x, max_y, max_z = np.array(volume.shape[:3]) - 1.001
        
        # 1. Clamp and Floor
        x = np.clip(x, 0, max_x)
        y = np.clip(y, 0, max_y)
        z = np.clip(z, 0, max_z)
        
        x0, y0, z0 = np.floor([x, y, z]).astype(int)
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        xd, yd, zd = x - x0, y - y0, z - z0

        # 2. Get 8 Neighbors
        neighbors = [
            volume[x0, y0, z0], volume[x1, y0, z0],
            volume[x0, y1, z0], volume[x1, y1, z0],
            volume[x0, y0, z1], volume[x1, y0, z1],
            volume[x0, y1, z1], volume[x1, y1, z1]
        ]

        # 3. CRITICAL: Axial Alignment (Hemisphere matching)
        # Match all neighbors to the first one (base_vec)
        base_vec = neighbors[0]
        aligned_neighbors = []
        for n in neighbors:
            if np.dot(n, base_vec) < 0:
                aligned_neighbors.append(-n)
            else:
                aligned_neighbors.append(n)

        # 4. Trilinear Blend
        # Interpolate along X
        c00 = aligned_neighbors[0]*(1-xd) + aligned_neighbors[1]*xd
        c01 = aligned_neighbors[4]*(1-xd) + aligned_neighbors[5]*xd
        c10 = aligned_neighbors[2]*(1-xd) + aligned_neighbors[3]*xd
        c11 = aligned_neighbors[6]*(1-xd) + aligned_neighbors[7]*xd

        # Interpolate along Y and Z
        c0 = c00*(1-yd) + c10*yd
        c1 = c01*(1-yd) + c11*yd
        res = c0*(1-zd) + c1*zd

        # 5. Normalize (Interpolation shrinks vectors)
        norm = np.linalg.norm(res)
        return res / norm if norm > 0 else res
   
    def _create_action_vectors(self):
        indices = [-1, 0, 1]
        vectors = [[dx, dy, dz] for dx in indices for dy in indices for dz in indices 
                   if not (dx == dy == dz == 0)]
        return np.array([v / np.linalg.norm(v) for v in vectors], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options and 'seed_world' in options:
            world_pos = np.array(options['seed_world'])
            self.pos = (np.append(world_pos, 1) @ self.inv_affine.T)[:3]
        else:
            seed_indices = np.argwhere(self.labels == 2)
            idx = self.np_random.integers(len(seed_indices))
            self.pos = seed_indices[idx].astype(np.float32)

        self.streamline = [self.pos.copy()]
        self.prev_dir = np.zeros((4, 3), dtype=np.float32)
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        
        gfa = self._get_interpolated_val(self.gfa_map, self.pos)
        return np.concatenate([self.pos, self.prev_dir.flatten(), [gfa], [self.steps/self.max_steps]]).astype(np.float32)

    def step(self, action_idx):
        self.steps += 1
        move_dir = self.actions[action_idx]

        if np.dot(move_dir, self.prev_dir[0]) < 0:
            move_dir = -move_dir

        new_pos = self.pos + move_dir * self.step_size
        valid, reason = self._is_valid_move(new_pos, move_dir)

        if not valid:
            return self._get_obs(), -1.0, True, False, {"reason": reason}

        self.prev_dir[1:] = self.prev_dir[:-1]
        self.prev_dir[0] = move_dir
        self.pos = new_pos
        self.streamline.append(new_pos.copy())

        return self._get_obs(), 0.1, False, (self.steps >= self.max_steps), {}

    def manual_step(self, move_dir):
        self.steps += 1
        move_dir = move_dir / np.linalg.norm(move_dir)

        new_pos = self.pos + move_dir * self.step_size
        valid, reason = self._is_valid_move(new_pos, move_dir)

        if not valid:
            return self._get_obs(), -1.0, True, False, {"reason": reason}

        self.prev_dir[1:] = self.prev_dir[:-1]
        self.prev_dir[0] = move_dir
        self.pos = new_pos
        self.streamline.append(new_pos.copy())

        return self._get_obs(), 0.1, False, (self.steps >= self.max_steps), {}
    
    def manual_step_rk2(self, move_dir):
        """
        Standard RK2 (Midpoint) integration for the agent's movement.
        """
        self.steps += 1
        
        # --- k1: Current Position Slope ---
        v1 = move_dir / np.linalg.norm(move_dir)
        
        # --- Midpoint Prediction ---
        mid_pos = self.pos + (v1 * self.step_size * 0.5)
        
        # --- k2: Slope at Midpoint ---
        # We look up the smooth interpolated vector at the midpoint
        v2 = self._get_interpolated_vec(self.peaks.peak_dirs[:,:,:,0,:], mid_pos)
        
        # Ensure v2 points in the same general direction as v1 (no U-turns)
        if np.dot(v1, v2) < 0:
            v2 = -v2
            
        # --- Final RK2 Update ---
        # The actual step is taken using the midpoint's direction
        new_pos = self.pos + v2 * self.step_size
        
        # Validation and State Update
        valid, reason = self._is_valid_move(new_pos, v2)
        if not valid:
            return self._get_obs(), -1.0, True, False, {"reason": reason}
            
        self.pos = new_pos
        self.prev_dir[1:] = self.prev_dir[:-1]
        self.prev_dir[0] = v2
        self.streamline.append(new_pos.copy())

        return self._get_obs(), 0.1, False, (self.steps >= self.max_steps), {}
        
    def manual_step_rk4(self, move_dir):
        """
        Standard RK4 (classical Runge–Kutta) integration for the agent's movement.
        """
        self.steps += 1

        # --- k1: slope at current position ---
        v1 = move_dir / np.linalg.norm(move_dir)

        # --- k2: slope at midpoint using v1 ---
        mid_pos1 = self.pos + v1 * (self.step_size * 0.5)
        v2 = self._get_interpolated_vec(self.peaks.peak_dirs[:,:,:,0,:], mid_pos1)
        if np.dot(v1, v2) < 0:  # avoid U-turn
            v2 = -v2

        # --- k3: slope at midpoint using v2 ---
        mid_pos2 = self.pos + v2 * (self.step_size * 0.5)
        v3 = self._get_interpolated_vec(self.peaks.peak_dirs[:,:,:,0,:], mid_pos2)
        if np.dot(v2, v3) < 0:
            v3 = -v3

        # --- k4: slope at endpoint using v3 ---
        end_pos = self.pos + v3 * self.step_size
        v4 = self._get_interpolated_vec(self.peaks.peak_dirs[:,:,:,0,:], end_pos)
        if np.dot(v3, v4) < 0:
            v4 = -v4

        # --- Weighted RK4 combination ---
        v_final = (v1 + 2*v2 + 2*v3 + v4) / 6.0
        v_final /= np.linalg.norm(v_final)  # normalize

        new_pos = self.pos + v_final * self.step_size

        # --- Validation and State Update ---
        valid, reason = self._is_valid_move(new_pos, v_final)
        if not valid:
            return self._get_obs(), -1.0, True, False, {"reason": reason}

        self.pos = new_pos
        self.prev_dir[1:] = self.prev_dir[:-1]
        self.prev_dir[0] = v_final
        self.streamline.append(new_pos.copy())

        return self._get_obs(), 0.1, False, (self.steps >= self.max_steps), {}

    def _compute_reward(self, old_pos, new_pos, move_dir):
        v_idx = tuple(np.floor(new_pos).astype(int))
        
        # 1. Survival Bonus (encourages longer, valid streamlines)
        reward = 0.05 
        
        # 2. GFA Reward (stay in high-signal areas)
        gfa_val = self._get_interpolated_val(self.gfa_map, new_pos)
        reward += 0.2 * gfa_val
        
        # 3. Alignment Reward (Peak following)
        p_dirs = self.peaks.peak_dirs[v_idx]
        p_vals = self.peaks.peak_values[v_idx]
        if len(p_vals) > 0:
            # Align with the peak that is most similar to current movement
            dots = np.abs(np.dot(p_dirs, move_dir))
            best_alignment = np.max(dots)
            reward += 0.5 * best_alignment
        
        # 4. Continuity Penalty (punish sharp turns)
        if np.linalg.norm(self.prev_dir) > 0:
            cos_sim = np.dot(self.prev_dir[0], move_dir)
            # If cos_sim is low (large angle), reward decreases
            reward -= 0.1 * (1 - cos_sim)
            
        # 5. THE GLOBAL GOAL (Target ROI)
        # Check if the new position is inside the Target Label (fMRI hub)
        if self.target_label is not None:
            if self.labels[v_idx] == self.target_label:
                reward += 50.0  # Massive "Win" bonus
                self.found_target = True # Signal for termination
                
            # 6. Distance-to-Goal Shaping (The "Compass")
            # This guides the agent toward the target even if there's no local peak
            if hasattr(self, 'target_coords'):
                dist_old = np.linalg.norm(old_pos - self.target_coords)
                dist_new = np.linalg.norm(new_pos - self.target_coords)
                reward += (dist_old - dist_new) * 2.0  # Positive reward for getting closer
            
        return reward

    def _is_valid_move(self, new_pos, move_dir):
        # 1. Boundary Check
        if any(new_pos < 0) or any(new_pos >= np.array(self.volume_shape) - 1):
            return False, "out_of_bounds"
            
        v_idx = tuple(np.floor(new_pos).astype(int))
        
        # 2. GFA/FA Threshold (Stopping Criterion)
        if self.gfa_map[v_idx] < self.fa_threshold:
            return False, "low_gfa"
            
        # 3. Curvature Check (Angle > 45 deg)
        if np.linalg.norm(self.prev_dir) > 0:
            cos_sim = np.dot(self.prev_dir[0], move_dir)
            if cos_sim < self.max_curvature_cos:
                return False, "high_curvature"
                
        return True, None

    def get_world_streamline(self):
        if not self.streamline: return np.array([])
        pts = np.array(self.streamline)
        return (np.c_[pts, np.ones(len(pts))] @ self.affine.T)[:, :3]
    # --------------------------------------------------
    # Visualization & Rendering (DIPY/FURY Integration)
    # --------------------------------------------------

    def render_wm_surface(self, scene, color=(1, 1, 1), opacity=0.2):
        """Adds a 3D contour of the White Matter volume to the scene."""
        wm_mask = (self.labels == 1) | (self.labels == 2)
        wm_surface = actor.contour_from_roi(
            data=wm_mask,
            affine=self.affine,
            color=color,
            opacity=opacity
        )
        scene.add(wm_surface)
        return scene

    def render_seeds_mask(self, scene, color=(1, 0, 0), point_radius=0.15):
        """Visualizes all potential starting points (label 2)."""
        seed_mask=(self.labels == 2)
        
        seeds= utils.seeds_from_mask(seed_mask, self.affine, density=self.vox_size)
        
        seed_points = actor.point(
            points=seeds,
            colors=color,
            point_radius=point_radius
        )
        scene.add(seed_points)
        return scene

    def render_current_streamline(self, scene, color=(0, 1, 1), linewidth=2):
        """Renders the path taken by the agent in the current episode."""
        world_path = self.get_world_streamline()
        if len(world_path) < 2:
            return scene
        
        streamline_actor = actor.line(
            [world_path],
            colors=color,
            linewidth=linewidth
        )
        scene.add(streamline_actor)
        return scene
    
    def render_bval_bvec(self, scene, mask):
        coords=np.argwhere(mask)
        hom_c=np.c_[coords,np.ones(len(coords))]
        world_c= hom_c @ self.affine.T
        points =world_c[:,:3]
        csa_peaks=self.peaks
        peak_dirs = csa_peaks.peak_dirs[coords[:,0], coords[:,1], coords[:,2]]  # (N,5,3)
        peak_vals = csa_peaks.peak_values[coords[:,0], coords[:,1], coords[:,2]]  # (N,5)
        lines = []
        colors = []

        for p, dirs, vals in zip(points, peak_dirs, peak_vals):
            for d, v in zip(dirs, vals):
                if v > 0:  # valid peak
                    start = p - d / 2   # scale for visibility
                    end   = p + d / 2
                    lines.append([start, end])
                    c = cm.viridis(v)[:3]  # RGB from colormap
                    colors.append(c)
        line_actor = actor.line(lines, colors=colors)
        scene.add(line_actor)
        return scene
    
    def render_streamlines(self,scene,streamlines):
        streamlines_actor= actor.line(
            streamlines, colors=colormap.line_colors(streamlines)
        )
        scene.add(streamlines_actor)
        return scene


