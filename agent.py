import numpy as np

class EuDXAgent:
    """
    Agent that replicates the custom_tracker logic within the RL Environment.
    """
    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        """
        Calculates the best action based on peak alignment.
        """
        curr_pos = obs[0:3]
        prev_dir = obs[3:6]
        
        v_idx = tuple(np.floor(curr_pos).astype(int))
        local_peaks = self.env.peaks.peak_dirs[v_idx] 
        local_values = self.env.peaks.peak_values[v_idx]
        
        if len(local_values) == 0 or np.max(local_values) <= 0:
            return 0 # No signal

        if np.linalg.norm(prev_dir) == 0:
            # First step: use strongest peak
            best_dir = local_peaks[0]
        else:
            # Align with previous direction
            dots = np.dot(local_peaks, prev_dir)
            best_idx = np.argmax(np.abs(dots))
            best_dir = local_peaks[best_idx]
            
            # Orient best_dir to point forward
            if np.dot(best_dir, prev_dir) < 0:
                best_dir = -best_dir

        # Find the index of the environment action closest to best_dir
        # Since env.actions are normalized, dot product is cosine similarity
        similarities = np.dot(self.env.actions, best_dir) # Discrete Action
        return np.argmax(similarities)

    def track_all(self, seeds_world):
        
        all_streamlines = []
        for seed in seeds_world:
            obs, _ = self.env.reset(options={'seed_world': seed})
            done = False
            
            while not done:
                action = self.predict(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
            streamline = self.env.get_world_streamline()
            if len(streamline) > 1:
                all_streamlines.append(streamline)
                
        return all_streamlines

class EnhancedEuDXAgent:
    """
    An upgraded agent that bypasses discrete action limitations 
    to match custom_tracker performance.
    """
    def __init__(self, env):
        self.env = env

    def predict_continuous(self, obs):
        """
        Calculates the exact peak vector instead of a discrete index.
        """
        curr_pos = obs[0:3]
        prev_dir = obs[3:6]
        
        v_idx = tuple(np.floor(curr_pos).astype(int))
        local_peaks = self.env.peaks.peak_dirs[v_idx]
        local_values = self.env.peaks.peak_values[v_idx]
        
        if len(local_values) == 0 or np.max(local_values) <= 0:
            return None

        if np.linalg.norm(prev_dir) == 0:
            best_dir = local_peaks[0]
        else:
            dots = np.dot(local_peaks, prev_dir)
            best_idx = np.argmax(np.abs(dots))
            best_dir = local_peaks[best_idx]
            
            if np.dot(best_dir, prev_dir) < 0:
                best_dir = -best_dir

        return best_dir

    def track_all(self, seeds_world):
        all_streamlines = []
        for seed in seeds_world:
            obs, _ = self.env.reset(options={'seed_world': seed})
            done = False
            
            while not done:
                # 1. Get exact peak vector
                best_dir = self.predict_continuous(obs)
                
                if best_dir is None:
                    break
                
                # 2. Inject this direction into the environment directly
                # We bypass the 'action_idx' to maintain precision
                obs, reward, done, truncated, info = self.env.manual_step(best_dir)
                if truncated: done = True
                
            streamline = self.env.get_world_streamline()
            if len(streamline) > 1:
                all_streamlines.append(streamline)
        return all_streamlines

class RewardDrivenAgent:
    """
    An agent that ignores deterministic peak-following and instead
    optimizes its path based solely on the Environment's reward signal.
    """
    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        """
        Decision Policy: Probe all 26 directions and pick the one 
        returning the highest reward from the environment.
        """
        best_reward = -np.inf
        best_action_idx = 0
        
        # Current state info from observation
        current_pos = obs[0:3]
        prev_dir = obs[3:6]

        # Iterate through all 26 possible discrete actions
        for action_idx in range(self.env.action_space.n):
            move_dir = self.env.actions[action_idx]
            
            # Enforce EuDX directionality (don't go backwards)
            # This ensures we don't calculate rewards for "U-turns"
            if np.linalg.norm(prev_dir) > 0:
                if np.dot(move_dir, prev_dir) < 0:
                    move_dir=-move_dir #continue 

            # Calculate potential new position
            potential_new_pos = current_pos + move_dir * self.env.step_size
            
            # Use the environment's own reward function to evaluate this move
            # Note: We call the internal method without actually stepping the env
            reward = self.env._compute_reward(current_pos, potential_new_pos, move_dir)
            
            if reward > best_reward:
                best_reward = reward
                best_action_idx = action_idx
                
        return best_action_idx

    def generate_streamline(self, seed_vox):
        """
        Standard tracking loop: reset -> predict (via reward) -> step.
        """
        obs, _ = self.env.reset()
        self.env.pos = seed_vox.astype(np.float32)
        self.env.streamline = [self.env.pos.copy()]
        
        done = False
        while not done:
            # The agent picks the action that yields the highest reward
            action = self.predict(obs)
            
            # Execute the step in the environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
        return self.env.get_world_streamline()

class ProbabilisticRewardDrivenAgent:
    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        current_pos = obs[0:3]
        prev_dir = obs[3:6]
        gfa = obs[6]
        
        # 1. FIXED: Random exploration (Epsilon-Greedy)
        # Using np.random.random() for 0.0-1.0 float comparison
        # Logic: If signal is weak (GFA < 0.3), 50% chance to explore randomly
        if gfa < 0.3 and np.random.random() < 0.5:
            # np.random.randint is preferred over random_integers
            return np.random.randint(0, self.env.action_space.n)
            
        best_reward = -np.inf
        best_action_idx = 0
        
        # 2. Iterate through actions
        for action_idx in range(self.env.action_space.n):
            move_dir = self.env.actions[action_idx]
            
            # FIXED: Axial Re-orientation
            # Instead of flipping move_dir, we align it to ensure no U-turns
            if np.linalg.norm(prev_dir) > 0:
                if np.dot(move_dir, prev_dir) < 0:
                    move_dir = -move_dir 

            potential_new_pos = current_pos + move_dir * self.env.step_size
            
            # 3. Evaluate Move
            # Note: We pass the target_label if the env is in pathfinding mode
            reward = self.env._compute_reward(
                current_pos, potential_new_pos, move_dir, 
                target_label=getattr(self.env, 'target_label', None)
            )
            
            if reward > best_reward:
                best_reward = reward
                best_action_idx = action_idx
                
        return best_action_idx

    def generate_streamline(self, seed_vox, target_label=None):
        # Set the goal in the environment context
        self.env.target_label = target_label
        self.env.found_target = False
        
        obs, _ = self.env.reset()
        self.env.pos = seed_vox.astype(np.float32)
        self.env.streamline = [self.env.pos.copy()]
        
        done = False
        while not done:
            action = self.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Break if we found the fMRI hub
            if getattr(self.env, 'found_target', False):
                break
                
            done = terminated or truncated
            
        return self.env.get_world_streamline()

class ContinuousRewardAgent:
    """
    An agent that optimizes its path in continuous 3D space by 
    probing a sphere of potential directions to maximize reward.
    """
    def __init__(self, env, num_samples=64):
        self.env = env
        self.num_samples = num_samples # Number of directions to probe on the sphere

    def _generate_sphere_samples(self, center_dir, span_angle=np.deg2rad(45)):
        """
        Generates sample vectors on a spherical cap centered around center_dir.
        Ensures the agent doesn't probe directions that violate curvature.
        """
        # Create random samples or use a Fibonacci sphere approach
        # For simplicity, we sample directions around the previous movement
        phi = np.random.uniform(0, 2 * np.pi, self.num_samples)
        costheta = np.random.uniform(np.cos(span_angle), 1, self.num_samples)
        theta = np.arccos(costheta)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        # Basic local coordinate system to align samples with center_dir
        # (This is a simplified rotation to align the 'z' axis with center_dir)
        samples = np.stack([x, y, z], axis=-1)
        
        # If no prev_dir, return global samples
        if np.linalg.norm(center_dir) == 0:
            return samples
            
        return samples # In a full implementation, rotate these to center_dir

    def predict(self, obs):
        """
        Policy: Evaluate 'num_samples' continuous directions and pick 
        the one returning the highest reward.
        """
        current_pos = obs[0:3]
        prev_dir = obs[3:6]

        # 1. Generate candidate vectors on the sphere
        # We focus our search within the valid curvature cone
        candidates = self._generate_sphere_samples(prev_dir)
        
        best_reward = -np.inf
        best_dir = prev_dir if np.linalg.norm(prev_dir) > 0 else np.array([1, 0, 0])

        # 2. Probe the reward landscape
        for move_dir in candidates:
            # Normalize candidate
            move_dir = move_dir / np.linalg.norm(move_dir)
            
            # Calculate potential reward
            potential_new_pos = current_pos + move_dir * self.env.step_size
            reward = self.env._compute_reward(current_pos, potential_new_pos, move_dir)
            
            if reward > best_reward:
                best_reward = reward
                best_dir = move_dir
                
        return best_dir

    def generate_streamline(self, seed_vox):
        obs, _ = self.env.reset()
        self.env.pos = seed_vox.astype(np.float32)
        self.env.streamline = [self.env.pos.copy()]
        
        done = False
        while not done:
            # Get high-precision best direction
            best_vector = self.predict(obs)
            
            # Use manual_step to bypass discrete action indexing
            obs, reward, terminated, truncated, info = self.env.manual_step(best_vector)
            done = terminated or truncated
            
        return self.env.get_world_streamline()
    
class LookAheadContinuousAgent:
    """
    An agent that evaluates directions based on 'Future Value' rather 
    than just immediate reward.
    """
    def __init__(self, env, look_ahead_steps=3, num_samples=32, gamma=0.9):
        self.env = env
        self.look_ahead_steps = look_ahead_steps # How many steps to "see" into the future
        self.num_samples = num_samples
        self.gamma = gamma # Discount factor for future rewards

    def _evaluate_path(self, start_pos, direction):
        """Simulates a path and returns the total discounted reward."""
        total_reward = 0
        current_pos = start_pos
        
        for t in range(self.look_ahead_steps):
            next_pos = current_pos + direction * self.env.step_size
            
            # Use internal reward logic
            r = self.env._compute_reward(current_pos, next_pos, direction)
            total_reward += (self.gamma ** t) * r
            
            # Update position for simulation
            current_pos = next_pos
            
        return total_reward

    def predict(self, obs):
        current_pos = obs[0:3]
        prev_dir = obs[3:6]
        
        # 1. Generate candidate directions (spherical cap)
        # We sample around the previous direction to maintain momentum
        candidates = self._generate_search_vectors(prev_dir)
        
        best_val = -np.inf
        best_dir = prev_dir if np.linalg.norm(prev_dir) > 0 else np.array([1, 0, 0])

        # 2. Look Ahead!
        for d in candidates:
            # How good is this direction over the next 3 steps?
            path_value = self._evaluate_path(current_pos, d)
            
            if path_value > best_val:
                best_val = path_value
                best_dir = d
                
        return best_dir

    def _generate_search_vectors(self, center_dir):
        # (Simplified: generates random vectors within a 45-degree cone)
        samples = []
        for _ in range(self.num_samples):
            vec = np.random.randn(3)
            vec /= np.linalg.norm(vec)
            # Ensure it points generally forward
            if np.linalg.norm(center_dir) > 0:
                if np.dot(vec, center_dir) < 0.7: # Within ~45 deg
                    continue
            samples.append(vec)
        return samples if samples else [center_dir]

    def generate_streamline(self, seed_vox):
        obs, _ = self.env.reset()
        self.env.pos = seed_vox.astype(np.float32)
        self.env.streamline = [self.env.pos.copy()]
        
        done = False
        while not done:
            best_vector = self.predict(obs)
            obs, reward, terminated, truncated, info = self.env.manual_step(best_vector)
            done = terminated or truncated
            
        return self.env.get_world_streamline()
    
class UltimateUnifiedAgent:
    """
    Combines:
    1. Continuous Action Space (Infinite angular precision)
    2. Look-Ahead Simulation (Long-term value estimation)
    3. Signal-Adaptive Exploration (Probabilistic behavior in low-SNR areas)
    """
    def __init__(self, env, num_samples=32, look_ahead_steps=3, gamma=0.9):
        self.env = env
        self.num_samples = num_samples
        self.look_ahead_steps = look_ahead_steps
        self.gamma = gamma

    def _generate_search_cone(self, center_dir, span_deg=45):
        """Generates candidate vectors in a cone around the current direction."""
        span_rad = np.deg2rad(span_deg)
        samples = []
        
        # If no previous direction, sample the whole sphere
        if np.linalg.norm(center_dir) == 0:
            for _ in range(self.num_samples):
                vec = np.random.randn(3)
                samples.append(vec / np.linalg.norm(vec))
            return np.array(samples)

        # Standard cone sampling logic
        center_dir = center_dir / np.linalg.norm(center_dir)
        for _ in range(self.num_samples):
            # Sample random vector and blend it with center_dir
            random_vec = np.random.randn(3)
            random_vec /= np.linalg.norm(random_vec)
            
            # Weighted average to stay within the 'cone'
            blended = center_dir + random_vec * np.tan(span_rad/2)
            samples.append(blended / np.linalg.norm(blended))
            
        return np.array(samples)

    def _evaluate_future(self, start_pos, direction):
        """Simulates path value over N steps."""
        total_val = 0
        curr_pos = start_pos
        sim_prev_dir = direction
        
        for t in range(self.look_ahead_steps):
            next_pos = curr_pos + direction * self.env.step_size
            # Query the environment reward logic
            r = self.env._compute_reward(curr_pos, next_pos, direction)
            total_val += (self.gamma ** t) * r
            curr_pos = next_pos
        return total_val

    def predict(self, obs):
        current_pos = obs[0:3]
        prev_dir = obs[3:6]
        gfa = obs[6]

        # 1. Probabilistic Logic: If signal is weak, explore randomly
        # This helps "tunnel" through noisy voxels or crossing junctions
        if gfa < 0.3 and np.random.random() < 0.3:
           # Try up to 10 times to find a valid random direction
            for _ in range(10):
                random_vec = np.random.randn(3)
                random_vec /= np.linalg.norm(random_vec)
                
                # Check Curvature: Is this random jump too sharp?
                if np.linalg.norm(prev_dir) > 0:
                    cos_sim = np.dot(random_vec, prev_dir)
                    # If angle is < 45 degrees, it's a valid 'exploratory' step
                    if cos_sim > self.env.max_curvature_cos:
                        return random_vec
        # 2. Continuous Search: Generate candidate vectors
        # We look ahead in a 45-degree cone to find the best biological path
        candidates = self._generate_search_cone(prev_dir, span_deg=45)
        
        best_val = -np.inf
        best_dir = prev_dir if np.linalg.norm(prev_dir) > 0 else np.array([1, 0, 0])

        # 3. Look-Ahead Evaluation
        for d in candidates:
            path_value = self._evaluate_future(current_pos, d)
            if path_value > best_val:
                best_val = path_value
                best_dir = d
                
        return best_dir

    def generate_streamline(self, seed_vox):
        obs, _ = self.env.reset()
        self.env.pos = seed_vox.astype(np.float32)
        self.env.streamline = [self.env.pos.copy()]
        
        done = False
        while not done:
            best_vector = self.predict(obs)
            
            # Step using high-precision manual_step
            obs, reward, terminated, truncated, info = self.env.manual_step(best_vector)
            done = terminated or truncated
            
        return self.env.get_world_streamline()
  
class MemoryAwareLookAheadAgent: # used updated env
    """
    An agent that uses a history of 4 directions to maintain 
    anatomical momentum and perform intelligent look-ahead.
    """
    def __init__(self, env, look_ahead_steps=3, num_samples=32, gamma=0.9):
        self.env = env
        self.look_ahead_steps = look_ahead_steps
        self.num_samples = num_samples
        self.gamma = gamma
        # Weights for the last 4 directions (Recent directions matter more)
        self.history_weights = np.array([0.5, 0.25, 0.15, 0.10])

    def _get_momentum_vector(self, prev_dirs):
        """
        Calculates a weighted average of the previous 4 directions.
        prev_dirs shape: (4, 3) where [0] is the most recent.
        """
        # If we just started, we won't have 4 directions yet
        active_dirs = 0
        momentum = np.zeros(3)
        
        for i in range(4):
            if np.linalg.norm(prev_dirs[i]) > 0:
                momentum += prev_dirs[i] * self.history_weights[i]
                active_dirs += 1
        
        if active_dirs == 0:
            return None
            
        return momentum / np.linalg.norm(momentum)

    def _generate_search_cone(self, momentum_dir, span_deg=30):
        """Generates candidates centered around the weighted momentum."""
        samples = []
        span_rad = np.deg2rad(span_deg)
        
        for _ in range(self.num_samples):
            random_vec = np.random.randn(3)
            random_vec /= np.linalg.norm(random_vec)
            
            # Blend the momentum with random jitter
            # Note: We use a narrower cone (30 deg) because momentum is more reliable
            blended = momentum_dir + random_vec * np.tan(span_rad/2)
            samples.append(blended / np.linalg.norm(blended))
            
        return np.array(samples)

    def predict(self, obs):
        # State: [pos(3), prev_dirs(4*3=12), gfa(1), time(1)] = 17 dims
        current_pos = obs[0:3]
        # Reshape the 12 direction values back into (4, 3)
        prev_dirs = obs[3:15].reshape(4, 3)
        gfa = obs[15]

        # 1. Calculate Momentum from History
        momentum_dir = self._get_momentum_vector(prev_dirs)

        # 2. Adaptive Search: If GFA is low, widen the search; if high, trust momentum
        span = 45 if gfa < 0.3 else 25
        
        if momentum_dir is None:
            # First step logic: random sphere search
            candidates = [np.random.randn(3) for _ in range(self.num_samples)]
            candidates = [v / np.linalg.norm(v) for v in candidates]
        else:
            candidates = self._generate_search_cone(momentum_dir, span_deg=span)

        # 3. Look-Ahead Evaluation
        best_val = -np.inf
        best_dir = momentum_dir if momentum_dir is not None else np.array([1,0,0])

        for d in candidates:
            # Simulate path based on this candidate
            path_value = 0
            sim_pos = current_pos
            for t in range(self.look_ahead_steps):
                next_pos = sim_pos + d * self.env.step_size
                # Use environment's interpolated reward
                r = self.env._compute_reward(sim_pos, next_pos, d)
                path_value += (self.gamma ** t) * r
                sim_pos = next_pos
            
            if path_value > best_val:
                best_val = path_value
                best_dir = d
                
        return best_dir

    def generate_streamline(self, seed_vox):
        obs, _ = self.env.reset()
        # Initialize the env position manually
        self.env.pos = seed_vox.astype(np.float32)
        self.env.streamline = [self.env.pos.copy()]
        
        done = False
        while not done:
            best_vector = self.predict(obs)
            obs, reward, terminated, truncated, info = self.env.manual_step_rk2(best_vector)
            done = terminated or truncated
            
        return self.env.get_world_streamline()
    
class FinalUnifiedAgent:

    """
    Combines:
    1. Continuous Action Space (infinite angular precision)
    2. Look-Ahead Simulation (long-term value estimation)
    3. Signal-Adaptive Exploration (probabilistic behavior in low-SNR areas)
    4. Momentum-Aware Search (weighted history of past directions)
    5. Adaptive Step Size (0.5 mm normally, 0.2 mm in low-GFA regions)
    """

    def __init__(self, env, num_samples=32, look_ahead_steps=3, gamma=0.9):
        self.env = env
        self.num_samples = num_samples
        self.look_ahead_steps = look_ahead_steps
        self.gamma = gamma
        # Weights for the last 4 directions (recent directions matter more)
        self.history_weights = np.array([0.5, 0.25, 0.15, 0.10])

    def _get_momentum_vector(self, prev_dirs):
        """Weighted average of previous 4 directions."""
        momentum = np.zeros(3)
        active_dirs = 0
        for i in range(4):
            if np.linalg.norm(prev_dirs[i]) > 0:
                momentum += prev_dirs[i] * self.history_weights[i]
                active_dirs += 1
        if active_dirs == 0:
            return None
        return momentum / np.linalg.norm(momentum)

    def _generate_search_cone(self, center_dir, span_deg=45):
        """Generates candidate vectors in a cone around the given direction."""
        span_rad = np.deg2rad(span_deg)
        samples = []

        if center_dir is None or np.linalg.norm(center_dir) == 0:
            # Sample whole sphere if no direction
            for _ in range(self.num_samples):
                vec = np.random.randn(3)
                samples.append(vec / np.linalg.norm(vec))
            return np.array(samples)

        center_dir = center_dir / np.linalg.norm(center_dir)
        for _ in range(self.num_samples):
            random_vec = np.random.randn(3)
            random_vec /= np.linalg.norm(random_vec)
            blended = center_dir + random_vec * np.tan(span_rad / 2)
            samples.append(blended / np.linalg.norm(blended))
        return np.array(samples)

    def _evaluate_future(self, start_pos, direction, step_size):
        """Simulates path value over N steps with adaptive step size."""
        total_val = 0
        curr_pos = start_pos
        for t in range(self.look_ahead_steps):
            next_pos = curr_pos + direction * step_size
            r = self.env._compute_reward(curr_pos, next_pos, direction)
            total_val += (self.gamma ** t) * r
            curr_pos = next_pos
        return total_val

    def predict(self, obs):
        """
        State format:
        - Simple: [pos(3), prev_dir(3), gfa(1)]
        - Memory-aware: [pos(3), prev_dirs(12), gfa(1), time(1)]
        """
        current_pos = obs[0:3]

        if len(obs) >= 17:  # Memory-aware format
            prev_dirs = obs[3:15].reshape(4, 3)
            gfa = obs[15]
            momentum_dir = self._get_momentum_vector(prev_dirs)
        else:  # Simple format
            prev_dir = obs[3:6]
            gfa = obs[6]
            prev_dirs = np.zeros((4, 3))
            prev_dirs[0] = prev_dir
            momentum_dir = prev_dir if np.linalg.norm(prev_dir) > 0 else None

        # 🔑 Adaptive step size
        step_size = 0.2 if gfa < 0.3 else 0.5

        # 1. Signal-Adaptive Exploration
        if gfa < 0.3 and np.random.random() < 0.3:
            for _ in range(10):
                random_vec = np.random.randn(3)
                random_vec /= np.linalg.norm(random_vec)
                if np.linalg.norm(prev_dirs[0]) > 0:
                    cos_sim = np.dot(random_vec, prev_dirs[0])
                    if cos_sim > self.env.max_curvature_cos:
                        return random_vec, step_size

        # 2. Adaptive Search Cone
        span = 45 if gfa < 0.3 else 25
        candidates = self._generate_search_cone(momentum_dir, span_deg=span)

        # 3. Look-Ahead Evaluation
        best_val = -np.inf
        best_dir = momentum_dir if momentum_dir is not None else np.array([1, 0, 0])
        for d in candidates:
            path_value = self._evaluate_future(current_pos, d, step_size)
            if path_value > best_val:
                best_val = path_value
                best_dir = d

        return best_dir, step_size

    def generate_streamline(self, seed_vox):
        obs, _ = self.env.reset()
        self.env.pos = seed_vox.astype(np.float32)
        self.env.streamline = [self.env.pos.copy()]

        done = False
        while not done:
            best_vector, step_size = self.predict(obs)

            # Override env step size dynamically
            self.env.step_size = step_size

            if hasattr(self.env, "manual_step_rk4"):
                obs, reward, terminated, truncated, info = self.env.manual_step_rk4(best_vector)
            else:
                obs, reward, terminated, truncated, info = self.env.manual_step(best_vector)

            done = terminated or truncated

        return self.env.get_world_streamline()

class BranchingStreamlineAgent:
    def __init__(self, env, peak_threshold=0.4, branching_angle=35):
        self.env = env
        self.peak_threshold = peak_threshold  # Min relative strength to branch
        self.branching_angle = branching_angle # Min angle to consider it a "new" path

    def predict_branches(self, obs):
        """
        Detects multiple valid directions in the current voxel.
        Returns a list of valid 'forward' vectors.
        """
        current_pos = obs[0:3]
        prev_dir = obs[3:6]
        
        v_idx = tuple(np.floor(current_pos).astype(int))
        local_peaks = self.env.peaks.peak_dirs[v_idx]
        local_values = self.env.peaks.peak_values[v_idx]

        if len(local_values) == 0 or np.max(local_values) <= 0:
            return []

        valid_branches = []
        max_val = np.max(local_values)

        for i, peak in enumerate(local_peaks):
            # 1. Strength Check: Only follow significant peaks
            if local_values[i] < (max_val * self.peak_threshold):
                continue
            
            # 2. Alignment: Ensure it's not a U-turn
            aligned_peak = peak if np.dot(peak, prev_dir) >= 0 else -peak
            
            if np.linalg.norm(prev_dir) > 0:
                cos_sim = np.dot(aligned_peak, prev_dir)
                # 3. Curvature Check: Must be within anatomical limits
                if cos_sim < self.env.max_curvature_cos:
                    continue
            
            valid_branches.append(aligned_peak)

        return valid_branches

    def track_with_branching(self, initial_seed):
        """
        The main loop that manages the 'Frontier' of active streamlines.
        """
        all_finished_streamlines = []

        initial_prev_dir = np.zeros(3, dtype=np.float32)
        # Queue stores: (current_pos, current_prev_dir, current_path_list)
        queue = [(initial_seed, initial_prev_dir, [initial_seed])]

        while queue:
            pos, prev_dir, history = queue.pop(0)
            
            # Reset environment internal state
            self.env.reset(options={'seed_vox': pos})
            # Manually set environment state
            self.env.pos = pos
            self.env.prev_dir[0] = prev_dir
            self.env.streamline = history
            
            done = False
            while not done:
                obs = self.env._get_obs()
                branches = self.predict_branches(obs)

                if not branches:
                    break
                
                # Primary branch continues current loop
                main_dir = branches[0]
                
                # Secondary branches are added to the queue as new seeds
                if len(branches) > 1:
                    for extra_dir in branches[1:]:
                        # Check if we've already branched here to avoid infinite loops
                        queue.append((self.env.pos.copy(), extra_dir.copy(), list(history)))

                # Execute step for the main branch
                obs, reward, terminated, truncated, _ = self.env.manual_step(main_dir)
                done = terminated or truncated

            final_sl = self.env.get_world_streamline()
            if len(final_sl) > 1:
                all_finished_streamlines.append(final_sl)
        
        return all_finished_streamlines

class UnifiedBranchingAgent:
    """
    Combines:
    1. Continuous Action Space (infinite angular precision)
    2. Look-Ahead Simulation (long-term value estimation)
    3. Signal-Adaptive Exploration (probabilistic behavior in low-SNR areas)
    4. Momentum-Aware Search (weighted history of past directions)
    5. Adaptive Step Size (0.5 mm normally, 0.2 mm in low-GFA regions)
    6. Branching on local peaks (strength + curvature + forward alignment)
    """

    def __init__(self, env, num_samples=32, look_ahead_steps=3, gamma=0.9,
                 history_weights=None, low_gfa_thresh=0.3, exploration_prob=0.3,
                 cone_span_low=45, cone_span_high=25, peak_threshold=0.4):
        self.env = env
        self.num_samples = num_samples
        self.look_ahead_steps = look_ahead_steps
        self.gamma = gamma
        self.low_gfa_thresh = low_gfa_thresh
        self.exploration_prob = exploration_prob
        self.cone_span_low = cone_span_low
        self.cone_span_high = cone_span_high
        self.peak_threshold = peak_threshold
        self.history_weights = (np.array([0.5, 0.25, 0.15, 0.10])
                                if history_weights is None else np.asarray(history_weights))

    # -------------------------
    # Momentum and sampling
    # -------------------------
    def _get_momentum_vector(self, prev_dirs):
        """Weighted average of previous 4 directions (recent matter more)."""
        momentum = np.zeros(3, dtype=np.float32)
        active_dirs = 0
        for i in range(4):
            if np.linalg.norm(prev_dirs[i]) > 0:
                momentum += prev_dirs[i] * self.history_weights[i]
                active_dirs += 1
        if active_dirs == 0:
            return None
        norm = np.linalg.norm(momentum)
        return momentum / norm if norm > 0 else None

    def _generate_search_cone(self, center_dir, span_deg):
        """Generates candidate unit vectors within a cone around center_dir; sphere if center_dir is None."""
        span_rad = np.deg2rad(span_deg)
        samples = []

        if center_dir is None or np.linalg.norm(center_dir) == 0:
            for _ in range(self.num_samples):
                vec = np.random.randn(3)
                vec /= np.linalg.norm(vec)
                samples.append(vec)
            return np.array(samples, dtype=np.float32)

        center_dir = center_dir / np.linalg.norm(center_dir)
        for _ in range(self.num_samples):
            random_vec = np.random.randn(3)
            random_vec /= np.linalg.norm(random_vec)
            blended = center_dir + random_vec * np.tan(span_rad / 2)
            blended /= np.linalg.norm(blended)
            samples.append(blended.astype(np.float32))
        return np.array(samples, dtype=np.float32)

    # -------------------------
    # Look-ahead evaluation
    # -------------------------
    def _evaluate_future(self, start_pos, direction, step_size):
        """Simulates discounted path reward over N steps with fixed step_size."""
        total_val = 0.0
        curr_pos = start_pos
        for t in range(self.look_ahead_steps):
            next_pos = curr_pos + direction * step_size
            r = self.env._compute_reward(curr_pos, next_pos, direction)
            total_val += (self.gamma ** t) * float(r)
            curr_pos = next_pos
        return total_val

    # -------------------------
    # Branching from local peaks
    # -------------------------
    def _valid_branches_from_peaks(self, current_pos, prev_dir):
        """
        Returns forward-aligned, curvature-valid peak directions above a relative strength threshold.
        """
        v_idx = tuple(np.floor(current_pos).astype(int))
        local_peaks = self.env.peaks.peak_dirs[v_idx]
        local_values = self.env.peaks.peak_values[v_idx]

        if len(local_values) == 0 or np.max(local_values) <= 0:
            return []

        valid = []
        max_val = float(np.max(local_values))

        for i, peak in enumerate(local_peaks):
            # 1) Strength gate
            if local_values[i] < (max_val * self.peak_threshold):
                continue

            # 2) Forward alignment (no U-turns)
            aligned_peak = peak if np.dot(peak, prev_dir) >= 0 else -peak

            # 3) Curvature gate
            if np.linalg.norm(prev_dir) > 0:
                cos_sim = float(np.dot(aligned_peak, prev_dir))
                if cos_sim < self.env.max_curvature_cos:
                    continue

            valid.append(aligned_peak.astype(np.float32))

        return valid

    # -------------------------
    # One-step prediction with exploration, cone sampling, look-ahead
    # -------------------------
    def predict(self, obs):
        """
        State format:
        - Simple: [pos(3), prev_dir(3), gfa(1)]
        - Memory-aware: [pos(3), prev_dirs(12), gfa(1), time(1)]
        Returns: (best_dir, step_size)
        """
        current_pos = obs[0:3]

        if len(obs) >= 17:
            prev_dirs = obs[3:15].reshape(4, 3)
            gfa = float(obs[15])
            momentum_dir = self._get_momentum_vector(prev_dirs)
        else:
            prev_dir = obs[3:6]
            gfa = float(obs[6])
            prev_dirs = np.zeros((4, 3), dtype=np.float32)
            prev_dirs[0] = prev_dir
            momentum_dir = prev_dir if np.linalg.norm(prev_dir) > 0 else None

        # Adaptive step size
        step_size = 0.2 if gfa < self.low_gfa_thresh else 0.5

        # Signal-adaptive exploration in low-GFA regions
        if gfa < self.low_gfa_thresh and np.random.random() < self.exploration_prob:
            for _ in range(10):
                random_vec = np.random.randn(3)
                random_vec /= np.linalg.norm(random_vec)
                if np.linalg.norm(prev_dirs[0]) > 0:
                    cos_sim = float(np.dot(random_vec, prev_dirs[0]))
                    if cos_sim > self.env.max_curvature_cos:
                        return random_vec.astype(np.float32), step_size

        # Cone span adapts to GFA
        span = self.cone_span_low if gfa < self.low_gfa_thresh else self.cone_span_high
        candidates = self._generate_search_cone(momentum_dir, span_deg=span)

        # Look-ahead evaluation
        best_val = -np.inf
        best_dir = momentum_dir if momentum_dir is not None else np.array([1, 0, 0], dtype=np.float32)
        for d in candidates:
            path_value = self._evaluate_future(current_pos, d, step_size)
            if path_value > best_val:
                best_val = path_value
                best_dir = d

        return best_dir.astype(np.float32), step_size

    # -------------------------
    # Single streamline (no branching)
    # -------------------------
    def generate_streamline(self, seed_vox):
        """Tracks a single streamline from seed_vox using predict()."""
        self.env.reset()
        self.env.pos = np.asarray(seed_vox, dtype=np.float32)
        self.env.streamline = [self.env.pos.copy()]

        done = False
        obs = self.env._get_obs()
        while not done:
            best_vector, step_size = self.predict(obs)
            self.env.step_size = step_size

            if hasattr(self.env, "manual_step_rk4"):
                obs, reward, terminated, truncated, info = self.env.manual_step_rk4(best_vector)
            else:
                obs, reward, terminated, truncated, info = self.env.manual_step(best_vector)

            done = bool(terminated) or bool(truncated)

        return self.env.get_world_streamline()

    # -------------------------
    # Streamline tracking with branching
    # -------------------------
    def track(self, initial_seed, enable_branching=True, max_frontier=1000, max_branches_per_seed=10):
        """
        Tracks from a seed with optional branching and performance limits.
        """
        all_finished = []
        # Queue stores: (pos, prev_dir, history, branch_count_for_this_path)
        frontier = [(np.asarray(initial_seed, dtype=np.float32),
                     np.zeros(3, dtype=np.float32),
                     [np.asarray(initial_seed, dtype=np.float32)],
                     0)]  # Start with 0 branches used

        while frontier:
            if len(frontier) > max_frontier:
                break

            pos, prev_dir, history, current_branch_count = frontier.pop(0)

            # Reset env to current node
            self.env.reset(options={'seed_vox': pos})
            self.env.pos = pos.copy()
            if hasattr(self.env, "prev_dir"):
                self.env.prev_dir[0] = prev_dir.copy()
            self.env.streamline = list(history)

            done = False
            while not done:
                obs = self.env._get_obs()

                # Determine if we are allowed to branch further
                can_still_branch = enable_branching and (current_branch_count < max_branches_per_seed)
                
                branches = self._valid_branches_from_peaks(obs[0:3], obs[3:6]) if can_still_branch else []

                if can_still_branch and branches:
                    # Follow the strongest valid peak in this loop
                    main_dir = branches[0]

                    # Queue extra branches only if budget allows
                    if len(branches) > 1:
                        for extra_dir in branches[1:]:
                            # Increment branch count for new paths spawned
                            frontier.append((self.env.pos.copy(),
                                             extra_dir.copy(),
                                             list(self.env.streamline),
                                             current_branch_count + 1))
                    
                    # Execute main branch step
                    if hasattr(self.env, "manual_step_rk4"):
                        obs, reward, terminated, truncated, _ = self.env.manual_step_rk4(main_dir)
                    else:
                        obs, reward, terminated, truncated, _ = self.env.manual_step(main_dir)
                    done = bool(terminated) or bool(truncated)
                    continue

                # Fallback to standard unified predict (no new branches spawned here)
                best_vector, step_size = self.predict(obs)
                self.env.step_size = step_size

                if hasattr(self.env, "manual_step_rk4"):
                    obs, reward, terminated, truncated, _ = self.env.manual_step_rk4(best_vector)
                else:
                    obs, reward, terminated, truncated, _ = self.env.manual_step(best_vector)
                done = bool(terminated) or bool(truncated)

            sl = self.env.get_world_streamline()
            if sl is not None and len(sl) > 1:
                all_finished.append(sl)

        return all_finished
