import numpy as np
import os
import time
from fury import window, actor

# Data and Class imports
from data import data, affine, labels, bvals, bvecs, vox_size
from env_main_updated import UnifiedTractographyEnv
from agent import BranchingStreamlineAgent,UnifiedBranchingAgent
from dipy.viz import colormap
from dipy.tracking import utils
from dipy.io.streamline import save_trk
from data import hardi_img
from dipy.io.stateful_tractogram import Space, StatefulTractogram

def main():
    # 1. Environment Setup
    # Matching your custom_tracker requirements: GFA < 0.25, Angle > 45 deg
    print("--- Initializing RL Environment ---")
    env = UnifiedTractographyEnv(
        data=data,
        affine=affine,
        labels=labels,
        bvals=bvals,
        bvecs=bvecs,
        vox_size=vox_size,
        step_size=0.5,      # Matches custom_tracker
        max_steps=500,       # Matches custom_tracker
        fa_threshold=0.25,   # Matches custom_tracker
        max_curvature_deg=45 # Matches custom_tracker
    )

    # 2. Agent Setup
    agent = BranchingStreamlineAgent(env, peak_threshold=0.4)

    # 3. Seed Selection
    # We grab all seeds from labels (label 2)
    mask=(labels==2)|(labels==1)
    scene=window.Scene()
    # env.render_wm_surface(scene)
    seeds_mask=labels==2
    # env.render_seeds_mask(scene)
    # env.render_bval_bvec(scene, seeds_mask)
    
    seed_indices = utils.seeds_from_mask((labels == 2),affine=np.eye(4),density=(1,1,1))
    np.random.shuffle(seed_indices)
    num_seeds =  len(seed_indices)
    # scene = window.Scene()
    all_streamlines = []

    print(f"Starting branching tracking for {num_seeds} seeds...")

    for i in range(num_seeds):
        seed = seed_indices[i].astype(np.float32)
        
        # This returns a LIST of streamlines because of branching
        branches = agent.track(seed)
        
        if branches:
            all_streamlines.extend(branches)
            print(f"Seed {i+1}: Found {len(branches)} branches.")

    # 4. Rendering in Fury
    if all_streamlines:
        print(f"Total streamlines generated: {len(all_streamlines)}")
        
        # Use coloring based on local orientation for visual clarity
        env.render_streamlines(scene,all_streamlines)
        env.render_wm_surface(scene)
        
        # Add the white matter surface at low opacity for anatomical context
        # env.render_wm_surface(scene, opacity=0.05)
    
        scene.reset_camera()
        
        print("Opening interactive window...")
        window.show(scene, size=(1024, 768), title="Branching Fiber Tracking")
    else:
        print("No streamlines found. Check peak thresholds.")
    # 5. Save Streamlines
    if all_streamlines:
        print("Saving streamlines to 'branching_tractography.trk'...")
        sft = StatefulTractogram(all_streamlines, hardi_img, Space.RASMM)
        save_trk(sft, 'branching_tractography.trk', bbox_valid_check=False)
        print("Streamlines saved successfully.")
    

    

if __name__ == "__main__":
    main()
