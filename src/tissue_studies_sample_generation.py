"""
tissue_studies_sample_generation.py
Extracts contiguous tissue samples based on a probability threshold and calculates tissue ratios.

Example use:
python tissue_studies_sample_generation.py --sub 5 --ses 2 --threshold 0.90
"""

import os
import json
import argparse
import numpy as np
import nibabel as nib
import multiprocessing as mp
import concurrent.futures
from nilearn.image import resample_to_img
from scipy.spatial.distance import cdist

def json_compatibility_encoder(obj):
    """Converts numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

def get_neighbors(center_vox):
    """Returns the 26 connected neighbor coordinates for a given voxel."""
    cube_range = np.arange(-1, 2)
    i, j, k = np.meshgrid(cube_range, cube_range, cube_range, indexing='ij')
    neighbor_relative_indices = np.stack((i, j, k), axis=-1).reshape(-1, 3)
    return center_vox + neighbor_relative_indices

def check_valid(center_vox, brain_shape):
    """Ensures a voxel and its immediate neighbors are within brain dimensions."""
    for pos in range(len(center_vox)):
        if not (1 < center_vox[pos] < brain_shape[pos] - 2):
            return False
    return True

def build_sample(seed_voxel, gm_data, wm_data, csf_data, threshold, output_dir, max_size=150):
    """Grows a contiguous sample of voxels meeting the threshold and computes tissue ratios."""
    valid_neighbors = set()
    checked = set()
    to_check = {tuple(seed_voxel)}
    
    while to_check and len(valid_neighbors) < max_size:
        looking = to_check.pop()
        
        if (gm_data[looking] >= threshold) and check_valid(looking, gm_data.shape):
            valid_neighbors.add(looking)
            neighbors = set(map(tuple, get_neighbors(looking).tolist()))
            neighbors.difference_update(neighbors & checked)
            to_check.update(neighbors)
            
        checked.add(looking)
    
    # Calculate tissue ratios (mean probability across the sample)
    sample_coords = list(valid_neighbors)
    if not sample_coords:
        return False
        
    gm_ratio = np.mean([gm_data[c] for c in sample_coords])
    wm_ratio = np.mean([wm_data[c] for c in sample_coords])
    csf_ratio = np.mean([csf_data[c] for c in sample_coords])
    
    formatted_seed = "_".join(map(str, seed_voxel))
    output_payload = {
        "metadata": {
            "seed_voxel": tuple(seed_voxel),
            "threshold_used": threshold,
            "sample_size": len(sample_coords),
            "tissue_ratios": {
                "gray_matter": gm_ratio,
                "white_matter": wm_ratio,
                "csf": csf_ratio
            }
        },
        "coordinates": sample_coords
    }
    
    file_path = os.path.join(output_dir, f'sample_seed_{formatted_seed}.json')
    with open(file_path, 'w') as f:
        json.dump(output_payload, f, default=json_compatibility_encoder, indent=4)
        
    return True

def build_paths(config, sub_id, ses_id):
    """
    Dynamically generates paths anchored safely to the project root directory.
    """
    # 1. Find the absolute path of the directory containing this script (src/) 
    # and go one level up to find 'my_project/' root.
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent 
    
    sub_str = f"{config['subj_prefix']}{sub_id:02d}"
    ses_str = f"{config['sess_prefix']}{ses_id:02d}"
    
    # 2. Safely anchor all relative config paths to the absolute project root
    base_data_dir = project_root / config['data_dir']
    func_dir = base_data_dir / sub_str / ses_str / 'func'
    anat_dir = base_data_dir / sub_str / 'anat'
    
    paths = {
        'bold': func_dir / f"{sub_str}_{ses_str}_{config['task_label']}_{config['space_label']}_{config['skull_stripped_suffix']}",
        'gm': anat_dir / f"{sub_str}_{config['probseg_space']}_label-GM_probseg.nii.gz",
        'wm': anat_dir / f"{sub_str}_{config['probseg_space']}_label-WM_probseg.nii.gz",
        'csf': anat_dir / f"{sub_str}_{config['probseg_space']}_label-CSF_probseg.nii.gz",
        'output_dir': project_root / config['data_output_dir'] / sub_str / ses_str
    }
        
    # Convert Path objects to strings
    return {k: str(v) for k, v in paths.items()}


def main(args):

    # Load the configuration file
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    # Generate all the relevant file paths
    paths = build_paths(config, args.sub, args.ses)
    
    os.makedirs(paths['output_dir'], exist_ok=True)
    
    # Load base images using the generated paths
    print(f"Loading data for Subject {args.sub}, Session {args.ses}...")
    brain_img = nib.load(paths['bold'])
    gm_ps_img = nib.load(paths['gm'])
    wm_ps_img = nib.load(paths['wm'])
    csf_ps_img = nib.load(paths['csf'])
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Resample probsegs to BOLD space
    bold_gm = resample_to_img(source_img=gm_ps_img, target_img=brain_img, interpolation='continuous').get_fdata()
    bold_wm = resample_to_img(source_img=wm_ps_img, target_img=brain_img, interpolation='continuous').get_fdata()
    bold_csf = resample_to_img(source_img=csf_ps_img, target_img=brain_img, interpolation='continuous').get_fdata()
    
    # Identify initial highly-probable GM seeds (e.g., >0.99) to act as roots for sampling
    idxs_99 = np.argwhere(bold_gm >= 0.99)
    rng = np.random.default_rng(seed=42)
    seeds = rng.choice(idxs_99, size=args.num_seeds, replace=False)
    
    # Filter seeds for spatial separation
    dist_matrix = cdist(seeds, seeds, metric='euclidean')
    closeness = dist_matrix.sum(axis=1)
    _, edges = np.histogram(closeness, bins='auto')
    sprouts = seeds[np.where(closeness > edges[5])[0]]
    
    print(f"Generated {len(sprouts)} viable, spatially distinct seed voxels. Processing...")

    context = mp.get_context('fork')
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {
            executor.submit(
                build_sample, vxl, bold_gm, bold_wm, bold_csf, args.threshold, args.output_dir
            ): i for i, vxl in enumerate(sprouts)
        }
        
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Catch any exceptions raised during processing
            
    print(f"Generation complete. Samples saved to {paths['output_dir']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tissue samples using a config file.")
    
    # User-friendly required arguments
    parser.add_argument("--config", type=str, default="config.json", help="Path to the JSON configuration file.")
    parser.add_argument("--sub", type=int, required=True, help="Subject ID (integer, e.g., 1 for sub-MSC01).")
    parser.add_argument("--ses", type=int, default=1, help="Session ID (integer, e.g., 1 for ses-func01).")
    
    # Optional overrides
    parser.add_argument("--threshold", type=float, default=None, help="Override default GM threshold.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers.")
    
    args = parser.parse_args()
    main(args)