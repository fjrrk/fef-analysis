"""
tissue_studies_variance_analysis.py
Calculates variance for tissue samples, fits a KDE, finds inflection points, and sub-samples coordinates.
"""

import os
import json
import glob
import argparse
import numpy as np
import nibabel as nib
from scipy import stats

def json_compatibility_encoder(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

def calculate_kde(variances):
    """Fits a Gaussian KDE to a 1D array of variances."""
    # Add minor noise to avoid singular matrix if all variances are perfectly identical
    if np.var(variances) == 0:
        variances += np.random.normal(0, 1e-6, size=variances.shape)
    return stats.gaussian_kde(variances)

def inflection_pointer(kde, data_min, data_max):
    """Finds inflection points in the KDE using the second derivative."""
    x_grid = np.linspace(data_min, data_max, 1000)
    kde_eval = kde(x_grid)
    
    kde_eval_d1 = np.gradient(kde_eval)
    kde_eval_d2 = np.gradient(kde_eval_d1)
    
    inflections = np.where(np.diff(np.sign(kde_eval_d2)))[0]
    return x_grid[inflections]

def segment_coordinates(coords, variances, inflections):
    """Groups coordinates into sub-samples based on their variance relative to inflection points."""
    segments = {}
    
    for coord, var in zip(coords, variances):
        # Determine which segment the variance falls into
        # If var < inflections[0], segment 0. If between 0 and 1, segment 1, etc.
        segment_idx = np.searchsorted(inflections, var)
        
        seg_key = f"segment_{segment_idx}"
        if seg_key not in segments:
            segments[seg_key] = []
            
        segments[seg_key].append(coord)
        
    return segments

def main(args):
    # Load the bold image to access the actual timeseries data
    brain_img = nib.load(args.bold_path)
    brain_data = brain_img.get_fdata()
    
    sample_files = glob.glob(os.path.join(args.input_dir, "*.json"))
    print(f"Found {len(sample_files)} sample files to process.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for file_path in sample_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Extract coordinates (assuming they were saved under the "coordinates" key)
        coords = data.get("coordinates", [])
        if not coords:
            continue
            
        # 1 & 2. Get timeseries and calculate variance for each coordinate
        # timeseries shape: (N_voxels, Timepoints)
        timeseries = np.array([brain_data[tuple(c)] for c in coords])
        variances = np.var(timeseries, axis=1, ddof=1)
        
        # 3. Fit PDF (KDE)
        kde = calculate_kde(variances)
        
        # 4. Find Inflection points
        inflections = inflection_pointer(kde, variances.min(), variances.max())
        
        # 5. Create Sub-samples
        sub_samples = segment_coordinates(coords, variances, inflections)
        
        # Prepare payload to save
        output_payload = {
            "metadata": data.get("metadata", {}),
            "variance_stats": {
                "min_var": variances.min(),
                "max_var": variances.max(),
                "inflection_points": inflections.tolist()
            },
            "sub_samples": sub_samples
        }
        
        filename = os.path.basename(file_path).replace(".json", "_segmented.json")
        out_path = os.path.join(args.output_dir, filename)
        
        with open(out_path, 'w') as f:
            json.dump(output_payload, f, default=json_compatibility_encoder, indent=4)
            
    print(f"Variance analysis and segmentation complete. Saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze variance and segment tissue samples.")
    parser.add_argument("--bold_path", type=str, required=True, help="Path to skull-stripped BOLD image.")
    parser.add_argument("--input_dir", type=str, default="./datasets/tissue_studies", help="Directory containing generated sample JSONs.")
    parser.add_argument("--output_dir", type=str, default="./datasets/segmented_studies", help="Directory to save the segmented JSON outputs.")
    
    args = parser.parse_args()
    main(args)