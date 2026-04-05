import pyvista as pv
import numpy as np

def save_batch_to_vtk(tensor, filename_prefix="flow"):
    """
    tensor: [B, C, X, Y, Z]
    """
    tensor = tensor.detach().cpu().numpy()
    
    B, C, X, Y, Z = tensor.shape
    
    for b in range(B):
        u = tensor[b, 0]
        v = tensor[b, 1]
        
        # Find non-zero voxels
        mask = (np.abs(u) + np.abs(v)) > 1e-6
        x, y, z = np.where(mask)
        
        if len(x) == 0:
            print("No data in batch", b)
            continue
        
        points = np.stack([x, y, z], axis=1).astype(np.float32)
        
        velocity = np.stack([
            u[mask],
            v[mask],
            np.zeros_like(u[mask])
        ], axis=1)
        
        grid = pv.PolyData(points)
        grid["velocity"] = velocity
        
        grid.set_active_vectors("velocity")
        
        grid.save(f"{filename_prefix}_{b}.vtp")