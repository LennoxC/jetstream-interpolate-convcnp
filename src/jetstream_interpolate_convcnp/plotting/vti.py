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
        
        # Create grid
        grid = pv.ImageData()
        grid.dimensions = (X, Y, Z)
        grid.spacing = (1, 1, 1)
        grid.origin = (0, 0, 0)
        
        # Flatten in Fortran order (VERY IMPORTANT for VTK)
        velocity = np.stack([u, v, np.zeros_like(u)], axis=-1)
        velocity = velocity.reshape(-1, 3, order="F")
        
        grid["velocity"] = velocity
        
        grid.save(f"{filename_prefix}_{b}.vti")