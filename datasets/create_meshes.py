import numpy as np
import os, sys, h5py
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import open3d as o3d
import multiprocessing
import traceback


root = "/export/livia/home/vision/Abahri/projects/MATE/MATE/data/scanobjectnn/main_split"
h5 = h5py.File(os.path.join(root, 'training_objectdataset.h5'), 'r')
points = np.array(h5['data']).astype(np.float32)
labels = np.array(h5['label']).astype(int)
            
h5.close()

def create_mesh_from_point_cloud(point_cloud):
    
    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)   
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)) 
    
    # Create mesh using Poisson Surface Reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    return mesh

    
def save_meshes_to_h5(meshes, file_path):
    with h5py.File(file_path, 'w') as h5f:
        for i, mesh in enumerate(meshes):
            # Save vertices and faces to the HDF5 file
            h5f.create_dataset(f'mesh_{i}/vertices', data=np.asarray(mesh.vertices))
            h5f.create_dataset(f'mesh_{i}/faces', data=np.asarray(mesh.triangles))            
       
def create_mesh_from_point_cloud_and_save(point_clouds, addr_save):
            
    #meshes = []
    for i, point_cloud in tqdm(enumerate(point_clouds)):
        meshes = []
        mesh = create_mesh_from_point_cloud(point_cloud)  
        meshes.append(mesh)

        # Save all meshes to an HDF5 file
        save_meshes_to_h5(meshes, addr_save + f'meshes_{i}.h5')        
    
    
    
################################ convert mesh to ply and save    
def load_meshes_from_h5(file_path):
    meshes = []
    with h5py.File(file_path, 'r') as h5f:
        for key in h5f.keys():
            vertices = np.array(h5f[f'{key}/vertices'])
            faces = np.array(h5f[f'{key}/faces'])
            
            # Create an Open3D TriangleMesh object
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            
            # Compute normals
            mesh.compute_vertex_normals()
            meshes.append(mesh)
    return meshes       


def save_meshes_as_ply(meshes, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for i, mesh in enumerate(meshes):
        ply_filename = os.path.join(output_dir, f'mesh_{i}.ply')
        o3d.io.write_triangle_mesh(ply_filename, mesh)
        print(f"Saved mesh {i} to {ply_filename}")
################################ convert mesh to ply and save    

        
if __name__ == "__main__":
    #addr_save = "/export/livia/home/vision/Abahri/projects/MATE/MATE/data/scanobjectnn/main_split/test_meshes/"
    addr_save = "/projets/ABahri/MATE/test_meshes/"   
    create_mesh_from_point_cloud_and_save(points, addr_save)          