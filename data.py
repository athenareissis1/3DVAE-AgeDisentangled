"""
this file performs tasks on the dataset being used 

- edits the dataset to remove specified vertices
- Removes files that contain a specified keyword in their name
- finds the x, y, z coordinates of specified vertices
- calculates measurements and propotions of dataset from coordiates

"""

import trimesh
import numpy as np
import os
import csv
import pandas as pd
import torch
import pymeshlab as ml
import json


def change_file_names(dataset_folder_path):
    # for each file in dataset folder, the file name is sf_xxx.obj where xxx is a number. We want to change it to f_xxx.obj
    count = 1
    for file_name in os.listdir(dataset_folder_path):
        print(f"Count: ({count})")
        count += 1
        old_file_path = os.path.join(dataset_folder_path, file_name)
        new_file_name = 'f_' + file_name.split('_')[1]
        new_file_path = os.path.join(dataset_folder_path, new_file_name)
        os.rename(old_file_path, new_file_path)


def create_subset_dataset(dataset_folder_path, dataset_metadata_path):
    """
    Creates a subset of the dataset by copying files corresponding to ages 0-17 years to a new folder.

    Parameters:
        dataset_folder_path (str): Path to the original dataset folder containing mesh files.
        dataset_metadata_path (str): Path to the CSV file containing metadata with 'id' and 'AgeYears'
    """

    metadata = pd.read_csv(dataset_metadata_path)
    valid_ids = metadata[(metadata['AgeYears'] >= 0) & (metadata['AgeYears'] <= 17)]['id'].astype(str).tolist()
    subset_folder_path = dataset_folder_path + "_subset_0_17"

    if not os.path.exists(subset_folder_path):
        os.makedirs(subset_folder_path) 

    for file_name in os.listdir(dataset_folder_path):
        if file_name.endswith(('.ply', '.obj', '.stl')) and not file_name.startswith('._'):
            file_id = file_name.split('.')[0]
            if file_id in valid_ids:
                original_file_path = os.path.join(dataset_folder_path, file_name)
                subset_file_path = os.path.join(subset_folder_path, file_name)
                try:
                    with open(original_file_path, 'rb') as src_file:
                        with open(subset_file_path, 'wb') as dst_file:
                            dst_file.write(src_file.read())
                except Exception as e:
                    print(f"Failed to copy {file_name}: {e}")


def delete_vertices_for_single_mesh(mesh_file, vertex_indices_to_remove, dataset, count, test=True):
    """
    Opens a mesh file, deletes vertices based on the given indices, and saves the modified mesh.

    Parameters:
        mesh_file (str): Path to the input mesh file.
        vertex_indices_to_remove (list or numpy.ndarray): Indices of vertices to remove.

    Returns:
        None
    """

    print(f"Count: ({count})")
    # Load the mesh
    mesh = trimesh.load(mesh_file)
    # take the last part of the path after the last '/' and save it as mesh_file_name
    mesh_file_name = os.path.basename(mesh_file)

    if not isinstance(vertex_indices_to_remove, np.ndarray):
        vertex_indices_to_remove = np.array(vertex_indices_to_remove)

    # Create a mask for vertices to keep
    keep_mask = np.ones(len(mesh.vertices), dtype=bool)
    # keep_mask[vertex_indices_to_remove] = False
    try:
        keep_mask[vertex_indices_to_remove] = False
    except IndexError:
        print(f"{mesh_file}: Some vertex indices are out of bounds. Skipping this mesh.")
        return

    # Filter vertices and faces
    new_vertices = mesh.vertices[keep_mask]
    
    # Update face indices to only use retained vertices
    old_to_new_indices = np.full(len(mesh.vertices), -1)
    old_to_new_indices[keep_mask] = np.arange(len(new_vertices))

    new_faces = []
    for face in mesh.faces:
        if all(keep_mask[face]):  # Keep faces with all vertices retained
            new_faces.append(old_to_new_indices[face])

    new_faces = np.array(new_faces)

    # Create the new mesh
    modified_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

    if test:
        if not os.path.exists("remove_triangle_tests"):
            os.makedirs("remove_triangle_tests")
        modified_mesh.export(f"remove_triangle_tests/{mesh_file_name}")
    else:
        # modified_mesh_path = os.path.join(f"/raid/compass/athena/data/smooth_{dataset}/{mesh_file_name}")
        # modified_mesh.export(modified_mesh_path)
        modified_mesh.export(mesh_file)
        print(f"Modified mesh saved to: {mesh_file}")


# Process the first 40 meshes in the specified folder
def delete_vertices_for_all_to_new_folder(folder_path, vertex_indices_to_remove, dataset):
    """
    Processes the first `max_files` meshes in a folder, removing specified vertices.

    Parameters:
        folder_path (str): Path to the folder containing mesh files.
        vertex_indices_to_remove (list or numpy.ndarray): Indices of vertices to remove.
        max_files (int): Maximum number of files to process.

    Returns:
        None
    """
    mesh_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.endswith(('.ply', '.obj', '.stl'))
    ]

    count = 1

    for i, mesh_file in enumerate(mesh_files):
        base = os.path.basename(mesh_file)

        if base == "f_1.obj":
            continue
        else:
            delete_vertices_for_single_mesh(mesh_file, vertex_indices_to_remove, dataset, count, test=False)
            count += 1

    
def verticies_difference(template_file, mesh_file):
    """
    Check the number assigned to the vertices in two meshes and print the difference.

    Parameters:
        template_file (str): Path to the template mesh file.
        mesh_file (str): Path to the mesh file to compare.

    Returns:
        list: List of vertex indices that are in `mesh` but not in `template`.
    """
    # Load the meshes
    template = trimesh.load(template_file)
    mesh = trimesh.load(mesh_file)

    # Extract vertices
    template_vertices = template.vertices
    mesh_vertices = mesh.vertices
    print(f"Template vertices: {len(template_vertices)}, Mesh vertices: {len(mesh_vertices)}")

    # Find the difference in vertices
    template_set = set(map(tuple, template_vertices))
    mesh_set = set(map(tuple, mesh_vertices))

    extra_vertices = list(mesh_set - template_set)

    # Find the indices of the extra vertices in the mesh
    extra_indices = [i for i, vertex in enumerate(mesh_vertices) if tuple(vertex) in extra_vertices]

    print(f"Extra vertices in mesh: {extra_indices}")
    return extra_indices


def remove_files_with_keyword(directory):
    """
    Removes files that contain the specified keyword in their name within the given directory.

    Parameters:
        directory (str): Path to the directory.
        keyword (str): Keyword to search for in file names.
    """

    # keyword = "test_data_modified"
    keyword = 'sf'

    count = 1

    for filename in os.listdir(directory):
        print(f"Count: ({count})")
        count += 1
        if keyword not in filename:
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")
        else:
            print(f"Keeping file: {filename}")


def get_xyz_coordinates(mesh, template_faces, triangle_index, u, v):
    """
    Gets the XYZ coordinates on the mesh that correspond to the triangle number and 2D barycentric coordinates.

    Parameters:
        mesh (trimesh.Trimesh): The mesh object.
        triangle_index (int): The index of the triangle.
        u (float): The first barycentric coordinate.
        v (float): The second barycentric coordinate.

    Returns:
        numpy.ndarray: The XYZ coordinates.
    """
    # try:
    #     triangle = mesh.faces[triangle_index]
    #     skip_mesh = False
    # except (IndexError, AttributeError):
    #     # print(f"Error getting faces on mesh {mesh.metadata['file_name']}.")
    #     with open(os.path.join("measurements", "missing_faces.txt"), "a") as file:
    #         file.write(f"{mesh.metadata['file_name']}\n")
    #     skip_mesh = True
    #     return None, skip_mesh

    # vertices = mesh.vertices[triangle]
    # w = 1 - u - v
    # xyz = u * vertices[0] + v * vertices[1] + w * vertices[2]
    # return xyz, skip_mesh

    if isinstance(mesh, trimesh.Trimesh):
        try:
            triangle = mesh.faces[triangle_index]
            skip_mesh = False
        except (IndexError, AttributeError):
            with open(os.path.join("measurements", "missing_faces.txt"), "a") as file:
                file.write(f"{mesh.metadata['file_name']}\n")
            skip_mesh = True
            return None, skip_mesh

        vertices = mesh.vertices[triangle]
    elif isinstance(mesh, torch.Tensor):
        if mesh.ndim != 2 or mesh.shape[1] != 3:
            raise ValueError("Input mesh must be a 2D array with shape (n, 3) representing 3D coordinates.")
        if triangle_index >= len(template_faces):
            raise IndexError("Triangle index out of bounds for the given faces.")
        triangle = template_faces[triangle_index]
        vertices = mesh[triangle].cpu().numpy()
        skip_mesh = False
    else:
        raise TypeError("Input mesh must be either a trimesh.Trimesh object or a torch.Tensor.")

    w = 1 - u - v
    xyz = u * vertices[0] + v * vertices[1] + w * vertices[2]
    return xyz, skip_mesh

def calculate_distances_in_folder(folder_path, template_path, reconstructions, mesh_names, dataset_type, output_directory):
    """
    Calculates the Euclidean distances between specified points on each mesh in a folder.

    Parameters:
        folder_path (str): Path to the folder containing mesh files.

    Returns:
        dict: A dictionary with mesh file names as keys and their corresponding distances as values.
    """

    # if "unified" in folder_path:
    if dataset_type == 'combined' or dataset_type == 'friday_combined':
        gn = (26021, 0.0978797972202301, 0.736789166927337)
        go_left = (31596, 0.14787405729293823, 0.204847782850265)
        go_right = (13220, 0.4579116106033325, 0.14548911154270172)
        n = (9654, 0.34022921323776245, 0.0014109929325059056)
        sto = (17479, 0.5679528117179871, 0.09499986469745636)
        zy_left = (30477, 0.23432278633117676, 0.5406232476234436)
        zy_right = (3468, 0.6796667575836182, 0.24974024295806885) 

    # else if dataset_type = "not_combined":
    else:
        gn = (53157, 0.3662373423576355, 0.5330255627632141)
        go_left = (54815, 0.35922741889953613, 0.6188949346542358)
        go_right = (51426, 0.8455120325088501, 0.02441496029496193)
        n = (16123, 0.35262376070022583, 0.08649962395429611)
        sto = (16270, 0.6295228004455566, 0.024552660062909126)
        zy_left = (43798, 0.35418692231178284, 0.16726090013980865)
        zy_right = (32542, 0.06230044364929199, 0.20505580306053162) 

    point_pairs = [(n, gn), (n, sto), (sto, gn), (zy_right, zy_left), (go_right, go_left)]
    distance_names = ["n-gn", "n-sto", "sto-gn", "zy_right-zy_left", "go-right-go-left"]

    if reconstructions is None:
        mesh_files = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.endswith('.obj')
        ]
        template_faces = None
    else:
        mesh_files = reconstructions
        template_faces = trimesh.load_mesh(template_path).faces

    distances = {}
    for i in range(len(mesh_files)):
        if reconstructions is None:
            mesh = trimesh.load(mesh_files[i])
        else:
            mesh = mesh_files[i]
        mesh_distances = {}
        for name, (point1, point2) in zip(distance_names, point_pairs):
            tri1, u1, v1 = point1
            tri2, u2, v2 = point2
            coord1, skip_mesh_1 = get_xyz_coordinates(mesh, template_faces, tri1, u1, v1)
            coord2, skip_mesh_2 = get_xyz_coordinates(mesh, template_faces, tri2, u2, v2)
            if skip_mesh_1 or skip_mesh_2:
                distance = ""
            else:
                distance = np.linalg.norm(coord1 - coord2)
            mesh_distances[name] = distance
        if reconstructions == None:
            distances[os.path.basename(mesh_files[i])] = mesh_distances
        else:
            distances[mesh_names[i]] = mesh_distances

    # save disctionary as cvs and save file to a folder
    output_csv_path = os.path.join(output_directory, f"{dataset_type}_distances.csv")
    with open(output_csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['Mesh File'] + distance_names
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for mesh_file, mesh_distances in distances.items():
            row = {'Mesh File': mesh_file}
            row.update(mesh_distances)
            writer.writerow(row)

    return distances


def add_proportions_age_gender_to_csv(folder_path, dataset_type, output_directory, dataset_metadata_path):
    """
    Reads the CSV file 'measurements/combined_distances.csv', calculates proportions for each row based on
    specified proportion names, and adds the proportions to the CSV file in the format 1:x. Also adds age information
    from 'preprocessing_data/combined_datasets_ages.csv'.
    """
    proportions_names = ["n-sto:n-gn", "n-sto:sto-gn", "sto-gn:n-gn", "zy_right-zy_left:go-right-go-left"]

    metadata_df = pd.read_csv(dataset_metadata_path)

    input_csv_path = os.path.join(output_directory, f"{dataset_type}_distances.csv")
    output_csv_path = os.path.join(output_directory, f"{dataset_type}_distances_with_proportions_age_gender.csv")

    # Read the CSV files
    df = pd.read_csv(input_csv_path)

    # Calculate proportions for each row and format them as 1:x
    for proportion_name in proportions_names:
        first, second = proportion_name.split(':')
        df[proportion_name] = df[first] / df[second]
        df[proportion_name] = df[proportion_name].apply(lambda x: f"{x:.3f}")

    if folder_path != None:
        # add age and gender column to the dataframe
        ages_dict = pd.Series(metadata_df.AgeYears.values, index=metadata_df.id).to_dict()
        if dataset_type == "combined":
            df['age'] = df['Mesh File'].str.replace(".obj", "").astype(int).map(ages_dict)
            gender_dict = pd.Series(metadata_df.gender.values, index=metadata_df.id).to_dict()
            df['gender'] = df['Mesh File'].str.replace(".obj", "").astype(int).map(gender_dict)
        else:
            df['age'] = df['Mesh File'].str.replace(".obj", "").str.replace("_", "").map(ages_dict)
            gender_dict = pd.Series(metadata_df.gender.values, index=metadata_df.id).to_dict()
            df['gender'] = df['Mesh File'].str.replace(".obj", "").str.replace("_", "").map(gender_dict)
    else:
        # Extract age from Mesh File and update Mesh File column
        df['age'] = df['Mesh File'].apply(lambda x: x.split('_')[1])
        df['Mesh File'] = df['Mesh File'].apply(lambda x: x.split('_')[0])
        
        # Map gender using the updated Mesh File column
        gender_dict = pd.Series(metadata_df.gender.values, index=metadata_df.id).to_dict()
        if dataset_type == "combined":
            df['gender'] = df['Mesh File'].apply(lambda x: gender_dict.get(int(x), 'Unknown'))
        else:
            df['gender'] = df['Mesh File'].apply(lambda x: gender_dict.get(x, 'Unknown'))


    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv_path, index=False)

    print(f"Proportions and age added and saved to: {output_csv_path}")


def distance_proportion_averages(dataset_type, output_directory):
    """
    Reads the CSV file 'measurements/combined_distances_with_proportions_and_age.csv', calculates average proportions
    for each unique integer age, and saves the results to a new CSV file.
    """
    proportions_names = ["n-sto:n-gn", "n-sto:sto-gn", "sto-gn:n-gn", "zy_right-zy_left:go-right-go-left"]
    input_csv_path = os.path.join(output_directory, f"{dataset_type}_distances_with_proportions_age_gender.csv")
    output_csv_path = os.path.join(output_directory, f"{dataset_type}_proportion_averages.csv")

    # Read the CSV file
    df = pd.read_csv(input_csv_path)

    # Fill NaN and infinite values with a specific value (e.g., 0)
    df['age'] = df['age'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Round down each value in the 'age' column to the nearest integer
    df['age'] = df['age'].apply(np.floor).astype(int)

    # Create a new DataFrame to store the average proportions per age
    avg_df = pd.DataFrame(columns=['age'] + ['gender'] + proportions_names)

    genders = ['male', 'female']

    # Calculate average proportions for each unique integer age for each gender
    for age in df['age'].unique():
        for gender in genders:
            age_group = df[df['age'] == age]
            gender_age_group = age_group[age_group['gender'] == gender]
            avg_proportions = {'age': age, 'gender': gender}
            for proportion_name in proportions_names:
                first, second = proportion_name.split(':')
                avg_first = gender_age_group[first].mean()
                avg_second = gender_age_group[second].mean()
                avg_proportions[proportion_name] = f"{avg_first / avg_second:.3f}"
            avg_df = pd.concat([avg_df, pd.DataFrame([avg_proportions])], ignore_index=True)

    # Order the DataFrame by the 'age' column in ascending order
    avg_df = avg_df.sort_values(by='age')

    # Save the new DataFrame to a CSV file
    avg_df.to_csv(output_csv_path, index=False)

    print(f"Average proportions per age saved to: {output_csv_path}")

def smooth_mesh(mesh_file, file_name='1216.obj', smooth=1, test=True):

    # file_name = 'f_1520.obj'
    # smooth = 1

    ms = ml.MeshSet()
    ms.load_new_mesh(mesh_file)

    ms.apply_coord_laplacian_smoothing(
        stepsmoothnum=smooth,   # Smoothing steps
        boundary=True,          # Smooth boundary vertices as well
        cotangentweight=True,   # Use cotangent weighting
        selected=False          # Affects entire mesh
    )

    folder = os.path.dirname(mesh_file)
    new_file_name = 's' + file_name
    new_file_path = os.path.join(folder, new_file_name)


    if test:
        ms.save_current_mesh(f"smooth_tests/{file_name}")
    else:
        # check if the folder exists, if not create it
        # if not os.path.exists(os.path.join(dataset_path, f'smooth_{dataset}')):
        #     os.makedirs(os.path.join(dataset_path, f'smooth_{dataset}'))
        # ms.save_current_mesh(os.path.join(dataset_path, f'smooth_{dataset}', f"{file_name}"))
        # ms.save_current_mesh(os.path.join(folder_path, file_name))
        ms.save_current_mesh(new_file_path)
        print(f"Smoothed mesh saved to: {new_file_path}")

def smooth_dataset(folder_path, dataset_metadata_path, dataset):

    dataset_folder_path = folder_path
    mesh_files = [
        # os.path.join(dataset_folder_path, f) for f in os.listdir(dataset_folder_path)
        # if f.endswith(('.ply', '.obj', '.stl'))]
        os.path.join(dataset_folder_path, f) for f in os.listdir(dataset_folder_path)
        if f.endswith(('.obj')) and not f.startswith('._')]

    metadata = pd.read_csv(dataset_metadata_path)[['id', 'Dataset']]

    count = 1

    for i, mesh_file in enumerate(mesh_files):

        print(f"Count: ({count})")
        count += 1

        file_name = os.path.basename(mesh_file)

        # if file_name contqains 'sf' then skip
        if 'sf' in file_name:
            print(f"Skipping {file_name} because it contains 'sf'.")
            continue

        # origin = metadata.loc[metadata['id'] == int(file_name.split('.')[0]), 'Dataset'].values[0]
        origin = metadata.loc[metadata['id'] == file_name.split('.')[0], 'Dataset'].values[0]

        if origin == 'LYHM':
            smooth_factor = 2
        elif origin == 'LSFM':
            smooth_factor = 1
        elif origin == 'MimicMe':
            smooth_factor = 1
        elif origin == 'FaceScape':
            smooth_factor = 0
        elif origin == 'Paeds':
            smooth_factor = 0

        smooth_mesh(mesh_file, file_name=file_name, smooth=smooth_factor, test=False)

def obj_to_ply(obj_folder_path, ply_folder_path, metadata_path):
    """
    Converts all mesh files in obj_folder_path from OBJ format to PLY format by loading and exporting the mesh,
    but only for meshes with ages between 0-17 years.

    Parameters:
        obj_folder_path (str): Path to the input OBJ mesh files.
        ply_folder_path (str): Path to the output PLY mesh files.
        metadata_path (str): Path to the CSV file containing metadata with 'id' and 'AgeYears' columns.

    """
    # Load metadata and filter for ages between 0-17
    metadata = pd.read_csv(metadata_path)
    valid_ids = metadata[(metadata['AgeYears'] >= 0) & (metadata['AgeYears'] <= 17)]['id'].astype(str).tolist()

    # Ensure the output folder exists
    if not os.path.exists(ply_folder_path):
        os.makedirs(ply_folder_path)

    converted_count = 0  # Counter for converted files

    # Iterate through all OBJ files in the input folder
    for file_name in os.listdir(obj_folder_path):
        if file_name.endswith('.obj'):
            file_id = file_name.replace('.obj', '')
            if file_id in valid_ids:
                obj_file_path = os.path.join(obj_folder_path, file_name)
                ply_file_path = os.path.join(ply_folder_path, file_name.replace('.obj', '.ply'))

                try:
                    # Load the OBJ file as a mesh
                    mesh = trimesh.load(obj_file_path)

                    # Export the mesh to PLY format
                    mesh.export(ply_file_path)

                    converted_count += 1
                    print(f"COUNT: {converted_count}")
                    # print(f"Converted {file_name} to {ply_file_path} in PLY format.")
                except Exception as e:
                    print(f"Failed to convert {file_name}: {e}")

    print(f"Total files converted: {converted_count}")

def dataset_split_obj_to_ply(dataset_split_path):
    # read dataseet split json file and recreate it replacing all .obj with .ply and save it as dataset_split_ply.json
    with open(dataset_split_path) as f:
        dataset_split = json.load(f)
    dataset_split_ply = {}
    for split, files in dataset_split.items():
        dataset_split_ply[split] = [file.replace('.obj', '.ply') for file in files]
    new_split_path = dataset_split_path.replace('.json', '_ply.json')
    # open for writing (create or truncate)
    with open(new_split_path, 'w') as f:
        json.dump(dataset_split_ply, f, indent=4)

        
# Example usage
if __name__ == "__main__":

    # folder_path = "/raid/compass/athena/data/"

    # dataset = "unified_normals_dataset"
    # dataset_type = "combined"
    dataset = "friday_unified_meshes_subset_0_17"
    dataset_type = "friday_unified"
    metadata_filename = "friday_all_datasets.csv"
    reconstructions = None
    mesh_names = None
    template_path = None
    output_directory = "measurements"

    # dataset = "DATA_BABIES_FACES" 
    # dataset_type = "not_combined"
    # reconstructions = None
    # mesh_names = None
    # template_path = None
    # output_directory = "measurements"

    dataset_folder_path = f"/raid/compass/athena/data/{dataset}"  # Replace with your folder path
    dataset_metadata_path = f"preprocessing_data/{metadata_filename}"
    dataset_split_path = f"precomputed/data_split_{dataset_type}.json"  # Replace with your desired output folder path for split dataset

    # create_subset_dataset(dataset_folder_path, dataset_metadata_path)
        
    # vertices_to_remove = [6945, 6946, 16087]   # Replace with the indices of vertices to remove # was [6945, 6946, 16087] 

    # delete_vertices_for_all_to_new_folder(dataset_folder_path, vertices_to_remove, dataset)
    # delete_vertices_for_single_mesh(f"/raid/compass/athena/data/{dataset}/f_2.obj", vertices_to_remove, dataset)
    # verticies_difference("preprocessing_data/face_combined_9_features.ply", f"/raid/compass/athena/data/{dataset}/f_2.obj")
    # verticies_difference("/raid/compass/athena/data/unified_normals_dataset/1.obj", f"/raid/compass/athena/data/{dataset}/f_1.obj")

    # remove_files_with_keyword(dataset_folder_path)

    # smooth_mesh(dataset_folder_path)
    # smooth_dataset(dataset_folder_path, dataset_metadata_path, dataset)

    # change_file_names(dataset_folder_path)

    # ply_folder_path = "/raid/compass/athena/data/PLY_friday_unified_meshes_subset_0_17"
    # obj_to_ply(dataset_folder_path, ply_folder_path, dataset_metadata_path)
    dataset_split_obj_to_ply(dataset_split_path)

    # calculate_distances_in_folder(dataset_folder_path, template_path, reconstructions, mesh_names, dataset_type, output_directory)
    # add_proportions_age_gender_to_csv(dataset_folder_path, dataset_type, output_directory, dataset_metadata_path)
    # distance_proportion_averages(dataset_type, output_directory)

