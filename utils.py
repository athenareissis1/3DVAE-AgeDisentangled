import os
import yaml
import trimesh
import torch
import pickle

import matplotlib.cm
import torch_geometric.transforms

import networkx as nx
import numpy as np
from collections import Counter
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def prepare_sub_folder(output_directory):
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print(f"Creating directory: {checkpoint_directory}")
        os.makedirs(checkpoint_directory)
    return checkpoint_directory


def load_template(mesh_path, data_type):
    mesh = trimesh.load_mesh(mesh_path, 'ply', process=False)
    feat_and_cont = extract_feature_and_contour_from_colour(mesh, data_type)
    mesh_verts = torch.tensor(mesh.vertices, dtype=torch.float,
                              requires_grad=False)
    face = torch.from_numpy(mesh.faces).t().to(torch.long).contiguous()
    mesh_colors = torch.tensor(mesh.visual.vertex_colors,
                               dtype=torch.float, requires_grad=False)
    data = Data(pos=mesh_verts, face=face, colors=mesh_colors,
                feat_and_cont=feat_and_cont)
    data = torch_geometric.transforms.FaceToEdge(False)(data)
    data.laplacian = torch.sparse_coo_tensor(
        *get_laplacian(data.edge_index, normalization='rw'))
    return data


def extract_feature_and_contour_from_colour(colored, data_type):
    # assuming that the feature is colored in red and its contour in black
    if isinstance(colored, torch_geometric.data.Data):
        assert hasattr(colored, 'colors')
        colored_trimesh = torch_geometric.utils.to_trimesh(colored)
        colors = colored.colors.to(torch.long).numpy()
    elif isinstance(colored, trimesh.Trimesh):
        colored_trimesh = colored
        colors = colored_trimesh.visual.vertex_colors
    else:
        raise NotImplementedError

    graph = nx.from_edgelist(colored_trimesh.edges_unique)
    one_rings_indices = [list(graph[i].keys()) for i in range(len(colors))]

    features = {}
    for index, (v_col, i_ring) in enumerate(zip(colors, one_rings_indices)):
        if str(v_col) not in features:
            features[str(v_col)] = {'feature': [], 'contour': []}

        if is_contour(colors, index, i_ring):
            features[str(v_col)]['contour'].append(index)
        else:
            features[str(v_col)]['feature'].append(index)

    # certain vertices on the contour have interpolated colours ->
    # assign them to adjacent region
    elem_to_remove = []
    for key, feat in features.items():
        if len(feat['feature']) < 3:
            elem_to_remove.append(key)
            for idx in feat['feature']:
                counts = Counter([str(colors[ri])
                                  for ri in one_rings_indices[idx]])
                most_common = counts.most_common(1)[0][0]
                if most_common == key:
                    break
                features[most_common]['feature'].append(idx)
                features[most_common]['contour'].append(idx)
    for e in elem_to_remove:
        features.pop(e, None)

    # reorder features to match the order of the non-combined data
    if 'combined' in data_type:
        order = [1, 0, 3, 6, 4, 2, 8, 7, 5]
        current_keys = list(features.keys())
        reordered_keys = [current_keys[i] for i in order if i < len(current_keys)]
        features = {key: features[key] for key in reordered_keys}

    # with b map
    # 0=eyes, 1=ears, 2=sides, 3=neck, 4=back, 5=mouth, 6=forehead,
    # 7=cheeks 8=cheekbones, 9=forehead, 10=jaw, 11=nose
    # key = list(features.keys())[11]
    # feature_idx = features[key]['feature']
    # contour_idx = features[key]['contour']

    # find surroundings
    # all_distances = self.compute_minimum_distances(
    #     colored.vertices, colored.vertices[contour_idx]
    # )
    # max_distance = max(all_distances)
    # all_distances[feature_idx] = max_distance
    # all_distances[contour_idx] = max_distance
    # threshold = 0.005
    # surrounding_idx = np.squeeze(np.argwhere(all_distances < threshold))
    # colored.visual.vertex_colors[surrounding_idx] = [0, 0, 0, 255]
    # colored.show()
    return features


def is_contour(colors, center_index, ring_indices):
    center_color = colors[center_index]
    ring_colors = [colors[ri] for ri in ring_indices]
    for r in ring_colors:
        if not np.array_equal(center_color, r):
            return True
    return False


def to_torch_sparse(spmat):
    return torch.sparse_coo_tensor(
        torch.LongTensor([spmat.tocoo().row, spmat.tocoo().col]),
        torch.FloatTensor(spmat.tocoo().data), torch.Size(spmat.tocoo().shape))


def batch_mm(sparse, matrix_batch):
    """
    :param sparse: Sparse matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns (b, n, k) -> (n, b, k) -> (n, b*k)
    matrix = matrix_batch.transpose(0, 1).reshape(sparse.shape[1], -1)

    # And then reverse the reshaping.
    return sparse.mm(matrix).reshape(sparse.shape[0],
                                     batch_size, -1).transpose(1, 0)


def errors_to_colors(values, min_value=None, max_value=None, cmap=None):
    device = values.device
    min_value = values.min() if min_value is None else min_value
    max_value = values.max() if max_value is None else max_value
    if min_value != max_value:
        values = (values - min_value) / (max_value - min_value)

    cmapper = matplotlib.cm.get_cmap(cmap)
    values = cmapper(values.cpu().detach().numpy(), bytes=True)
    return torch.tensor(values[:, :, :3]).to(device)


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(
                      os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

def age_per_feature_new_ages(z, new_ages, swapped_feature, latent_size, age_latent_size, latent_regions, bs):

    latent_per_feature_size = (latent_size - age_latent_size) // len(latent_regions)

    age_latents = z[:, -age_latent_size:]
    swapped_latent_index = latent_regions[swapped_feature][0] // latent_per_feature_size

    # make a gt_age matrix of size [16,age_latent_size]
    gt_feature_ages = torch.zeros([bs ** 2, age_latent_size],
                                    device=age_latents.device,
                                    dtype=age_latents.dtype)

    # make new gt_age matrix with swapped feature ages
    for j in range(bs):
        for i in range(bs):
            gt_feature_ages[i * bs + j, ::] = new_ages[i, ::]
            if i != j:
                gt_feature_ages[i * bs + j, swapped_latent_index-1] = new_ages[j]

    # # try now for 45 age latets (or x age)
    # for j in range(bs):
    #     for i in range(bs):
    #         # Repeat each new_age value 5 times to fill the corresponding 5 latents
    #         gt_feature_ages[i * bs + j, ::] = new_ages[i, ::]
    #         if i != j:
    #             # Update the swapped latent index group (5 latents) with the new_age[j]
    #             start_idx = (swapped_latent_index - 1) * 5
    #             end_idx = start_idx + 5
    #             gt_feature_ages[i * bs + j, start_idx:end_idx] = new_ages[j].repeat(5)


    return gt_feature_ages



def modify_age_latent_based_on_gt(precomputed_storage_path, gt_age, min_delta, max_age):
    """
    Modify the age latent based on the ground truth age (gt_age).
    The new age is randomly selected from:
    - [0, gt_age - min_delta] or
    - [gt_age + min_delta, max_age]

    Args:
        gt_age (torch.Tensor): The ground truth age tensor (batch_size, 1).
        min_delta (float): The minimum difference for the modification.
        max_age (float): The maximum age value (normalized).

    Returns:
        torch.Tensor: The modified age latent tensor.
    """

    # option = 1 # 1: select from [0, gt_age - min_delta] or [gt_age + min_delta, max_age]
    option = 2 # 2: select from [0,max_age] excluding gt_age

    lower_range = (gt_age - min_delta).clamp(min=0.0)
    upper_range = (gt_age + min_delta).clamp(max=max_age)

    batch_size = gt_age.size(0)

    # Create a mask for valid ages
    all_ages = torch.arange(0, max_age + 1, device=gt_age.device).unsqueeze(0).repeat(batch_size, 1)

    if option == 1:
        valid_mask = (all_ages < lower_range.unsqueeze(1)) | (all_ages > upper_range.unsqueeze(1))
    elif option == 2:
        valid_mask = (all_ages != gt_age.unsqueeze(1))

    # Filter valid ages
    valid_ages = [all_ages[i][valid_mask[i]] for i in range(batch_size)]

    # Randomly select a valid age for each batch element
    modified_age_latent = torch.stack([
        valid_ages[i][torch.randint(0, len(valid_ages[i]), (1,))] for i in range(batch_size)
    ])

    # normalise the modified age latent

    storage_path = os.path.join(*precomputed_storage_path)
    try:
        with open(storage_path, 'rb') as file:
            age_mean, age_std = \
                pickle.load(file)
    except FileNotFoundError:
        print("Could not find normalise stats file")

    # normalise age
    modified_age_latent = (modified_age_latent - age_mean) / age_std

    return modified_age_latent
