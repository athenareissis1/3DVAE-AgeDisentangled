import os
import yaml
import trimesh
import torch

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


def load_template(mesh_path, attribute_to_remove=None):
    mesh = trimesh.load_mesh(mesh_path, 'ply', process=False)
    feat_and_cont = extract_feature_and_contour_from_colour(mesh)

    mask_save = np.ones(mesh.vertices.shape[0], dtype=bool)
    if attribute_to_remove != 'none':
        a2r = attribute_to_remove
        feat_and_cont[a2r]['feature'] += feat_and_cont[a2r]['contour']
        feat_and_cont[a2r]['feature'].sort()
        mesh_verts, mesh_faces, mask_save = remove_mesh_vertices(
            mesh.vertices, mesh.faces, feat_and_cont[a2r]['feature']
        )
        mesh = trimesh.Trimesh(
            mesh_verts, mesh_faces,
            vertex_colors=mesh.visual.vertex_colors[mask_save],
            process=False)
        feat_and_cont = extract_feature_and_contour_from_colour(mesh)

    mesh_verts = torch.tensor(mesh.vertices, dtype=torch.float,
                              requires_grad=False)
    face = torch.from_numpy(mesh.faces).t().to(torch.long).contiguous()
    mesh_colors = torch.tensor(mesh.visual.vertex_colors,
                               dtype=torch.float, requires_grad=False)
    data = Data(pos=mesh_verts, face=face, colors=mesh_colors,
                feat_and_cont=feat_and_cont, mask_verts=mask_save)
    data = torch_geometric.transforms.FaceToEdge(False)(data)
    data.laplacian = torch.sparse_coo_tensor(
        *get_laplacian(data.edge_index, normalization='rw'))
    return data


def extract_feature_and_contour_from_colour(colored):
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



def remove_mesh_vertices(v_original, f_original, v_idxs_to_remove):
    keep_v_mask = np.ones(v_original.shape[0], dtype=bool)
    keep_v_mask[v_idxs_to_remove] = 0

    # remove faces containing a vertex that needs to be removed
    remove_f_idxs = []
    for v_idx in v_idxs_to_remove:
        indices = np.argwhere(f_original == v_idx)[:, 0]
        remove_f_idxs.extend(indices.tolist())
    remove_f_idxs = sorted(list(dict.fromkeys(remove_f_idxs)))
    keep_f_mask = np.ones(f_original.shape[0], dtype=bool)
    keep_f_mask[remove_f_idxs] = 0
    updated_faces = f_original[keep_f_mask, :]

    # remove vertices
    new_vertices = v_original[keep_v_mask]

    # update indices of faces with new vertex indexing
    val_old = np.argwhere(keep_v_mask == 1)[:, 0]
    val_new = np.arange(0, np.sum(keep_v_mask))
    arr = np.empty(updated_faces.max() + 1, dtype=val_new.dtype)
    arr[val_old] = val_new
    new_faces = arr[updated_faces]

    return new_vertices, new_faces, keep_v_mask
