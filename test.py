import os
import json
import pickle
import tqdm
import trimesh
import torch
import pytorch3d.loss
import random
import neptune

import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch.nn as nn
import torch.optim as optim
from torchvision.io import write_video
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, TensorDataset

from pytorch3d.renderer import BlendParams
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.loss.chamfer import _handle_pointcloud_input
from pytorch3d.ops.knn import knn_points

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score, mutual_info_score, accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

# from disentanglement_lib.evaluation.metrics import mig, dci, sap_score, beta_vae

import scipy
from scipy.stats import mode

from sap_score import sap
from evaluation_metrics import compute_all_metrics, jsd_between_point_cloud_sets

from data import calculate_distances_in_folder, add_proportions_age_gender_to_csv, distance_proportion_averages

import utils

class Tester:
    def __init__(self, model_manager, norm_dict,
                 train_load, val_load, test_load, out_dir, config, logging):
    
        self.log = neptune.init_run(
            project=logging['logging']['neptune_project'], 
            api_token=logging['logging']['neptune_api'],
            custom_run_id=os.path.basename(out_dir)
            )

        self._manager = model_manager
        self._manager.eval()
        self._device = model_manager.device
        self._norm_dict = norm_dict
        self._normalized_data = config['data']['normalize_data']
        self._out_dir = out_dir
        self._config = config
        self._logging = logging
        self._train_loader = train_load
        self._val_loader = val_load
        self._test_loader = test_load
        self._is_vae = self._manager.is_vae
        self.latent_stats = self.compute_latent_stats(train_load)
        self._data_type = config['data']['dataset_type'].split("_", 1)[1]

        self.coma_landmarks = [
            1337, 1344, 1163, 878, 3632, 2496, 2428, 2291, 2747,
            3564, 1611, 2715, 3541, 1576, 3503, 3400, 3568, 1519,
            203, 183, 870, 900, 867, 3536]
        self.uhm_landmarks = [
            10754, 10826, 9123, 10667, 19674, 28739, 4831, 19585,
            8003, 22260, 12492, 27386, 1969, 31925, 31158, 20963,
            1255, 9881, 32055, 45778, 5355, 27515, 18482, 33691]

    def __call__(self):
        self.set_renderings_size(512)
        self.set_rendering_background_color([1, 1, 1])

        # # Qualitative evaluations
        # if self._config['data']['swap_features']:
        #     self.latent_swapping(next(iter(self._test_loader)).x)
        # self.per_variable_range_experiments(use_z_stats=False)
        # self.random_generation_and_rendering(n_samples=16)
        # self.random_generation_and_save(n_samples=16)
        # self.interpolate() # not working
        # if self._config['data']['dataset_type'] == 'faces':
        #     self.direct_manipulation()

        # # Quantitative evaluation
        # self.evaluate_gen(self._test_loader, n_sampled_points=2048) # takes a while to run
        # recon_errors = self.reconstruction_errors(self._test_loader)
        # train_set_diversity = self.compute_diversity_train_set()
        # diversity = self.compute_diversity()
        # specificity = self.compute_specificity() # takes a while to run
        # metrics = {'recon_errors': recon_errors,
        #            'train_set_diversity': train_set_diversity,
        #            'diversity': diversity,
        #            'specificity': specificity}

        # outfile_path = os.path.join(self._out_dir, 'eval_metrics.json')
        # with open(outfile_path, 'w') as outfile:
        #     json.dump(metrics, outfile)

        # # TEST TO RUN (run all on val set then once model is finalised move to test set)
        # self.per_variable_range_experiments(use_z_stats=False)
        # self.random_generation_and_rendering(n_samples=16)

        if self._config['model']['age_disentanglement'] or self._config['model']['age_per_feature']:
            dataset = self._test_loader # self._val_loader,
            # self.dataset_split()
            # self.age_encoder_decoder_accuracy(self._train_loader, dataset)
            self.age_prediction_MLP(self._train_loader, dataset)
            self.age_latent_changing(dataset)
            self.tsne_visualization(self._train_loader, self._val_loader, self._test_loader)
            self.stats_tests_correlation(self._train_loader, self._val_loader, self._test_loader)
            self.proportions(dataset)
            self.plot_proportions()

            # relatives tests
            self.relatives_aging_diff_new()
            self.relatives_aging()
        
        self._manager.log_hyperparameters(self.log, self._config, self._logging)
        
        self.log.stop()
            

    def _unnormalize_verts(self, verts, dev=None):
        d = self._device if dev is None else dev
        return verts * self._norm_dict['std'].to(d) + \
            self._norm_dict['mean'].to(d)

    def set_renderings_size(self, size):
        self._manager.renderer.rasterizer.raster_settings.image_size = size

    def set_rendering_background_color(self, color=None):
        color = [1, 1, 1] if color is None else color
        blend_params = BlendParams(background_color=color)
        self._manager.default_shader.blend_params = blend_params
        self._manager.simple_shader.blend_params = blend_params

    def compute_latent_stats(self, data_loader):
        storage_path = os.path.join(self._out_dir, 'z_stats.pkl')
        try:
            with open(storage_path, 'rb') as file:
                z_stats = pickle.load(file)
        except FileNotFoundError:
            latents_list = []
            for data in tqdm.tqdm(data_loader):
                latents_list.append(self._manager.encode(
                    data.x.to(self._device)).detach().cpu())
            latents = torch.cat(latents_list, dim=0)
            z_means = torch.mean(latents, dim=0)
            z_stds = torch.std(latents, dim=0)
            z_mins, _ = torch.min(latents, dim=0)
            z_maxs, _ = torch.max(latents, dim=0)
            z_stats = {'means': z_means, 'stds': z_stds,
                       'mins': z_mins, 'maxs': z_maxs}

            with open(storage_path, 'wb') as file:
                pickle.dump(z_stats, file)
        return z_stats

    @staticmethod
    def string_to_color(rgba_string, swap_bw=True):
        rgba_string = rgba_string[1:-1]  # remove [ and ]
        rgb_values = rgba_string.split()[:-1]
        colors = [int(c) / 255 for c in rgb_values]
        if colors == [1., 1., 1.] and swap_bw:
            colors = [0., 0., 0.]
        return tuple(colors)

    def per_variable_range_experiments(self, z_range_multiplier=1,
                                       use_z_stats=True):
        
        # gives 54
        latent_size = self._manager.model_latent_size
        # gives 9
        age_latent_size = self._manager._age_latent_size
        # gives 45
        feature_latent_size = latent_size - age_latent_size
        # gives 5
        if self._config['data']['swap_features']:
            individual_feature_latent_size = feature_latent_size // len(self._manager.latent_regions)
        else:
            individual_feature_latent_size = 5 ## make dynamic 

        if self._is_vae and not use_z_stats and not (self._config['model']['age_disentanglement'] or self._config['model']['age_per_feature']):
            z_means = torch.zeros(latent_size)
            z_mins = -3 * z_range_multiplier * torch.ones(latent_size)
            z_maxs = 3 * z_range_multiplier * torch.ones(latent_size)
        elif self._is_vae and not use_z_stats and (self._config['model']['age_disentanglement'] or self._config['model']['age_per_feature']):
            latent_size = self._manager.model_latent_size   
            z_means = torch.zeros(latent_size)
            z_mins = -3 * z_range_multiplier * torch.ones(latent_size)
            z_maxs = 3 * z_range_multiplier * torch.ones(latent_size)
            z_mins[-age_latent_size:] = self.latent_stats['mins'][-age_latent_size:] * z_range_multiplier
            z_maxs[-age_latent_size:] = self.latent_stats['maxs'][-age_latent_size:] * z_range_multiplier
        else:
            z_means = self.latent_stats['means']
            z_mins = self.latent_stats['mins'] * z_range_multiplier
            z_maxs = self.latent_stats['maxs'] * z_range_multiplier

        # Create video perturbing each latent variable from min to max.
        # Show generated mesh and error map next to each other
        # Frames are all concatenated along the same direction. A black frame is
        # added before start perturbing the next latent variable

        ##### AGE TEST #####

        if self._config['model']['age_disentanglement'] or self._config['model']['age_per_feature']:

            # change only the age latent variables all at the same time from their own min to max

            n_steps_age = 17
            z_age_mins = z_mins[-age_latent_size:]
            z_age_maxs = z_maxs[-age_latent_size:]

            z = z_means.repeat(n_steps_age, 1)
            z[:, -age_latent_size:] = torch.stack(
                [torch.linspace(z_age_mins[i], z_age_maxs[i], n_steps_age) for i in range(age_latent_size)],
                dim=1).to(self._device)

            gen_verts = self._manager.generate(z.to(self._device))

            if self._normalized_data:
                gen_verts = self._unnormalize_verts(gen_verts)

            differences_from_first = self._manager.compute_vertex_errors(
                gen_verts, gen_verts[0].expand(gen_verts.shape[0], -1, -1))
            renderings = self._manager.render(gen_verts).detach().cpu()
            differences_renderings = self._manager.render(
                gen_verts, differences_from_first,
                error_max_scale=5).cpu().detach()
            frames = torch.cat([renderings, differences_renderings], dim=-1)
            all_frames_age = torch.cat([frames, torch.zeros_like(frames)[:2, ::]])
                
            
            file_path_age = os.path.join(self._out_dir, 'latent_exploration_all_age_latents_[min-max].mp4')
            write_video(file_path_age, all_frames_age.permute(0, 2, 3, 1) * 255, fps=4)
            self.log['test/latent_exploration_all_age_latents_[min-max].mp4'].upload(file_path_age)

            # changing all age latent values [0-17]

            storage_path = os.path.join(self._manager._precomputed_storage_path, f'normalise_age_{self._data_type}.pkl')
            with open(storage_path, 'rb') as file:
                age_train_mean, age_train_std = \
                    pickle.load(file)

            age_range = range(0, 18)
            normalized_age_range = [(age - age_train_mean) / age_train_std for age in age_range]

            z = z_means.repeat(len(normalized_age_range), 1)

            z[:, -age_latent_size:] = torch.tensor(normalized_age_range, device=self._device).unsqueeze(1).repeat(1, age_latent_size)

            gen_verts = self._manager.generate(z.to(self._device))

            if self._normalized_data:
                gen_verts = self._unnormalize_verts(gen_verts)

            differences_from_first = self._manager.compute_vertex_errors(
                gen_verts, gen_verts[0].expand(gen_verts.shape[0], -1, -1)
            )
            renderings = self._manager.render(gen_verts).detach().cpu()
            differences_renderings = self._manager.render(gen_verts, differences_from_first, error_max_scale=5).cpu().detach()

            frames = torch.cat([renderings, differences_renderings], dim=-1)

            file_path = os.path.join(self._out_dir, 'latent_exploration_all_age_latents_[0-17].mp4')
            write_video(file_path, frames.permute(0, 2, 3, 1) * 255, fps=4)
            self.log['test/latent_exploration_all_age_latents_[0-17].mp4'].upload(file_path)


        #### NOT AGE TESTS ####
        n_steps = 10
        all_frames, all_rendered_differences, max_distances = [], [], []
        all_renderings = []
        # change each latent variable one by one
        for i in tqdm.tqdm(range(z_means.shape[0])):
            z = z_means.repeat(n_steps, 1)
            z[:, i] = torch.linspace(
                z_mins[i], z_maxs[i], n_steps).to(self._device)

            gen_verts = self._manager.generate(z.to(self._device))

            if self._normalized_data:
                gen_verts = self._unnormalize_verts(gen_verts)

            differences_from_first = self._manager.compute_vertex_errors(
                gen_verts, gen_verts[0].expand(gen_verts.shape[0], -1, -1))
            max_distances.append(differences_from_first[-1, ::])
            renderings = self._manager.render(gen_verts).detach().cpu()
            all_renderings.append(renderings)
            differences_renderings = self._manager.render(
                gen_verts, differences_from_first,
                error_max_scale=5).cpu().detach()
            all_rendered_differences.append(differences_renderings)
            frames = torch.cat([renderings, differences_renderings], dim=-1)
            all_frames.append(
                torch.cat([frames, torch.zeros_like(frames)[:2, ::]]))

        file_path = os.path.join(self._out_dir, 'latent_exploration.mp4')
        write_video(file_path, torch.cat(all_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=4)
        self.log['test/latent_exploration.mp4'].upload(file_path)


        # Same video as before, but effects of perturbing each latent variables
        # are shown in the same frame. Only error maps are shown.
        grid_frames = []
        grid_nrows = 6
        if self._config['data']['swap_features']:
            z_size = self._config['model']['latent_size']
            grid_nrows = z_size // len(self._manager.latent_regions)

        stacked_frames = torch.stack(all_rendered_differences)

        # change order for age feature latent to be with feature latents
        if self._config['model']['age_per_feature']: # and self._config['data']['swap_features']:
            assert age_latent_size > 1

            new_order_indices = []
            for i in range(age_latent_size):
                new_order_indices.extend(range(i * individual_feature_latent_size, (i + 1) * individual_feature_latent_size))  # Add 5 feature tensors
                new_order_indices.append(feature_latent_size + i)  # Add the corresponding age tensor

            reordered_tensor = stacked_frames[new_order_indices]

            stacked_frames = reordered_tensor

        for i in range(stacked_frames.shape[1]):
            grid_frames.append(
                make_grid(stacked_frames[:, i, ::], padding=10,
                          pad_value=1, nrow=grid_nrows))
        save_image(grid_frames[-1],
                   os.path.join(self._out_dir, 'latent_exploration_tiled.png'))
        self.log['test/latent_exploration_tiled.png'].upload(os.path.join(self._out_dir, 'latent_exploration_tiled.png'))
        file_path = os.path.join(self._out_dir, 'latent_exploration_tiled.mp4')
        write_video(file_path, torch.stack(grid_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=1)
        self.log['test/latent_exploration_tiled.mp4'].upload(file_path)

        # Same as before, but only output meshes are used
        stacked_frames_meshes = torch.stack(all_renderings)
        grid_frames_m = []
        for i in range(stacked_frames_meshes.shape[1]):
            grid_frames_m.append(
                make_grid(stacked_frames_meshes[:, i, ::], padding=10,
                          pad_value=1, nrow=grid_nrows))
        file_path = os.path.join(self._out_dir, 'latent_exploration_outs_tiled.mp4')
        write_video(file_path, torch.stack(grid_frames_m, dim=0).permute(0, 2, 3, 1) * 255, fps=4)
        self.log['test/latent_exploration_outs_tiled.mp4'].upload(file_path)

        # Create a plot showing the effects of perturbing latent variables in
        # each region of the face
        df = pd.DataFrame(columns=['mean_dist', 'z_var', 'region'])
        df_row = 0
        for zi, vert_distances in enumerate(max_distances):
            for region, indices in self._manager.template.feat_and_cont.items():
                regional_distances = vert_distances[indices['feature']]
                mean_regional_distance = torch.mean(regional_distances)
                df.loc[df_row] = [mean_regional_distance.item(), zi, region]
                df_row += 1

        sns.set_theme(style="ticks")
        palette = {k: self.string_to_color(k) for k in
                   self._manager.template.feat_and_cont.keys()}
        grid = sns.FacetGrid(df, col="region", hue="region", palette=palette,
                             col_wrap=4, height=3)

        grid.map(plt.plot, "z_var", "mean_dist", marker="o")
        plt.savefig(os.path.join(self._out_dir, 'latent_exploration_split.svg'))
        self.log['test/latent_exploration_split.svg'].upload(os.path.join(self._out_dir, 'latent_exploration_split.svg'))

        sns.relplot(data=df, kind="line", x="z_var", y="mean_dist",
                    hue="region", palette=palette)
        plt.savefig(os.path.join(self._out_dir, 'latent_exploration.svg'))
        plt.savefig(os.path.join(self._out_dir, 'latent_exploration.png'))
        self.log['test/latent_exploration.png'].upload(os.path.join(self._out_dir, 'latent_exploration.png'))

    def random_latent(self, n_samples, z_range_multiplier=1):
        if self._is_vae:  # sample from normal distribution if vae
            z = torch.randn([n_samples, self._manager.model_latent_size])
        else:
            z_means = self.latent_stats['means']
            z_mins = self.latent_stats['mins'] * z_range_multiplier
            z_maxs = self.latent_stats['maxs'] * z_range_multiplier

            uniform = torch.rand([n_samples, z_means.shape[0]],
                                 device=z_means.device)
            z = uniform * (z_maxs - z_mins) + z_mins
        return z

    def random_generation(self, n_samples=16, z_range_multiplier=1,
                          denormalize=True):
        z = self.random_latent(n_samples, z_range_multiplier)
        gen_verts = self._manager.generate(z.to(self._device))
        if self._normalized_data and denormalize:
            gen_verts = self._unnormalize_verts(gen_verts)
        return gen_verts

    def random_generation_and_rendering(self, n_samples=16,
                                        z_range_multiplier=1):
        gen_verts = self.random_generation(n_samples, z_range_multiplier)
        renderings = self._manager.render(gen_verts).cpu()
        grid = make_grid(renderings, padding=10, pad_value=1)
        file_path = os.path.join(self._out_dir, 'random_generation.png')
        save_image(grid, file_path)
        self.log['test/random_generation'].upload(file_path)

    def random_generation_and_save(self, n_samples=16, z_range_multiplier=1):
        out_mesh_dir = os.path.join(self._out_dir, 'random_meshes')
        if not os.path.isdir(out_mesh_dir):
            os.mkdir(out_mesh_dir)

        gen_verts = self.random_generation(n_samples, z_range_multiplier)

        self.save_batch(gen_verts, out_mesh_dir)

    def save_batch(self, batch_verts, out_mesh_dir):
        for i in range(batch_verts.shape[0]):
            mesh = trimesh.Trimesh(
                batch_verts[i, ::].cpu().detach().numpy(),
                self._manager.template.face.t().cpu().numpy())
            mesh.export(os.path.join(out_mesh_dir, str(i) + '.ply'))

    def reconstruction_errors(self, data_loader):
        print('Compute reconstruction errors')
        data_errors = []
        for data in tqdm.tqdm(data_loader):
            if self._config['data']['swap_features']:
                data.x = data.x[self._manager.batch_diagonal_idx, ::]
            data = data.to(self._device)
            gt = data.x

            recon = self._manager.forward(data)[0]

            if self._normalized_data:
                gt = self._unnormalize_verts(gt)
                recon = self._unnormalize_verts(recon)

            errors = self._manager.compute_vertex_errors(recon, gt)
            data_errors.append(torch.mean(errors.detach(), dim=1))
        data_errors = torch.cat(data_errors, dim=0)
        return {'mean': torch.mean(data_errors).item(),
                'median': torch.median(data_errors).item(),
                'max': torch.max(data_errors).item()}

    def compute_diversity_train_set(self):
        print('Computing train set diversity')
        previous_verts_batch = None
        mean_distances = []
        for data in tqdm.tqdm(self._train_loader):
            if self._config['data']['swap_features']:
                x = data.x[self._manager.batch_diagonal_idx, ::]
            else:
                x = data.x

            current_verts_batch = x
            if self._normalized_data:
                current_verts_batch = self._unnormalize_verts(
                    current_verts_batch, x.device)

            if previous_verts_batch is not None:
                verts_batch_distances = self._manager.compute_vertex_errors(
                    previous_verts_batch, current_verts_batch)
                mean_distances.append(torch.mean(verts_batch_distances, dim=1))
            previous_verts_batch = current_verts_batch
        return torch.mean(torch.cat(mean_distances, dim=0)).item()

    def compute_diversity(self, n_samples=10000):
        print('Computing generative model diversity')
        samples_per_batch = 20
        mean_distances = []
        for _ in tqdm.tqdm(range(n_samples // samples_per_batch)):
            verts_batch_distances = self._manager.compute_vertex_errors(
                self.random_generation(samples_per_batch),
                self.random_generation(samples_per_batch))
            mean_distances.append(torch.mean(verts_batch_distances, dim=1))
        return torch.mean(torch.cat(mean_distances, dim=0)).item()

    def compute_specificity(self, n_samples=100):
        print('Computing generative model specificity')
        min_distances = []
        for _ in tqdm.tqdm(range(n_samples)):
            sample = self.random_generation(1)

            mean_distances = []
            for data in self._train_loader:
                if self._config['data']['swap_features']:
                    x = data.x[self._manager.batch_diagonal_idx, ::]
                else:
                    x = data.x

                if self._normalized_data:
                    x = self._unnormalize_verts(x.to(self._device))
                else:
                    x = x.to(self._device)

                v_dist = self._manager.compute_vertex_errors(
                    x, sample.expand(x.shape[0], -1, -1))
                mean_distances.append(torch.mean(v_dist, dim=1))
            min_distances.append(torch.min(torch.cat(mean_distances, dim=0)))
        return torch.mean(torch.stack(min_distances)).item()

    def evaluate_gen(self, data_loader, n_sampled_points=None):
        all_sample = []
        all_ref = []
        for data in tqdm.tqdm(data_loader):
            if self._config['data']['swap_features']:
                data.x = data.x[self._manager.batch_diagonal_idx, ::]
            data = data.to(self._device)
            if self._normalized_data:
                data.x = self._unnormalize_verts(data.x)

            ref = data.x
            sample = self.random_generation(data.x.shape[0])

            if n_sampled_points is not None:
                subset_idxs = np.random.choice(ref.shape[1], n_sampled_points)
                ref = ref[:, subset_idxs]
                sample = sample[:, subset_idxs]

            all_ref.append(ref)
            all_sample.append(sample)

        sample_pcs = torch.cat(all_sample, dim=0)
        ref_pcs = torch.cat(all_ref, dim=0)
        print("Generation sample size:%s reference size: %s"
              % (sample_pcs.size(), ref_pcs.size()))

        # Compute metrics
        metrics = compute_all_metrics(
            sample_pcs, ref_pcs, self._config['optimization']['batch_size'])
        metrics = {k: (v.cpu().detach().item()
                       if not isinstance(v, float) else v) for k, v in
                   metrics.items()}
        print(metrics)

        sample_pcl_npy = sample_pcs.cpu().detach().numpy()
        ref_pcl_npy = ref_pcs.cpu().detach().numpy()
        jsd = jsd_between_point_cloud_sets(sample_pcl_npy, ref_pcl_npy)
        print("JSD:%s" % jsd)
        metrics["jsd"] = jsd

        outfile_path = os.path.join(self._out_dir, 'eval_metrics_gen.json')
        with open(outfile_path, 'w') as outfile:
            json.dump(metrics, outfile)

    def latent_swapping(self, v_batch=None):
        if v_batch is None:
            v_batch = self.random_generation(2, denormalize=False)
        else:
            assert v_batch.shape[0] >= 2
            v_batch = v_batch.to(self._device)
            if self._config['data']['swap_features']:
                v_batch = v_batch[self._manager.batch_diagonal_idx, ::]
            v_batch = v_batch[:2, ::]

        z = self._manager.encode(v_batch)
        z_0, z_1 = z[0, ::], z[1, ::]

        swapped_verts = []
        for key, z_region in self._manager.latent_regions.items():
            z_swap = z_0.clone()
            z_swap[z_region[0]:z_region[1]] = z_1[z_region[0]:z_region[1]]
            swapped_verts.append(self._manager.generate(z_swap))

        all_verts = torch.cat([v_batch, torch.cat(swapped_verts, dim=0)], dim=0)

        if self._normalized_data:
            all_verts = self._unnormalize_verts(all_verts)

        out_mesh_dir = os.path.join(self._out_dir, 'latent_swapping')
        if not os.path.isdir(out_mesh_dir):
            os.mkdir(out_mesh_dir)
        self.save_batch(all_verts, out_mesh_dir)

        source_dist = self._manager.compute_vertex_errors(
            all_verts, all_verts[0, ::].expand(all_verts.shape[0], -1, -1))
        target_dist = self._manager.compute_vertex_errors(
            all_verts, all_verts[1, ::].expand(all_verts.shape[0], -1, -1))

        renderings = self._manager.render(all_verts)
        renderings_source = self._manager.render(all_verts, source_dist, 5)
        renderings_target = self._manager.render(all_verts, target_dist, 5)
        grid = make_grid(torch.cat(
            [renderings, renderings_source, renderings_target], dim=-2),
            padding=10, pad_value=1, nrow=renderings.shape[0])
        save_image(grid, os.path.join(out_mesh_dir, 'latent_swapping.png'))

    def fit_vertices(self, target_verts, lr=5e-3, iterations=250,
                     target_noise=0, target_landmarks=None):
        # Scale and position target_verts
        target_verts = target_verts.unsqueeze(0).to(self._device)
        if target_landmarks is None:
            target_landmarks = target_verts[:, self.coma_landmarks, :]
        target_landmarks = target_landmarks.to(self._device)

        if target_noise > 0:
            target_verts = target_verts + (torch.randn_like(target_verts) *
                                           target_noise /
                                           self._manager.to_mm_const)
            target_landmarks = target_landmarks + (
                torch.randn_like(target_landmarks) *
                target_noise / self._manager.to_mm_const)

        z = self.latent_stats['means'].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr)
        gen_verts = None
        for i in range(iterations):
            optimizer.zero_grad()
            gen_verts = self._manager.generate_for_opt(z.to(self._device))
            if self._normalized_data:
                gen_verts = self._unnormalize_verts(gen_verts)

            if i < iterations // 3:
                er = self._manager.compute_mse_loss(
                    gen_verts[:, self.uhm_landmarks, :], target_landmarks)
            else:
                er, _ = pytorch3d.loss.chamfer_distance(gen_verts, target_verts)

            er.backward()
            optimizer.step()
        return gen_verts, target_verts.squeeze()

    def fit_coma_data(self, base_dir='meshes2fit',
                      noise=0, export_meshes=False):
        print(f"Fitting CoMA meshes with noise = {noise} mm")
        out_mesh_dir = os.path.join(self._out_dir, 'fitting')
        if not os.path.isdir(out_mesh_dir):
            os.mkdir(out_mesh_dir)

        names_and_scale = {}
        for dirpath, _, fnames in os.walk(base_dir):
            for f in fnames:
                if f.endswith('.ply'):
                    if f[:5] in ['03274', '03275', '00128', '03277']:
                        names_and_scale[f] = 9
                    else:
                        names_and_scale[f] = 8

        dataframes = []
        for m_id, scale in tqdm.tqdm(names_and_scale.items()):
            df_id = m_id.split('.')[0]
            subd = False
            mesh_path = os.path.join(base_dir, m_id)
            target_mesh = trimesh.load_mesh(mesh_path, 'ply', process=False)
            target_verts = torch.tensor(
                target_mesh.vertices, dtype=torch.float,
                requires_grad=False, device=self._device)

            # scale and translate to match template. Values manually computed
            target_verts *= scale
            target_verts[:, 1] += 0.15

            # If target mesh was subdivided use original target to retrieve its
            # landmarks
            target_landmarks = None
            if 'subd' in m_id:
                subd = True
                df_id = m_id.split('_')[0]
                base_path = os.path.join(base_dir, m_id.split('_')[0] + '.ply')
                base_mesh = trimesh.load_mesh(base_path, 'ply', process=False)
                base_verts = torch.tensor(
                    base_mesh.vertices, dtype=torch.float,
                    requires_grad=False, device=self._device)
                target_landmarks = base_verts[self.coma_landmarks, :]
                target_landmarks = target_landmarks.unsqueeze(0)
                target_landmarks *= scale
                target_landmarks[:, 1] += 0.15

            out_verts, t_verts = self.fit_vertices(
                target_verts, target_noise=noise,
                target_landmarks=target_landmarks)

            closest_p_errors = self._manager.to_mm_const * \
                self._dist_closest_point(out_verts, target_verts.unsqueeze(0))

            dataframes.append(pd.DataFrame(
                {'id': df_id, 'noise': noise, 'subdivided': subd,
                 'errors': closest_p_errors.squeeze().detach().cpu().numpy()}))

            if export_meshes:
                mesh_name = m_id.split('.')[0]
                out_mesh = trimesh.Trimesh(
                    out_verts[0, ::].cpu().detach().numpy(),
                    self._manager.template.face.t().cpu().numpy())
                out_mesh.export(os.path.join(
                    out_mesh_dir, mesh_name + f"_fit_{str(noise)}" + '.ply'))
                target_mesh.vertices = t_verts.detach().cpu().numpy()
                target_mesh.export(os.path.join(
                    out_mesh_dir, mesh_name + f"_t_{str(noise)}" + '.ply'))
        return pd.concat(dataframes)

    def fit_coma_data_different_noises(self, base_dir='meshes2fit'):
        noises = [0, 2, 4, 6, 8]
        dataframes = []
        for n in noises:
            dataframes.append(self.fit_coma_data(base_dir, n, True))
        df = pd.concat(dataframes)
        df.to_pickle(os.path.join(self._out_dir, 'coma_fitting.pkl'))

        sns.set_theme(style="ticks")
        plt.figure()
        sns.lineplot(data=df, x='noise', y='errors',
                     markers=True, dashes=False, ci='sd')
        plt.savefig(os.path.join(self._out_dir, 'coma_fitting.svg'))

        plt.figure()
        sns.boxplot(data=df, x='noise', y='errors', showfliers=False)
        plt.savefig(os.path.join(self._out_dir, 'coma_fitting_box.svg'))

        plt.figure()
        sns.violinplot(data=df[df.errors < 3], x='noise', y='errors',
                       split=False)
        plt.savefig(os.path.join(self._out_dir, 'coma_fitting_violin.svg'))

    @staticmethod
    def _point_mesh_distance(points, verts, faces):
        points = points.squeeze()
        verts_packed = verts.to(points.device)
        faces_packed = torch.tensor(faces, device=points.device).t()
        first_idx = torch.tensor([0], device=points.device)

        tris = verts_packed[faces_packed]

        point_to_face = point_face_distance(points, first_idx, tris,
                                            first_idx, points.shape[0])
        return point_to_face / points.shape[0]

    @staticmethod
    def _dist_closest_point(x, y):
        # for each point on x return distance to closest point in y
        x, x_lengths, x_normals = _handle_pointcloud_input(x, None, None)
        y, y_lengths, y_normals = _handle_pointcloud_input(y, None, None)
        x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
        cham_x = x_nn.dists[..., 0]
        return cham_x

    def direct_manipulation(self, z=None, indices=None, new_coords=None,
                            lr=0.1, iterations=50, affect_only_zf=True):
        if z is None:
            z = self.latent_stats['means'].unsqueeze(0)
            # z = self.random_latent(1)
            z = z.clone().detach().requires_grad_(True)
        if indices is None and new_coords is None:
            indices = [8816, 8069, 8808]
            new_coords = torch.tensor([[-0.0108174, 0.0814601, 0.664498],
                                       [-0.1821480, 0.0190682, 0.419531],
                                       [-0.0096422, 0.3058790, 0.465528]])
        new_coords = new_coords.unsqueeze(0).to(self._device)

        colors = self._manager.template.colors.to(torch.long)
        features = [str(colors[i].cpu().detach().numpy()) for i in indices]
        assert all(x == features[0] for x in features)

        zf_idxs = self._manager.latent_regions[features[0]]

        optimizer = torch.optim.Adam([z], lr)
        initial_verts = self._manager.generate_for_opt(z.to(self._device))
        if self._normalized_data:
            initial_verts = self._unnormalize_verts(initial_verts)
        gen_verts = None
        for i in range(iterations):
            optimizer.zero_grad()
            gen_verts = self._manager.generate_for_opt(z.to(self._device))
            if self._normalized_data:
                gen_verts = self._unnormalize_verts(gen_verts)

            loss = self._manager.compute_mse_loss(
                gen_verts[:, indices, :], new_coords)
            loss.backward()

            if affect_only_zf:
                z.grad[:, :zf_idxs[0]] = 0
                z.grad[:, zf_idxs[1]:] = 0
            optimizer.step()

        # Save output meshes
        out_mesh_dir = os.path.join(self._out_dir, 'direct_manipulation')
        if not os.path.isdir(out_mesh_dir):
            os.mkdir(out_mesh_dir)

        initial_mesh = trimesh.Trimesh(
            initial_verts[0, ::].cpu().detach().numpy(),
            self._manager.template.face.t().cpu().numpy())
        initial_mesh.export(os.path.join(out_mesh_dir, 'initial.ply'))

        new_mesh = trimesh.Trimesh(
            gen_verts[0, ::].cpu().detach().numpy(),
            self._manager.template.face.t().cpu().numpy())
        new_mesh.export(os.path.join(out_mesh_dir, 'new.ply'))

        for i, coords in zip(indices, new_coords[0, ::].detach().cpu().numpy()):
            sphere = trimesh.creation.icosphere(radius=0.01)
            sphere.vertices = sphere.vertices + coords
            sphere.export(os.path.join(out_mesh_dir, f'target_{i}.ply'))

            sphere = trimesh.creation.icosphere(radius=0.01)
            sphere.vertices += initial_verts[0, i, :].cpu().detach().numpy()
            sphere.export(os.path.join(out_mesh_dir, f'selected_{i}.ply'))

    def interpolate(self):
        with open(os.path.join('precomputed', f'data_split_{self._data_type}.json'), 'r') as fp:
            data = json.load(fp)
        test_list = data['test']
        meshes_root = self._test_loader.dataset.root

        # Pick first test mesh and find most different mesh in test set
        v_1 = None
        distances = [0]
        for i, fname in enumerate(test_list):
            mesh_path = os.path.join(meshes_root, fname)
            mesh = trimesh.load_mesh(mesh_path, process=False)
            mesh_verts = torch.tensor(mesh.vertices, dtype=torch.float,
                                      requires_grad=False, device='cpu')
            if i == 0:
                v_1 = mesh_verts
            else:
                distances.append(
                    self._manager.compute_mse_loss(v_1, mesh_verts).item())

        m_2_path = os.path.join(
            meshes_root, test_list[np.asarray(distances).argmax()] + '.ply')
        m_2 = trimesh.load_mesh(m_2_path, 'ply', process=False)
        v_2 = torch.tensor(m_2.vertices, dtype=torch.float, requires_grad=False)

        v_1 = (v_1 - self._norm_dict['mean']) / self._norm_dict['std']
        v_2 = (v_2 - self._norm_dict['mean']) / self._norm_dict['std']

        z_1 = self._manager.encode(v_1.unsqueeze(0).to(self._device))
        z_2 = self._manager.encode(v_2.unsqueeze(0).to(self._device))

        features = list(self._manager.template.feat_and_cont.keys())

        # Interpolate per feature
        if self._config['data']['swap_features']:
            z = z_1.repeat(len(features) // 2, 1)
            all_frames, rows = [], []
            for feature in features:
                zf_idxs = self._manager.latent_regions[feature]
                z_1f = z_1[:, zf_idxs[0]:zf_idxs[1]]
                z_2f = z_2[:, zf_idxs[0]:zf_idxs[1]]
                z[:, zf_idxs[0]:zf_idxs[1]] = self.vector_linspace(
                    z_1f, z_2f, len(features) // 2).to(self._device)

                gen_verts = self._manager.generate(z.to(self._device))
                if self._normalized_data:
                    gen_verts = self._unnormalize_verts(gen_verts)

                renderings = self._manager.render(gen_verts).cpu()
                all_frames.append(renderings)
                rows.append(make_grid(renderings, padding=10,
                            pad_value=1, nrow=len(features)))
                z = z[-1, :].repeat(len(features) // 2, 1)

            save_image(
                torch.cat(rows, dim=-2),
                os.path.join(self._out_dir, 'interpolate_per_feature.png'))
            write_video(
                os.path.join(self._out_dir, 'interpolate_per_feature.mp4'),
                torch.cat(all_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=4)

        # Interpolate per variable
        z = z_1.repeat(3, 1)
        all_frames = []
        for z_i in range(self._manager.model_latent_size):
            z_1f = z_1[:, z_i]
            z_2f = z_2[:, z_i]
            z[:, z_i] = torch.linspace(z_1f.item(),
                                       z_2f.item(), 3).to(self._device)

            gen_verts = self._manager.generate(z.to(self._device))
            if self._normalized_data:
                gen_verts = self._unnormalize_verts(gen_verts)

            renderings = self._manager.render(gen_verts).cpu()
            all_frames.append(renderings)
            z = z[-1, :].repeat(3, 1)

        write_video(
            os.path.join(self._out_dir, 'interpolate_per_variable.mp4'),
            torch.cat(all_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=4)

        # Interpolate all features
        zs = self.vector_linspace(z_1, z_2, len(features))

        gen_verts = self._manager.generate(zs.to(self._device))
        if self._normalized_data:
            gen_verts = self._unnormalize_verts(gen_verts)

        renderings = self._manager.render(gen_verts).cpu()
        im = make_grid(renderings, padding=10, pad_value=1, nrow=len(features))
        save_image(im, os.path.join(self._out_dir, 'interpolate_all.png'))


    # AGE TESTS

    ## maybe put this in model manager to make it more general and call that in here
    def process_data(self, loader, datasets, diagonal):

        """
        
        This function processes the data from the loader and returns the feature latents, age latents and ground truth ages seperately.
    
        """

        feature_latents_list = []
        age_latents_list = []
        gt_age_list = []
        gt_age_norm_list = []
        data_dataset = []

        for batch in tqdm.tqdm(loader):

            gt_ages_batch = batch.age
            gt_ages_norm_batch = batch.norm_age
            file_name = batch.fname

            if datasets is not None:
                for fname in file_name:
                    if 'combined' in str(self._config['data']['dataset_type']):
                        dataset_name = datasets[datasets['id'] == int(fname)]['Dataset'].values[0]
                    else:
                        dataset_name = datasets[datasets['id'] == fname]['Dataset'].values[0] 
                    data_dataset.append(dataset_name)

            if diagonal and self._config['data']['swap_features']:
                data = batch.x[self._manager.batch_diagonal_idx, ::] 
            else:
                data = batch.x

            z = self._manager.encode(data.to(self._device)).detach()
            z_features = z[:, :-self._config['model']['age_latent_size']]
            z_ages = z[:, -self._config['model']['age_latent_size']:]

            bs = self._config['optimization']['batch_size']

            if self._config['data']['swap_features'] and not diagonal:
                swapped = batch.swapped
                gt_ages_batch = self._manager._gt_age(bs, z_ages, gt_ages_batch, swapped)
                gt_ages_norm_batch = self._manager._gt_age(bs, z_ages, gt_ages_norm_batch, swapped)

            for i in range(data.shape[0]):
                feature_latents_list.append(z_features[i])
                age_latents_list.append(z_ages[i])
                gt_age_list.append(gt_ages_batch[i])
                gt_age_norm_list.append(gt_ages_norm_batch[i])

        feature_latents = torch.stack(feature_latents_list).detach().cpu().numpy()
        age_latents = torch.stack(age_latents_list).detach().cpu().numpy()
        gt_ages = torch.stack(gt_age_list).detach().cpu().numpy()
        gt_ages_norm = torch.stack(gt_age_norm_list).detach().cpu().numpy()

        # if self._config['model']['age_per_feature']==False and self._config['data']['swap_features']==False:
        #     gt_ages = gt_ages.reshape(-1, 1)
        #     gt_ages_norm = gt_ages_norm.reshape(-1, 1)

        # need to give off all for either 1 or 9 age latents 
    
        return feature_latents, age_latents, gt_ages, gt_ages_norm, data_dataset
    
    def set_seed(self, seed):

        """
        
        This function makes sure all random operation produce the same results each time the code is run.
        
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def age_latent_changing(self, data_loader):

        """
        
        This function generates an image of 4 subjects from a batch and changes the age latent to 5 differenet ages and displays them with their difference maps compared to the orginal mesh. 
        A subjects goes through the encoder, the age latent is changed, then goes through the decoder. 

        10 different tests are run. Changing the age for each feature (9) and then changing the age for all features (1).

        Output: 10 images of a batch with 5 different ages & difference maps when compared to the:
         - first age (0)

        """

        padding_value = 10

        storage_path = os.path.join(self._manager._precomputed_storage_path, f'normalise_age_{self._data_type}.pkl')
        with open(storage_path, 'rb') as file:
            age_train_mean, age_train_std = \
                pickle.load(file)
        age_latent_ranges = self._config['testing']['age_latent_changing']
        age_latent_ranges_original = age_latent_ranges.copy()
        for i in range(len(age_latent_ranges)):
            age_latent_ranges[i] = (age_latent_ranges[i] - age_train_mean) / age_train_std

        batch = next(iter(data_loader))

        original_ages = batch.age.numpy()

        if self._config['data']['swap_features']:
            batch = batch.x[self._manager.batch_diagonal_idx, ::] 
        else:
            batch = batch.x

        z = self._manager.encode(batch.to(self._device)).detach()

        error_scale = 5

        age_latent_size = self._config['model']['age_latent_size']
        latent_size = self._config['model']['latent_size'] - age_latent_size

        age_latent_ranges = torch.tensor(age_latent_ranges) 
        expanded_tensor = z.unsqueeze(1).expand(-1, age_latent_ranges.size(0)-1, -1)

        if self._config['model']['age_per_feature']:
            assert age_latent_size > 1
            range_size = age_latent_size + 1
        else:
            range_size = 1


        # changing each features age latent value [0,4,8,12,17]

        for i in range(range_size):

            output = []
            expanded_tensor_copy = expanded_tensor.clone()

            for j in range(len(age_latent_ranges)-1):

                if self._config['model']['age_per_feature'] and i != age_latent_size:
                    expanded_tensor_copy[:, j, i+latent_size] = age_latent_ranges[j+1]
                    name = i
                else:
                    expanded_tensor_copy[:, j, latent_size:] = age_latent_ranges[j+1]
                    name = "all"

            gen_verts = self._manager.generate(expanded_tensor_copy.to(self._device))

            if self._normalized_data:
                gen_verts = self._unnormalize_verts(gen_verts)

            renderings = self._manager.render(gen_verts).cpu()

            for j in range(z.size(0)):

                output.extend(renderings[j*(len(age_latent_ranges)-1):(j+1)*(len(age_latent_ranges)-1)])
                first_index = j*(len(age_latent_ranges)-1)

                for k in range(len(age_latent_ranges)-1):

                    k_index = (j*(len(age_latent_ranges)-1)) + k

                    differences_from_first = self._manager.compute_vertex_errors(gen_verts[k_index].unsqueeze(0), gen_verts[first_index].unsqueeze(0))
                    differences_renderings_first = self._manager.render(gen_verts[k_index].unsqueeze(0), differences_from_first, error_max_scale=error_scale).cpu().detach()
                    output.append(differences_renderings_first.squeeze())

            # create image

            stacked_frames = torch.stack(output)
            file_path = os.path.join(self._out_dir, f'age_latent_changing_{age_latent_ranges_original}_{name}.png')
            grid = make_grid(stacked_frames, padding=padding_value, pad_value=255, nrow=len(age_latent_ranges)-1) 
            save_image(grid, file_path)
            self.log[f'age_latent_changing_{age_latent_ranges_original}_{name}'].upload(file_path)


        line_to_add = 'age_latent_changing original ages: ' + str(original_ages)
        filename = os.path.join(self._out_dir, 'results.txt')
        self.log['age_latent_changing_original_ages'].upload(str(original_ages))

        if not os.path.exists(filename):
            with open(filename, 'w') as file:
                file.write('')
        else:
            print(f"{filename} already exists.")

        with open(filename, 'a') as file:
            file.write(line_to_add)
            file.write('\n' * 2)

        
        # make pre/post model mesh

        def render_and_diff(pre, post):
            pre_render = self._manager.render(pre).cpu()
            post_render = self._manager.render(post).cpu()
            difference_pre_pre = self._manager.compute_vertex_errors(pre, pre)
            difference_rendering_pre_pre = self._manager.render(pre, difference_pre_pre, error_max_scale=error_scale).cpu().detach()
            difference_pre_post = self._manager.compute_vertex_errors(post, pre)
            difference_rendering_pre_post = self._manager.render(post, difference_pre_post, error_max_scale=error_scale).cpu().detach()
            return pre_render.squeeze(), post_render.squeeze(), difference_rendering_pre_pre.squeeze(), difference_rendering_pre_post.squeeze()

        output = []
        for i in range(z.size(0)):
            pre = batch[i, ::].clone().unsqueeze(0)
            if self._normalized_data:
                pre = self._unnormalize_verts(pre.to(self._device))
            z_gen = z[i, ::].clone()
            gen_output = self._manager.generate(z_gen.to(self._device))
            if self._normalized_data:
                gen_output = self._unnormalize_verts(gen_output)

            pre_render, post_render, difference_rendering_pre, difference_rendering_post = render_and_diff(pre, gen_output)
            output.extend([pre_render, post_render, difference_rendering_pre, difference_rendering_post])  # Assuming you want to add the difference_rendering twice as in the original code

        stacked_frames = torch.stack(output)
        file_path = os.path.join(self._out_dir, 'pre_post_mesh.png')
        grid = make_grid(stacked_frames, padding=padding_value, pad_value=255, nrow=2) 
        save_image(grid, file_path)
        self.log[f'test/pre_post_mesh.png'].upload(file_path)


        # # ##################

        # # generate a colour bar for /plots 

        # stacked_frames = torch.stack(output)
        # grid = make_grid(stacked_frames, padding=10, pad_value=1, nrow=len(age_latent_ranges)) 

        # # Convert the tensor to a numpy array and transpose the dimensions for matplotlib
        # grid_np = grid.numpy().transpose((1, 2, 0))

        # plt.figure(figsize=(10, 10))
        # plt.imshow(grid_np, interpolation='nearest', vmin=0, vmax=error_scale, cmap='plasma')  # Set the colorbar range to 0-5 and colormap to 'plasma'
        # cbar = plt.colorbar(cmap='plasma')  # Set the colorbar colormap to 'plasma'
        # cbar.set_label('Error (mm)')
        # plt.savefig(os.path.join(self._out_dir, f'colour bar_{age_latent_ranges_original}.png'))
        # plt.close()

        # # ##################


    def age_per_feature_new_ages(self, z, gt_age, swapped_feature):
        assert self._config['data']['swap_features']

        latent_size = self._manager.model_latent_size
        age_latent_size = self._config['model']['age_latent_size']
        latent_regions = self._manager.latent_regions
        bs = self._config['optimization']['batch_size']

        gt_feature_ages = utils.age_per_feature_new_ages(z, gt_age, swapped_feature, latent_size, age_latent_size, latent_regions, bs)

        return gt_feature_ages

    # def age_encoder_check(self, train_loader, test_loader):

    #     """
        
    #     This function calculates the mean absolute difference between the actual age and the age latent after being passed through the encoder once only on the main diagonal. This checks how well the encoder generates the correct age. 

    #     Output: scatter plot of predicted age against ground truth age  
        
    #     """

    #     storage_path = os.path.join(self._manager._precomputed_storage_path, f'normalise_age_{self._data_type}.pkl')
    #     age_latent_size = self._config['model']['age_latent_size']

    #     with open(storage_path, 'rb') as file:
    #         age_train_mean, age_train_std = \
    #             pickle.load(file)

    #     for j in range(2):

    #         if j == 0:
    #             data_loader = train_loader
    #         else:
    #             data_loader = test_loader
    
    #         age_original = []
    #         age_predict = []
    #         age_std = []
    #         all_diff = []

    #         age_original_all = []
    #         age_predict_all = []

    #         for data in tqdm.tqdm(data_loader):

    #             batch = data.x
    #             # if swap_features is true AND number of age latents = 1, then use only the diagonal of the batch, if false use the whole batch
    #             if self._config['data']['swap_features']:
    #                 batch = batch[self._manager.batch_diagonal_idx, ::]

    #             z = self._manager.encode(batch.to(self._device)).detach()

    #             gt_age = data.age.squeeze()
    #             age_vector = gt_age.unsqueeze(1).repeat(1, 9)

    #             z_age = z[:, -age_latent_size:]
    #             z_age = (z_age * age_train_std) + age_train_mean

    #             age_original_all.extend(age_vector.tolist())
    #             age_predict_all.extend(z_age.tolist())

    #             if self._config['model']['age_per_feature']:
    #                 std = torch.std(z_age, dim=1)
    #                 z_age = torch.mean(z_age, dim=1)
    #             else:
    #                 std = torch.zeros(z_age.size(0)) 

    #             z_age = z_age.squeeze()

    #             age_diff = abs(torch.tensor(gt_age).to(z_age.device) - torch.tensor(z_age))

    #             for i in range(z.shape[0]):
    #                 age_predict.append(z_age[i].item())
    #                 age_original.append(gt_age[i].item())   
    #                 age_std.append(std[i].item())                 
    #                 all_diff.append(age_diff[i].item())
                
    #         age_std_scaled = [s * 100 for s in age_std]
    #         average_diff = np.mean(all_diff)


    #         if j == 0:
    #             train_age_original = age_original
    #             train_age_predict = age_predict
    #             train_age_std = age_std_scaled
    #             train_average_diff = average_diff
    #         else:
    #             test_age_original = age_original
    #             test_age_predict = age_predict
    #             test_age_std = age_std_scaled
    #             test_average_diff = average_diff

    #     # plot graph of actual age against age latent from encoder
    #     age_range = self._config['data']['dataset_age_range']
    #     min_age, max_age = map(int, age_range.split('-'))

    #     # plt.figure(figsize=(6, 6))
    #     # plt.clf()

    #     # # # Check the scaled standard deviation values
    #     # # print("Train Age Std Scaled:", train_age_std)
    #     # # print("Test Age Std Scaled:", test_age_std)

    #     # # Define base marker size and scaling factor
    #     # base_marker_size = 50
    #     # scaling_factor = 200
    #     # max_marker_size = 300

    #     # max_std = max(max(train_age_std), max(test_age_std))

    #     # # Normalize the std values relative to the maximum std
    #     # train_age_std_normalized = [s / max_std for s in train_age_std]
    #     # test_age_std_normalized = [s / max_std for s in test_age_std]

    #     # # Scale marker sizes dynamically based on normalized std
    #     # train_marker_sizes = [min(base_marker_size + s * scaling_factor, max_marker_size) for s in train_age_std_normalized]
    #     # test_marker_sizes = [min(base_marker_size + s * scaling_factor, max_marker_size) for s in test_age_std_normalized]

    #     # # Scatter plot with dynamic marker sizes
    #     # plt.scatter(train_age_original, train_age_predict, s=train_marker_sizes, color='yellow', marker='x', label='Train dataset')
    #     # plt.scatter(test_age_original, test_age_predict, s=test_marker_sizes, color='blue', marker='o', label='Test dataset')

    #     # # Add diagonal line
    #     # plt.plot([min_age, max_age], [min_age, max_age], 'r--')

    #     # # Fixed marker sizes for legend
    #     # legend_handles = [
    #     #     plt.scatter([], [], color='yellow', marker='x', s=base_marker_size, label='Train dataset'),
    #     #     plt.scatter([], [], color='blue', marker='o', s=base_marker_size, label='Test dataset')
    #     # ]
    #     # plt.legend(handles=legend_handles, loc='upper left')

    #     # # Add title, labels, and text
    #     # plt.title(f'Age prediction on age latent')
    #     # plt.xlabel('Ground truth age (years)')
    #     # plt.ylabel('Predicted age (years)')
    #     # plt.text(0.30, 0.1, f'Mean absolute difference (train) = {round(train_average_diff, 2)} years', transform=plt.gca().transAxes)
    #     # plt.text(0.30, 0.05, f'Mean absolute difference (test) = {round(test_average_diff, 2)} years', transform=plt.gca().transAxes)

    #     # # # Add average standard deviation values as text
    #     # # plt.text(0.30, 0.2, f'Average std (train) = {round(np.mean(train_age_std), 2)}', transform=plt.gca().transAxes)
    #     # # plt.text(0.30, 0.15, f'Average std (test) = {round(np.mean(test_age_std), 2)}', transform=plt.gca().transAxes)

    #     # # Set ticks
    #     # plt.xticks(range(0, 18))
    #     # plt.yticks(range(0, 18))

    #     # # Save the plot
    #     # file_path = os.path.join(self._out_dir, f'age_encoder_check_{age_range}.png')
    #     # plt.savefig(file_path)
    #     # self.log[f'test/age_encoder_check_{age_range}'].upload(file_path)


    #     # Second plot: Each of the 9 age latents vs GT age for the test set
    #     plt.figure(figsize=(8, 6))
    #     plt.clf()

    #     # Define colors for the 9 age latents
    #     colors = plt.cm.tab10.colors  # Use a colormap with 10 distinct colors

    #     # Convert age_original_all and age_predict_all to numpy arrays for easier manipulation
    #     age_original_all = np.array(age_original_all)
    #     age_predict_all = np.array(age_predict_all)

    #     features = ["Temporal", "Eyes", "Cheekbones", "Cheeks", "Jaw", "Forehead", "Chin", "Lips", "Nose"]

    #     # Initialize a list to store MAE for each latent
    #     mae_per_latent = []

    #     # Iterate over each of the 9 age latents and plot it against the GT age
    #     for latent_idx in range(9):  # Assuming there are 9 age latents
    #         gt_values = age_original_all[:, latent_idx]  # Extract GT values for the current latent
    #         pred_values = age_predict_all[:, latent_idx]  # Extract predicted values for the current latent
            
    #         # Calculate MAE for the current latent
    #         mae = np.mean(np.abs(np.array(pred_values) - np.array(gt_values)))
    #         mae_per_latent.append(mae)
            
    #         # Plot the scatter for the current latent
    #         plt.scatter(gt_values, pred_values, color=colors[latent_idx % len(colors)], label=f'{features[latent_idx]} (MAE: {mae:.2f})', alpha=0.7)

    #     # Add a diagonal line for reference
    #     plt.plot([0, 17], [0, 17], 'r--', label='Ideal')  # Assuming age range is 0-18

    #     # Set ticks
    #     plt.xticks(range(0, 18))
    #     plt.yticks(range(0, 18))

    #     # Add title, labels, and legend
    #     plt.title('Encoder: Predicted vs Ground Truth for Each Age Latent')
    #     plt.xlabel('Ground Truth Age (years)')
    #     plt.ylabel('Predicted Age Latent Value')
    #     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the plot
    #     plt.grid(True)

    #     # Add total MAE for all latents as text on the graph
    #     total_mae = np.mean(mae_per_latent)
    #     plt.text(1.01, 0.45, f'Total MAE: {total_mae:.2f}', transform=plt.gca().transAxes, fontsize=10, color='black')

    #     # Save the plot
    #     file_path = os.path.join(self._out_dir, 'encoder_accuracy_scatter_plot.png')
    #     plt.savefig(file_path, bbox_inches='tight')  # Save with tight layout to include the legend
    #     self.log['test/encoder_accuracy_scatter_plot'].upload(file_path)


    def age_encoder_decoder_accuracy(self, train_loader, test_loader):

        """
        
        This function encodes the subjects, changes the age latent to a random age, decodes, then encodes again and meaures if the second encoding can generate 
        the same age that it was assigned to after the first encoding. If the encoder check test shows the encoder works well, this tests will show how well the decoder performs.

        Output: two scatter plots. One for assigned age vs predicted age after second encoding and one for GT age vs predicted age after second encoding. Both the training and val/test set are plotted
        
        """

        age_latent_size = self._config['model']['age_latent_size']

        for j in range(2):

            if j == 0:
                data_loader = train_loader
            else:
                data_loader = test_loader

            age_latents_gt = []
            age_preds_encoder = []
            age_latents_rand = []
            age_preds_decoder = []

            for batch in tqdm.tqdm(data_loader):

                if self._config['data']['swap_features']:
                    x = batch.x[self._manager.batch_diagonal_idx, ::]   
                else:
                    x = batch.x

                gt_age = batch.age 

                z = self._manager.encode(x.to(self._device)).detach()
                
                storage_path = os.path.join(self._manager._precomputed_storage_path, f'normalise_age_{self._data_type}.pkl')
                with open(storage_path, 'rb') as file:
                    age_train_mean, age_train_std = \
                        pickle.load(file)
                    
                age_range = self._config['data']['dataset_age_range']
                age_lower, age_upper = map(int, age_range.split('-'))

                for i in tqdm.tqdm(range(z.shape[0])):
                    age_preds = z[i][-age_latent_size:]
                    age_preds = (age_preds * age_train_std) + age_train_mean
                    age_preds_encoder.append(age_preds.tolist())

                    age_latent_rand = random.randrange(age_lower, age_upper)
                    age_latent_rand = [age_latent_rand] * age_latent_size
                    age_latents_rand.append(age_latent_rand)
                    age_latent_rand_norm = [(val - age_train_mean) / age_train_std for val in age_latent_rand]
                    z[i, -age_latent_size:] = torch.tensor(age_latent_rand_norm)

                gen_verts = self._manager.generate(z.to(self._device))
                z_2 = self._manager.encode(gen_verts.to(self._device)).detach()

                for i in tqdm.tqdm(range(z.shape[0])):
                    age_pred = z_2[i][-age_latent_size:]
                    age_pred = (age_pred * age_train_std) + age_train_mean
                    age_preds_decoder.append(age_pred.tolist())

                    age_gt = [gt_age[i].item()] * age_latent_size
                    age_latents_gt.append(age_gt)

        ### PLOT RESULTS ###

            # Convert age_random_all and age_predict_all to numpy arrays for easier manipulation
            age_latents_rand = np.array(age_latents_rand)
            age_preds_decoder = np.array(age_preds_decoder)
            age_latents_gt = np.array(age_latents_gt)
            age_preds_encoder = np.array(age_preds_encoder)

            for i in range(2):

                if i == 0:
                    test_name = 'encoder'
                else:
                    test_name = 'decoder'

                # Second plot: Each of the 9 age latents vs GT age for the test set
                plt.figure(figsize=(8, 6))
                plt.clf()

                # Define colors for the 9 age latents
                colors = plt.cm.tab10.colors  # Use a colormap with 10 distinct colors

                features = ["Temporal", "Eyes", "Cheekbones", "Cheeks", "Jaw", "Forehead", "Chin", "Lips", "Nose"]

                # Initialize a list for errors
                mae_per_latent = []
                mse_per_latent = []

                # containers for per-age MAE
                max_age_int = 17
                per_age_abs_errors = [[] for _ in range(max_age_int + 1)]
                all_abs_errors = []

                # Iterate over each of the age latents and plot it against the GT age
                for latent_idx in range(self._config['model']['age_latent_size']): 
                    if i == 0:
                        gt_values = age_latents_gt[:, latent_idx]
                        pred_values = age_preds_encoder[:, latent_idx]
                    else:   
                        gt_values = age_latents_rand[:, latent_idx]
                        pred_values = age_preds_decoder[:, latent_idx]
                    
                    # Calculate MAE for the current latent
                    mae = np.mean(np.abs(np.array(pred_values) - np.array(gt_values)))
                    mae_per_latent.append(mae)

                    # Calculate MSE for the current latent
                    mse = np.mean((np.array(pred_values) - np.array(gt_values))**2)
                    mse_per_latent.append(mse)

                    # --- NEW: accumulate absolute errors per integer GT age bin ---
                    gt_int = np.clip(np.floor(gt_values).astype(int), 0, max_age_int)
                    abs_err = np.abs(np.array(pred_values) - np.array(gt_values))
                    for age_bin in range(max_age_int + 1):
                        mask = (gt_int == age_bin)
                        if np.any(mask):
                            per_age_abs_errors[age_bin].extend(abs_err[mask].tolist())
                    # --------------------------------------------------------------
                    
                    # Plot the scatter for the current latent
                    plt.scatter(gt_values, pred_values, color=colors[latent_idx % len(colors)], label=f'{features[latent_idx]} (MAE: {mae:.2f})', alpha=0.7)
                    # plt.scatter(gt_values, pred_values, color=colors[latent_idx % len(colors)], label=f'{features[latent_idx]} (MAE: {mae:.2f}, MSE: {mse:.2f})', alpha=0.7)

                # Add a diagonal line for reference
                plt.plot([0, 17], [0, 17], 'r--', label='Ideal')  # Assuming age range is 0-18

                # Set ticks
                plt.xticks(range(0, 18))
                plt.yticks(range(0, 18))

                if i == 0:
                    plt.title('Encoder: Predicted vs Ground Truth for Each Age Latent')
                    plt.xlabel('Ground Truth Age (years)')
                    plt.ylabel('Predicted Age Latent Value')
                else:
                    plt.title('Decoder: Predicted vs Randomly Assigned for Each Age Latent')
                    plt.xlabel('Randomly Assigned Age (years)')
                    plt.ylabel('Predicted Age Latent Value')
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the plot
                plt.grid(True)

                # Add total MAE for all latents as text on the graph
                total_mae = np.mean(mae_per_latent)
                plt.text(1.02, 0.47, f'Total MAE: {total_mae:.2f}', transform=plt.gca().transAxes, fontsize=10, color='black')

                # --- NEW: compute and display per-integer-age MAE block ---
                per_age_mae_lines = []
                for age_bin in range(max_age_int + 1):
                    if len(per_age_abs_errors[age_bin]) > 0:
                        age_mae = np.mean(per_age_abs_errors[age_bin])
                        per_age_mae_lines.append(f'Age {age_bin} (MAE: {age_mae:.2f})')
                    else:
                        per_age_mae_lines.append(f'Age {age_bin} (MAE: -)')

                if per_age_mae_lines:
                    per_age_text = "MAE per age (years):\n" + "\n".join(per_age_mae_lines)

                    plt.text(1.02, 0.42, per_age_text, transform=plt.gca().transAxes, fontsize=10, fontfamily='sans-serif', color='black', va='top',
                        bbox=dict(
                            boxstyle='round',
                            facecolor='white',
                            edgecolor='black',
                            alpha=0.8
                        )
                    )
                
                # Add total MAE for all ages as text on the graph
                total_mae = np.mean(abs_err)
                plt.text(1.02, -0.29, f'Total MAE: {total_mae:.2f}', transform=plt.gca().transAxes, fontsize=10, color='black')
                # ------------------------------------------------------------

                # Save the plot
                if j == 0:
                    test_name =  test_name + '_train'
                # else:
                #     test_name = 'test_' + test_name
                file_path = os.path.join(self._out_dir, f'{test_name}_accuracy_scatter_plot.png')
                plt.savefig(file_path, bbox_inches='tight')  # Save with tight layout to include the legend
                self.log[f'test/{test_name}_accuracy_scatter_plot'].upload(file_path)


    def dataset_split(self):

        """
        
        This function calculates how many data subjects there are for each age range group and original dataset to understand it's distribution. 
        
        It saves the results in a .png.  

        Output: three graphs:
            1. distribution of age 
            2. age split for train, val and test sets
            3. original dataset split
        
        """

        precomputed_storage_path = self._config['data']['precomputed_path']
        precomputed_data_path = os.path.join(precomputed_storage_path, f'data_split_{self._data_type}.json')
        metadata_path = self._config['data']['dataset_metadata_path']

        # Load precomputed data and metadata
        with open(precomputed_data_path, 'r') as file:
            data_split = json.load(file)  

        metadata = pd.read_csv(metadata_path)  

        # Initialize lists to store ages and their associated split type
        ages_list = []
        ages_decimals_list = []
        data_type_list = []
        dataset_list = []
        fname_list = []

        # Iterate over train, val, and test splits in the precomputed data
        for split_type, ids in data_split.items():
            # remove _ and .obj from ids
            ids =  [id.replace('_', '').replace('.obj', '') for id in ids]
            # Look up the entry in the metadata file using the 'id' column
            for entry_id in ids:
                if 'combined' in self._data_type:
                    entry_id = int(entry_id)
                dataset_row = metadata.loc[metadata['id'] == entry_id, 'Dataset']
                age_row = metadata.loc[metadata['id'] == entry_id, 'AgeYears']   
                age_decimals_row = metadata.loc[metadata['id'] == entry_id, 'age']     
                fname = entry_id        

                # If the ID exists in the metadata, add its age and split type to the lists
                if not dataset_row.empty and not age_row.empty:
                    ages_list.append(age_row.values[0])
                    ages_decimals_list.append(age_decimals_row.values[0])
                    data_type_list.append(split_type)  
                    dataset_list.append(dataset_row.values[0])
                    fname_list.append(fname)
        
        total_subjects = len(ages_list)

    # create a .txt file to state for train, val and test how many subjects there are for each age group and list the fnames that are in that group

        age_range = self._config['data']['dataset_age_range']
        age_lower, age_upper = map(int, age_range.split('-'))

        storage_path = os.path.join(precomputed_storage_path, f'{self._data_type}_age_split_fnames_{age_range}.txt')
        
        if not os.path.exists(storage_path):

            # 1) integer-binned summary: split -> age(int) -> list[fname]
            split_age_fnames = {'train': {}, 'val': {}, 'test': {}}
            for split, age, fname in zip(data_type_list, ages_decimals_list, fname_list):
                age_int = int(np.floor(age))
                age_int = max(age_lower, min(age_int, age_upper))
                if split not in split_age_fnames:
                    split_age_fnames[split] = {}
                split_dict = split_age_fnames[split]
                if age_int not in split_dict:
                    split_dict[age_int] = []
                split_dict[age_int].append(fname)

            # 2) decimal-age summary: split -> exact_age(float) -> list[fname]
            split_decimal_fnames = {'train': {}, 'val': {}, 'test': {}}
            for split, age, fname in zip(data_type_list, ages_decimals_list, fname_list):
                if split not in split_decimal_fnames:
                    split_decimal_fnames[split] = {}
                # use the raw float as key; you can round if needed, e.g. round(age, 3)
                age_key = float(age)
                if int(age_key) > 4:
                    continue
                if age_key not in split_decimal_fnames[split]:
                    split_decimal_fnames[split][age_key] = []
                split_decimal_fnames[split][age_key].append(fname)

            
            with open(storage_path, 'w') as f:
                f.write(f'Dataset type: {self._data_type}\n')
                f.write(f'Age range: {age_range}\n')
                f.write(f'Total subjects: {total_subjects}\n\n')

                # --- integer-binned section (existing behaviour) ---
                for split in ['train', 'val', 'test']:
                    if split not in split_age_fnames:
                        continue
                    f.write(f'=== {split.upper()} (integer bins, floor(age)) ===\n')
                    for age_int in range(age_lower, age_upper + 1):
                        fnames = split_age_fnames[split].get(age_int, [])
                        f.write(f'Age {age_int}: count={len(fnames)}\n')
                        if fnames:
                            f.write('  fnames: ' + ', '.join(str(x) for x in fnames) + '\n')
                    f.write('\n')

                # --- decimal-age section (exact ages, arbitrary counts) ---
                for split in ['train', 'val', 'test']:
                    if split not in split_decimal_fnames:
                        continue
                    f.write(f'=== {split.upper()} (exact decimal ages) ===\n')
                    # sort by age
                    for age_key in sorted(split_decimal_fnames[split].keys()):
                        fnames = split_decimal_fnames[split][age_key]
                        f.write(f'Age {age_key:.3f}: count={len(fnames)}\n')
                        f.write('  fnames: ' + ', '.join(str(x) for x in fnames) + '\n')
                    f.write('\n')

            print(f"Wrote age/split summary to {storage_path}")

    # create age distribution graph of all data 

        # age_range = self._config['data']['dataset_age_range']
        # age_lower, age_upper = map(int, age_range.split('-'))

        storage_path = os.path.join(precomputed_storage_path, f'{self._data_type}_age_distribution_{age_range}.png')

        if self._data_type == 'lyhm':
            bins_num = 12
        else:
            bins_num = age_upper-age_lower + 2

        # Define the bin edges
        bin_edges = np.linspace(min(ages_list), max(ages_list)+1, bins_num)
        bin_labels = [f"{int(bin_edges[i])}" for i in range(len(bin_edges))]
        label_points = [bin_edges[i] for i in range(len(bin_edges))]
        mid_points = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]

        if not os.path.exists(storage_path):

            # Plot the histogram - 0-2 mean up to and including 2 years old for example
            plt.figure(figsize=(10,6))
            n, bins, patches = plt.hist(ages_list, bins=bin_edges, alpha=0.7, edgecolor="k")
            # plt.hist(ages_list, bins=bin_edges, alpha=0.7)
            plt.xticks(label_points, bin_labels)
            plt.legend(loc='upper right')
            plt.xlabel("Age (years)")
            plt.ylabel("Frequency")
            plt.title("Age Distribution")

            # Annotate the bars with the frequency count and percentage
            for i in range(len(n)):
                plt.text(mid_points[i], n[i] + 5, f"{int(n[i])}\n({(n[i] / len(ages_list)) * 100:.1f}%)", 
                        ha='center', va='bottom', color='black', fontsize=9)

            plt.annotate(f'Total number of subjects: {total_subjects}', xy=(0.75, 0.95), xycoords='axes fraction')
            plt.tight_layout()
            plt.savefig(storage_path)
            plt.clf()

        else:
            print(f"{storage_path} already exists.")

        # self.log['dataset/distribution'].upload(storage_path)

    # create age split for train, val & test graph if it does not already exist 

        storage_path = os.path.join(precomputed_storage_path, f'{self._data_type}_data_split_{age_range}.png')

        if not os.path.exists(storage_path):

            # Initialize separate lists for train, test, and val ages
            train_ages = []
            test_ages = []
            val_ages = []

            # Iterate over data_types and ages simultaneously
            for data_type, age in zip(data_type_list, ages_list):
                if data_type == 'train':
                    train_ages.append(age)
                elif data_type == 'test':
                    test_ages.append(age)
                elif data_type == 'val':
                    val_ages.append(age)
            plt.clf()


            # # Determine the overall min and max age across all datasets
            # overall_min_age = min(ages_list)
            # overall_max_age = max(ages_list)

            # # Define the bin edges
            # bin_edges = np.linspace(overall_min_age, overall_max_age, bins_num)

            # # Plotting (overlaying bars)
            # plt.figure(figsize=(10, 6))
            # plt.hist(train_ages, bins=bin_edges, alpha=0.5, label='Train', edgecolor='black')
            # plt.hist(val_ages, bins=bin_edges, alpha=0.5, label='Validation', edgecolor='black')
            # plt.hist(test_ages, bins=bin_edges, alpha=0.5, label='Test', edgecolor='black')
            # plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8))
            # plt.xlabel('Age')
            # plt.ylabel('Frequency')
            # plt.title('Age distribution in Train, Validation and Test sets')

            # Calculate the histogram counts for each split
            n_train, _ = np.histogram(train_ages, bins=bin_edges)
            n_val, _ = np.histogram(val_ages, bins=bin_edges)
            n_test, _ = np.histogram(test_ages, bins=bin_edges)

            # Plotting (stacking bars)
            plt.figure(figsize=(10, 6))
            plt.hist([train_ages, val_ages, test_ages], bins=bin_edges, stacked=True, label=['Train', 'Validation', 'Test'], edgecolor='black', alpha=0.5)
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
            plt.xlabel('Age (years)')
            plt.ylabel('Frequency')
            plt.title('Train, Validation and Test sets split wrt Age')

            bin_labels = [f"{int(bin_edges[i])}" for i in range(len(bin_edges))]
            label_points = [bin_edges[i] for i in range(len(bin_edges))]
            plt.xticks(label_points, bin_labels)

            # Annotate the bars with the total count of subjects for each age group
            mid_points = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
            for i in range(len(mid_points)):
                total_count = int(n_train[i] + n_val[i] + n_test[i])  # Sum of counts across train, val, and test
                plt.text(mid_points[i], total_count + 0, f"{total_count}", 
                        ha='center', va='bottom', color='black', fontsize=9)

            if 'combined' in self._data_type:
                x = 0.02
            else:
                x = 0.60

            # Annotate min and max for each set
            plt.annotate(f'Total number of subjects: {total_subjects}', xy=(x, 0.95), xycoords='axes fraction')
            plt.annotate(f'Train min: {min(train_ages)}, max: {max(train_ages)}, count: {len(train_ages)}', xy=(x, 0.90), xycoords='axes fraction')
            plt.annotate(f'Validation min: {min(val_ages)}, max: {max(val_ages)}, count: {len(val_ages)}', xy=(x, 0.85), xycoords='axes fraction')
            plt.annotate(f'Test min: {min(test_ages)}, max: {max(test_ages)}, count: {len(test_ages)}', xy=(x, 0.80), xycoords='axes fraction')

            plt.savefig(storage_path)
        else:
            print(f"{storage_path} already exists.")

    # plot the original dataset split against age

        storage_path = os.path.join(precomputed_storage_path, f'{self._data_type}_dataset_split_{age_range}.png')

        # bin_edges = bin_edges - 1

        if not os.path.exists(storage_path):

            # Initialize separate lists for train, test, and val ages
            lyhm = []
            lsfm = []
            necker = []
            facescape = []
            mimicme = []

            # Iterate over data_types and ages simultaneously
            for dataset, age in zip(dataset_list, ages_list):
                if dataset == 'LYHM':
                    lyhm.append(age)
                elif dataset == 'LSFM':
                    lsfm.append(age)
                elif dataset == 'Paeds':
                    necker.append(age)
                elif dataset == 'FaceScape':
                    facescape.append(age)
                elif dataset == 'MimicMe':
                    mimicme.append(age)
            plt.clf()

            # Calculate the histogram counts for each split
            n_lyhm, _ = np.histogram(lyhm, bins=bin_edges)
            n_lsfm, _ = np.histogram(lsfm, bins=bin_edges)
            n_necker, _ = np.histogram(necker, bins=bin_edges)
            n_facescape, _ = np.histogram(facescape, bins=bin_edges)
            n_mimicme, _ = np.histogram(mimicme, bins=bin_edges)

            # Plotting (stacking bars)
            plt.figure(figsize=(10, 6))
            if 'combine' in self._data_type:
                plt.hist([lyhm, lsfm, necker, facescape, mimicme], bins=bin_edges, stacked=True, label=['LYHM', 'LSFM', 'Necker', 'FaceScape', 'MimicMe'], edgecolor='black', alpha=0.5)
            else:
                plt.hist([lsfm, necker], bins=bin_edges, stacked=True, label=['LSFM', 'Necker'], edgecolor='black', alpha=0.5)
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
            plt.xlabel('Age (years)')
            plt.ylabel('Frequency')
            plt.title('Dataset Origin Distribution Stacked by Age')

            bin_labels = [f"{int(bin_edges[i])}" for i in range(len(bin_edges))]
            label_points = [bin_edges[i] for i in range(len(bin_edges))]
            plt.xticks(label_points, bin_labels)

            # Annotate the bars with the total count of subjects for each age group
            mid_points = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
            for i in range(len(mid_points)):
                total_count = int(n_lyhm[i] + n_lsfm[i] + n_necker[i] + n_facescape[i] + n_mimicme[i])  # Sum of counts across train, val, and test
                plt.text(mid_points[i], total_count + 0, f"{total_count}", 
                        ha='center', va='bottom', color='black', fontsize=9)

            # Annotate min and max for each set
            plt.annotate(f'Total number of subjects: {total_subjects}', xy=(x, 0.95), xycoords='axes fraction')
            if 'combine' in self._data_type:
                plt.annotate(f'LYHM count: {len(lyhm)}', xy=(x, 0.90), xycoords='axes fraction')
            plt.annotate(f'LSFM count: {len(lsfm)}', xy=(x, 0.85), xycoords='axes fraction')
            plt.annotate(f'Necker count: {len(necker)}', xy=(x, 0.80), xycoords='axes fraction')
            if 'combine' in self._data_type:
                plt.annotate(f'FaceScape count: {len(facescape)}', xy=(x, 0.75), xycoords='axes fraction')
                plt.annotate(f'MimicMe count: {len(mimicme)}', xy=(x, 0.70), xycoords='axes fraction')

            plt.savefig(storage_path)
        else:
            print(f"{storage_path} already exists.")

        # self.log['dataset/distribution_split'].upload(storage_path)

    def age_prediction_MLP(self, train_loader, val_loader):
        """
        This function trains a MLP model to predict the age of the subjects based on the feature latents. 

        If disentanglement is successful, the model should NOT be able to predict the age of the subjects based on the feature latents.

        Output: plot of training loss and scatter plot of predicted age against ground truth age
        """

        train_feature_latents, _, train_gt_age, _, _ = self.process_data(train_loader, datasets=None, diagonal=True)
        val_feature_latents, _, val_gt_age, _, _ = self.process_data(val_loader, datasets=None, diagonal=True)

        sc = StandardScaler()
        train_feature_latents_scaled = sc.fit_transform(train_feature_latents)
        val_feature_latents_scaled = sc.transform(val_feature_latents)

        train_feature_latents = torch.tensor(train_feature_latents_scaled, dtype=torch.float32)
        val_feature_latents = torch.tensor(val_feature_latents_scaled, dtype=torch.float32)

        train_gt_age_tensor = torch.tensor(train_gt_age, dtype=torch.float32).view(-1, 1)
        val_gt_age_tensor = torch.tensor(val_gt_age, dtype=torch.float32).view(-1, 1)

        self.set_seed(42)

        train_dataset = TensorDataset(train_feature_latents, train_gt_age_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        input_size = train_feature_latents.shape[1]

        # Define the MLP model, loss function, and optimizer
        model = nn.Sequential(
            nn.Linear(input_size, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)) 
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        def train_model(model, criterion, optimizer, dataloader, epochs=100):
            model.train()
            losses = []
            for epoch in range(epochs):
                for inputs, targets in dataloader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                if epoch % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
                losses.append(loss.item())
                self.log['test/MLP_loss'].log(loss.item())
            return losses

        losses = train_model(model, criterion, optimizer, train_loader)

        # Plot the losses
        plt.figure()
        plt.clf()
        plt.plot(losses)
        plt.title('MLP Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        file_path = os.path.join(self._out_dir, 'mlp_training_loss.png')
        plt.savefig(file_path)
        self.log['test/mlp_training_loss'].upload(file_path)

        # Evaluate the model
        def evaluate_model(model, X, y):
            model.eval()
            with torch.no_grad():
                predictions = model(X).view(-1)
                mae = torch.mean(torch.abs(predictions - y.squeeze_(1)))
            return predictions.numpy(), mae.item()

        train_ages_pred, train_mean_age_diff = evaluate_model(model, train_feature_latents, train_gt_age_tensor)
        val_ages_pred, val_mean_age_diff = evaluate_model(model, val_feature_latents, val_gt_age_tensor)

        # Plot the results
        age_range = self._config['data']['dataset_age_range']
        min_age, max_age = map(int, age_range.split('-'))

        # Define base marker size
        base_marker_size = 50

        plt.figure(figsize=(6, 6))
        plt.clf()

        # Scatter plot with fixed marker sizes
        plt.scatter(train_gt_age, train_ages_pred, s=base_marker_size, color='yellow', marker='x', label='Train dataset')
        plt.scatter(val_gt_age, val_ages_pred, s=base_marker_size, color='green', marker='o', label='Validation dataset')
        plt.plot([0, max_age], [0, max_age], 'r--')

        # Add title, labels, and text
        plt.title('Age prediction on feature latents')
        plt.xlabel('Ground truth age (years)')
        plt.ylabel('Predicted age (years)')
        plt.text(0.25, 0.1, f'Mean absolute difference (train) = {round(train_mean_age_diff, 2)} years', transform=plt.gca().transAxes)
        plt.text(0.25, 0.05, f'Mean absolute difference (val) = {round(val_mean_age_diff, 2)} years', transform=plt.gca().transAxes)

        # Fixed marker sizes for legend
        legend_handles = [
            plt.scatter([], [], color='yellow', marker='x', s=base_marker_size, label='Train dataset'),
            plt.scatter([], [], color='green', marker='o', s=base_marker_size, label='Validation dataset')
        ]
        plt.legend(handles=legend_handles, loc='upper left')

        # Set ticks
        plt.xticks(range(0, 18))
        plt.yticks(range(0, 18))

        file_path = os.path.join(self._out_dir, f'mlp_age_prediction_{age_range}.png')
        plt.savefig(file_path)
        self.log['test/mlp_age_prediction'].upload(file_path)



    def age_prediction_MLP_OLD(self, train_loader, val_loader):

        """
        
        This function trains a MLP model to predict the age of the subjects based on the feature latents. 

        If disentanglement is successful, the model should NOT be able to predict the age of the subjects based on the feature latents.

        Output: plot of training loss and scatter plot of predicted age against ground truth age
        
        """

        train_feature_latents, _, train_gt_age, _, _ = self.process_data(train_loader, datasets=None, diagonal=True)
        val_feature_latents, _, val_gt_age, _, _ = self.process_data(val_loader, datasets=None, diagonal=True)

        # train_gt_age = mode(train_gt_age, axis=1).mode
        # val_gt_age = mode(val_gt_age, axis=1).mode

        sc = StandardScaler()
        train_feature_latents_scaled = sc.fit_transform(train_feature_latents)
        val_feature_latents_scaled = sc.transform(val_feature_latents)

        train_feature_latents = torch.tensor(train_feature_latents_scaled, dtype=torch.float32)
        val_feature_latents = torch.tensor(val_feature_latents_scaled, dtype=torch.float32)

        train_gt_age_tensor = torch.tensor(train_gt_age, dtype=torch.float32).view(-1, 1)
        val_gt_age_tensor = torch.tensor(val_gt_age, dtype=torch.float32).view(-1, 1)

        self.set_seed(42)

        train_dataset = TensorDataset(train_feature_latents, train_gt_age_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        input_size = train_feature_latents.shape[1]

        # Define the MLP model, loss function, and optimizer
        model = nn.Sequential(
            nn.Linear(input_size, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)) 
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        def train_model(model, criterion, optimizer, dataloader, epochs=100):
            model.train()
            losses = []
            for epoch in range(epochs):
                for inputs, targets in dataloader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                if epoch % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
                losses.append(loss.item())
                self.log['test/MLP_loss'].log(loss.item())
            return losses

        losses = train_model(model, criterion, optimizer, train_loader)

        # Plot the losses
        plt.figure()
        plt.clf()
        plt.plot(losses)
        plt.title('MLP Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        file_path = os.path.join(self._out_dir, 'mlp_training_loss.png')
        plt.savefig(file_path)
        self.log['test/mlp_training_loss'].upload(file_path)

        # Evaluate the model
        def evaluate_model(model, X, y):
            model.eval()
            with torch.no_grad():
                predictions = model(X).view(-1)
                mae = torch.mean(torch.abs(predictions - y.squeeze_(1)))
            return predictions.numpy(), mae.item()

        train_ages_pred, train_mean_age_diff = evaluate_model(model, train_feature_latents, train_gt_age_tensor)
        val_ages_pred, val_mean_age_diff = evaluate_model(model, val_feature_latents, val_gt_age_tensor)

        # Plot the results
        age_range = self._config['data']['dataset_age_range']
        min_age, max_age = map(int, age_range.split('-'))

        plt.figure(figsize=(5, 5))
        plt.clf()
        plt.scatter(train_gt_age, train_ages_pred, color='yellow', marker='x', label='Train dataset')
        plt.scatter(val_gt_age, val_ages_pred, color='green', label='Validation dataset')
        plt.plot([0, max_age], [0, max_age], 'r--')

        plt.title('Age prediction on feature latents')
        plt.xlabel('Ground truth age (years)')
        plt.ylabel('Predicted age (years)')
        plt.text(0.25, 0.1, f'Mean absolute difference (train) = {round(train_mean_age_diff, 2)} years', transform=plt.gca().transAxes)
        plt.text(0.25, 0.05, f'Mean absolute difference (val) = {round(val_mean_age_diff, 2)} years', transform=plt.gca().transAxes)
        plt.legend(loc='upper left')
        plt.xticks(range(0, 18))
        plt.yticks(range(0, 18))

        file_path = os.path.join(self._out_dir, f'mlp_age_prediction_{age_range}.png')

        plt.savefig(file_path)

        self.log['test/mlp_age_prediction'].upload(file_path)

    # def dataset_type(self, data_loader, datasets):

    #     data_dataset = []

    #     for batch in tqdm.tqdm(data_loader):

    #         file_name = batch.fname

    #         for fname in file_name:
    #             # dataset_id = fname.split('.')[0]
    #             dataset_name = datasets[datasets['id'] == int(fname)]['dataset'].values[0]
    #             data_dataset.append(dataset_name)

    #     return data_dataset



    def tsne_visualization(self, train_loader, val_loader, test_loader):
        """
        This function performs t-SNE on the feature latents and visualizes 
        their clustering based on age.


        Outputs:
        1. If `age_per_feature` is True:
        - Scatter plots for t-SNE results:
            - One for each identity latent region (non-age latents) wrt age and dataset origin.
            - One for all non-age latents combined wrt age and dataset origin.

        2. If `age_per_feature` is False:
        - Scatter plots for t-SNE results:
            - One for all non-age latents combined.
            - One for all non-age latents grouped by dataset.

        """

        random_state = 42
        self.set_seed(random_state)
        diagonal = True
        cmap='viridis'

        # read csv file
        datasets = pd.read_csv(self._config['data']['dataset_metadata_path'], usecols=['id', 'Dataset'])

        train_feature_latents, train_age_latents, train_gt_ages, _, train_dataset = self.process_data(train_loader, datasets=datasets, diagonal=diagonal)
        val_feature_latents, val_age_latents, val_gt_ages, _, val_dataset = self.process_data(val_loader, datasets=datasets,diagonal=diagonal)
        test_feature_latents, test_age_latents, test_gt_ages, _, test_dataset = self.process_data(test_loader, datasets=datasets,diagonal=diagonal)

        feature_latents = np.concatenate((train_feature_latents, val_feature_latents, test_feature_latents), axis=0)
        age_latents = np.concatenate((train_age_latents, val_age_latents, test_age_latents), axis=0)
        gt_ages = np.concatenate((train_gt_ages, val_gt_ages, test_gt_ages), axis=0)
        datasets = [train_dataset, val_dataset, test_dataset]
        datasets = np.concatenate(datasets, axis=0)

        if self._config['model']['age_per_feature']:
            range_size = self._config['model']['age_latent_size'] + 2
        else:
            range_size = 2

        for i in range(range_size):
            if self._config['model']['age_per_feature'] and i < self._config['model']['age_latent_size']:
                latents_per_feature = (self._manager.model_latent_size - self._config['model']['age_latent_size']) // self._config['model']['age_latent_size']
                subset_latents = feature_latents[:, (i*latents_per_feature):(i*latents_per_feature)+latents_per_feature]
                name = i
                gt_feature = gt_ages
                gt_dataset_str = datasets
                unique_values = np.unique(gt_dataset_str)
                value_to_number = {value: idx for idx, value in enumerate(unique_values)}
                gt_feature_numeric = np.array([value_to_number[value] for value in gt_dataset_str])
                gt_dataset = gt_feature_numeric
                cmap_dataset = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(unique_values))))
            elif i == self._config['model']['age_latent_size']:
                subset_latents = feature_latents
                name = "all"
                gt_feature = gt_ages
            else: 
                subset_latents = feature_latents
                name = "all_dataset"
                gt_feature_str = datasets
                unique_values = np.unique(gt_feature_str)
                value_to_number = {value: idx for idx, value in enumerate(unique_values)}
                gt_feature_numeric = np.array([value_to_number[value] for value in gt_feature_str])
                gt_feature = gt_feature_numeric
                cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(unique_values))))

            sc = StandardScaler()
            subset_latents_scaled = sc.fit_transform(subset_latents)

            tsne = TSNE(n_components=2, random_state=random_state)
            tsne_results = tsne.fit_transform(subset_latents_scaled)

            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=gt_feature, cmap=cmap, alpha=0.6)

            if name != "all_dataset":
                plt.colorbar(scatter, label='Age')
                plt.title(f't-SNE Visualization of Feature Latents Region {name} by Age')
            else:
                cbar = plt.colorbar(scatter, ticks=range(len(unique_values)), label='Datasets')
                cbar.ax.set_yticklabels(unique_values)
                plt.title(f't-SNE Visualization of Feature Latents Region {name} by Dataset')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')

            file_path = os.path.join(self._out_dir, f'tsne_feature_latents_subset_{name}.png')
            plt.savefig(file_path)
            self.log[f'test/tsne_feature_latents_subset_{name}'].upload(file_path)

            # if name == "all_dataset":
            #     file_path_svg = os.path.join(self._out_dir, f'tsne_feature_latents_subset_{name}.svg')
            #     plt.savefig(file_path_svg)

            plt.close()

            if self._config['model']['age_per_feature'] and i < self._config['model']['age_latent_size']:
                plt.figure(figsize=(8, 6))
                scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=gt_dataset, cmap=cmap_dataset, alpha=0.6)
                cbar = plt.colorbar(scatter, ticks=range(len(unique_values)), label='Datasets')
                cbar.ax.set_yticklabels(unique_values)
                plt.title(f't-SNE Visualization of Feature Latents Region {name} by Dataset')
                plt.xlabel('t-SNE Dimension 1')
                plt.ylabel('t-SNE Dimension 2')

                file_path = os.path.join(self._out_dir, f'tsne_feature_latents_subset_{name}_dataset.png')
                plt.savefig(file_path)
                # self.log[f'test/tsne_feature_latents_subset_{name}'].upload(file_path)

                plt.close()

        # # Perform t-SNE on age latents and visualize
        # tsne_age_latents = TSNE(n_components=2, random_state=random_state)
        # age_latents_scaled = sc.fit_transform(age_latents)
        # tsne_age_results = tsne_age_latents.fit_transform(age_latents_scaled)

        # plt.figure(figsize=(8, 6))
        # plt.title('t-SNE Visualization of Age Latents by Identity')
        # plt.xlabel('t-SNE Dimension 1')
        # plt.ylabel('t-SNE Dimension 2')

        # file_path = os.path.join(self._out_dir, 'tsne_age_latents_by_identity.png')
        # plt.savefig(file_path)
        # self.log['test/tsne_age_latents_by_identity'].upload(file_path)
        # plt.close()

        # plt.figure(figsize=(8, 6))
        # scatter = plt.scatter(tsne_age_results[:, 0], tsne_age_results[:, 1], c=gt_feature_numeric, cmap=cmap, alpha=0.6)
        # cbar = plt.colorbar(scatter, ticks=range(len(unique_values)), label='Datasets')
        # cbar.ax.set_yticklabels(unique_values)
        # plt.title('t-SNE Visualization of Age Latents by Dataset')
        # plt.xlabel('t-SNE Dimension 1')
        # plt.ylabel('t-SNE Dimension 2')

        # file_path = os.path.join(self._out_dir, 'tsne_age_latents_by_dataset.png')
        # plt.savefig(file_path)
        # self.log['test/tsne_age_latents_by_dataset'].upload(file_path)
        # plt.close()

    ##### STATISTICAL TESTS #####

    def compute_mi(self, latents, factors):
        """
        Compute Mutual Information Gap (MIG) - measures the gap between the top two mutual information scores.
        latents: (N, D) latent representations (e.g., identity latents)
        factors: (N, K) ground truth factors (e.g., age latents)
        """
        
        mi_matrix = np.zeros((latents.shape[1], factors.shape[1]))

        for i in range(latents.shape[1]): 
            for j in range(factors.shape[1]):  
                mi_matrix[i, j] = mutual_info_regression(latents[:, i].reshape(-1, 1), factors[:, j]).mean()

        mi = np.mean(mi_matrix)
        
        return mi
    
    def compute_dci(self, latents, factors):
        """
        Compute DCI (Disentanglement, Completeness, Informativeness) for the entire latent representations.
        latents: (N, D1) latent representations (e.g., identity latents)
        factors: (N, D2) ground truth factors (e.g., age latents)
        Returns:
            disentanglement: Disentanglement score
            completeness: Completeness score
            informativeness: Informativeness score
        """
        # Train a regressor to predict the entire factors from the entire latents
        model = GradientBoostingRegressor()
        model.fit(latents, factors)

        # Compute feature importances
        importance_matrix = np.abs(model.feature_importances_).reshape(-1, factors.shape[1])

        # Compute disentanglement
        disentanglement_scores = 1.0 - scipy.stats.entropy(importance_matrix.T + 1e-10, base=latents.shape[1])
        disentanglement = np.mean(disentanglement_scores)

        # Compute completeness
        completeness_scores = 1.0 - scipy.stats.entropy(importance_matrix + 1e-10, base=factors.shape[1])
        completeness = np.mean(completeness_scores)

        # Compute informativeness (R^2 score)
        informativeness = model.score(latents, factors)

        return disentanglement, completeness, informativeness
    
    def compute_sap(self, latents, factors, continuous_factors=True):
        """
        Compute SAP (Separated Attribute Predictability) - measures the difference in prediction error for the top two latent dimensions most predictive of each factor.
        latents: (N, D) latent representations (e.g., identity latents)
        factors: (N, K) ground truth factors (e.g., age latents)
        continuous_factors: Whether the factors are continuous (True) or discrete (False)
        """
        sap_scores = []
        for j in range(factors.shape[1]):  # Iterate over factors
            errors = []
            for i in range(latents.shape[1]):  # Iterate over latent dimensions
                model = LinearRegression() if continuous_factors else GradientBoostingRegressor()
                model.fit(latents[:, i].reshape(-1, 1), factors[:, j])
                predictions = model.predict(latents[:, i].reshape(-1, 1))
                error = mean_squared_error(factors[:, j], predictions)
                errors.append(error)
            errors = np.sort(errors)
            sap_scores.append(errors[1] - errors[0])  # Difference between top two errors
        return np.mean(sap_scores)

    def compute_beta_vae_score(self, latents, factors):
        """
        Compute Beta-VAE Score - measures disentanglement by training a classifier to predict factors from latents.
        latents: (N, D) latent representations (e.g., identity latents)
        factors: (N, K) ground truth factors (e.g., age latents)
        """
        accuracies = []
        for j in range(factors.shape[1]):  # Iterate over factors
            model = LinearRegression()
            model.fit(latents, factors[:, j])
            predictions = model.predict(latents)
            accuracy = mean_squared_error(factors[:, j], predictions)
            accuracies.append(accuracy)
        return np.mean(accuracies)

    # def stats_tests_correlation_new(self, train_loader, val_loader, test_loader):

    #     # Your numpy arrays
    #     age_latents = np.random.rand(332, 9)  # Replace with your actual data
    #     identity_latents = np.random.rand(332, 45)  # Replace with your actual data

    #     # SAP Metric
    #     sap_results = sap_score.compute_sap(
    #         mus_test=identity_latents,  # Learned representations
    #         ys=age_latents,             # Ground truth factors
    #         continuous_factors=True,     # Set to True if ground truth factors are continuous
    #         num_train=300,
    #         num_test=32
    #     )
    #     print("SAP Metric Results:", sap_results)

    #     # DCI Metric
    #     dci_results = dci.compute_dci(
    #         mus_test=identity_latents,  # Learned representations
    #         ys=age_latents              # Ground truth factors
        # )
        # print("DCI Metric Results:", dci_results)

    def stats_tests_correlation(self, train_loader, val_loader, test_loader):
        """
        Perform statistical tests to check if age is disentangled from feature latents.

        """

        # process_data outputs: feature_latents, age_latents, gt_ages, gt_ages_norm, data_dataset
        train_identity_latents, train_age_latents, _, _, _ = self.process_data(train_loader, datasets=None, diagonal=True)
        val_identity_latents, val_age_latents, _, _, _ = self.process_data(val_loader, datasets=None, diagonal=True)
        test_identity_latents, test_age_latents, _, _, _ = self.process_data(test_loader, datasets=None, diagonal=True)

        ####### DISENTANGLEMENT_LIB PYTORCH #######

        # from disentanglement_lib.mig import _compute_mig
        from disentanglement_lib.dci import _compute_dci
        from disentanglement_lib.sap_score import _compute_sap

        identity_latents_train_val = np.concatenate((train_identity_latents, val_identity_latents), axis=0)
        identity_latents_test = test_identity_latents
        age_latents_train_val = np.concatenate((train_age_latents, val_age_latents), axis=0)
        age_latents_test = test_age_latents

        ####### SAP and DCI SCORES #######

        sap_score = _compute_sap(identity_latents_train_val.T, age_latents_train_val.T, identity_latents_test.T, age_latents_test.T, continuous_factors=True)
        print("SAP Score (age in id):", sap_score)
        sap_score = list(sap_score.values())[0]
        self.log['test/sap_score_age_in_id'] = sap_score
        sap_score = _compute_sap(age_latents_train_val.T, identity_latents_train_val.T, age_latents_test.T, identity_latents_test.T, continuous_factors=True)
        print("SAP Score (id in age):", sap_score)
        sap_score = list(sap_score.values())[0]
        self.log['test/sap_score_id_in_age'] = sap_score

        # dci_score = _compute_dci(identity_latents_train_val.T, age_latents_train_val.T, identity_latents_test.T, age_latents_test.T)
        # dci_score = dci_score["disentanglement"]
        # print("DCI Score (age in id):", dci_score)
        # # self.log['test/dci_d_score_age_in_id'] = dci_score
        # dci_score = _compute_dci(age_latents_train_val.T, identity_latents_train_val.T, age_latents_test.T, identity_latents_test.T)
        # dci_score = dci_score["disentanglement"]
        # print("DCI Score (id in age):", dci_score)
        # # self.log['test/dci_d_score_id_in_age'] = dci_score


        # === FEATURE-LEVEL SAP TESTS ===
        # feature_sap_results_age_in_id = {}
        # feature_dci_results_id_in_age = {}
        feature_r2_results_id_in_age = {}
        feature_r2_results_age_in_id = {}
        features = ["Temporal", "Eyes", "Cheekbones", "Cheeks", "Jaw", "Forehead", "Chin", "Lips", "Nose"]

        # Assuming 45 id latents (5 per feature) and 9 age latents
        num_features = age_latents_train_val.shape[1]
        id_latent_size = identity_latents_train_val.shape[1]
        id_per_feature = id_latent_size // num_features

        model_age_in_id = LinearRegression()
        model_id_in_age = LinearRegression()

        for i in range(num_features):
            id_inds = list(range(i * id_per_feature, (i + 1) * id_per_feature))
            age_ind = i

            feature_name = features[i]
            id_train_sub = identity_latents_train_val[:, id_inds]
            id_test_sub = identity_latents_test[:, id_inds]
            age_train_sub = age_latents_train_val[:, age_ind]
            age_test_sub = age_latents_test[:, age_ind]

            age_train_sub = age_train_sub.reshape(-1, 1)
            age_test_sub = age_test_sub.reshape(-1, 1)

            # sap_feat_age_in_id = _compute_sap(id_train_sub.T, age_train_sub.T,
            #                         id_test_sub.T, age_test_sub.T,
            #                         continuous_factors=True)
            # sap_feat_val_age_in_id = list(sap_feat_age_in_id.values())[0]
            # feature_sap_results_age_in_id[feature_name] = sap_feat_val_age_in_id
            # self.log[f"test/feature_sap_age_in_id/{feature_name}"] = sap_feat_val_age_in_id
            # print(f"{feature_name} (age in id): SAP={sap_feat_val_age_in_id}")

            # dci_feat_id_in_age = _compute_dci(age_train_sub.T, id_train_sub.T,
            #                         age_test_sub.T, id_test_sub.T)
            # # dci_feat_val_id_in_age = list(dci_feat_id_in_age.values())[0]
            # dci_feat_val_id_in_age = dci_feat_id_in_age["disentanglement"]
            # feature_dci_results_id_in_age[feature_name] = dci_feat_val_id_in_age
            # # self.log[f"test/feature_sap_id_in_age/{feature_name}"] = sap_feat_val_id_in_age
            # # print(f"{feature_name} (id in age): DCI Disentanglement={dci_feat_val_id_in_age}")

            # Fit the model for "age in identity"
            model_age_in_id.fit(id_train_sub, age_train_sub)
            r2 = r2_score(age_test_sub, model_age_in_id.predict(id_test_sub), multioutput='variance_weighted')
            feature_r2_results_age_in_id[feature_name] = r2

            # Fit the model for "identity in age"
            model_id_in_age.fit(age_train_sub, id_train_sub)
            r2 = r2_score(id_test_sub, model_id_in_age.predict(age_test_sub), multioutput='variance_weighted')
            feature_r2_results_id_in_age[feature_name] = r2

        # print("Feature-level SAP results (age in id):", feature_sap_results_age_in_id)
        # print("Feature-level DCI results (id in age):", feature_dci_results_id_in_age)
        print("Feature-level R² results (age in id):", feature_r2_results_age_in_id)
        print("Feature-level R² results (id in age):", feature_r2_results_id_in_age)
        self.log["test/feature_r2_age_in_id"] = feature_r2_results_age_in_id
        self.log["test/feature_r2_id_in_age"] = feature_r2_results_id_in_age
    
        ####### MI SCORE #######

        # # Concatenate the feature latents and age labels from train, val, and test sets
        # identity_latents = np.concatenate((train_identity_latents, val_identity_latents, test_identity_latents), axis=0)
        # age_latents = np.concatenate((train_age_latents, val_age_latents, test_age_latents), axis=0)
        # # gt_age_norms = np.concatenate((train_gt_age_norm, val_gt_age_norm, test_gt_age_norm), axis=0)

        # # mig_score = _compute_mig(identity_latents.T, age_latents.T)
        # # print("MIG Score (id vs age):", mig_score)
        # # mig_score = _compute_mig(age_latents.T, identity_latents.T)
        # # print("MIG Score (age vs id):", mig_score)

        # # # not from dis_lib
        # # # MI: How much age information is in identity latents
        # # mi_matrix = np.zeros((identity_latents.shape[1], age_latents.shape[1]))
        # # for i in range(identity_latents.shape[1]):
        # #     for j in range(age_latents.shape[1]):
        # #         mi_matrix[i, j] = mutual_info_regression(identity_latents[:, i].reshape(-1, 1), age_latents[:, j]).mean()
        # # mi_score= np.mean(mi_matrix)
        # # print("MI Score:", mi_score)
        # # mi_score = float(mi_score)
        # # self.log["test/mi_score"].log([mi_score]) 


        ####### DISENTANGLEMENT_LIB TENSORFLOW #######

        # # Import gin and disentanglement_lib utilities
        # import gin
        # import disentanglement_lib.evaluation.metrics.utils as disentanglement_utils

        # # Register make_discretizer as an external configurable
        # gin.external_configurable(disentanglement_utils.make_discretizer, module='disentanglement_lib.utils')

        # # Set gin configuration for MIG
        # gin.bind_parameter('discretizer.num_bins', 10)
        # gin.bind_parameter('discretizer.discretizer_fn', disentanglement_utils.make_discretizer)

        # # === STEP 1: Debugging Latent Shapes ===
        # print("Identity latents shape:", identity_latents.shape)
        # print("Age latents shape:", age_latents.shape)

        # # === STEP 2: Dataset Wrapper ===
        # class SimpleDataset:
        #     def __init__(self, factors):
        #         self.factors = factors
        #         self.num_samples = self.factors.shape[0]

        #     def sample(self, num_samples, random_state):
        #         # Ensure we don't sample more than the available data
        #         if num_samples > self.num_samples:
        #             num_samples = self.num_samples
        #         idx = random_state.choice(self.num_samples, num_samples, replace=False)
        #         obs = idx[:, None]  # obs contains sample indices
        #         print("Sampled indices:", idx)  # Debugging
        #         print("Sampled factors shape:", self.factors[idx].shape)  # Debugging
        #         return obs, self.factors[idx]

        # # Create datasets for age and identity factors
        # identity_factors_dataset = SimpleDataset(identity_latents)
        # age_factors_dataset = SimpleDataset(age_latents)

        # # === STEP 3: Representation Function ===
        # def make_repr_fn(latents, name=""):
        #     def fn(obs):
        #         indices = obs[:, 0].astype(int)
        #         rep = latents[indices]
        #         print(f"[{name}] Representation shape: {rep.shape}")  # Debugging
        #         return rep
        #     return fn

        # # Create representation functions
        # identity_repr_fn = make_repr_fn(identity_latents, name="identity")
        # age_repr_fn = make_repr_fn(age_latents, name="age")

        # # === STEP 4: Compute Metrics ===
        # random_state = np.random.RandomState(0)
        # num_samples = identity_latents.shape[0]
        # num_train = int(num_samples * 0.8)
        # num_test = num_samples - num_train

        # results = {}

        # print("Debugging MIG Inputs:")
        # print("Age factors dataset shape:", age_factors_dataset.factors.shape)
        # print("Number of training samples:", num_train)
        # print("Random state:", random_state)

        # # --- Age info in Identity latents ---
        # results["mig_identity_age"] = mig.compute_mig(
        #     age_factors_dataset, identity_repr_fn, random_state, num_train=num_train
        # )['discrete_mig']
        # results["dci_identity_age"] = dci.compute_dci(
        #     age_factors_dataset, identity_repr_fn, random_state, num_train=num_train, num_test=num_test
        # )['disentanglement']
        # results["sap_identity_age"] = sap_score.compute_sap(
        #     age_factors_dataset, identity_repr_fn, random_state, num_train=num_train, num_test=num_test
        # )['SAP_score']
        # results["beta_identity_age"] = beta_vae.compute_beta_vae_sklearn(
        #     age_factors_dataset, identity_repr_fn, random_state, num_train=num_train, num_eval=num_test
        # )['eval_accuracy']

        # # --- Identity info in Age latents ---
        # results["mig_age_identity"] = mig.compute_mig(
        #     identity_factors_dataset, age_repr_fn, random_state, num_train=num_train
        # )['discrete_mig']
        # results["dci_age_identity"] = dci.compute_dci(
        #     identity_factors_dataset, age_repr_fn, random_state, num_train=num_train, num_test=num_test
        # )['disentanglement']
        # results["sap_age_identity"] = sap_score.compute_sap(
        #     identity_factors_dataset, age_repr_fn, random_state, num_train=num_train, num_test=num_test
        # )['SAP_score']
        # results["beta_age_identity"] = beta_vae.compute_beta_vae_sklearn(
        #     identity_factors_dataset, age_repr_fn, random_state, num_train=num_train, num_eval=num_test
        # )['eval_accuracy']

        # # === STEP 5: Print Results ===
        # for k, v in results.items():
        #     print(f"{k}: {v:.4f}")

        # # Log results
        # for k, v in results.items():
        #     self.log[f'test/{k}'] = v

        # ####### NEW IMPLEMENTATION #######

        # # age_min, age_max = map(int, self._config['data']['dataset_age_range'].split('-'))

        # # # Test 1: Age Information in Identity Latents
        # # print("Running Test 1: Age Information in Identity Latents")
        # # mig_score = self.compute_mi(identity_latents, age_latents)
        # # print("MIG Score (Age in Identity):", mig_score)

        # # dci_scores_dis, dci_scores_comp, dci_scores_info = self.compute_dci(identity_latents, age_latents)
        # # print("DCI Scores Disentanglement (Age in Identity):", dci_scores_dis)
        # # print("DCI Scores Completeness (Age in Identity):", dci_scores_comp)
        # # print("DCI Scores Informativeness (Age in Identity):", dci_scores_info)

        # sap_score = self._compute_sap(identity_latents, age_latents, continuous_factors=True)
        # print("SAP Score (Age in Identity):", sap_score)

        # beta_vae_score = self.compute_beta_vae_score(identity_latents, age_latents)
        # print("Beta-VAE Score (Age in Identity):", beta_vae_score)

        # # self.log['test/mig_score_age_in_identity'] = str(mig_score)
        # # self.log['test/dci_score_age_in_identity'] = str(dci_scores)
        # # self.log['test/sap_score_age_in_identity'] = str(sap_score)
        # # self.log['test/beta_vae_score_age_in_identity'] = str(beta_vae_score)

        # # Test 2: Identity Information in Age Latents
        # print("Running Test 2: Identity Information in Age Latents")
        # mig_score = self.compute_mi(age_latents, identity_latents)
        # # dci_scores_dis, dci_scores_comp, dci_scores_info = self.compute_dci(age_latents, identity_latents)
        # sap_score = self.compute_sap(age_latents, identity_latents, continuous_factors=True)
        # beta_vae_score = self.compute_beta_vae_score(age_latents, identity_latents)

        # print("MIG Score (Identity in Age):", mig_score)
        # # print("DCI Scores Disentanglement (Identity in Age):", dci_scores_dis)
        # # print("DCI Scores Completeness (Age in Identity):", dci_scores_comp)
        # # print("DCI Scores Informativeness (Age in Identity):", dci_scores_info)
        # print("SAP Score (Identity in Age):", sap_score)
        # print("Beta-VAE Score (Identity in Age):", beta_vae_score)

        # # self.log['test/mig_score_identity_in_age'] = str(mig_score)
        # # self.log['test/dci_score_identity_in_age'] = str(dci_scores)
        # # self.log['test/sap_score_identity_in_age'] = str(sap_score)
        # # self.log['test/beta_vae_score_identity_in_age'] = str(beta_vae_score)

        ####### OLD IMPLEMENTATION #######

        # if self._config['data']['swap_features']:
        #     latents_per_feature = (self._manager.model_latent_size - self._config['model']['age_latent_size']) // len(self._manager.latent_regions)
        #     num_groups = identity_latents.shape[1] // latents_per_feature  # should give 9 groups

        # age_latents_size = self._config['model']['age_latent_size']

        # if age_latents_size == 1:

        #     sap_score = sap(factors=age_latents, codes=identity_latents, continuous_factors=True, nb_bins=10, regression=True)

        #     # Mutual Information (all feature latents vs single age latent)
        #     mi_all_all = mutual_info_regression(identity_latents, age_latents.ravel())

        #     if self._config['data']['swap_features']:
        
        #         # Mutual Information (each subset of feature latents vs single age latent)
        #         mi_subsets_single_age = []
        #         for i in range(num_groups):
        #             mi = mutual_info_regression(identity_latents[:, i*latents_per_feature:(i+1)*latents_per_feature], age_latents.ravel())
        #             mi_subsets_single_age.append(mi)
                
        #         # Mutual Information (each feature latent vs single age latent)
        #         mi_each_feature_single_age = []
        #         for i in range(identity_latents.shape[1]):
        #             mi = mutual_info_regression(identity_latents[:, i].reshape(-1, 1), age_latents.ravel())
        #             mi_each_feature_single_age.append(mi)
            
        # else:

        #     sap_score = sap(factors=age_latents, codes=identity_latents, continuous_factors=True, nb_bins=10, regression=True)

        #     # Mutual Information (all feature latents vs all age latents) 
        #     # compute the MI for each pair of shape latent and age latent.
        #     mi_matrix = np.zeros((identity_latents.shape[1], age_latents.shape[1]))

        #     for i in range(identity_latents.shape[1]): 
        #         for j in range(age_latents.shape[1]):  
        #             mi_matrix[i, j] = mutual_info_regression(identity_latents[:, i].reshape(-1, 1), age_latents[:, j]).mean()

        #     mi_all_all = np.mean(mi_matrix)

        #     if self._config['data']['swap_features']:

        #         # Mutual Information (each subset of feature latents vs each single age latent)
        #         mi_subsets_each_age = []
        #         for i in range(num_groups):
        #             mi = mutual_info_regression(identity_latents[:, i*latents_per_feature:(i+1)*latents_per_feature], age_latents[:, i]).mean()
        #             mi_subsets_each_age.append(mi)
        #         mi_subsets_each_age_mean = np.array(mi_subsets_each_age).mean()
        #         mi_subsets_each_age = [[mi_subsets_each_age_mean], mi_subsets_each_age]
                
        #         # Mutual Information (each feature latent vs corresponding age latent)
        #         mi_each_feature_each_age = []
        #         for i in range(num_groups):
        #             for j in range(latents_per_feature):
        #                 mi = mutual_info_regression(identity_latents[:, i*latents_per_feature + j].reshape(-1, 1), age_latents[:, i])
        #                 mi_each_feature_each_age.append(mi[0])
        #         mi_each_feature_each_age_mean = np.array(mi_each_feature_each_age).mean()
        #         mi_each_feature_each_age = [[mi_each_feature_each_age_mean], mi_each_feature_each_age]

        # # All values desired to be low
        # print("- SAP Score:")
        # print(sap_score)

        # print("- Mutual Information between all shape_latents and all age_latents:")
        # print(mi_all_all)

        # self.log['test/sap'] = str(sap_score)
        # self.log['test/mi_all_all'] = str(mi_all_all)

        # if self._config['data']['swap_features'] and age_latents_size != 1:
        #     self.log['test/mi_subsets_each_age'] = str(mi_subsets_each_age)
        #     self.log['test/mi_each_feature_each_age'] = str(mi_each_feature_each_age)



    def proportions(self, data_loader):
        """
        This function takes the first batch from the input data, passes them through the encoder,
        changes the age latents to all equal the same value for all integer ages between 'age_range',
        and then passes all these changed ages data through the generator.
        """

        self.set_seed(42)

        folder_path = None
        if 'combined' in self._config['data']['dataset_type']:
            dataset_type = 'combined'
        else:
            dataset_type = "not-combined"

        storage_path = os.path.join(self._manager._precomputed_storage_path, f'normalise_age_{self._data_type}.pkl')
        with open(storage_path, 'rb') as file:
            age_train_mean, age_train_std = pickle.load(file)

        age_range = self._config['data']['dataset_age_range']
        age_lower, age_upper = map(int, age_range.split('-'))

        all_gen_verts = []
        all_mesh_names = []

        count = 0

        for batch in tqdm.tqdm(data_loader):
            gt_ages = batch.age.numpy()
            file_names = batch.fname

            if self._config['data']['swap_features']:
                batch = batch.x[self._manager.batch_diagonal_idx, ::]
            else:
                batch = batch.x

            count += len(batch)

            z = self._manager.encode(batch.to(self._device)).detach()
            z_copy = z.clone()

            age_latent_size = self._config['model']['age_latent_size']

            for age in range(age_lower, age_upper + 1):
                age_latent_value = (age - age_train_mean) / age_train_std
                z_copy[:, -age_latent_size:] = age_latent_value

                gen_verts = self._manager.generate(z_copy.to(self._device))

                if self._normalized_data:
                    gen_verts = self._unnormalize_verts(gen_verts)

                if dataset_type == 'combined':
                    mesh_names = [f'{file_name.item()}_{age}' for file_name in file_names]
                else:
                    mesh_names = [f'{file_name}_{age}' for file_name in file_names]

                all_gen_verts.append(gen_verts)
                all_mesh_names.append(mesh_names)

        all_gen_verts = torch.cat(all_gen_verts, dim=0)
        all_mesh_names = [item for sublist in all_mesh_names for item in sublist]

        template_path = self._config['data']['template_path']
        output_directory = self._out_dir
        calculate_distances_in_folder(folder_path, template_path, all_gen_verts, all_mesh_names, dataset_type, output_directory)
        add_proportions_age_gender_to_csv(folder_path, dataset_type, output_directory)
        distance_proportion_averages(dataset_type, output_directory)

        # renderings = self._manager.render(all_gen_verts).cpu()
        # grid = make_grid(renderings, padding=10, pad_value=1, nrow=batch.size(0))
        # file_path = os.path.join(self._out_dir, 'age_latent_modification.png')
        # save_image(grid, file_path)
        # self.log['test/age_latent_modification'].upload(file_path)

        # Save gt_ages to results.txt
        results_file_path = os.path.join(self._out_dir, 'results.txt')
        with open(results_file_path, 'a') as file:
            file.write('gt_age for proportions test\n')
            file.write(str(gt_ages.tolist()) + '\n\n')

    def plot_proportions(self):
        output_directory = self._out_dir
        if 'combined' in self._config['data']['dataset_type']:
            dataset_type = 'combined'
        else:
            dataset_type = "not-combined"

        # Read the CSV files
        csv_path1 = os.path.join(output_directory, f"{dataset_type}_proportion_averages.csv")
        df1 = pd.read_csv(csv_path1)

        csv_path2 = os.path.join("measurements", f"{dataset_type}_proportion_averages.csv")
        df2 = pd.read_csv(csv_path2)

        csv_path3 = "measurements/farkas_proportion_averages.csv"
        df3 = pd.read_csv(csv_path3)

        csv_path4 = "measurements/facebase_proportion_averages.csv"
        df4 = pd.read_csv(csv_path4)

        age_range = self._config['data']['dataset_age_range']
        age_lower, age_upper = map(int, age_range.split('-'))
        df2 = df2[(df2['age'] >= age_lower) & (df2['age'] <= age_upper)]
        df3 = df3[(df3['age'] >= age_lower) & (df3['age'] <= age_upper)]
        df4 = df4[(df4['age'] >= age_lower) & (df4['age'] <= age_upper)]

        def calculate_mse(df1, df2, proportion_name):
            mse = mean_squared_error(df2[proportion_name], df1[proportion_name])
            return mse

        proportion_columns = ['n-sto:n-gn', 'n-sto:sto-gn', 'sto-gn:n-gn', 'zy_right-zy_left:go-right-go-left']
        
        # Plot the data
        def plot_comparison(df_a, df_b, proportion_name, label_a, label_b, output_directory, mse):
            plt.figure(figsize=(10, 6))
            colors = {'male': 'blue', 'female': 'green'}
            for gender in df_a['gender'].unique():
                gender_data_a = df_a[df_a['gender'] == gender]
                gender_data_b = df_b[df_b['gender'] == gender]
                plt.plot(gender_data_a['age'], gender_data_a[proportion_name], label=f'{gender} ({label_a})', linestyle='-', color=colors[gender])
                plt.plot(gender_data_b['age'], gender_data_b[proportion_name], label=f'{gender} ({label_b})', linestyle='--', color=colors[gender])
            plt.xlabel('Age', fontsize=22)
            plt.ylabel('Proportion Value', fontsize=22)
            plt.title(f'{label_a} vs {label_b}', fontsize=26)
            plt.suptitle(f'Proportion {proportion_name}', fontsize=10)
            if mse is not None:
                plt.text(0.02, 1.05, f'MSE: {mse:.5f}', fontsize=18, color='black', transform=plt.gca().transAxes, ha='left', va='top')
            plt.legend(fontsize=22, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True)
            plt.xticks(range(0, 18), fontsize=22)  # Set x-axis ticks to show each integer value from 0 to 17
            plt.yticks(fontsize=22)
            file_path = os.path.join(output_directory, f'proportions_{proportion_name}_{label_a}_vs_{label_b}.png')
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
            self.log[f'test/proportions_{proportion_name}_{label_a}_vs_{label_b}'].upload(file_path)

        # Plot each proportion for the three comparisons
        for proportion in proportion_columns:
            # Farkas vs. FaceBase
            mse = None
            plot_comparison(df3, df4, proportion, 'Farkas', 'FaceBase', output_directory, mse)

            # FaceBase vs. Dataset
            mse = None
            plot_comparison(df4, df2, proportion, 'FaceBase', 'Dataset', output_directory, mse)

            # Dataset vs. Model
            mse = calculate_mse(df2, df1, proportion)
            plot_comparison(df2, df1, proportion, 'Dataset', 'Model', output_directory, mse)


    # def relatives_aging_diff(self):

    #     # Read the CSV file
    #     relatives_metadata = pd.read_csv('preprocessing_data/relatives_metadata.csv', usecols=['name', 'id', 'age', 'gender'])

    #     if 'combined' in self._config['data']['dataset_type']:
    #         dataset_type = 'combined'
    #     else:
    #         dataset_type = "not_combined"

    #     # Get all mesh files in the specified folder
    #     mesh_files = glob(f"/raid/compass/athena/data/relatives/{dataset_type}/*.obj")

    #     # Group by unique 'name'
    #     grouped = relatives_metadata.groupby('name')

    #     # get age normalization stats
    #     age_norm_path = os.path.join(self._manager._precomputed_storage_path, f'normalise_age_{self._data_type}.pkl')
    #     try:
    #         with open(age_norm_path, 'rb') as file:
    #             age_mean, age_std = \
    #                 pickle.load(file)
    #     except FileNotFoundError:
    #         print("Could not find normalise stats file")

    #     # Generate difference maps and save images

    #     # make relatives directory in model output directory
    #     output_dir = os.path.join(self._out_dir, 'relatives_aging')
    #     os.makedirs(output_dir, exist_ok=True)

    #     # Iterate through each name
    #     for name, group in grouped:

    #         # if name != 'livia':
    #         #     continue

    #         original_rendering = [] # orginal meshes before passing through model
    #         aged_rendering = [] # mesh after changing age latens
    #         difference_rendering = [] # different between the youngest mesh and the aged meshes after passing all through model 
    #         difference_original_rendering = [] # difference between the original meshes and the aged meshes after passing all through model

    #         # Get all mesh files that match the 'id' in the group
    #         matching_files = [f for f in mesh_files if any(str(id) in f for id in group['id'])]

    #         # only use people with more than one mesh
    #         if len(matching_files) < 2:
    #             continue

    #         # Sort the group by age to get the youngest
    #         group = group.sort_values(by='age')
    #         youngest = group.iloc[0]
    #         others = group.iloc[1:]

    #         # Load the youngest mesh
    #         youngest_original_mesh_path = [f for f in matching_files if str(youngest['id']) in f][0]
    #         youngest_original_mesh = trimesh.load_mesh(youngest_original_mesh_path, 'obj', process=False)
    #         youngest_original_verts = torch.tensor(youngest_original_mesh.vertices, dtype=torch.float, requires_grad=False, device=self._device)
    #         youngest_original_verts_unqueeze = youngest_original_verts.unsqueeze(0)
    #         youngest_original_verts_rend = self._manager.render(youngest_original_verts.unsqueeze(0)).cpu()
    #         original_rendering.append(youngest_original_verts_rend)

    #         # Normalize the youngest vertices
    #         if self._normalized_data:
    #             youngest_original_verts_norm = (youngest_original_verts - self._norm_dict['mean'].to(self._device)) / self._norm_dict['std'].to(self._device)

    #         # Encode and decode (generate) the youngest vertices
    #         z = self._manager.encode(youngest_original_verts_norm.unsqueeze(0).to(self._device)).detach()
    #         youngest_verts = self._manager.generate(z.to(self._device))

    #         # Unnormalize the decoder output and render to mesh
    #         if self._normalized_data:
    #             youngest_verts = self._unnormalize_verts(youngest_verts)
    #         youngest_rendering = self._manager.render(youngest_verts).cpu()
    #         aged_rendering.append(youngest_rendering)

    #         # Compute differences between the model output youngest and itself (should be zero)
    #         differences = self._manager.compute_vertex_errors(youngest_verts, youngest_verts)
    #         difference_render = self._manager.render(youngest_verts, differences, error_max_scale=5).cpu().detach()
    #         difference_rendering.append(difference_render)

    #         # Compute differences between the model output youngest and original youngest before being passed through the model
    #         differences_original = self._manager.compute_vertex_errors(youngest_original_verts_unqueeze, youngest_verts)
    #         difference_original_render = self._manager.render(youngest_original_verts_unqueeze, differences_original, error_max_scale=5).cpu().detach()
    #         difference_original_rendering.append(difference_original_render)

    #         # Iterate through the other ages for the same name 
    #         for _, row in others.iterrows():
    #             # replace age latents with age of next mesh of same person
    #             age_latent_value = (row['age'] - age_mean) / age_std
    #             z_copy = z.clone()
    #             z_copy[:, -self._config['model']['age_latent_size']:] = age_latent_value

    #             # Generate the aged vertices and render to mesh
    #             aged_verts = self._manager.generate(z_copy.to(self._device))
    #             if self._normalized_data:
    #                 aged_verts = self._unnormalize_verts(aged_verts)
    #             aged_render = self._manager.render(aged_verts).cpu()
    #             aged_rendering.append(aged_render)

    #             # Load the original mesh for the current row
    #             original_id = row['id']
    #             original_mesh_path = [f for f in matching_files if str(original_id) in f][0]
    #             original_mesh = trimesh.load_mesh(original_mesh_path, 'obj', process=False)
    #             original_verts = torch.tensor(original_mesh.vertices, dtype=torch.float, requires_grad=False, device=self._device)

    #             # Normalize the original vertices
    #             if self._normalized_data:
    #                 original_verts = (original_verts - self._norm_dict['mean'].to(self._device)) / self._norm_dict['std'].to(self._device)

    #             # Compute the differences between the aged mesh and the youngest mesh (both after passing through the model)
    #             differences = self._manager.compute_vertex_errors(aged_verts, youngest_verts)
    #             difference_render = self._manager.render(aged_verts, differences, error_max_scale=5).cpu().detach()
    #             difference_rendering.append(difference_render)

    #             # Compute differences between the model output aged and original aged before being passed through the model
    #             differences_original = self._manager.compute_vertex_errors(aged_verts, original_verts)
    #             difference_original_render = self._manager.render(aged_verts, differences_original, error_max_scale=5).cpu().detach()
    #             difference_original_rendering.append(difference_original_render)

    #             # Render the original meshes
    #             if self._normalized_data:
    #                 original_verts = self._unnormalize_verts(original_verts)
    #             original_render = self._manager.render(original_verts.unsqueeze(0)).cpu()
    #             original_rendering.append(original_render)

    #         max_batch_size = len(original_rendering)

    #         # Concatenate the lists of tensors along the batch dimension (dim=0)
    #         original_rendering_tensor = torch.cat(original_rendering, dim=0)
    #         aged_rendering_tensor = torch.cat(aged_rendering, dim=0)
    #         difference_rendering_tensor = torch.cat(difference_rendering, dim=0)
    #         difference_original_rendering_tensor = torch.cat(difference_original_rendering, dim=0)

    #         # Create a grid with the original meshes, aged meshes, and difference maps stacked on top of each other
    #         grid = make_grid(torch.cat([original_rendering_tensor, aged_rendering_tensor, difference_rendering_tensor, difference_original_rendering_tensor], dim=0), padding=10, pad_value=1, nrow=max_batch_size)
    #         file_path = os.path.join(output_dir, f'{name}_difference.png')
    #         save_image(grid, file_path)
    #         self.log[f'test/{name}_difference'].upload(file_path)

    def relatives_aging_diff_new(self):

        """"
        change: one ID image and one age image 

        1. original meshs, then model reconstructions of originals, then difference maps betweem the top and bottom and maybe differcne maps in different colours. 

        2. original meshes, ageing difference maps, youngest and age it, aging difference maps, oldest and de-age it, de-aging difference maps 
        
        """

        # Read the CSV file
        relatives_metadata = pd.read_csv('preprocessing_data/relatives_metadata.csv', usecols=['name', 'id', 'AgeYears', 'gender'])

        if 'combined' in self._config['data']['dataset_type']:
            dataset_type = 'combined'
        else:
            dataset_type = "not_combined"

        # # Get all mesh files in the specified folder
        mesh_file_path_lead = f'/raid/compass/athena/data/relatives/{dataset_type}'
        # mesh_files = glob(f"{mesh_file_path_lead}/*.obj")

        # Group by unique 'name'
        grouped = relatives_metadata.groupby('name')

        # get age normalization stats
        age_norm_path = os.path.join(self._manager._precomputed_storage_path, f'normalise_age_{self._data_type}.pkl')
        try:
            with open(age_norm_path, 'rb') as file:
                age_mean, age_std = \
                    pickle.load(file)
        except FileNotFoundError:
            print("Could not find normalise stats file")

        # Generate difference maps and save images

        # make relatives directory in model output directory
        output_dir = os.path.join(self._out_dir, 'relatives_aging')
        os.makedirs(output_dir, exist_ok=True)

        results_filename = os.path.join(self._out_dir, 'results.txt')
        if not os.path.exists(results_filename):
            with open(results_filename, 'w') as file:
                file.write('')

        z_age_means = []

        # Iterate through each name
        for name, group in grouped:

            # if name != 'livia':
            #     continue

            line_to_add = f'Name: {name}'
            with open(results_filename, 'a') as file:
                file.write(line_to_add)
                file.write('\n')

            original_rendering = [] # orginal meshes before passing through model
            reconstructed_rendering = [] # mesh after passing through model
            aged_rendering = [] # mesh after taking youngest and changing age latens to older 
            de_aged_rendering = [] # mesh after taking oldest and changing age latens to younger

            difference_pre_post_rendering = [] # difference between the original meshes and the reconstructed meshes
            difference_original_rendering = [] # different between the youngest original mesh and the aged original meshes
            # difference_reconstructed_rendering = [] # difference between the meshes after passing all through model
            difference_aged_rendering = [] # difference between the aged meshes 
            difference_de_aged_rendering = [] # difference between the de-aged meshes
            z_originals = [] # z values of the original meshes

            # # Get all mesh files that match the 'id' in the group
            # matching_files = [f for f in mesh_files if any(str(id) in f for id in group['id'])]

            # only use people with more than one mesh
            if len(group) < 2:
                continue

            # Sort the group by age to get the youngest
            group = group.sort_values(by='AgeYears')
            ages = group["AgeYears"].to_numpy()
            # youngest = group.iloc[0] # gives youngest age value
            # oldest = group.iloc[-1] # gives youngest age value
            # if len(group) > 2:
            #     others = group.iloc[1:-1]

            #### PRE & POST PROCESSING - ID TEST ####

            original_mesh_path = [f'{mesh_file_path_lead}/{id}.obj' for id in group['id']]
            for i in range(len(original_mesh_path)):
                # get the original mesh
                mesh_path = original_mesh_path[i]
                original_mesh = trimesh.load_mesh(mesh_path, 'obj', process=False)
                original_verts = torch.tensor(original_mesh.vertices, dtype=torch.float, requires_grad=False, device=self._device)
                original_verts_unqueeze = original_verts.unsqueeze(0)
                if i==0:
                    youngest_original_verts_unqueeze = original_verts_unqueeze
                original_verts_rend = self._manager.render(original_verts_unqueeze).cpu()
                original_rendering.append(original_verts_rend)

                # get differences between the original meshes
                differences = self._manager.compute_vertex_errors(original_verts_unqueeze, youngest_original_verts_unqueeze)
                difference_render = self._manager.render(original_verts_unqueeze, differences, error_max_scale=5).cpu().detach()
                difference_original_rendering.append(difference_render)

                # get reconstructed mesh
                if self._normalized_data:
                    original_verts_norm = (original_verts - self._norm_dict['mean'].to(self._device)) / self._norm_dict['std'].to(self._device)
                z = self._manager.encode(original_verts_norm.unsqueeze(0).to(self._device)).detach()
                z_originals.append(z)
                reconstructed_verts = self._manager.generate(z.to(self._device))
                if self._normalized_data:
                    reconstructed_verts = self._unnormalize_verts(reconstructed_verts)
                # if i==0:
                #     youngest_reconstructed_verts = reconstructed_verts
                reconstructed_verts_rendering = self._manager.render(reconstructed_verts).cpu()
                reconstructed_rendering.append(reconstructed_verts_rendering)

                # get differences between the original meshes and reconstructed meshes (use new colour)
                differences = self._manager.compute_vertex_errors(reconstructed_verts, original_verts_unqueeze)
                difference_render = self._manager.render(reconstructed_verts, differences, error_max_scale=5).cpu().detach()
                difference_pre_post_rendering.append(difference_render)

                # # get differences between the reconstructed meshes
                # differences = self._manager.compute_vertex_errors(reconstructed_verts, youngest_reconstructed_verts)
                # difference_render = self._manager.render(reconstructed_verts, differences, error_max_scale=5).cpu().detach()
                # difference_reconstructed_rendering.append(difference_render)

                z_age = z[:, -self._config['model']['age_latent_size']:] * age_std + age_mean
                z_age_mean = z_age.mean().item()
                line_to_add = f'Age: {ages[i]}, z_age: {z_age}, z_age_mean: {z_age_mean}'

                with open(results_filename, 'a') as file:
                    file.write(line_to_add)
                    file.write('\n' * 2)

                z_age_means.append({
                    'name': name,
                    'original_age': ages[i],
                    'z_ages_mean': z_age_mean
                })
            
            max_batch_size = len(original_rendering)
            
            # Concatenate the lists of tensors along the batch dimension (dim=0)
            original_rendering_tensor = torch.cat(original_rendering, dim=0)
            difference_original_rendering_tensor = torch.cat(difference_original_rendering, dim=0)
            reconstructed_rendering_tensor = torch.cat(reconstructed_rendering, dim=0)
            # difference_reconstructed_rendering_tensor = torch.cat(difference_reconstructed_rendering, dim=0)
            # cat_tensors = torch.cat([original_rendering_tensor, difference_original_rendering_tensor, reconstructed_rendering_tensor, difference_reconstructed_rendering_tensor], dim=0)
            difference_pre_post_rendering_tensor = torch.cat(difference_pre_post_rendering, dim=0)
            cat_tensors = torch.cat([original_rendering_tensor, reconstructed_rendering_tensor, difference_pre_post_rendering_tensor], dim=0)


            # Create a grid with the original meshes, aged meshes, and difference maps stacked on top of each other
            grid = make_grid(cat_tensors, padding=10, pad_value=1, nrow=max_batch_size)
            file_path = os.path.join(output_dir, f'{name}_pre_post.png')
            save_image(grid, file_path)
            self.log[f'{name}_pre_post'].upload(file_path)


            ##### AGE & DE-AGE MESHES - AGE TEST #####

            # take the youngest mesh and change the age latents to older ages without changing the youngest original z_age values 

            # # Load the youngest mesh
            z_youngest = z_originals[0]
            youngest_verts = self._manager.generate(z_youngest.to(self._device))

            # Unnormalize the decoder output and render to mesh
            if self._normalized_data:
                youngest_verts = self._unnormalize_verts(youngest_verts)
            youngest_rendering = self._manager.render(youngest_verts).cpu()
            aged_rendering.append(youngest_rendering)

            # Compute differences between the model output youngest and itself (should be zero)
            differences = self._manager.compute_vertex_errors(youngest_verts, youngest_verts)
            difference_render = self._manager.render(youngest_verts, differences, error_max_scale=5).cpu().detach()
            difference_aged_rendering.append(difference_render)

            # Iterate through the other ages for the same name 
            for _, row in group.iloc[1:].iterrows():
                # replace age latents with age of next mesh of same person
                age_latent_value = (row['AgeYears'] - age_mean) / age_std
                z_youngest_copy = z_youngest.clone()
                z_youngest_copy[:, -self._config['model']['age_latent_size']:] = age_latent_value

                # Generate the aged vertices and render to mesh
                aged_verts = self._manager.generate(z_youngest_copy.to(self._device))
                if self._normalized_data:
                    aged_verts = self._unnormalize_verts(aged_verts)
                aged_render = self._manager.render(aged_verts).cpu()
                aged_rendering.append(aged_render)

                # Compute differences between the model output youngest and itself (should be zero)
                differences = self._manager.compute_vertex_errors(aged_verts, youngest_verts)
                difference_render = self._manager.render(aged_verts, differences, error_max_scale=5).cpu().detach()
                difference_aged_rendering.append(difference_render)


            # take the oldest mesh and change the age latents to younger ages without changing the oldest original z_age values 

            # # Load the oldest mesh
            z_oldest = z_originals[-1]
            oldest_verts = self._manager.generate(z_oldest.to(self._device))

            # Unnormalize the decoder output and render to mesh
            if self._normalized_data:
                oldest_verts = self._unnormalize_verts(oldest_verts)

            verts = []
            # Iterate through the other ages for the same name 
            for _, row in group.iloc[:-1].iterrows():
                # replace age latents with age of younger version meshes of same person
                age_latent_value = (row['AgeYears'] - age_mean) / age_std
                z_oldest_copy = z_oldest.clone()
                z_oldest_copy[:, -self._config['model']['age_latent_size']:] = age_latent_value

                # Generate the aged vertices and render to mesh
                de_aged_verts = self._manager.generate(z_oldest_copy.to(self._device))
                if self._normalized_data:
                    de_aged_verts = self._unnormalize_verts(de_aged_verts)
                verts.append(de_aged_verts)
                de_aged_render = self._manager.render(de_aged_verts).cpu()
                de_aged_rendering.append(de_aged_render)

                # Compute differences between the oldest model output and de-aged mesh
                differences = self._manager.compute_vertex_errors(de_aged_verts, verts[0])
                difference_render = self._manager.render(de_aged_verts, differences, error_max_scale=5).cpu().detach()
                difference_de_aged_rendering.append(difference_render)

            oldest_rendering = self._manager.render(oldest_verts).cpu()
            de_aged_rendering.append(oldest_rendering)

            # Compute differences between the model output oldest and itself (should be zero)
            differences = self._manager.compute_vertex_errors(oldest_verts, verts[0])
            difference_render = self._manager.render(oldest_verts, differences, error_max_scale=5).cpu().detach()
            difference_de_aged_rendering.append(difference_render)

            max_batch_size = len(aged_rendering)
            
            # Concatenate the lists of tensors
            aged_rendering_tensor = torch.cat(aged_rendering, dim=0)
            difference_aged_rendering_tensor = torch.cat(difference_aged_rendering, dim=0)
            de_aged_rendering_tensor = torch.cat(de_aged_rendering, dim=0)
            difference_de_aged_rendering_tensor = torch.cat(difference_de_aged_rendering, dim=0)
            cat_tensors = torch.cat([original_rendering_tensor, difference_original_rendering_tensor,  aged_rendering_tensor, difference_aged_rendering_tensor, de_aged_rendering_tensor, difference_de_aged_rendering_tensor], dim=0)

            # Create a grid with the original meshes, aged meshes, and difference maps stacked on top of each other
            grid = make_grid(cat_tensors, padding=10, pad_value=1, nrow=max_batch_size)
            file_path = os.path.join(output_dir, f'{name}_age_de_age.png')
            save_image(grid, file_path)
            self.log[f'test/{name}_age_de_age'].upload(file_path)

        df = pd.DataFrame(z_age_means)

        # Plot the data
        plt.figure(figsize=(10, 6))

        # Plot original age vs z_ages_mean
        plt.scatter(df['original_age'], df['z_ages_mean'], label='Original Age vs Mean z_age', color='blue', marker='o')
        
        # Add labels for each point (name)
        for i, row in df.iterrows():
            plt.text(row['original_age'], row['z_ages_mean'], row['name'], fontsize=8, ha='right', va='bottom')
        
        # Add diagonal line
        plt.plot([0, 17], [0, 17], 'r--')

        # Add labels, legend, and title
        plt.xlabel('Ground Truth Age (Years)')
        plt.ylabel('Mean z_age')
        plt.title('GT Age vs Mean z_age')
        plt.legend()
        plt.grid(True)

        # Set x and y axis ticks to go in steps of 1
        plt.xticks(range(0, 18, 1))  # x-axis ticks from 0 to 17 in steps of 1
        plt.yticks(range(0, 18, 1))  # y-axis ticks from 0 to 17 in steps of 1


        # Save the plot
        plot_path = os.path.join(output_dir, 'age_vs_z_age_mean.png')
        plt.savefig(plot_path)

            
    def relatives_aging(self):
        if 'combined' in self._config['data']['dataset_type']:
            dataset_type = 'combined'
        else:
            dataset_type = "not_combined"
    
        relatives_dir = f'/raid/compass/athena/data/relatives/{dataset_type}/'
        output_dir = os.path.join(self._out_dir, 'relatives_aging')
        os.makedirs(output_dir, exist_ok=True)

        mesh_files = [f for f in os.listdir(relatives_dir) if f.endswith('_1.obj')]

        for mesh_file in tqdm.tqdm(mesh_files):
            print(mesh_file)
            mesh_path = os.path.join(relatives_dir, mesh_file)
            mesh = trimesh.load_mesh(mesh_path, 'obj', process=False)
            mesh_verts = torch.tensor(mesh.vertices, dtype=torch.float, requires_grad=False, device=self._device)

            if self._normalized_data:
                mesh_verts = (mesh_verts - self._norm_dict['mean'].to(self._device)) / self._norm_dict['std'].to(self._device)

            z = self._manager.encode(mesh_verts.unsqueeze(0).to(self._device)).detach()

            age_latent_size = self._config['model']['age_latent_size']
            z_mins = self.latent_stats['mins'][-age_latent_size:]
            z_maxs = self.latent_stats['maxs'][-age_latent_size:]
            n_steps = 18
            z_traversal = torch.stack([torch.linspace(z_min, z_max, n_steps) for z_min, z_max in zip(z_mins, z_maxs)], dim=1).to(self._device)

            all_frames = []
            for i in range(n_steps):
                z_copy = z.clone()
                z_copy[:, -age_latent_size:] = z_traversal[i]

                gen_verts = self._manager.generate(z_copy.to(self._device))

                if self._normalized_data:
                    gen_verts = self._unnormalize_verts(gen_verts)

                if i == 0:
                    gen_verts_youngest = gen_verts

                renderings = self._manager.render(gen_verts).detach().cpu()
                differences = self._manager.compute_vertex_errors(gen_verts, gen_verts_youngest)
                difference_renderings = self._manager.render(gen_verts, differences, error_max_scale=5).cpu().detach()

                combined_renderings = torch.cat([renderings, difference_renderings], dim=-1)
                all_frames.append(combined_renderings)

            file_path = os.path.join(output_dir, f'{mesh_file[:-4]}_aging.mp4')
            name = mesh_file.split('_')[1]
            file_path = os.path.join(output_dir, f'{name}_aging.mp4')
            write_video(file_path, torch.cat(all_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=4)
            # self.log[f'test/{mesh_file[:-4]}_aging.mp4'].upload(file_path)

    @staticmethod
    def vector_linspace(start, finish, steps):
        ls = []
        for s, f in zip(start[0], finish[0]):
            ls.append(torch.linspace(s, f, steps))
        res = torch.stack(ls)
        return res.t()


if __name__ == '__main__':
    import argparse
    import utils
    from data_generation_and_loading import get_data_loaders
    from model_manager import ModelManager
    import pandas as pd
    from glob import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='none',
                        help="ID of experiment")
    parser.add_argument('--output_path', type=str, default='.',
                        help="outputs path")
    opts = parser.parse_args()
    model_name = opts.id

    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_dir = os.path.join(output_directory, 'checkpoints')

    configurations = utils.get_config(
        os.path.join(output_directory, "config.yaml"))

    logging = utils.get_config("logging.yaml")

    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print("GPU not available, running on CPU")
    else:
        device = torch.device('cuda')
    
    if configurations['model']['age_disentanglement'] or configurations['model']['age_per_feature']:
        configurations['model']['latent_size'] += configurations['model']['age_latent_size']

    manager = ModelManager(
        configurations=configurations, device=device,
        precomputed_storage_path=configurations['data']['precomputed_path'])
    manager.resume(checkpoint_dir)

    train_loader, val_loader, test_loader, normalization_dict = \
        get_data_loaders(configurations, manager.template)

    tester = Tester(manager, normalization_dict, train_loader, val_loader, test_loader,
                    output_directory, configurations, logging)

    tester()
    # tester.direct_manipulation()
    # tester.fit_coma_data_different_noises()
    # tester.set_renderings_size(512)
    # tester.set_rendering_background_color()
    # tester.interpolate()
    # tester.latent_swapping(next(iter(test_loader)).x)
    # tester.per_variable_range_experiments()
    # tester.random_generation_and_rendering(n_samples=16)
    # tester.random_generation_and_save(n_samples=16)
    # print(tester.reconstruction_errors(test_loader))
    # print(tester.compute_specificity(train_loader, 100))
    # print(tester.compute_diversity_train_set())
    # print(tester.compute_diversity())
    # tester.evaluate_gen(test_loader, n_sampled_points=2048)

