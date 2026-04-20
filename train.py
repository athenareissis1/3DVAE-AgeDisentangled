import os
import argparse
import shutil
import tqdm
import torch.nn
import wandb

import utils
from data_generation_and_loading import FaceGenerator, BodyGenerator
from data_generation_and_loading import get_data_loaders
from model_manager import ModelManager
from test import Tester

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configurations/default.yaml',
                    help="Path to the configuration file.")
parser.add_argument('--id', type=str, default='none', help="ID of experiment")
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--generate_data', action='store_true')
parser.add_argument('--resume', action='store_true')
opts = parser.parse_args()
config = utils.get_config(opts.config)

##### INITALISE LOGGING #####

# logging_config = utils.get_config("logging.yaml")

# Set these BEFORE wandb.init()
os.environ["WANDB_DIR"] = config['wandb']['dir']
os.environ["WANDB_DISABLE_CODE"] = "true"
os.environ["WANDB_CONSOLE"] = "off"

wandb_run = wandb.init(
    entity=config['wandb']['entity'],
    project=config['wandb']['project'],
    name=opts.id,
    id=opts.id,
    dir=os.environ["WANDB_DIR"],
    save_code=False,
    resume="allow",
    settings=wandb.Settings(_disable_stats=True),
    config=config
)

# assert logging_config is not None
assert opts.id is not None
assert wandb_run is not None

#############################

if opts.id != 'none':
    model_name = opts.id
else:
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_dir = utils.prepare_sub_folder(output_directory)

shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

if not torch.cuda.is_available():
    device = torch.device('cpu')
    print("GPU not available, running on CPU")
else:
    device = torch.device('cuda')

if opts.generate_data:
    if 'faces' in str(config['data']['dataset_type']):
        data_generator = FaceGenerator(config['data']['pca_path'],
                                       config['data']['dataset_path'])
    else:
        data_generator = BodyGenerator(config['data']['dataset_path'])
    data_generator(config['data']['number_of_meshes'],
                   config['data']['std_pca_latent'], opts.generate_data)

if config['model']['age_disentanglement'] or config['model']['age_per_feature']:

    no_remainder = config['model']['latent_size'] % config['model']['age_latent_size'] == 0
    assert no_remainder
        
    config['model']['latent_size'] += config['model']['age_latent_size']

if config['model']['disease_classification']:
    config['model']['latent_size'] += 1

loss_keys = ['reconstruction', 'kl', 'dip', 'factor',
        'latent_consistency', 'laplacian', 'age', 'age_remove_mlp', 
        'age_reconstruction_mlp', 'contrastive', 'mi', 'latent_similarity', 
        'adversarial', 'adversarial_latent', 'discriminator', 'discriminator_latent', 
        'discriminator_latent_real', 'discriminator_latent_fake', 'edit_id_consistency', 
        'edit_age_consistency', 'cycle_recon_consistency', 'disease_classification', 'tot']

manager = ModelManager(
    configurations=config, device=device,
    precomputed_storage_path=config['data']['precomputed_path'], loss_keys=loss_keys)

train_loader, validation_loader, test_loader, normalization_dict = \
    get_data_loaders(config, manager.template)

train_visualization_batch = next(iter(train_loader))
validation_visualization_batch = next(iter(validation_loader))

# manager.render_and_show_batch(train_visualization_batch, normalization_dict)

if opts.resume:
    start_epoch = manager.resume(checkpoint_dir)
else:
    start_epoch = 0

# manager.log_hyperparameters(wandb_run, config, logging_config)

for epoch in tqdm.tqdm(range(start_epoch, config['optimization']['epochs'])):
    manager.run_epoch(train_loader, device, train=True)
    manager.log_losses(wandb_run, epoch, 'train')

    manager.run_epoch(validation_loader, device, train=False)
    manager.log_losses(wandb_run, epoch, 'validation')

    if (epoch + 1) % config['logging_frequency']['tb_renderings'] == 0:
        manager.log_images(train_visualization_batch, wandb_run, epoch,
                           normalization_dict, 'train', error_max_scale=2)
        manager.log_images(validation_visualization_batch, wandb_run, epoch,
                           normalization_dict, 'validation', error_max_scale=2)
    if (epoch + 1) % config['logging_frequency']['save_weights'] == 0:
        manager.save_weights(checkpoint_dir, epoch)

wandb_run.finish()

Tester(manager, normalization_dict, train_loader, validation_loader, test_loader,
       output_directory, config, wandb_run)()
