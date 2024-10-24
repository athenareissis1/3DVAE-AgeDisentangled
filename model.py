"""
Code from https://github.com/theswgong/spiralnet_plus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch.autograd import Function


class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))
        x = self.layer(x)
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)


def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralEnblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, down_transform):
        out = F.elu(self.conv(x))
        out = Pool(out, down_transform)
        return out


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralDeblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out))
        return out
    
# create gradient reverse layer function

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, latent_size, age_disentanglement, swap_features, batch_diagonal_idx, old_experiment,
                 spiral_indices, down_transform, up_transform, mlp_dropout, mlp_layer_2, mlp_layer_3, model_version, extra_layers, detach_features, is_vae=False):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.age_disentanglement = age_disentanglement
        self.swap_features = swap_features
        self.batch_diagonal_idx = batch_diagonal_idx
        self.old_experiment = old_experiment
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.mlp_dropout = mlp_dropout
        self.mlp_layer_2 = mlp_layer_2
        self.mlp_layer_3 = mlp_layer_3
        self.model_version = model_version
        self.extra_layers = extra_layers
        self.detach_features = detach_features
        self.num_vert = self.down_transform[-1].size(0)
        self.is_vae = is_vae

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx]))
            else:
                self.en_layers.append(
                    SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx]))
                
        if self.age_disentanglement and self.old_experiment==False and self.model_version!=2.3:
            self.en_layers.append(
                nn.Linear(self.num_vert * out_channels[-1], latent_size-1))
        else:
            self.en_layers.append(
                nn.Linear(self.num_vert * out_channels[-1], latent_size))


        if self.is_vae:  # add another linear layer for logvar
            self.en_layers.append(
                nn.Linear(self.num_vert * out_channels[-1], latent_size))
            
            
        # ------------- # 
            

        # MLP_features
        self.mlp_feature_layers = nn.Sequential(nn.Linear(self.latent_size - 1, self.latent_size - 1), 
                                       nn.ReLU(), 
                                       nn.BatchNorm1d(self.latent_size - 1), 
                                       nn.Dropout(self.mlp_dropout), 

                                       nn.Linear(self.latent_size - 1, self.latent_size - 1), 
                                       nn.ReLU(), 
                                       nn.BatchNorm1d(self.latent_size - 1), 
                                       nn.Dropout(self.mlp_dropout), 

                                       nn.Linear(self.latent_size - 1, self.latent_size - 1)
                                       ) 

        # ------------- # 

        # # MLP_features
        # self.mlp_feature_layers = nn.Sequential(nn.Linear(self.latent_size - 1, self.latent_size - 1),
        #                                         nn.ReLU(), 
        #                                         nn.Linear(self.latent_size - 1, self.latent_size - 1), 
        #                                         nn.ReLU(), 
        #                                         nn.Linear(self.latent_size - 1, self.latent_size - 1)
        #                                         ) 
        # # self.reset_parameters_mlp(self.mlp_feature_layers)
    
        # MLP_loss
        self.mlp_loss_layers = nn.Sequential(GradientReversalLayer(alpha=1), 
                                       
                                       nn.Linear(self.latent_size-1, self.mlp_layer_2), 
                                       nn.ReLU(), 
                                       nn.BatchNorm1d(self.mlp_layer_2), 
                                       nn.Dropout(self.mlp_dropout), 
                                       
                                       nn.Linear(self.mlp_layer_2, self.mlp_layer_3), 
                                       nn.ReLU(), 
                                       nn.BatchNorm1d(self.mlp_layer_3), 
                                       nn.Dropout(self.mlp_dropout), 

                                       nn.Linear(self.mlp_layer_3, 1)
                                       ) 

        # ------------- # 

        # # MLP_loss
        # self.regressor = nn.Sequential(GradientReversalLayer(alpha=1), 
                                       
        #                                nn.Linear(self.latent_size-1, self.mlp_layer_2), 
        #                                nn.SiLU(), 
        #                                nn.BatchNorm1d(self.mlp_layer_2), 
        #                                nn.Dropout(self.mlp_dropout), 
                                       
        #                                nn.Linear(self.mlp_layer_2, self.mlp_layer_3), 
        #                                nn.SiLU(), 
        #                                nn.BatchNorm1d(self.mlp_layer_3), 
        #                                nn.Dropout(self.mlp_dropout), 

        #                                nn.Linear(self.mlp_layer_3, 1)
        #                                ) 
        # self.reset_parameters_mlp(self.regressor)

        # self.feature_layer1 = nn.Linear(self.latent_size - 1, self.latent_size - 1)
        # self.feature_layer2 = nn.Linear(self.latent_size - 1, self.latent_size - 1)

        # ---------- # 

        # decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_size, self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx - 1],
                                  out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
            else:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx], out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
        self.de_layers.append(
            SpiralConv(out_channels[0], in_channels, self.spiral_indices[0]))
        
        
        self.reset_parameters_mlp(self.mlp_feature_layers)
        self.reset_parameters_mlp(self.mlp_loss_layers)
        self.reset_parameters()


    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'mlp' in name:
                continue 
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def reset_parameters_mlp(self, model):
        for name, layer in model.named_children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)


    def encode(self, x):
        n_linear_layers = 2 if self.is_vae else 1
        for i, layer in enumerate(self.en_layers):
            if i < len(self.en_layers) - n_linear_layers:
                x = layer(x, self.down_transform[i])

        x = x.view(-1, self.en_layers[-1].weight.size(1))
        mu = self.en_layers[-1](x)

        if self.is_vae:
            logvar = self.en_layers[-2](x)
        else:
            mu = torch.sigmoid(mu)
            logvar = None
        return mu, logvar

    def mlp_feature(self, features):

        features_no_age = self.mlp_feature_layers(features)

        return features_no_age

    def mlp_loss(self, features):

        if self.swap_features:
            features = features[self.batch_diagonal_idx, ::]

        pred_age = self.mlp_loss_layers(features)

        return pred_age 
    
    def decode(self, x):
        num_layers = len(self.de_layers)
        num_features = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.up_transform[num_features - i])
            else:
                x = layer(x)
        return x
    
    
    def forward(self, x):

        # MODEL 1 - AGE & MLP losses
        if self.model_version == 1 and self.extra_layers == False and self.detach_features == False:

            mu, logvar = self.encode(x)
            if self.is_vae and self.training:
                z = self._reparameterize(mu, logvar, self.age_disentanglement, self.old_experiment, self.model_version)
            else:
                z = mu
            z_features = z[:, :-1]

            if self.age_disentanglement:
                mlp_output = self.mlp_loss(z_features)
            else:
                mlp_output = 0

            out = self.decode(z)

        # MODEL 2 - AGE & MLP losses
        elif self.model_version == 2 and self.extra_layers == False and self.detach_features == False:

            mu, logvar = self.encode(x)
            mu_features = mu[:, :-1]

            if self.age_disentanglement:
                mlp_output = self.mlp_loss(mu_features)
            else:
                mlp_output = 0

            if self.is_vae and self.training:
                z = self._reparameterize(mu, logvar, self.age_disentanglement, self.old_experiment, self.model_version)
            else:
                z = mu  

            out = self.decode(z)
        
        # MODEL 1.1 - Add linear layers 
        elif self.model_version == 1 and self.extra_layers == True and self.detach_features == False:

            mu, logvar = self.encode(x)
            if self.is_vae and self.training:
                z1 = self._reparameterize(mu, logvar, self.age_disentanglement, self.old_experiment, self.model_version)
            else:
                z1 = mu
            z1_features = z1[:, :-1]
            z1_age = z1[:, -1]

            z2_features = self.mlp_feature(z1_features)

            if self.age_disentanglement:
                mlp_output = self.mlp_loss(z2_features)
            else:
                mlp_output = 0

            z2 = torch.cat((z2_features, z1_age.unsqueeze(1)), dim=1)
            out = self.decode(z2)
            z = z2

        # MODEL 2.1 - Add linear layers 
        elif (self.model_version == 2 or self.model_version == 2.3)  and self.extra_layers == True and self.detach_features == False:
            mu1, logvar = self.encode(x)

            mu1_features = mu1[:, :-1]
            mu1_age = mu1[:, -1]

            mu2_features = self.mlp_feature(mu1_features)

            if self.age_disentanglement:
                mlp_output = self.mlp_loss(mu2_features)
            else:
                mlp_output = 0

            mu2 = torch.cat((mu2_features, mu1_age.unsqueeze(1)), dim=1)

            if self.is_vae and self.training:
                z = self._reparameterize(mu2, logvar, self.age_disentanglement, self.old_experiment, self.model_version)
            else:
                z = mu2 

            out = self.decode(z)
            mu = mu2

        # MODEL 2.2 - .clone().detach() features 
        elif self.model_version == 2 and self.extra_layers == True and self.detach_features == True:
            mu1, logvar = self.encode(x)

            mu1_features = mu1[:, :-1]
            mu1_features_detached = mu1[:, :-1].clone().detach()
            mu1_age = mu1[:, -1]

            mu2_features = self.mlp_feature(mu1_features)
            mu1_features_detached = self.mlp_feature(mu1_features_detached)

            if self.age_disentanglement:
                mlp_output = self.mlp_loss(mu1_features_detached)
            else:
                mlp_output = 0

            mu2 = torch.cat((mu2_features, mu1_age.unsqueeze(1)), dim=1)

            if self.is_vae and self.training:
                z = self._reparameterize(mu2, logvar, self.age_disentanglement, self.old_experiment, self.model_version)
            else:
                z = mu2

            out = self.decode(z)
            mu = mu2

        return out, z, mu, logvar, mlp_output

    @staticmethod
    def _reparameterize(mu, logvar, age_disentanglement, old_experiment, model_version):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if age_disentanglement and old_experiment==False and model_version!=2.3:
            mu_feat = mu[:,:-1]
            mu_age = mu[:, -1].view(-1, 1)
            z = mu_feat + eps * std
            z = torch.cat((z, mu_age), dim=1)
        else:
            z = mu + eps * std
        return z


class FactorVAEDiscriminator(nn.Module):
    def __init__(self, latent_dim=10):
        """
        Model Architecture
        ------------
        - 6 layer multi-layer perceptron, each with 1000 hidden units
        - Leaky ReLu activations
        - Output 2 logits
        """
        super(FactorVAEDiscriminator, self).__init__()

        # Activation parameters
        self.neg_slope = 0.2
        self.leaky_relu = nn.LeakyReLU(self.neg_slope, True)

        # Layer parameters
        self.z_dim = latent_dim
        self.hidden_units = 1000
        # theoretically 1 with sigmoid but bad results => use 2 and softmax
        out_units = 2

        # Fully connected layers
        self.lin1 = nn.Linear(self.z_dim, self.hidden_units)
        self.lin2 = nn.Linear(self.hidden_units, self.hidden_units)
        self.lin3 = nn.Linear(self.hidden_units, self.hidden_units)
        self.lin4 = nn.Linear(self.hidden_units, self.hidden_units)
        self.lin5 = nn.Linear(self.hidden_units, self.hidden_units)
        self.lin6 = nn.Linear(self.hidden_units, out_units)

        self.reset_parameters()

    def forward(self, z):
        z = self.leaky_relu(self.lin1(z))
        z = self.leaky_relu(self.lin2(z))
        z = self.leaky_relu(self.lin3(z))
        z = self.leaky_relu(self.lin4(z))
        z = self.leaky_relu(self.lin5(z))
        z = self.lin6(z)
        return z

    def reset_parameters(self):
        self.apply(self.weights_init)

    @staticmethod
    def weights_init(layer):
        if isinstance(layer, nn.Linear):
            x = layer.weight
            return nn.init.kaiming_uniform_(x, a=0.2, nonlinearity='leaky_relu')
