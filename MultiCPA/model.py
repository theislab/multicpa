# Author: Kemal Inecik
# Email: k.inecik@gmail.com

import json
from abc import abstractmethod
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions.utils import (
    logits_to_probs,
    probs_to_logits,
    lazy_property,
    broadcast_all
)


class NB(torch.nn.Module):

    def __init__(self):
        super(NB, self).__init__()

    def forward(self, yhat, y, eps=1e-8, reduction="mean"):
        """Negative binomial log-likelihood loss. It assumes targets `y` with n
        rows and d columns, but estimates `yhat` with n rows and 2d columns.
        The columns 0:d of `yhat` contain estimated means, the columns d:2*d of
        `yhat` contain estimated variances. This module assumes that the
        estimated mean and inverse dispersion are positive---for numerical
        stability, it is recommended that the minimum estimated variance is
        greater than a small number (1e-3).
        Parameters
        ----------
        yhat: Tensor
                Torch Tensor of reconstructed data.
        y: Tensor
                Torch Tensor of ground truth data.
        eps: Float
                numerical stability constant.
        """
        dim = yhat.size(1) // 2
        # means of the negative binomial (has to be positive support)
        mu = yhat[:, :dim]
        # inverse dispersion parameter (has to be positive support)
        theta = yhat[:, dim:]

        if theta.ndimension() == 1:
            # In this case, we reshape theta for broadcasting
            theta = theta.view(1, theta.size(0))
        t1 = torch.lgamma(theta + eps) + torch.lgamma(y + 1.0) - \
             torch.lgamma(y + theta + eps)
        t2 = (theta + y) * torch.log(1.0 + (mu / (theta + eps))) + \
             (y * (torch.log(theta + eps) - torch.log(mu + eps)))
        final = t1 + t2
        final = _nan2inf(final)
        return _reduction(final, reduction=reduction)


class NBMixture(torch.nn.Module):
    """
    Negative binomial mixture distribution.

    From https://github.com/YosefLab/scvi-tools/blob/12402cd8ec70aa135eee66af5a7c67224474167d/scvi/distributions/_negative_binomial.py
    """

    def __init__(self):
        super(NBMixture, self).__init__()

    def mixture_probs(self, mixture_logits):
        return logits_to_probs(mixture_logits, is_binary=True)

    def _gamma(self, theta, mu):
        concentration = theta
        rate = theta / mu
        # Important remark: Gamma is parametrized by the rate = 1/scale!
        gamma_d = torch.distributions.Gamma(concentration=concentration, rate=rate)
        return gamma_d

    def sample(self, mu_1, mu_2, theta_1, theta_2, pi_logits, sample_shape=None):
        with torch.no_grad():
            pi = self.mixture_probs(pi_logits)
            mixing_sample = torch.distributions.Bernoulli(pi).sample()
            mu = mu_1 * mixing_sample + mu_2 * (1 - mixing_sample)
            if theta_2 is None:
                theta = theta_1
            else:
                theta = self.theta1 * mixing_sample + self.theta2 * (1 - mixing_sample)
            gamma_d = self._gamma(theta, mu)
            if sample_shape is None:
                sample_shape = mu_1.size
            p_means = gamma_d.sample(sample_shape)

            # Clamping as distributions objects can have buggy behaviors when
            # their parameters are too high
            l_train = torch.clamp(p_means, max=1e8)
            counts = torch.distributions.Poisson(
                l_train
            ).sample()  # Shape : (n_samples, n_cells_batch, n_features)
            return counts

    def forward(self, y, mu_1, mu_2, theta_1, theta_2, pi_logits, eps=1e-8, reduction="mean"):
        """
        Negative log likelihood (scalar) of a minibatch according to a mixture nb model.

        pi_logits is the probability (logits) to be in the first component.
        For totalVI, the first component should be background.

        Parameters
        ----------
        y
            Observed data
        mu_1
            Mean of the first negative binomial component (has to be positive support) (shape: minibatch x features)
        mu_2
            Mean of the second negative binomial (has to be positive support) (shape: minibatch x features)
        theta_1
            First inverse dispersion parameter (has to be positive support) (shape: minibatch x features)
        theta_2
            Second inverse dispersion parameter (has to be positive support) (shape: minibatch x features)
            If None, assume one shared inverse dispersion parameter.
        pi_logits
            Probability of belonging to mixture component 1 (logits scale)
        eps
            Numerical stability constant
        """

        if theta_2 is not None:
            log_nb_1 = -NB(y, mu_1, theta_1, reduction="none")
            log_nb_2 = -NB(y, mu_2, theta_2, reduction="none")
        # this is intended to reduce repeated computations
        else:
            theta = theta_1
            if theta.ndimension() == 1:
                theta = theta.view(
                    1, theta.size(0)
                )  # In this case, we reshape theta for broadcasting

            log_theta_mu_1_eps = torch.log(theta + mu_1 + eps)
            log_theta_mu_2_eps = torch.log(theta + mu_2 + eps)
            lgamma_x_theta = torch.lgamma(y + theta)
            lgamma_theta = torch.lgamma(theta)
            lgamma_x_plus_1 = torch.lgamma(y + 1)

            log_nb_1 = (
                    theta * (torch.log(theta + eps) - log_theta_mu_1_eps)
                    + y * (torch.log(mu_1 + eps) - log_theta_mu_1_eps)
                    + lgamma_x_theta
                    - lgamma_theta
                    - lgamma_x_plus_1
            )
            log_nb_2 = (
                    theta * (torch.log(theta + eps) - log_theta_mu_2_eps)
                    + y * (torch.log(mu_2 + eps) - log_theta_mu_2_eps)
                    + lgamma_x_theta
                    - lgamma_theta
                    - lgamma_x_plus_1
            )

        logsumexp = torch.logsumexp(torch.stack((log_nb_1, log_nb_2 - pi_logits)), dim=0)
        softplus_pi = F.softplus(-pi_logits)

        log_mixture_nb = -(logsumexp - softplus_pi)
        log_mixture_nb = _nan2inf(log_mixture_nb)
        return _reduction(log_mixture_nb, reduction=reduction)


def _reduction(val, reduction):
    if reduction == "mean":
        return torch.mean(val)
    elif reduction == "sum":
        return torch.sum(val, dim=-1)
    elif reduction == "none":
        return val
    else:
        raise NotImplementedError(f"The reduction method \'{reduction}\' is not implemented!")


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


def _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6):
    r"""
    NB parameterizations conversion.

    From https://github.com/YosefLab/scvi-tools/blob/12402cd8ec70aa135eee66af5a7c67224474167d/scvi/distributions/_negative_binomial.py

    Parameters
    ----------
    mu
        mean of the NB distribution.
    theta
        inverse overdispersion.
    eps
        constant used for numerical log stability. (Default value = 1e-6)

    Returns
    -------
    type
        the number of failures until the experiment is stopped
        and the success probability.
    """
    if not (mu is None) == (theta is None):
        raise ValueError(
            "If using the mu/theta NB parameterization, both parameters must be specified"
        )
    logits = (mu + eps).log() - (theta + eps).log()
    total_count = theta
    return total_count, logits


class GaussianMixture(torch.nn.Module):
    """
    Gaussian mixture distribution.

    TODO: implement this distribution as an alternative to the Negative Binomial Mixture distribution
    """

    def __init__(self):
        super(GaussianMixture, self).__init__()
        raise NotImplementedError("Implement GaussianMixture()!")

    def sample(self, mu_1, mu_2, theta_1, theta_2, pi_logits, sample_shape=None):
        ## https://github.com/ldeecke/gmm-torch/blob/master/gmm.py
        pass  # TODO

    def forward(self, y, mu_1, mu_2, theta_1, theta_2, pi_logits, eps=1e-8, reduction="mean"):
        ## https://github.com/ldeecke/gmm-torch/blob/master/gmm.py
        pass  # TODO


class Gaussian(torch.nn.Module):
    """
    Gaussian log-likelihood loss. It assumes targets `y` with n rows and d
    columns, but estimates `yhat` with n rows and 2d columns. The columns 0:d
    of `yhat` contain estimated means, the columns d:2*d of `yhat` contain
    estimated variances. This module assumes that the estimated variances are
    positive---for numerical stability, it is recommended that the minimum
    estimated variance is greater than a small number (1e-3).
    """

    def __init__(self):
        super(Gaussian, self).__init__()

    def forward(self, yhat, y, reduction="mean"):
        dim = yhat.size(1) // 2
        mean = yhat[:, :dim]
        variance = yhat[:, dim:]

        term1 = variance.log().div(2)
        term2 = (y - mean).pow(2).div(variance.mul(2))
        return _reduction(term1 + term2, reduction=reduction)


class KLDUniversal(torch.nn.Module):

    def __init__(self):
        super(KLDUniversal, self).__init__()

    def forward(self, distribution_object_1, distribution_object_2, reduction="mean"):
        kl = torch.distributions.kl_divergence(distribution_object_1, distribution_object_2)
        # TODO: Implement here!


class KLD(torch.nn.Module):
    """Compute the Kullback-Leibler divergence between a Gaussian and a standard normal Gaussian prior."""
    # 𝐷𝐾𝐿(𝑞𝜙(𝑧|𝑥(𝑖))||𝑝𝜃(𝑧)) where 𝑝𝜃(𝑧)~N(μ, σ^2) and μ=0, σ=1
    def __init__(self):
        super(KLD, self).__init__()

    def forward(self, mu, logvar, reduction='mean'):
        kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1. - logvar, dim=1)
        return _reduction(kl, reduction=reduction)


class KLDGaussians(torch.nn.Module):
    """Compute the Kullback-Leibler divergence between two Gaussian distributions."""

    def __init__(self):
        super(KLDGaussians, self).__init__()

    def forward(self, mu_1, logvar_1, mu_2, logvar_2, reduction='mean'):
        kl = 0.5 * torch.sum((logvar_2 - logvar_1 +
                              (torch.exp(logvar_1) + (mu_1 - mu_2) ** 2) / torch.exp(logvar_2) - 1.), dim=1)
        # correction based on:
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # https://github.com/SchubertLab/mvTCR/blob/master/tcr_embedding/models/losses/kld.py
        return _reduction(kl, reduction=reduction)


class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear", full_last_layer_act=False):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.full_last_layer_act = full_last_layer_act
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        elif self.activation == "sigm":
            self.sigmoid = torch.nn.Sigmoid()
        else:
            raise ValueError("last_layer_act must be one of 'linear', 'ReLU' or 'sigm'.")

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        if self.activation == "ReLU":
            x = self.network(x)
            if self.full_last_layer_act:
                dim = x.size(1)
            else:
                # apply the activation only on the estimated means
                dim = x.size(1) // 2
            return torch.cat((self.relu(x[:, :dim]), x[:, dim:]), dim=1)
        elif self.activation == "sigm":
            x = self.network(x)
            if self.full_last_layer_act:
                dim = x.size(1)
            else:
                # apply the activation only on the estimated means
                dim = x.size(1) // 2
            return torch.cat((self.sigmoid(x[:, :dim]), x[:, dim:]), dim=1)
        return self.network(x)


class GeneralizedSigmoid(torch.nn.Module):
    """
    Sigmoid, log-sigmoid or linear functions for encoding dose-response for
    drug perturbations.
    """

    def __init__(self, dim, device, nonlin='sigmoid'):
        """Sigmoid modeling of continuous variable.
        Params
        ------
        nonlin : str (default: logsigm)
            One of logsigm, sigm.
        """
        super(GeneralizedSigmoid, self).__init__()
        self.nonlin = nonlin
        self.beta = torch.nn.Parameter(
            torch.ones(1, dim, device=device),
            requires_grad=True
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(1, dim, device=device),
            requires_grad=True
        )

    def forward(self, x):
        if self.nonlin == 'logsigm':
            c0 = self.bias.sigmoid()
            return (torch.log1p(x) * self.beta + self.bias).sigmoid() - c0
        elif self.nonlin == 'sigm':
            c0 = self.bias.sigmoid()
            return (x * self.beta + self.bias).sigmoid() - c0
        else:
            return x

    def one_drug(self, x, i):
        if self.nonlin == 'logsigm':
            c0 = self.bias[0][i].sigmoid()
            return (torch.log1p(x) * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        elif self.nonlin == 'sigm':
            c0 = self.bias[0][i].sigmoid()
            return (x * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        else:
            return x


class BaseModel(torch.nn.Module):
    """Base module which implements methods shared across all main models."""

    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def set_hparams_(self):
        raise NotImplementedError

    @abstractmethod
    def move_inputs_(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

    @classmethod
    def defaults(self):
        """
        Returns the list of default hyper-parameters for ComPert
        """
        return self.set_hparams_(self, 0, "")


    def reparameterize(self, mu, log_var):
        """
        Apply the reparametrization trick.

        From  https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/

        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        z = mu + (eps * std)  # sampling as if coming from the input space
        return z

    def kl_annealing(self, epoch, max_epoch, kl_annealing_frac):
        # TODO: it does not fit. adding this makes the problem even higher
        """
        Calculate KLD annealing factor, i.e. KLD needs to get warmup
        From https://github.com/SchubertLab/mvTCR/blob/master/tcr_embedding/models/base_model.py
        # Check: https://arxiv.org/pdf/1511.06349.pdf section 3.1

        :param e: current epoch
        :param kl_annealing_epochs: total number of warmup epochs
        :return:
        """
        if kl_annealing_frac == 0 or kl_annealing_frac is None:
            alpha = 1.0
        else:
            alpha = min(1.0, epoch / (max_epoch * kl_annealing_frac))

        return alpha

    def compute_drug_embeddings_(self, drugs):
        """
        Compute sum of drug embeddings, each of them multiplied by its
        dose-response curve.
        """

        if self.doser_type == 'mlp':
            doses = []
            for d in range(drugs.size(1)):
                this_drug = drugs[:, d].view(-1, 1)
                doses.append(self.dosers[d](this_drug).sigmoid() * this_drug.gt(0))
            return torch.cat(doses, 1) @ self.drug_embeddings.weight
        else:
            return self.dosers(drugs) @ self.drug_embeddings.weight

    def early_stopping(self, score):
        """
        Decays the learning rate, and possibly early-stops training.
        """
        self.scheduler_autoencoder.step()
        self.scheduler_adversary.step()
        self.scheduler_dosers.step()

        if score > self.best_score:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1

        return self.patience_trials > self.patience

    def specific_threshold(self, score, epoch, epoch_thr, score_thr):
        if epoch > epoch_thr:
            if score < score_thr:
                raise ValueError("Not passed threshold.")
        return False

class ComPert(BaseModel):
    """
    The ComPert autoencoder extended by the possibilities to encode the data using a VAE.
    Furthermore, protein data can be encoded/ decoded by concatenating the protein data to the rna data.
    """

    def __init__(
            self,
            num_genes,
            num_drugs,
            num_cell_types,
            device="cpu",
            seed=0,
            patience=5,
            loss_ae='gauss',
            doser_type='logsigm',
            decoder_activation='linear',
            hparams="",
            is_vae=False,
            num_proteins=None, ):
        super(ComPert, self).__init__()

        # set generic attributes
        self.num_genes = num_genes
        self.num_proteins = num_proteins
        self.num_drugs = num_drugs
        self.num_cell_types = num_cell_types
        self.device = device
        self.seed = seed
        self.loss_ae = loss_ae
        self.is_vae = is_vae
        # early-stopping
        self.patience = patience
        self.best_score = -1e3
        self.patience_trials = 0

        # set hyperparameters
        self.set_hparams_(seed, hparams)

        vae_dim_factor = 1
        if is_vae:
            vae_dim_factor = 2

        if num_proteins is not None:
            self.num_inputs = num_genes + num_proteins

        # set models
        self.encoder = MLP(
            [self.num_inputs] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [self.hparams["dim"] * vae_dim_factor])

        self.decoder = MLP(
            [self.hparams["dim"]] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [self.num_inputs * 2], last_layer_act=decoder_activation)

        self.adversary_drugs = MLP(
            [self.hparams["dim"]] +
            [self.hparams["adversary_width"]] *
            self.hparams["adversary_depth"] +
            [num_drugs])

        self.adversary_cell_types = MLP(
            [self.hparams["dim"]] +
            [self.hparams["adversary_width"]] *
            self.hparams["adversary_depth"] +
            [num_cell_types])

        # set dosers
        self.doser_type = doser_type
        if doser_type == 'mlp':
            self.dosers = torch.nn.ModuleList()
            for _ in range(num_drugs):
                self.dosers.append(
                    MLP([1] +
                        [self.hparams["dosers_width"]] *
                        self.hparams["dosers_depth"] +
                        [1],
                        batch_norm=False))
        else:
            self.dosers = GeneralizedSigmoid(num_drugs, self.device,
                                             nonlin=doser_type)

        self.drug_embeddings = torch.nn.Embedding(
            num_drugs, self.hparams["dim"])
        self.cell_type_embeddings = torch.nn.Embedding(
            num_cell_types, self.hparams["dim"])

        # losses
        if self.loss_ae == 'nb':
            self.loss_autoencoder = NB()
        else:
            self.loss_autoencoder = Gaussian()
        if self.is_vae:
            self.kl_loss = KLD()
        self.loss_adversary_drugs = torch.nn.BCEWithLogitsLoss()
        self.loss_adversary_cell_types = torch.nn.CrossEntropyLoss()
        self.iteration = 0

        self.to(self.device)

        # optimizers
        self.optimizer_autoencoder = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.drug_embeddings.parameters()) +
            list(self.cell_type_embeddings.parameters()),
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"])

        self.optimizer_adversaries = torch.optim.Adam(
            list(self.adversary_drugs.parameters()) +
            list(self.adversary_cell_types.parameters()),
            lr=self.hparams["adversary_lr"],
            weight_decay=self.hparams["adversary_wd"])

        self.optimizer_dosers = torch.optim.Adam(
            self.dosers.parameters(),
            lr=self.hparams["dosers_lr"],
            weight_decay=self.hparams["dosers_wd"])

        # learning rate schedulers
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.hparams["step_size_lr"])

        self.scheduler_adversary = torch.optim.lr_scheduler.StepLR(
            self.optimizer_adversaries, step_size=self.hparams["step_size_lr"])

        self.scheduler_dosers = torch.optim.lr_scheduler.StepLR(
            self.optimizer_dosers, step_size=self.hparams["step_size_lr"])

        self.history = {'epoch': [], 'stats_epoch': []}

    def set_hparams_(self, seed, hparams):
        """
        Set hyper-parameters to (i) default values if `seed=0`, (ii) random
        values if `seed != 0`, or (iii) values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        default = (seed == 0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.hparams = {
            # "delay_adversary": -1 if default else
            # int(np.random.choice([50, 100, 200])),
            "dim": 256 if default else
            int(np.random.choice([128, 256, 512])),
            "dosers_width": 64 if default else
            int(np.random.choice([32, 64, 128])),
            "dosers_depth": 2 if default else
            int(np.random.choice([1, 2, 3])),
            "dosers_lr": 1e-3 if default else
            float(10 ** np.random.uniform(-4, -2)),
            "dosers_wd": 1e-7 if default else
            float(10 ** np.random.uniform(-8, -5)),
            "autoencoder_width": 512 if default else
            int(np.random.choice([256, 512, 1024])),
            "autoencoder_depth": 4 if default else
            int(np.random.choice([3, 4, 5])),
            "adversary_width": 128 if default else
            int(np.random.choice([64, 128, 256])),
            "adversary_depth": 3 if default else
            int(np.random.choice([2, 3, 4])),
            "reg_adversary": 5 if default else
            float(10 ** np.random.uniform(-2, 2)),
            "penalty_adversary": 3 if default else
            float(10 ** np.random.uniform(-2, 1)),
            "autoencoder_lr": 1e-3 if default else
            float(10 ** np.random.uniform(-4, -2)),
            "adversary_lr": 3e-4 if default else
            float(10 ** np.random.uniform(-5, -3)),
            "autoencoder_wd": 1e-6 if default else
            float(10 ** np.random.uniform(-8, -4)),
            "adversary_wd": 1e-4 if default else
            float(10 ** np.random.uniform(-6, -3)),
            "adversary_steps": 3 if default else
            int(np.random.choice([1, 2, 3, 4, 5])),
            "batch_size": 128 if default else
            int(np.random.choice([64, 128, 256, 512])),
            "step_size_lr": 45 if default else
            int(np.random.choice([15, 25, 45])),
            "kl_annealing_frac": 0.3 if default else
            float(np.random.uniform(0, 8) / 10),
            "recon_weight_pro": 0.1 if default else
            float(np.random.choice([0.01, 0.1, 1, 10])),
            "kl_weight": 1 if default else
            float(np.random.choice([0.1, 1, 2, 5, 10])),
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams

    def move_inputs_(self, genes, drugs, cell_types, proteins, raw_genes, raw_proteins):
        """
        Move minibatch tensors to CPU/GPU.
        """
        if genes.device.type != self.device:
            genes = genes.to(self.device)
            drugs = drugs.to(self.device)
            cell_types = cell_types.to(self.device)
            if proteins is not None:
                proteins = proteins.to(self.device)
            if raw_genes is not None:
                raw_genes = raw_genes.to(self.device)
            if raw_proteins is not None:
                raw_proteins = raw_proteins.to(self.device)
        return genes, drugs, cell_types, proteins, raw_genes, raw_proteins

    def get_latent(self, genes, drugs, cell_types, proteins=None,
                   return_latent_treated=False, use_latent_mean=False):
        """
        Return the latent representation of the inputs.
        """

        genes, drugs, cell_types, proteins, _, _ = self.move_inputs_(
            genes, drugs, cell_types, proteins, None, None)

        if self.num_proteins is not None:
            encoder_input = torch.cat([genes, proteins], dim=-1)
        else:
            encoder_input = genes

        latent_basal = self.encoder(encoder_input)

        latent_mean = None
        latent_log_var = None
        if self.is_vae:
            # convert variance estimates to a positive value in [1e-3, \infty)
            dim = latent_basal.size(1) // 2
            latent_basal[:, dim:] = \
                latent_basal[:, dim:].exp().add(1).log().add(1e-3)
            latent_mean = latent_basal[:, :dim]
            latent_log_var = latent_basal[:, dim:].log()
            latent_basal = self.reparameterize(latent_mean, latent_log_var)
            if use_latent_mean:
                latent_basal = latent_mean

        drug_emb = self.compute_drug_embeddings_(drugs)
        cell_emb = self.cell_type_embeddings(cell_types.argmax(1))

        latent_treated = latent_basal + drug_emb + cell_emb

        if return_latent_treated:
            return latent_basal, latent_treated
        return latent_basal

    def predict(self, genes, drugs, cell_types, proteins=None,
                return_latent_basal=False, use_latent_mean=False):
        """
        Predict "what would have the gene expression `genes` been, had the
        cells in `genes` with cell types `cell_types` been treated with
        combination of drugs `drugs`.
        """

        genes, drugs, cell_types, proteins, _, _ = self.move_inputs_(
            genes, drugs, cell_types, proteins, None, None)

        if self.num_proteins is not None:
            # concatenate genes and proteins
            encoder_input = torch.cat([genes, proteins], dim=-1)
        else:
            encoder_input = genes

        latent_basal = self.encoder(encoder_input)

        latent_mean = None
        latent_log_var = None
        if self.is_vae:
            # convert variance estimates to a positive value in [1e-3, \infty)
            dim = latent_basal.size(1) // 2
            latent_basal[:, dim:] = \
                latent_basal[:, dim:].exp().add(1).log().add(1e-3)
            latent_mean = latent_basal[:, :dim]
            latent_log_var = latent_basal[:, dim:].log()
            latent_basal = self.reparameterize(latent_mean, latent_log_var)
            if use_latent_mean:
                latent_basal = latent_mean

        drug_emb = self.compute_drug_embeddings_(drugs)
        cell_emb = self.cell_type_embeddings(cell_types.argmax(1))

        latent_treated = latent_basal + drug_emb + cell_emb
        reconstructions = self.decoder(latent_treated)

        # convert variance estimates to a positive value in [1e-3, \infty)
        dim = reconstructions.size(1) // 2
        reconstructions[:, dim:] = reconstructions[:, dim:].exp().add(1).log().add(1e-3)

        if self.loss_ae == 'nb':
            reconstructions[:, :dim] = reconstructions[:, :dim].exp().add(1).log().add(1e-4)
            # gene_reconstructions[:, :dim] = torch.clamp(gene_reconstructions[:, :dim], min=1e-4, max=1e4)
            # gene_reconstructions[:, dim:] = torch.clamp(gene_reconstructions[:, dim:], min=1e-6, max=1e6)

        if self.num_proteins is not None:
            # split gene and protein reconstructions
            gene_reconstructions = torch.cat([
                reconstructions[:, :dim][:, :self.num_genes],
                reconstructions[:, dim:][:, :self.num_genes]], dim=-1)
            protein_reconstructions = torch.cat([
                reconstructions[:, :dim][:, self.num_genes:],
                reconstructions[:, dim:][:, self.num_genes:]], dim=-1)
        else:
            gene_reconstructions = reconstructions
            protein_reconstructions = None

        if return_latent_basal:
            return gene_reconstructions, protein_reconstructions, \
                   latent_basal, latent_mean, latent_log_var

        return gene_reconstructions, protein_reconstructions

    def sample(self, genes, drugs, cell_types, proteins):
        """
        Sample from the decoder distributions.
        """

        with torch.no_grad():
            gene_reconstructions, protein_reconstructions, = self.predict(
                self, genes, drugs, cell_types, proteins)

        dim_genes = gene_reconstructions.size(1) // 2
        if self.loss_ae == 'nb':
            counts_genes, logits_genes = _convert_mean_disp_to_counts_logits(
                gene_reconstructions[:, dim_genes:],
                gene_reconstructions[:, :dim_genes])
            dist_genes = torch.distributions.negative_binomial.NegativeBinomial(
                total_count=counts_genes, logits=logits_genes)
        else:
            dist_genes = torch.distributions.Normal(
                gene_reconstructions[:, dim_genes:],
                gene_reconstructions[:, :dim_genes].log().div(2).exp())
        sample_genes = dist_genes.sample().cpu()

        sample_proteins = None
        if self.num_proteins is not None:
            dim_proteins = protein_reconstructions.size(1) // 2
            if self.loss_ae == 'nb':
                counts_proteins, logits_proteins = _convert_mean_disp_to_counts_logits(
                    protein_reconstructions[:, dim_proteins:],
                    protein_reconstructions[:, :dim_proteins])
                dist_proteins = torch.distributions.negative_binomial.NegativeBinomial(
                    total_count=counts_proteins, logits=logits_proteins)
            else:
                dist_proteins = torch.distributions.Normal(
                    protein_reconstructions[:, dim_proteins:],
                    protein_reconstructions[:, :dim_proteins].log().div(2).exp())
            sample_proteins = dist_proteins.sample().cpu()

        return sample_genes, sample_proteins

    def update(self, genes, drugs, cell_types, proteins,
               raw_genes, raw_proteins, epoch, max_epoch):
        """
        Update ComPert's parameters given a minibatch of genes, drugs, and
        cell types.
        """

        genes, drugs, cell_types, proteins, raw_genes, raw_proteins = self.move_inputs_(
            genes, drugs, cell_types, proteins, raw_genes, raw_proteins)

        if self.loss_ae == 'nb':
            # use raw counts as true values
            genes = raw_genes
            proteins = raw_proteins

        gene_reconstructions, protein_reconstructions, latent_basal, latent_mean, latent_log_var = \
            self.predict(genes, drugs, cell_types, proteins, return_latent_basal=True)
        # TODO: Aslında embeddingleri falan calculate etmesine gerek yok discriminator'leri eğitirken.

        reconstruction_loss_genes = self.loss_autoencoder(
            gene_reconstructions, genes, reduction="mean")

        if self.num_proteins is not None:
            reconstruction_loss_proteins = self.loss_autoencoder(
                protein_reconstructions, proteins, reduction="mean")
            reconstruction_loss = \
                reconstruction_loss_genes + \
                self.hparams["recon_weight_pro"] * \
                reconstruction_loss_proteins
        else:
            reconstruction_loss = reconstruction_loss_genes

        if self.is_vae:
            current_alpha = self.kl_annealing(epoch, max_epoch, self.hparams["kl_annealing_frac"])
            kl_loss = current_alpha * self.kl_loss(latent_mean, latent_log_var, reduction="mean")

        adversary_drugs_predictions = self.adversary_drugs(
            latent_basal)
        adversary_drugs_loss = self.loss_adversary_drugs(
            adversary_drugs_predictions, drugs.gt(0).float())

        adversary_cell_types_predictions = self.adversary_cell_types(
            latent_basal)
        adversary_cell_types_loss = self.loss_adversary_cell_types(
            adversary_cell_types_predictions, cell_types.argmax(1))

        # two place-holders for when adversary is not executed
        adversary_drugs_penalty = torch.Tensor([0])
        adversary_cell_types_penalty = torch.Tensor([0])

        if self.iteration % self.hparams["adversary_steps"]: #and self.iteration > self.hparams["delay_adversary"]:
            adversary_drugs_penalty = torch.autograd.grad(
                adversary_drugs_predictions.sum(),
                latent_basal,
                create_graph=True)[0].pow(2).mean()

            adversary_cell_types_penalty = torch.autograd.grad(
                adversary_cell_types_predictions.sum(),
                latent_basal,
                create_graph=True)[0].pow(2).mean()

            self.optimizer_adversaries.zero_grad()
            (adversary_drugs_loss +
             self.hparams["penalty_adversary"] *
             adversary_drugs_penalty +
             adversary_cell_types_loss +
             self.hparams["penalty_adversary"] *
             adversary_cell_types_penalty).backward()
            self.optimizer_adversaries.step()
        else:
            self.optimizer_autoencoder.zero_grad()
            self.optimizer_dosers.zero_grad()
            loss = (reconstruction_loss -
                    self.hparams["reg_adversary"] *
                    adversary_drugs_loss -
                    self.hparams["reg_adversary"] *
                    adversary_cell_types_loss)
            if self.is_vae:
                loss += self.hparams["kl_weight"] * kl_loss
            loss.backward()
            self.optimizer_autoencoder.step()
            self.optimizer_dosers.step()

        self.iteration += 1

        stats_dict = {
            "loss_reconstruction": reconstruction_loss.item(),
            "loss_adv_drugs": adversary_drugs_loss.item(),
            "loss_adv_cell_types": adversary_cell_types_loss.item(),
            "penalty_adv_drugs": adversary_drugs_penalty.item(),
            "penalty_adv_cell_types": adversary_cell_types_penalty.item(),
        }

        if self.num_proteins is not None:
            stats_dict["loss_reconstruction_genes"] = reconstruction_loss_genes.item()
            stats_dict["loss_reconstruction_proteins"] = reconstruction_loss_proteins.item()

        if self.is_vae:
            stats_dict["kl_weight"] = current_alpha
            stats_dict["kl_loss"] = kl_loss.item()

        return stats_dict


class TotalComPert(BaseModel):
    """
    The ComPert autoencoder combined with the concepts of the totalVI model for integrating rna and protein modalities.
    The genes and proteins are concatenated and encoded using a single encoder. Afterwards, genes and proteins
    are decoded separately. The genes are decoded using a default decoder. The proteins are decoded using five different
    decoders: background mean, foreground mean, protein mixing of background and foreground and protein dispersion.
    The loss of the proteins is computed by using a Negative Binomial Mixture distribution.

    Parts of it taken from https://github.com/YosefLab/scvi-tools/tree/12402cd8ec70aa135eee66af5a7c67224474167d
    """

    def __init__(
            self,
            num_genes,
            num_drugs,
            num_cell_types,
            device="cpu",
            seed=0,
            patience=5,
            loss_ae='gauss',
            doser_type='logsigm',
            decoder_activation='linear',
            hparams="",
            is_vae=False,
            num_proteins=None, ):
        super(TotalComPert, self).__init__()
        # set generic attributes
        self.num_genes = num_genes
        self.num_proteins = num_proteins
        self.num_drugs = num_drugs
        self.num_cell_types = num_cell_types
        self.device = device
        self.seed = seed
        self.loss_ae = loss_ae
        self.is_vae = is_vae
        # early-stopping
        self.patience = patience
        self.best_score = -1e3
        self.patience_trials = 0

        # set hyperparameters
        self.set_hparams_(seed, hparams)

        vae_dim_factor = 1
        if is_vae:
            vae_dim_factor = 2

        if num_proteins is None:
            raise ValueError("Please specify protein expressions for this model!")

        self.num_inputs = num_genes + num_proteins

        # set models
        self.encoder = MLP(
            [self.num_inputs] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [self.hparams["dim"] * vae_dim_factor])

        self.rna_decoder = MLP(
            [self.hparams["dim"]] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [num_genes * 2], last_layer_act=decoder_activation)

        self.protein_back_decoder = MLP(
            [self.hparams["dim"]] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [num_proteins * vae_dim_factor])

        self.protein_fore_scale_decoder = MLP(
            [self.hparams["dim"]] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [num_proteins],
            last_layer_act='ReLU', full_last_layer_act=True)

        self.protein_mixing_decoder = MLP(
            [self.hparams["dim"]] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [num_proteins])

        self.protein_dispersion_decoder = MLP(
            [self.hparams["dim"]] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [num_proteins])

        self.adversary_drugs = MLP(
            [self.hparams["dim"]] +
            [self.hparams["adversary_width"]] *
            self.hparams["adversary_depth"] +
            [num_drugs])

        self.adversary_cell_types = MLP(
            [self.hparams["dim"]] +
            [self.hparams["adversary_width"]] *
            self.hparams["adversary_depth"] +
            [num_cell_types])

        self.background_pro_alpha = torch.nn.Parameter(
            torch.randn(num_proteins))

        self.background_pro_log_beta = torch.nn.Parameter(
            torch.clamp(torch.randn(num_proteins), -10, 1))

        # set dosers
        self.doser_type = doser_type
        if doser_type == 'mlp':
            self.dosers = torch.nn.ModuleList()
            for _ in range(num_drugs):
                self.dosers.append(
                    MLP([1] +
                        [self.hparams["dosers_width"]] *
                        self.hparams["dosers_depth"] +
                        [1],
                        batch_norm=False))
        else:
            self.dosers = GeneralizedSigmoid(num_drugs, self.device,
                                             nonlin=doser_type)

        self.drug_embeddings = torch.nn.Embedding(
            num_drugs, self.hparams["dim"])
        self.cell_type_embeddings = torch.nn.Embedding(
            num_cell_types, self.hparams["dim"])

        # losses
        if self.loss_ae == 'nb':
            self.loss_autoencoder_genes = NB()
            self.loss_autoencoder_proteins = NBMixture()
        else:
            self.loss_autoencoder_genes = Gaussian()
            self.loss_autoencoder_proteins = GaussianMixture()
        if self.is_vae:
            self.kl_loss_latent = KLD()
            self.kl_loss_proteins_back = KLDGaussians()
        self.loss_adversary_drugs = torch.nn.BCEWithLogitsLoss()
        self.loss_adversary_cell_types = torch.nn.CrossEntropyLoss()
        self.iteration = 0

        self.to(self.device)

        # optimizers
        self.optimizer_autoencoder = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.rna_decoder.parameters()) +
            list(self.protein_back_decoder.parameters()) +
            list(self.protein_fore_scale_decoder.parameters()) +
            list(self.protein_mixing_decoder.parameters()) +
            list(self.protein_dispersion_decoder.parameters()) +
            list(self.drug_embeddings.parameters()) +
            list(self.cell_type_embeddings.parameters()) +
            [self.background_pro_alpha] +
            [self.background_pro_log_beta],
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"])

        self.optimizer_adversaries = torch.optim.Adam(
            list(self.adversary_drugs.parameters()) +
            list(self.adversary_cell_types.parameters()),
            lr=self.hparams["adversary_lr"],
            weight_decay=self.hparams["adversary_wd"])

        self.optimizer_dosers = torch.optim.Adam(
            self.dosers.parameters(),
            lr=self.hparams["dosers_lr"],
            weight_decay=self.hparams["dosers_wd"])

        # learning rate schedulers
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.hparams["step_size_lr"])

        self.scheduler_adversary = torch.optim.lr_scheduler.StepLR(
            self.optimizer_adversaries, step_size=self.hparams["step_size_lr"])

        self.scheduler_dosers = torch.optim.lr_scheduler.StepLR(
            self.optimizer_dosers, step_size=self.hparams["step_size_lr"])

        self.history = {'epoch': [], 'stats_epoch': []}

    def set_hparams_(self, seed, hparams):
        """
        Set hyper-parameters to (i) default values if `seed=0`, (ii) random
        values if `seed != 0`, or (iii) values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        default = (seed == 0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.hparams = {
            #"delay_adversary": -1 if default else
            #int(np.random.choice([50, 100, 200])),
            "dim": 256 if default else
            int(np.random.choice([128, 256, 512])),
            "dosers_width": 64 if default else
            int(np.random.choice([32, 64, 128])),
            "dosers_depth": 2 if default else
            int(np.random.choice([1, 2, 3])),
            "dosers_lr": 1e-3 if default else
            float(10 ** np.random.uniform(-4, -2)),
            "dosers_wd": 1e-7 if default else
            float(10 ** np.random.uniform(-8, -5)),
            "autoencoder_width": 512 if default else
            int(np.random.choice([256, 512, 1024])),
            "autoencoder_depth": 4 if default else
            int(np.random.choice([3, 4, 5])),
            "adversary_width": 128 if default else
            int(np.random.choice([64, 128, 256])),
            "adversary_depth": 3 if default else
            int(np.random.choice([2, 3, 4])),
            "reg_adversary": 5 if default else
            float(10 ** np.random.uniform(-2, 2)),
            "penalty_adversary": 3 if default else
            float(10 ** np.random.uniform(-2, 1)),
            "autoencoder_lr": 1e-3 if default else
            float(10 ** np.random.uniform(-4, -2)),
            "adversary_lr": 3e-4 if default else
            float(10 ** np.random.uniform(-5, -3)),
            "autoencoder_wd": 1e-6 if default else
            float(10 ** np.random.uniform(-8, -4)),
            "adversary_wd": 1e-4 if default else
            float(10 ** np.random.uniform(-6, -3)),
            "adversary_steps": 3 if default else
            int(np.random.choice([1, 2, 3, 4, 5])),
            "batch_size": 128 if default else
            int(np.random.choice([64, 128, 256, 512])),
            "step_size_lr": 45 if default else
            int(np.random.choice([15, 25, 45])),
            "kl_annealing_frac": 0.3 if default else
            float(np.random.uniform(0, 8) / 10),
            "recon_weight_pro": 0.1 if default else
            float(np.random.choice([0.01, 0.1, 1, 10])),
            "kl_weight": 1 if default else
            float(np.random.choice([0.1, 1, 2, 5, 10])),
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams

    def move_inputs_(self, genes, drugs, cell_types, proteins, raw_genes, raw_proteins):
        """
        Move minibatch tensors to CPU/GPU.
        """
        if genes.device.type != self.device:
            genes = genes.to(self.device)
            drugs = drugs.to(self.device)
            cell_types = cell_types.to(self.device)
            if proteins is not None:
                proteins = proteins.to(self.device)
            if raw_genes is not None:
                raw_genes = raw_genes.to(self.device)
            if raw_proteins is not None:
                raw_proteins = raw_proteins.to(self.device)
        return genes, drugs, cell_types, proteins, raw_genes, raw_proteins

    def get_latent(self, genes, drugs, cell_types, proteins=None,
                   return_latent_treated=False, use_latent_mean=False):
        """
        Return the latent representation of the inputs.
        """

        genes, drugs, cell_types, proteins, _, _ = self.move_inputs_(
            genes, drugs, cell_types, proteins, None, None)

        encoder_input = torch.cat([genes, proteins], dim=-1)
        latent_basal = self.encoder(encoder_input)

        if self.is_vae:
            # convert variance estimates to a positive value in [1e-3, \infty)
            dim = latent_basal.size(1) // 2
            latent_basal[:, dim:] = \
                latent_basal[:, dim:].exp().add(1).log().add(1e-3)
            latent_mean = latent_basal[:, :dim]
            latent_log_var = latent_basal[:, dim:].log()
            latent_basal = self.reparameterize(latent_mean, latent_log_var)
            if use_latent_mean:
                latent_basal = latent_mean

        drug_emb = self.compute_drug_embeddings_(drugs)
        cell_emb = self.cell_type_embeddings(cell_types.argmax(1))

        latent_treated = latent_basal + drug_emb + cell_emb

        if return_latent_treated:
            return latent_basal, latent_treated
        return latent_basal

    def predict(self, genes, drugs, cell_types, proteins=None,
                return_latent_basal=False, use_latent_mean=False):
        """
        Predict "what would have the gene expression `genes` been, had the
        cells in `genes` with cell types `cell_types` been treated with
        combination of drugs `drugs`.
        """

        genes, drugs, cell_types, proteins, _, _ = self.move_inputs_(
            genes, drugs, cell_types, proteins, None, None)

        # concatenate genes and proteins
        encoder_input = torch.cat([genes, proteins], dim=-1)
        latent_basal = self.encoder(encoder_input)

        latent_mean = None
        latent_log_var = None
        if self.is_vae:
            # convert variance estimates to a positive value in [1e-3, \infty)
            dim = latent_basal.size(1) // 2
            latent_basal[:, dim:] = \
                latent_basal[:, dim:].exp().add(1).log().add(1e-3)
            latent_mean = latent_basal[:, :dim]
            latent_log_var = latent_basal[:, dim:].log()
            latent_basal = self.reparameterize(latent_mean, latent_log_var)
            if use_latent_mean:
                latent_basal = latent_mean

        drug_emb = self.compute_drug_embeddings_(drugs)
        cell_emb = self.cell_type_embeddings(cell_types.argmax(1))

        latent_treated = latent_basal + drug_emb + cell_emb

        gene_reconstructions = self.rna_decoder(latent_treated)

        # convert variance estimates to a positive value in [1e-3, \infty)
        dim = gene_reconstructions.size(1) // 2
        gene_reconstructions[:, dim:] = \
            gene_reconstructions[:, dim:].exp().add(1).log().add(1e-3)

        if self.loss_ae == 'nb':
            gene_reconstructions[:, :dim] = \
                gene_reconstructions[:, :dim].exp().add(1).log().add(1e-4)
            # gene_reconstructions[:, :dim] = torch.clamp(gene_reconstructions[:, :dim], min=1e-4, max=1e4)
            # gene_reconstructions[:, dim:] = torch.clamp(gene_reconstructions[:, dim:], min=1e-6, max=1e6)

        protein_back = self.protein_back_decoder(latent_treated)
        protein_fore_scale = self.protein_fore_scale_decoder(latent_treated) + 1 + 1e-8
        protein_mixing = self.protein_mixing_decoder(latent_treated)

        if self.is_vae:
            # convert variance estimates to a positive value
            dim = protein_back.size(1) // 2
            alpha = protein_back[:, :dim]
            beta = protein_back[:, dim:].exp()
            log_protein_back = self.reparameterize(alpha, 2 * beta.log())
            protein_back = log_protein_back.exp()
        else:
            protein_back = protein_back.exp().add(1).log().add(1e-4)

        protein_fore = protein_fore_scale * protein_back
        protein_mixing_transformed = 1 / (1 + torch.exp(-protein_mixing))
        protein_reconstructions = (1 - protein_mixing_transformed) * protein_fore + \
                                  protein_mixing_transformed * protein_back

        protein_dispersion = self.protein_dispersion_decoder(latent_treated)

        # convert variance estimates to a positive value in [1e-3, \infty)
        protein_dispersion = \
            protein_dispersion.exp().add(1).log().add(1e-3)

        protein_reconstructions = torch.cat(
            [protein_reconstructions, protein_dispersion], dim=-1)

        if return_latent_basal:
            return gene_reconstructions, protein_reconstructions, \
                   latent_basal, latent_mean, latent_log_var, \
                   protein_back, protein_fore, protein_mixing, alpha, beta

        return gene_reconstructions, protein_reconstructions

    def sample(self, genes, drugs, cell_types, proteins):
        """
        Sample from the decoder distributions.
        """

        with torch.no_grad():
            gene_reconstructions, protein_reconstructions, _, _, _, \
            protein_back, protein_fore, protein_mixing, _, _ = self.predict(
                self, genes, drugs, cell_types, proteins, return_latent_basal=True)

        dim_genes = gene_reconstructions.size(1) // 2
        if self.loss_ae == 'nb':
            counts_genes, logits_genes = _convert_mean_disp_to_counts_logits(
                gene_reconstructions[:, dim_genes:],
                gene_reconstructions[:, :dim_genes])
            dist_genes = torch.distributions.negative_binomial.NegativeBinomial(
                total_count=counts_genes, logits=logits_genes)
        else:
            dist_genes = torch.distributions.Normal(
                gene_reconstructions[:, dim_genes:],
                gene_reconstructions[:, :dim_genes].log().div(2).exp())
        sample_genes = dist_genes.sample().cpu()

        sample_proteins = None
        if self.num_proteins is not None:
            dim_proteins = protein_reconstructions.size(1) // 2
            if self.loss_ae == 'nb':
                protein_dispersion = protein_reconstructions[:, dim_proteins:]
                sample_proteins = NBMixture().sample(
                    protein_back, protein_fore, protein_dispersion, None, protein_mixing)
            else:
                protein_dispersion = protein_reconstructions[:, dim_proteins:]
                sample_proteins = GaussianMixture().sample(
                    protein_back, protein_fore, protein_dispersion, None, protein_mixing)

        return sample_genes, sample_proteins

    def update(self, genes, drugs, cell_types, proteins,
               raw_genes, raw_proteins, epoch, max_epoch):
        """
        Update ComPert's parameters given a minibatch of genes, drugs, and
        cell types.
        """

        genes, drugs, cell_types, proteins, raw_genes, raw_proteins = self.move_inputs_(
            genes, drugs, cell_types, proteins, raw_genes, raw_proteins)

        if self.loss_ae == 'nb':
            # use raw counts as true values
            genes = raw_genes
            proteins = raw_proteins

        gene_reconstructions, protein_reconstructions, \
        latent_basal, latent_mean, latent_log_var, \
        protein_back, protein_fore, protein_mixing, alpha, beta = \
            self.predict(genes, drugs, cell_types, proteins, return_latent_basal=True)

        reconstruction_loss_genes = self.loss_autoencoder_genes(
            gene_reconstructions, genes, reduction="mean")

        if self.num_proteins is not None:
            dim = protein_reconstructions.size(1) // 2
            protein_dispersion = protein_reconstructions[:, dim:]
            reconstruction_loss_proteins = self.loss_autoencoder_proteins(
                proteins, protein_back, protein_fore, protein_dispersion, None, protein_mixing, reduction="mean")
            reconstruction_loss = \
                reconstruction_loss_genes + \
                self.hparams["recon_weight_pro"] * \
                reconstruction_loss_proteins
        else:
            reconstruction_loss = reconstruction_loss_genes

        if self.is_vae:
            current_alpha = self.kl_annealing(epoch, max_epoch, self.hparams["kl_annealing_frac"])
            py_back_alpha_prior = self.background_pro_alpha
            py_back_beta_prior = torch.log(torch.exp(self.background_pro_log_beta) ** 2)

            kl_loss_latent = current_alpha * self.kl_loss_latent(
                latent_mean, latent_log_var, reduction="mean")
            kl_loss_proteins_back = current_alpha * self.kl_loss_proteins_back(
                alpha, beta, py_back_alpha_prior, py_back_beta_prior, reduction="mean")

        adversary_drugs_predictions = self.adversary_drugs(
            latent_basal)
        adversary_drugs_loss = self.loss_adversary_drugs(
            adversary_drugs_predictions, drugs.gt(0).float())

        adversary_cell_types_predictions = self.adversary_cell_types(
            latent_basal)
        adversary_cell_types_loss = self.loss_adversary_cell_types(
            adversary_cell_types_predictions, cell_types.argmax(1))

        # two place-holders for when adversary is not executed
        adversary_drugs_penalty = torch.Tensor([0])
        adversary_cell_types_penalty = torch.Tensor([0])

        if self.iteration % self.hparams["adversary_steps"]:# and self.iteration > self.hparams["delay_adversary"]:
            adversary_drugs_penalty = torch.autograd.grad(
                adversary_drugs_predictions.sum(),
                latent_basal,
                create_graph=True)[0].pow(2).mean()

            adversary_cell_types_penalty = torch.autograd.grad(
                adversary_cell_types_predictions.sum(),
                latent_basal,
                create_graph=True)[0].pow(2).mean()

            self.optimizer_adversaries.zero_grad()
            (adversary_drugs_loss +
             self.hparams["penalty_adversary"] *
             adversary_drugs_penalty +
             adversary_cell_types_loss +
             self.hparams["penalty_adversary"] *
             adversary_cell_types_penalty).backward()
            self.optimizer_adversaries.step()
        else:
            self.optimizer_autoencoder.zero_grad()
            self.optimizer_dosers.zero_grad()
            loss = (reconstruction_loss -
                    self.hparams["reg_adversary"] *
                    adversary_drugs_loss -
                    self.hparams["reg_adversary"] *
                    adversary_cell_types_loss)
            if self.is_vae:
                kl_loss = self.hparams["kl_weight"] * (kl_loss_latent + kl_loss_proteins_back)
                loss += kl_loss
            loss.backward()
            self.optimizer_autoencoder.step()
            self.optimizer_dosers.step()

        self.iteration += 1

        stats_dict = {
            "loss_reconstruction": reconstruction_loss.item(),
            "loss_adv_drugs": adversary_drugs_loss.item(),
            "loss_adv_cell_types": adversary_cell_types_loss.item(),
            "penalty_adv_drugs": adversary_drugs_penalty.item(),
            "penalty_adv_cell_types": adversary_cell_types_penalty.item(),
        }

        if self.num_proteins is not None:
            stats_dict["loss_reconstruction_genes"] = reconstruction_loss_genes.item()
            stats_dict["loss_reconstruction_proteins"] = reconstruction_loss_proteins.item()

        if self.is_vae:
            stats_dict["kl_weight"] = current_alpha
            stats_dict["kl_loss_latent"] = kl_loss_latent.item()
            stats_dict["kl_loss_proteins_back"] = kl_loss_proteins_back.item()

        return stats_dict


class PoEComPert(BaseModel):
    """
    The ComPert autoencoder extended by the possibility to integrate rna and protein data using a product-of-experts.
    The rna and protein data is encoded/ decoded using separate encoders/ decoders.
    """

    def __init__(
            self,
            num_genes,
            num_drugs,
            num_cell_types,
            device="cpu",
            seed=0,
            patience=5,
            loss_ae='gauss',
            doser_type='logsigm',
            decoder_activation='linear',
            hparams="",
            is_vae=False,
            num_proteins=None, ):
        super(PoEComPert, self).__init__()
        # set generic attributes
        self.num_genes = num_genes
        self.num_proteins = num_proteins
        self.num_drugs = num_drugs
        self.num_cell_types = num_cell_types
        self.device = device
        self.seed = seed
        self.loss_ae = loss_ae
        self.is_vae = is_vae
        # early-stopping
        self.patience = patience
        self.best_score = -1e3
        self.patience_trials = 0

        # set hyperparameters
        self.set_hparams_(seed, hparams)

        if not self.is_vae:
            print("This model only works with VAE turned on. Parameter is set to True.")
            self.is_vae = True

        vae_dim_factor = 1
        if self.is_vae:
            vae_dim_factor = 2

        # set models
        self.rna_encoder = MLP(
            [self.num_genes] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [self.hparams["dim"] * vae_dim_factor])

        self.rna_decoder = MLP(
            [self.hparams["dim"]] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [self.num_genes * 2], last_layer_act=decoder_activation)

        self.protein_encoder = MLP(
            [self.num_proteins] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [self.hparams["dim"] * vae_dim_factor])

        self.protein_decoder = MLP(
            [self.hparams["dim"]] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [self.num_proteins * 2], last_layer_act=decoder_activation)

        self.adversary_drugs = MLP(
            [self.hparams["dim"]] +
            [self.hparams["adversary_width"]] *
            self.hparams["adversary_depth"] +
            [num_drugs])

        self.adversary_cell_types = MLP(
            [self.hparams["dim"]] +
            [self.hparams["adversary_width"]] *
            self.hparams["adversary_depth"] +
            [num_cell_types])

        # set dosers
        self.doser_type = doser_type
        if doser_type == 'mlp':
            self.dosers = torch.nn.ModuleList()
            for _ in range(num_drugs):
                self.dosers.append(
                    MLP([1] +
                        [self.hparams["dosers_width"]] *
                        self.hparams["dosers_depth"] +
                        [1],
                        batch_norm=False))
        else:
            self.dosers = GeneralizedSigmoid(num_drugs, self.device,
                                             nonlin=doser_type)

        self.drug_embeddings = torch.nn.Embedding(
            num_drugs, self.hparams["dim"])
        self.cell_type_embeddings = torch.nn.Embedding(
            num_cell_types, self.hparams["dim"])

        # losses
        if self.loss_ae == 'nb':
            self.loss_autoencoder = NB()
        else:
            self.loss_autoencoder = Gaussian()
        if self.is_vae:
            self.kl_loss = KLD()
        self.loss_adversary_drugs = torch.nn.BCEWithLogitsLoss()
        self.loss_adversary_cell_types = torch.nn.CrossEntropyLoss()
        self.iteration = 0

        self.to(self.device)

        # optimizers
        self.optimizer_autoencoder = torch.optim.Adam(
            list(self.rna_encoder.parameters()) +
            list(self.rna_decoder.parameters()) +
            list(self.protein_encoder.parameters()) +
            list(self.protein_decoder.parameters()) +
            list(self.drug_embeddings.parameters()) +
            list(self.cell_type_embeddings.parameters()),
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"])

        self.optimizer_adversaries = torch.optim.Adam(
            list(self.adversary_drugs.parameters()) +
            list(self.adversary_cell_types.parameters()),
            lr=self.hparams["adversary_lr"],
            weight_decay=self.hparams["adversary_wd"])

        self.optimizer_dosers = torch.optim.Adam(
            self.dosers.parameters(),
            lr=self.hparams["dosers_lr"],
            weight_decay=self.hparams["dosers_wd"])

        # learning rate schedulers
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.hparams["step_size_lr"])

        self.scheduler_adversary = torch.optim.lr_scheduler.StepLR(
            self.optimizer_adversaries, step_size=self.hparams["step_size_lr"])

        self.scheduler_dosers = torch.optim.lr_scheduler.StepLR(
            self.optimizer_dosers, step_size=self.hparams["step_size_lr"])

        self.history = {'epoch': [], 'stats_epoch': []}

    def set_hparams_(self, seed, hparams):
        """
        Set hyper-parameters to (i) default values if `seed=0`, (ii) random
        values if `seed != 0`, or (iii) values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        default = (seed == 0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.hparams = {
            #"delay_adversary": -1 if default else
            #int(np.random.choice([50, 100, 200])),
            "dim": 256 if default else
            int(np.random.choice([128, 256, 512])),
            "dosers_width": 64 if default else
            int(np.random.choice([32, 64, 128])),
            "dosers_depth": 2 if default else
            int(np.random.choice([1, 2, 3])),
            "dosers_lr": 1e-3 if default else
            float(10 ** np.random.uniform(-4, -2)),
            "dosers_wd": 1e-7 if default else
            float(10 ** np.random.uniform(-8, -5)),
            "autoencoder_width": 512 if default else
            int(np.random.choice([256, 512, 1024])),
            "autoencoder_depth": 4 if default else
            int(np.random.choice([3, 4, 5])),
            "adversary_width": 128 if default else
            int(np.random.choice([64, 128, 256])),
            "adversary_depth": 3 if default else
            int(np.random.choice([2, 3, 4])),
            "reg_adversary": 5 if default else
            float(10 ** np.random.uniform(-2, 2)),
            "penalty_adversary": 3 if default else
            float(10 ** np.random.uniform(-2, 1)),
            "autoencoder_lr": 1e-3 if default else
            float(10 ** np.random.uniform(-4, -2)),
            "adversary_lr": 3e-4 if default else
            float(10 ** np.random.uniform(-5, -3)),
            "autoencoder_wd": 1e-6 if default else
            float(10 ** np.random.uniform(-8, -4)),
            "adversary_wd": 1e-4 if default else
            float(10 ** np.random.uniform(-6, -3)),
            "adversary_steps": 3 if default else
            int(np.random.choice([1, 2, 3, 4, 5])),
            "batch_size": 128 if default else
            int(np.random.choice([64, 128, 256, 512])),
            "step_size_lr": 45 if default else
            int(np.random.choice([15, 25, 45])),
            "kl_annealing_frac": 0.3 if default else
            float(np.random.uniform(0, 8) / 10),
            "recon_weight_pro": 0.1 if default else
            float(np.random.choice([0.01, 0.1, 1, 10])),
            "kl_weight": 1 if default else
            float(np.random.choice([0.1, 1, 2, 5, 10])),
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams

    def move_inputs_(self, genes, drugs, cell_types, proteins, raw_genes, raw_proteins):
        """
        Move minibatch tensors to CPU/GPU.
        """
        if genes.device.type != self.device:
            genes = genes.to(self.device)
            drugs = drugs.to(self.device)
            cell_types = cell_types.to(self.device)
            if proteins is not None:
                proteins = proteins.to(self.device)
            if raw_genes is not None:
                raw_genes = raw_genes.to(self.device)
            if raw_proteins is not None:
                raw_proteins = raw_proteins.to(self.device)
        return genes, drugs, cell_types, proteins, raw_genes, raw_proteins

    def product_of_experts(self, mu_rna, mu_protein, logvar_rna, logvar_protein):
        """
        Mixture of latent distributions using a product-of-experts.

        From https://github.com/SchubertLab/mvTCR/blob/master/tcr_embedding/models/poe.py.
        """

        # formula: var_joint = inv(inv(var_prior) + sum(inv(var_modalities)))
        # sum up all inverse vars, logvars first needs to be converted to var,
        # last 1.0 is coming from the prior
        logvar_joint = 1.0 / torch.exp(logvar_rna) + \
                       1.0 / torch.exp(logvar_protein) + 1.0
        logvar_joint = torch.log(1.0 / logvar_joint)  # inverse and convert to logvar

        # formula: mu_joint = (mu_prior*inv(var_prior) +
        # sum(mu_modalities*inv(var_modalities))) * var_joint,
        # where mu_prior = 0.0
        mu_joint = mu_rna * (1.0 / torch.exp(logvar_rna)) + \
                   mu_protein * (1.0 / torch.exp(logvar_protein))
        mu_joint = mu_joint * torch.exp(logvar_joint)

        return mu_joint, logvar_joint

    def get_latent(self, genes, drugs, cell_types, proteins=None,
                   return_latent_treated=False, use_latent_mean=False):
        """
        Return the joint latent representation of the inputs.
        """

        genes, drugs, cell_types, proteins, _, _ = self.move_inputs_(
            genes, drugs, cell_types, proteins, None, None)

        rna_latent_basal = self.rna_encoder(genes)
        protein_latent_basal = self.protein_encoder(proteins)

        rna_dim = rna_latent_basal.size(1) // 2
        rna_latent_basal[:, rna_dim:] = \
            rna_latent_basal[:, rna_dim:].exp().add(1).log().add(1e-3)
        rna_latent_mean = rna_latent_basal[:, :rna_dim]
        rna_latent_log_var = rna_latent_basal[:, rna_dim:].log()

        protein_dim = protein_latent_basal.size(1) // 2
        protein_latent_basal[:, protein_dim:] = \
            protein_latent_basal[:, protein_dim:].exp().add(1).log().add(1e-3)
        protein_latent_mean = protein_latent_basal[:, :protein_dim]
        protein_latent_log_var = protein_latent_basal[:, protein_dim:].log()

        joint_latent_mean, joint_latent_log_var = self.product_of_experts(
            rna_latent_mean, protein_latent_mean, rna_latent_log_var, protein_latent_log_var)
        joint_latent_basal = self.reparameterize(joint_latent_mean, joint_latent_log_var)

        if use_latent_mean:
            joint_latent_basal = joint_latent_mean

        drug_emb = self.compute_drug_embeddings_(drugs)
        cell_emb = self.cell_type_embeddings(cell_types.argmax(1))

        joint_latent_treated = joint_latent_basal + drug_emb + cell_emb

        if return_latent_treated:
            return joint_latent_basal, joint_latent_treated
        return joint_latent_basal

    def predict(self, genes, drugs, cell_types, proteins=None,
                return_latent_basal=False, use_latent_mean=False):
        """
        Predict "what would have the gene expression `genes` been, had the
        cells in `genes` with cell types `cell_types` been treated with
        combination of drugs `drugs`.
        """

        genes, drugs, cell_types, proteins, _, _ = self.move_inputs_(
            genes, drugs, cell_types, proteins, None, None)

        rna_latent_basal = self.rna_encoder(genes)
        protein_latent_basal = self.protein_encoder(proteins)

        rna_dim = rna_latent_basal.size(1) // 2
        rna_latent_basal[:, rna_dim:] = \
            rna_latent_basal[:, rna_dim:].exp().add(1).log().add(1e-3)
        rna_latent_mean = rna_latent_basal[:, :rna_dim]
        rna_latent_log_var = rna_latent_basal[:, rna_dim:].log()
        rna_latent_basal = self.reparameterize(rna_latent_mean, rna_latent_log_var)

        protein_dim = protein_latent_basal.size(1) // 2
        protein_latent_basal[:, protein_dim:] = \
            protein_latent_basal[:, protein_dim:].exp().add(1).log().add(1e-3)
        protein_latent_mean = protein_latent_basal[:, :protein_dim]
        protein_latent_log_var = protein_latent_basal[:, protein_dim:].log()
        protein_latent_basal = self.reparameterize(protein_latent_mean, protein_latent_log_var)

        joint_latent_mean, joint_latent_log_var = self.product_of_experts(
            rna_latent_mean, protein_latent_mean, rna_latent_log_var, protein_latent_log_var)
        joint_latent_basal = self.reparameterize(joint_latent_mean, joint_latent_log_var)

        if use_latent_mean:
            rna_latent_basal = rna_latent_mean
            protein_latent_basal = protein_latent_mean
            joint_latent_basal = joint_latent_mean

        latent_mean = [rna_latent_mean, protein_latent_mean, joint_latent_mean]
        latent_log_var = [rna_latent_log_var, protein_latent_log_var, joint_latent_log_var]
        latent_basal = [rna_latent_basal, protein_latent_basal, joint_latent_basal]

        drug_emb = self.compute_drug_embeddings_(drugs)
        cell_emb = self.cell_type_embeddings(cell_types.argmax(1))

        # TODO: Bence burası saçma. drug_emb ve cell_emb'i böyle toplama yerine diğerlerini concat edip sonrasında
        #  büyük birer drug_emb/cell_emb matrixleriyle toplamak daha mantıklı. çünkü burada aynı embedding'in iki
        #  modaliteye de ayrı ayrı uygun olması üzerinde baskı var.
        latent_treated = [latent_basal_ + drug_emb + cell_emb for latent_basal_ in latent_basal]

        gene_reconstructions = []
        for latent_treated_ in [latent_treated[0], latent_treated[2]]:
            gene_reconstructions.append(self.rna_decoder(latent_treated_))

        protein_reconstructions = []
        for latent_treated_ in [latent_treated[1], latent_treated[2]]:
            protein_reconstructions.append(self.protein_decoder(latent_treated_))

        # TODO: Ne olmuş lan burada. bunlar ayrı ayrı decodera girmiş. sonuçta da yan yana duruyorlar.

        for i in range(len(gene_reconstructions)):
            # convert variance estimates to a positive value in [1e-3, \infty)
            dim = gene_reconstructions[i].size(1) // 2
            gene_reconstructions[i][:, dim:] = \
                gene_reconstructions[i][:, dim:].exp().add(1).log().add(1e-3)

            if self.loss_ae == 'nb':
                gene_reconstructions[i][:, :dim] = \
                    gene_reconstructions[i][:, :dim].exp().add(1).log().add(1e-4)

        for i in range(len(protein_reconstructions)):
            # convert variance estimates to a positive value in [1e-3, \infty)
            dim = protein_reconstructions[i].size(1) // 2
            protein_reconstructions[i][:, dim:] = \
                protein_reconstructions[i][:, dim:].exp().add(1).log().add(1e-3)

            if self.loss_ae == 'nb':
                protein_reconstructions[i][:, :dim] = \
                    protein_reconstructions[i][:, :dim].exp().add(1).log().add(1e-4)

        if return_latent_basal:
            return gene_reconstructions, protein_reconstructions, \
                   latent_basal, latent_mean, latent_log_var

        return gene_reconstructions, protein_reconstructions

    def sample(self, genes, drugs, cell_types, proteins):
        """
        Sample from the decoder distributions.
        """

        with torch.no_grad():
            gene_reconstructions, protein_reconstructions, = self.predict(
                self, genes, drugs, cell_types, proteins)

        # TODO: İşe bak sadece joint'e bakıyor. neden ayrı ayrı soktuk o zaman.

        gene_reconstructions = gene_reconstructions[-1]
        protein_reconstructions = protein_reconstructions[-1]

        dim_genes = gene_reconstructions.size(1) // 2
        if self.loss_ae == 'nb':
            counts_genes, logits_genes = _convert_mean_disp_to_counts_logits(
                gene_reconstructions[:, dim_genes:],
                gene_reconstructions[:, :dim_genes])
            dist_genes = torch.distributions.negative_binomial.NegativeBinomial(
                total_count=counts_genes, logits=logits_genes)
        else:
            dist_genes = torch.distributions.Normal(
                gene_reconstructions[:, dim_genes:],
                gene_reconstructions[:, :dim_genes].log().div(2).exp())

        sample_genes = dist_genes.sample().cpu()
        sample_proteins = None

        if self.num_proteins is not None:
            dim_proteins = protein_reconstructions.size(1) // 2
            if self.loss_ae == 'nb':
                counts_proteins, logits_proteins = _convert_mean_disp_to_counts_logits(
                    protein_reconstructions[:, dim_proteins:],
                    protein_reconstructions[:, :dim_proteins])
                dist_proteins = torch.distributions.negative_binomial.NegativeBinomial(
                    total_count=counts_proteins, logits=logits_proteins)
            else:
                dist_proteins = torch.distributions.Normal(
                    protein_reconstructions[:, dim_proteins:],
                    protein_reconstructions[:, :dim_proteins].log().div(2).exp())
            sample_proteins = dist_proteins.sample().cpu()

        return sample_genes, sample_proteins

    def update(self, genes, drugs, cell_types, proteins,
               raw_genes, raw_proteins, epoch, max_epoch):
        """
        Update ComPert's parameters given a minibatch of genes, drugs, and
        cell types.
        """

        genes, drugs, cell_types, proteins, raw_genes, raw_proteins = self.move_inputs_(
            genes, drugs, cell_types, proteins, raw_genes, raw_proteins)

        if self.loss_ae == 'nb':
            # use raw counts as true values
            genes = raw_genes
            proteins = raw_proteins

        gene_reconstructions, protein_reconstructions, latent_basal, latent_mean, latent_log_var = \
            self.predict(genes, drugs, cell_types, proteins, return_latent_basal=True)

        reconstruction_loss_genes = []
        for i, reconstruction_ in enumerate(gene_reconstructions):
            reconstruction_loss_genes.append(
                self.loss_autoencoder(reconstruction_, genes, reduction="mean"))

        reconstruction_loss_proteins = []
        for i, reconstruction_ in enumerate(protein_reconstructions):
            reconstruction_loss_proteins.append(
                self.loss_autoencoder(reconstruction_, proteins, reduction="mean"))

        reconstruction_loss_genes = 0.5 * sum(reconstruction_loss_genes)
        reconstruction_loss_proteins = 0.5 * sum(reconstruction_loss_proteins)

        # TODO: ortalamasını aldı. decoder'ın hem sadece protein hem protein+rna'i decode etmesini beklemek mantıklı mı

        reconstruction_loss = \
            reconstruction_loss_genes + \
            self.hparams["recon_weight_pro"] * \
            reconstruction_loss_proteins

        current_alpha = self.kl_annealing(epoch, max_epoch, self.hparams["kl_annealing_frac"])

        kl_loss = \
            1.0 / 3.0 * current_alpha * \
            (self.kl_loss(latent_mean[0], latent_log_var[0], reduction="mean") +
             self.kl_loss(latent_mean[1], latent_log_var[1], reduction="mean") +
             self.kl_loss(latent_mean[2], latent_log_var[2], reduction="mean"))

        adversary_drugs_loss, adversary_drugs_predictions = [], []
        adversary_cell_types_loss, adversary_cell_types_predictions = [], []
        for i, latent_basal_ in enumerate(latent_basal):
            adversary_drugs_predictions.append(self.adversary_drugs(
                latent_basal_))
            adversary_drugs_loss.append(self.loss_adversary_drugs(
                adversary_drugs_predictions[i], drugs.gt(0).float()))

            adversary_cell_types_predictions.append(self.adversary_cell_types(
                latent_basal_))
            adversary_cell_types_loss.append(self.loss_adversary_cell_types(
                adversary_cell_types_predictions[i], cell_types.argmax(1)))

        # TODO: bence yine aynı problem: bir adversalial nn'in hem protein, hem rna, hem de poe'yi tanımasını beklemek?!

        adversary_drugs_loss = 1.0 / 3.0 * sum(adversary_drugs_loss)
        adversary_cell_types_loss = 1.0 / 3.0 * sum(adversary_cell_types_loss)

        # two place-holders for when adversary is not executed
        adversary_drugs_penalty = torch.Tensor([0])
        adversary_cell_types_penalty = torch.Tensor([0])

        if self.iteration % self.hparams["adversary_steps"]: # and self.iteration > self.hparams["delay_adversary"]:
            adversary_drugs_penalty = sum(  # TODO: Weird alert!
                [tmp.pow(2).mean() for tmp in torch.autograd.grad(
                    [adversary_drugs_predictions_.sum() for
                     adversary_drugs_predictions_ in
                     adversary_drugs_predictions],
                    latent_basal,
                    create_graph=True)]).div(3)

            adversary_cell_types_penalty = sum(
                [tmp.pow(2).mean() for tmp in torch.autograd.grad(
                    [adversary_cell_types_predictions_.sum() for
                     adversary_cell_types_predictions_ in
                     adversary_cell_types_predictions],
                    latent_basal,
                    create_graph=True)]).div(3)

            self.optimizer_adversaries.zero_grad()
            (adversary_drugs_loss +
             self.hparams["penalty_adversary"] *
             adversary_drugs_penalty +
             adversary_cell_types_loss +
             self.hparams["penalty_adversary"] *
             adversary_cell_types_penalty).backward()
            self.optimizer_adversaries.step()
        else:
            self.optimizer_autoencoder.zero_grad()
            self.optimizer_dosers.zero_grad()
            loss = (reconstruction_loss -
                    self.hparams["reg_adversary"] *
                    adversary_drugs_loss -
                    self.hparams["reg_adversary"] *
                    adversary_cell_types_loss)
            if self.is_vae:
                loss += self.hparams["kl_weight"] * kl_loss
            loss.backward()
            self.optimizer_autoencoder.step()
            self.optimizer_dosers.step()

        self.iteration += 1

        stats_dict = {
            "loss_reconstruction": reconstruction_loss.item(),
            "loss_adv_drugs": adversary_drugs_loss.item(),
            "loss_adv_cell_types": adversary_cell_types_loss.item(),
            "penalty_adv_drugs": adversary_drugs_penalty.item(),
            "penalty_adv_cell_types": adversary_cell_types_penalty.item(),
        }

        if self.num_proteins is not None:
            stats_dict["loss_reconstruction_genes"] = reconstruction_loss_genes.item()
            stats_dict["loss_reconstruction_proteins"] = reconstruction_loss_proteins.item()

        if self.is_vae:
            stats_dict["kl_weight"] = current_alpha
            stats_dict["kl_loss"] = kl_loss.item()

        return stats_dict


class TotalPoEComPert(BaseModel):
    """
    The ComPert autoencoder combined with the concepts of the totalVI model and a product-of-experts for integrating
    rna and protein modalities. The genes and proteins are encoded using separate encoders. Afterwards, genes and
    proteins are also decoded separately. The genes are decoded using a default decoder. The proteins are decoded using
    five different decoders: background mean, foreground mean, protein mixing of background and foreground and protein
    dispersion. The loss of the proteins is computed by using a Negative Binomial Mixture distribution.
    """

    def __init__(
            self,
            num_genes,
            num_drugs,
            num_cell_types,
            device="cpu",
            seed=0,
            patience=5,
            loss_ae='gauss',
            doser_type='logsigm',
            decoder_activation='linear',
            hparams="",
            is_vae=False,
            num_proteins=None, ):
        super(TotalPoEComPert, self).__init__()
        # set generic attributes
        self.num_genes = num_genes
        self.num_proteins = num_proteins
        self.num_drugs = num_drugs
        self.num_cell_types = num_cell_types
        self.device = device
        self.seed = seed
        self.loss_ae = loss_ae
        self.is_vae = is_vae
        # early-stopping
        self.patience = patience
        self.best_score = -1e3
        self.patience_trials = 0

        # set hyperparameters
        self.set_hparams_(seed, hparams)

        if not self.is_vae:
            print("This model only works with VAE turned on. Parameter is set to True.")
            self.is_vae = True

        vae_dim_factor = 1
        if is_vae:
            vae_dim_factor = 2

        if num_proteins is None:
            raise ValueError("Please specify protein expressions for this model!")

        # set models
        self.rna_encoder = MLP(
            [self.num_genes] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [self.hparams["dim"] * vae_dim_factor])

        self.rna_decoder = MLP(
            [self.hparams["dim"]] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [self.num_genes * 2], last_layer_act=decoder_activation)

        self.protein_encoder = MLP(
            [self.num_proteins] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [self.hparams["dim"] * vae_dim_factor])

        self.protein_back_decoder = MLP(
            [self.hparams["dim"]] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [num_proteins * vae_dim_factor])

        self.protein_fore_scale_decoder = MLP(
            [self.hparams["dim"]] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [num_proteins],
            last_layer_act='ReLU', full_last_layer_act=True)

        self.protein_mixing_decoder = MLP(
            [self.hparams["dim"]] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [num_proteins])

        self.protein_dispersion_decoder = MLP(
            [self.hparams["dim"]] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [num_proteins])

        self.adversary_drugs = MLP(
            [self.hparams["dim"]] +
            [self.hparams["adversary_width"]] *
            self.hparams["adversary_depth"] +
            [num_drugs])

        self.adversary_cell_types = MLP(
            [self.hparams["dim"]] +
            [self.hparams["adversary_width"]] *
            self.hparams["adversary_depth"] +
            [num_cell_types])

        self.background_pro_alpha = torch.nn.Parameter(
            torch.randn(num_proteins))

        self.background_pro_log_beta = torch.nn.Parameter(
            torch.clamp(torch.randn(num_proteins), -10, 1))

        # set dosers
        self.doser_type = doser_type
        if doser_type == 'mlp':
            self.dosers = torch.nn.ModuleList()
            for _ in range(num_drugs):
                self.dosers.append(
                    MLP([1] +
                        [self.hparams["dosers_width"]] *
                        self.hparams["dosers_depth"] +
                        [1],
                        batch_norm=False))
        else:
            self.dosers = GeneralizedSigmoid(num_drugs, self.device,
                                             nonlin=doser_type)

        self.drug_embeddings = torch.nn.Embedding(
            num_drugs, self.hparams["dim"])
        self.cell_type_embeddings = torch.nn.Embedding(
            num_cell_types, self.hparams["dim"])

        # losses
        if self.loss_ae == 'nb':
            self.loss_autoencoder_genes = NB()
            self.loss_autoencoder_proteins = NBMixture()
        else:
            self.loss_autoencoder_genes = Gaussian()
            self.loss_autoencoder_proteins = GaussianMixture()
        if self.is_vae:
            self.kl_loss_latent = KLD()
            self.kl_loss_proteins_back = KLDGaussians()
        self.loss_adversary_drugs = torch.nn.BCEWithLogitsLoss()
        self.loss_adversary_cell_types = torch.nn.CrossEntropyLoss()
        self.iteration = 0

        self.to(self.device)

        # optimizers
        self.optimizer_autoencoder = torch.optim.Adam(
            list(self.rna_encoder.parameters()) +
            list(self.rna_decoder.parameters()) +
            list(self.protein_encoder.parameters()) +
            list(self.protein_back_decoder.parameters()) +
            list(self.protein_fore_scale_decoder.parameters()) +
            list(self.protein_mixing_decoder.parameters()) +
            list(self.protein_dispersion_decoder.parameters()) +
            list(self.drug_embeddings.parameters()) +
            list(self.cell_type_embeddings.parameters()) +
            [self.background_pro_alpha] +
            [self.background_pro_log_beta],
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"])

        self.optimizer_adversaries = torch.optim.Adam(
            list(self.adversary_drugs.parameters()) +
            list(self.adversary_cell_types.parameters()),
            lr=self.hparams["adversary_lr"],
            weight_decay=self.hparams["adversary_wd"])

        self.optimizer_dosers = torch.optim.Adam(
            self.dosers.parameters(),
            lr=self.hparams["dosers_lr"],
            weight_decay=self.hparams["dosers_wd"])

        # learning rate schedulers
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.hparams["step_size_lr"])

        self.scheduler_adversary = torch.optim.lr_scheduler.StepLR(
            self.optimizer_adversaries, step_size=self.hparams["step_size_lr"])

        self.scheduler_dosers = torch.optim.lr_scheduler.StepLR(
            self.optimizer_dosers, step_size=self.hparams["step_size_lr"])

        self.history = {'epoch': [], 'stats_epoch': []}

    def set_hparams_(self, seed, hparams):
        """
        Set hyper-parameters to (i) default values if `seed=0`, (ii) random
        values if `seed != 0`, or (iii) values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        default = (seed == 0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.hparams = {
            #"delay_adversary": -1 if default else
            #int(np.random.choice([50, 100, 200])),
            "dim": 256 if default else
            int(np.random.choice([128, 256, 512])),
            "dosers_width": 64 if default else
            int(np.random.choice([32, 64, 128])),
            "dosers_depth": 2 if default else
            int(np.random.choice([1, 2, 3])),
            "dosers_lr": 1e-3 if default else
            float(10 ** np.random.uniform(-4, -2)),
            "dosers_wd": 1e-7 if default else
            float(10 ** np.random.uniform(-8, -5)),
            "autoencoder_width": 512 if default else
            int(np.random.choice([256, 512, 1024])),
            "autoencoder_depth": 4 if default else
            int(np.random.choice([3, 4, 5])),
            "adversary_width": 128 if default else
            int(np.random.choice([64, 128, 256])),
            "adversary_depth": 3 if default else
            int(np.random.choice([2, 3, 4])),
            "reg_adversary": 5 if default else
            float(10 ** np.random.uniform(-2, 2)),
            "penalty_adversary": 3 if default else
            float(10 ** np.random.uniform(-2, 1)),
            "autoencoder_lr": 1e-3 if default else
            float(10 ** np.random.uniform(-4, -2)),
            "adversary_lr": 3e-4 if default else
            float(10 ** np.random.uniform(-5, -3)),
            "autoencoder_wd": 1e-6 if default else
            float(10 ** np.random.uniform(-8, -4)),
            "adversary_wd": 1e-4 if default else
            float(10 ** np.random.uniform(-6, -3)),
            "adversary_steps": 3 if default else
            int(np.random.choice([1, 2, 3, 4, 5])),
            "batch_size": 128 if default else
            int(np.random.choice([64, 128, 256, 512])),
            "step_size_lr": 45 if default else
            int(np.random.choice([15, 25, 45])),
            "kl_annealing_frac": 0.3 if default else
            float(np.random.uniform(0, 8) / 10),
            "recon_weight_pro": 0.1 if default else
            float(np.random.choice([0.01, 0.1, 1, 10])),
            "kl_weight": 1 if default else
            float(np.random.choice([0.1, 1, 2, 5, 10])),
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams

    def move_inputs_(self, genes, drugs, cell_types, proteins, raw_genes, raw_proteins):
        """
        Move minibatch tensors to CPU/GPU.
        """
        if genes.device.type != self.device:
            genes = genes.to(self.device)
            drugs = drugs.to(self.device)
            cell_types = cell_types.to(self.device)
            if proteins is not None:
                proteins = proteins.to(self.device)
            if raw_genes is not None:
                raw_genes = raw_genes.to(self.device)
            if raw_proteins is not None:
                raw_proteins = raw_proteins.to(self.device)
        return genes, drugs, cell_types, proteins, raw_genes, raw_proteins

    def product_of_experts(self, mu_rna, mu_protein, logvar_rna, logvar_protein):
        """
        Mixture of latent distributions using a product-of-experts.

        From https://github.com/SchubertLab/mvTCR/blob/master/tcr_embedding/models/poe.py.
        """

        # formula: var_joint = inv(inv(var_prior) + sum(inv(var_modalities)))
        # sum up all inverse vars, logvars first needs to be converted to var,
        # last 1.0 is coming from the prior
        logvar_joint = 1.0 / torch.exp(logvar_rna) + \
                       1.0 / torch.exp(logvar_protein) + 1.0
        logvar_joint = torch.log(1.0 / logvar_joint)  # inverse and convert to logvar

        # formula: mu_joint = (mu_prior*inv(var_prior) +
        # sum(mu_modalities*inv(var_modalities))) * var_joint,
        # where mu_prior = 0.0
        mu_joint = mu_rna * (1.0 / torch.exp(logvar_rna)) + \
                   mu_protein * (1.0 / torch.exp(logvar_protein))
        mu_joint = mu_joint * torch.exp(logvar_joint)

        return mu_joint, logvar_joint

    def get_latent(self, genes, drugs, cell_types, proteins=None,
                   return_latent_treated=False, use_latent_mean=False):
        """
        Return the joint latent representation of the inputs.
        """

        genes, drugs, cell_types, proteins, _, _ = self.move_inputs_(
            genes, drugs, cell_types, proteins, None, None)

        rna_latent_basal = self.rna_encoder(genes)
        protein_latent_basal = self.protein_encoder(proteins)

        rna_dim = rna_latent_basal.size(1) // 2
        rna_latent_basal[:, rna_dim:] = \
            rna_latent_basal[:, rna_dim:].exp().add(1).log().add(1e-3)
        rna_latent_mean = rna_latent_basal[:, :rna_dim]
        rna_latent_log_var = rna_latent_basal[:, rna_dim:].log()

        protein_dim = protein_latent_basal.size(1) // 2
        protein_latent_basal[:, protein_dim:] = \
            protein_latent_basal[:, protein_dim:].exp().add(1).log().add(1e-3)
        protein_latent_mean = protein_latent_basal[:, :protein_dim]
        protein_latent_log_var = protein_latent_basal[:, protein_dim:].log()

        joint_latent_mean, joint_latent_log_var = self.product_of_experts(
            rna_latent_mean, protein_latent_mean, rna_latent_log_var, protein_latent_log_var)
        joint_latent_basal = self.reparameterize(joint_latent_mean, joint_latent_log_var)

        if use_latent_mean:
            joint_latent_basal = joint_latent_mean

        drug_emb = self.compute_drug_embeddings_(drugs)
        cell_emb = self.cell_type_embeddings(cell_types.argmax(1))

        joint_latent_treated = joint_latent_basal + drug_emb + cell_emb

        if return_latent_treated:
            return joint_latent_basal, joint_latent_treated
        return joint_latent_basal

    def predict(self, genes, drugs, cell_types, proteins=None,
                return_latent_basal=False, use_latent_mean=False):
        """
        Predict "what would have the gene expression `genes` been, had the
        cells in `genes` with cell types `cell_types` been treated with
        combination of drugs `drugs`.
        """

        genes, drugs, cell_types, proteins, _, _ = self.move_inputs_(
            genes, drugs, cell_types, proteins, None, None)

        rna_latent_basal = self.rna_encoder(genes)
        protein_latent_basal = self.protein_encoder(proteins)

        rna_dim = rna_latent_basal.size(1) // 2
        rna_latent_basal[:, rna_dim:] = \
            rna_latent_basal[:, rna_dim:].exp().add(1).log().add(1e-3)
        rna_latent_mean = rna_latent_basal[:, :rna_dim]
        rna_latent_log_var = rna_latent_basal[:, rna_dim:].log()
        rna_latent_basal = self.reparameterize(rna_latent_mean, rna_latent_log_var)

        protein_dim = protein_latent_basal.size(1) // 2
        protein_latent_basal[:, protein_dim:] = \
            protein_latent_basal[:, protein_dim:].exp().add(1).log().add(1e-3)
        protein_latent_mean = protein_latent_basal[:, :protein_dim]
        protein_latent_log_var = protein_latent_basal[:, protein_dim:].log()
        protein_latent_basal = self.reparameterize(protein_latent_mean, protein_latent_log_var)

        joint_latent_mean, joint_latent_log_var = self.product_of_experts(
            rna_latent_mean, protein_latent_mean, rna_latent_log_var, protein_latent_log_var)
        joint_latent_basal = self.reparameterize(joint_latent_mean, joint_latent_log_var)

        if use_latent_mean:
            rna_latent_basal = rna_latent_mean
            protein_latent_basal = protein_latent_mean
            joint_latent_basal = joint_latent_mean

        latent_mean = [rna_latent_mean, protein_latent_mean, joint_latent_mean]
        latent_log_var = [rna_latent_log_var, protein_latent_log_var, joint_latent_log_var]
        latent_basal = [rna_latent_basal, protein_latent_basal, joint_latent_basal]

        drug_emb = self.compute_drug_embeddings_(drugs)
        cell_emb = self.cell_type_embeddings(cell_types.argmax(1))

        latent_treated = [latent_basal_ + drug_emb + cell_emb for latent_basal_ in latent_basal]

        gene_reconstructions = []
        for latent_treated_ in [latent_treated[0], latent_treated[2]]:
            gene_reconstructions_ = self.rna_decoder(latent_treated_)

            # convert variance estimates to a positive value in [1e-3, \infty)
            dim = gene_reconstructions_.size(1) // 2
            gene_reconstructions_[:, dim:] = \
                gene_reconstructions_[:, dim:].exp().add(1).log().add(1e-3)

            if self.loss_ae == 'nb':
                gene_reconstructions_[:, :dim] = \
                    gene_reconstructions_[:, :dim].exp().add(1).log().add(1e-4)
                # gene_reconstructions[:, :dim] = torch.clamp(gene_reconstructions[:, :dim], min=1e-4, max=1e4)
                # gene_reconstructions[:, dim:] = torch.clamp(gene_reconstructions[:, dim:], min=1e-6, max=1e6)

            gene_reconstructions.append(gene_reconstructions_)

        protein_reconstructions, protein_back, protein_fore, protein_mixing, \
        alpha, beta = [], [], [], [], [], []
        for latent_treated_ in [latent_treated[1], latent_treated[2]]:
            protein_back_ = self.protein_back_decoder(latent_treated_)
            protein_fore_scale_ = self.protein_fore_scale_decoder(latent_treated_) + 1 + 1e-8
            protein_mixing_ = self.protein_mixing_decoder(latent_treated_)

            if self.is_vae:
                # convert variance estimates to a positive value
                dim = protein_back_.size(1) // 2
                alpha_ = protein_back_[:, :dim]
                beta_ = protein_back_[:, dim:].exp()
                log_protein_back_ = self.reparameterize(alpha_, 2 * beta_.log())
                protein_back_ = log_protein_back_.exp()
            else:
                protein_back_ = protein_back_.exp().add(1).log().add(1e-4)

            protein_fore_ = protein_fore_scale_ * protein_back_
            protein_mixing_transformed_ = 1 / (1 + torch.exp(-protein_mixing_))
            protein_reconstructions_ = (1 - protein_mixing_transformed_) * protein_fore_ + \
                                       protein_mixing_transformed_ * protein_back_

            protein_dispersion_ = self.protein_dispersion_decoder(latent_treated_)

            # convert variance estimates to a positive value in [1e-3, \infty)
            protein_dispersion_ = \
                protein_dispersion_.exp().add(1).log().add(1e-3)

            protein_reconstructions.append(torch.cat(
                [protein_reconstructions_, protein_dispersion_], dim=-1))

            protein_back.append(protein_back_)
            protein_fore.append(protein_fore_)
            protein_mixing.append(protein_mixing_)
            alpha.append(alpha_)
            beta.append(beta_)

        if return_latent_basal:
            return gene_reconstructions, protein_reconstructions, \
                   latent_basal, latent_mean, latent_log_var, \
                   protein_back, protein_fore, protein_mixing, alpha, beta

        return gene_reconstructions, protein_reconstructions

    def sample(self, genes, drugs, cell_types, proteins):
        """
        Sample from the decoder distributions.
        """

        with torch.no_grad():
            gene_reconstructions, protein_reconstructions, _, _, _, \
            protein_back, protein_fore, protein_mixing, _, _ = self.predict(
                self, genes, drugs, cell_types, proteins, return_latent_basal=True)

        gene_reconstructions = gene_reconstructions[-1]
        protein_reconstructions = protein_reconstructions[-1]

        dim_genes = gene_reconstructions.size(1) // 2
        if self.loss_ae == 'nb':
            counts_genes, logits_genes = _convert_mean_disp_to_counts_logits(
                gene_reconstructions[:, dim_genes:],
                gene_reconstructions[:, :dim_genes])
            dist_genes = torch.distributions.negative_binomial.NegativeBinomial(
                total_count=counts_genes, logits=logits_genes)
        else:
            dist_genes = torch.distributions.Normal(
                gene_reconstructions[:, dim_genes:],
                gene_reconstructions[:, :dim_genes].log().div(2).exp())

        sample_genes = dist_genes.sample().cpu()
        sample_proteins = None

        if self.num_proteins is not None:
            dim_proteins = protein_reconstructions.size(1) // 2
            if self.loss_ae == 'nb':
                protein_dispersion = protein_reconstructions[:, dim_proteins:]
                sample_proteins = NBMixture().sample(
                    protein_back, protein_fore, protein_dispersion, None, protein_mixing)
            else:
                protein_dispersion = protein_reconstructions[:, dim_proteins:]
                sample_proteins = GaussianMixture().sample(
                    protein_back, protein_fore, protein_dispersion, None, protein_mixing)

        return sample_genes, sample_proteins

    def update(self, genes, drugs, cell_types, proteins,
               raw_genes, raw_proteins, epoch, max_epoch):
        """
        Update ComPert's parameters given a minibatch of genes, drugs, and
        cell types.
        """

        genes, drugs, cell_types, proteins, raw_genes, raw_proteins = self.move_inputs_(
            genes, drugs, cell_types, proteins, raw_genes, raw_proteins)

        if self.loss_ae == 'nb':
            # use raw counts as true values
            genes = raw_genes
            proteins = raw_proteins

        gene_reconstructions, protein_reconstructions, \
        latent_basal, latent_mean, latent_log_var, \
        protein_back, protein_fore, protein_mixing, alpha, beta = \
            self.predict(genes, drugs, cell_types, proteins, return_latent_basal=True)

        reconstruction_loss_genes = []
        for i, reconstruction_ in enumerate(gene_reconstructions):
            reconstruction_loss_genes.append(
                self.loss_autoencoder_genes(reconstruction_, genes, reduction="mean"))

        reconstruction_loss_proteins = []
        for i in range(len(protein_reconstructions)):
            dim = protein_reconstructions[i].size(1) // 2
            protein_dispersion = protein_reconstructions[i][:, dim:]
            reconstruction_loss_proteins.append(self.loss_autoencoder_proteins(
                proteins[i], protein_back[i], protein_fore[i],
                protein_dispersion[i], None, protein_mixing[i], reduction="mean"))

        reconstruction_loss_genes = 0.5 * sum(reconstruction_loss_genes)
        reconstruction_loss_proteins = 0.5 * sum(reconstruction_loss_proteins)

        reconstruction_loss = \
            reconstruction_loss_genes + \
            self.hparams["recon_weight_pro"] * \
            reconstruction_loss_proteins

        current_alpha = self.kl_annealing(epoch, max_epoch, self.hparams["kl_annealing_frac"])
        py_back_alpha_prior = self.background_pro_alpha
        py_back_beta_prior = torch.log(torch.exp(self.background_pro_log_beta) ** 2)

        kl_loss_latent = \
            1.0 / 3.0 * current_alpha * \
            (self.kl_loss_latent(latent_mean[0], latent_log_var[0], reduction="mean") +
             self.kl_loss_latent(latent_mean[1], latent_log_var[1], reduction="mean") +
             self.kl_loss_latent(latent_mean[2], latent_log_var[2], reduction="mean"))
        kl_loss_proteins_back = \
            1.0 / 2.0 * current_alpha * \
            (self.kl_loss_proteins_back(alpha[0], beta[0], py_back_alpha_prior, py_back_beta_prior, reduction="mean") +
             self.kl_loss_proteins_back(alpha[1], beta[1], py_back_alpha_prior, py_back_beta_prior, reduction="mean"))

        adversary_drugs_loss, adversary_drugs_predictions = [], []
        adversary_cell_types_loss, adversary_cell_types_predictions = [], []
        for i, latent_basal_ in enumerate(latent_basal):
            adversary_drugs_predictions.append(self.adversary_drugs(
                latent_basal_))
            adversary_drugs_loss.append(self.loss_adversary_drugs(
                adversary_drugs_predictions[i], drugs.gt(0).float()))

            adversary_cell_types_predictions.append(self.adversary_cell_types(
                latent_basal_))
            adversary_cell_types_loss.append(self.loss_adversary_cell_types(
                adversary_cell_types_predictions[i], cell_types.argmax(1)))

        adversary_drugs_loss = 1.0 / 3.0 * sum(adversary_drugs_loss)
        adversary_cell_types_loss = 1.0 / 3.0 * sum(adversary_cell_types_loss)

        # two place-holders for when adversary is not executed
        adversary_drugs_penalty = torch.Tensor([0])
        adversary_cell_types_penalty = torch.Tensor([0])

        if self.iteration % self.hparams["adversary_steps"]:  # and self.iteration > self.hparams["delay_adversary"]:
            adversary_drugs_penalty = sum(
                [tmp.pow(2).mean() for tmp in torch.autograd.grad(
                    [adversary_drugs_predictions_.sum() for
                     adversary_drugs_predictions_ in
                     adversary_drugs_predictions],
                    latent_basal,
                    create_graph=True)]).div(3)

            adversary_cell_types_penalty = sum(
                [tmp.pow(2).mean() for tmp in torch.autograd.grad(
                    [adversary_cell_types_predictions_.sum() for
                     adversary_cell_types_predictions_ in
                     adversary_cell_types_predictions],
                    latent_basal,
                    create_graph=True)]).div(3)

            self.optimizer_adversaries.zero_grad()
            (adversary_drugs_loss +
             self.hparams["penalty_adversary"] *
             adversary_drugs_penalty +
             adversary_cell_types_loss +
             self.hparams["penalty_adversary"] *
             adversary_cell_types_penalty).backward()
            self.optimizer_adversaries.step()
        else:
            self.optimizer_autoencoder.zero_grad()
            self.optimizer_dosers.zero_grad()
            loss = (reconstruction_loss -
                    self.hparams["reg_adversary"] *
                    adversary_drugs_loss -
                    self.hparams["reg_adversary"] *
                    adversary_cell_types_loss)
            if self.is_vae:
                kl_loss = self.hparams["kl_weight"] * (kl_loss_latent + kl_loss_proteins_back)
                loss += kl_loss
            loss.backward()
            self.optimizer_autoencoder.step()
            self.optimizer_dosers.step()

        self.iteration += 1

        stats_dict = {
            "loss_reconstruction": reconstruction_loss.item(),
            "loss_adv_drugs": adversary_drugs_loss.item(),
            "loss_adv_cell_types": adversary_cell_types_loss.item(),
            "penalty_adv_drugs": adversary_drugs_penalty.item(),
            "penalty_adv_cell_types": adversary_cell_types_penalty.item(),
        }

        if self.num_proteins is not None:
            stats_dict["loss_reconstruction_genes"] = reconstruction_loss_genes.item()
            stats_dict["loss_reconstruction_proteins"] = reconstruction_loss_proteins.item()

        if self.is_vae:
            stats_dict["kl_weight"] = current_alpha
            stats_dict["kl_loss_latent"] = kl_loss_latent.item()
            stats_dict["kl_loss_proteins_back"] = kl_loss_proteins_back.item()

        return stats_dict
