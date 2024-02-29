import math
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap

from munkres import Munkres
from itertools import cycle
from scipy.sparse import load_npz
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix
from torch.autograd import Function
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

def coo_matrix_to_pytorch_tensor(mat):
    return torch.sparse_coo_tensor(np.array([mat.row, mat.col]), mat.data, mat.shape)

'''
class SingleModalityDataSet(Dataset):

    def __init__(self, file_mat, file_chr, **kwargs):
        self.file_mat = load_npz(file_mat)
        self.chr = np.loadtxt(file_chr, dtype=int).flatten().tolist()
        self.lens = self.file_mat.shape[0]
        self.dense_mat = self.file_mat.toarray().astype(np.float32)
        self.depths = np.sum(self.file_mat, axis=1).astype(np.float32)

    def __getitem__(self, index):
        feats = torch.from_numpy(self.dense_mat[index]).flatten()
        depths = torch.from_numpy(self.depths[index]).flatten()
        indexs = torch.tensor([index, ], dtype=torch.long)
        return feats, depths, indexs

    def __len__(self):
        return self.lens

    def get_features(self):
        return self.file_mat.shape[1]

'''
class SingleModalityDataSet(Dataset):
    

    def __init__(self, file_mat, file_chr = None, file_sample = None):
        self.file_mat = load_npz(file_mat)#.tocoo()
        self.chr = None
        self.sample = None
        if file_chr is not None:
            self.chr = np.loadtxt(file_chr, dtype=int).flatten().tolist()
        if file_sample is not None:
            self.sample = pd.read_csv(file_sample, delimiter = '\t', names = ['barcode', 'sample']).drop(columns = ['barcode'])
            self.sample['sample'] = self.sample['sample'].astype('category')
            self.sample = pd.get_dummies(self.sample).to_numpy().astype(np.float32)
        self.lens = self.file_mat.shape[0]
        self.dense_mat = self.file_mat.todense().astype(np.float32)
        #self.sparse_mat = coo_matrix_to_pytorch_tensor(self.file_mat)
        #print(f'Sparse matrix shape: {self.sparse_mat.shape}')
        self.depths = np.sum(self.file_mat, axis=1).astype(np.float32)

    def __getitem__(self, index):
        #print(f'Fetching index {index} from mat shape of {self.sparse_mat.shape}')
        #print(self.sparse_mat[5000])
        feats = torch.from_numpy(self.dense_mat[index]).flatten()#coo_matrix_to_pytorch_tensor(self.file_mat[index].tocoo()).to_dense()#self.sparse_mat[index]
        depths = torch.from_numpy(self.depths[index]).flatten()
        indexs = torch.tensor([index], dtype=torch.long)
        if self.sample is not None:
            sample = torch.tensor(self.sample[index, :]).flatten()
            return feats, (depths, sample), indexs
        return feats, depths, indexs

    def __len__(self):
        return self.lens

    def get_features(self):
        return self.file_mat.shape[1]


class MFeatDataSet(Dataset):
    '''Mixed-modal feature'''

    def __init__(self, file_mat, has_filename=False):
        self.file_mat = sio.loadmat(file_mat)
        self.lens = len(self.file_mat['X'])
        self.has_filename = has_filename

    def __getitem__(self, index):
        if self.has_filename:
            feat, file, modality = self.file_mat['X'][index]
        else:
            feat, modality = self.file_mat['X'][index]
        feat = feat.squeeze().astype(np.float32)
        cluster_label = self.file_mat['y'][0][index]
        cluster_label = np.float32(cluster_label) - 1
        modality_label = np.float32(modality[0])

        return np.float32(index), feat, modality_label, cluster_label

    def __len__(self):
        return self.lens


class SFeatDataSet(Dataset):
    '''Single modal feature'''

    def __init__(self, file_mat):
        self.file_mat = sio.loadmat(file_mat)
        self.lens = len(self.file_mat['X'])

    def __getitem__(self, index):
        feat = self.file_mat['X'][index][0].squeeze().astype(np.float32)
        return feat

    def __len__(self):
        return self.lens


def best_map(L1, L2):
    # L1 should be the ground-truth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = (L1 == Label1[i]).astype(float)
        for j in range(nClass2):
            ind_cla2 = (L2 == Label2[j]).astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def get_ar(y_true, y_pred):
    return metrics.adjusted_rand_score(y_true, y_pred)


def get_nmi(y_true, y_pred):
    return metrics.normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')


def get_fpr(y_true, y_pred):
    n_samples = np.shape(y_true)[0]
    c = contingency_matrix(y_true, y_pred, sparse=True)
    tk = np.dot(c.data, np.transpose(c.data)) - n_samples  # TP
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples  # TP+FP
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples  # TP+FN
    precision = 1. * tk / pk if tk != 0. else 0.
    recall = 1. * tk / qk if tk != 0. else 0.
    f = 2 * precision * recall / (precision + recall) if (precision +
                                                          recall) != 0. else 0.
    return f, precision, recall


def get_purity(y_true, y_pred):
    c = metrics.confusion_matrix(y_true, y_pred)
    return 1. * c.max(axis=0).sum() / np.shape(y_true)[0]


def calculate_metrics(y, y_pred):
    y_new = best_map(y, y_pred)
    acc = metrics.accuracy_score(y, y_new)
    ar = get_ar(y, y_pred)
    nmi = get_nmi(y, y_pred)
    f, p, r = get_fpr(y, y_pred)
    purity = get_purity(y, y_pred)
    return acc, ar, nmi, f, p, r, purity


def check_dir_exist(dir_):
    if not os.path.isdir(dir_):
        os.mkdir(dir_)


def check_invalid_vals(container, infos=None, logger=None, ignore_error=False, return_value=False, callback_fn=None):
    check_items = None
    if isinstance(container, (list, tuple)):
        check_items = enumerate(container)
    elif isinstance(container, dict):
        check_items = container.items()
    elif isinstance(container, torch.Tensor):
        check_items = enumerate([container])
    else:
        check_items = enumerate([])

    invalid_exist = False
    output_func = print if logger is None else logger.info
    for k, v in check_items:
        if isinstance(v, torch.Tensor):
            v = v.float()
            if torch.isnan(v).any() or torch.isinf(v).any():
                output_func(f'Item {k} in ({infos}) with shape of {v.shape} contains invalid value (Nan: {torch.isnan(v).any()}, Inf: {torch.isinf(v).any()}):\n{v}')
                invalid_exist = True

    if invalid_exist and callback_fn is not None:
        callback_fn()

    if invalid_exist and not ignore_error:
        raise RuntimeError('Invalid value found! Check error information above.')

    if return_value:
        return invalid_exist


def check_gradient_vals(module: nn.Module, step: int, logger=None, module_name=''):
    output_func = print if logger is None else logger.info

    for m_name, m in module.named_children():
        for name, param in m.named_parameters():
            output_func(f'Step {step} {module._get_name() if not module_name else module_name}.{m_name}.{name}->\nParam:\n{param}\nGrad:\n{param.grad}')
        check_gradient_vals(m, step, logger, f'{module_name}.{m_name}')


def check_parameter_vals(module: nn.Module, step: int, logger=None, module_name='', callback_fn=None):
    for m_name, m in module.named_children():
        for name, param in m.named_parameters():
            para_name = f'{module._get_name() if not module_name else module_name}.{m_name}.{name}'
            check_invalid_vals(
                [param, param.grad],
                [f'{para_name}::param', f'{para_name}::grad'],
                logger=logger,
                ignore_error=True,
                callback_fn=callback_fn
            )
        check_parameter_vals(m, step, logger, f'{module_name}.{m_name}', callback_fn)


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if not hasattr(param.grad, 'data'):
                continue
            param.grad.data.clamp_(-grad_clip, grad_clip)

def balanced_sampling(x_true, threshold: float = 1e-4):
    pos_mask = x_true > threshold
    pos_nums = pos_mask.float().sum()

    # check whether the threshold setting is failed
    if pos_nums == x_true.numel():
        raise Exception(
            f"With current threshold ({threshold}), all samples are positive."
            f"The maximum value in the sample is {x_true.max()}."
        )

    # check whether the threshold setting contains imbalance problem
    pos_prob = pos_nums / x_true.numel()
    if pos_prob > 0.5:
        raise Warning(
            f"With current threshold ({threshold}), there are more than"
            f"half of samples are masked as positives."
        )

    neg_prob = pos_nums / (x_true.numel() - pos_nums)
    binomial = torch.distributions.Binomial(1, torch.ones_like(x_true) * neg_prob)
    neg_sample_mask = binomial.sample() == 1
    neg_mask = torch.logical_and(torch.logical_not(pos_mask), neg_sample_mask)
    return pos_mask, neg_mask


def focal_bce_with_logits(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    threshold: float,
    gamma: float = 0.
):
    """compute bce loss with logits (ref: https://paperswithcode.com/method/focal-loss#:~:text=Focal%20loss%20applies%20a%20modulating,in%20the%20correct%20class%20increases.)

    :param x_pred: prediction logits
    :param x_true: binary ground truth
    :param threshold: threshold for determining postive class
    :param gamma: focal loss gamma, defaults to 0.
    :return: focal loss
    """
    x_pred_dist = torch.sigmoid(x_pred)
    pos_cls_mask = x_true > threshold
    neg_cls_mask = torch.logical_not(pos_cls_mask)

    bce_loss = F.binary_cross_entropy_with_logits(x_pred, x_true, reduction="none")

    bce_loss = torch.pow(1 - x_pred_dist, gamma) * bce_loss * pos_cls_mask.float()\
        + torch.pow(x_pred_dist, gamma) * bce_loss * neg_cls_mask.float()
    return bce_loss

def masked_bce(x_pred, x_true):
    """Masked BCE Loss
    we only compute the balance loss from the postive value and the negative value.
    """
    pos_mask, neg_mask = balanced_sampling(x_true, 0.5)

    bce_loss = focal_bce_with_logits(
        x_pred,
        x_true,
        threshold=0.5,
        gamma=1
    )
    pos_loss = (bce_loss * pos_mask.float()).sum()
    neg_loss = (bce_loss * neg_mask.float()).sum()

    """
    Note:

    The correct loss considering both the positive part and the negative part should be
        loss = (pos_loss + neg_loss) / (pos_mask.float().sum() + neg_mask.float().sum())
    where positive samples and negative samples are weighted the same for training.
    However, in practice, it doesn't work. The inituition reason is negative samples are
    much more easier to predict than positive samples. Therefore, we need to give
    different weights for positive samples and negative samples.

    We can compute the weighted loss as:
        loss = (pos_loss + neg_loss) / (pos_mask.float().sum() + neg_loss.float().sum())
    or
        use the focal bce loss?
    """
    loss = (pos_loss + neg_loss) / (pos_mask.float().sum() + neg_mask.float().sum())
    return loss


def centroids_penalty(embed: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    embed = embed.unsqueeze(1)  # [b, 1, d]
    centrods = centroids.unsqueeze(0)  # [1, c, d]

    dist = torch.abs(embed - centrods).mean(-1)  # [b, c]
    min_dist = torch.min(dist, dim=-1)[0]  # get minimal values
    loss = min_dist.mean()
    return loss


def apprx_kl(mu, sigma):
    '''Adapted from https://github.com/dcmoyer/invariance-tutorial/
    Function to calculate approximation for KL(q(z|x)|q(z))
        Args:
            mu: Tensor, (B, z_dim)
            sigma: Tensor, (B, z_dim)
    '''
    var = sigma.pow(2)
    var_inv = var.reciprocal()

    first = torch.matmul(var, var_inv.T)

    r = torch.matmul(mu * mu, var_inv.T)
    r2 = (mu * mu * var_inv).sum(axis=1)

    second = 2 * torch.matmul(mu, (mu * var_inv).T)
    second = r - second + (r2 * torch.ones_like(r)).T

    r3 = var.log().sum(axis=1)
    third = (r3 * torch.ones_like(r)).T - r3

    return 0.5 * (first + second + third)


class StableExpOp(Function):
    max_value = 1e6
    min_value = 1e-5
    max_threshold = math.log(max_value)
    min_threshold = math.log(min_value)

    @staticmethod
    def forward(ctx, x):
        exp_val = torch.exp(x)
        ctx.save_for_backward(x, exp_val)
        appr_exp_val = torch.where(
            x < StableExpOp.max_threshold,
            exp_val,
            StableExpOp.max_value + (x - StableExpOp.max_threshold) * StableExpOp.max_value
        )
        appr_exp_val = torch.where(
            x > StableExpOp.min_threshold,
            appr_exp_val,
            StableExpOp.min_value
        )
        return appr_exp_val

    @staticmethod
    def backward(ctx, grad_output):
        x, exp_val = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.where(
                x < StableExpOp.max_threshold,
                exp_val * grad_output,
                StableExpOp.max_value * grad_output
            )
            grad_input = torch.where(
                x > StableExpOp.min_threshold,
                grad_input,
                StableExpOp.min_value * grad_output
            )
        return grad_input


stable_exp = StableExpOp.apply


def nb_loss(preds, theta, truth, tb_step=None):
    """Calculates negative binomial loss as defined in the NB class in link above"""
    scale_factor = 1.0
    eps = 1e-6
    mean = True
    debug = False

    y_true = truth
    y_pred = preds * scale_factor

    if debug:  # Sanity check before loss calculation
        assert not torch.isnan(y_pred).any(), y_pred
        assert not torch.isinf(y_pred).any(), y_pred
        assert not (y_pred < 0).any()  # should be non-negative
        assert not (theta < 0).any()

    # Clip theta values
    theta = torch.clamp(theta, max=1e6)

    t1 = (
        torch.lgamma(theta + eps)
        + torch.lgamma(y_true + 1.0)
        - torch.lgamma(y_true + theta + eps)
    )
    t2 = (theta + y_true) * torch.log1p(y_pred / (theta + eps)) + (
        y_true * (torch.log(theta + eps) - torch.log(y_pred + eps))
    )
    if debug:  # Sanity check after calculating loss
        assert not torch.isnan(t1).any(), t1
        assert not torch.isinf(t1).any(), (t1, torch.sum(torch.isinf(t1)))
        assert not torch.isnan(t2).any(), t2
        assert not torch.isinf(t2).any(), t2

    retval = t1 + t2
    if debug:
        assert not torch.isnan(retval).any(), retval
        assert not torch.isinf(retval).any(), retval

    # if tb is not None and tb_step is not None:
    #    tb.add_histogram("nb/t1", t1, global_step=tb_step)
    #    tb.add_histogram("nb/t2", t2, global_step=tb_step)

    return torch.mean(retval) if mean else retval


def total_variation(x):
    """
    Given a 2D input (where one dimension is a batch dimension, the actual values are
    one dimensional) compute the total variation (within a 1 position shift)
    """
    t = torch.sum(torch.abs(x[:, :-1] - x[:, 1:]))
    return t


def zinb_loss(preds, theta_disp, pi_dropout, truth, tb_step=None):
    ridge_lambda = 10
    tv_lambda = 0.1
    eps = 1e-6
    scale_factor = 1.0
    debug = False

    if debug:
        assert not (pi_dropout > 1.0).any()
        assert not (pi_dropout < 0.0).any()
    nb_case = nb_loss(preds, theta_disp, truth, tb_step=tb_step) - torch.log(
        1.0 - pi_dropout + eps
    )

    y_true = truth
    y_pred = preds * scale_factor
    theta = torch.clamp(theta_disp, min=0, max=1e6)

    zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
    zero_case = -torch.log(pi_dropout + ((1.0 - pi_dropout) * zero_nb) + eps)
    result = torch.where(y_true < 1e-8, zero_case, nb_case)

    print('--')
    print(torch.log(1.0 - pi_dropout + eps))
    print(nb_case.mean())
    print(zero_case.mean())
    print(result.mean())

    # Ridge regularization on pi dropout term
    ridge = ridge_lambda * torch.pow(pi_dropout, 2)
    result += ridge

    print(ridge.mean())

    # Total variation regularization on pi dropout term
    tv = tv_lambda * total_variation(pi_dropout)
    result += tv

    print(tv.mean())

    # if tb is not None and tb_step is not None:
    #    tb.add_histogram("zinb/nb_case", nb_case, global_step=tb_step)
    #    tb.add_histogram("zinb/zero_nb", zero_nb, global_step=tb_step)
    #    tb.add_histogram("zinb/zero_case", zero_case, global_step=tb_step)
    #    tb.add_histogram("zinb/ridge", ridge, global_step=tb_step)
    #    tb.add_histogram("zinb/zinb_loss", result, global_step=tb_step)

    retval = torch.mean(result)
    # if debug:
    #     assert retval.item() > 0
    return retval


def scvi_log_nb_positive(x, mu, theta, eps=1e-8):
    """
    Taken from scVI log_likelihood.py - scVI invocation is:
    reconst_loss = -log_nb_positive(x, px_rate, px_r).sum(dim=-1)
    scVI decoder outputs px_scale, px_r, px_rate, px_dropout
    px_scale is subject to Softmax
    px_r is just a Linear layer
    px_rate = torch.exp(library) * px_scale

    mu = mean of NB
    theta = indverse dispersion parameter

    Here, x appears to correspond to y_true in the below negative_binom_loss (aka the observed counts)
    """
    # if theta.ndimension() == 1:
    #     theta = theta.view(
    #         1, theta.size(0)
    #     )  # In this case, we reshape theta for broadcasting

    log_theta_mu_eps = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)  # Present (in negative) for DCA
        - torch.lgamma(x + 1)
    )

    return res.mean()


def log_zinb_positive(x, mu, theta, pi, eps=1E-6):
    """
    Note: All inputs are torch Tensors

    log likelihood (scalar) of a minibatch according to a zinb model.

    Notes:
    We parametrize the bernoulli using the logits, hence the softplus functions appearing
    Variables:
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    pi: logit of the dropout parameter (real support) (shape: minibatch x genes)
    eps: numerical stability constant
    """
    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    softplus_pi = F.softplus(-pi)
    theta = torch.clamp(theta, max=1e5)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    # first part with zero cases
    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    # second part with non-zero cases
    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta + eps)
        - torch.lgamma(theta + eps)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    # total loss
    res = mul_case_zero + mul_case_non_zero
    return res


def negative_binom_loss(
        scale_factor=1.0,
        eps=1e-10,
        mean=True,
        debug=False,
        tb=None):
    """
    Return a function that calculates the binomial loss
    https://github.com/theislab/dca/blob/master/dca/loss.py

    combination of the Poisson distribution and a gamma distribution is a negative binomial distribution
    """

    def loss(preds, theta, truth, tb_step: int = None):
        """Calculates negative binomial loss as defined in the NB class in link above"""
        y_true = truth
        y_pred = preds * scale_factor

        if debug:  # Sanity check before loss calculation
            assert not torch.isnan(y_pred).any(), y_pred
            assert not torch.isinf(y_pred).any(), y_pred
            assert not (y_pred < 0).any()  # should be non-negative
            assert not (theta < 0).any()

        # Clip theta values
        theta = torch.clamp(theta, max=1e6)

        t1 = (
            torch.lgamma(theta + eps)
            + torch.lgamma(y_true + 1.0)
            - torch.lgamma(y_true + theta + eps)
        )
        t2 = (theta + y_true) * torch.log1p(y_pred / (theta + eps)) + (
            y_true * (torch.log(theta + eps) - torch.log(y_pred + eps))
        )
        if debug:  # Sanity check after calculating loss
            assert not torch.isnan(t1).any(), t1
            assert not torch.isinf(t1).any(), (t1, torch.sum(torch.isinf(t1)))
            assert not torch.isnan(t2).any(), t2
            assert not torch.isinf(t2).any(), t2

        retval = t1 + t2

        return torch.mean(retval) if mean else retval

    return loss


def regularized_bce(x_pred, x_true, omega=0.05, logger=None):
    res = nn.BCELoss(weight=(x_pred <= 0.5) * x_true / omega + 1)
    # res = nn.BCELoss()
    return res(x_pred, x_true)


def log_nb_positive(
    x,
    mu,
    theta,
    eps=1e-8,
    log_fn=torch.log,
    lgamma_fn=torch.lgamma,
):
    """
    Log likelihood (scalar) of a minibatch according to a nb model.
    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    return res


def plot_single_embedding_umap(embedding_name: str, embedding: np.ndarray, labels: list, umap_config: dict):
    reducer = umap.UMAP(**umap_config)
    umap_embedded = reducer.fit_transform(embedding)

    # prepare label array
    label_encoder = LabelEncoder()
    label_array = label_encoder.fit_transform(labels)

    # compute score
    silhouette = silhouette_score(umap_embedded, label_array)

    # plot umap
    fig, axes = plt.subplots()
    for i, c in enumerate(np.unique(labels)):
        mask = label_array == label_encoder.transform([c])[0]
        feature = umap_embedded[mask]
        axes.scatter(
            feature[:, 0], feature[:, 1], label=c, s=0.7, marker='.'
        )
    axes.legend(markerscale=4, fontsize=6)
    axes.set_title(
        f'UMAP of {embedding_name.upper()} Embedding (silhouette={silhouette:.3f})'
    )
    output = {'plot': (fig, axes), 'umap_embedding': umap_embedded, 'silhouette_score': silhouette}
    return output


def plot_joint_embedding_umap(embedding_name: str, embedding1: np.ndarray, embedding2: np.ndarray, labels: list, umap_config: dict):
    embedding = np.concatenate((embedding1, embedding2), axis=0)
    reducer = umap.UMAP(**umap_config)
    umap_embedded = reducer.fit_transform(embedding)
    umap_embedded1, umap_embedded2 = np.split(umap_embedded, 2)

    # prepare label array
    label_encoder = LabelEncoder()
    label_array = label_encoder.fit_transform(labels)

    # compute score
    silhouette = silhouette_score(umap_embedded, np.tile(label_array, 2))

    # color
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

    # plot umap
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, c in enumerate(np.unique(labels)):
        mask = label_array == label_encoder.transform([c])[0]
        color = next(colors)

        feature1 = umap_embedded1[mask]
        axes[0].scatter(
            feature1[:, 0], feature1[:, 1], label=c, s=0.7, marker='.', c=color, alpha=0.8
        )
        axes[1].scatter(
            feature1[:, 0], feature1[:, 1], label=c, s=0.7, marker='.', c=color, alpha=0.8
        )

        feature2 = umap_embedded2[mask]
        axes[0].scatter(
            feature2[:, 0], feature2[:, 1], label=f'{c} Cross', s=0.5, marker='v', c=color, alpha=0.8
        )
        axes[2].scatter(
            feature2[:, 0], feature2[:, 1], label=f'{c} Cross', s=0.5, marker='v', c=color, alpha=0.8
        )
    axes[0].legend(markerscale=4, fontsize=6)
    axes[0].set_title(
        f'UMAP of {embedding_name.upper()} Embedding (silhouette={silhouette:.3f})'
    )
    output = {'plot': (fig, axes), 'umap_embedding': umap_embedded, 'silhouette_score': silhouette}
    return output
