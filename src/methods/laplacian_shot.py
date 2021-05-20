import torch.nn.functional as F
import argparse
import torch
from torch import tensor
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy import sparse
import math
from numpy import linalg as LA
from typing import Dict, Tuple
from numpy import ndarray

from .method import FSmethod
from .utils import extract_features, compute_centroids
from ..metrics import Metric


class LaplacianShot(FSmethod):
    """
    Implementation of LaplacianShot (ICML 2020) https://arxiv.org/abs/2006.15486
    """
    def __init__(self,
                 args: argparse.Namespace):
        self.knn = args.knn
        self.lmd = args.lmd
        self.iter = args.iter
        self.episodic_training = False
        self.proto_rect = args.proto_rect
        self.batch = False
        self.normalize_features = args.normalize_features
        self.extract_batch_size = args.extract_batch_size
        self.center = args.center_features
        self.merge_support = args.merge_support
        if args.val_batch_size > 1:
            raise ValueError("For now, only val_batch_size=1 is support for LaplacianShot")
        super().__init__(args)

    def record_info(self,
                    metrics: dict,
                    task_ids: tuple,
                    iteration: int,
                    new_time: float,
                    Y: tensor,
                    kernel: tensor,
                    unary: tensor,
                    y_q: tensor) -> None:
        """
        inputs:
            support : tensor of shape [n_task, s_shot, feature_dim]
            query : tensor of shape [n_task, q_shot, feature_dim]
            y_s : tensor of shape [n_task, s_shot]
            y_q : tensor of shape [n_task, q_shot] :
        """
        if metrics:
            kwargs = {}
            kwargs['Y'] = Y
            kwargs['lmd'] = self.lmd
            kwargs['preds'] = np.argmax(Y, axis=1)
            kwargs['gt'] = y_q
            kwargs['kernel'] = kernel
            kwargs['unary'] = unary

            for metric_name in metrics:
                metrics[metric_name].update(task_ids[0],
                                            task_ids[1],
                                            iteration,
                                            **kwargs)

    def init_prototypes(self,
                        support: ndarray,
                        query: ndarray,
                        y_s: ndarray,
                        y_q: ndarray) -> ndarray:
        """
        inputs:
            support : tensor of shape [s_shot, feature_dim]
            query : tensor of shape [q_shot, feature_dim]
            y_s : tensor of shape [s_shot]
            y_q : tensor of shape [q_shot]

        outputs :
            centroids : tensor of shape [num_class, feature_dim]
        """
        # num_classes = np.unique(y_s).shape[0]
        # one_hot = np.eye(num_classes)[y_s]  # [shot, num_classes]
        # counts = np.expand_dims(one_hot.sum(0), 1)  # [num_classes, 1]
        # weights = one_hot.T.dot(support)
        # centroids = weights / counts

        centroids = compute_centroids(tensor(support), tensor(y_s)).numpy()  # [batch, num_class, d]
        if self.proto_rect:
            eta = support.mean(0) - query.mean(0) # shift
            query = query + eta[np.newaxis, :]
            query_aug = np.concatenate((support, query), axis=0)
            support_ = torch.from_numpy(centroids)
            query_aug = torch.from_numpy(query_aug)
            distance = get_metric('cosine')(support_, query_aug)
            predict = torch.argmin(distance, dim=1)
            cos_sim = F.cosine_similarity(query_aug[:, None, :], support_[None, :, :], dim=2)
            cos_sim = 10 * cos_sim
            W = F.softmax(cos_sim, dim=1)
            support_list = [(W[predict == i, i].unsqueeze(1) * query_aug[predict == i]).mean(0, keepdim=True) for i in predict.unique()]
            centroids = torch.cat(support_list, dim=0).numpy()
        return centroids

    def create_affinity(self, X: ndarray) -> ndarray:
        """
        inputs:
            X : ndarray of shape [shot, feature_dim]

        """
        N, D = X.shape
        # print('Compute Affinity ')
        nbrs = NearestNeighbors(n_neighbors=self.knn).fit(X)
        dist, knnind = nbrs.kneighbors(X)

        row = np.repeat(range(N), self.knn - 1)
        col = knnind[:, 1:].flatten()
        data = np.ones(X.shape[0] * (self.knn - 1))
        W = sparse.csc_matrix((data, (row, col)), shape=(N, N), dtype=np.float)
        return W

    def l2_distance(self, samples_1: ndarray, samples_2: ndarray) -> ndarray:
        """
        inputs:
            samples_1 : tensor of shape [shot_1, feature_dim]
            samples_2 : tensor of shape [shot_2, feature_dim]

        """
        l2_dist = (-2 * samples_1.dot(samples_2.T) \
                   + (samples_1**2).sum(1).reshape((-1, 1)) \
                   + (samples_2**2).sum(1).reshape((1, -1)))  #
        return l2_dist

    def normalize(self, Y_in: ndarray) -> ndarray:
        """
        Inputs:
            - Y_in: [shot, num_classes]

        Outputs:
            - Y_out: [shot, num_classes]
        """
        maxcol = np.max(Y_in, axis=1)
        Y_in = Y_in - maxcol[:, np.newaxis]
        N = Y_in.shape[0]
        size_limit = 150000
        if N > size_limit:
            batch_size = 1280
            Y_out = []
            num_batch = int(math.ceil(1.0 * N / batch_size))
            for batch_idx in range(num_batch):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, N)
                tmp = np.exp(Y_in[start:end, :])
                tmp = tmp / (np.sum(tmp, axis=1)[:, None])
                Y_out.append(tmp)
            del Y_in
            Y_out = np.vstack(Y_out)
        else:
            Y_out = np.exp(Y_in)
            Y_out = Y_out / (np.sum(Y_out, axis=1)[:, None])

        return Y_out

    def entropy_energy(self, Y: ndarray, unary: ndarray, kernel: ndarray) -> ndarray:
        """
        Inputs:
            - Y: [shot, num_classes]
            - unary: [shot, num_classes]
            - kernel: [shot, shot]

        Outputs:
            - E: []
        """
        tot_size = Y.shape[0]
        pairwise = kernel.dot(Y)
        if self.batch == False:
            temp = (unary * Y) + (-self.lmd * pairwise * Y)
            E = (Y * np.log(np.maximum(Y, 1e-20)) + temp).sum()
        else:
            batch_size = 1024
            num_batch = int(math.ceil(1.0 * tot_size / batch_size))
            E = 0
            for batch_idx in range(num_batch):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, tot_size)
                temp = (unary[start:end] * Y[start:end]) + (-self.lmd * pairwise[start:end] * Y[start:end])
                E = E + (Y[start:end] * np.log(Y[start:end] + 1e-20) + temp).sum()

        return E

    # def bound_update(self, unary, kernel, y_q, , batch=False):
    #     """
    #     """

    def forward(self,
                model: torch.nn.Module,
                support: tensor,
                query: tensor,
                y_s: tensor,
                y_q: tensor,
                metrics: Dict[str, Metric] = None,
                task_ids: Tuple[int, int] = None)-> Tuple[tensor, tensor]:
        """
        """

        model.eval()
        with torch.no_grad():
            feat_s, feat_q = extract_features(self.extract_batch_size,
                                              support,
                                              query,
                                              model)
        feat_s, feat_q = feat_s[0].cpu().numpy(), feat_q[0].cpu().numpy()
        y_s, y_q = y_s[0].cpu().numpy(), y_q[0].cpu().numpy()
        if self.knn == 'auto':
            self.knn = int(y_q.shape[0] / (np.unique(y_s).shape[0] * 5))
        # Perform required normalizations
        if self.center:
            feat_s -= self.train_mean.numpy()
            feat_q -= self.train_mean.numpy()

        if self.normalize_features:
            feat_s /= LA.norm(feat_s, 2, 1)[:, None]
            feat_q /= LA.norm(feat_q, 2, 1)[:, None]

        n_query = feat_q.shape[0]
        n_support = feat_s.shape[0]
        num_class = np.unique(y_s).shape[0]
        # Initialize weights
        centroids = self.init_prototypes(feat_s, feat_q, y_s, y_q)
        unary = self.l2_distance(feat_q, centroids)  # [shot, num_class]
        if self.merge_support:
            kernel = self.create_affinity(np.concatenate([feat_s, feat_q], 0))  # [shot, shot]
            unary = np.concatenate([np.zeros((n_support, num_class)), unary], 0)
        else:
            kernel = self.create_affinity(feat_q)  # [shot, shot]

        oldE = float('inf')
        Y = self.normalize(-unary)  # [shot, num_classes]
        if self.merge_support:
            Y[:n_support] = np.eye(num_class)[y_s]
        self.record_info(iteration=0,
                         task_ids=task_ids,
                         metrics=metrics,
                         new_time=0,
                         Y=Y[-n_query:],
                         kernel=kernel[-n_query:, -n_query:],
                         unary=unary[-n_query:],
                         y_q=y_q)
        keep_updating = True
        for i in range(1, self.iter):
            if keep_updating:
                X = -unary + self.lmd * kernel.dot(Y)  # [shot, num_classes]
                Y = self.normalize(X)
                if self.merge_support:
                    Y[:n_support] = np.eye(num_class)[y_s]
                E = self.entropy_energy(Y, unary, kernel)  # []
            self.record_info(iteration=i,
                             task_ids=task_ids,
                             metrics=metrics,
                             new_time=0,
                             Y=Y[-n_query:],
                             kernel=kernel[-n_query:, -n_query:],
                             unary=unary[-n_query:],
                             y_q=y_q)
            # print('entropy_energy is ' +repr(E) + ' at iteration ',i)
            if (i > 1 and (abs(E - oldE) <= 1e-6 * abs(oldE))):
                # print('Converged')
                keep_updating = False

            else:
                oldE = E.copy()
        soft_preds_q = tensor(Y[-n_query:]).unsqueeze(0)
        return None, soft_preds_q


def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]
