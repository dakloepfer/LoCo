import os
from typing import List, Union

import einops as ein
import fast_pytorch_kmeans as fpk
import numpy as np
import torch
from einops import rearrange, reduce
from torch.nn import functional as F


# VLAD global descriptor implementation
class VLAD:
    """
    An implementation of VLAD algorithm given database and query
    descriptors.

    Constructor arguments:
    - num_clusters:     Number of cluster centers for VLAD
    - desc_dim:         Descriptor dimension. If None, then it is
                        inferred when running `fit` method.
    - intra_norm:       If True, intra normalization is applied
                        when constructing VLAD
    - norm_descs:       If True, the given descriptors are
                        normalized before training and predicting
                        VLAD descriptors. Different from the
                        `intra_norm` argument.
    - dist_mode:        Distance mode for KMeans clustering for
                        vocabulary (not residuals). Must be in
                        {'euclidean', 'cosine'}.
    - vlad_mode:        Mode for descriptor assignment (to cluster
                        centers) in VLAD generation. Must be in
                        {'soft', 'hard'}
    - soft_temp:        Temperature for softmax (if 'vald_mode' is
                        'soft') for assignment
    - cache_dir:        Directory to cache the VLAD vectors. If
                        None, then no caching is done. If a str,
                        then it is assumed as the folder path. Use
                        absolute paths.

    Notes:
    - Arandjelovic, Relja, and Andrew Zisserman. "All about VLAD."
        Proceedings of the IEEE conference on Computer Vision and
        Pattern Recognition. 2013.
    """

    def __init__(
        self,
        num_clusters: int,
        desc_dim: Union[int, None] = None,
        intra_norm: bool = True,
        norm_descs: bool = True,
        distance_metric: str = "cosine",
        vlad_mode: str = "hard",
        soft_temp: float = 1.0,
        cache_dir: Union[str, None] = None,
    ) -> None:

        self.num_clusters = num_clusters
        self.desc_dim = desc_dim
        self.intra_norm = intra_norm
        self.norm_descs = norm_descs
        self.mode = distance_metric
        self.vlad_mode = str(vlad_mode).lower()
        assert self.vlad_mode in ["soft", "hard"]
        self.soft_temp = soft_temp

        # Set in the training phase
        self.c_centers = None
        self.kmeans = None

        # Set the caching
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir = os.path.abspath(os.path.expanduser(self.cache_dir))
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
                print(f"Created cache directory: {self.cache_dir}")
            else:
                print("Warning: Cache directory already exists: " f"{self.cache_dir}")
        else:
            print("VLAD caching is disabled.")

    def can_use_cache_vlad(self):
        """
        Checks if the cache directory is a valid cache directory.
        For it to be valid, it must exist and should at least
        include the cluster centers file.

        Returns:
        - True if the cache directory is valid
        - False if
            - the cache directory doesn't exist
            - exists but doesn't contain the cluster centers
            - no caching is set in constructor
        """
        if self.cache_dir is None:
            return False
        if not os.path.exists(self.cache_dir):
            return False
        if os.path.exists(f"{self.cache_dir}/cluster_centers.pt"):
            return True
        else:
            return False

    # Generate cluster centers
    def fit(self, train_descs):
        """Using the training descriptors calculated using the feature_extractor and the vocab_imgs, generate the cluster
        centers (vocabulary). Function expects all descriptors in
        a single list (see `fit_and_generate` for a batch of
        images).
        If the cache directory is valid, then retrieves cluster
        centers from there (the `train_descs` are ignored).
        Otherwise, stores the cluster centers in the cache
        directory (if using caching).

            Parameters
            ----------
            train_descs (b x q x d torch.Tensor):
                a batch of q descriptors of dimension d used for training the vocabulary.

            vocab_imgs (iterable):
                an iterable that returns the images used to generate the vocabulary.
        """
        # Clustering to create vocabulary
        self.kmeans = fpk.KMeans(self.num_clusters, mode=self.mode)

        # Check if cache exists
        if self.can_use_cache_vlad():
            print("Using cached cluster centers")
            self.c_centers = torch.load(f"{self.cache_dir}/cluster_centers.pt")
            self.kmeans.centroids = self.c_centers
            if self.desc_dim is None:
                self.desc_dim = self.c_centers.shape[1]
                print(f"Desc dim set to {self.desc_dim}")
        else:
            train_descs = rearrange(train_descs, "b q d -> (b q) d")

            if self.desc_dim is None:
                self.desc_dim = train_descs.shape[1]
            if self.norm_descs:
                train_descs = F.normalize(train_descs, dim=-1)
            self.kmeans.fit(train_descs)
            self.c_centers = self.kmeans.centroids
            if self.cache_dir is not None:
                print("Caching cluster centers")
                torch.save(self.c_centers, f"{self.cache_dir}/cluster_centers.pt")

    def generate(
        self, descriptors: Union[np.ndarray, torch.Tensor], cache_name=None
    ) -> torch.Tensor:
        """Given a batch of sets of descriptors, generate a VLAD vector. Call
        `fit` before using this method.

            Parameters
            ----------
            descriptors (batch_size x n_descriptors x desc_dim Union[np.ndarray, torch.Tensor]):
                a batch of descriptors which are turned into VLAD descriptors

            Returns
            -------
            vlads (batch_size x (n_clusters * desc_dim) torch.Tensor):
                the corresponding VLAD descriptors.
        """
        if cache_name is not None:
            cache_path = os.path.join(self.cache_dir, cache_name)
            if os.path.exists(cache_path):
                print(f"Using cached VLAD vectors from {cache_path}...")
                return torch.load(cache_path)

        if len(descriptors.shape) == 2:
            return self.generate_single(descriptors)

        res = [self.generate_single(desc) for desc in descriptors]
        try:  # most likely pytorch
            res = torch.stack(res)
        except TypeError:
            try:
                res = np.stack(res)
            except TypeError:  # leave it as a list
                res = res

        if cache_name is not None:
            print(f"Caching VLAD vectors to {cache_path}...")
            torch.save(res, cache_path)

        return res

    def generate_single(
        self, descriptors: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Given a set of descriptors, generate a VLAD vector. Call
        `fit` before using this method. Uses a serialised method that uses less memory.

            Parameters
            ----------
            descriptors (n_descriptors x desc_dim Union[np.ndarray, torch.Tensor]):
                a set of descriptors which are turned into a VLAD vector.

            Returns
            -------
            vlads (n_clusters * desc_dim torch.Tensor):
                the corresponding VLAD descriptors.
        """
        assert self.kmeans is not None
        assert self.c_centers is not None

        if type(descriptors) == np.ndarray:
            descriptors = torch.from_numpy(descriptors).to(torch.float)
        if self.norm_descs:
            descriptors = F.normalize(descriptors, dim=-1)

        raw_vlad = torch.zeros(self.num_clusters * self.desc_dim)
        if self.vlad_mode == "hard":
            # get labels for assignment of descriptors
            labels = self.kmeans.predict(descriptors)  # q

            residuals = descriptors - self.c_centers[labels]  # q d
            used_clusters = set(labels.numpy())
            for k in used_clusters:
                # Sum of residuals for the descriptors in the cluster
                cd_sum = reduce((labels == k)[:, None] * residuals, "q d -> d", "sum")
                if self.intra_norm:
                    cd_sum = F.normalize(cd_sum, dim=0)
                raw_vlad[k * self.desc_dim : (k + 1) * self.desc_dim] = cd_sum

        else:  # Soft cluster assignment
            residuals = descriptors[:, None, :] - self.c_centers[None, :, :]  # q c d
            # Cosine similarity: 1 = close, -1 = away
            cos_sims = F.cosine_similarity(  # [q, c]
                descriptors[:, None, :],
                self.c_centers[None, :, :],
                dim=2,
            )
            assignment = F.softmax(self.soft_temp * cos_sims, dim=1)  # [q, c]

            for k in range(self.num_clusters):

                cd_sum = reduce(
                    assignment[:, k, None, None] * residuals, "q c d -> d", "sum"
                )
                if self.intra_norm:
                    cd_sum = F.normalize(cd_sum, dim=0)
                raw_vlad[k * self.desc_dim : (k + 1) * self.desc_dim] = cd_sum

        # Normalize the VLAD vector
        vlads = F.normalize(raw_vlad, dim=-1)
        return vlads

    def generate_vectorised(
        self, descriptors: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Given a batch of sets of descriptors, generate a VLAD vector. Call
        `fit` before using this method.

            Parameters
            ----------
            descriptors (batch_size x n_descriptors x desc_dim Union[np.ndarray, torch.Tensor]):
                a batch of descriptors which are turned into VLAD descriptors

            Returns
            -------
            vlads (batch_size x (n_clusters * desc_dim) torch.Tensor):
                the corresponding VLAD descriptors.
        """
        assert self.kmeans is not None
        assert self.c_centers is not None

        if type(descriptors) == np.ndarray:
            descriptors = torch.from_numpy(descriptors).to(torch.float)
        if self.norm_descs:
            descriptors = F.normalize(descriptors, dim=-1)

        if len(descriptors.shape) == 2:
            descriptors = descriptors.unsqueeze(0)
        batch_size = descriptors.shape[0]

        residuals = descriptors[:, :, None, :] - self.c_centers[None, None, :, :]

        if self.vlad_mode == "hard":
            # Get labels for assignment of descriptors

            labels = rearrange(
                self.kmeans.predict(rearrange(descriptors, "b q d -> (b q) d")),
                "(b q) -> b q",
                b=batch_size,
            )
            assignment = F.one_hot(labels, num_classes=self.num_clusters)

        else:  # Soft cluster assignment
            # Cosine similarity: 1 = close, -1 = away
            cos_sims = F.cosine_similarity(  # [b, q, c]
                descriptors[:, :, None, :],
                self.c_centers[None, None, :, :],
                dim=3,
            )
            assignment = F.softmax(self.soft_temp * cos_sims, dim=2)  # [b, q, c]

        cd_sum = torch.sum(
            assignment[:, :, None, :, None] * residuals[:, :, :, None, :],
            dim=(1, 2),
        )  # b c d
        if self.intra_norm:
            cd_sum = F.normalize(cd_sum, dim=-1)

        # Normalize the VLAD vector
        vlads = F.normalize(rearrange(cd_sum, "b c d -> b (c d)"), dim=-1)
        return vlads

    def generate_patch_descriptors(self, descriptors, cache_name=None):
        """Given a batch of sets of descriptors, get the VLAD versions of each patch descriptor. Call
        `fit` before using this method.

            Parameters
            ----------
            descriptors (batch_size x ... x desc_dim Union[np.ndarray, torch.Tensor]):
                a batch of descriptors which are turned into VLAD descriptors

            Returns
            -------
            vlads (batch_size x (n_clusters * desc_dim) torch.Tensor):
                the corresponding VLAD descriptors.
        """
        if cache_name is not None:
            cache_path = os.path.join(self.cache_dir, cache_name)
            if os.path.exists(cache_path):
                print(f"Using cached VLAD vectors from {cache_path}...")
                return torch.load(cache_path)

        if len(descriptors.shape) == 2:
            return self.generate_single_patch_descs(descriptors)

        res = [self.generate_single_patch_descs(desc) for desc in descriptors]
        try:  # most likely pytorch
            res = torch.stack(res)
        except TypeError:
            try:
                res = np.stack(res)
            except TypeError:  # leave it as a list
                res = res

        if cache_name is not None:
            print(f"Caching VLAD vectors to {cache_path}...")
            torch.save(res, cache_path)

        return res

    def generate_single_patch_descs(
        self, descriptors: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Given a set of descriptors, generate a VLAD vector for each descriptor. Call
        `fit` before using this method. Uses a serialised method that uses less memory.

            Parameters
            ----------
            descriptors (... x desc_dim Union[np.ndarray, torch.Tensor]):
                a set of descriptors which are turned into a VLAD vector.

            Returns
            -------
            vlads (... x n_clusters * desc_dim torch.Tensor):
                the corresponding VLAD descriptors.
        """
        assert self.kmeans is not None
        assert self.c_centers is not None

        if type(descriptors) == np.ndarray:
            descriptors = torch.from_numpy(descriptors).to(torch.float)
        if self.norm_descs:
            descriptors = F.normalize(descriptors, dim=-1)

        descriptor_shape = descriptors.shape[:-1]
        descriptor_dim = descriptors.shape[-1]
        assert descriptor_dim == self.desc_dim

        descriptors = descriptors.view(-1, self.desc_dim)
        raw_vlad = torch.zeros(len(descriptors), self.num_clusters * self.desc_dim)

        if self.vlad_mode == "hard":
            # get labels for assignment of descriptors
            labels = self.kmeans.predict(descriptors)  # q

            residuals = descriptors - self.c_centers[labels]  # q d
            used_clusters = set(labels.numpy())
            for k in used_clusters:
                # Sum of residuals for the descriptors in the cluster
                cd = (labels == k)[:, None] * residuals  # q d
                if self.intra_norm:
                    cd = F.normalize(cd, dim=-1)
                raw_vlad[:, k * self.desc_dim : (k + 1) * self.desc_dim] = cd

        else:  # Soft cluster assignment
            residuals = descriptors[:, None, :] - self.c_centers[None, :, :]  # q c d
            # Cosine similarity: 1 = close, -1 = away
            cos_sims = F.cosine_similarity(  # [q, c]
                descriptors[:, None, :],
                self.c_centers[None, :, :],
                dim=2,
            )
            assignment = F.softmax(self.soft_temp * cos_sims, dim=1)  # [q, c]

            for k in range(self.num_clusters):

                cd = reduce(
                    assignment[:, k, None, None] * residuals, "q c d -> q d", "sum"
                )
                if self.intra_norm:
                    cd = F.normalize(cd, dim=-1)
                raw_vlad[:, k * self.desc_dim : (k + 1) * self.desc_dim] = cd

        # Normalize the VLAD vector
        vlads = F.normalize(raw_vlad, dim=-1)

        vlads = vlads.reshape(*descriptor_shape, self.num_clusters * self.desc_dim)
        return vlads
