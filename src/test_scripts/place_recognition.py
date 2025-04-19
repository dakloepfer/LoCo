import os

import cv2
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from loguru import logger as loguru_logger
from MixVPR.main import VPRModel as MixVPR
from tqdm import tqdm

from src.data.vpr_dataset import VPR_Dataset
from src.lightning.smoothap_module import PLModule
from src.models.vlad import VLAD
from src.utils.misc import (
    calc_neighbourhood_dists,
    extract_patch_descriptors,
    generate_all_combination_configs,
    get_downsampled_from_integral,
    get_integral_features,
    get_rank_zero_only_logger,
    get_top_k_accuracy,
    get_top_k_recall,
)

DATASET_NAMES = {
    "indoor": ["baidu", "gardens", "17places"],
    "outdoor": ["pitts30k", "st_lucia", "robotcar"],
    "underground": ["hawkins", "laurel_caverns"],
    "underwater": ["eiffel"],
}

loguru_logger = get_rank_zero_only_logger(loguru_logger)


def prune_coarse_vpr_configs(configs):
    """Remove configs with unnecessary combinations of parameters."""

    # remove unnecessary parameters
    pruned_configs = []

    for config in configs:
        # nothing to prune yet, as only one kind of coarse VPR is implemented
        pruned_configs.append(config)

    return pruned_configs


def prune_fine_vpr_configs(configs):
    """Remove configs with unnecessary combinations of parameters."""

    # remove unnecessary parameters
    pruned_configs = []

    for config in configs:
        if config.match_strategy.lower() == "mutual_nn":
            new_config = config.clone()
            for key in config:
                if key in ["nn_tolerance", "neighbourhood_tolerance", "lowes_ratio"]:
                    del new_config[key]
        elif config.match_strategy.lower() == "approx_mutual_nn":
            new_config = config.clone()
            for key in config:
                if key in ["neighbourhood_tolerance", "lowes_ratio"]:
                    del new_config[key]
        elif config.match_strategy.lower().startswith("filter"):
            new_config = config.clone()
            unneeded_keys = ["nn_tolerance"]
            if "lowes" not in config.match_strategy.lower():
                unneeded_keys.append("lowes_ratio")
            if "neighbourhood_consistency" not in config.match_strategy.lower():
                unneeded_keys.append("neighbourhood_tolerance")

            for key in config:
                if key in unneeded_keys:
                    del new_config[key]

        if config.score_strategy.lower() == "rapid_spatial_scoring":
            new_config = config.clone()
            for key in config:
                if key in []:
                    del new_config[key]
        elif config.score_strategy.lower() == "rapid_spatial_scoring2d":
            new_config = config.clone()
            for key in config:
                if key in []:
                    del new_config[key]
        elif config.score_strategy.lower() == "number":
            new_config = config.clone()
            for key in config:
                if key in ["use_original_rss"]:
                    del new_config[key]
        elif config.score_strategy.lower() == "ransac":
            new_config = config.clone()
            for key in config:
                if key in ["use_original_rss"]:
                    del new_config[key]

        pruned_configs.append(config)

    return pruned_configs


def generate_ablation_configs(config):
    config.defrost()

    new_coarse_configs = []
    for coarse_config in config.coarse:
        new_coarse_configs += generate_all_combination_configs(coarse_config)

    new_coarse_configs = prune_coarse_vpr_configs(new_coarse_configs)
    config.coarse = new_coarse_configs

    new_fine_configs = []
    for fine_config in config.fine:
        new_fine_configs += generate_all_combination_configs(fine_config)
    new_fine_configs = prune_fine_vpr_configs(new_fine_configs)

    config.fine = new_fine_configs
    config.freeze()
    return config


def test_place_recognition(config, profiler=None):

    feature_extractor = PLModule(config, profiler=profiler)
    if len(config.device_list) >= 1:
        feature_extractor = feature_extractor.to(config.device_list[0])

    vpr_config = config.eval.vpr
    config_file = os.path.join(config.save_dir, config.exp_name, "config.txt")
    if os.path.exists(config_file):
        raise ValueError(
            f"Config file path {config_file} exists! Please make sure you're not overwriting anything important"
        )
    if not os.path.exists(os.path.dirname(config_file)):
        os.makedirs(os.path.dirname(config_file))
    with open(config_file, "w+") as f:
        f.write(f"config: {config}")

    config.cache_dir = os.path.join(config.save_dir, "cache")
    os.makedirs(config.cache_dir, exist_ok=True)

    output_file = os.path.join(config.save_dir, config.exp_name, f"vpr_results.txt")
    if os.path.exists(output_file):
        raise ValueError(
            f"Output file path {output_file} exists! Please make sure you're not overwriting anything important"
        )
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    with open(output_file, "w+") as f:
        f.write(
            f"Visual Place Recognition Ablations for {config.exp_name}\n\
            Model ID: {feature_extractor.model_id}\
            ==========================================================\n\n\n"
        )
    vpr_config = generate_ablation_configs(vpr_config)

    for vpr_dataset_name in vpr_config.dataset_names:

        vpr_dataset = VPR_Dataset(
            vpr_dataset_name,
            vpr_config.img_size,
            vpr_config.datasets_folder,
            config.data.batch_size,
            config.data.num_workers,
        )

        with open(output_file, "a") as f:
            f.write(
                f"\n\nVisual Place Recognition Results on {vpr_dataset_name} ({vpr_dataset.dataset.db_num} ref and {vpr_dataset.dataset.query_num} query images): \
                    \n==============================================================\n\n "
            )

        gt_indices = vpr_dataset.get_gt_indices()

        for coarse_vpr_config in vpr_config.coarse:
            coarse_topk_indices, vlad_instance = run_coarse_vpr(
                config,
                coarse_vpr_config,
                vpr_dataset,
                feature_extractor,
                img_size=vpr_config.img_size,
                datasets_folder=vpr_config.datasets_folder,
            )
            if coarse_topk_indices is None:
                continue

            coarse_recalls = get_top_k_recall(
                vpr_config.top_k, coarse_topk_indices, gt_indices
            )
            coarse_accuracies = get_top_k_accuracy(
                vpr_config.top_k, coarse_topk_indices, gt_indices
            )

            # write to output file
            with open(output_file, "a") as f:
                f.write(
                    f"\nAfter using the following configuration for the COARSE place recognition: \n {coarse_vpr_config} \n\
                        \n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
                )
                for top_k, recall in zip(vpr_config.top_k, coarse_recalls):
                    f.write(f"Recall @ {top_k}: \t{recall}\n")

                f.write("\n")
                for top_k, accuracy in zip(vpr_config.top_k, coarse_accuracies):
                    f.write(f"Accuracy @ {top_k}: \t{accuracy}\n")

                f.write("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")

            for fine_vpr_config in vpr_config.fine:
                fine_topk_indices = run_fine_vpr(
                    config,
                    fine_vpr_config,
                    vpr_dataset,
                    feature_extractor,
                    coarse_topk_indices,
                    vlad_instance,
                )

                recalls = get_top_k_recall(
                    vpr_config.top_k, fine_topk_indices, gt_indices
                )
                accuracies = get_top_k_accuracy(
                    vpr_config.top_k, fine_topk_indices, gt_indices
                )

                # write to output file
                with open(output_file, "a") as f:
                    f.write(
                        f"\nAfter using the following configuration for the FINE place recognition: \n {fine_vpr_config} \n\
                            \n_______________________________________________________________\n"
                    )
                    for top_k, recall in zip(vpr_config.top_k, recalls):
                        f.write(f"Recall @ {top_k}: \t{recall}\n")

                    f.write("\n")
                    for top_k, accuracy in zip(vpr_config.top_k, accuracies):
                        f.write(f"Accuracy @ {top_k}: \t{accuracy}\n")

                    f.write(
                        "\n_______________________________________________________________\n"
                    )

            with open(output_file, "a") as f:
                f.write(
                    f"\noooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo\n"
                )


def run_coarse_vpr(general_config, config, dataset, feature_extractor, **kwargs):

    if config.type.lower() == "vlad":
        img_size = kwargs["img_size"]
        datasets_folder = kwargs["datasets_folder"]
        n_clusters = config.n_clusters
        vocab_dataset_names = config.vocab_dataset_names

        loguru_logger.info(
            f"Calculating Coarse Place Recognition Performance for {dataset.name} using VLAD, with {n_clusters} clusters and vocabulary set {vocab_dataset_names}"
        )
        if type(vocab_dataset_names) is str:
            vocab_dataset_names = DATASET_NAMES[vocab_dataset_names]
        if len(vocab_dataset_names) == 1:
            if vocab_dataset_names[0] != dataset.dataset.dataset_name:
                print(
                    f"Skipping using just {vocab_dataset_names[0]} as the vocabulary for VPR dataset {dataset.dataset.dataset_name} -- when using only one vocabulary dataset, it should match the VPR dataset!"
                )
                return None, None

        vlad_cache_dir = os.path.join(
            general_config.cache_dir,
            general_config.exp_name,
            f"vlad_cache_{n_clusters}_{vocab_dataset_names}",
        )

        if not os.path.exists(os.path.join(vlad_cache_dir, "cluster_centers.pt")):
            vocab_features = []
            for vocab_dataset in vocab_dataset_names:
                ds = VPR_Dataset(
                    vocab_dataset,
                    img_size,
                    datasets_folder,
                    general_config.data.batch_size,
                    general_config.data.num_workers,
                )
                loguru_logger.info(
                    f"Extracting patch descriptors for {vocab_dataset} db images"
                )
                vocab_features.append(
                    extract_patch_descriptors(
                        feature_extractor,
                        ds.get_db_imgs(),
                        cache_dir=os.path.join(
                            general_config.cache_dir, f"{vocab_dataset}_db_features"
                        ),
                    )
                )
            vocab_features = torch.cat(vocab_features, dim=0)
            vocab_features = rearrange(vocab_features, "b c h w -> b (h w) c")
        else:
            vocab_features = None  # not needed, just load the clusters from the cache

        vlad = VLAD(n_clusters, cache_dir=vlad_cache_dir)
        vlad.fit(vocab_features)

        loguru_logger.info(f"Extracting patch descriptors for {dataset.name} db images")
        db_features = extract_patch_descriptors(
            feature_extractor,
            dataset.get_db_imgs(),
            cache_dir=os.path.join(
                general_config.cache_dir, f"{dataset.name}_db_features"
            ),
        )
        db_features = rearrange(db_features, "b c h w -> b (h w) c")

        loguru_logger.info(f"Generating VLADs for {dataset.name} db images")
        db_vlads = vlad.generate(db_features, cache_name=f"{dataset.name}_db_vlads")
        if not general_config.debug:
            del db_features

        loguru_logger.info(f"Extracting patch descriptors for {dataset.name} queries")
        query_features = extract_patch_descriptors(
            feature_extractor,
            dataset.get_query_imgs(),
            cache_dir=os.path.join(
                general_config.cache_dir, f"{dataset.name}_query_features"
            ),
        )
        query_features = rearrange(query_features, "b c h w -> b (h w) c")

        loguru_logger.info(f"Generating VLADs for {dataset.name} query images")
        query_vlads = vlad.generate(
            query_features, cache_name=f"{dataset.name}_query_vlads"
        )
        if not general_config.debug:
            del query_features

        gt_indices = dataset.get_gt_indices()

        ## get top k indices
        db_vlads = db_vlads.cpu().numpy()
        query_vlads = query_vlads.cpu().numpy()
        vlad_dim = db_vlads.shape[1]

        # use cosine similarity
        index = faiss.IndexFlatIP(vlad_dim)
        index.add(db_vlads)
        dists, top_k_indices = index.search(query_vlads, config.coarse_top_k)

        if (
            general_config.debug and False
        ):  # TODO probably only temporary, remove later on
            idxs = top_k_indices[:, :20]
            # good_retrievals = [
            #     np.any(np.isin(idxs[i][:5], gt_indices[i])) for i in range(len(idxs))
            # ]
            bad_retrieval_indices = np.arange(
                100
            )  # np.nonzero(~np.array(good_retrievals))[0]
            retrieved_imgs = []
            gt_imgs = []
            from src.utils.visualisations import colour_patches

            if not os.path.exists(
                f"{dataset.dataset.dataset_name}_hard_{args.exp_name}_retrievals/"
            ):
                os.makedirs(
                    f"{dataset.dataset.dataset_name}_hard_{args.exp_name}_retrievals/"
                )
            for i in bad_retrieval_indices:
                print(".", end="", flush=True)
                if len(idxs[i]) > 0 and len(gt_indices[i]) > 0:

                    retrieved_imgs.append(
                        torch.stack([dataset.dataset[ret] for ret in idxs[i]])
                    )
                    gt_imgs.append(
                        torch.stack([dataset.dataset[gt] for gt in gt_indices[i]])
                    )
                    colour_patches(
                        dataset.dataset[dataset.dataset.db_num + i][None],
                        [],
                        f"{dataset.dataset.dataset_name}_hard_{args.exp_name}_retrievals/hard_queryimg_{dataset.dataset.dataset_name}{i}.png",
                    )
                    colour_patches(
                        retrieved_imgs[-1],
                        [],
                        f"{dataset.dataset.dataset_name}_hard_{args.exp_name}_retrievals/bad_retrieved_{dataset.dataset.dataset_name}{i}.png",
                    )
                    colour_patches(
                        gt_imgs[-1],
                        [],
                        f"{dataset.dataset.dataset_name}_hard_{args.exp_name}_retrievals/correct_retrievals_{dataset.dataset.dataset_name}{i}.png",
                    )
                else:
                    print(
                        f"for query {i}, no retrievals ({len(idxs[i])}) or no ground truth ({len(gt_indices[i])}) found!"
                    )
                    retrieved_imgs.append(torch.zeros_like(retrieved_imgs[-1]))
                    gt_imgs.append(torch.zeros_like(gt_imgs[-1]))

            breakpoint()

    elif config.type.lower() == "gem":
        img_size = kwargs["img_size"]
        datasets_folder = kwargs["datasets_folder"]
        exponent = torch.tensor(config.gem_exponent).float()
        loguru_logger.info(
            f"Calculating Coarse Place Recognition Performance for {dataset.name} using GeM, with an exponent of {exponent}"
        )

        loguru_logger.info(f"Extracting patch descriptors for {dataset.name} db images")
        db_features = extract_patch_descriptors(
            feature_extractor,
            dataset.get_db_imgs(),
            cache_dir=os.path.join(
                general_config.cache_dir, f"{dataset.name}_db_features"
            ),
        )
        db_features = rearrange(db_features, "b c h w -> b (h w) c")

        db_descriptors = F.normalize(db_features, p=2, dim=-1)
        db_descriptors = torch.pow(
            torch.sum(torch.pow(db_descriptors, exponent), dim=1), 1 / exponent
        )
        db_descriptors = F.normalize(db_descriptors, p=2, dim=-1)

        loguru_logger.info(
            f"Extracting patch descriptors for {dataset.name} query images"
        )
        query_features = extract_patch_descriptors(
            feature_extractor,
            dataset.get_query_imgs(),
            cache_dir=os.path.join(
                general_config.cache_dir, f"{dataset.name}_query_features"
            ),
        )
        query_features = rearrange(query_features, "b c h w -> b (h w) c")

        query_descriptors = F.normalize(query_features, p=2, dim=-1)
        query_descriptors = torch.pow(
            torch.sum(torch.pow(query_descriptors, exponent), dim=1), 1 / exponent
        )
        query_descriptors = F.normalize(query_descriptors, p=2, dim=-1)

        # use cosine similarity
        index = faiss.IndexFlatIP(768)
        index.add(db_descriptors)
        dists, top_k_indices = index.search(query_descriptors, config.coarse_top_k)

        vlad = None

    elif config.TYPE.lower() == "mixvpr":

        mixvpr = MixVPR(
            backbone_arch="resnet50",
            layers_to_crop=[4],
            agg_arch="MixVPR",
            agg_config={
                "in_channels": 1024,
                "in_h": 20,
                "in_w": 20,
                "out_channels": 1024,
                "mix_depth": 4,
                "mlp_ratio": 1,
                "out_rows": 4,
            },
        )

        state_dict = torch.load(
            "pretrained_weights/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt"
        )
        mixvpr.load_state_dict(state_dict)
        mixvpr.eval()

        loguru_logger.info(
            f"Extracting MixVPR descriptors for {dataset.name} db images"
        )
        db_descriptors = []
        for img in tqdm(dataset.get_db_imgs(), "extracting db descriptors"):
            db_descriptors.append(mixvpr(img).squeeze())
        db_descriptors = torch.cat(db_descriptors, dim=0)
        db_descriptors = F.normalize(db_descriptors, p=2, dim=-1)

        loguru_logger.info(
            f"Extracting MixVPR descriptors for {dataset.name} query images"
        )
        query_descriptors = []
        for img in tqdm(dataset.get_query_imgs(), "extracting query descriptors"):
            query_descriptors.append(mixvpr(img).squeeze())
        query_descriptors = torch.cat(query_descriptors, dim=0)
        query_descriptors = F.normalize(query_descriptors, p=2, dim=-1)

        # use cosine similarity
        index = faiss.IndexFlatIP(4096)
        index.add(db_descriptors)
        dists, top_k_indices = index.search(query_descriptors, config.coarse_top_k)

        vlad = None
    else:
        raise NotImplementedError(
            "Only VLAD-based place recognition implemented so far!"
        )

    return top_k_indices, vlad


def fine_vpr_extract_patch_descriptors(
    general_config, config, dataset, feature_extractor, vlad_instance
):
    # these should already be cashed from the coarse stage
    loguru_logger.info(f"Extracting patch descriptors for {dataset.name} queries")
    query_features = extract_patch_descriptors(
        feature_extractor,
        dataset.get_query_imgs(),
        cache_dir=os.path.join(
            general_config.cache_dir, f"{dataset.name}_query_features"
        ),
    )
    loguru_logger.info(f"Extracting patch descriptors for {dataset.name} db images")
    db_features = extract_patch_descriptors(
        feature_extractor,
        dataset.get_db_imgs(),
        cache_dir=os.path.join(general_config.cache_dir, f"{dataset.name}_db_features"),
    )

    if config.use_vlad_descriptors:
        query_features = rearrange(query_features, "b c h w -> b h w c")
        query_features = vlad_instance.generate_patch_descriptors(
            query_features, cache_name=f"{dataset.name}_query_patch_vlads"
        )
        query_features = rearrange(query_features, "b h w c -> b c h w")
        db_features = rearrange(db_features, "b c h w -> b h w c")
        db_features = vlad_instance.generate_patch_descriptors(
            db_features, cache_name=f"{dataset.name}_db_patch_vlads"
        )
        db_features = rearrange(db_features, "b h w c -> b c h w")

    if type(config.patch_size) is int:
        patch_size = config.patch_size
        patch_stride = config.patch_size
    else:
        patch_size, patch_stride = config.patch_size
    if patch_size == 1 and patch_stride == 1:
        pass
    else:
        # downsample using integral feature map
        integral_query = get_integral_features(query_features)
        integral_db = get_integral_features(db_features)

        query_features = get_downsampled_from_integral(
            integral_query, patch_size, patch_stride
        )  # b c h w
        db_features = get_downsampled_from_integral(
            integral_db, patch_size, patch_stride
        )  # b c h w

    query_features = F.normalize(query_features, p=2, dim=1)
    db_features = F.normalize(db_features, p=2, dim=1)

    return query_features, db_features


def fine_vpr_calc_matches(config, query, topk_db, strategy):
    """Create matches for the fine VPR using one of several strategies.

    Parameters
    ----------
    config (Namespace):
        the config for the fine VPR

    query (f x hq x wq tensor):
        the features for one query image

    topk_db (k x f x hdb x wdb):
        the features for the top-k db images for that query

    strategy (str):
        One of ["mutual_nn", "approx_mutual_nn", "filter_{neighbourhood_consistency}_{lowes}"]

    Returns
    -------
    match_query_to_db (k x hq x wq x hdb x wdb tensor):
        for each of the k db-images, index [k, i1, j1, i2, j2] is True iff patch i1, j1 in the query image is matched to patch i2, j2 in the db image

    match_db_to_query (k x hdb x wdb x hq*wq tensor):
        for each of the k db-images, index [k, i1, j1, i2, j2] is True iff patch i1, j1 in the db image is matched to patch i2, j2 in the query image
    """
    k, f, hdb, wdb = topk_db.shape
    hq, wq = query.shape[-2:]
    device = query.device

    # calculate cosine similarity
    similarity = einsum(topk_db, query, "k f hdb wdb, f hq wq -> k hdb wdb hq wq")
    similarity = rearrange(similarity, "k hdb wdb hq wq -> k (hdb wdb) (hq wq)")

    if strategy == "mutual_nn":
        ## get mutual nearest neighbours

        # for each db image: for each query patch, which patch in that db image is most similar
        query_to_db = similarity.argmax(dim=1)  # k x h*w; db_img x query_patch

        # for each db image: for each patch in that db image, which query patch is most similar
        db_to_query = similarity.argmax(dim=2)  # k x h*w; db_img x db_patch

        # for each db image: index i is True iff query-patch i is part of a mutual-NN pair with a patch in that db image
        # for each db image: index i is True iff query-patch i maps to a db-patch j that then itself maps back to query-patch i
        mutual_nn_query_mask = torch.gather(
            db_to_query, dim=1, index=query_to_db
        ) == torch.arange(
            hq * wq, device=device
        )  # k x h*w; db_img x query_patch

        # for each db image: index i is True iff patch i in that db image is part of a mutual-NN pair with a query-patch
        # for each db image: index i is True iff db-patch i maps to a query-patch j that then itself maps back to db-patch i
        mutual_nn_db_mask = torch.gather(
            query_to_db, dim=1, index=db_to_query
        ) == torch.arange(
            hdb * wdb, device=device
        )  # k x h*w; db_img x db_patch

        # mutual-nn is 1-to-1 so the number of mutual nearest neighbours should be the same for both query and db
        assert torch.allclose(
            mutual_nn_db_mask.sum(dim=1), mutual_nn_query_mask.sum(dim=1)
        )

        match_query_to_db = (
            repeat(
                torch.arange(hdb * wdb, device=device),
                "hdbwdb -> k hqwq hdbwdb",
                k=k,
                hqwq=hq * wq,
            )
            == query_to_db[:, :, None]
        )
        match_query_to_db[~mutual_nn_query_mask] = False

        match_db_to_query = (
            repeat(
                torch.arange(hq * wq, device=device),
                "hqwq -> k hdbwdb hqwq",
                k=k,
                hdbwdb=hdb * wdb,
            )
            == db_to_query[:, :, None]
        )
        match_db_to_query[~mutual_nn_db_mask] = False

        match_query_to_db = rearrange(
            match_query_to_db,
            "k (hq wq) (hdb wdb) -> k hq wq hdb wdb",
            hq=hq,
            wq=wq,
            hdb=hdb,
            wdb=wdb,
        )
        match_db_to_query = rearrange(
            match_db_to_query,
            "k (hdb wdb) (hq wq) -> k hdb wdb hq wq",
            hq=hq,
            wq=wq,
            hdb=hdb,
            wdb=wdb,
        )

    elif strategy == "approx_mutual_nn":
        ## get approximate mutual nearest neighbours
        # it counts as the nearest neighbour even if the reverse match is up to TOLERANCE away

        # for each db image: for each query patch, which patch in that db image is most similar
        query_to_db = similarity.argmax(dim=1)  # k x hq*wq

        # for each db image: for each patch in that db image, which query patch is most similar
        db_to_query = similarity.argmax(dim=2)  # k x hdb*wdb

        # x, y pixel coordinates in the given db image that the given query patch maps to
        query_to_db_xy = torch.stack(
            (query_to_db % wdb, query_to_db // wdb), dim=2
        )  # db_img x query_patch x 2

        # x, y pixel coordinates in the query image that the given db patch in the given db image maps to
        db_to_query_xy = torch.stack(
            (db_to_query % wq, db_to_query // wq), dim=2
        )  # db_img x db_patch x 2

        query_xy = torch.stack(
            (
                torch.arange(hq * wq, device=device) % wq,
                torch.arange(hq * wq, device=device) // wq,
            ),
            dim=1,
        )
        db_xy = torch.stack(
            (
                torch.arange(hdb * wdb, device=device) % wdb,
                torch.arange(hdb * wdb, device=device) // wdb,
            ),
            dim=1,
        )

        # for each db image: for each query-patch i the distance between itself and the query-patch that the db patch that query-patch i maps to maps back to
        query_source_reverse_match_distance = torch.linalg.norm(
            torch.gather(
                db_to_query_xy,
                dim=1,
                index=repeat(query_to_db, "b hw -> b hw t", t=2),
            ).float()
            - query_xy[None].float(),
            ord=2,
            dim=2,
        )  # k x hq*wq

        # for each db image: for each patch i in that image the distance between itself and the patch that the query patch that patch i maps to maps back to
        db_source_reverse_match_distance = torch.linalg.norm(
            torch.gather(
                query_to_db_xy,
                dim=1,
                index=repeat(db_to_query, "b hw -> b hw t", t=2),
            ).float()
            - db_xy[None].float(),
            ord=2,
            dim=2,
        )  # k x hdb*wdb

        # for each db image: index i is True iff query-patch i is part of a approx mutual-NN pair with a patch in that db image
        # for each db image: index i is True iff query-patch i maps to a db-patch j that then itself maps back to (somewhere close to) query-patch i
        approx_mutual_nn_query_mask = (
            query_source_reverse_match_distance < config.nn_tolerance
        )

        # for each db image: index i is True iff patch i in that db image is part of a approx mutual-NN pair with a query-patch
        # for each db image: index i is True iff db-patch i maps to a query-patch j that then itself maps back to (somewhere close to) db-patch i
        approx_mutual_nn_db_mask = (
            db_source_reverse_match_distance < config.nn_tolerance
        )

        match_query_to_db = (
            repeat(
                torch.arange(hdb * wdb, device=device),
                "hdbwdb -> k hqwq hdbwdb",
                k=k,
                hqwq=hq * wq,
            )
            == query_to_db[:, :, None]
        )
        match_query_to_db[~approx_mutual_nn_query_mask] = False

        match_db_to_query = (
            repeat(
                torch.arange(hq * wq, device=device),
                "hqwq -> k hdbwdb hqwq",
                k=k,
                hdbwdb=hdb * wdb,
            )
            == db_to_query[:, :, None]
        )
        match_db_to_query[~approx_mutual_nn_db_mask] = False

        match_query_to_db = rearrange(
            match_query_to_db,
            "k (hq wq) (hdb wdb) -> k hq wq hdb wdb",
            hq=hq,
            wq=wq,
            hdb=hdb,
            wdb=wdb,
        )
        match_db_to_query = rearrange(
            match_db_to_query,
            "k (hdb wdb) (hq wq) -> k hdb wdb hq wq",
            hq=hq,
            wq=wq,
            hdb=hdb,
            wdb=wdb,
        )

    elif strategy.startswith("filter"):

        # for each db image: for each query patch, which patch in that db image is most similar
        query_to_db = similarity.argmax(dim=1)  # k x hq*wq

        # for each db image: for each patch in that db image, which query patch is most similar
        db_to_query = similarity.argmax(dim=2)  # k x hdb*wdb

        # x, y pixel coordinates in the given db image that the given query patch maps to
        query_to_db_xy = torch.stack(
            (query_to_db % wdb, query_to_db // wdb), dim=2
        )  # db_img x query_patch x 2

        # x, y pixel coordinates in the query image that the given db patch in the given db image maps to
        db_to_query_xy = torch.stack(
            (db_to_query % wq, db_to_query // wq), dim=2
        )  # db_img x db_patch x 2

        # for each db image: index i is True iff patch i in the query image has a match with a patch in that db image that was not filtered out
        query_mask = torch.ones((k, hq, wq), device=device, dtype=torch.bool)

        # for each db image: index i is True iff patch i in that db image has a match with a patch in the query image that was not filtered out
        db_mask = torch.ones((k, hdb, wdb), device=device, dtype=torch.bool)

        if "lowes" in strategy:  # use Lowe's ratio test
            # the ratio in similarity between the best match and the best match that is not in the neighbourhood of the best match has to be > config.lowes_ratio

            raise NotImplementedError

        if "neighbourhood_consistency" in strategy:
            # the distance between the match of patch i and the matches of the patches in the neighbourhood of patch i has to be on average < config.neighbourhood_tolerance
            # neighbourhood_size = 1

            query_neighbourhood_dists, valid_mask = calc_neighbourhood_dists(
                rearrange(query_to_db_xy, "k (hq wq) t -> k hq wq t", hq=hq, wq=wq)
            )  # k hq wq 8
            mean_query_neighbourhood_dists = (
                query_neighbourhood_dists * valid_mask
            ).sum(dim=3) / valid_mask.sum(dim=3)
            query_mask = query_mask & (
                mean_query_neighbourhood_dists < config.neighbourhood_tolerance
            )

            db_neighbourhood_dists, valid_mask = calc_neighbourhood_dists(
                rearrange(
                    db_to_query_xy, "k (hdb wdb) t -> k hdb wdb t", hdb=hdb, wdb=wdb
                )
            )  # k hdb wdb 8
            mean_db_neighbourhood_dists = (db_neighbourhood_dists * valid_mask).sum(
                dim=3
            ) / valid_mask.sum(dim=3)
            db_mask = db_mask & (
                mean_db_neighbourhood_dists < config.neighbourhood_tolerance
            )

        match_query_to_db = (
            repeat(
                torch.arange(hdb * wdb, device=device),
                "hdbwdb -> k hqwq hdbwdb",
                k=k,
                hqwq=hq * wq,
            )
            == query_to_db[:, :, None]
        )
        match_query_to_db = rearrange(
            match_query_to_db,
            "k (hq wq) (hdb wdb) -> k hq wq hdb wdb",
            hq=hq,
            wq=wq,
            hdb=hdb,
            wdb=wdb,
        )
        match_query_to_db[~query_mask] = False

        match_db_to_query = (
            repeat(
                torch.arange(hq * wq, device=device),
                "hqwq -> k hdbwdb hqwq",
                k=k,
                hdbwdb=hdb * wdb,
            )
            == db_to_query[:, :, None]
        )
        match_db_to_query = rearrange(
            match_db_to_query,
            "k (hdb wdb) (hq wq) -> k hdb wdb hq wq",
            hq=hq,
            wq=wq,
            hdb=hdb,
            wdb=wdb,
        )
        match_db_to_query[~db_mask] = False

    else:
        raise NotImplementedError(
            f"Strategy {strategy} for generating Matches for fine VPR not implemented!"
        )
    assert torch.all(match_query_to_db.sum((3, 4)) <= 1)
    assert torch.all(match_db_to_query.sum((3, 4)) <= 1)

    return match_query_to_db, match_db_to_query


def fine_vpr_calc_scores(config, match_query_to_db, match_db_to_query, strategy):
    """Generate scores for how well the query image matches each of the db images.

    Parameters
    ----------
    config (Namespace):
        the config for the fine VPR

    match_query_to_db (k x hq x wq x hdb x wdb tensor):
        for each of the k db-images, index [k, i1, j1, i2, j2] is True iff patch i1, j1 in the query image is matched to patch i2, j2 in the db image

    match_db_to_query (k x hdb x wdb x hq x wq tensor):
        for each of the k db-images, index [k, i1, j1, i2, j2] is True iff patch i1, j1 in the db image is matched to patch i2, j2 in the query image

    strategy (str):
        One of ["number", "rapid_spatial_scoring", "rapid_spatial_scoring_2d", "RANSAC"].

    Returns
    -------
    scores (k tensor):
        the scores for each of the k db-images
    """
    device = match_query_to_db.device
    k, hq, wq, hdb, wdb = match_query_to_db.shape

    if strategy == "number":
        scores = match_query_to_db.sum((1, 2, 3, 4)) + match_db_to_query.sum(
            (1, 2, 3, 4)
        )

    elif strategy == "rapid_spatial_scoring":

        query_mask = rearrange(
            match_query_to_db, "k hq wq hdb wdb -> k hq wq (hdb wdb)"
        ).any(
            dim=-1
        )  # k hq wq
        db_mask = rearrange(
            match_db_to_query, "k hdb wdb hq wq -> k hdb wdb (hq wq)"
        ).any(
            dim=-1
        )  # k hdb wdb

        query_to_db_xy = torch.zeros((k, hq, wq, 2), device=device)
        query_to_db_xy[query_mask] = torch.nonzero(match_query_to_db[query_mask])[
            :, 1:
        ].float()

        db_to_query_xy = torch.zeros((k, hdb, wdb, 2), device=device)
        db_to_query_xy[db_mask] = torch.nonzero(match_db_to_query[db_mask])[
            :, 1:
        ].float()

        query_xy = rearrange(
            torch.stack(
                (
                    torch.arange(hq * wq, device=device) % wq,
                    torch.arange(hq * wq, device=device) // wq,
                ),
                dim=1,
            ),
            "(hq wq) t -> hq wq t",
            hq=hq,
            wq=wq,
            t=2,
        )
        db_xy = rearrange(
            torch.stack(
                (
                    torch.arange(hdb * wdb, device=device) % wdb,
                    torch.arange(hdb * wdb, device=device) // wdb,
                ),
                dim=1,
            ),
            "(hdb wdb) t -> hdb wdb t ",
            hdb=hdb,
            wdb=wdb,
            t=2,
        )

        ## calculate scores matching the query to database images
        spatial_dists = query_to_db_xy - query_xy[None]  # k hq wq 2

        mean_spatial_dists = torch.sum(
            spatial_dists.float() * query_mask[:, :, :, None],
            dim=(1, 2),
        ) / torch.clamp(
            query_mask.sum(dim=(1, 2))[:, None], min=1
        )  # k 2

        deltas = torch.abs(spatial_dists.float() - mean_spatial_dists[:, None, None, :])

        if config.use_original_rss:
            anchor_val = torch.tensor(
                [wdb, hdb], dtype=torch.float, device=device
            )  # maximum values of the x and y coordinates in the database images
            anchor_val = repeat(anchor_val, "two -> k hq wq two", k=k, hq=hq, wq=wq)
        else:
            raise NotImplementedError

        query_scores = torch.sum(
            query_mask[:, :, :, None] * torch.square(anchor_val - deltas),
            dim=(1, 2, 3),
        ) / (hq * wq)

        ## calculate scores matching database images to the query
        spatial_dists = db_to_query_xy - db_xy[None]  # k hdb wdb 2

        mean_spatial_dists = torch.sum(
            spatial_dists.float() * db_mask[:, :, :, None],
            dim=(1, 2),
        ) / torch.clamp(
            db_mask.sum(dim=(1, 2))[:, None], min=1
        )  # k 2

        deltas = torch.abs(spatial_dists.float() - mean_spatial_dists[:, None, None, :])

        if config.use_original_rss:
            anchor_val = torch.tensor(
                [wq, hq], dtype=torch.float, device=device
            )  # maximum values of the x and y coordinates in the query image
            anchor_val = repeat(
                anchor_val, "two -> k hdb wdb two", k=k, hdb=hdb, wdb=wdb
            )
        else:
            raise NotImplementedError

        db_scores = torch.sum(
            db_mask[:, :, :, None] * torch.square(anchor_val - deltas),
            dim=(1, 2, 3),
        ) / (hdb * wdb)

        scores = query_scores + db_scores

    elif strategy == "rapid_spatial_scoring_2d":

        query_mask = rearrange(
            match_query_to_db, "k hq wq hdb wdb -> k hq wq (hdb wdb)"
        ).any(
            dim=-1
        )  # k hq wq
        db_mask = rearrange(
            match_db_to_query, "k hdb wdb hq wq -> k hdb wdb (hq wq)"
        ).any(
            dim=-1
        )  # k hdb wdb

        query_to_db_xy = torch.zeros((k, hq, wq, 2), device=device)
        query_to_db_xy[query_mask] = torch.nonzero(match_query_to_db[query_mask])[
            :, 1:
        ].float()

        db_to_query_xy = torch.zeros((k, hdb, wdb, 2), device=device)
        db_to_query_xy[db_mask] = torch.nonzero(match_db_to_query[db_mask])[
            :, 1:
        ].float()

        query_xy = rearrange(
            torch.stack(
                (
                    torch.arange(hq * wq, device=device) % wq,
                    torch.arange(hq * wq, device=device) // wq,
                ),
                dim=1,
            ),
            "(hq wq) t -> hq wq t",
            hq=hq,
            wq=wq,
            t=2,
        )
        db_xy = rearrange(
            torch.stack(
                (
                    torch.arange(hdb * wdb, device=device) % wdb,
                    torch.arange(hdb * wdb, device=device) // wdb,
                ),
                dim=1,
            ),
            "(hdb wdb) t -> hdb wdb t ",
            hdb=hdb,
            wdb=wdb,
            t=2,
        )

        ## calculate scores matching the query to database images
        spatial_dists = query_to_db_xy - query_xy[None]  # k hq wq 2

        mean_spatial_dists = torch.sum(
            spatial_dists.float() * query_mask[:, :, :, None],
            dim=(1, 2),
        ) / torch.clamp(
            query_mask.sum(dim=(1, 2))[:, None], min=1
        )  # k 2

        deltas = torch.linalg.norm(
            spatial_dists.float() - mean_spatial_dists[:, None, None, :], ord=2, dim=-1
        )  # k hq wq

        if config.use_original_rss:
            # just the length of the diagonal
            anchor_val = torch.sqrt(torch.tensor([wdb ^ 2 + hdb ^ 2], device=device))
            anchor_val = repeat(anchor_val, "() -> k hq wq", k=k, hq=hq, wq=wq)
        else:
            raise NotImplementedError

        query_scores = torch.sum(
            query_mask * torch.square(anchor_val - deltas), dim=(1, 2)
        ) / (hq * wq)

        ## calculate scores matching database images to the query
        spatial_dists = db_to_query_xy - db_xy[None]  # k hdb wdb 2

        mean_spatial_dists = torch.sum(
            spatial_dists.float() * db_mask[:, :, :, None],
            dim=(1, 2),
        ) / torch.clamp(
            db_mask.sum(dim=(1, 2))[:, None], min=1
        )  # k 2

        deltas = torch.linalg.norm(
            spatial_dists.float() - mean_spatial_dists[:, None, None, :], ord=2, dim=-1
        )  # k hdb wdb

        if config.use_original_rss:
            # just the length of the diagonal
            anchor_val = torch.sqrt(torch.tensor([wq ^ 2 + hq ^ 2], device=device))
            anchor_val = repeat(anchor_val, "() -> k hdb wdb", k=k, hdb=hdb, wdb=wdb)
        else:
            raise NotImplementedError

        db_scores = torch.sum(
            db_mask * torch.square(anchor_val - deltas), dim=(1, 2)
        ) / (hdb * wdb)

        scores = query_scores + db_scores

    elif strategy == "ransac":
        # threshold is patch stride
        if type(config.patch_size) is int:
            ransac_threshold = config.patch_size
        else:
            ransac_threshold = config.patch_size[1]

        scores = torch.zeros(k)

        query_mask = (
            rearrange(match_query_to_db, "k hq wq hdb wdb -> k hq wq (hdb wdb)")
            .any(dim=-1)
            .cpu()
            .numpy()
        )  # k hq wq
        db_mask = (
            rearrange(match_db_to_query, "k hdb wdb hq wq -> k hdb wdb (hq wq)")
            .any(dim=-1)
            .cpu()
            .numpy()
        )  # k hdb wdb

        query_to_db_xy = torch.zeros((k, hq, wq, 2), device=device)
        query_to_db_xy[query_mask] = torch.nonzero(match_query_to_db[query_mask])[
            :, 1:
        ].float()
        query_to_db_xy = query_to_db_xy.cpu().numpy().astype(np.float32)

        db_to_query_xy = torch.zeros((k, hdb, wdb, 2), device=device)
        db_to_query_xy[db_mask] = torch.nonzero(match_db_to_query[db_mask])[
            :, 1:
        ].float()
        db_to_query_xy = db_to_query_xy.cpu().numpy().astype(np.float32)

        query_xy = rearrange(
            np.stack((np.arange(hq * wq) % wq, np.arange(hq * wq) // wq), axis=1),
            "(hq wq) t -> hq wq t",
            hq=hq,
            wq=wq,
            t=2,
        )
        db_xy = rearrange(
            np.stack((np.arange(hdb * wdb) % wdb, np.arange(hdb * wdb) // wdb), axis=1),
            "(hdb wdb) t -> hdb wdb t ",
            hdb=hdb,
            wdb=wdb,
            t=2,
        )

        n_query_to_db_matches = match_query_to_db.sum((1, 2, 3, 4))
        n_db_to_query_matches = match_db_to_query.sum((1, 2, 3, 4))

        for i in range(k):  # for each db image; OpenCV does not allow vectorised
            # compute query-to-db score
            if n_query_to_db_matches[i] < 4:
                query_score = 0
            else:
                points_in_query = query_xy[query_mask[i]]
                points_in_db = query_to_db_xy[i][query_mask[i]]

                _, mask = cv2.findHomography(
                    points_in_query, points_in_db, cv2.RANSAC, ransac_threshold
                )
                query_score = mask.sum() / (hq * wq)

            # compute db-to-query score
            if n_db_to_query_matches[i] < 4:
                db_score = 0
            else:
                points_in_db = db_xy[db_mask[i]]
                points_in_query = db_to_query_xy[i][db_mask[i]]

                _, mask = cv2.findHomography(
                    points_in_db, points_in_query, cv2.RANSAC, ransac_threshold
                )
                db_score = mask.sum() / (hdb * wdb)

            scores[i] = query_score + db_score

    else:
        raise NotImplementedError(
            f"Strategy {strategy} for calculating matching scores not implemented!"
        )

    return scores


def run_fine_vpr(
    general_config,
    config,
    dataset,
    feature_extractor,
    coarse_topk_indices,
    vlad_instance,
):
    if config.match_strategy.lower() == "none":  # only use the coarse VPR
        return coarse_topk_indices

    else:
        loguru_logger.info(
            f"Calculating Fine Place Recognition Performance using {config.match_strategy} for matching and {config.score_strategy} for scoring"
        )

        coarse_topk_indices = torch.from_numpy(coarse_topk_indices)

        query_features, db_features = fine_vpr_extract_patch_descriptors(
            general_config, config, dataset, feature_extractor, vlad_instance
        )

        # do this in a for-loop for memory reasons
        n_queries = len(query_features)
        fine_topk_indices = -torch.ones_like(coarse_topk_indices)

        avg_number_matches = 0

        for i in tqdm(
            range(n_queries),
            f"Computing Fine-VPR scores for query images using {config.match_strategy} and {config.score_strategy}",
        ):
            query = query_features[i].to(general_config.device_list[0])
            topk_db = db_features[coarse_topk_indices[i]].to(
                general_config.device_list[0]
            )

            match_query_to_db, match_db_to_query = fine_vpr_calc_matches(
                config, query, topk_db, config.match_strategy.lower()
            )
            avg_number_matches += match_query_to_db.sum().float()
            avg_number_matches += match_db_to_query.sum().float()

            scores = fine_vpr_calc_scores(
                config,
                match_query_to_db,
                match_db_to_query,
                config.score_strategy.lower(),
            )

            # re-rank using scores
            fine_topk_indices[i] = coarse_topk_indices[i][
                scores.cpu().sort(descending=True, stable=True)[1]
            ]
        avg_number_matches /= n_queries * 2 * coarse_topk_indices.shape[1]
        loguru_logger.info(
            f"Matching Strategy {config.match_strategy} kept {avg_number_matches} / {query_features.shape[-1] * query_features.shape[-2]} matches on average"
        )

        assert torch.all(fine_topk_indices != -1)
        return fine_topk_indices
