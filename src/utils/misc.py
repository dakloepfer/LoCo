import contextlib
import copy
import inspect
import os
from argparse import ArgumentParser, Namespace
from itertools import chain
from typing import Any, Dict, List, Tuple, Union

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from loguru import _Logger, logger
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm
from yacs.config import CfgNode as CN

import wandb

#### Generally useful tensor / matrix operations


def batched_2d_index_select(input_tensor, indices):
    """
    Arguments:
        input_tensor: b x c x h x w
        indices: b x n x 2 (y, x in the last dimension)
    Returns:
        output: b x n x c

    output[b,n,c] = input_tensor[b, c, indices[b,n,0], indices[b,n,1]]
    """
    b, c, h, w = input_tensor.shape
    input_tensor = rearrange(input_tensor, "b c h w -> b (h w) c")
    indices = indices[:, :, 0] * w + indices[:, :, 1]
    indices = repeat(indices, "b n -> b n c", c=c)
    output = torch.gather(input_tensor, dim=1, index=indices)
    return output


def square_idxs_to_condensed(square_idxs, n):
    """Convert from an index in the square matrix of size n*n to the corresponding index in the condensed (distance) matrix of length n*(n-1)/2.

    Parameters
    ----------
    square_idxs (n_idxs x 2 tensor):
        a batch of indices in the square matrix.

    n (int):
        the size of the square matrix.

    Returns
    -------
    condensed_idxs (n_idxs tensor):
        the corresponding indices in the condensed matrix.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if (square_idxs[:, 0] == square_idxs[:, 1]).any():
        raise ValueError("square_idxs must not contain diagonal elements")

    condensed_idxs = torch.zeros_like(square_idxs[:, 0])

    lower_triangle = square_idxs[:, 0] > square_idxs[:, 1]
    condensed_idxs[lower_triangle] = (
        (n - 1) * square_idxs[lower_triangle, 1]
        - square_idxs[lower_triangle, 1] * (square_idxs[lower_triangle, 1] + 1) // 2
        + square_idxs[lower_triangle, 0]
        - 1
    )
    upper_triangle = square_idxs[:, 0] < square_idxs[:, 1]
    condensed_idxs[upper_triangle] = (
        (n - 1) * square_idxs[upper_triangle, 0]
        - square_idxs[upper_triangle, 0] * (square_idxs[upper_triangle, 0] + 1) // 2
        + square_idxs[upper_triangle, 1]
        - 1
    )

    return condensed_idxs


def squareform(condensed_mat, n, low_mem=False, upper_only=False):
    """Generate a full n x n distance matrix from a condensed distance matrix.

    Parameters
    ----------
    condensed_mat (batch_size x n*(n-1)/2 tensor):
        the batch of condensed distance matrices, in row-major order.

    n (int):
        the dimension of the full distance matrix.

    low_mem (bool, optional):
        Whether to use the low-memory-usage (but somewhat slower because not parallelised) version, by default False

    upper_only (bool, optional):
        Whether to only assign the upper triangular part of the full distance matrix, by default False. If this is True, the lower triangular part and the diagonal will be filled with inf.

    Returns
    -------
    square_mat (batch_size x n x n tensor):
        The square distance matrix.
    """
    if len(condensed_mat.shape) == 1:
        condensed_mat = condensed_mat.unsqueeze(0)
        no_batch = True
    else:
        no_batch = False
    batch_size = condensed_mat.shape[0]

    if upper_only:
        square_mat = torch.full(
            (batch_size, n, n),
            float("inf"),
            dtype=condensed_mat.dtype,
            device=condensed_mat.device,
        )
    else:
        square_mat = torch.zeros(
            (batch_size, n, n), dtype=condensed_mat.dtype, device=condensed_mat.device
        )

    if low_mem:
        condensed_idx = 0
        for i in range(n):
            square_mat[:, i, i + 1 :] = condensed_mat[
                :, condensed_idx : condensed_idx + n - i - 1
            ]
            condensed_idx += n - i - 1
    else:
        square_mat[:, *torch.triu_indices(n, n, 1)] = condensed_mat

    if not upper_only:
        square_mat += square_mat.clone().transpose(-1, -2)

    if no_batch:
        square_mat = square_mat.squeeze(0)

    return square_mat


def condensed_idxs_to_square(condensed_idxs, n):
    """Convert from an index in the condensed (distance) matrix of length n*(n-1)/2 to the corresponding index in the square matrix of size n*n.

    Parameters
    ----------
    condensed_idxs (n_idxs tensor):
        a batch of indices in the condensed matrix.

    n (int):
        the size of the square matrix.

    Returns
    -------
    square_idxs (n_idxs x 2 tensor):
        the corresponding indices in the square matrix.
    """
    if torch.max(condensed_idxs) >= n * (n - 1) // 2:
        raise ValueError("condensed_idxs must be smaller than n*(n-1)/2")
    if len(condensed_idxs.shape) != 1:
        condensed_idxs = condensed_idxs.squeeze()
    # breakpoint()
    square_idxs = torch.zeros(
        condensed_idxs.shape[0],
        2,
        dtype=condensed_idxs.dtype,
        device=condensed_idxs.device,
    )

    ## Original equation, easier to interpret
    # b = 1 / 2 - n
    # square_idxs[:, 0] = torch.floor(
    #     -b - torch.sqrt(b**2 - 2 * condensed_idxs)
    # ).to(condensed_idxs.dtype)

    if n**2 > 1e5:
        # use double precision to hopefully avoid most numerical issues due to lack of precision
        # a threshold of 1e6 is probably safe, but I'm being conservative here
        square_idxs[:, 0] = torch.floor(
            n - torch.sqrt((n**2 - n - 2 * condensed_idxs).double() + 0.25) - 0.5
        ).to(condensed_idxs.dtype)
    else:
        square_idxs[:, 0] = torch.floor(
            n - torch.sqrt(n**2 - n - 2 * condensed_idxs + 0.25) - 0.5
        ).to(condensed_idxs.dtype)

    square_idxs[:, 1] = (
        condensed_idxs
        + 1
        + ((square_idxs[:, 0] / 2) * (square_idxs[:, 0] + 3 - 2 * n)).to(
            condensed_idxs.dtype
        )
    )

    return square_idxs


def invert_se3(T):
    """Invert an SE(3) transformation matrix."""
    assert T.shape[-2:] == (4, 4), "T must be of shape (..., 4, 4)"

    rot = T[..., :3, :3]
    trans = T[..., :3, 3]

    if type(T) == torch.Tensor:
        inv_T = torch.zeros_like(T)
        inv_rot = rot.transpose(-1, -2)
        inv_trans = torch.einsum("...ij,...j->...i", -inv_rot, trans)

    else:  # numpy
        inv_T = np.zeros_like(T)
        inv_rot = np.swapaxes(rot, -1, -2)
        inv_trans = np.einsum("...ij,...j->...i", -inv_rot, trans)

    inv_T[..., :3, :3] = inv_rot
    inv_T[..., :3, 3] = inv_trans
    inv_T[..., 3, 3] = 1.0

    return inv_T


def so3_rotation_angle(R, use_deg=True, eps=1e-4):
    """Compute the angle of rotation of an SO(3) rotation matrix."""
    assert R.shape[-2:] == (3, 3), "Rotation matrix must be of shape (..., 3, 3)"

    trace = einsum(R, "... i i -> ...")

    if ((trace < -1.0 - eps) + (trace > 3.0 + eps)).any():
        raise ValueError(
            "A rotation matrix has trace outside valid range [-1-eps,3+eps]."
        )

    cos = (trace - 1.0) / 2.0

    if type(R) == torch.Tensor:
        angle = torch.acos(torch.clamp(cos, -1.0, 1.0))
    else:  # numpy
        angle = np.arccos(np.clip(cos, -1.0, 1.0))

    if use_deg:
        angle = angle * 180.0 / np.pi

    return angle


#### Geometry Utils


def calc_neighbourhood_dists(match_locations):
    """For each patch i, calculate the distanc between the patch that patch i is matched to and the patches that its neighbours (adjacent patches) are matched to. Only works for neighbourhood size 1.

    Parameters
    ----------
    match_locations (batch_size x height x width x 2 tensor):
        for each patch, the location (x, y) of the patch it is matched to.

    Returns
    -------
    neighbourhood_dists (batch_size x height x width x 8 tensor):
        for each patch i, for each neighbour of patch i (there are 8 of them), the distance between the patch that patch i is matched to and the patch that the respective neighbour is matched to.

    valid_mask (batch_size x height x width x 8 bool tensor):
        a mask that is True if the respective neighbour is valid (i.e. not out of bounds), and False otherwise.
    """
    if len(match_locations.shape) == 3:
        match_locations = match_locations.unsqueeze(0)
        squeeze_result = True
    else:
        squeeze_result = False

    b, h, w, _ = match_locations.shape
    device = match_locations.device
    neighbourhood_size = 8

    match_locations = rearrange(match_locations, "b h w t -> b (h w) t")

    # h w indices of all neighbours
    neighbourhood_idxs = torch.stack(
        (
            repeat(
                torch.arange(h, device=device),
                "h -> b (h w) n",
                b=b,
                h=h,
                w=w,
                n=neighbourhood_size,
            ),
            repeat(
                torch.arange(w, device=device),
                "w -> b (h w) n",
                b=b,
                h=h,
                w=w,
                n=neighbourhood_size,
            ),
        ),
        dim=-1,
    )
    neighbourhood_idxs += repeat(
        torch.tensor(
            [
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [0, -1],
                [0, 1],
                [1, -1],
                [1, 0],
                [1, 1],
            ],
            dtype=torch.long,
            device=device,
        ),
        "n t -> b (h w) n t",
        b=b,
        h=h,
        w=w,
        n=neighbourhood_size,
        t=2,
    )
    valid_mask = (
        (neighbourhood_idxs[..., 0] >= 0)
        & (neighbourhood_idxs[..., 0] < h)
        & (neighbourhood_idxs[..., 1] >= 0)
        & (neighbourhood_idxs[..., 1] < w)
    )
    neighbourhood_idxs[~valid_mask] = 0

    # into hw indices
    neighbourhood_idxs = neighbourhood_idxs[..., 0] * w + neighbourhood_idxs[..., 1]
    neighbourhood_matches = torch.gather(
        repeat(
            match_locations,
            "b (h w) t -> b (h w) n t",
            b=b,
            h=h,
            w=w,
            n=neighbourhood_size,
            t=2,
        ),
        dim=1,
        index=repeat(
            neighbourhood_idxs,
            "b (h w) n -> b (h w) n t",
            b=b,
            h=h,
            w=w,
            n=neighbourhood_size,
            t=2,
        ),
    )

    neighbourhood_dists = torch.linalg.norm(
        match_locations[:, :, None, :].float() - neighbourhood_matches.float(),
        ord=2,
        dim=-1,
    )

    neighbourhood_dists = rearrange(
        neighbourhood_dists, "b (h w) n -> b h w n", b=b, h=h, w=w, n=neighbourhood_size
    )
    valid_mask = rearrange(
        valid_mask, "b (h w) n -> b h w n", b=b, h=h, w=w, n=neighbourhood_size
    )

    if squeeze_result:
        neighbourhood_dists = neighbourhood_dists.squeeze(0)
        valid_mask = valid_mask.squeeze(0)

    return neighbourhood_dists, valid_mask


#### Hydra Config Utils


def conf_to_string(conf):
    """Turn OmegaConf into string."""
    config_string = ""

    for key, value in conf.items():
        if len(config_string) > 0:
            config_string += "-"
        config_string += str(key)
        if type(value) in [dict, DictConfig]:
            config_string += f"_{conf_to_string(value)}"
        else:
            config_string += f"_{str(value)}"

    return config_string


#### YACS Config Utils


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def upper_config(dict_cfg):
    if not isinstance(dict_cfg, dict):
        return dict_cfg
    return {k.upper(): upper_config(v) for k, v in dict_cfg.items()}


def cfg_to_dict(cfg: CN):
    """Convert a yacs config node to a dictionary."""
    return {k: cfg_to_dict(v) if isinstance(v, CN) else v for k, v in cfg.items()}


def equal_configs(cfg1, cfg2) -> bool:
    """Check whether two configs are equal."""
    for k, v in cfg1.items():
        if k not in cfg2:
            return False
        if isinstance(v, CN) or isinstance(v, DictConfig):
            if not equal_configs(v, cfg2[k]):
                return False
        else:
            if v != cfg2[k]:
                return False
    for k in cfg2:
        if k not in cfg1:
            return False

    return True


def apply_to_subconfig(func, cfg: CN, subcfg_name: str, **kwargs):
    subcfg_name_list = subcfg_name.strip().split(".")
    if len(subcfg_name_list) == 1:
        setattr(cfg, subcfg_name, func(getattr(cfg, subcfg_name), **kwargs))
    else:
        apply_to_subconfig(
            func,
            getattr(cfg, subcfg_name_list[0]),
            ".".join(subcfg_name_list[1:]),
            **kwargs,
        )


def make_list(x, n=-1):
    if isinstance(x, list):
        return x
    else:
        return [x for _ in range(n)]


def generate_new_configs(configs, key):
    new_configs = []
    for config in configs:
        for value in config[key]:
            new_config = copy.deepcopy(config)
            new_config[key] = value
            new_configs.append(new_config)
    return new_configs


def generate_all_combination_configs(config):
    """Given a source config (some of) whose values are lists, generate a list of all the possible configs by making all combinations of the lists."""

    new_configs = [config]
    for key in config:
        if type(config[key]) in [list, ListConfig]:
            new_configs = generate_new_configs(new_configs, key)

    new_configs = remove_duplicate_configs(new_configs)
    return new_configs


def remove_duplicate_configs(configs):
    """Remove duplicate configs from the list."""

    filtered_configs = []

    for config in configs:
        config_already_added = False
        for filtered_config in filtered_configs:
            if equal_configs(config, filtered_config):
                config_already_added = True
                break

        if not config_already_added:
            filtered_configs.append(config)

    return filtered_configs


#### A bunch of systems or Pytorch Lightning-related utils


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument

    Usage:
        with tqdm_joblib(tqdm(desc="My calculation", total=10)) as progress_bar:
            Parallel(n_jobs=16)(delayed(sqrt)(i**2) for i in range(10))

    When iterating over a generator, directly use of tqdm is also a solutin (but monitor the task queuing, instead of finishing)
        ret_vals = Parallel(n_jobs=args.world_size)(
                    delayed(lambda x: _compute_cov_score(pid, *x))(param)
                        for param in tqdm(combinations(image_ids, 2),
                                          desc=f'Computing cov_score of [{pid}]',
                                          total=len(image_ids)*(len(image_ids)-1)/2))
    Src: https://stackoverflow.com/a/58936697
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def str_to_bool_or_str(val: str) -> Union[str, bool]:
    """Possibly convert a string representation of truth to bool. Returns the input otherwise. Based on the python
    implementation distutils.utils.strtobool.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values are 'n', 'no', 'f', 'false', 'off', and '0'.
    """
    lower = val.lower()
    if lower in ("y", "yes", "t", "true", "on", "1"):
        return True
    if lower in ("n", "no", "f", "false", "off", "0"):
        return False
    return val


def str_to_bool(val: str) -> bool:
    """Convert a string representation of truth to bool.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.

    Raises:
        ValueError:
            If ``val`` isn't in one of the aforementioned true or false values.

    >>> str_to_bool('YES')
    True
    >>> str_to_bool('FALSE')
    False
    """
    val_converted = str_to_bool_or_str(val)
    if isinstance(val_converted, bool):
        return val_converted
    raise ValueError(f"invalid truth value {val_converted}")


def str_to_bool_or_int(val: str) -> Union[bool, int, str]:
    """Convert a string representation to truth of bool if possible, or otherwise try to convert it to an int.

    >>> str_to_bool_or_int("FALSE")
    False
    >>> str_to_bool_or_int("1")
    True
    >>> str_to_bool_or_int("2")
    2
    >>> str_to_bool_or_int("abc")
    'abc'
    """
    val_converted = str_to_bool_or_str(val)
    if isinstance(val_converted, bool):
        return val_converted
    try:
        return int(val_converted)
    except ValueError:
        return val_converted


def _gpus_allowed_type(x) -> Union[int, str]:
    if "," in x:
        return str(x)
    else:
        return int(x)


def _int_or_float_type(x) -> Union[int, float]:
    if "." in str(x):
        return float(x)
    else:
        return int(x)


def _get_abbrev_qualified_cls_name(cls):
    assert isinstance(cls, type), repr(cls)
    if cls.__module__.startswith("pytorch_lightning."):
        # Abbreviate.
        return f"pl.{cls.__name__}"
    else:
        # Fully qualified.
        return f"{cls.__module__}.{cls.__qualname__}"


def get_init_arguments_and_types(cls) -> List[Tuple[str, Tuple, Any]]:
    r"""Scans the class signature and returns argument names, types and default values.

    Returns:
        List with tuples of 3 values:
        (argument name, set with argument types, argument default value).

    Examples:

        >>> from pytorch_lightning import Trainer
        >>> args = get_init_arguments_and_types(Trainer)

    """
    cls_default_params = inspect.signature(cls).parameters
    name_type_default = []
    for arg in cls_default_params:
        arg_type = cls_default_params[arg].annotation
        arg_default = cls_default_params[arg].default
        try:
            arg_types = tuple(arg_type.__args__)
        except AttributeError:
            arg_types = (arg_type,)

        name_type_default.append((arg, arg_types, arg_default))

    return name_type_default


def _parse_args_from_docstring(docstring: str) -> Dict[str, str]:
    arg_block_indent = None
    current_arg = None
    parsed = {}
    for line in docstring.split("\n"):
        stripped = line.lstrip()
        if not stripped:
            continue
        line_indent = len(line) - len(stripped)
        if stripped.startswith(("Args:", "Arguments:", "Parameters:")):
            arg_block_indent = line_indent + 4
        elif arg_block_indent is None:
            continue
        elif line_indent < arg_block_indent:
            break
        elif line_indent == arg_block_indent:
            current_arg, arg_description = stripped.split(":", maxsplit=1)
            parsed[current_arg] = arg_description.lstrip()
        elif line_indent > arg_block_indent:
            parsed[current_arg] += f" {stripped}"
    return parsed


def parse_argparser(cls, arg_parser: Union[ArgumentParser, Namespace]) -> Namespace:
    """Parse CLI arguments, required for custom bool types. Copied from https://pytorch-lightning.readthedocs.io/en/1.3.8/_modules/pytorch_lightning/utilities/argparse.html#parse_argparser"""
    args = (
        arg_parser.parse_args()
        if isinstance(arg_parser, ArgumentParser)
        else arg_parser
    )

    types_default = {
        arg: (arg_types, arg_default)
        for arg, arg_types, arg_default in get_init_arguments_and_types(cls)
    }

    modified_args = {}
    for k, v in vars(args).items():
        if k in types_default and v is None:
            # We need to figure out if the None is due to using nargs="?" or if it comes from the default value
            arg_types, arg_default = types_default[k]
            if bool in arg_types and isinstance(arg_default, bool):
                # Value has been passed as a flag => It is currently None, so we need to set it to True
                # We always set to True, regardless of the default value.
                # Users must pass False directly, but when passing nothing True is assumed.
                # i.e. the only way to disable something that defaults to True is to use the long form:
                # "--a_default_true_arg False" becomes False, while "--a_default_false_arg" becomes None,
                # which then becomes True here.

                v = True

        modified_args[k] = v
    return Namespace(**modified_args)


def add_pl_argparse_args(cls, parent_parser: ArgumentParser):
    """Extends existing argparse by default attributes for a Pytorch Lightning Trainer.
    Replaces the (annoyingly deprecated) PytorchLightning function pl.Trainer.add_argparse_args().

    Code largely copied from https://pytorch-lightning.readthedocs.io/en/1.3.8/_modules/pytorch_lightning/utilities/argparse.html#add_argparse_args , with use_argument_group=True.

    Args:
        cls: a Lightning class (in practice a Trainer)
        parent_parser:
            The custom CLI arguments parser, which will be extended by the default Trainer default arguments.

    Returns:
        the parent parser with added default Trainer arguments.
    """
    parent_parser_args = [action.dest for action in parent_parser._actions]
    group_name = _get_abbrev_qualified_cls_name(cls)
    parser = parent_parser.add_argument_group(group_name)

    ignore_arg_names = ["self", "args", "kwargs"]
    if hasattr(cls, "get_deprecated_arg_names"):
        ignore_arg_names += cls.get_deprecated_arg_names()

    allowed_types = (str, int, float, bool)

    # Get symbols from cls or init function.
    for symbol in (cls, cls.__init__):
        args_and_types = get_init_arguments_and_types(symbol)
        args_and_types = [x for x in args_and_types if x[0] not in ignore_arg_names]
        if len(args_and_types) > 0:
            break

    args_help = _parse_args_from_docstring(cls.__init__.__doc__ or cls.__doc__ or "")

    for arg, arg_types, arg_default in args_and_types:
        if arg in parent_parser_args:
            continue
        arg_types = [at for at in allowed_types if at in arg_types]
        if not arg_types:
            # skip argument with not supported type
            continue
        arg_kwargs = {}
        if bool in arg_types:
            arg_kwargs.update(nargs="?", const=True)
            # if the only arg type is bool
            if len(arg_types) == 1:
                use_type = str_to_bool
            elif int in arg_types:
                use_type = str_to_bool_or_int
            elif str in arg_types:
                use_type = str_to_bool_or_str
            else:
                # filter out the bool as we need to use more general
                use_type = [at for at in arg_types if at is not bool][0]
        else:
            use_type = arg_types[0]

        if arg == "gpus" or arg == "tpu_cores":
            use_type = _gpus_allowed_type

        # hack for types in (int, float)
        if len(arg_types) == 2 and int in set(arg_types) and float in set(arg_types):
            use_type = _int_or_float_type

        # hack for track_grad_norm
        if arg == "track_grad_norm":
            use_type = float

        parser.add_argument(
            f"--{arg}",
            dest=arg,
            default=arg_default,
            type=use_type,
            help=args_help.get(arg),
            **arg_kwargs,
        )

    return parent_parser


def pl_from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs):
    """Manually re-implemented function to create a Pytorch Lightning Trainer from CLI arguments, as originally used in Trainer.from_argparse_args(), which has (annoyingly) been deprecated.

    Code largely taken copied from https://pytorch-lightning.readthedocs.io/en/1.3.8/_modules/pytorch_lightning/utilities/argparse.html#from_argparse_args , with the cls argument fixed as cls=Trainer.

    Args:
        cls: a Lightning class (in practice, a Trainer)
        args: The parser or namespace to take arguments from. Only known arguments will be
            parsed and passed to the :class:`Trainer`.
        **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
            These must be valid Trainer arguments.

    Returns:
        :class:`Trainer`
            Initialised Trainer object.
    """

    if isinstance(args, ArgumentParser):
        args = parse_argparser(cls, args)

    params = vars(args)

    # we only want to pass in valid Trainer args from the parser, the rest may be user specific
    valid_kwargs = inspect.signature(cls.__init__).parameters
    trainer_kwargs = dict(
        (name, params[name]) for name in valid_kwargs if name in params
    )
    trainer_kwargs.update(**kwargs)

    return cls(**trainer_kwargs)


def convert_cfg_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary"""
    if not isinstance(cfg_node, CN):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_cfg_to_dict(v, key_list + [k])
        return cfg_dict


#### Different Logging-related Utils


def average_dict_list(dict_list: List[Dict[Any, Any]]) -> Dict[Any, Any]:
    """Average a list of dictionaries, recursively if needed.

    Parameters
    ----------
    dict_list (List[Dict[Any, Any]]):
        the list of dictionaries to average. All dictionaries must have the same keys.

    Returns
    -------
    Dict[Any, Any]:
        a dictionary with the same keys, but with values averaged over the list.
    """
    if len(dict_list) == 0:
        return {}

    if len(dict_list) == 1:
        return dict_list[0]

    return_dict = {}
    for k in dict_list.keys():
        if isinstance(dict_list[0][k], dict):
            return_dict[k] = average_dict_list([d[k] for d in dict_list])
        else:
            return_dict[k] = sum([d[k] for d in dict_list]) / len(dict_list)

    return return_dict


def log_on(condition, message, level):
    if condition:
        assert level in ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]
        logger.log(level, message)


def get_rank_zero_only_logger(logger: _Logger):
    if rank_zero_only.rank == 0:
        return logger
    else:
        for _level in logger._core.levels.keys():
            level = _level.lower()
            setattr(logger, level, lambda x: None)
        logger._log = lambda x: None
    return logger


def flattenList(x):
    return list(chain(*x))


def add_statistics_to_logdict(log_dict, raw_samples_dict, log_wandb_histogram=False):
    for key, value in raw_samples_dict.items():
        if type(value) == float:
            log_dict[key] = value
            continue
        v = value.detach().float()
        if v.numel() == 0:
            log_dict[key] = v
            continue
        log_dict[f"mean_{key}"] = v.mean()
        log_dict[f"min_{key}"] = v.min()
        log_dict[f"max_{key}"] = v.max()
        log_dict[f"stdev_{key}"] = v.std()

        if log_wandb_histogram:
            w = torch.zeros_like(v)
            w[:] = v
            w[torch.isinf(w)] = 1e10 * torch.sign(w[torch.isinf(w)])
            w[torch.isnan(w)] = 0
            wandb.log(
                {f"raw_samples/{key}": wandb.Histogram(w.flatten().cpu().numpy())}
            )

    return log_dict


#### Metrics Utils


def get_top_k_recall(top_ks, topk_retrievals, gt_indices):
    """Given a database and query (or queries), calculate the recall for the top 'k'
    retrievals (closest in database for each query).

        Parameters
        ----------
        top_ks (list of ints):
            The k-values to calculate the recall for.

        topk_retrievals (n_queries x k numpy array or tensor):

        gt_indices (dict):
            a dictionary that for each query index contains the indices of the true database items.

        Returns
        -------
        recalls (list of floats):
            the recall for each of the k-values (percentage of correct retrievals).

    """
    if type(topk_retrievals) == torch.Tensor:
        topk_retrievals = topk_retrievals.cpu().numpy()

    topk_retrievals = topk_retrievals[:, : max(top_ks)]

    n_queries = len(topk_retrievals)
    assert n_queries == len(gt_indices)

    recalls = np.zeros(len(top_ks))
    for query_idx, retrievals in enumerate(topk_retrievals):
        gt_idxs = gt_indices[query_idx]

        correct_retrievals = np.isin(retrievals, gt_idxs)

        for i, k in enumerate(top_ks):
            if np.any(correct_retrievals[:k]):
                recalls[i] += 1

    recalls /= n_queries

    return recalls


def get_top_k_accuracy(top_ks, topk_retrievals, gt_indices):
    """Given a database and query (or queries), calculate the accuracy for the top 'k'
    retrievals (closest in database for each query). Basically the average fraction of top-k retrievals that are in the ground truth. When k > len(gt_indices[i]), the query is skipped in the calculation.

        Parameters
        ----------
        top_ks (list of ints):
            The k-values to calculate the recall for.

        topk_retrievals (n_queries x k numpy array or tensor):

        gt_indices (dict):
            a dictionary that for each query index contains the indices of the true database items.

        Returns
        -------
        accuracies (list of floats):
            the recall for each of the k-values (average fraction of correct retrievals).

    """

    if type(topk_retrievals) == torch.Tensor:
        topk_retrievals = topk_retrievals.cpu().numpy()

    topk_retrievals = topk_retrievals[:, : max(top_ks)]

    n_queries = len(topk_retrievals)
    assert n_queries == len(gt_indices)

    accuracies = np.zeros(len(top_ks))
    n_queries_used = np.zeros(len(top_ks))
    for query_idx, retrievals in enumerate(topk_retrievals):
        gt_idxs = gt_indices[query_idx]

        correct_retrievals = np.isin(retrievals, gt_idxs)

        for i, k in enumerate(top_ks):
            if k > len(gt_idxs):
                continue
            accuracies[i] += correct_retrievals[:k].sum() / k
            n_queries_used[i] += 1

    accuracies /= np.clip(n_queries_used, a_min=1e-8, a_max=None)

    return accuracies


#### Other


def extract_patch_descriptors(feature_extractor, imgs, cache_dir=None):
    """Extract patch descriptors from a feature extractor for a list of images. Potentially cache the results.

    Parameters
    ----------
    feature_extractor (PyTorch Module):
        the feature extractor to use.

    imgs (iterable):
        the images to extract the descriptors from.

    cache_dir (str, optional):
        the directory in which to cache the result / from which to read the result, by default None

    Returns
    -------
    batch_size x channels x height x width tensor:
        the extracted patch descriptors.

    """
    cache_filepath = os.path.join(
        cache_dir, f"patch_descriptors_{feature_extractor.model_id}.pkl"
    )

    if cache_dir is not None:
        if os.path.exists(cache_filepath):
            logger.info(f"Using cached patch descriptors from {cache_filepath}...")
            patch_descriptors = torch.load(cache_filepath)
            return patch_descriptors

    patch_descriptors = []
    for img in tqdm(imgs, "Extracting Patch Descriptors"):
        img = img.to(feature_extractor.device)
        descs = feature_extractor(img).cpu()
        patch_descriptors.append(descs)

    patch_descriptors = torch.cat(patch_descriptors, dim=0)

    if cache_dir is not None:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        logger.info(f"Caching patch descriptors to {cache_filepath}...")
        torch.save(patch_descriptors, cache_filepath)

    return patch_descriptors


def get_integral_features(feat_in):
    """Generate integral features from input features. Output is shape (... x h+1 x w+1), padded with zeros"""
    feat_out = torch.cumsum(torch.cumsum(feat_in, dim=-1), dim=-2)
    feat_out = F.pad(feat_out, (1, 0, 1, 0), "constant", 0)
    return feat_out


def get_downsampled_from_integral(integral_features, patch_size, patch_stride):

    if len(integral_features.shape) == 3:
        integral_features = integral_features.unsqueeze(0)

    b, c, h, w = integral_features.shape
    downsample_kernel = torch.ones(c, 1, 2, 2, device=integral_features.device)
    downsample_kernel[:, :, 0, -1] = -1
    downsample_kernel[:, :, -1, 0] = -1

    feat_regions = F.conv2d(
        integral_features,
        downsample_kernel,
        stride=patch_stride,
        groups=c,
        dilation=patch_size,
    )
    return (feat_regions / (patch_size**2)).squeeze(0)
