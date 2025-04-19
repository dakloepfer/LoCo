"""Utils related to data loading and processing."""

import collections
from typing import Callable, Dict, Optional, Tuple, Type, Union

import torch.nn.functional as F
from torch.utils.data._utils.collate import (
    default_collate_err_msg_format,
    default_collate_fn_map,
)


def variable_keys_collate(
    batch,
    *,
    collate_fn_map: Optional[
        Dict[Union[Type, Tuple[Type, ...]], Callable]
    ] = default_collate_fn_map,
):
    r"""
    Extends the default collate function to allow for samples that do not all contain the same keys, which now can lead to the collated output batch being a dictionary with values that have different batch sizes.

    Args:
        batch: a single batch to be collated
        collate_fn_map: Optional dictionary mapping from element type to the corresponding collate function.
          If the element type isn't present in this dictionary,
          this function will go through each key of the dictionary in the insertion order to
          invoke the corresponding collate function if the element type is a subclass of the key.

    Examples:
        >>> # samples do not all share the same keys
        >>> input = [{'every_third_sample': 0}, {'sample': 1}, {'sample': 2}, {'every_third_sample': 3}, {'sample': 4}, {'sample': 5}, {'every_third_sample': 6}, {'sample': 7}, {'sample': 8}, {'every_third_sample': 9}]
        >>> output = variable_keys_collate(input)
        >>> # Values in output have different batch sizes
        >>> print(output)
        >>> {'every_third_sample': tensor([0, 3, 6, 9]), 'sample': tensor([1, 2, 4, 5, 7, 8])}

    Note:
        Each collate function requires a positional argument for batch and a keyword argument
        for the dictionary of collate functions as `collate_fn_map`.
    """
    elem = batch[0]
    elem_type = type(elem)

    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](
                    batch, collate_fn_map=collate_fn_map
                )

    if isinstance(elem, collections.abc.Mapping):
        collated_batch = {}
        for d in batch:
            for key, value in d.items():
                if key in collated_batch:
                    collated_batch[key].append(value)
                else:
                    collated_batch[key] = [value]
        for key in collated_batch:
            collated_batch[key] = variable_keys_collate(
                collated_batch[key], collate_fn_map=collate_fn_map
            )

        try:
            return elem_type(collated_batch)
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return collated_batch
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(
            *(
                variable_keys_collate(samples, collate_fn_map=collate_fn_map)
                for samples in zip(*batch)
            )
        )
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [
                variable_keys_collate(samples, collate_fn_map=collate_fn_map)
                for samples in transposed
            ]  # Backwards compatibility.
        else:
            try:
                return elem_type(
                    [
                        variable_keys_collate(samples, collate_fn_map=collate_fn_map)
                        for samples in transposed
                    ]
                )
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [
                    variable_keys_collate(samples, collate_fn_map=collate_fn_map)
                    for samples in transposed
                ]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def crop_to_aspect_ratio(img, aspect_ratio, return_borders=False):
    """Crop the input image to the given aspect ratio.

    Args:
        img: The input image (... x h x w).
        aspect_ratio: The target aspect ratio (width / height)
        return_borders: If True, return the borders that were cropped (x, y).

    Returns:
        The cropped image.
    """
    h, w = img.shape[-2:]

    target_w = int(round(h * aspect_ratio))
    if w < target_w:  # crop height, ie y
        target_h = int(round(w / aspect_ratio))
        pad = (h - target_h) // 2
        img = img[..., pad : pad + target_h, :]
        borders = (0, pad)
    elif w > target_w:  # crop width, ie x
        pad = (w - target_w) // 2
        img = img[..., pad : pad + target_w]
        borders = (pad, 0)
    else:
        borders = (0, 0)

    if return_borders:
        return img, borders
    else:
        return img


def center_padding(images, patch_size):
    """Center pad a batch of images to a multiple of the patch size.

    Parameters
    ----------
    images (batch_size x channels x height x width tensor):
        the batch of images

    patch_size (int):
        The size of the patch in pixels

    Returns
    -------
    padded images (batch_size x channels x padded_height x padded_width tensor):
        The padded images.
    """
    _, _, h, w = images.shape
    diff_h = h % patch_size
    diff_w = w % patch_size

    if diff_h == 0 and diff_w == 0:
        return images

    pad_h = patch_size - diff_h
    pad_w = patch_size - diff_w

    pad_t = pad_h // 2
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    pad_b = pad_h - pad_t

    images = F.pad(images, (pad_l, pad_r, pad_t, pad_b))
    return images
