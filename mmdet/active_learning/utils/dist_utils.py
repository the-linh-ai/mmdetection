import os
import pickle
import socket
import logging
from tempfile import mkdtemp
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor as T
import torch.distributed as dist

from mmcv.utils import print_log
from mmdet.utils import get_root_logger
from mmdet.utils.comm import synchronize, is_main_process


def setup_ddp(config):
    """
    Setup params for CUDA, GPU & distributed training.
    Note that this function is to be called before any logger is initialized,
    thus logging in this function is done simply with `logging` module.
    """
    logging.info(f"Local_rank: {config.local_rank}")
    world_size = os.environ.get("WORLD_SIZE")
    config.distributed_world_size = int(world_size) if world_size else 1
    logging.info(f"Distributed world size: {world_size}")

    if world_size is None:  # single-node single-gpu (no DDP)
        device = "cuda"
        config.distributed = False

    else:  # distributed training
        device = str(torch.device("cuda", config.local_rank))
        torch.cuda.set_device(device)

        torch.distributed.init_process_group(
            backend="gloo",
            world_size=config.distributed_world_size,
            rank=config.local_rank,
        )

        config.distributed = True
        logging.info(
            f"Initialized host {socket.gethostname()} as d.rank {config.local_rank} "
            f"on device={config.device}, world size={config.distributed_world_size}"
        )

    config.device = device

    return config


def broadcast_object(obj: Any = None):
    """
    Broadcase any arbitrary pickable object from the master process
    to all other processes. The master process must call this function
    with `obj` specified. Other processes must call this function
    without any arguments.
    """
    if not dist.is_initialized():
        return obj

    obj = [obj]
    torch.distributed.broadcast_object_list(obj, src=0)
    return obj[0]


def all_gather_list(
    data: Any,
    group=None,
    max_size: int = None,
    device: str = None,
):
    """Gathers arbitrary data from all nodes into a list.
    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.
    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
    """
    SIZE_STORAGE_BYTES = 4  # int32 to encode the payload size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    enc = pickle.dumps(data)
    enc_size = len(enc)

    if max_size is None:
        # All processes must use the same buffer size
        max_size = enc_size + SIZE_STORAGE_BYTES
        max_sizes = gather([max_size], device, buffer_size=1000)[0]
        max_size = max(max_sizes)
        max_size = round(max_size * 1.1)  # take 110%

    elif enc_size + SIZE_STORAGE_BYTES > max_size:
        raise ValueError(
            f'encoded data exceeds max_size, this can be fixed by increasing '
            f'buffer size: {enc_size}'
        )

    buffer_size = max_size * world_size

    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()

    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    assert enc_size < 256 ** SIZE_STORAGE_BYTES, 'Encoded object size should be less than {} bytes'.format(
        256 ** SIZE_STORAGE_BYTES)

    size_bytes = enc_size.to_bytes(SIZE_STORAGE_BYTES, byteorder='big')

    cpu_buffer[0:SIZE_STORAGE_BYTES] = torch.ByteTensor(list(size_bytes))
    cpu_buffer[SIZE_STORAGE_BYTES: enc_size + SIZE_STORAGE_BYTES] = torch.ByteTensor(list(enc))

    start = rank * max_size
    size = enc_size + SIZE_STORAGE_BYTES
    buffer[start: start + size].copy_(cpu_buffer[:size])

    if group is None:
        dist.all_reduce(buffer)
    else:
        dist.all_reduce(buffer, group=group)

    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size: (i + 1) * max_size]
            size = int.from_bytes(out_buffer[0:SIZE_STORAGE_BYTES], byteorder='big')
            if size > 0:
                result.append(pickle.loads(bytes(out_buffer[SIZE_STORAGE_BYTES: size + SIZE_STORAGE_BYTES].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list requires all '
            'workers to enter the function together, so this error usually indicates '
            'that the workers have fallen out of sync somehow. Workers can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one worker to finish an epoch '
            'while other workers are still iterating over their portions of the data.'
        )


def gather(
    objects_to_sync: List[Any],
    device,
    buffer_size: int = None,
) -> List[Tuple]:
    """
    Helper function to gather arbitrary objects.

    Args:
        objects_to_sync (List[Any]): List of any arbitrary objects. This list
            should be of the same size across all processes.

    Returns
        gathered_objects (List[Tuple]):  List of size `num_objects`, where
            `num_objects` is the number of objects in `objects_to_sync` input.
            Each element in this list is a tuple of gathered objects from
            multiple processes (of the same object).
    """
    if not dist.is_initialized():
        return [[obj] for obj in objects_to_sync]

    local_rank = dist.get_rank()
    distributed_world_size = dist.get_world_size()

    if distributed_world_size > 1:
        # For tensors that reside on GPU, we first need to detach it from its
        # computation graph, clone it, and then transfer it to CPU
        on_gpus = []
        copied_objects_to_sync = []
        for object in objects_to_sync:
            if isinstance(object, T) and object.is_cuda:
                on_gpus.append(1)
                copied_object_to_sync = torch.empty_like(  # clone, detach and transfer to CPU
                    object, device="cpu"
                ).copy_(object).detach_()
                copied_objects_to_sync.append(copied_object_to_sync)
            else:
                on_gpus.append(0)
                copied_objects_to_sync.append(object)

        global_objects_to_sync = all_gather_list(
            [local_rank, copied_objects_to_sync],
            max_size=buffer_size,
            device=device,
        )
        # Sort gathered objects according to ranks, so that all processes
        # will receive the same objects in the same order
        global_objects_to_sync = sorted(global_objects_to_sync, key=lambda x: x[0])

        gathered_objects = []
        for rank, items in global_objects_to_sync:
            if rank == local_rank:
                gathered_objects.append(objects_to_sync)  # not the copied ones
            else:
                # `items` is a list of objects from `local_rank=rank`
                # If any object originally resides on GPU, we need
                # to transfer it back
                assert len(items) == len(on_gpus)
                copied_items = []
                for item, on_gpu in zip(items, on_gpus):
                    if on_gpu:
                        item = item.to(device)
                    copied_items.append(item)
                gathered_objects.append(copied_items)

    else:
        gathered_objects = [objects_to_sync]

    gathered_objects = list(zip(*gathered_objects))
    return gathered_objects


def gather_list(
    object_list: List[Any],
    device,
    buffer_size: int = None,
):
    """
    Higher-level wrapper of `gather`. This function gathers `object_list`
    from all processes, appending all to a single list. Note that this
    utility is different from `gather` in the sense that `object_list`
    here can be viewed as a single object in `objects_to_sync` of
    `gather`.

    Args:
        object_list (List[Any]): List of any arbitrary objects. This list
            can be of different size across all processes.

    Returns:
        gathered_objects (List[Any]): List of gathered objects, whose size
            is equal to sum of sizes of individual processes' `object_list`.
    """
    objects = gather([object_list], device, buffer_size=buffer_size)[0]
    objects = sum(objects, [])  # flatten
    return objects


def gather_and_compare(
    obj: Any,
    device,
    buffer_size: int = None,
):
    """
    Higher-level wrapper of `gather`. Gather `obj` from all process and compare
    the gathered objects one-by-one. This requires all processes to enter this
    function together. Also, since the comparison is done with `==`, the input
    object is expected to be a primitive python object (e.g., str, list, etc.)
    """
    if not dist.is_initialized():
        return

    gathered_objects = gather([obj], device, buffer_size)[0]
    # Do comparison with the first object
    res = []
    for obj in gathered_objects:
        res.append(obj == gathered_objects[0])

    if not all(res):
        obj_str = "\n".join(
            str(gathered_obj)[:3000] for gathered_obj in gathered_objects
        )
        raise RuntimeError(
            f"Out of sync: comparison with the first process failed: list of "
            f"gathered objects for each process are:\n{obj_str}"
        )


def synchronize_weights(model: nn.Module, device: str):
    """
    Synchronize model weights from master process to all other processes.
    """
    if is_main_process():
        # Save to a temporary file
        state_dict = model.state_dict()
        save_path = os.path.join(mkdtemp(), "model.pth")
        torch.save(state_dict, save_path)
        # Send save path to other processes
        broadcast_object(save_path)

    else:
        # Load model from temp dir
        save_path = broadcast_object()
        state_dict = torch.load(save_path, map_location=device)
        model.load_state_dict(state_dict)

    # Delete saved model
    synchronize()
    if is_main_process():
        os.remove(save_path)

    synchronize()
    print_log("Synchronize weights successfully!", logger=get_root_logger())
