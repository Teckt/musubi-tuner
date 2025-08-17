from concurrent.futures import ThreadPoolExecutor
import gc
import time
from typing import Optional
import torch
import torch.nn as nn


import logging

logger = logging.getLogger(__name__)

def clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    logger.info(f"SWAP: Cleaning memory on device: {device}")
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        logger.info(f"SWAP: CUDA cache cleared on {device}")
    if device.type == "xpu":
        torch.xpu.empty_cache()
        logger.info(f"SWAP: XPU cache cleared on {device}")
    if device.type == "mps":
        torch.mps.empty_cache()
        logger.info(f"SWAP: MPS cache cleared on {device}")


def synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def swap_weight_devices_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__
    
    logger.info(f"SWAP: Starting CUDA weight swap - Moving {layer_to_cpu.__class__.__name__} to CPU and {layer_to_cuda.__class__.__name__} to {device}")

    weight_swap_jobs = []

    # This is not working for all cases (e.g. SD3), so we need to find the corresponding modules
    # for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
    #     print(module_to_cpu.__class__, module_to_cuda.__class__)
    #     if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
    #         weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
    swap_count = 0
    for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
        if hasattr(module_to_cuda, "weight") and module_to_cuda.weight is not None:
            module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
            if module_to_cpu is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))
                swap_count += 1
            else:
                if module_to_cuda.weight.data.device.type != device.type:
                    logger.info(f"SWAP: Moving {module_to_cuda_name} weight directly to {device} (no CPU counterpart)")
                    module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

    logger.info(f"SWAP: Found {swap_count} weight tensors to swap between CPU and {device}")
    
    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # cuda to cpu
        logger.info(f"SWAP: Moving {swap_count} weight tensors from {device} to CPU...")
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.record_stream(stream)
            module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

        stream.synchronize()

        # cpu to cuda
        logger.info(f"SWAP: Moving {swap_count} weight tensors from CPU to {device}...")
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
            module_to_cuda.weight.data = cuda_data_view

    stream.synchronize()
    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value
    logger.info(f"SWAP: CUDA weight swap completed for {layer_to_cpu.__class__.__name__}")


def swap_weight_devices_no_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """
    not tested
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__
    
    logger.info(f"SWAP: Starting non-CUDA weight swap - Moving {layer_to_cpu.__class__.__name__} between CPU and {device}")

    weight_swap_jobs = []
    for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    logger.info(f"SWAP: Found {len(weight_swap_jobs)} weight tensors to swap")

    # device to cpu
    logger.info(f"SWAP: Moving {len(weight_swap_jobs)} weight tensors from {device} to CPU...")
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

    synchronize_device(device)

    # cpu to device
    logger.info(f"SWAP: Moving {len(weight_swap_jobs)} weight tensors from CPU to {device}...")
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
        module_to_cuda.weight.data = cuda_data_view

    synchronize_device(device)
    logger.info(f"SWAP: Non-CUDA weight swap completed for {layer_to_cpu.__class__.__name__}")


def weighs_to_device(layer: nn.Module, device: torch.device):
    logger.info(f"SWAP: Moving all weights in {layer.__class__.__name__} to {device}")
    weight_count = 0
    for module in layer.modules():
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data = module.weight.data.to(device, non_blocking=True)
            weight_count += 1
    logger.info(f"SWAP: Moved {weight_count} weight tensors to {device}")


class Offloader:
    """
    common offloading class
    """

    def __init__(self, block_type: str, num_blocks: int, blocks_to_swap: int, device: torch.device, debug: bool = False):
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.debug = debug

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)

    def _submit_move_blocks(self, blocks, block_idx_to_cpu, block_idx_to_cuda):
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            start_time = time.perf_counter()
            logger.info(
                f"SWAP: [{self.block_type}] Starting block swap - Block {bidx_to_cpu} → CPU, Block {bidx_to_cuda} → {'CUDA' if self.cuda_available else 'device'}"
            )

            self.swap_weight_devices(block_to_cpu, block_to_cuda)

            elapsed = time.perf_counter() - start_time
            logger.info(f"SWAP: [{self.block_type}] Block swap completed in {elapsed:.2f}s - blocks {bidx_to_cpu} and {bidx_to_cuda}")
            
            if self.debug:
                print(
                    f"[{self.block_type}] Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'}"
                )
                print(f"[{self.block_type}] Moved blocks {bidx_to_cpu} and {bidx_to_cuda} in {elapsed:.2f}s")
            return bidx_to_cpu, bidx_to_cuda  # , event

        block_to_cpu = blocks[block_idx_to_cpu]
        block_to_cuda = blocks[block_idx_to_cuda]

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_cuda, block_to_cuda
        )

    def _wait_blocks_move(self, block_idx):
        if block_idx not in self.futures:
            return

        start_time = time.perf_counter()
        logger.info(f"SWAP: [{self.block_type}] Waiting for block {block_idx} to finish moving...")
        
        if self.debug:
            print(f"[{self.block_type}] Wait for block {block_idx}")

        future = self.futures.pop(block_idx)
        _, bidx_to_cuda = future.result()

        assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

        elapsed = time.perf_counter() - start_time
        logger.info(f"SWAP: [{self.block_type}] Block {block_idx} move completed, waited {elapsed:.2f}s")
        
        if self.debug:
            print(f"[{self.block_type}] Waited for block {block_idx}: {elapsed:.2f}s")


class ModelOffloader(Offloader):
    """
    supports forward offloading
    """

    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        debug: bool = False,
    ):
        super().__init__(block_type, num_blocks, blocks_to_swap, device, debug)

        self.supports_backward = supports_backward
        self.forward_only = not supports_backward  # forward only offloading: can be changed to True for inference

        if self.supports_backward:
            # register backward hooks
            self.remove_handles = []
            for i, block in enumerate(blocks):
                hook = self.create_backward_hook(blocks, i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)

    def set_forward_only(self, forward_only: bool):
        self.forward_only = forward_only

    def __del__(self):
        if self.supports_backward:
            for handle in self.remove_handles:
                handle.remove()

    def create_backward_hook(self, blocks: list[nn.Module], block_index: int) -> Optional[callable]:
        # -1 for 0-based index
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        # create  hook
        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if self.debug:
                print(f"Backward hook for block {block_index}")

            if swapping:
                self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        logger.info(f"SWAP: [{self.block_type}] Preparing block devices before forward pass...")
        logger.info(f"SWAP: [{self.block_type}] Moving first {self.num_blocks - self.blocks_to_swap} blocks to {self.device}")
        logger.info(f"SWAP: [{self.block_type}] Moving last {self.blocks_to_swap} blocks' weights to CPU")

        if self.debug:
            print(f"[{self.block_type}] Prepare block devices before forward")

        for i, b in enumerate(blocks[0 : self.num_blocks - self.blocks_to_swap]):
            logger.info(f"SWAP: [{self.block_type}] Moving block {i} to {self.device}")
            b.to(self.device)
            weighs_to_device(b, self.device)  # make sure weights are on device

        for i, b in enumerate(blocks[self.num_blocks - self.blocks_to_swap :], start=self.num_blocks - self.blocks_to_swap):
            logger.info(f"SWAP: [{self.block_type}] Moving block {i} structure to {self.device}, weights to CPU")
            b.to(self.device)  # move block to device first. this makes sure that buffers (non weights) are on the device
            weighs_to_device(b, "cpu")  # make sure weights are on cpu

        synchronize_device(self.device)
        clean_memory_on_device(self.device)
        logger.info(f"SWAP: [{self.block_type}] Block device preparation completed")

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        logger.info(f"SWAP: [{self.block_type}] Requested to wait for block {block_idx}")
        self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(self, blocks: list[nn.Module], block_idx: int):
        # check if blocks_to_swap is enabled
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        # if backward is enabled, we do not swap blocks in forward pass more than blocks_to_swap, because it should be on GPU
        if not self.forward_only and block_idx >= self.blocks_to_swap:
            return

        block_idx_to_cpu = block_idx
        block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
        block_idx_to_cuda = block_idx_to_cuda % self.num_blocks  # this works for forward-only offloading
        
        logger.info(f"SWAP: [{self.block_type}] Submitting forward block move - block {block_idx_to_cpu} → CPU, block {block_idx_to_cuda} → device")
        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
