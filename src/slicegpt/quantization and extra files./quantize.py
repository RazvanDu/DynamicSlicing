# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import torch
from tqdm import tqdm

import numpy as np
import re
import math

from slicegpt.config import config
from slicegpt.model_adapter import LayerAdapter, ModelAdapter
from slicegpt.model_utils import get_layer0_inputs, get_signals
from slicegpt.utils import cleanup_memory, map_tensors


def quantize_dynamic(data, bits):
    # Calculate scale factor based on the number of bits

    #print(f"The data in quant function is:{data}")

    data_range = torch.max(data) - torch.min(data)
    data_range = 1 if data_range == 0 else data_range

    #print(f"The bits are {bits}")

    range_values = 2 ** bits

    scale = (range_values - 1) / data_range

    zeropoint = (-scale * torch.min(data) - (range_values / 2)).round()
    data_quant = torch.clip((data * scale + zeropoint).round(), -(range_values), range_values - 1)
    data_dequant = (data_quant - zeropoint) / scale

    #data_dequant[data_dequant== 0] = 10 ** (-(bits + 2))

    return data_dequant


def quantize_matrix_zones(matrix, quantization_limit_one, quantization_limit_two,
                          quant_second_zone_percent, quant_third_zone_percent):
    # print(f"The original vectt/mat is: {matrix}")
    quantized_matrix = matrix.clone()  # Clone the matrix to avoid modifying the original

    # print(f"The copied vectt/mat is: {matrix}")

    matrix_type = str(matrix.dtype)

    # Use regular expression to find numbers in the string
    original_format = int(re.findall(r'\d+', matrix_type)[0])

    bits_zone1 = int(original_format)

    # print(f"The bitzone is: {bits_zone1}")

    bits_zone2 = int(quant_second_zone_percent)
    bits_zone3 = int(quant_third_zone_percent)
    # print(f"The parameters are: quant_second_zone_percent: {quant_second_zone_percent}, quant_third_zone_percent: {quant_third_zone_percent}\n ")
    #print(f"The bitzone 1 is: {bits_zone1}, bitzone2: {bits_zone2}, and bitszone3: {bits_zone3}")

    #print(f"bit zone1: {bits_zone1}, bits_zone2: {bits_zone2}, bits_zone3: P{bits_zone3}")

    if matrix.dim() == 1:

        # print(f"The vect before quant is: {matrix}")

        # Treat the single dimension as columns
        cols = matrix.size(0)
        col_zone1_end = int(cols * quantization_limit_one)
        col_zone2_end = int(cols * quantization_limit_two)

        if col_zone1_end == col_zone2_end:
            col_zone2_end += 2
        if col_zone1_end == 0:
            col_zone1_end += 1
        # print(f"The zone parameters are: quantiz_limit_one: {quantization_limit_two}, quantiz_limit_two: {quantization_limit_two}\n"
        #      f"with the ranges: 0:{col_zone1_end}:{col_zone2_end} for columns, with initial dimension:{cols}")
        # Apply quantization based on column zones

        quantized_matrix[col_zone1_end:col_zone2_end] = quantize_dynamic(quantized_matrix[col_zone1_end:col_zone2_end],
                                                                         bits_zone2)
        quantized_matrix[col_zone2_end:] = quantize_dynamic(quantized_matrix[col_zone2_end:], bits_zone3)

        # print(f"The vect after quant is: {matrix}")


    else:

        #print(f"The matrix before quant is: {matrix}")

        # Define the zones based on matrix dimensions
        rows, cols = matrix.size()  # Get the dimensions of the matrix
        row_zone1_end = int(rows * quantization_limit_one)
        col_zone1_end = int(cols * quantization_limit_one)
        row_zone2_end = int(rows * quantization_limit_two)
        col_zone2_end = int(cols * quantization_limit_two)

        if col_zone1_end == col_zone2_end:
            col_zone2_end += 2
        if col_zone1_end == 0:
            col_zone1_end += 1

        if row_zone1_end == row_zone2_end:
            row_zone2_end += 2
        if row_zone1_end == 0:
            row_zone1_end += 1


        #print(
        #    f"The zone parameters are: quantiz_limit_one: {quantization_limit_one}, quantiz_limit_two: {quantization_limit_two}\n"
        #    f"with the ranges: 0:{col_zone1_end}:{col_zone2_end} for columns, with initial dimension:{cols}\n"
        #    f"with the ranges: 0:{row_zone1_end}:{row_zone2_end} for rows, with initial_dimension: {rows}")

        # Zone 2 - precision lower than Zone 1
        # We ensure that zone 1 is excluded by starting from row_zone1_end


        #print(f"\n The zone2_slice1 is: {row_zone1_end}:{row_zone2_end}, : {col_zone2_end} ")
        zone2_slice_1 = quantized_matrix[row_zone1_end:row_zone2_end, :col_zone2_end]
        quantized_matrix[row_zone1_end:row_zone2_end, :col_zone2_end] = quantize_dynamic(zone2_slice_1, bits_zone2)

        #print(f"\n The zone2_slice2 is: :{row_zone1_end}, {col_zone1_end}: {col_zone2_end} ")
        zone2_slice_2 = quantized_matrix[:row_zone1_end, col_zone1_end:col_zone2_end]
        quantized_matrix[:row_zone1_end, col_zone1_end:col_zone2_end] = quantize_dynamic(zone2_slice_2, bits_zone2)

        # Zone 3 - precision decreased by 2 more bits than Zone 2
        # This includes the area from row_zone2_end to the end of the matrix, and col_zone2_end to the end of the matrix
        # We also include the area from row_zone1_end to row_zone2_end for columns beyond col_zone2_end
        #print(f"\n The zone3_slice1 is: {row_zone2_end}:, :  ")
        zone3_slice_1 = quantized_matrix[row_zone2_end:, :]
        quantized_matrix[row_zone2_end:, :] = quantize_dynamic(zone3_slice_1, bits_zone3)

        #print(f"\n The zone3_slice1 is: :{row_zone2_end}, {col_zone2_end}:  ")
        zone3_slice_2 = quantized_matrix[:row_zone2_end, col_zone2_end:]
        quantized_matrix[:row_zone2_end, col_zone2_end:] = quantize_dynamic(zone3_slice_2, bits_zone3)

        #print(f"The matrix after quant is: {quantized_matrix}")
        #exit()

    return quantized_matrix


def slice_particular_layer_percent(nr_layers, initial_dimension, layer_number, percentage):

    new_dim = np.full(nr_layers + 1, initial_dimension)


    new_dimension = initial_dimension * (1 - percentage)

    print(f"The new dimension will be: {new_dimension}, with initial dimension {initial_dimension}, and cut percent: {percentage}")

    new_dim[layer_number] = new_dimension

    new_dim = np.round(new_dim).astype(int)

    print(f"The new vector will be: {new_dim}")

    return new_dim

def get_slice_dimension_by_cut_vector(initial_dimension, vector_cut):

    new_dim = np.array(vector_cut) * initial_dimension

    print(f"New vector will be: {new_dim}")

    new_dim = np.round(new_dim).astype(int)

    print(f"The new vector will be: {new_dim}")

    return new_dim



def rotate_attention_inputs(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_attention_inputs(layer_adapter: LayerAdapter, new_embedding_dimension: int, quantization_limit_one: float,
                           quantization_limit_two: float, quant_second_zone_percent: float, quant_third_zone_percent: float) -> None:
    # Slice the  WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.weight.data = quantize_matrix_zones(W.weight.data, quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)
        #new_data = quantize_matrix_zones(W.weight.data)
        #check_tensors(new_data,W.weight.data)

        W.in_features = new_embedding_dimension
        #print(f"Slice attention input- dimension {W.weight.data.shape}")

    layer_adapter.layer.attn_shortcut_Q = layer_adapter.layer.attn_shortcut_Q[:new_embedding_dimension, :]


def rotate_attention_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer_adapter.get_attention_output()

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def slice_attention_output(layer_adapter: LayerAdapter, new_embedding_dimension: int, quantization_limit_one: float,
                           quantization_limit_two: float, quant_second_zone_percent: float, quant_third_zone_percent: float) -> None:
    # Slice output matrix of the self-attention layer.
    W = layer_adapter.get_attention_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    W.weight.data = quantize_matrix_zones(W.weight.data, quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)
    #new_data = quantize_matrix_zones(W.weight.data)
    #check_tensors(new_data, W.weight.data)
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
        W.bias.data = quantize_matrix_zones(W.bias.data, quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)
        #new_data = quantize_matrix_zones(W.bias.data)
        #check_tensors(new_data, W.bias.data)

    W.out_features = new_embedding_dimension
    #(f"Slice attention output- dimension {W.weight.data.shape}")


def rotate_mlp_input(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_mlp_input(layer_adapter: LayerAdapter, new_embedding_dimension: int, quantization_limit_one: float,
                           quantization_limit_two: float, quant_second_zone_percent: float, quant_third_zone_percent: float) -> None:
    # Slice the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.weight.data = quantize_matrix_zones(W.weight.data, quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)
        #new_data = quantize_matrix_zones(W.weight.data)
        #check_tensors(new_data, W.weight.data)
        #print(f"Slice mlp input- dimension {W.weight.data.shape}")
        W.in_features = new_embedding_dimension


def rotate_mlp_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the MLP output weights and bias.
    W = layer_adapter.get_mlp_output()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def slice_mlp_output(layer_adapter: LayerAdapter, new_embedding_dimension: int, quantization_limit_one: float,
                           quantization_limit_two: float, quant_second_zone_percent: float, quant_third_zone_percent: float) -> None:
    # Slice the MLP output weights and bias.
    W = layer_adapter.get_mlp_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    W.weight.data = quantize_matrix_zones(W.weight.data, quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)
    #new_data = quantize_matrix_zones(W.weight.data)
    #check_tensors(new_data, W.weight.data)

    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
        W.bias.data = quantize_matrix_zones(W.bias.data, quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)
        #new_data = quantize_matrix_zones(W.bias.data)
        #check_tensors(new_data, W.bias.data)

    W.out_features = new_embedding_dimension
    #print(f"Slice mlp output- dimension {W.weight.data.shape}")


def rotate_embeddings(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in model_adapter.get_embeddings():
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    # Run GC and cleanup GPU memory
    cleanup_memory()


def slice_embeddings(model_adapter: ModelAdapter, new_embedding_dimension: int, quantization_limit_one: float,
                           quantization_limit_two: float, quant_second_zone_percent: float, quant_third_zone_percent: float) -> None:
    # Slice the embeddings.
    for W in model_adapter.get_embeddings():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.weight.data = quantize_matrix_zones(W.weight.data, quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)
        #new_data = quantize_matrix_zones(W.weight.data)
        #check_tensors(new_data, W.weight.data)


######## poate pusca
def slice_embeddings2(model_adapter: ModelAdapter, new_embedding_dimensions: np.array,quantization_limit_one: float,
                           quantization_limit_two: float, quant_second_zone_percent: float, quant_third_zone_percent: float) -> None:
    # Slice the embeddings
    for i, W in enumerate(model_adapter.get_embeddings()):
        #print("LOOKING AT LAYER ", i, new_embedding_dimensions[i])
        W.weight.data = W.weight.data[:, :new_embedding_dimensions[0]]
        W.weight.data = quantize_matrix_zones(W.weight.data, quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)
        #new_data = quantize_matrix_zones(W.weight.data)
        #check_tensors(new_data, W.weight.data)


        logging.info(W.weight.data.shape)
        #print(f"Slice emb- dimension {W.weight.data.shape}")

def rotate_head(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_adapter.get_lm_head()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_head(model_adapter: ModelAdapter, new_embedding_dimension: int, quantization_limit_one: float,
                           quantization_limit_two: float, quant_second_zone_percent: float, quant_third_zone_percent: float) -> None:
    # Slice the head.
    lm_head = model_adapter.get_lm_head()
    lm_head.weight.data = lm_head.weight.data[:, :new_embedding_dimension]
    lm_head.weight.data = quantize_matrix_zones(lm_head.weight.data, quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)
    #new_data = quantize_matrix_zones(lm_head.weight.data)
    #check_tensors(new_data, lm_head.weight.data)

    lm_head.in_features = new_embedding_dimension

    #print(f"Slice head- dimension {lm_head.weight.data.shape}")

def quantize(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    cut_vector: list,
    slice_layer_number: int,
    slice_percentage: float,
    quantization_limit_one: float,
    quantization_limit_two:float,
    quant_second_zone_percent : float,
    quant_third_zone_percent: float,
    do_slice_head: bool = False,
    ignore_tokens: list[int] | None = None,
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations
    """
    if model_adapter.parallel_blocks:

        rotate_and_slice_parallel(model_adapter, dataloader, cut_vector, slice_layer_number, slice_percentage,
                                  quantization_limit_one, quantization_limit_two,
                                  quant_second_zone_percent, quant_third_zone_percent,
                                  do_slice_head, ignore_tokens)
    else:
        rotate_and_slice_sequential(model_adapter, dataloader, cut_vector ,slice_layer_number, slice_percentage,
                                    quantization_limit_one, quantization_limit_two,
                                    quant_second_zone_percent, quant_third_zone_percent,
                                do_slice_head, ignore_tokens)


@torch.no_grad()
def rotate_and_slice_sequential(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    cut_vector: list,
    slice_layer_number: int,
    slice_percentage: float,
    quantization_limit_one: float,
    quantization_limit_two:float,
    quant_second_zone_percent : float,
    quant_third_zone_percent: float,
    do_slice_head: bool = False,
    ignore_tokens: list[int] | None = None,
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations

    This method works for models where the MLP block is computed after the attention block.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if ignore_tokens:
            ignore_masks.append(
                torch.stack([batch["input_ids"] == ignore_token for ignore_token in ignore_tokens]).any(dim=0)
            )

    _, Q = pca_calc(inps, ignore_masks)
    Q = Q.to(device=config.device)

    logging.info("Rotate and slice layers")
    layers = model_adapter.get_layers()

    #new_dimensions =slice_particular_layer_percent(len(layers), model_adapter.hidden_size, slice_layer_number,
   #                                                slice_percentage)

    new_dimensions = get_slice_dimension_by_cut_vector(model_adapter.hidden_size, cut_vector)
    print(new_dimensions)

    rotate_embeddings(model_adapter, Q)
    slice_embeddings2(model_adapter, new_dimensions, quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)

    logging.info("Rotate and slice layers")
    #layers = model_adapter.get_layers()
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = Q.T.clone().to(dtype=dtype)

        new_imp_emb_dimension = new_dimensions[idx]
        new_out_emb_dimension = new_dimensions[idx + 1]

        # rotate and slice the attention inputs to match previous layer
        rotate_attention_inputs(layer_adapter, Q)
        slice_attention_inputs(layer_adapter, new_imp_emb_dimension, quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent) # match matmul part

        # get signal between attention and mlp, rotate and slice

        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[:, :, :new_imp_emb_dimension].cpu(),
                args[i],
            )

        mlp_ln_inputs, _ = get_signals(layer_adapter, args, kwargs)
        _, Q = pca_calc(mlp_ln_inputs, ignore_masks)
        Q = Q.to(device=config.device, dtype=torch.float64)

        layer.attn_shortcut_Q = torch.matmul(layer.attn_shortcut_Q, Q.to(dtype=dtype)[:, :new_imp_emb_dimension]) # match 2 lines below
        rotate_attention_output(layer_adapter, Q)
        slice_attention_output(layer_adapter, new_imp_emb_dimension, quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent) # this must match slice_mlp_input

        layer.mlp_shortcut_Q = Q.T.clone().to(dtype=dtype)[:new_imp_emb_dimension, :]
        rotate_mlp_input(layer_adapter, Q)
        slice_mlp_input(layer_adapter, new_imp_emb_dimension, quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)
        _, inps = get_signals(layer_adapter, args, kwargs)

        # Run GC and cleanup GPU memory
        cleanup_memory()

        # now compute the outputs of the current layer/inputs for the next layer
        # with slicing between Attention and mlp.
        _, inps = get_signals(layer_adapter, args, kwargs)
        _, Q = pca_calc(inps, ignore_masks)

        layer.mlp_shortcut_Q = torch.matmul(layer.mlp_shortcut_Q, Q.to(dtype=dtype))
        # optionally slice the mlp/head connection in the last layer

        dim = new_out_emb_dimension
        if layer_adapter is layers[-1]:
            if not do_slice_head:
                dim = model_adapter.hidden_size

        rotate_mlp_output(layer_adapter, Q)
        slice_mlp_output(layer_adapter, dim, quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)
        layer_adapter.layer.mlp_shortcut_Q = layer_adapter.layer.mlp_shortcut_Q[:, :dim]

        layer.to('cpu')

        # Run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    rotate_head(model_adapter, Q)
    if do_slice_head:
        slice_head(model_adapter, new_dimensions[-1], quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)

    logging.info("Rotate and slice layers done")


@torch.no_grad()
def rotate_and_slice_parallel(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    cut_vector: list,
    slice_layer_number: int,
    slice_percentage: float,
    quantization_limit_one: float,
    quantization_limit_two:float,
    quant_second_zone_percent : float,
    quant_third_zone_percent: float,
    do_slice_head: bool = False,
    ignore_tokens: list[int] | None = None,
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations

    This version works for models where the MLP block and the attention block are computed in parallel.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if ignore_tokens:
            ignore_masks.append(
                torch.stack([batch["input_ids"] == ignore_token for ignore_token in ignore_tokens]).any(dim=0)
            )

    _, Q = pca_calc(inps, ignore_masks)
    Q = Q.to(device=config.device)

    logging.info("Rotate and slice layers")
    layers = model_adapter.get_layers()

    new_dimensions = get_slice_dimension_by_cut_vector(model_adapter.hidden_size, cut_vector)
    #new_dimensions = slice_particular_layer_percent(len(layers), model_adapter.hidden_size, slice_layer_number,
    #                                                slice_percentage)
    rotate_embeddings(model_adapter, Q)
    slice_embeddings2(model_adapter, new_dimensions, quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)

    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = Q.T.clone().to(dtype=dtype)

        new_imp_emb_dimension = new_dimensions[idx]
        new_out_emb_dimension = new_dimensions[idx + 1]

        # rotate and slice the inputs to match previous layer (both attention and mlp)
        rotate_attention_inputs(layer_adapter, Q)
        rotate_mlp_input(layer_adapter, Q)

        slice_attention_inputs(layer_adapter, int(new_imp_emb_dimension), quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)
        slice_mlp_input(layer_adapter, int(new_imp_emb_dimension), quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)

        # update the input signals to this layer, and re-run it
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[:, :, : int(new_imp_emb_dimension)].cpu(),                args[i],
            ) #aici inainte era input
            #print("Testing for the input signal to this layer ", args[i].shape)

        # the simpler equivalent of get_signals
        outputs = []
        layer = layer.to(config.device)
        for layer_args_batch, layer_kwargs_batch in zip(args, kwargs):
            layer_args_batch, layer_kwargs_batch = map_tensors(
                [layer_args_batch, layer_kwargs_batch], device=config.device
            )
            out = layer(*layer_args_batch, **layer_kwargs_batch)
            if isinstance(out, tuple):
                out = out[layer_adapter.hidden_states_output_position]
            out = out.cpu()
            outputs.append(out)

        inps = outputs
        _, Q = pca_calc(inps, ignore_masks)

        # update shorcut matrix
        layer.attn_shortcut_Q = torch.matmul(layer.attn_shortcut_Q, Q.to(dtype=dtype))

        # optionally slice the mlp/head connection in the last layer

        dim = int(new_out_emb_dimension)
        if layer_adapter is layers[-1]:
            if not do_slice_head:
                dim = model_adapter.hidden_size


        #print(f"\n\ndim is:{dim}\n\n" )

        rotate_mlp_output(layer_adapter, Q)
        rotate_attention_output(layer_adapter, Q)
        slice_mlp_output(layer_adapter, dim, quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)
        slice_attention_output(layer_adapter, dim, quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)

        # slice the shortcut (there is only one, we use attn_shortcut buffer)

        # it was output dimension
        layer.attn_shortcut_Q = layer.attn_shortcut_Q[:int(new_imp_emb_dimension), : dim]
        #print("Testing for layer attn shortcut q ", layer.attn_shortcut_Q.shape)
        layer.to('cpu')

        # Run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    rotate_head(model_adapter, Q)
    if do_slice_head:
        slice_head(model_adapter, new_dimensions[-1], quantization_limit_one, quantization_limit_two,
                        quant_second_zone_percent, quant_third_zone_percent)

    logging.info("Rotate and slice layers done")


@torch.no_grad()
def rotate(model_adapter: ModelAdapter, dataloader: torch.utils.data.DataLoader[torch.Tensor]) -> None:
    """
    Rotate a model.
    TODO: Make this gpu memory efficient.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype  # Get the dtype of the model.

    # List of layers to rotate.
    layers = model_adapter.get_layers()

    # Get the input of the first layer norm and calculate the Q_1
    inps, args, kwargs = [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)

    _, Q_1 = pca_calc(inps)
    Q_1 = Q_1.to(device=config.device)

    # Rotate the embeddings.
    rotate_embeddings(model_adapter, Q_1)

    # Rotate the rest of the model.
    logging.info("Rotate layers")
    for layer_adapter in tqdm(layers, unit="layer", desc="Rotating"):
        layer = layer_adapter.layer
        # Extract the inputs and outputs of the second layernorm input and calculate the Q_3
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(inp, args[i])
        mlp_ln_inputs, outs = get_signals(layer_adapter, args, kwargs)
        _, Q_3 = pca_calc(mlp_ln_inputs)
        Q_3 = Q_3.to(device=config.device)
        _, Q_5 = pca_calc(outs)
        Q_5 = Q_5.to(device=config.device)

        # Rotate the Q, K and V matrices of the self-attention layer.
        rotate_attention_inputs(layer_adapter, Q_1)

        # Set the shortcut rotation matrix of the self-attention layer.
        layer.attn_shortcut_Q = torch.matmul(Q_1.clone().T, Q_3.clone()).to(device="cpu", dtype=dtype)

        # Rotate the Attention output matrix
        rotate_attention_output(layer_adapter, Q_3)

        # Rotate the MLP input
        rotate_mlp_input(layer_adapter, Q_3)

        # Set the shortcut rotation matrix of the MLP.
        layer.mlp_shortcut_Q = torch.matmul(Q_3.clone().T, Q_5.clone()).to(device="cpu", dtype=dtype)

        # Rotate MLP output
        rotate_mlp_output(layer_adapter, Q_5)

        # Run GC and cleanup GPU memory
        cleanup_memory()

        inps = outs  # The inputs to the next layer are the outputs from this one!
        Q_1 = Q_5  # first rotation in the next layer is the last one in this...

    rotate_head(model_adapter, Q_5)
    logging.info("Rotate layers done")


def slice_rotated_model(model_adapter: ModelAdapter, new_embedding_dimension: int, do_slice_head: bool = False) -> None:
    """
    TODO: Make this gpu memory efficient.
    """
    model_adapter.model.eval()

    # slice embeddings
    slice_embeddings(model_adapter, new_embedding_dimension)

    # List of layers to sice.
    layers = model_adapter.get_layers()

    for layer_adapter in layers:
        layer = layer_adapter.layer
        slice_attention_inputs(layer_adapter, new_embedding_dimension)

        slice_mlp_input(layer_adapter, new_embedding_dimension)

        # slice mlp shortcut
        if layer_adapter.layer.mlp_shortcut_Q is not None:
            layer_adapter.layer.mlp_shortcut_Q = layer_adapter.layer.mlp_shortcut_Q[:new_embedding_dimension, :]

        # optionally slice the mlp/head connection in the last layer
        dim = new_embedding_dimension
        if layer_adapter is layers[-1]:
            if not do_slice_head:
                dim = model_adapter.hidden_size

        slice_mlp_output(layer_adapter, dim)
        if layer_adapter.layer.mlp_shortcut_Q is None:  # parallel case
            layer.attn_shortcut_Q = layer.attn_shortcut_Q[:new_embedding_dimension, :dim]
            slice_attention_output(layer_adapter, dim)
        else:  # sequential case
            layer.attn_shortcut_Q = layer.attn_shortcut_Q[:new_embedding_dimension, :new_embedding_dimension]
            layer.mlp_shortcut_Q = layer.mlp_shortcut_Q[:new_embedding_dimension, :dim]
            slice_attention_output(layer_adapter, new_embedding_dimension)

    if do_slice_head:
        slice_head(model_adapter, new_embedding_dimension)


@torch.no_grad()
def pca_calc(
    X: list[torch.Tensor], ignore_masks: list[torch.Tensor] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run PCA on a list of batched data. Returns the eigenvalues and eigenvectors.
    """
    # Run GC and cleanup GPU memory
    cleanup_memory()

    H = None
    for idx, X_batch in enumerate(X):
        if ignore_masks:
            X_batch[ignore_masks[idx]] = 0

        X_batch = X_batch.double().to(device=config.device)
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.
        H = H_batch if H is None else H + H_batch

    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=config.device)
    H[diag, diag] = H[diag, diag] + damp
    X_eig = torch.linalg.eigh(H)
    identity_matrix = torch.eye(H.shape[0], device=config.device, dtype=torch.float64)
    del H
    #print(X_eig.size())
    index = torch.argsort(X_eig[0], descending=True)
    #eig_val = X_eig[0][index]
    #eigen_vec = X_eig[1][:, index]



    eig_val = X_eig[0]
    eigen_vec = X_eig[1]

    condition_number = eig_val.max() / eig_val[eig_val > 0].min()
    print(condition_number)

    return eig_val, identity_matrix#eigen_vec

