# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import torch
from tqdm import tqdm

import numpy as np

from .config import config
from .model_adapter import LayerAdapter, ModelAdapter
from .model_utils import get_layer0_inputs, get_signals
from .utils import cleanup_memory, map_tensors

'''
def slicing_vector_generation(nr_layers, initial_dimension):
    new_dim = []
    for n_layer in range(nr_layers + 1):
        new_dim.append(initial_dimension - ((30 / 100 - n_layer / nr_layers * 20 / 100) * initial_dimension))
    new_dim = np.array(new_dim).astype('int')
    return new_dim
'''
def read_slicing_dimensions():
    file_path = "/storage/paulclotan/SmartSliceGPT/experiment_cut_search/layer_dimensions.txt"
    with open(file_path, 'r') as file:
        new_dim = [int(line.strip()) for line in file.readlines()]
    return np.array(new_dim)
'''
def slicing_vector_generation(nr_layers, initial_dimension):
    target_avg_value = initial_dimension * 0.7  # 30% reduction target

    diff = 0

    # Generate initial values randomly within ±20% of the initial dimension
    new_dim = np.random.uniform(1 - diff, 1 + diff, nr_layers + 1) * target_avg_value

    # Adjust the generated values to ensure the average is exactly as required
    new_dim_adjusted = adjust_values_to_target_average(new_dim, target_avg_value, diff)

    # Ensure the array contains integers
    new_dim_adjusted = np.round(new_dim_adjusted).astype(int)

    return new_dim_adjusted
'''

'''
// slicing a single layer, one by one with given parameters to the script.
def slice_particular_layer(nr_layers, initial_dimension, layer_number, amount, add_or_substract):
    # Create a vector with all elements set to the initial_dimension
    cut_dimension = initial_dimension * 0.7 # 0.7
    #print(initial_dimension, cut_dimension)
    new_dim = np.full(nr_layers + 1, cut_dimension)

    # Check if the layer_number is within the range of the layers

    # Modify the dimension of the specified layer based on the add_or_substract parameter
    #print(f"add or substract is: {add_or_substract}, with the type: {type(add_or_substract)}")
    if add_or_substract == False:
        # Ensure that the layer dimension cannot be less than 0 after subtraction
        new_dim[layer_number] = max(new_dim[layer_number] - amount, 0)
        #print(f"The new dim of the layer was substracted, new dim:{new_dim[layer_number]}")
    else:
        new_dim[layer_number] += amount
        #print(f"The new dim of the layer was added, new dim:{new_dim[layer_number]}")

    new_dim = np.round(new_dim).astype(int)

    return new_dim

#manual testing
def slice_particular_layer(nr_layers, initial_dimension, layer_number, amount, add_or_substract):
    # Create a vector with all elements set to the initial_dimension
    cut_dimension = initial_dimension * 0.7 # 0.7
    #print(initial_dimension, cut_dimension)
    new_dim = np.full(nr_layers + 1, cut_dimension)

    # Check if the layer_number is within the range of the layers

    # Modify the dimension of the specified layer based on the add_or_substract parameter
    #print(f"add or substract is: {add_or_substract}, with the type: {type(add_or_substract)}")
    if not add_or_substract:
        new_dim[layer_number] = max(new_dim[layer_number] - amount, 0)
        print(f"The new dim of the layer was substracted, new dim:{new_dim[layer_number]}")
    else:
        new_dim[layer_number] += amount
        print(f"The new dim of the layer was added, new dim:{new_dim[layer_number]}")

    new_dim = np.round(new_dim).astype(int)

    #print(f"hardcoded vector version for testing the perplexity: {new_dim}" )

    return new_dim
'''
'''
FUNCTION TO PERFORM PERCENTUAL CUT PER LAYER
'''
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

    print(f"Cut vector is: {vector_cut}")

    new_dim = np.round(new_dim).astype(int)

    print(f"The new vector will be: {new_dim}")

    return new_dim


def rotate_attention_inputs(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_attention_inputs(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the  WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
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


def slice_attention_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice output matrix of the self-attention layer.
    W = layer_adapter.get_attention_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension
    #(f"Slice attention output- dimension {W.weight.data.shape}")


def rotate_mlp_input(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_mlp_input(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
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


def slice_mlp_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP output weights and bias.
    W = layer_adapter.get_mlp_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
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


def slice_embeddings(model_adapter: ModelAdapter, new_embedding_dimension: int) -> None:
    # Slice the embeddings.
    for W in model_adapter.get_embeddings():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]

######## poate pusca
def slice_embeddings2(model_adapter: ModelAdapter, new_embedding_dimensions: np.array) -> None:
    # Slice the embeddings
    for i, W in enumerate(model_adapter.get_embeddings()):
        #print("LOOKING AT LAYER ", i, new_embedding_dimensions[i])
        W.weight.data = W.weight.data[:, :new_embedding_dimensions[0]]
        logging.info(W.weight.data.shape)
        #print(f"Slice emb- dimension {W.weight.data.shape}")

def rotate_head(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_adapter.get_lm_head()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_head(model_adapter: ModelAdapter, new_embedding_dimension: int) -> None:
    # Slice the head.

    print(f"\n\n\nSlicinggus der headus\n\n\n\n")
    lm_head = model_adapter.get_lm_head()
    lm_head.weight.data = lm_head.weight.data[:, :new_embedding_dimension]
    lm_head.in_features = new_embedding_dimension

    #print(f"Slice head- dimension {lm_head.weight.data.shape}")

def rotate_and_slice(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    cut_vector: list,
    slice_layer_number: int,
    slice_percentage: float,
    new_embedding_dimension: int,
    single_layer_cut: int,
    metric_to_use: int = 1,
    do_slice_head: bool = False,
    ignore_tokens: list[int] | None = None,
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations
    """
    if model_adapter.parallel_blocks:

        rotate_and_slice_parallel(model_adapter, dataloader, cut_vector, slice_layer_number, slice_percentage,
                                new_embedding_dimension, single_layer_cut, do_slice_head, ignore_tokens)
    else:
        rotate_and_slice_sequential(model_adapter, dataloader, cut_vector ,slice_layer_number, slice_percentage,
                                new_embedding_dimension, single_layer_cut, do_slice_head, ignore_tokens)


@torch.no_grad()
def rotate_and_slice_sequential(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    cut_vector: list,
    slice_layer_number: int,
    slice_percentage: float,
    new_embedding_dimension: int,
    single_layer_cut: int,
    do_slice_head: bool = False,
    ignore_tokens: list[int] | None = None,
    double_pattern_cut: bool = False,
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations

    This method works for models where the MLP block is computed after the attention block.
    """


    #double_pattern_cut - used to determine if we are determining a pattern for the slicing,
    #or executing a variation of the cutting. will be judged based on single_layer_cut - if 1(true) - we are evaluating the patter, if not we are slicing
    if single_layer_cut == 0: # we are using a pattern to cut the vector
        double_pattern_cut = True
    # write the logic to adapt the code to be able to run for perplexity graph. current logic
    #assumes that the opt and llama models will be always cut with 2 values per layer, not with one.
    #that logic for perplexity cutting graphs does not work.

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

    if single_layer_cut == 0:
        print("Vector cut")
        new_dimensions = get_slice_dimension_by_cut_vector(model_adapter.hidden_size, cut_vector)
    else:
        print("single layer cut")
        new_dimensions = slice_particular_layer_percent(len(layers), model_adapter.hidden_size, slice_layer_number,
                                                        slice_percentage)
    print(new_dimensions)

    rotate_embeddings(model_adapter, Q)
    slice_embeddings2(model_adapter, new_dimensions)

    logging.info("Rotate and slice layers")
    #layers = model_adapter.get_layers()

    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        if double_pattern_cut:
            idx = idx * 2
            print(f"Doubling the index")
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = Q.T.clone().to(dtype=dtype)

        new_imp_emb_dimension = new_dimensions[idx]
        new_out_emb_dimension = new_dimensions[idx + 1]

        # rotate and slice the attention inputs to match previous layer
        rotate_attention_inputs(layer_adapter, Q)
        if idx > 0 and double_pattern_cut:
            slice_attention_inputs(layer_adapter, new_dimensions[idx-1])
            print(f"Doubling the index2")

        else:
            slice_attention_inputs(layer_adapter, new_dimensions[idx])# new imp_dimensions
        #slice_attention_inputs(layer_adapter, new_imp_dimensiions) # match matmul part

        # get signal between attention and mlp, rotate and slice

        if idx > 0 and double_pattern_cut:
            for i, inp in enumerate(inps):
                args[i] = layer_adapter.get_updated_args(
                    torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[:, :, :new_dimensions[idx-1]].cpu(),
                    args[i],
                )
        else:
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
        slice_attention_output(layer_adapter, new_imp_emb_dimension) # this must match slice_mlp_input

        layer.mlp_shortcut_Q = Q.T.clone().to(dtype=dtype)[:new_imp_emb_dimension, :]
        rotate_mlp_input(layer_adapter, Q)
        slice_mlp_input(layer_adapter, new_imp_emb_dimension)
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
        slice_mlp_output(layer_adapter, dim)
        layer_adapter.layer.mlp_shortcut_Q = layer_adapter.layer.mlp_shortcut_Q[:, :dim]

        layer.to('cpu')

        # Run GC and cleanup GPU memory
        cleanup_memory()



    # rotate and slice head
    rotate_head(model_adapter, Q)
    if do_slice_head:
        slice_head(model_adapter, new_dimensions[-1])

    logging.info("Rotate and slice layers done")


@torch.no_grad()
def rotate_and_slice_parallel(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    cut_vector: list,
    slice_layer_number: int,
    slice_percentage: float,
    new_embedding_dimension: int,
    single_layer_cut: int,
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

    if single_layer_cut == 0:
        print("Vector cut")
        new_dimensions = get_slice_dimension_by_cut_vector(model_adapter.hidden_size, cut_vector)
    else:
        print("single layer cut")
        new_dimensions = slice_particular_layer_percent(len(layers), model_adapter.hidden_size, slice_layer_number,
                                                    slice_percentage)
    rotate_embeddings(model_adapter, Q)
    slice_embeddings2(model_adapter, new_dimensions)

    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = Q.T.clone().to(dtype=dtype)

        new_imp_emb_dimension = new_dimensions[idx]
        new_out_emb_dimension = new_dimensions[idx + 1]

        # rotate and slice the inputs to match previous layer (both attention and mlp)
        rotate_attention_inputs(layer_adapter, Q)
        rotate_mlp_input(layer_adapter, Q)

        slice_attention_inputs(layer_adapter, int(new_imp_emb_dimension))
        slice_mlp_input(layer_adapter, int(new_imp_emb_dimension))

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
        slice_mlp_output(layer_adapter, dim)
        slice_attention_output(layer_adapter, dim)

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
        slice_head(model_adapter, new_dimensions[-1])

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
    X: list[torch.Tensor], ignore_masks: list[torch.Tensor] | None = None,
    metric_to_use : int = 1
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
    del H
    index = torch.argsort(X_eig[0], descending=True)
    eig_val = X_eig[0][index]
    eigen_vec = X_eig[1][:, index]

    #print(f"\n\n\nThe column mean is: {eigen_vec.mean(dim=0)}")
    #print(f"The mean for each line is: {eigen_vec.mean(dim=1)}\n\n\n")


    # Assuming eigen_vec is a NumPy array and you already have it converted to a PyTorch tensor
    # Sample eigen_vec as a NumPy array (replace this with your actual eigen_vec)


    def compute_skewness(array):
        mean = torch.mean(array)
        diffs = array - mean
        var = torch.mean(torch.pow(diffs, 2.0))
        std = torch.pow(var, 0.5)
        zscores = diffs / std
        skews = torch.mean(torch.pow(zscores, 3.0))
        return skews

    def compute_coefficient_of_variation(array):
        mean = torch.mean(torch.abs(array))
        std = torch.std(torch.abs(array), dim=None)
        cv = std / mean
        print(f"\nThe std is: {std} and the mean is: {mean}\n")
        return cv

    # Compute skewness for each column (axis=0)
    if metric_to_use == 1:
        matrix_cv = compute_coefficient_of_variation(eigen_vec)
    else:
        matrix_cv = compute_skewness(torch.abs(eigen_vec)) # experiment matAbs
        #matrix_cv = compute_skewness(eigen_vec)


    print(f"\n\n\nSkewness for each column: {matrix_cv}")


    condition_number = eig_val.max() / eig_val[eig_val > 0].min()
    #print(condition_number)
    return eig_val, eigen_vec

