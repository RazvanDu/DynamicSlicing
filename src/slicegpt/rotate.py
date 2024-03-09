# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import config
from .model_adapter import LayerAdapter, ModelAdapter
from .model_utils import get_layer0_inputs, get_signals
from .slicing_scheduler import ConfigSlicingScheduler, ConstSlicingScheduler, SlicingScheduler
from .utils import cleanup_memory, map_tensors


def calculate_perplexity(model, dataloader, tokenizer):
    """
    Calculate the perplexity of the model on the given dataloader.
    This is a placeholder function; you'll need to implement it based on your specific model and task.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss
            total_loss += loss.item() * inputs.size(0)
            total_tokens += attention_mask.sum().item()

    average_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(average_loss))
    return perplexity.item()


def adjust_layer_dimensions(model_adapter, layer_pair):
    """
    Adjust the dimensions of the selected pair of layers.
    This is a placeholder function; the implementation depends on how your layers are structured and how you wish to modify them.
    """
    # Example: Modify this function to adjust the dimensions of the model's layers as needed
    pass


def optimize_model_perplexity(model_adapter, dataloader, slicing_scheduler, tokenizer, device='cuda'):
    """
    Optimizes the model's layers to minimize perplexity by repeatedly adjusting the dimensions
    of unique pairs of layers.

    Args:
    - model_adapter: A ModelAdapter instance for accessing and modifying the model.
    - dataloader: DataLoader instance providing the data.
    - slicing_scheduler: SlicingScheduler instance to manage slicing operations.
    - tokenizer: Tokenizer for the model.
    - device: The device to perform computations on.
    """
    improvement = True
    iteration = 0
    while improvement:
        improvement = False
        original_model_state = copy.deepcopy(model_adapter.model.state_dict())
        best_perplexity = calculate_perplexity(model_adapter.model.to(device), dataloader, tokenizer)
        original_perplexity = best_perplexity
        best_config = None

        # Generate all unique layer pairs
        layer_indices = list(range(len(model_adapter.get_layers())))
        unique_layer_pairs = [(i, j) for i in layer_indices for j in layer_indices if i < j]
        random.shuffle(unique_layer_pairs)  # Shuffle to ensure randomness

        for _ in tqdm(range(30), desc=f"Iteration {iteration}"):
            if not unique_layer_pairs:
                break

            layer_pair = unique_layer_pairs.pop()
            adjust_layer_dimensions(model_adapter, layer_pair)  # Placeholder: Implement based on your model

            # Placeholder: Implement the rotation and slicing operation for the new configuration
            # rotate_and_slice_parallel(model_adapter, dataloader, slicing_scheduler, apply_mask=True, final_orientation='pca')

            new_perplexity = calculate_perplexity(model_adapter.model.to(device), dataloader, tokenizer)

            if new_perplexity < best_perplexity:
                best_perplexity = new_perplexity
                best_config = layer_pair
                improvement = True
                # Save the best model state found so far
                best_model_state = copy.deepcopy(model_adapter.model.state_dict())

            # Revert model to original state before next adjustment
            model_adapter.model.load_state_dict(original_model_state)

        if improvement:
            # Apply the best configuration found in this iteration
            model_adapter.model.load_state_dict(best_model_state)
            print(
                f"Improvement found in iteration {iteration}: Perplexity improved from {original_perplexity} to {best_perplexity}, adjusting layers {best_config}.")
        else:
            print("No further improvement found.")

        iteration += 1


# Example usage placeholder: You will need to define or replace `model_adapter`, `dataloader`, `slicing_scheduler`, and `tokenizer` with your actual objects or variables.
# optimize_model_perplexity(model_adapter, dataloader, slicing_scheduler, tokenizer)


def adjust_values_to_target_average(values, target_avg, diff):
    current_avg = np.mean(values)
    while not np.isclose(current_avg, target_avg, rtol=1e-03):

        random_index = np.random.choice(len(values))

        #print(current_avg, target_avg, random_index)

        modifier = -1

        if current_avg < target_avg:
            modifier = 1

        values[random_index] += modifier
        values[random_index] = np.clip(values[random_index], target_avg * (1 - diff), target_avg * (1 + diff))

        current_avg = np.mean(values)

    return values

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

import numpy as np


def slice_particular_layer(nr_layers, initial_dimension, layer_number, amount, add_or_substract):
    # Create a vector with all elements set to the initial_dimension
    cut_dimension = initial_dimension * 0.7
    #print(initial_dimension, cut_dimension)
    new_dim = np.full(nr_layers + 1, cut_dimension)

    # Check if the layer_number is within the range of the layers
    if not (0 <= layer_number < nr_layers):
        raise ValueError(f"layer_number must be between 0 and {nr_layers - 1}, inclusive")

    # Modify the dimension of the specified layer based on the add_or_substract parameter
    if add_or_substract == False:
        # Ensure that the layer dimension cannot be less than 0 after subtraction
        new_dim[layer_number] = max(new_dim[layer_number] - amount, 0)
        print(f"The new dim of the layer was substracted, new dim:{new_dim[layer_number]}")
    else:
        new_dim[layer_number] += amount
        print(f"The new dim of the layer was added, new dim:{new_dim[layer_number]}")

    new_dim = np.round(new_dim).astype(int)

    return new_dim


#def slicing_vector_generation(nr_layers, initial_dimension):
#    target_avg_value = initial_dimension * 0.7

    # Initialize the new_dim list
   # new_dim = []

    # Generate values within ±20% of the target_avg_value
    #for _ in range(nr_layers + 1):
    #    # Calculate a random adjustment within ±20% of the target_avg_value
    #    adjustment = np.random.uniform(-0.2, 0.2) * target_avg_value
    #    # Apply the adjustment to get a new value and ensure it's within the bounds of the initial dimension
    #    #new_value = np.clip(target_avg_value + adjustment, initial_dimension * 0.8, initial_dimension * 1.2)
    #    new_dim.append(target_avg_value + adjustment)

    #print("AVERAGE CUT: ", sum(new_dim)/len(new_dim))

    #for n_layer in range(nr_layers + 1):
    #    new_dim.append(initial_dimension - ((30 / 100 - (nr_layers - n_layer) / nr_layers * 20 / 100) * initial_dimension))
        #new_dim.append((1-0.20) * initial_dimension)

    #new_dim = np.array(new_dim).astype('int')
    #return new_dim


def rotate_attention_inputs(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_attention_inputs(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the WQ, WK and WV matrices of the self-attention layer.
    layer_nr = 0
    logging.info(f"slice_attention_inputs {new_embedding_dimension}")
    for W in layer_adapter.get_attention_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension

    layer_adapter.layer.attn_shortcut_Q = nn.Parameter(layer_adapter.layer.attn_shortcut_Q[:new_embedding_dimension, :])


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
    logging.info(f"slice_attention_output {new_embedding_dimension}")
    W = layer_adapter.get_attention_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension


def rotate_mlp_input(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_mlp_input(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP input weights.
    logging.info(f"sslice_mlp_input {new_embedding_dimension}")
    for W in layer_adapter.get_mlp_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
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
    logging.info(f"slice_mlp_output {new_embedding_dimension}")
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension


def rotate_embeddings(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in model_adapter.get_embeddings():
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    # Run GC and cleanup GPU memory
    cleanup_memory()

def slice_embeddings2(model_adapter: ModelAdapter, new_embedding_dimensions: np.array) -> None:
    # Slice the embeddings
    logging.info(f"slice_embeddings {new_embedding_dimensions}")
    for i, W in enumerate(model_adapter.get_embeddings()):
        W.weight.data = W.weight.data[:, : new_embedding_dimensions[i]]
        logging.info(W.weight.data.shape)


def slice_embeddings(model_adapter: ModelAdapter, new_embedding_dimensions: dict[int, int]) -> None:
    # Slice the embeddings
    logging.info(f"slice_embeddings {new_embedding_dimensions}")
    for i, W in enumerate(model_adapter.get_embeddings()):
        W.weight.data = W.weight.data[:, : new_embedding_dimensions[i]]
        logging.info(W.weight.data.shape)


def rotate_head(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_adapter.get_lm_head()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_head(model_adapter: ModelAdapter, new_embedding_dimension: int) -> None:
    # Slice the head.
    logging.info(f"slice_head {new_embedding_dimension}")
    lm_head = model_adapter.get_lm_head()
    lm_head.weight.data = lm_head.weight.data[:, :new_embedding_dimension]
    lm_head.in_features = new_embedding_dimension
    #print(f"The head shape after slicing is:{lm_head}")

'''
def rotate_and_slice(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations
    """
    logging.info("\n\nrotate and slice func")
    if model_adapter.parallel_blocks:
        logging.info("\n\nparal branch")
        rotate_and_slice_parallel(model_adapter, dataloader, slicing_scheduler, apply_mask, final_orientation)
    else:
        logging.info("\n\nseq branch")
        rotate_and_slice_sequential(model_adapter, dataloader, slicing_scheduler, apply_mask, final_orientation)

'''

def rotate_and_slice(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slice_layer_number: int,
    slice_dimension: int,
    add_dimension: bool,
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations
    """
    logging.info("\n\nrotate and slice func")
    if model_adapter.parallel_blocks:
        logging.info("\n\nparal branch")
        rotate_and_slice_parallel(model_adapter, dataloader, slicing_scheduler, slice_layer_number, slice_dimension,
                                  add_dimension, apply_mask, final_orientation)
    else:
        logging.info("\n\nseq branch")
        rotate_and_slice_sequential(model_adapter, dataloader, slicing_scheduler, apply_mask, final_orientation)

@torch.no_grad()
def rotate_and_slice_sequential(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice the provided model, with interleaved slicing and PCA calculations.

    This method works for models where the MLP block is computed after the attention block.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype
    new_dimensions = slicing_vector_generation(len(model_adapter.get_layers()), model_adapter.hidden_size)
    logging.info(f"The dimensions will be:{new_dimensions}")

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=False)

    # rotate and slice embeddings
    eig_val, Q = pca_calc(inps, ignore_masks)
    Q = Q.to(device=config.device)
    if final_orientation == 'random':
        R = random_orthogonal_upper_left(Q.shape[0], int(new_dimensions[0]))#slicing_scheduler.get_embedding_dimensions()[0])
        Q = Q @ R.to(Q.device)
    rotate_embeddings(model_adapter, Q)

    slice_embeddings2(model_adapter, new_dimensions)
    logging.info(f"slice_embeddings out {new_dimensions}")
    #slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    logging.info("Rotate and slice layers")
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):

        ##
        logging.info(idx)
        logging.info("Hallo1")

        new_imp_emb_dimension = new_dimensions[idx]
        new_out_emb_dimension = new_dimensions[idx + 1]
        #new_out_emb_dimension = new_dimensions[idx + 1]

        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        # rotate and slice the attention inputs to match previous layer
        rotate_attention_inputs(layer_adapter, Q)
        logging.info(f"slice_attention_inputs out {int(new_imp_emb_dimension)}")
        slice_attention_inputs(layer_adapter, int(new_imp_emb_dimension)) #slicing_scheduler.get_attention_input_dimension(idx))

        # get signal between attention and mlp, rotate and slice
        logging.info(f"inputs between attention and mlp. rorate and slice {int(new_imp_emb_dimension)}")
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[
                    :, :, :  int(new_imp_emb_dimension) #slicing_scheduler.get_attention_input_dimension(idx)
                ].cpu(),
                args[i],
            )

        mlp_ln_inputs, _ = get_signals(layer_adapter, args, kwargs)
        eig_val, Q = pca_calc(mlp_ln_inputs, ignore_masks)
        Q = Q.to(device=config.device, dtype=torch.float64)

        if final_orientation == 'random':
            logging.info(f"final orientation out {int(new_out_emb_dimension)}")
            R = random_orthogonal_upper_left(
                Q.shape[0], int(new_out_emb_dimension)#slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)
            )
            Q = Q @ R.to(Q.device)
        logging.info(f"layer attn shortcut Q out {int(new_out_emb_dimension)}")
        layer.attn_shortcut_Q = nn.Parameter(
            torch.matmul(
                layer.attn_shortcut_Q,
                Q.to(dtype=dtype)[:, : int(new_out_emb_dimension)]#slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)],
            )
        )
        rotate_attention_output(layer_adapter, Q)
        logging.info(f"slice attention output out {int(new_out_emb_dimension)}")
        slice_attention_output(
            layer_adapter, int(new_out_emb_dimension) #slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)
        )

        logging.info(f"layer mlp shortcut Q out {int(new_imp_emb_dimension)} ")
        layer.mlp_shortcut_Q = nn.Parameter(
            Q.T.clone().to(dtype=dtype)[: int(new_imp_emb_dimension)] #slicing_scheduler.get_mlp_input_dimension(idx), :]
        )
        rotate_mlp_input(layer_adapter, Q)

        logging.info(f"slice mlp input out {int(new_imp_emb_dimension)} ")
        slice_mlp_input(layer_adapter, int(new_imp_emb_dimension)) # slicing_scheduler.get_mlp_input_dimension(idx))

        # Run GC and cleanup GPU memory
        cleanup_memory()

        # now compute the outputs of the current layer/inputs for the next layer
        # with slicing between Attention and mlp.
        _, inps = get_signals(layer_adapter, args, kwargs)
        eig_val, Q = pca_calc(inps, ignore_masks)
        if final_orientation == 'random':
            logging.info(f"random ort upper out {int(new_out_emb_dimension)} ")
            R = random_orthogonal_upper_left(Q.shape[0], int(new_out_emb_dimension))#slicing_scheduler.get_mlp_output_dimension(idx))
            Q = Q @ R.to(Q.device)

        logging.info(f"layer mlp shorcut Q out {int(new_out_emb_dimension)}")
        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(layer.mlp_shortcut_Q, Q.to(dtype=dtype)))

        # optionally slice the mlp/head connection in the last layer
        rotate_mlp_output(layer_adapter, Q)
        logging.info(f"slice_mlp_output out {int(new_out_emb_dimension)}")
        slice_mlp_output(layer_adapter, int(new_out_emb_dimension))#slicing_scheduler.get_mlp_output_dimension(idx))
        logging.info(f"slice_mlp_output out {int(new_out_emb_dimension)}")
        layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[:, : int(new_out_emb_dimension)]) # slicing_scheduler.get_mlp_output_dimension(idx)])

        layer.to('cpu')

        # Run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    rotate_head(model_adapter, Q)
    if slicing_scheduler.do_slice_head:
        logging.info(f"slice_mlp_output out {int(new_dimensions[-1])}")

        slice_head(model_adapter, int(new_dimensions[-1]))#slicing_scheduler.get_head_dimension())

    # update model's slicing config
    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Rotate and slice layers done")


@torch.no_grad()
def rotate_and_slice_parallel(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    slice_layer_number: int,
    slice_dimension: int,
    add_dimension: bool,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
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
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=True)

    #generate the new dimension matrix

    new_dimensions = slice_particular_layer(len(layers), model_adapter.hidden_size, slice_layer_number, slice_dimension, add_dimension)
    # new_dimensions = slicing_vector_generation(len(layers), model_adapter.hidden_size)
    logging.info(f"The dimensions will be: {new_dimensions}")
    print(f"The dimensions will be: {new_dimensions}")




    # rotate and slice embeddings
    _, Q = pca_calc(inps, ignore_masks)
    Q = Q.to(device=config.device)
    if final_orientation == 'random':
        R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q = Q @ R.to(Q.device)
    rotate_embeddings(model_adapter, Q)
    # modify the embed dimension for the first emb layer
    #slicing_scheduler.set_second_embedding_dimensions(new_dimensions[0])
    #
    #print(f"the new embedding dimension will be: {slicing_scheduler.get_embedding_dimensions()}")

    slice_embeddings2(model_adapter, new_dimensions)

    #slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    logging.info("Rotate and slice layers")
    layers = model_adapter.get_layers()
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):

        #
        logging.info(idx)
        logging.info("Helo2")
        new_imp_emb_dimension = new_dimensions[idx]
        new_out_emb_dimension = new_dimensions[idx + 1]
        """
        if idx < len(layers):
            new_out_emb_dimension = new_dimensions[idx + 1]
        else:
            new_out_emb_dimension = new_dimensions[idx]
        """
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        # rotate and slice the inputs to match previous layer (both attention and mlp)
        rotate_attention_inputs(layer_adapter, Q)
        rotate_mlp_input(layer_adapter, Q)

        #print(f"attention dimension: {new_emb_dimension}")
        slice_attention_inputs(layer_adapter, int(new_imp_emb_dimension)) #slicing_scheduler.get_attention_input_dimension(idx))

        #print(f"mlp dimension: {new_emb_dimension}")
        slice_mlp_input(layer_adapter, int(new_imp_emb_dimension)) # slicing_scheduler.get_attention_input_dimension(idx))

        # update the input signals to this layer, and re-run it
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[
                    :, :, : int(new_imp_emb_dimension)  #slicing_scheduler.get_attention_input_dimension(idx)
                ].cpu(),
                args[i],
            )

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

        if final_orientation == 'random':
            R = random_orthogonal_upper_left(Q.shape[0], int(new_out_emb_dimension)) #slicing_scheduler.get_mlp_output_dimension(idx))
            Q = Q @ R.to(Q.device)

        # update shortcut matrix
        layer.attn_shortcut_Q = nn.Parameter(torch.matmul(layer.attn_shortcut_Q, Q.to(dtype=dtype)))

        # optionally slice the mlp/head connection in the last layer
        rotate_mlp_output(layer_adapter, Q)
        rotate_attention_output(layer_adapter, Q)
        slice_mlp_output(layer_adapter, int(new_out_emb_dimension)) # slicing_scheduler.get_mlp_output_dimension(idx))
        slice_attention_output(layer_adapter, int(new_out_emb_dimension)) # slicing_scheduler.get_mlp_output_dimension(idx))

        # slice the shortcut (there is only one, we use attn_shortcut buffer)
        layer.attn_shortcut_Q = nn.Parameter(
            layer.attn_shortcut_Q[:, : int(new_out_emb_dimension)] #slicing_scheduler.get_mlp_output_dimension(idx)]
        )

        layer.to('cpu')

        # Run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    rotate_head(model_adapter, Q)
    logging.info(f"slicing head with: {new_dimensions[-1]}")
    #slice_head(model_adapter, slicing_scheduler.get_head_dimension())
    slice_head(model_adapter, new_dimensions[-1])

    if slicing_scheduler.do_slice_head:
        logging.info(f"slicing head with: {slicing_scheduler.get_head_dimension()}")
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    # update model's slicing config
    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
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
        layer.attn_shortcut_Q = nn.Parameter(torch.matmul(Q_1.clone().T, Q_3.clone()).to(device="cpu", dtype=dtype))

        # Rotate the Attention output matrix
        rotate_attention_output(layer_adapter, Q_3)

        # Rotate the MLP input
        rotate_mlp_input(layer_adapter, Q_3)

        # Set the shortcut rotation matrix of the MLP.
        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(Q_3.clone().T, Q_5.clone()).to(device="cpu", dtype=dtype))

        # Rotate MLP output
        rotate_mlp_output(layer_adapter, Q_5)

        # Run GC and cleanup GPU memory
        cleanup_memory()

        inps = outs  # The inputs to the next layer are the outputs from this one!
        Q_1 = Q_5  # first rotation in the next layer is the last one in this...

    rotate_head(model_adapter, Q_5)
    logging.info("Rotate layers done")


def slice_rotated_model(model_adapter: ModelAdapter, slicing_scheduler: SlicingScheduler | None = None) -> None:
    """
    TODO: Make this gpu memory efficient.
    """
    model_adapter.model.eval()
    layers = model_adapter.get_layers()

    if not slicing_scheduler:
        logging.info("No Slicing Scheduler")
        if model_adapter.slicing_conf.const_dimension is not None:
            # backward compatibility for when no config is available
            slicing_scheduler = ConstSlicingScheduler(model_adapter.slicing_conf.const_dimension)
            slicing_scheduler.setup(
                hidden_size=model_adapter.hidden_size,
                layers_num=len(layers),
                parallel_blocks=model_adapter.parallel_blocks,
            )
        else:
            slicing_scheduler = ConfigSlicingScheduler(model_adapter.slicing_conf)

    # slice embeddings
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    # slice layers
    for i, layer_adapter in enumerate(layers):
        layer = layer_adapter.layer
        # slice attn weights 2nd dim, attn shortcut 1st dim
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(i))

        logging.info(i)
        logging.info("Helo3")

        # slice mlp input 2nd dimension
        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(i))

        # slice mlp shortcut 1st dimension
        # slice mlp shortcut
        if not model_adapter.parallel_blocks:
            layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[: slicing_scheduler.get_mlp_input_dimension(i), :])

        # slice mlp weights 1st dimension
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(i))

        if model_adapter.parallel_blocks:  # parallel case
            layer.attn_shortcut_Q = nn.Parameter(
                layer.attn_shortcut_Q[:, : slicing_scheduler.get_attention_output_dimension(i, match_head_dim=True)]
            )
            slice_attention_output(
                layer_adapter, slicing_scheduler.get_attention_output_dimension(i, match_head_dim=True)
            )
        else:  # sequential case
            layer.attn_shortcut_Q = nn.Parameter(
                layer.attn_shortcut_Q[:, : slicing_scheduler.get_attention_output_dimension(i, match_head_dim=False)]
            )
            layer.mlp_shortcut_Q = nn.Parameter(
                layer.mlp_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(i)]
            )

            # slice attention weights 1st dimension
            slice_attention_output(
                layer_adapter, slicing_scheduler.get_attention_output_dimension(i, match_head_dim=False)
            )

    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())


def random_orthogonal_upper_left(total_dim, upper_block_dim):
    """
    Create a square matrix where the upper left block is a random orthogonal matrix, and the remainder is the identity.
    """
    A = np.random.rand(upper_block_dim, upper_block_dim)
    Q, _ = np.linalg.qr(A)
    R = np.eye(total_dim)
    R[:upper_block_dim, :upper_block_dim] = Q
    return torch.from_numpy(R)


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
            X_batch[ignore_masks[idx] == 0] = 0

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
    return eig_val, eigen_vec
