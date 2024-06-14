# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os



os.environ["WANDB_SERVICE_WAIT"] = "300"
os.environ['TRANSFORMERS_CACHE'] = '/storage/paulclotan/SmartSliceGPT/models'


import os
os.environ["WANDB_SERVICE_WAIT"] = "300"
os.environ['TRANSFORMERS_CACHE'] = '/storage/paulclotan/SmartSliceGPT/models'

import argparse
import logging
import os
import torch
import wandb
import sys
import random


#slice gpt dependency not found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate, utils
from slicegpt.config import config
from slicegpt.layerwrapper import WrappedGPT


from datasets import load_dataset


from torch import nn


import json
import lm_eval
from lm_eval import tasks
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM

utils.configure_logging()



def argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="OPT model to load; pass `facebook/opt-125m`.",
        choices=[
            # OPT models
            "facebook/opt-125m",
            "facebook/opt-1.3b",
            "facebook/opt-2.7b",
            "facebook/opt-6.7b",
            "facebook/opt-13b",
            "facebook/opt-30b",
            "facebook/opt-66b",
            # LLAMA 2 Models
            'meta-llama/Llama-2-7b-hf',
            'meta-llama/Llama-2-13b-hf',
            'meta-llama/Llama-2-70b-hf',
            'togethercomputer/Llama-2-7B-32K-Instruct',
            'Secbone/llama-2-13B-instructed',
            #LLAMAmodels
            'meta-llama/Meta-Llama-3-8B',
            'meta-llama/Meta-Llama-3-8B-Instruct',
            # Phi-2 model
            'microsoft/phi-2',
            # mistral
            'mistralai/Mistral-7B-v0.1',
            'mistralai/Mistral-7B-Instruct-v0.2',
            'mistralai/Mistral-7B-v0.2',
            'mistralai/Mistral-7B-v0.3',
            'mistralai/Mistral-7B-Instruct-v0.1',


        ],
        default="facebook/opt-125m",
    )
    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument(
        "--cal-dataset",
        type=str,
        help="Dataset to calibrate and calculate perplexity on.",
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default="wikitext2",
    )
    parser.add_argument(
        "--cal-nsamples",
        type=int,
        help="Number of samples of the calibration data to load.",
        default=128,
    )
    parser.add_argument("--cal-batch-size", type=int, default=1, help="Batch size for loading the calibration data.")
    parser.add_argument(
        "--cal-max-seqlen", type=int, default=2048, help="Maximum sequence length for the calibration data."
    )
    parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument(
        "--round-interval",
        type=int,
        default=8,
        help="Interval for rounding the weights (the best value may depend on your hardware)",
    )
    parser.add_argument(
        "--ppl-eval-seqlen", type=int, default=2048, help="Sequence length for evaluating the perplexity."
    )
    parser.add_argument("--ppl-eval-batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
    parser.add_argument(
        "--ppl-eval-nsamples", type=int, default=128, help="Number of samples to evaluate the perplexity on."
    )
    parser.add_argument("--eval-baseline", action="store_true", help="Evaluate the baseline model.")
    parser.add_argument("--eval-fused-model", action="store_true", help="Evaluate the fused model.")
    parser.add_argument("--ppl-only", action="store_true", help="Evaluate the loaded model without doing compression.")
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )

    parser.add_argument("--save-dir", type=str, default=None, help="Path to save the model.")
    parser.add_argument("--load-model-path", type=str, default=None, help="Path to load the sliced model from.")
    parser.add_argument("--cov-limit", type=float, default=1.0, help="Covariance limit.")

    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))

    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.",
    )

    # add arguments to set the slicing size and what layer we are currently slicing

    parser.add_argument("--slice-layer", type=int, default=1, help="The layer we are currently slicing.")
    parser.add_argument("--slice-percentage", type=float, default=0,
                        help="Percentage we cut from the original model")

    parser.add_argument("--accuracy-limit", type=int, default=-1,
                        help="The limit used in the evaluation of the results.")


    #parser.add_argument("--slice-dimension", type=int, default=50,
    #                    help="The dimension we are adding/ reducing from that certain layer")
    #parser.add_argument("--add-dimension", type=bool, default=False,
    #                    help="Default: the amount is subtracted. Add the param: True, to add dimension")

    #parse the vector
    parser.add_argument(
        "--vector-cut",
        type=float,
        nargs='+',
        help="Vector of integers, representing the new cutting dimensions",
        default=[0]  # Default to a vector containing a single zero, adjust as necessary
    )

    parser.add_argument("--single-layer-cut", type=int, default=0,
                        help="Two cutting modes. 0- cut based on vector-instance of cut percentage"
                            "1 - the cut is done for only 1 layer, for a given layer nr and percentage")

    # zero-shot-task arguments
    parser.add_argument(
        '--tasks',
        nargs='+',
        default = None,
        choices=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "boolq", "gsm8k_cot", "mlqa_en", "xlsum_en", "mmlu", "social_iqa"],
    )

    parser.add_argument('--num-fewshot', type=int, default=0, help="Number of fewshots for all tasks.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluating with lm eval harness.")

    parser.add_argument("--metric-to-use", type=int, default=1, help="0 for skewness, 1 for coefficient of vairation.")

    args = parser.parse_args()

    logging.debug(f'Parsed arguments:')
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')

    if not 0 <= args.sparsity < 1:
        raise argparse.ArgumentTypeError(f"Sparsity should be in the range [0, 1)")

    if not 0 <= args.slice_percentage < 1:
        raise argparse.ArgumentTypeError(f"The cut percentage must be in the interval [0,1)")

    if args.device:
        config.device = torch.device(args.device)

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    else:
        raise argparse.ArgumentTypeError(f"Data type should be one of 'fp16', 'fp32'")

    return args

#####functii

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids



def get_c4(nsamples, seed, seqlen, tokenizer, device):
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', 'en', split='train[10:20]', cache_dir="/storage/paulclotan/SmartSliceGPT/datasets")
    valdata = load_dataset('allenai/c4', 'en', split='validation[10:20]', cache_dir="/storage/paulclotan/SmartSliceGPT/datasets")

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

def prepare_calibration_input(model, dataloader, device, model_name):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]

    # TODO: !!! ENABLE THIS IF WE END UP USING BIG MODELS/DEVICE MAPS SOMEHOW!

    #if "model.embed_tokens" in model.hf_device_map:
    #    device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype

    print("TESTING", model.seqlen, model.config.hidden_size)

    # TODO: WHY IS THIS DIVIDED BY 2???

    if model_name == "meta-llama/Llama-2-7b-hf":
        model_seqlen = int(model.seqlen/2)
    else:
        model_seqlen = 2048


    inps = torch.zeros((128, model_seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids

def get_loaders(name, device, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    #if 'wikitext2' in name:
    #    return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    #if "c4" in name:
    return get_c4(nsamples, seed, seqlen, tokenizer, device)
def wanda_data(model, tokenizer, model_name, device):
    print("loading calibration data")
    dataloader, _ = get_loaders("c4", device, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, model_name)
    return inps, outs, attention_mask, position_ids

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

#####
def main() -> None:
    logging.info("Running SliceGPT perplexity experiment")

    #initialize_tasks()

    # TODO: ADAPT THIS!
    nsamples = 128

    args = argparser()
    '''
    print(f"current argunemts are: layer: {args.slice_layer} and type {type(args.slice_layer)}"
          f"\n with the slicing dimension {args.slice_dimension} "
          f"\n with the add_dimension: {args.add_dimension}"
          f"\nrunning on the device: {config.device}"
          f"\nmodel: {args.model}"
          f"\ndataset: {args.cal_dataset}")
    '''
    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    try:
        wandb.init(project="slicegpt", config=args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project="slicegpt", mode='disabled')

    if args.load_model_path:
        # load the model from load_model_path to compute perplexity and skip rotation and slicing
        logging.info(f"Loading sliced {args.model} model from {args.load_model_path} with sparsity {args.sparsity}")
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            args.model, args.load_model_path, args.sparsity, args.hf_token
        )
    else:
        # load one of the pre-trained modelspython3.11 mean_cut_experiment_accuracy_cov_auto.py --model mistralai/Mistral-7B-v0.1 --source-for-vector wikitext2 --cuda-device cuda:3 --dataset wikitext2 --tasks hellaswag winogrande arc_easy piqa mmlu arc_challenge --accuracy-limit 100 --cov-limit -1.0
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(args.model, token=args.hf_token, dtype=config.dtype)

    print("MODEL", model_adapter)
    print("MODEL", model_adapter.model)
    print("MODEL", model_adapter.model.model)
    print("MODEL", model_adapter.model.model.layers)

    part_model = model_adapter.model.to(args.device)
    model = part_model.model

    """
    # idk what is happening. please help --- ;( help is coming!
    inps, outs, attention_mask, position_ids = wanda_data(part_model, tokenizer, args.model, device=args.device)
    #.to(args.device)
    layers = model.layers

    layer_scores_original = []
    layer_scores_modified = []

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)


        # TODO: FIX IF MULTIPLE GPU STUFF
        #if f"model.layers.{i}" in model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
        #    dev = model.hf_device_map[f"model.layers.{i}"]
        #    inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(
        #        dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])#, device = args.device)#, device = args.device)

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        avgg_original = 0
        avgg_modified = 0
        saved = 0

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1)))

              
            if 'down_proj' in name:
                mean = torch.mean(torch.abs(subset[name].weight.data))
                std = torch.std(torch.abs(subset[name].weight.data), dim=None)
                saved = std / mean
                continue
            
            #prnt(f"\n\nThe W metric is: {W_metric}")

            mean = torch.mean(torch.abs(subset[name].weight.data))
            std = torch.std(torch.abs(subset[name].weight.data), dim=None)
            cv = std / mean
            avgg_original += cv
            #print(f"\nORIGINAL MATRIX: The std is: {std}, the mean is: {mean}, the CoV is: {cv}\n")
            mean = torch.mean(torch.abs(W_metric))
            std = torch.std(torch.abs(W_metric), dim=None)
            cv = std / mean
            avgg_modified += cv
            #print(f"MODIFIED MATRIX: The std is: {std}, the mean is: {mean}, the CoV is: {cv}\n")

        avgg_original /= (len(subset))
        avgg_modified /= (len(subset))

        print(f"pruning layer {i}\n")
        #print(f"ORIGINAL LAYER {avgg_original}\n")
        if args.cov_limit > 0:
            print(f"\n\n\nSkewness for each column: {min(float(args.cov_limit), avgg_modified)}")
            print(f"\nThe current max covariance is: {args.cov_limit}")
        else:
            print(f"\n\n\nSkewness for each column: {avgg_modified}")
            print(f"\nThe covariance is unlimited")
        #print(f"\n\n\nSkewness for each column: {saved}")
        print(f"PART 1 - {avgg_modified}\n")
        #print(f"PART 2 - {saved}\n")

        #layer_scores_original.append((avgg_original.item(), i))
        layer_scores_modified.append((min(2, avgg_modified.item()), i))

    layer_scores_original.sort()
    layer_scores_original.reverse()
    layer_scores_modified.sort()
    layer_scores_modified.reverse()

    print("ORIGINAL", layer_scores_original)
    print("MODIFIED", layer_scores_modified)

    layer_scores_original = [value[1] for value in layer_scores_original]
    layer_scores_modified = [value[1] for value in layer_scores_modified]

    print("ORIGINAL LAYERS", layer_scores_original)
    print("MODIFIED LAYERS", layer_scores_modified)
    """
    print(f"new cutting dimensions are {args.vector_cut}")
    def reset_model_device() -> None:
        if args.distribute_model:
            # distribute model across available GPUs
            gpu_utils.distribute_model(model)
        else:
            model.to(config.device)

    dataset = data_utils.get_dataset(args.cal_dataset)
    train_dataset, test_dataset = dataset["train"], dataset["validation"]
    train_loader = data_utils.prepare_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_seqlen=args.cal_max_seqlen,
        batch_size=args.cal_batch_size,
        nsamples=args.cal_nsamples,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )
    test_loader = data_utils.prepare_dataloader(
        dataset=test_dataset,
        tokenizer=tokenizer,
        max_seqlen=args.ppl_eval_seqlen,
        batch_size=args.ppl_eval_batch_size,
        nsamples=args.ppl_eval_nsamples,
        seed=args.seed,
    )

    # evaluate perplexity and exit if sliced model is loaded or if ppl_only is set
    if args.load_model_path or args.ppl_only:
        reset_model_device()
        dataset_ppl = gpu_utils.evaluate_ppl(model_adapter, test_loader)
        logging.info(f'Loaded model perplexity: {dataset_ppl}')
        wandb.log({"original_ppl": dataset_ppl})
        return

    # original ppl
    if args.eval_baseline:
        reset_model_device()
        dataset_ppl = gpu_utils.evaluate_ppl(model_adapter, test_loader)
        logging.info(f'Original ppl: {dataset_ppl:.4f}')
        wandb.log({"original_ppl": dataset_ppl})
        model.cpu()
        utils.cleanup_memory()

    # added this to solve different device problem
    model_adapter.model.to(config.device)

    # replace modules with compressible equivalents
    layernorm_fusion.replace_layers(model_adapter)

    # fuse layernorms and add rotations to skip connections
    layernorm_fusion.fuse_modules(model_adapter)

    # don't run this on large and/or distributed models
    if args.eval_fused_model and not args.distribute_model:
        model.to(config.device)

        dataset_ppl = gpu_utils.evaluate_ppl(model_adapter, test_loader)
        logging.info(f'Post-fusion: {dataset_ppl:.4f}')
        wandb.log({"post_fusion_ppl": dataset_ppl})

        model.cpu()

        # run GC and cleanup GPU memory
        utils.cleanup_memory()

    original_param_count = sum(int(p.nelement()) for p in model.parameters())
    logging.info(f'Original model parameters: {original_param_count:,d}')

    # compute new embedding dimension given the desired sparsity level
    new_embedding_dimension = int((1 - args.sparsity) * model_adapter.hidden_size)
    # round (down) to the nearest multiple of round_interval
    new_embedding_dimension = new_embedding_dimension - (new_embedding_dimension % args.round_interval)
    #logging.info(
    #    f"New embedding dimension: {new_embedding_dimension} (sparsity {100*(1 - new_embedding_dimension / model_adapter.hidden_size):.4f} %)"
    #)

    ignore_tokens = [tokenizer.pad_token_id]

    rotate.rotate_and_slice(model_adapter, train_loader, args.vector_cut,
                            args.slice_layer, args.slice_percentage, new_embedding_dimension,
                            args.single_layer_cut, args.metric_to_use, ignore_tokens=ignore_tokens)
    #rotate.rotate_and_slice(model_adapter, train_loader, args.slice_layer, args.slice_dimension,
    #                        args.add_dimension, new_embedding_dimension, ignore_tokens=ignore_tokens)
    # used to cut layer by layer,+- a given quantity
    """"""
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        model_file = os.path.join(args.save_dir, os.path.basename(args.model) + "_" + str(args.sparsity) + ".pt")
        torch.save(model.state_dict(), model_file)
        logging.info(f"Saved sliced model to {args.save_dir}")

    reset_model_device()

    model_adapter.model.to(config.device)


    dataset_ppl = gpu_utils.evaluate_ppl(model_adapter, test_loader)
    logging.info(f'After rotating and slicing {dataset_ppl:.4f}')
    print(f'After rotating and slicing {dataset_ppl:.4f}')
    wandb.log({"sliced_ppl": dataset_ppl})
    
    sliced_param_count = sum(int(p.nelement()) for p in model.parameters())
    sliced_fraction = 1.0 - sliced_param_count / original_param_count
    logging.info(f'Sliced model parameters: {sliced_param_count:,d} (sliced fraction {sliced_fraction:.4f})')
    print(f'Sliced model parameters: {sliced_param_count:,d} (sliced fraction {sliced_fraction:.4f})')



    ### LM Eval Harness ###
    # the lm eval harness ties the weights, but this should not be done for sliced models unless the lm_head was sliced
    # necesary for opt to run. do not understand why atm.
    model_adapter.model.tie_weights = lambda: None
    model_adapter.model.to(config.device)


    print(f"\n\n\nThe model adap is: {model_adapter}")
    print(f"\nThe model is: {model_adapter.model}")
    print(f"\nThe device is: {config.device}")



    #model_adapter.model.eval()
    #setting the tasks here
    #args.tasks = ["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande"]
    hflm = HFLM(pretrained=model_adapter.model, tokenizer=tokenizer)# , batch_size=args.batch_size) , device = args.device

    #print(f"\n\n\n The tasts are: {args.tasks}\n\n")
    if args.tasks is None:
        task_names = None
        print(f"\n\n\n The tasks are : {task_names}. Stopping.")
        exit()
    else:
        print(f"\n\n\n The function.. has the output: {lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)}")
        task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)
    task_names = args.tasks
    #print(f"\n\n\n The tasts are after if: {task_names}\n\n")


    if args.accuracy_limit < 0:
        results = lm_eval.simple_evaluate(hflm, tasks=task_names)[
            # batch_size=args.batch_size, num_fewshot=args.num_fewshot) [
            'results'
        ]
        print(f"No limit here")
    else:
        print(f"\n Evaluating with the limit {args.accuracy_limit}")
        results = lm_eval.simple_evaluate(hflm, tasks=task_names, limit=args.accuracy_limit)[ #batch_size=args.batch_size, num_fewshot=args.num_fewshot) [
            'results'
        ]

    #logging.info(json.dumps(results, indent=2))



    '''
    used to compute the average accuracy. not needed atm. should be most likely removed in the future
    def calculate_avg_accuracy(task_names, results):
        n_tasks = len(task_names)
        acc_cumul = sum(
            result.get('acc_norm,none', result['acc,none']) for task, result in results.items() if 'mmlu' not in task
        )

        questions_per_mmlu_task = {
            task_name: lm_eval.tasks.get_task_dict([task_name])[task_name].dataset["test"].num_rows
            for task_name in task_names
            if 'mmlu' in task_name
        }

        if not questions_per_mmlu_task:
            return acc_cumul / n_tasks

        # Calculate average accuracy for mmlu tasks, weighted by number of questions in each task
        acc_mmlu = sum(
            result.get('acc_norm,none', result['acc,none']) * questions_per_mmlu_task[task]
            for task, result in results.items()
            if 'mmlu' in task
        )
        acc_mmlu_avg = acc_mmlu / sum(questions_per_mmlu_task.values())
        wandb.log({'acc_mmlu_avg': acc_mmlu_avg})

        return (acc_cumul + acc_mmlu_avg) / (n_tasks - len(questions_per_mmlu_task) + 1)


    acc_avg = calculate_avg_accuracy(task_names, results)
    
    wandb.log({'acc_avg': acc_avg})
    logging.info(f"Average accuracy across tasks: {acc_avg}.")
    #print(f"\nAverage accuracy across tasks: {acc_avg}.")
    
    
    '''
    print(f"\n\nThe accuracy results are: {results}\n\n")





if __name__ == "__main__":
    main()
