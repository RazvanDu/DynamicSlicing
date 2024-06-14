# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
os.environ["WANDB_SERVICE_WAIT"] = "300"
os.environ['TRANSFORMERS_CACHE'] = '/storage/paulclotan/SmartSliceGPT/models'


import argparse
import logging
import os

import torch

import json

import lm_eval
import wandb
from lm_eval import tasks
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM

from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, utils, quantize
from slicegpt.config import config

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
            # Phi-2 model
            'microsoft/phi-2',
        ],
        default="facebook/opt-125m",
    )
    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp32")
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
    # parser.add_argument("--slice-dimension", type=int, default=50,
    #                    help="The dimension we are adding/ reducing from that certain layer")
    # parser.add_argument("--add-dimension", type=bool, default=False,
    #                    help="Default: the amount is subtracted. Add the param: True, to add dimension")

    # parse the vector
    parser.add_argument(
        "--vector-cut",
        type=float,
        nargs='+',
        help="Vector of integers, representing the new cutting dimensions",
        default=[0]  # Default to a vector containing a single zero, adjust as necessary
    )

    parser.add_argument("--quantization-limit-one", type=float, default=1.0, help="The quantization limit between the first and second area [should be between 0 and 1].")
    parser.add_argument("--quantization-limit-two", type=float, default=1.0, help="The quantization limit between the second and third area. [Should be between 0 and 1]")

    parser.add_argument("--quant-second-zone-percent", type=float, default=1.0,
                        help="Used as a int. ex- value 2 means that the type of this zone, will for example be from float16 -> float8")
    parser.add_argument("--quant-third-zone-percent", type=float, default=1.0,
                        help="Used as a int. ex- value 2 means that the type of this zone, will for example be from float16 -> float8")

    # zero-shot-task arguments
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande"],
        choices=lm_eval_utils.MultiChoice(tasks.ALL_TASKS),
    )
    parser.add_argument('--num-fewshot', type=int, default=0, help="Number of fewshots for all tasks.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluating with lm eval harness.")




    args = parser.parse_args()

    logging.debug(f'Parsed arguments:')
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')

    if not 0 <= args.sparsity < 1:
        raise argparse.ArgumentTypeError(f"Sparsity should be in the range [0, 1)")

    for arg in [args.quantization_limit_one, args.quantization_limit_two]:
        if not (0 <= arg <= 1):
            raise ValueError(f"All parameters should be between 0 and 1. Invalid value: {arg}")

        # Check if quantization-limit-one is less than quantization-limit-two
    #if args.quantization_limit_one > args.quantization_limit_two:
    #    raise ValueError("quantization-limit-one should be less than quantization-limit-two")

        # Check if quant-second-zone-percent is less than quantization-third-zone-percent
    #if args.quant_second_zone_percent > args.quant_third_zone_percent:
    #    raise ValueError("quant-second-zone-percent should be less than quantization-third-zone-percent")


    if args.device:
        config.device = torch.device(args.device)

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    else:
        raise argparse.ArgumentTypeError(f"Data type should be one of 'fp16', 'fp32'")

    return args


def main() -> None:
    logging.info("Running SliceGPT perplexity experiment")
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
        # load one of the pre-trained models
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(args.model, token=args.hf_token, dtype=config.dtype)

    model = model_adapter.model
    def reset_model_device() -> None:
        if args.distribute_model:
            # distribute model across available GPUs
            gpu_utils.distribute_model(model_adapter)
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

    # replace modules with compressible equivalents
    # replace modules with compressible equivalents
    layernorm_fusion.replace_layers(model_adapter)

    # fuse layernorms and add rotations to skip connections
    layernorm_fusion.fuse_modules(model_adapter)

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


    ignore_tokens = [tokenizer.pad_token_id]
    quantize.quantize(model_adapter, train_loader, args.vector_cut,
                      args.slice_layer, args.slice_percentage,
                      args.quantization_limit_one, args.quantization_limit_two,
                      args.quant_second_zone_percent, args.quant_third_zone_percent,
                      ignore_tokens=ignore_tokens)


    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        model_file = os.path.join(args.save_dir, os.path.basename(args.model) + "_" + str(args.sparsity) + ".pt")
        torch.save(model.state_dict(), model_file)
        logging.info(f"Saved sliced model to {args.save_dir}")

    reset_model_device()
    dataset_ppl = gpu_utils.evaluate_ppl(model_adapter, test_loader)
    logging.info(f'After rotating and slicing {dataset_ppl:.4f}')
    print(f'After rotating and slicing {dataset_ppl:.4f}')
    wandb.log({"sliced_ppl": dataset_ppl})

    sliced_param_count = sum(int(p.nelement()) for p in model.parameters())
    sliced_fraction = 1.0 - sliced_param_count / original_param_count
    logging.info(f'Sliced model parameters: {sliced_param_count:,d} (sliced fraction {sliced_fraction:.4f})')
    print(f'Sliced model parameters: {sliced_param_count:,d} (sliced fraction {sliced_fraction:.4f})')



    ### LM Eval Harness ###


    hflm = HFLM(pretrained=model_adapter.model, tokenizer=tokenizer, batch_size=args.batch_size)

    print(f"\n\n\n The tasts are: {args.tasks}\n\n")
    if args.tasks is None:
        task_names = tasks.ALL_TASKS
        print(f"\n\n\n The tasks.all_tasks is: {tasks.ALL_TASKS}")
    else:
        print(f"\n\n\n The function.. has the output: { lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)}")
        task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)

    print(f"\n\n\n The tasts are after if: {task_names}\n\n")

    results = lm_eval.simple_evaluate(hflm, tasks=task_names, num_fewshot=args.num_fewshot, batch_size=args.batch_size)[
        'results'
    ]
    logging.info(json.dumps(results, indent=2))


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
    logging.info(f"Average accuracy across tasks: {acc_avg}")



if __name__ == "__main__":
    main()
