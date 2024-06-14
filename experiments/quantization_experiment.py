import numpy as np
import subprocess
import shlex
import argparse
import re
import os


# Function to run the command and return the perplexity
def run_command(command):
    command_to_run = shlex.split(command)
    result = subprocess.run(command_to_run, capture_output=True, text=True)
    output = result.stdout
    perplexity_match = re.search(r"After rotating and slicing (\d+\.\d+)", output)
    print(perplexity_match)
    if perplexity_match:
        return perplexity_match.group(1)
    else:
        return "Error: Perplexity not found."


# Function to parse the command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Run quantization experiments with given parameters.')
    parser.add_argument('--model', type=str, required=True, help='Model to use.')
    parser.add_argument('--cuda-device', type=str, required=True, help='CUDA device to use.')
    parser.add_argument('--cal-dataset', type=str, required=True, help='Dataset on which to perform calibration.')
    return parser.parse_args()


args = parse_args()

# The string for vector-cut with 50 ones
vector_cut = "1 " * 50

# Create the output directory including the model and dataset
output_dir = f"/experiments/old_experiments/quantization/"
os.makedirs(output_dir, exist_ok=True)

# Construct the output file path with the dataset in the filename
base_filename = f"quantization_results_{args.model.replace('/', '-')}_{args.cal_dataset}_flexible_bits2"
save_path = os.path.join(output_dir, base_filename + ".txt")

with open(save_path, 'w') as file:
    file.write(f"Experiment Summary:\nModel: {args.model}\nCUDA Device: {args.cuda_device}\nCalibration Dataset: {args.cal_dataset}\n\n")
    file.flush()


    # Add the base perplexity line with all scaling parameters set to 1
    base_command = (f"python3.11 run_slicegpt_quantization.py --model {args.model} --save-dir "
                    f"/storage/paulclotan/SmartSliceGPT/save_work2 --sparsity 0.25 --no-wandb "
                    f"--hf-token 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 --vector-cut {vector_cut} "
                    f"--cal-dataset {args.cal_dataset} --device {args.cuda_device} "
                    f"--quantization-limit-one 1 --quantization-limit-two 1 "
                    f"--quant-second-zone-percent 1 --quant-third-zone-percent 1")
    #base_perplexity = run_command(base_command)
    #file.write(f"Base Perplexity: {base_perplexity}\n\n")
    #file.flush()

    # Iterate over the ranges for the parameters and run the command
    for limit_one in [0.2]:
        file.write("\n\n\n\n")

        for limit_two in [0.2, 0.4, 0.6, 0.8]:

            if limit_one > limit_two:
                continue
            file.write("\n\n")
            for second_zone in [16]:
                file.write("\n")
                for third_zone in [8]:
                    if second_zone > third_zone:

                        command = (f"python3.11 run_slicegpt_quantization.py --model {args.model} --save-dir "
                                   f"/storage/paulclotan/SmartSliceGPT/save_work2 --sparsity 0.25 --no-wandb "
                                   f"--hf-token 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 --vector-cut {vector_cut} "
                                   f"--cal-dataset {args.cal_dataset} --device {args.cuda_device} "
                                   f"--quantization-limit-one {limit_one} --quantization-limit-two {limit_two} "
                                   f"--quant-second-zone-percent {second_zone} --quant-third-zone-percent {third_zone}")
                        perplexity = run_command(command)
                        file.write(f"Params: Limit One: {limit_one}, Limit Two: {limit_two}, "
                                   f"Second Zone: {second_zone}, Third Zone: {third_zone}, "
                                   f"Perplexity: {perplexity}\n")
                        file.flush()


