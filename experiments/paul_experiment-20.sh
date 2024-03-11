#!/bin/bash

# The base command without the changing parameters
base_command="python3.11 run_slicegpt_perplexity.py --model microsoft/phi-2 --save-dir /storage/paulclotan/SmartSliceGPT/save_work2 --sparsity 0.25 --no-wandb --device cuda:0 --hf-token hf_ucILpakjSHaCoAHxZkBwPQLGdGorpzhcqvclear"

# Define the output file
output_file="/storage/paulclotan/SmartSliceGPT/experiments/output-20.txt"

# Make sure the output file is empty before we start appending
> "$output_file"

# Run the initial command with --slice-dimension 0 and --add-dimension True
#echo -e "Running initial command with --slice-dimension 0 --add-dimension True -> Initial perplexity determination" >> "$output_file"
#eval $base_command --slice-layer 1 --slice-dimension 0 --add-dimension True >> "$output_file"
sleep 2

# Loop 32 times for each slice layer
for i in {0..33}; do
    # Echo a header for the iteration with True
    echo -e "Running with --slice-layer $i --slice-dimension 20 --add-dimension True" >> "$output_file"
    # Append the command output with --slice-layer as the loop counter, --slice-dimension 50, and --add-dimension True to the output file
    eval $base_command --slice-layer $i --slice-dimension 20 --add-dimension True >> "$output_file"

    sleep 2
    # Echo a header for the iteration with False
    echo -e "\nRunning with --slice-layer $i --slice-dimension 20 --add-dimension False" >> "$output_file"
    # Append the command output with --slice-layer as the loop counter, --slice-dimension 50, and --add-dimension False to the output file
    eval $base_command --slice-layer $i --slice-dimension 20 >> "$output_file"
    sleep 2

done