#!/bin/bash

# Check if exactly 4 arguments are passed
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <model_name> <slice_percentage> <dataset> <gpu_device>"
    exit 1
fi

# Assign command line arguments to variables
model_name=$1
slice_percentage=$2
dataset=$3
gpu_device=$4

# Specify your hf_token directly in the script
hf_token="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"

# The base command with the changing parameters as variables and the hf_token included
base_command="python3.11 run_slicegpt_perplexity.py --model $model_name --save-dir /storage/paulclotan/SmartSliceGPT/save_work2 --sparsity 0.25 --no-wandb --device $gpu_device --cal-dataset $dataset --hf-token $hf_token"

# Define the output file
model_name_save=${model_name//\//-}
output_file="/storage/paulclotan/SmartSliceGPT/experiments/experiment-output-folder/output-${slice_percentage}-${model_name_save}-${dataset}.txt"

# Make sure the output file is empty before we start appending
> "$output_file"

# Append experiment details to the output file
echo "Experiment Details:" >> "$output_file"
echo "Model being evaluated: $model_name" >> "$output_file"
echo "Cut percentage: $slice_percentage %" >> "$output_file"
echo "Output folder: /storage/paulclotan/SmartSliceGPT/save_work2" >> "$output_file"
echo "Dataset: $dataset" >> "$output_file"
echo "GPU Device: $gpu_device" >> "$output_file"
echo "-----------------------------------" >> "$output_file"
echo "" >> "$output_file"


# Echo a header for the iteration with True
echo -e "Baseline ppl for default cut: ">> "$output_file"
eval $base_command --slice-percentage 0 >> "$output_file"

# Loop 32 times for each slice layer
for i in {0..40}; do
    # Echo a header for the iteration with True
    echo -e "\n\nRunning with --slice-layer $i --slice-percentage: $slice_percentage --add-dimension True" >> "$output_file"
    # Append the command output with --slice-layer as the loop counter, --slice-dimension from variable, and --add-dimension True to the output file
    eval $base_command --slice-layer $i --slice-percentage $slice_percentage >> "$output_file"

    sleep 2

done


