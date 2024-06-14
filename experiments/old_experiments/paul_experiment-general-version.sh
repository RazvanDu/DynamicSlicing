#!/bin/bash

# Check if exactly 4 arguments are passed
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <model_name> <slice_dimension> <dataset> <gpu_device>"
    exit 1
fi

# Assign command line arguments to variables
model_name=$1
slice_dimension=$2
dataset=$3
gpu_device=$4

# Specify your hf_token directly in the script
hf_token="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"

# The base command with the changing parameters as variables and the hf_token included
base_command="python3.11 run_slicegpt_perplexity.py --model $model_name --save-dir /storage/paulclotan/SmartSliceGPT/save_work2 --sparsity 0.25 --no-wandb --device $gpu_device --cal-dataset $dataset --hf-token $hf_token"

# Define the output file
model_name_save=${model_name//\//-}
output_file="/storage/paulclotan/SmartSliceGPT/experiments/output-${slice_dimension}-${model_name_save}-${dataset}.txt"

# Make sure the output file is empty before we start appending
> "$output_file"

# Append experiment details to the output file
echo "Experiment Details:" >> "$output_file"
echo "Model being evaluated: $model_name" >> "$output_file"
echo "Cut dimension: $slice_dimension" >> "$output_file"
echo "Output folder: /storage/paulclotan/SmartSliceGPT/save_work2" >> "$output_file"
echo "Dataset: $dataset" >> "$output_file"
echo "GPU Device: $gpu_device" >> "$output_file"
echo "-----------------------------------" >> "$output_file"
echo "" >> "$output_file"

echo -e "\n\n"
# Echo a header for the iteration with True
echo -e "Baseline ppl for default cut: ">> "$output_file"
eval $base_command --slice-dimension 0 --add-dimension True >> "$output_file"

# Loop 32 times for each slice layer
for i in {0..40}; do
    echo -e "\n\n"
    # Echo a header for the iteration with True
    echo -e "\nRunning with --slice-layer $i --slice-dimension $slice_dimension --add-dimension True" >> "$output_file"
    # Append the command output with --slice-layer as the loop counter, --slice-dimension from variable, and --add-dimension True to the output file
    eval $base_command --slice-layer $i --slice-dimension $slice_dimension --add-dimension True >> "$output_file"

    sleep 2
    # Echo a header for the iteration with False
    echo -e "\nRunning with --slice-layer $i --slice-dimension $slice_dimension --add-dimension False" >> "$output_file"
    # Append the command output with --slice-layer as the loop counter, --slice-dimension from variable, and --add-dimension False to the output file
    eval $base_command --slice-layer $i --slice-dimension $slice_dimension >> "$output_file"
    sleep 2

done


