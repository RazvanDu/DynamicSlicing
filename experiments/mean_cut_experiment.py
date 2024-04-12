import numpy as np
import subprocess
import shlex
import argparse
import re
import os

cut_mean = 0.3

def parse_args():
    parser = argparse.ArgumentParser(description='Run mean cut experiment with given parameters.')
    print("Usage:  --model <model_name> --cuda_device <gpu_device ex: cuda:0> --vector-cut <cut_dimensions> --source for vector <source_name> --dataset <dataset_name>")
    parser.add_argument('--model', type=str, default='microsoft/phi-2', help='Model to use.')
    parser.add_argument('--source-for-vector', type=str, default='wikitext2', help='Source from where the vector-cut is taken from')
    parser.add_argument('--cut-percent', type=str, default='80%',
                        help='The percent for the cut-vector used to generate it')
    parser.add_argument('--dataset', type=str, default='wikitext2', help='Dataset on which we evaluate.')
    parser.add_argument('--cuda-device', type=str, default='cuda:0', help='CUDA device to use.')
    parser.add_argument(
        "--vector-cut",
        type=float,
        nargs='+',
        help="Vector of floats, representing the new cutting dimensions",
        default=[0]  # Default to a vector containing a single zero, adjust as necessary
    )
    return parser.parse_args()


args = parse_args()
# perplexity values that give the form of the graph - to be computed with percentage-cut-experiment.py


model_name_save = args.model.replace("/", "-")
print(model_name_save)

save_path = (f"/storage/paulclotan/SmartSliceGPT/experiments/experiment-output-folder/perplexity_results_{model_name_save}_"
             f"{args.dataset}_on_{args.source_for_vector}_{args.cut_percent}_cut_percent.txt")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, 'w') as file:
    file.write(
        f"Experiment Summary:"
        f"\nModel: {args.model}"
        f"\nDataset: {args.dataset}"
        f"\nSource for Vector-Cut: {args.source_for_vector}"
        f"\nThe cut vector was realized by cutting {args.cut_percent}%"
        f"\nVector: {args.vector_cut}"
        f"\nCUDA Device: {args.cuda_device}\n\n")


    # Adjust the mean of the graph
    for amount in np.arange(0, cut_mean + 0.01, 0.01):

        add_to_mean = cut_mean - amount
        print(f"Running for: variation: {amount} with add to mean: {add_to_mean}")

        values = args.vector_cut

        print(f"The value list is: {values}, with the type {type(values)}")

        layers = list(range(len(values) + 1))

        # Invert the graph
        min_graph = min(values)
        max_graph = max(values)
        values = [(max_graph - value) for value in values]

        # Normalize the values
        min_graph = min(values)
        max_graph = max(values)
        values = [value / max_graph for value in values]


        mean_graph = np.mean(values)
        values = [value * amount / mean_graph + add_to_mean for value in values]
        new_mean_graph = np.mean(values)

        # Adjust dimensions
        adjust_dimensions = np.array(values)
        adjusted_dimensions = (1 - adjust_dimensions)
        adjusted_dimensions = np.append(adjusted_dimensions, 1)

        #convert the dimensions to a string(to be able to give them as an argument)
        adjusted_dimensions = ' '.join(map(str, adjusted_dimensions))

        print(f"the adjusted dimensions are: {adjusted_dimensions}")

        #run the run_slice_gpt
        command = (f"python3.11 run_slicegpt_perplexity.py --model {args.model} --save-dir "
                   f"/storage/paulclotan/SmartSliceGPT/save_work2 --sparsity 0.25 --no-wandb "
                   f"--device {args.cuda_device} --hf-token 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 "
                   f"--vector-cut {adjusted_dimensions} --cal-dataset {args.dataset}")

        command_to_run = shlex.split(command)


        result = subprocess.run(command_to_run, capture_output=True, text=True)
        output = result.stdout
        # Extract perplexity value using regex
        perplexity_match = re.search(r"After rotating and slicing (\d+\.\d+)", output)
        print(perplexity_match.group(1))
        if perplexity_match:
            perplexity_value = perplexity_match.group(1)
            print(f"\nVariation {amount}%, adding to the mean: {add_to_mean}, perplexity: {perplexity_value}\n")
            file.write(f"\nVariation {amount}%, adding to the mean: {add_to_mean}, perplexity: {perplexity_value}\n")


# values vector for PHI-2 alpaca, basecut 30%, cut vector: 80% layerwise: values = [
#             3.1669, 2.9433, 3.0739, 3.1901, 3.294, 3.2841, 3.2285, 3.2603, 3.3077, 3.3068,
#             3.3112, 3.2988, 3.2954, 3.2786, 3.3106, 3.3605, 3.432, 3.4543, 3.4634, 3.5715,
#             3.6416, 3.7818, 3.8982, 3.9862, 4.1137, 4.236, 4.4062, 4.5342, 4.6055, 4.5875,
#             4.6307, 4.6348
#         ]
# PHI-2 WIKITEXt, cut vector: 80% layerwise: [13.9559, 11.3622, 12.4713, 14.0541, 14.078, 13.719, 14.0593, 13.5304, 13.7168, 13.7191, 13.8314, 13.0987, 12.7281, 12.3108, 12.3131, 12.3014, 12.5094, 12.997, 13.167, 13.1269, 13.4453, 13.7476, 14.3025, 14.9632, 15.331, 16.2428, 17.5334, 19.2541, 20.1539, 21.16, 21.5895, 21.8606]
# ``` &#8203;``【oaicite:0】``&#8203;