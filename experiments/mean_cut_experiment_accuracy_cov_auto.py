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
    parser.add_argument('--dataset', type=str, default='wikitext2', help='Dataset on which we evaluate.')
    parser.add_argument('--accuracy-limit', type=int, default=-1, help='CUDA device to use.')
    parser.add_argument('--cov-limit', type=float, default=1.0, help='The covariance limit.')

    parser.add_argument(
        '--tasks',
        nargs='+',
        default= None,
        choices=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "boolq", "gsm8k_cot", "mlqa_en", "xlsum_en", "mmlu", "social_iqa"],
    )
    parser.add_argument('--cuda-device', type=str, default='cuda:0', help='CUDA device to use.')

    return parser.parse_args()

args = parse_args()
# perplexity values that give the form of the graph - to be computed with percentage-cut-experiment.py


model_name_save = args.model.replace("/", "-")
print(model_name_save)

save_path = (f"/storage/paulclotan/SmartSliceGPT/experiments/experiment_fixed_result/activation_accur_perp_results_{model_name_save}_"
             f"_cov_benchmark_limit_{args.accuracy_limit}_cov_limit_{args.cov_limit}_final.txt")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, 'w') as file:
    file.write(
        f"Experiment Summary." 
        f"\nThis experiment is based on the pattern found in the coeficient of variation."
        f"\nThe limit used for this experiment is: {args.accuracy_limit}"
        f"\nModel: {args.model}"
        f"\nDataset: {args.dataset}"
        f"\nSource for Vector-Cut: {args.source_for_vector}"
        f"\nEvaluated on: {args.tasks}%"
        f"\nCUDA Device: {args.cuda_device}"
        f"\nCov limit: {args.cov_limit}")
    file.flush()

    # get the pattern:

    command = (f"python3.11 run_slicegpt_perplexity.py --model {args.model} --save-dir "
               f"/storage/paulclotan/SmartSliceGPT/save_work2 --sparsity 0.25 --no-wandb "
               f"--device {args.cuda_device} --hf-token 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 "
               f"--cal-dataset {args.source_for_vector} --single-layer-cut 1 --cov-limit {args.cov_limit} ")

    # Run the command and capture the output
    result = subprocess.run(shlex.split(command), capture_output=True, text=True)
    output = result.stdout

    # Extract skewness values from the output string
    skewness_values = re.findall(r'Skewness for each column: (-?\d+\.\d+e?-?\d*)', output)

    # Apply the absolute value function to each skewness value and convert to float
    #abs_skewness_values = [str(abs(float(value))) for value in skewness_values] # without abs for the other abs experiment(matrix abs experiment- no abs here)

    abs_skewness_values = [str(float(value)) for value in skewness_values] # without abs for the other abs experiment(matrix abs experiment- no abs here)

    # Create the list of values separated by one space
    skewness_list_str = ' '.join(abs_skewness_values)



    file.write(
        f"\nVector: {skewness_list_str}\n\n\n")
    file.flush()

    # Adjust the mean of the graph

    for amount in np.arange(0, cut_mean + 0.01, 0.02):

        add_to_mean = cut_mean - amount
        print(f"Running for: variation: {amount} with add to mean: {add_to_mean}")

        string_list = skewness_list_str.split()

        # Convert each string in the list to a float
        float_vector = [float(num) for num in string_list]

        values = float_vector

        print(f"\nThe value list is: {values}, with the type {type(values)}")

        layers = list(range(len(values) + 1))

        # Invert the graph
        min_graph = min(values)
        max_graph = max(values)
        values = [(max_graph - value) for value in values]

        # Normalize the values
        min_graph = min(values)
        max_graph = max(values)
        values = [(value - min_graph) / (max_graph - min_graph) for value in values]


        mean_graph = np.mean(values)
        values = [value * amount / mean_graph + add_to_mean for value in values]
        new_mean_graph = np.mean(values)

        # Adjust dimensions
        adjust_dimensions = np.array(values)
        adjusted_dimensions = (1 - adjust_dimensions)
        adjusted_dimensions = np.append(adjusted_dimensions, 1)

        #convert the dimensions toa a string(to be able to give them as an argument)
        adjusted_dimensions = ' '.join(map(str, adjusted_dimensions))

        print(f"the adjusted dimensions are: {adjusted_dimensions}")

        string_tasks = ' '.join(args.tasks)
        #run the run_slice_gpt
        command = (f"python3.11 run_slicegpt_perplexity.py --model {args.model} --save-dir "
                   f"/storage/paulclotan/SmartSliceGPT/save_work2 --sparsity 0.25 --no-wandb "
                   f"--device {args.cuda_device} --hf-token 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 "
                   f"--vector-cut {adjusted_dimensions} --cal-dataset {args.dataset} --single-layer-cut 0 "
                   f"--tasks {string_tasks} --accuracy-limit {args.accuracy_limit}")
        print(string_tasks,"The type of the tasks is", type(string_tasks))
        command_to_run = shlex.split(command)


        result = subprocess.run(command_to_run, capture_output=True, text=True)
        output = result.stdout

        # Extract perplexity value using regex

        perplexity_match = re.search(r"After rotating and slicing (\d+\.\d+)", output)
        #print(perplexity_match.group(1))
        if perplexity_match:
            perplexity_value = perplexity_match.group(1)
            print(f"\nVariation {amount}%, adding to the mean: {add_to_mean}, perplexity: {perplexity_value}\n")
            file.write(f"\n\n\nVariation {amount}%, adding to the mean: {add_to_mean}, perplexity: {perplexity_value}\n")


        # Extract perplexity value using regex

        accuracy_results_match = re.search(r"The accuracy results are: (\{.*\})", output, re.DOTALL)
        if accuracy_results_match:
            accuracy_results = accuracy_results_match.group(1)

            print(
                f"Variation {amount}%, adding to the mean: {add_to_mean}, Average accuracy across tasks is: {accuracy_results}")
            file.write(
                f"\nVariation {amount}%, adding to the mean: {add_to_mean}, Average accuracy: {accuracy_results}\n")
            file.flush()
        else:
            file.write("No accuracy results found.")
        file.flush()


