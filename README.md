# Dynamic Slicing

DynamicSlicing is a repository for running experiments on model compression and evaluation, based on the methodology presented in our paper (https://aclanthology.org/2024.findings-emnlp.579/).

## Getting Started

Clone the repository and ensure that Python 3.11 or above is installed. Install any dependencies listed in `requirements.txt` by running:
```bash
pip install -r requirements.txt
```

## Running Experiments
### Step 1: Calculate LR Score

The first step involves calculating the `LR (Layer Redundancy)` score for a model from Hugging Face. The `calculate_LR.py` script takes a model name as input and outputs the LR score.
```bash
python3.11 calculate_LR.py <model_name>
```
**Example:**
```bash
python3.11 calculate_LR.py mistralai/Mistral-7B-v0.1
```
Note: The `model_name` must match the name of the model from the Hugging face platform.

### Step 2: Run Mean Cut Accuracy Experiments
Once the LR score is calculated, the second step is to run accuracy experiments based on a cutting pattern determined by the first script. The `mean_cut_accuracy_experiments.py` script allows you to test different compression levels across various datasets and configurations.

**Command:**
```bash
python3.11 mean_cut_accuracy_experiments.py --model <model_name> --source-for-vector <dataset> --cuda-device <device> --dataset <evaluation_dataset> --tasks <task_list> --accuracy-limit <limit> --vector-cut <cut_pattern> --mean <mean_value>
```
#### Parameters

Each parameter tailors the experiment to specific needs. Below is a breakdown:

- **model**: Must match the model parameter from `calculate_LR.py`.
- **cuda-device**: Device for execution (e.g., `cuda:2` or `cpu`).
- **tasks**: List of datasets to evaluate for accuracy.
- **dataset**: The main dataset for perplexity evaluation (default is `wikitext2`).
- **accuracy-limit**: Number of samples to evaluate (higher values increase computation time).
- **vector-cut**: Takes output from the `calculate_LR.py` script, defining the cutting pattern.
- **mean**: Mean value to use for the experiment (e.g., `0.3`, `0.35`, `0.4`).

**Example**
```bash
python3.11 mean_cut_accuracy_experiments.py --model mistralai/Mistral-7B-v0.1 --source-for-vector wikitext2 --cuda-device cuda:2 --dataset wikitext2 --tasks hellaswag winogrande arc_easy piqa mmlu arc_challenge --accuracy-limit 1000 --vector-cut 0.0 0.4291 ... 1.0 --mean 0.4
```
## Important mentions
This codebase incorporates functionalities from the following repositories:
1. [ShortGPT](https://github.com/sramshetty/ShortGPT)
2. [Transformer Compression by Microsoft](https://github.com/microsoft/TransformerCompression)

## Citation

To cite the paper please use the following citation:

### BibTeX
```bash
@inproceedings{dumitru-etal-2024-change,
    title = "Change Is the Only Constant: Dynamic {LLM} Slicing based on Layer Redundancy",
    author = "Dumitru, Razvan-Gabriel  and
      Clotan, Paul Ioan  and
      Yadav, Vikas  and
      Peteleaza, Darius  and
      Surdeanu, Mihai",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.579",
    pages = "9912--9920",
    abstract = "This paper introduces a novel model compression approach through dynamic layer-specific pruning in Large Language Models (LLMs), enhancing the traditional methodology established by SliceGPT. By transitioning from constant to dynamic slicing, our method leverages the newly proposed Layer Redundancy (LR) score, which assesses how much change each layer changes its input by measuring the cosine similarity of the input to the output of the layer. We use this score to prune parts of individual layers based on redundancy in such a way that the average pruned percentage for all layers is a fixed value. We conducted extensive experiments using models like Llama3-8B and Mistral-7B on multiple datasets, evaluating different slicing bases and percentages to determine optimal configurations that balance efficiency and performance. Our findings show that our dynamic slicing approach not only maintains but, in many cases, enhances model performance compared to the baseline established by constant slicing methods. For instance, in several settings, we see performance improvements of up to 5{\%} over the SliceGPT baseline.Additionally, a perplexity decrease by as much as 7{\%} was observed across multiple benchmarks, validating the effectiveness of our method. The code, model weights, and datasets are open-sourced at - https://github.com/RazvanDu/DynamicSlicing",
}

```

