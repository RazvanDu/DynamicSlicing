from datasets import load_dataset
import torch
from torch.utils.data import DataLoader


from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from transformers import default_data_collator, Trainer, TrainingArguments
import argparse
from short_hf import ShortHFModel

def argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="model to lead",
        choices=[
            # LLAMAmodels
            'meta-llama/Meta-Llama-3-8B',
            # mistral
            'mistralai/Mistral-7B-v0.1',

        ],
        default="mistralai/Mistral-7B-v0.1",
    )

    args = parser.parse_args()
    return args


device_id = 0
args = argparser()

#introduce the path for your environment
cache_dir_datasets = '/mnt/' + 'razvandu/LayerAveraging/datasets_' + str(device_id)
model_name = args.model


print("WORKING ON MODEL", model_name)

data = load_dataset("pg19", split="validation", cache_dir=cache_dir_datasets)  # authors sample 10,000 texts to compute block influences
dataloader = DataLoader(
    data,
    batch_size=1,
    shuffle=True,
)

MAX_SEQ_LEN = 1024
short_model = ShortHFModel(
    model_name=model_name,
    layers_path="model.layers",
    n_prune_layers=9
)

# sample generation
gen = short_model.model.generate(
    short_model.tokenizer(["I am an avid fan of "], return_tensors='pt').input_ids.to("cuda:4"),
    max_new_tokens=20
)

short_model.tokenizer.batch_decode(gen, skip_special_tokens=True)

for i, batch in enumerate(dataloader):
    prompts = batch['text']

    short_model.eval_importance(
        prompts=prompts,
        max_seq_len=MAX_SEQ_LEN,
        stride=256,
        max_gen_len=0
    )

    to_sort = [(short_model.importances[i], i) for i in range(len(short_model.importances))]

    to_sort.sort()
    to_sort.reverse()

    max_value = max([to_sort[i][0] for i in range(len(to_sort))])
    min_value = min([to_sort[i][0] for i in range(len(to_sort))])
    normalized = [(1-(to_sort[i][0]-min_value)/(max_value-min_value), to_sort[i][1]) for i in range(len(to_sort))]
    normalized_2 = [(to_sort[i][1], 1 - (to_sort[i][0] - min_value) / (max_value - min_value)) for i in
                    range(len(to_sort))]
    normalized_2.sort()
    normalized.reverse()

    to_sort = [to_sort[i][1] for i in range(len(to_sort))]

    print("Z", to_sort)

print(short_model.importances)