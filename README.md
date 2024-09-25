This file contains a short tutorial on how to run the experiments presented in the paper.

There are 2 commands that need to be executed to successfully run the experiments.

The first command is: python3.11 calculate_LR.py 'model_name'
example: python3.11 calculate_LR.py mistralai/Mistral-7B-v0.1

The model name needs to exactly match the name of the model as it is named in Huggingface.
This command will output the LR score presented in the paper.

Next, we will run the second command, that will execute the experiments based on the cutting pattern provided by the previous script.
Command: python3.11 mean_cut_accuracy_experiments.py --model 'model' --source-for-vector wikitext2 --cuda-device 'cuda_device' --dataset wikitext2 --tasks 'tasks' --accuracy-limit 'acc_limit' --vector-cut 'vec_cut' --mean 'mean'

The script has the following main parameters: 
-model (must match the model parameter from the previous script)
-cuda-device: the particular GPU, or CPU, that we want to run the experiment on
-tasks the datasets we want to assess the accuracies on
-dataset: the dataset on which the perplexity will be evaluated(to be initialized as wikitext2)
-accuracy-limit: the number of sample points on which we want to evaluate the accuracy(higher limits imply longer computation times)
-vector-cut: this parameter takes the output of the previous script. (Defines the cutting pattern)
-mean: the mean of the for which we want to run the experiment(e.g. 0.3, 0.35, 0.4)


example: python3.11 mean_cut_accuracy_experiments.py --model mistralai/Mistral-7B-v0.1 --source-for-vector wikitext2 --cuda-device cuda:2 --dataset wikitext2 --tasks hellaswag winogrande arc_easy piqa mmlu arc_challenge --accuracy-limit 1000 --vector-cut 0.0 0.42911685545068057 0.606256316573843 0.6857787985571174 0.6245429330633783 0.6642601683627527 0.7004455348049525 0.717172568095607 0.7472846146330077 0.783581615334039 0.7841917909981923 0.8073249613388788 0.830985124477407 0.847718599550241 0.8280651268462332 0.8458608116661329 0.8422941995742359 0.8855863985095019 0.8263995687771349 0.8620272453114192 0.9187727049851694 0.9589514397512818 0.9823687497309113 0.9825129694416351 0.9978177690620187 1.0 0.9991571428569818 0.9886628647074337 0.9604258119483695 0.9334049076887766 0.9276028258206697 0.4929959305050077 --mean 0.4

IMPORTANT MENTIONS:
This codebase uses certain features from the following GitHub repositories: 
1. https://github.com/sramshetty/ShortGPT
2. https://github.com/microsoft/TransformerCompression
