Ein Vergleich unterschiedlicher Lernstrategien für Model Agnostic Meta-Learning
==============
***code of bachelorthesis***
# MAML added Features (main.py):
MAML forked from https://github.com/cbfinn/maml (CBFINN_README.md)
- ANALYSE: validate mutliple models and store output layerwise
  - model_json_file: json inputfile to load models and analyse them
  - analyse_path: path of layerwise output
  - get_label: store all labels
  - update_batch_size_val_test: K examples per class in support dataset of validation data

  - dream: deep dream (gradient descent on input) [not working]
    - hard_coded_label: ---
- TRAIN: train model
  - no_dense_update: dont update dense layer in inner loop
  - freezing_layer: dont update the last k layer in inner loop
  - second_dense: add a second dense layer at the end of the model
  - reset_last_layer_training_and_test: reset last layer before inner loop train/test (RBIL)
  - reset_last_layer_test: reset last layer before inner loop test

# ANALYSE DATA (analyse.py)
CCA: calculation of PWCCA forked from https://github.com/google/svcca
```
usage: analyse.py [-h] [-p MODEL_PATHS] [-n MODEL_NAMES] [-l LABELS_PATH] [-d DIR] [-v] [-e] [-i] [-c] [-k] [-s]

options:
  -h, --help            show this help message and exit
  -p MODEL_PATHS, --model_paths MODEL_PATHS
                        list of model paths: /path/to/model1,path/to/model2
  -n MODEL_NAMES, --model_names MODEL_NAMES
                        list of model names: model1,model2
  -l LABELS_PATH, --labels_path LABELS_PATH
                        path to labels: /path/to/labels for cosine accuracy
  -d DIR, --dir DIR     directionary
  -v, --verbose         verbose
  -e, --exclusive_layer_sim_analyse
                        exclusive layer similarity
  -i, --inclusive_layer_sim_analyse
                        inclusive layer similarity
  -c, --cca             pwcca (calculation of PWCCA forked from https://github.com/google/svcca)
  -k, --cka             cka
  -s, --sim_acc         inclusive cosine accuracy
```


# COMBINE DATA (combine_data_files.py)
combine data of analyse.py to one file
```
usage: combine_data_files.py [-h] [-i INPUT_DIRS] [-o OUTPUT_DIR]

options:
  -h, --help            show this help message and exit
  -i INPUT_DIRS, --input_dirs INPUT_DIRS
                        list of input dirs: /path/to/inputs/of/strat1,/path/to/inputs/of/strat2
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        output_dir: /path/to/output/dir
```

# Latex helper (removed)

## create_plots/plot_data.ipynb
jupyther notebook to create latex plots
## create_plots/creat_figure.ipynb
jupyther notebook to create latex figures included ref to plots
# Latex Directionary of Latex files and compiled pdf: (removed)
- latex
- latex/out/philipp_ba.pdf
# reproduce models and output of bachelorthesis:
## MAML
### Omniglot train
train model

```shell
python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/
```
### Omniglot analyse
get layerwise output of validation dataset on trained model

```shell
[MAML_train_command] --train=false --test_set=True --model_json_file=models.json --num_testpoints=100 --update_batch_size_val_test=2 --analyse=True 
```

### MiniImageNet train
train model

```shell
python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --num_classes=5 --update_lr=0.01 --num_updates=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True
```
### MiniImageNet analyse
get layerwise output of validation dataset on trained model

```shell
[MAML_train_command] --train=false --test_set=True --model_json_file=models.json --num_testpoints=100 --update_batch_size_val_test=2 --analyse=True 
```


## MAML Variation
### ANIL (Almost No Inner Loop)
```shell
[MAML_test_command] --freezing_layer=4
```
### RBIL (Reinitialize Befor Inner Loop)
```shell
[MAML_test_command] --reset_last_layer_training_and_test=True

```

## models.json:
```json
[
  {
    "dir": "dir/to/model",
    "iterations": [0, 1000, 2000, 3000, 4000, 5000, analyse_model_iteration],
    "layers":[0,1, analyse_layer_output]
  }
]

```
