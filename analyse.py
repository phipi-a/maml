import sys
import time

sys.path.insert(0, './..')
import analyse_utils
import numpy as np
import os
from svcca import cca_core, pwcca
import importlib

importlib.reload(pwcca)

importlib.reload(analyse_utils)
importlib.reload(cca_core)
import json
from datetime import datetime
import argparse


def store_json(data, filename):
    f = open(filename, 'w', encoding='utf-8')
    json.dump(data, f, indent=4)


def sec_to_string(seconds):
    minutes = int(seconds / 60)
    hours = int(minutes / 60)
    seconds = seconds % 60
    minutes = minutes % 60
    return "{0:02d}:{1:02d}:{2:02d}".format(int(hours), int(minutes), int(seconds))


parser = argparse.ArgumentParser()

# -db DATABSE -u USERNAME -p PASSWORD -size 20
parser.add_argument('-p', '--model_paths', help='list of model paths: /path/to/model1,path/to/model2', type=str)
parser.add_argument('-n', '--model_names', help='list of model names: model1,model2', type=str)
parser.add_argument("-l", "--labels_path", help="path to labels: /path/to/labels for cosine accuracy")
# parser.add_argument("-i", "--inner_steps", help="innerloop iterations", type=int)
parser.add_argument("-d", "--dir", help="directionary", type=str)
parser.add_argument("-v", "--verbose", help="verbose", action='store_true')
parser.add_argument("-e", "--exclusive_layer_sim_analyse", help="exclusive layer similarity", action='store_true')
parser.add_argument("-i", "--inclusive_layer_sim_analyse", help="inclusive layer similarity", action='store_true')
parser.add_argument("-c", "--cca", help="pwcca (calculation of PWCCA forked from https://github.com/google/svcca)", action='store_true')
parser.add_argument("-k", "--cka", help="cka", action='store_true')
parser.add_argument("-s", "--sim_acc", help="inclusive cosine accuracy", action='store_true')

args = parser.parse_args()
sim_cal_func="pwcca"
if args.cca:
    sim_cal_func = "cca"
if args.cka:
    sim_cal_func = "cka"


labels = np.load(args.labels_path, allow_pickle=True)["arr_0"]
model_paths = [item for item in args.model_paths.split(',')]
model_names = [item for item in args.model_names.split(',')]
# inner_steps=args.inner_steps
data_directionary = args.dir
inclusive_layer_sim_analyse=args.inclusive_layer_sim_analyse
exclusive_layer_sim_analyse=args.exclusive_layer_sim_analyse
verbose = args.verbose
sim_acc= args.sim_acc
# labels=np.load("labels_analyse.npz",allow_pickle=True)["arr_0"]
# model_paths=["cls_20.mbs_16.ubs_1.numstep5.updatelr0.1batchnorm_1/data","cls_20.mbs_16.ubs_1.numstep5.updatelr0.1batchnorm.fr4_1/data","cls_20.mbs_16.ubs_1.numstep5.updatelr0.1batchnorm.reset_layer_train_test_1/data"]
# model_names=["MAML","ANIL","RRAIL"]
# inner_steps=10
# data_directionary="./json_data"
# verbose=False

data_directionary = data_directionary + "/" + datetime.now().strftime("%d%m%y_%H%M%S") + "/"
os.makedirs(data_directionary, exist_ok=True)
meta_iteras_list = list(range(0, 60000, 1000))

all_start = time.time()

cos_acc_path = data_directionary + "cos_acc.json"
meta_sim_path = data_directionary + "meta_sim.json"
inner_sim_path = data_directionary + "inner_sim.json"
cos_acc = {}
meta_sim = {}
inner_sim = {}
if not exclusive_layer_sim_analyse:
    for model_path, model_name in zip(model_paths, model_names):
        model_start = time.time()
        meta_iteras_cos_acc = []
        meta_iteras_meta_sim = []
        meta_iteras_inner_sim = []
        print("load compare meta_iteration: 59000", "of model", model_name, "...", end=" ", flush=True)
        start = time.time()
        last_meta_iteration = analyse_utils.load_iter(59000, data_path=model_path)
        end = time.time()
        print("Time: " + sec_to_string(end - start), flush=True)
        for itera in meta_iteras_list:
            itera_start = time.time()
            inner_loop_iters = []
            print("load meta_iteration:", itera, "of model", model_name, "...", end=" ", flush=True)
            start = time.time()
            meta_iteration = analyse_utils.load_iter(itera, data_path=model_path)
            end = time.time()
            print("Time: " + sec_to_string(end - start), flush=True)

            if sim_acc:
            # Calculate Cosine Accuracy
                print("Calculate Cosine Accuracy...", end=" ", flush=True)
                start = time.time()
                meta_iteras_cos_acc.append({"meta-iteration": itera,
                                            "innerloop": analyse_utils.calc_cos_acc(meta_iteration, labels, verbose=verbose, inner_loop_iters_list=[0,10] )})
                end = time.time()
                print("Time: " + sec_to_string(end - start), flush=True)

            # Meta-Learning Similarity
            print("Calculate Meta-Learning Similarity...", end=" ", flush=True)
            start = time.time()
            meta_iteras_meta_sim.append({"meta-iteration": itera,
                                         "innerloop": analyse_utils.calc_meta_sim(meta_iteration, last_meta_iteration,sim_cal_func,
                                                                                  verbose=verbose,inner_loop_iters_list=[0,10])})
            end = time.time()
            print("Time: " + sec_to_string(end - start), flush=True)

            # Calculate Innerloop Similarity

            print("Calculate Innerloop Similarity...", end=" ", flush=True)
            start = time.time()
            meta_iteras_inner_sim.append(
                {"meta-iteration": itera, "innerloop": analyse_utils.calc_inner_sim(meta_iteration,sim_cal_func, verbose=verbose)})
            end = time.time()
            print("Time: " + sec_to_string(end - start), flush=True)

            itera_end = time.time()
            print("Iteration Calculation Time: " + sec_to_string(itera_end - itera_start), flush=True)
        print(model_name, end=" ", flush=True)
        if sim_acc:
            cos_acc[model_name] = meta_iteras_cos_acc
        meta_sim[model_name] = meta_iteras_meta_sim
        inner_sim[model_name] = meta_iteras_inner_sim
        model_end = time.time()
        print("Model Calculation Time: " + sec_to_string(model_end - model_start), flush=True)
    if sim_acc:
        print("save cos_acc.json:", cos_acc_path, flush=True)
        store_json(cos_acc, cos_acc_path)

    print("save meta_sim.json:", meta_sim_path, flush=True)
    store_json(meta_sim, meta_sim_path)

    print("save inner_sim.json:", inner_sim_path, flush=True)
    store_json(inner_sim, inner_sim_path)

# Calculate Innerloop Similarity
if inclusive_layer_sim_analyse or exclusive_layer_sim_analyse:
    layer_sim_path = data_directionary + "layer_sim.json"

    print("Calculate Layer Similarity...", end=" ", flush=True)
    start = time.time()
    pre_layer_sim, post_layer_sim = analyse_utils.calc_sim_arrays(model_paths, model_names,sim_cal_func,verbose=verbose)
    end = time.time()
    print("Time: " + sec_to_string(end - start), flush=True)
    print("save layer_sim.json:", layer_sim_path, flush=True)
    store_json({"pre_innerloop": pre_layer_sim, "post_innerloop": post_layer_sim}, layer_sim_path)

all_end = time.time()
print("All Calculation Time: " + sec_to_string(all_end - all_start), flush=True)

print("#################", flush=True)


