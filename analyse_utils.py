from zipfile import ZipFile as zf
import numpy as np
from svcca import pwcca, cca_core
import os
import itertools
import shutil
import multiprocessing
import scipy.spatial
import time
import numpy as np
import math

def sec_to_string(seconds):
    minutes = int(seconds / 60)
    hours = int(minutes / 60)
    seconds = seconds % 60
    minutes = minutes % 60
    return "{0:02d}:{1:02d}:{2:02d}".format(int(hours), int(minutes), int(seconds))

def load_iter(iteration, data_path="./data"):
    data = np.load(data_path + '/iter_' + str(iteration) + '.npz', allow_pickle=True)["arr_0"].item()
    return data


def cos_sim(a, b):
    return 1-scipy.spatial.distance.cosine(a,b)

def centering(K):
    K=K-K.mean(axis=1)
    return K-K.mean(axis=0)

def linear_CKA(X, Y):
    X=centering(X)
    Y = centering(Y)
    a=np.trace(Y@X.T@X@Y.T)
    b=np.trace(X@X.T@ X@X.T)
    c=np.trace(Y@Y.T@ Y@Y.T)
    return a/np.sqrt(b*c)

def calculate_similarity(X,Y,sim_cal_func):
    if X.shape==Y.shape and np.allclose(X,Y):
        return 1.0
    X = X + np.random.rand(*(X.shape)) * 0.000001
    Y = Y + np.random.rand(*(Y.shape)) * 0.000001

    #print("pre:",np.cov(X, X)[:numx, :numx])
    if sim_cal_func=="pwcca":
        return pwcca.compute_pwcca(X, Y, epsilon=1e-8)[0]
    elif sim_cal_func == "cca":
        return np.mean(cca_core.get_cca_similarity(X, Y, epsilon=1e-8)["cca_coef1"])
    elif sim_cal_func == "cka":
        x=linear_CKA(X, Y)
        return x


def calc_single_cos_acc(single_layer_output, labels, inner_loop_iter):
    itera_acc = []
    for s, test_iteration in enumerate(single_layer_output):
        iteration_res = []
        num_classes = labels.shape[3]

        test_iteration_labels = np.argmax(labels[s, 0], axis=1)

        layer_output_size = test_iteration[0, 0, 0].size
        output_by_cls = [[] for i in range(num_classes)]
        for i, label in enumerate(test_iteration_labels):
            output_by_cls[label].append(test_iteration[inner_loop_iter, 0, i].reshape(layer_output_size))
        train_obc, test_obc = np.split(output_by_cls, [1], axis=1)
        for class_idx, outputs in enumerate(test_obc):
            for output in outputs:
                cos_sim_out = []
                for cmp_output in train_obc[:, 0]:
                    cos_sim_out.append(cos_sim(cmp_output, output))
                iteration_res.append(np.argmax(cos_sim_out) == class_idx)

        itera_acc.append(iteration_res.count(True) / len(iteration_res))
    return np.asarray(itera_acc).mean().astype(np.float64), np.asarray(itera_acc).std().astype(np.float64)


def calc_cos_acc(meta_iteration, labels, images_per_class=2, layers=["layer_" + str(i) for i in range(5)],
                 verbose=False, inner_loop_iters_list=list(range(11))):
    inner_loop_iters = []
    for inner_loop_iter in inner_loop_iters_list:
        inner_loop_dict = {}
        inner_loop_dict["acc_mean"] = meta_iteration["all_acc"]["means"][inner_loop_iter].astype(np.float64)
        inner_loop_dict["acc_std"] = meta_iteration["all_acc"]["stds"][inner_loop_iter].astype(np.float64)
        layers_sim_acc = {k: [] for k in layers}
        for layer in layers:
            mean, std_dev = calc_single_cos_acc(meta_iteration["layer_output"][layer], labels, inner_loop_iter)
            layers_sim_acc[layer] = {"acc_mean": mean, "acc_std": std_dev}
        inner_loop_dict["layerwise"] = layers_sim_acc
        if verbose:
            print("innerloop iteration", inner_loop_iter, inner_loop_dict)
        inner_loop_dict["iteration"] = inner_loop_iter
        inner_loop_iters.append(inner_loop_dict)
    # if verbose:
    # print(inner_loop_iters[-1]["layerwise_cos_acc"][layers[-1]]," acc_mean:",inner_loop_iters[-1]["acc_means"])
    return inner_loop_iters


def calc_meta_sim(meta_iteration, last_meta_iteration,sim_cal_func, inner_loop_iters_list=[0, 10],
                  layers=["layer_" + str(i) for i in range(5)], verbose=False):
    inner_loop_iters = []
    for inner_loop_iter in inner_loop_iters_list:
        inner_loop_dict = {}
        layers_meta_sim = {k: [] for k in layers}
        for layer in layers:
            m_temp = meta_iteration["layer_output"][layer][:, inner_loop_iter]
            if len(m_temp.shape) > 5:
                # conv layer
                m_temp = m_temp.mean(axis=(3, 4))
            layer_output = m_temp.reshape(-1, m_temp.shape[-1]).T
            m_temp = last_meta_iteration["layer_output"][layer][:, inner_loop_iter]
            if len(m_temp.shape) > 5:
                # conv layer
                m_temp = m_temp.mean(axis=(3, 4))
            layer_output_cmp = m_temp.reshape(-1, m_temp.shape[-1]).T
            # output: variable x datapoints
            layers_meta_sim[layer] = calculate_similarity(layer_output, layer_output_cmp, sim_cal_func)
        inner_loop_dict["layerwise"] = layers_meta_sim
        if verbose:
            print("innerloop iteration", inner_loop_iter, inner_loop_dict)
        inner_loop_dict["iteration"] = inner_loop_iter
        inner_loop_iters.append(inner_loop_dict)
    # if verbose:
    # print(inner_loop_iters[-1]["layerwise_cos_acc"][layers[-1]]," acc_mean:",inner_loop_iters[-1]["acc_means"])
    return inner_loop_iters


def calc_inner_sim(meta_iteration,sim_cal_func, layers=["layer_" + str(i) for i in range(5)], verbose=False):
    inner_loop_dict = {}
    layers_meta_sim = {k: [] for k in layers}
    for layer in layers:
        m_temp = meta_iteration["layer_output"][layer][:, 0]
        if len(m_temp.shape) > 5:
            # conv layer
            m_temp = m_temp.mean(axis=(3, 4))
        layer_output_pre = m_temp.reshape(-1, m_temp.shape[-1]).T
        m_temp = meta_iteration["layer_output"][layer][:, -1]
        if len(m_temp.shape) > 5:
            # conv layer
            m_temp = m_temp.mean(axis=(3, 4))
        layer_output_post = m_temp.reshape(-1, m_temp.shape[-1]).T

        layers_meta_sim[layer] = calculate_similarity(layer_output_pre, layer_output_post, sim_cal_func)
    inner_loop_dict["layerwise"] = layers_meta_sim
    if verbose:
        print("inner_loop_sim", inner_loop_dict)
    return inner_loop_dict


def compare_models(model_a, model_b, same_model, innerloop_iteration,sim_cal_func, layers=["layer_" + str(i) for i in range(5)],verbose=False):
    sim = []
    if same_model:
        comb_layer_names = list(itertools.combinations(layers, 2))
    else:
        comb_layer_names = itertools.product(layers, layers)
    for layer_a, layer_b in comb_layer_names:
        if verbose:
            print("compare layer",layer_a,layer_b, flush=True)
        m_temp = model_a["layer_output"][layer_a][:, innerloop_iteration]
        if len(m_temp.shape) > 5:
            # conv layer
            m_temp = m_temp.mean(axis=(3, 4))
        layer_output_a = m_temp.reshape(-1, m_temp.shape[-1]).T

        m_temp = model_b["layer_output"][layer_b][:, innerloop_iteration]
        if len(m_temp.shape) > 5:
            # conv layer
            m_temp = m_temp.mean(axis=(3, 4))
        layer_output_b = m_temp.reshape(-1, m_temp.shape[-1]).T
        sim.append({"layer_a": layer_a, "layer_b": layer_b,
                    "sim": calculate_similarity(layer_output_a, layer_output_b, sim_cal_func)})
    return sim


def calc_sim_arrays(model_paths, model_names, sim_cal_func, verbose=False, pre_innerloop=0, post_innerloop=10):
    models = {}
    for model_path, model_name in zip(model_paths, model_names):
        if verbose:
            print("load meta_iteration: 59000", "of model", model_name, "...", end=" ", flush=True)
        start = time.time()
        models[model_name] = load_iter(59000, data_path=model_path)
        end = time.time()
        if verbose:
            print("Time: " + sec_to_string(end - start))

    comb_model_names = list(itertools.combinations(model_names, 2))
    comb_model_names = comb_model_names + list(zip(model_names, model_names))
    # compare
    pre_model_sim = []
    post_model_sim = []
    for model_names_a, model_names_b in comb_model_names:
        if verbose:
            print("compare model", model_names_a, model_names_b, flush=True)
        model_a = models[model_names_a]
        model_b = models[model_names_b]
        pre_model_sim.append({"model_a": model_names_a, "model_b": model_names_b,
                              "sim": compare_models(model_a, model_b, model_names_a == model_names_b, pre_innerloop,sim_cal_func,verbose=verbose)})
        post_model_sim.append({"model_a": model_names_a, "model_b": model_names_b,
                               "sim": compare_models(model_a, model_b, model_names_a == model_names_b, post_innerloop,sim_cal_func,verbose=verbose)})
    return pre_model_sim, post_model_sim