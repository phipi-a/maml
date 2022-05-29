"""
Usage Instructions:
    10-shot sinusoid:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10

    10-shot sinusoid baselines:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10 --baseline=oracle
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10

    5-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/

    20-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/

    5-way 1-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True

    5-way 5-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet5shot/ --num_filters=32 --max_pool=True

    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.

    For omniglot and miniimagenet training, acquire the dataset online, put it in the correspoding data directory, and see the python script instructions in that directory to preprocess the data.

    Note that better sinusoid results can be achieved by using a larger network.
"""
import csv
import os
import zipfile
import matplotlib.pyplot as plt
# from pympler import asizeof
import numpy as np
import pickle
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.python.platform import flags
import os
import json
from os.path import exists
from tensorflow.contrib.framework.python.framework import checkpoint_utils

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

###################### new
flags.DEFINE_string('model_json_file', None, 'model dir')
flags.DEFINE_bool('analyse', False, 'analyse')
flags.DEFINE_bool('hard_coded_label', False, 'hard_coded_label')

flags.DEFINE_bool('dream', False, 'deep_dream')
flags.DEFINE_integer('dream_steps', 200000, 'deep_steps')

flags.DEFINE_bool('get_label', False, 'get_label')
flags.DEFINE_string('analyse_path', "./", 'analyse_path')

flags.DEFINE_integer('update_batch_size_val_test', -1, 'batchsize for validation in test')

flags.DEFINE_integer('num_testpoints', 600, 'number of testpoints')
flags.DEFINE_bool('freezing_meta', False, 'freezing_meta')
flags.DEFINE_bool("no_dense_update",False, "no_dense_update")

###################### new
## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000,
                     'number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5,
                     'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.')  # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')

###################### new
flags.DEFINE_integer('freezing_layer', 0, 'number of freezing layer')
flags.DEFINE_bool('second_dense', False, '2 dense layer')
flags.DEFINE_bool('reset_last_layer_training_and_test', False, 'if True, reinitialization last layer before each step')
flags.DEFINE_bool('reset_last_layer_test', False, 'if True, reinitialization last layer before each test step')
###################### new
## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1,
                     'number of examples used for gradient update during training (use if you want to test with a different number).')

flags.DEFINE_string('model_dir','./',' ')
flags.DEFINE_float('train_update_lr', -1,
                   'value of inner gradient step step during training. (use if you want to test with a different value)')  # 0.1 for omniglot
# Opening JSON file

from maml import MAML
from data_generator import DataGenerator

if FLAGS.analyse:
    f = open(FLAGS.model_json_file)
    model_meta_data = json.loads(f.read())


# ./logs/MAML/omniglot20way/cls_20.mbs_16.ubs_1.numstep5.updatelr0.1batchnorm.fr4/
def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 100
    # store_output = 100
    # save_output = 1200
    SAVE_INTERVAL = 1000
    if FLAGS.datasource == 'sinusoid':
        PRINT_INTERVAL = 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = 5 * PRINT_INTERVAL

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes  # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []
    end = []
    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        print("iteration:", itr + 1)
        feed_dict = {}
        if 'generate' in dir(data_generator):
            batch_x, batch_y, amp, phase = data_generator.generate()

            if FLAGS.baseline == 'oracle':
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                for i in range(FLAGS.meta_batch_size):
                    batch_x[i, :, 1] = amp[i]
                    batch_x[i, :, 2] = phase[i]

            inputa = batch_x[:, :num_classes * FLAGS.update_batch_size, :]
            labela = batch_y[:, :num_classes * FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes * FLAGS.update_batch_size:, :]  # b used for testing
            labelb = batch_y[:, num_classes * FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}

        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates - 1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates - 1]])

        #call reinit for last layer befor training
        if "set_random_op" in dir(model) and model.set_random_op[0] is not None:
            if model.set_random_op[2] is None:
                sess.run(model.set_random_op[:2])
            else:
                sess.run(model.set_random_op)
        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr != 0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        #write_meta_graph
        if itr % SAVE_INTERVAL == 0 or itr == FLAGS.pretrain_iterations + FLAGS.metatrain_iterations - 1:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr), write_meta_graph=False)

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0 and FLAGS.datasource != 'sinusoid':
            ##call reinit for last layer
            if "set_random_op" in dir(model) and model.set_random_op[0] is not None:
                if model.set_random_op[2] is None:
                    sess.run(model.set_random_op[:2])
                else:
                    sess.run(model.set_random_op)
            if 'generate' not in dir(data_generator):
                feed_dict = {}
                if model.classification:
                    input_tensors = [model.metaval_total_accuracy1,
                                     model.metaval_total_accuracies2[FLAGS.num_updates - 1],
                                     model.summ_op]
                else:
                    input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates - 1],
                                     model.summ_op]
            else:
                batch_x, batch_y, amp, phase = data_generator.generate(train=False)
                inputa = batch_x[:, :num_classes * FLAGS.update_batch_size, :]
                inputb = batch_x[:, num_classes * FLAGS.update_batch_size:, :]
                labela = batch_y[:, :num_classes * FLAGS.update_batch_size, :]
                labelb = batch_y[:, num_classes * FLAGS.update_batch_size:, :]
                feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb,
                             model.meta_lr: 0.0}
                if model.classification:
                    input_tensors = [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates - 1]]
                else:
                    input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates - 1]]

            result = sess.run(input_tensors, feed_dict)
            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))


# calculated for omniglot
NUM_TEST_POINTS = FLAGS.num_testpoints


#try to visualize features (deepdream)
def dream(model, sess,fw,metaval_input_tensors,lb_dream,input_dream=None, result_path=None):
    model.construct_model(input_tensors=metaval_input_tensors, prefix='dream_', no_fw=fw == None)
    #if input_dream!=None:
    #    model.input_dream.assign(input_dream)
    np.set_printoptions(precision=1, suppress=True)
    if fw!=None:
        feed={i: d[0] for i, d in zip(model.old_w, fw)}
    else:
        feed={}
    feed[model.label_dream] = lb_dream

    sess.run(tf.global_variables_initializer())
    f=FLAGS.dream_steps-1
    for i in range(FLAGS.dream_steps):
        # if i % 100 == 0:
        #     rn.append(model.val_loss_dream, )
        # if i % 1000 == 0:
        #     rn.append(model.input_dream)
        model.input_dream
        result=sess.run([model.dream_op,model.val_loss_dream, model.input_dream,model.out], feed_dict=feed)
        if i % 1000 == 0:
            print(i, "loss:", result[1],result[-1])
        #if i % 1000 == 0:
             #plt.imsave('img/iter' + str(i) + '.jpeg', result[-1])
        if result[1]<0.01 and i>10000:
            f=i
            break
    res=sess.run([model.val_loss_dream, model.input_dream,model.layerwise_out_dream], feed_dict=feed)
    print("loss:",res[0])

    return {"input": res[1], "loss": res[0], "layer_out":res[2],"iterations":f}


def test(model, saver, sess, exp_string, data_generator, test_num_updates=None, layers=[]):
    num_classes = data_generator.num_classes  # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []
    end = []
    for k in range(NUM_TEST_POINTS):
        if 'generate' not in dir(data_generator):
            feed_dict = {}
            feed_dict = {model.meta_lr: 0.0}
        else:
            batch_x, batch_y, amp, phase = data_generator.generate(train=False)

            if FLAGS.baseline == 'oracle':  # NOTE - this flag is specific to sinusoid
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                batch_x[0, :, 1] = amp[0]
                batch_x[0, :, 2] = phase[0]

            inputa = batch_x[:, :num_classes * FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes * FLAGS.update_batch_size:, :]
            labela = batch_y[:, :num_classes * FLAGS.update_batch_size, :]
            labelb = batch_y[:, num_classes * FLAGS.update_batch_size:, :]

            feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb,
                         model.meta_lr: 0.0}
        if model.classification:
            #add output_layervise
            result = sess.run(
                [[model.metaval_total_accuracy1] + model.metaval_total_accuracies2, model.output_layervise,model.fw], feed_dict)

        else:  # this is for sinusoid
            result = sess.run([model.total_loss1] + model.total_losses2, feed_dict)
        metaval_accuracies.append(result[0])

        #reformat layerwise output
        rere = result[1]
        if FLAGS.dream:
            return result[-1]
        rere = [list(x) for x in zip(*rere)]
        rere = [np.asarray(i) for i in rere]
        end.append(rere)
    # reformat layerwise output
    end = [list(x) for x in zip(*end)]
    new_dict = dict()
    for k, i in enumerate(end):
        if k in layers:
            new_dict["layer_" + str(k)] = np.asarray(i)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    if FLAGS.analyse:
        return new_dict, {"metaval_accuracies": metaval_accuracies, "means": means, "stds": stds, "ci95": ci95}
    os.makedirs(FLAGS.logdir + '/' + exp_string)
    out_filename = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + str(
        FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    out_pkl = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(
        FLAGS.update_lr) + '.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update' + str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)


def main():
    if FLAGS.datasource == 'sinusoid':
        if FLAGS.train:
            test_num_updates = 5
        else:
            test_num_updates = 10
    else:
        if FLAGS.datasource == 'miniimagenet':
            if FLAGS.train == True:
                test_num_updates = 1  # eval on at least one update during training
            else:
                test_num_updates = 10
        else:
            test_num_updates = 10

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1
    if FLAGS.datasource == 'sinusoid':
        data_generator = DataGenerator(FLAGS.update_batch_size * 2, FLAGS.meta_batch_size)
    else:
        if FLAGS.metatrain_iterations == 0 and FLAGS.datasource == 'miniimagenet':
            assert FLAGS.meta_batch_size == 1
            assert FLAGS.update_batch_size == 1
            data_generator = DataGenerator(1, FLAGS.meta_batch_size)  # only use one datapoint,
        else:
            if FLAGS.datasource == 'miniimagenet':  # TODO - use 15 val examples for imagenet?
                if FLAGS.train:
                    uuu = FLAGS.update_batch_size +15

                else:
                    # overwrite update_batch_size_val_test via FLAG
                    uuu = FLAGS.update_batch_size * 2
                    if FLAGS.update_batch_size_val_test != -1:
                        uuu = FLAGS.update_batch_size + FLAGS.update_batch_size_val_test
                data_generator = DataGenerator(uuu,
                                                   FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
            else:
                uuu = FLAGS.update_batch_size * 2
                # overwrite update_batch_size_val_test via FLAG
                if FLAGS.update_batch_size_val_test != -1:
                    uuu = FLAGS.update_batch_size + FLAGS.update_batch_size_val_test
                data_generator = DataGenerator(uuu, FLAGS.meta_batch_size,
                                               config={})  # only use one datapoint for testing to save memory

    dim_output = data_generator.dim_output
    if FLAGS.baseline == 'oracle':
        assert FLAGS.datasource == 'sinusoid'
        dim_input = 3
        FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
        FLAGS.metatrain_iterations = 0
    else:
        dim_input = data_generator.dim_input

    if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'omniglot':
        tf_data_load = True
        num_classes = data_generator.num_classes

        if FLAGS.train:  # only construct training model if needed
            random.seed(5)
            image_tensor, label_tensor = data_generator.make_data_tensor()
            inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
            inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
            labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
            labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
            input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

        random.seed(6)
        image_tensor, label_tensor = data_generator.make_data_tensor(train=False)

        inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
        inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
        labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
        metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
    else:
        tf_data_load = False
        input_tensors = None
    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train or not tf_data_load:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load and not FLAGS.dream:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')


    if FLAGS.dream:
        model = MAML(28 * 28, 20)
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    all_vars = [x for x in all_vars if
                "w5_1" not in x.name and "b5_1" not in x.name and "w6" not in x.name and "b6" not in x.name and "input_dream" not in x.name and "model/Variable_4" not in x.name]
    saver = loader = tf.train.Saver(all_vars, max_to_keep=2500)

    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_' + str(FLAGS.num_classes) + '.mbs_' + str(FLAGS.meta_batch_size) + '.ubs_' + str(
        FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)
    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')

    if FLAGS.reset_last_layer_training_and_test:
        exp_string += '.reset_layer_train_test'
    if FLAGS.reset_last_layer_test:
        exp_string += '.reset_layer_test'
    if FLAGS.freezing_layer != 0:
        exp_string += '.fr' + str(FLAGS.freezing_layer)
    if FLAGS.freezing_meta:
        exp_string += '.fr_meta'
    if FLAGS.second_dense:
        exp_string += '.second_dense'
    print(exp_string)
    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)


    if not FLAGS.analyse:
        if FLAGS.resume or not FLAGS.train:
            model_file = tf.train.latest_checkpoint(FLAGS.model_dir)
            if FLAGS.test_iter > 0:
                model_file = model_file[:model_file.rindex('model')] + 'model' + str(FLAGS.test_iter)
            if model_file:
                ind1 = model_file.rindex('model')
                resume_itr = int(model_file[ind1 + 5:])
                print("Restoring model weights from " + model_file)
                print(checkpoint_utils.list_variables(model_file))
                all_variables = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
                tf.global_variables_initializer().run()
                temp_saver = tf.train.Saver(
                    var_list=[v for v in all_variables]) #if "ExponentialMovingAverage" not in v.name and "moving" not in v.name])
                temp_saver.restore(sess, model_file)
    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        #analyse model
        if FLAGS.analyse:
            if FLAGS.get_label:
                labelbs = []
                for i in range(FLAGS.num_testpoints):
                    labelbs.append(sess.run(labelb))
                np.savez_compressed(FLAGS.analyse_path+'/labels_analyse.npz',
                                    np.asarray(labelbs))
                print(np.asarray(labelbs))
                exit(0)
            for model_meta in model_meta_data:
                model_file = tf.train.latest_checkpoint(model_meta["dir"])
                path = FLAGS.analyse_path+"analyse/" + ([i for i in model_meta["dir"].split("/") if i][-1])

                num = 0
                s=path
                while os.path.exists(s):
                    num += 1
                    s = path + "_" + str(num)
                path = s + "/data"
                os.makedirs(path)
                k=[]
                for i in model_meta["iterations"]:
                    print(model_file)
                    nmodel_file = model_file[:model_file.rindex('model')] + 'model' + str(i)
                    if nmodel_file:
                        print("Restoring model weights from " + nmodel_file)
                        coord.request_stop()
                        coord.join(threads)

                        sess.close()
                        sess = tf.InteractiveSession()
                        tf.global_variables_initializer().run()

                        coord = tf.train.Coordinator()
                        threads = tf.train.start_queue_runners(coord=coord)
                        saver.restore(sess, nmodel_file)

                        if FLAGS.dream:
                            #sess.run(assign_weights(model.weights,model.fast_w))
                            fw=None
                            st=[]
                            lb=np.zeros(20)
                            np.random.seed(1)
                            for testi in range(FLAGS.num_testpoints):
                                print("testpoint:",testi)
                                lb = np.zeros(20)
                                input_d = np.ones((28,28))
                                if FLAGS.hard_coded_label:
                                    lb[0]=testi%20
                                    input_d = np.random.uniform(size=(28, 28))
                                else:
                                    lb = np.random.rand(20)
                                a=dream(model, sess, fw, metaval_input_tensors,lb, input_d)
                                fw=test(model, saver, sess, exp_string, data_generator, test_num_updates,
                                     layers=model_meta["layers"])
                                #print("store output in", path + '/deep_iter_' + str(i) + '.npz ...')
                                b=dream(model, sess, fw,metaval_input_tensors, lb,input_d)
                                st.append((a,b))
                            k.append(st)
                        else:
                            layerout, all_acc = test(model, saver, sess, exp_string, data_generator, test_num_updates,
                                                     layers=model_meta["layers"])
                            print("store output in", path + '/iter_' + str(i) + '.npz ...')
                            np.savez_compressed(path + '/iter_' + str(i) + '.npz',
                                                np.asarray({"layer_output": layerout, "all_acc": all_acc}))
                            print("store completed")
                if FLAGS.dream:
                    np.savez_compressed(path + '/deep_iter_' + str(i) + '.npz',
                                        np.asarray(k))
        else:
            test(model, saver, sess, exp_string, data_generator, test_num_updates)

if __name__ == "__main__":
    main()
