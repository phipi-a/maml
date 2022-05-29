""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS
if FLAGS.analyse:
    tf.random.set_random_seed(2)
try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)


from utils import mse, xent, conv_block, normalize, dream_loss




class MAML:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5):
        """ must call construct_model() after initializing MAML! """
        self.new_weights = {}
        #self.store_weight=tf.Variable(False)
        self.set_random_op = [None] * 4
        self.number_of_n = 20
        #self.input_dream=tf.Variable(tf.clip_by_value(tf.random.normal([28,28], 0, 1, tf.float32, seed=1),0,1), name="input_dream")

        self.dim_input = dim_input
        self.dim_output = dim_output
        if not FLAGS.second_dense:
            self.number_of_n = self.dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.classification = False
        self.test_num_updates = test_num_updates
        if FLAGS.datasource == 'sinusoid':
            self.dim_hidden = [40, 40]
            self.loss_func = mse
            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights
        elif FLAGS.datasource == 'omniglot' or FLAGS.datasource == 'miniimagenet':
            self.loss_func = xent
            self.classification = True
            if FLAGS.conv:
                self.dim_hidden = FLAGS.num_filters
                self.forward = self.forward_conv
                self.construct_weights = self.construct_conv_weights
            else:
                self.dim_hidden = [256, 128, 64, 64]
                self.forward = self.forward_fc
                self.construct_weights = self.construct_fc_weights
            if FLAGS.datasource == 'miniimagenet':
                self.channels = 3
            else:
                self.channels = 1
            self.img_size = int(np.sqrt(self.dim_input / self.channels))
        else:
            raise ValueError('Unrecognized data source.')

    def construct_model(self, input_tensors=None, prefix='metatrain_', no_fw=True):
        # a: training data for inner gradient, b: test data for meta gradient
        if 'dream' in prefix:

            self.input_dream  = self.construct_input_dream()


            self.label_dream = tf.placeholder(tf.float32, shape=(20))
            if not no_fw:
                self.old_w = [tf.placeholder(tf.float32, shape=val.shape) for val in list(self.weights.values())]
                for i, val in enumerate(list(self.weights.values())):
                    val.assign(self.old_w[i])

            loss,self.layerwise_out_dream=self.forward(self.input_dream, self.weights, reuse=tf.AUTO_REUSE, layerwise=True)
            loss ,out = dream_loss(loss, self.label_dream,self.input_dream)
            self.out = out
            # input_dream = tf.Print(input_dream,[input_dream],"pre:")
            opt = tf.train.AdamOptimizer(
                FLAGS.meta_lr)

            self.dream_op = opt.minimize(loss, var_list=[self.input_dream])

            # input_dream = tf.Print(input_dream, [input_dream], "post:")
            self.val_loss_dream = loss
            return

        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as training_scope:
            if 'weights' in dir(self) or 'input_dream' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights

                #input_dream=tf.Print(input_dream,[input_dream],"reuse")
            else:

                # Define the weights
                self.weights = weights = self.construct_weights()

                #input_dream=tf.Print(input_dream,[input_dream],"construct")



            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            outputbs = [[]] * num_updates
            lossesb = [[]] * num_updates
            accuraciesb = [[]] * num_updates

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []
                layeroutput=[]

                if self.classification:
                    task_accuraciesb = []
                #
                # inputb = tf.Print(inputb, [tf.math.reduce_mean(self.weights["b2"])], "weights")
                task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
                task_lossa = self.loss_func(task_outputa, labela)
                # inputb = tf.Print(inputb, [tf.math.reduce_mean(inputa)], "input")
                # inputb=tf.Print(inputb,[tf.math.reduce_mean(task_outputa)],"output")


                output, output_la = self.forward(inputb, weights, reuse=True, layerwise=True)
                layeroutput.append(output_la)

                grads = tf.gradients(task_lossa, list(weights.values()))
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                updating_keys = list(weights.keys())[
                                FLAGS.freezing_layer * 2:]  # weights.keys() ['conv1', 'b1', 'conv2', 'b2', 'conv3', 'b3', 'conv4', 'b4', 'w5', 'b5']
                if FLAGS.no_dense_update:
                    updating_keys=[k for k in updating_keys if k !="b5" and k != "w5"]
                update_lr_new = dict(
                    zip(weights.keys(), [self.update_lr if k in updating_keys else 0 for k in weights.keys()]))
                #                print(update_lr_new)
                fast_weights = dict(
                    zip(weights.keys(), [weights[key] - update_lr_new[key] * gradients[key] for key in weights.keys()]))
                #validation
                output,output_la = self.forward(inputb, fast_weights, reuse=True, layerwise=True)
                layeroutput.append(output_la)

                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))

                    fast_weights = dict(zip(fast_weights.keys(),
                                            [fast_weights[key] - update_lr_new[key] * gradients[key] for key in
                                             fast_weights.keys()]))
                    #validation

                    output,output_la = self.forward(inputb, fast_weights, reuse=True ,layerwise=True)
                    layeroutput.append(output_la)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]
                if FLAGS.dream and 'val' in prefix:
                    pass

                    #self.weights=fast_weights
                if self.classification:
                    task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1),
                                                                 tf.argmax(labela, 1))
                    for j in range(num_updates):
                        task_accuraciesb.append(
                            tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1),
                                                        tf.argmax(labelb, 1)))
                    task_output.extend([task_accuracya, task_accuraciesb, layeroutput,list(fast_weights.values())])

                return task_output

            # weights['conv1'] = tf.Print(weights['conv1'], [weights['w5']], "A: ")
            if FLAGS.reset_last_layer_training_and_test:
                dim = [self.dim_hidden, self.number_of_n]
                if FLAGS.datasource == 'miniimagenet':
                    dim = [self.dim_hidden * 5 * 5, self.dim_output]

                new_random = tf.random_normal(dim)
                self.set_random_op[0] = self.new_weights["w5"].assign(new_random)
                self.set_random_op[1] = self.new_weights["b5"].assign(tf.zeros([self
                                                                               .number_of_n]))
                weights["w5"] = self.new_weights["w5"]
                weights["b5"] = self.new_weights["b5"]
                if FLAGS.second_dense:
                    new_random = tf.random_normal([self.number_of_n, self.dim_output])
                    self.set_random_op[2] = self.new_weights["w6"].assign(new_random)
                    self.set_random_op[3] = self.new_weights["b6"].assign(tf.zeros([self.dim_output]))
                    weights["w6"] = self.new_weights["w6"]
                    weights["b6"] = self.new_weights["b6"]
            if FLAGS.reset_last_layer_test and ('train' not in prefix):

                dim=[self.dim_hidden, self.number_of_n]
                if FLAGS.datasource == 'miniimagenet':
                    dim=[self.dim_hidden * 5 * 5, self.dim_output]

                new_random = tf.random_normal(dim)
                self.set_random_op[0] = self.new_weights["w5"].assign(new_random)
                self.set_random_op[1] = self.new_weights["b5"].assign(tf.zeros([self.number_of_n]))
                weights["w5"] = self.new_weights["w5"]
                weights["b5"] = self.new_weights["b5"]

                if FLAGS.second_dense:
                    new_random = tf.random_normal([self.number_of_n, self.dim_output])
                    self.set_random_op[2] = self.new_weights["w6"].assign(new_random)
                    self.set_random_op[3] = self.new_weights["b6"].assign(tf.zeros([self.dim_output]))
                    weights["w6"] = self.new_weights["w6"]
                    weights["b6"] = self.new_weights["b6"]
            # weights['conv1'] = tf.Print(weights['conv1'], [weights['w5']], "B: ")
            if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)




            out_dtype = [tf.float32, [tf.float32] * num_updates, tf.float32, [tf.float32] * num_updates]#,[[tf.float32]*layer_output_size * num_layers]* num_updates


            if self.classification:
                out_dtype.extend([tf.float32, [tf.float32] * num_updates,[[tf.float32] * 5 ] * (num_updates+1),[i.dtype for i in list(weights.values())]])
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb),
                               dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            if self.classification:
                outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb,output_la_iter , fw = result
            else:
                outputas, outputbs, lossesa, lossesb,output_la_iter = result

        ## Performance & Optimization
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j
                                                  in range(num_updates)]
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs
            if self.classification:
                self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.total_accuracies2 = total_accuracies2 = [
                    tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            if FLAGS.freezing_meta:
                freeze_layers=["w5","b5"]
            else:
                freeze_layers=[]
            var_list=[val for key,val in weights.items() if key not in freeze_layers]
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1, var_list=var_list)

            if FLAGS.metatrain_iterations > 0:
                if FLAGS.freezing_meta:
                    freeze_layers = ["w5", "b5"]
                else:
                    freeze_layers = []
                var_list = [val for key, val in weights.items() if key not in freeze_layers]
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                if not FLAGS.freezing_meta:
                    self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates - 1])
                    if FLAGS.datasource == 'miniimagenet':
                        gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs if grad!=None]
                    self.metatrain_op = optimizer.apply_gradients(gvs)
                else:
                    self.metatrain_op = optimizer.minimize(self.total_losses2[FLAGS.num_updates - 1], var_list=var_list)
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size)
                                                          for j in range(num_updates)]
            if self.classification:
                self.fw=fw
                self.output_layervise = output_la_iter
                self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(
                    FLAGS.meta_batch_size)
                self.metaval_total_accuracies2 = total_accuracies2 = [
                    tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]



        ## Summaries
        #sum_range = FLAGS.num_updates if 'train' in prefix else self.test_num_updates
        tf.summary.scalar(prefix + 'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix + 'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):#TODO edit num_update
            tf.summary.scalar(prefix + 'Post-update loss, step ' + str(j + 1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix + 'Post-update accuracy, step ' + str(j + 1), total_accuracies2[j])

    ### Network construction functions (fc networks and conv networks)
    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1, len(self.dim_hidden)):
            weights['w' + str(i + 1)] = tf.Variable(
                tf.truncated_normal([self.dim_hidden[i - 1], self.dim_hidden[i]], stddev=0.01))
            weights['b' + str(i + 1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w' + str(len(self.dim_hidden) + 1)] = tf.Variable(
            tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b' + str(len(self.dim_hidden) + 1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    def forward_fc(self, inp, weights, reuse=False):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
        for i in range(1, len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)],
                               activation=tf.nn.relu, reuse=reuse, scope=str(i + 1))
        return tf.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)]) + weights[
            'b' + str(len(self.dim_hidden) + 1)]


    def construct_input_dream(self):
        init=tf.random_uniform_initializer(minval=0., maxval=1.)
        input_dream=tf.Variable(init([28,28], dtype=tf.float32))
        return input_dream

    def construct_conv_weights(self):
        weights = {}

        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3

        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
        if FLAGS.datasource == 'miniimagenet':
            # assumes max pooling
            weights['w5'] = tf.get_variable('w5', [self.dim_hidden * 5 * 5, self.dim_output],
                                            initializer=fc_initializer)
            self.new_weights["w5"] = tf.Variable(tf.random_normal([self.dim_hidden * 5 * 5, self.dim_output]), name='w5')
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
            self.new_weights["b5"]= tf.Variable(tf.zeros([self.dim_output]), name='b5')
        else:

            weights['w5'] = tf.Variable(tf.random_normal([self.dim_hidden, self.number_of_n]), name='w5')
            self.new_weights["w5"] = tf.Variable(tf.random_normal([self.dim_hidden, self.number_of_n]), name='w5')
            weights['b5'] = tf.Variable(tf.zeros([self.number_of_n]), name='b5')
            self.new_weights["b5"] = tf.Variable(tf.zeros([self.number_of_n]), name='b5')
            if FLAGS.second_dense:
                weights['w6'] = tf.Variable(tf.random_normal([self.number_of_n, self.dim_output]), name='w6')
                self.new_weights["w6"] = tf.Variable(tf.random_normal([self.number_of_n, self.dim_output]), name='w6')
                weights['b6'] = tf.Variable(tf.zeros([self.dim_output]), name='b6')
                self.new_weights["b6"] = tf.Variable(tf.zeros([self.dim_output]), name='b6')
        return weights

    def forward_conv(self, inp, weights, reuse=False, scope='', layerwise=False):
        # reuse is for the normalization parameters.
        la_o=[]
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope + '0')
        la_o.append(hidden1)
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1')
        la_o.append(hidden2)
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2')
        la_o.append(hidden3)
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3')

        if FLAGS.datasource == 'miniimagenet':
            # last hidden layer is 6x6x64-ish, reshape to a vector
            hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])
        else:
            hidden4 = tf.reduce_mean(hidden4, [1, 2])
        la_o.append(hidden4)
        hidden5 = tf.matmul(hidden4, weights['w5']) + weights['b5']
        la_o.append(hidden5)

        if not FLAGS.second_dense:
            if layerwise:
                return la_o[-1], la_o
            else:
                return la_o[-1]
        la_o.append(tf.matmul(hidden5, weights['w6']) + weights['b6'])

        if layerwise:
            return la_o[-1], la_o
        else:
            return la_o[-1]

