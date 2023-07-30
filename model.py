import os
import tensorflow as tf
from xembedding.model.BaseModel import *
from xembedding.feature.WeiFeature import *
import xembedding as xbd

class PLE(BaseModel):
    def __init__(self, **kwargs):
        self.params = kwargs['params']

    def dnn(self, mode, inputs, units, name):
        outputs = tf.layers.dense(
            inputs,
            units,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.params["model_l2"]),
            name=name,
            use_bias=True
        )
        outputs = tf.layers.batch_normalization(outputs, training=True if mode == tf.estimator.ModeKeys.TRAIN else False)
        if mode == tf.estimator.ModeKeys.TRAIN:
            outputs = tf.layers.dropout(outputs, rate=self.params['model_dropout_keep_deep'])
        outputs = tf.nn.relu(outputs)
        return outputs



    def cgc_net(self, inputs, name, mode, is_last=False):
        specific_experts_outputs = []
        for i in range(self.params['num_tasks']): #2
            for j in range(self.params['specific_expert_num']): #1
                expert_network = self.dnn(mode, inputs[i], self.params['expert_hidden_units'], name + '_task_' + self.params['task_names'][i] + '_expert_' + str(j))
                specific_experts_outputs.append(expert_network)
                # specific expert shape Tensor("Relu:0", shape=(?, 128), dtype=float32, device=/job:chief/task:0)
                print('specific expert shape {}'.format(expert_network))
        shared_experts_outputs = []
        for k in range(self.params['shared_expert_num']):
            expert_network = self.dnn(mode, inputs[-1], self.params['expert_hidden_units'], name + '_expert_shared' + str(k))
            shared_experts_outputs.append(expert_network)
            # share expert shape Tensor("Relu_4:0", shape=(?, 128), dtype=float32, device=/job:chief/task:0)
            print('share expert shape {}'.format(expert_network))

        cgc_outputs = []
        for i in range(self.params['num_tasks']):
            cur_expert_num = self.params['specific_expert_num'] + self.params['shared_expert_num']
            cur_experts = specific_experts_outputs[i*self.params['specific_expert_num']:(i+1)*self.params['specific_expert_num']] + shared_experts_outputs
            cur_experts = tf.stack(cur_experts, axis=1)
            # cgc experts shape Tensor("stack:0", shape=(?, 2, 128), dtype=float32, device=/job:chief/task:0)
            print('cgc experts shape {}'.format(cur_experts))

            gate_input = self.dnn(mode, inputs[i], self.params['gate_hidden_units'], name + '_gate_specific_' + self.params['task_names'][i])
            # gate input shape Tensor("Relu_5:0", shape=(?, 128), dtype=float32, device=/job:chief/task:0)
            print('gate input shape {}'.format(gate_input))
            gate_output = tf.layers.dense(
                gate_input,
                cur_expert_num, #2
                use_bias=False,
                activation=tf.nn.softmax,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.params["model_l2"]),
                name=name + '_gate_softmax_specific_' + self.params['task_names'][i]
            )
            # gate output shape Tensor("level_0_gate_softmax_specific_is_click/Softmax:0", shape=(?, 2), dtype=float32, device=/job:chief/task:0)
            print('gate output shape {}'.format(gate_output))
            gate_output = tf.expand_dims(gate_output, axis=-1)
            # gate output shape Tensor("ExpandDims:0", shape=(?, 2, 1), dtype=float32, device=/job:chief/task:0)
            print('gate output shape {}'.format(gate_output))
            gate_mul_output = tf.reduce_sum(tf.multiply(cur_experts, gate_output), axis=1)
            # gate output shape Tensor("Sum_1:0", shape=(?, 128), dtype=float32, device=/job:chief/task:0)
            print('gate output shape {}'.format(gate_mul_output))
            cgc_outputs.append(gate_mul_output)

        if not is_last:
            cur_expert_num = self.params['num_tasks'] * self.params['specific_expert_num'] + self.params['shared_expert_num']
            cur_experts = specific_experts_outputs + shared_experts_outputs
            cur_experts = tf.stack(cur_experts, 1)
            # not last expert shape Tensor("stack_2:0", shape=(?, 3, 128), dtype=float32, device=/job:chief/task:0)
            print('not last expert shape {}'.format(cur_experts))
            gate_input = self.dnn(mode, inputs[-1], self.params['gate_hidden_units'], name + '_gate_shared')
            # share gate input shape Tensor("Relu_5:0", shape=(?, 128), dtype=float32, device=/job:chief/task:0)
            print('share gate input shape {}'.format(gate_input))
            gate_output = tf.layers.dense(
                gate_input,
                cur_expert_num, #3
                use_bias=False,
                activation=tf.nn.softmax,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.params["model_l2"]),
                name=name + '_gate_softmax_share'
            )
            # share gate output shape Tensor("level_0_gate_softmax_share/Softmax:0", shape=(?, 3), dtype=float32, device=/job:chief/task:0)
            print('share gate output shape {}'.format(gate_output))
            gate_output = tf.expand_dims(gate_output, axis=2)
            # share gate output shape Tensor("ExpandDims_2:0", shape=(?, 3, 1), dtype=float32, device=/job:chief/task:0)
            print('share gate output shape {}'.format(gate_output))
            gate_mul_ouput = tf.reduce_sum(tf.multiply(cur_experts, gate_output), axis=1)
            # share gate output shape Tensor("Sum_1:0", shape=(?, 128), dtype=float32, device=/job:chief/task:0)
            print('share gate output shape {}'.format(gate_mul_output))
            cgc_outputs.append(gate_mul_ouput)
        return cgc_outputs



    def model_fn(self, features, labels, mode, params):
        print('features {}'.format(features))
        nolimit_features, limit_features = self.params['group_column']
        select_features = [xbd.feature_column.wei_embedding_column(
                                                    limit_features,
                                                    mean=0,
                                                    std=0.01,
                                                    combiner='mean',
                                                    steps_to_live=self.params['steps_to_live'],
                                                    dimension=self.params['model_embedding_dimension'],
                                                    learning_rate=self.params['learning_rate']
                                                )]

        select_features.extend([xbd.feature_column.wei_embedding_column(
                                                    nolimit_features,
                                                    mean=0,
                                                    std=0.01,
                                                    combiner='mean',
                                                    steps_to_live=None,
                                                    dimension=self.params['model_embedding_dimension'],
                                                    learning_rate=self.params['learning_rate']
                                                )])
        # embedding shape Tensor("Reshape_2:0", shape=(?, 1152), dtype=float32, device=/job:chief/task:0)
        feature_embedding = tf.reshape(xbd.feature_column.input_layer(features, select_features), shape=(-1, len(self.params['feature_columns'])*self.params['model_embedding_dimension']))
        ple_inputs = [feature_embedding] * (self.params['num_tasks'] + 1)
        print('ple inputs shape {}'.format(ple_inputs))
        ple_outputs = []
        for i in range(self.params['num_levels']): #2
            if i == self.params['num_levels'] - 1:
                ple_outputs = self.cgc_net(ple_inputs, 'level_' + str(i), mode, is_last=True)
            else:
                ple_outputs = self.cgc_net(ple_inputs, 'level_' + str(i), mode, is_last=False)
                ple_inputs = ple_outputs
        # ple output shape [<tf.Tensor 'Sum_3:0' shape=(?, 128) dtype=float32>, <tf.Tensor 'Sum_4:0' shape=(?, 128) dtype=float32>]
        print('ple output shape {}'.format(ple_outputs))
        prediction_keys = {}
        for task_name, ple_output in zip(self.params['task_names'], ple_outputs):
            # tower output shape Tensor("Relu_11:0", shape=(?, 128), dtype=float32, device=/job:chief/task:0)
            tower_output = self.dnn(mode, ple_output, self.params['tower_hidden_units'], 'tower_' + task_name)
            logit = tf.layers.dense(
                tower_output,
                1,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(params["model_l2"]),
                name='tower_dense_' + task_name
            )
            print('logit shape {}'.format(logit))
            logit = tf.reshape(logit, (-1, 1))
            prediction = tf.reshape(tf.nn.sigmoid(logit), (-1, 1))
            predicted_classes = tf.to_int32(prediction > 0.5)
            prediction_keys[task_name + '_pred'] = prediction
            prediction_keys[task_name + '_class_ids'] = predicted_classes
            prediction_keys[task_name + '_logits'] = logit

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = prediction_keys
            for task_name in self.params['task_names']:
                if task_name in features:
                    predictions[task_name + '_label'] = features[task_name]
            if self.params['mode'] == 'predict':
                for task_name in self.params['task_names']:
                    predictions.pop(task_name + '_logits')
                    predictions.pop(task_name + '_class_ids')
                for key in self.params['predict_columns']:
                    if key in features:
                        predictions[key] = features[key]
            export_keys = {}
            for task_name in self.params['task_names']:
                export_keys[task_name] = prediction_keys[task_name + '_pred']
            export_outputs = {
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    tf.estimator.export.PredictOutput(export_keys)
            }
            return tf.estimator.EstimatorSpec(
                mode,
                predictions=predictions,
                export_outputs=export_outputs,
                prediction_hooks=[xbd.InitEmbeddingHook()])
        loss = 0
        for task_name in self.params['task_names']:
            if task_name in features:
                task_label = tf.reshape(features[task_name], (-1, 1))
                # task label type Tensor("Reshape_7:0", shape=(?,), dtype=float32, device=/job:chief/task:0)
                task_logit = prediction_keys[task_name + '_logits']
                # task_label = tf.Print(task_label, [tf.shape(task_label), task_label], 'task_label', summarize=10)
                # task_logit = tf.Print(task_logit, [tf.shape(task_logit), task_logit], 'task_logit', summarize=10)
                # logit shape Tensor("Reshape_3:0", shape=(?,), dtype=float32, device=/job:chief/task:0)
                task_loss = tf.losses.sigmoid_cross_entropy(task_label, task_logit)
                loss += task_loss
                prediction_keys[task_name + '_loss'] = task_loss
                prediction_keys[task_name + '_pos_num'] = tf.reduce_sum(task_label)
                prediction_keys[task_name + '_label'] = task_label
        print('prediction_keys {}'.format(prediction_keys))
        def eval_fn():
            metrics = {}
            for task_name in self.params['task_names']:
                metrics[task_name + '_accuracy'] = tf.metrics.accuracy(labels=prediction_keys[task_name + '_label'], predictions=prediction_keys[task_name + '_class_ids'], name=task_name + '_accuracy')
                metrics[task_name + '_auc'] = tf.metrics.auc(labels=prediction_keys[task_name + '_label'], predictions=prediction_keys[task_name + '_pred'], name=task_name + '_auc')
            return metrics

        if mode == tf.estimator.ModeKeys.EVAL:
            metrics = eval_fn()
            hook_msg = {}
            for k, v in metrics.items():
                hook_msg[k] = v[0]
            for task_name in self.params['task_names']:
                hook_msg[task_name + '_pos_num'] = prediction_keys[task_name + '_pos_num']
            eval_hook = tf.train.LoggingTensorHook(hook_msg, every_n_iter=1)
            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                eval_metric_ops=metrics,
                evaluation_hooks=[eval_hook,xbd.InitEmbeddingHook()])

        assert mode == tf.estimator.ModeKeys.TRAIN
        metrics = eval_fn()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        for k, v in metrics.items():
            update_ops.append(v[1])
        hook_msg = {}
        for task_name in self.params['task_names']:

            hook_msg[task_name + '_loss'] = prediction_keys[task_name + '_loss']
            hook_msg[task_name + '_pos_num'] = prediction_keys[task_name + '_pos_num']
            hook_msg[task_name + '_auc'] = metrics[task_name + '_auc'][0]
            hook_msg[task_name + '_label_mean'] = tf.reduce_mean(prediction_keys[task_name + '_label'])
            hook_msg[task_name + '_pred_mean'] = tf.reduce_mean(prediction_keys[task_name + '_class_ids'])

            tf.summary.scalar(task_name + '_loss', prediction_keys[task_name + '_loss'])
            tf.summary.scalar(task_name + '_label_mean', hook_msg[task_name + '_label_mean'])
            tf.summary.scalar(task_name + '_pred_mean', hook_msg[task_name + '_pred_mean'])
            tf.summary.scalar(task_name + '_auc', metrics[task_name + '_auc'][0])

        hook_msg['global_step'] = tf.train.get_global_step()
        hook_msg['loss'] = loss
        train_hook = tf.train.LoggingTensorHook(hook_msg, every_n_iter=100)
        hooks = [train_hook]
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdagradOptimizer(learning_rate=params["learning_rate"])
            if self.params['runtime_is_sync'] is True:
                optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=3, total_num_replicas=3)
                sync_replicas_hook = optimizer.make_session_run_hook(self.params['is_chief'])
                hooks.append(sync_replicas_hook)
            optimizer = xbd.DistributedOptimizer(optimizer)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())
        hooks.append(xbd.InitEmbeddingHook())
        if 'runtime_evaluator' in self.params and self.params['runtime_evaluator'] is not None:
            stream_evaluator = self.params['runtime_evaluator']
            stream_evaluator_hook = stream_evaluator.get_hook(
                    [prediction_keys[each + '_label'] for each in self.params['task_names']],
                    [prediction_keys[each + '_pred'] for each in self.params['task_names']],
                    self.params['runtime_batchsize']
                )
            hooks.append(stream_evaluator_hook)
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op,
            training_chief_hooks=[xbd.RegisterEmbeddingHook(),
                                  GenFeatureConfHook([self.params['model_root_dir'],
                                                      os.path.join(self.params['model_root_dir'],'model_version=%s' % self.params['model_version'])],
                                                     feature_spec=self.get_serving_input_fn(False)())
                                  ],
            training_hooks=hooks)

    def get_model_fn(self):
        return self.model_fn

    def get_serving_input_fn(self, serving=True, raw_feature_tensor=False):
        def serving_input_receiver_fn():
            tf.add_to_collection("IS_EXAMPLE","1")
            feature_columns = [
                x for x in [ x for x in self.params['feature_columns']]
                if isinstance(x, xbd.feature_column.WeiFeatureColumn) is False
            ]
            wei_feature_columns = [
                x for x in [ x for x in self.params['feature_columns']]
                if isinstance(x, xbd.feature_column.WeiFeatureColumn) is True
            ]

            feature_spec = {}
            for column in feature_columns:
                config = column._parse_example_spec
                for key, value in six.iteritems(config):
                    print("feature_column key:%s" % (key))
                    feature_spec[key] = tf.FixedLenFeature([], dtype=tf.string)
            for column in wei_feature_columns:
                key = column.key_
                feature_spec[key] = tf.FixedLenFeature([], dtype=tf.string)

            print("xxxserving_input_reciver_fnxxxxx:%s" % (feature_spec))
            if not serving:
                return feature_spec
            receiver_tensors = {}
            for f in feature_spec.keys():
                receiver_tensors[f] = tf.placeholder(dtype=tf.string, name=f)
            return tf.estimator.export.ServingInputReceiver(receiver_tensors, receiver_tensors)

        def serving_input_receiver_fn_raw_tensor():
            tf.add_to_collection("IS_EXAMPLE","0")
            features, receiver_tensors = xbd.get_receiver_tensors_from_dataschema_v2(
                [ x for x in self.params['feature_columns']],
                self.params['sample_data_schema'])
            if not serving:
                return receiver_tensors
            return tf.estimator.export.ServingInputReceiver(features=features,
                                                            receiver_tensors=receiver_tensors)
        if raw_feature_tensor:
            return  serving_input_receiver_fn_raw_tensor
        else:
            return serving_input_receiver_fn
