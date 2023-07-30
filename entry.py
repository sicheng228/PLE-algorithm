# coding=utf-8
import os
import tensorflow as tf
import sys
import logging
import xembedding as xbd
import time

sys.path.append("../../")
from xembedding.source.BaseSource import *
from xembedding.source.KafkaSource import KafkaSource
from xembedding.source.FileSource import FileSource
from model import PLE

logger = logging.getLogger('tensorflow')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"
formatter = logging.Formatter(
    '%(asctime)s - %(filename)s - %(levelname)s:%(message)s')
for h in logger.handlers:
    h.setFormatter(formatter)
tf.logging.set_verbosity(tf.logging.INFO)
pid = os.getpid()

print("<<<<<<<<<<<START<<<<XBD PID:%d" % (pid))

class ExportHook(tf.train.SessionRunHook):
    def __init__(self, export_xbd, interval_in_sec = 300):
        self.export_xbd = export_xbd
        self.interval_in_sec = interval_in_sec
        self.last_export_time = 0
    def after_run(self, run_context, run_values):
        curr_time = time.time()
        if curr_time - self.last_export_time >= self.interval_in_sec:
            self.export_xbd.export()
            self.last_export_time = curr_time


class xRuntime():
    def __init__(self, params, mode):
        self.params = params
        self.mode = mode
        stream_evaluator = None
        device_filters = None
        if 'runtime_stream_eval' in self.params and self.params['runtime_stream_eval'] is True:
            stream_evaluator = xbd.StreammingEvaluator(
                eval_dir=params['runtime_stream_eval_dir'],
                batch_size=params['runtime_batchsize'],
                task_name_list=self.params['task_names'],
                dequeue_cnt=100)

            device_filters = stream_evaluator.get_device_filters()
            self.params['runtime_evaluator'] = stream_evaluator
        else:
            device_filters = self.default_device_filter()
        session_config = tf.ConfigProto(allow_soft_placement=True, device_filters=device_filters)
        #############################estimator的config
        self.run_config = tf.estimator.RunConfig(keep_checkpoint_max=5, #文件最多保留个数
                                                 log_step_count_steps=100, #记录loss等的频率
                                                 session_config=session_config)
        self.params['is_chief'] = self.run_config.is_chief

        #############################准备数据
        if self.params['sample_data_schema'] is not None:
            if 'TF_CONFIG' in os.environ:
                if 'predict_offline' in self.params and self.params['predict_offline'] is True:
                    self.source = FileSource(
                        batchsize=self.params['runtime_batchsize'],
                        label_column=self.params['sample_label_column'] if mode == 'train' else None,
                        sample_addr=self.params['sample_addr'],
                        sample_schema=self.params['sample_data_schema'],
                        need_split=False,
                        train_data_ratio=0.9,
                        worker_num=self.run_config.num_worker_replicas,
                        worker_index=self.run_config.global_id_in_cluster,
                        decode_op='xbd',
                        map_parallel=10,
                        prefetch_size=5,
                        use_quote_delim=False)
                else:
                    self.source = KafkaSource(
                        batchsize=self.params['runtime_batchsize'],
                        label_column=self.params['sample_label_column'],
                        sample_addr=self.params['sample_addr'],
                        sample_schema=self.params['sample_data_schema'],
                        shuffle_size=False,
                        need_split=False,
                        train_data_ratio=0.9,
                        worker_num=self.run_config.num_worker_replicas,
                        worker_index=self.run_config.global_id_in_cluster,
                        #过滤feed场景数据
                        feature_filter=self.get_feature_filter() if self.params['add_feature_filter'] else None,
                        decode_op='xbd',
                        map_parallel=10,
                        prefetch_size=5,
                        use_quote_delim=False)
            else:
                print(" now TF_CONFIG not find ")
                if 'predict_offline' in self.params and self.params['predict_offline'] is True:
                    self.source = FileSource(
                        batchsize=self.params['runtime_batchsize'],
                        label_column=self.params['sample_label_column'] if mode == 'train' else None,
                        sample_addr=self.params['sample_addr'],
                        sample_schema=self.params['sample_data_schema'],
                        shuffle_size=False,
                        need_split=False,
                        train_data_ratio=0.9,
                        decode_op='xbd',
                        map_parallel=10,
                        prefetch_size=5,
                        use_quote_delim=False)
                else:
                    self.source = KafkaSource(
                        batchsize=self.params['runtime_batchsize'],
                        label_column=self.params['sample_label_column'],
                        sample_addr=self.params['sample_addr'],
                        sample_schema=self.params['sample_data_schema'],
                        feature_filter=self.get_feature_filter() if self.params['add_feature_filter'] else None,
                        shuffle_size=False,
                        need_split=False,
                        train_data_ratio=0.9,
                        decode_op='xbd',
                        map_parallel=10,
                        prefetch_size=5,
                        use_quote_delim=False)
        self.model = PLE(params=self.params)
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model.get_model_fn(),
            config=self.run_config,
            model_dir=self.params['model_dir'],
            params=self.params)

    def get_feature_filter(self):
        if self.params['scene_id'] == "notfeed":
            def feature_filter(features_dict, label):
                print("mp kafka filter:", features_dict)
                containerid = features_dict["c_containerid"]
                def func_true(): return tf.constant(True, dtype=tf.bool)
                def func_false(): return tf.constant(False, dtype=tf.bool)
                result = tf.reshape(tf.case(
                    {
                        tf.equal(containerid, 'board'): func_true,
                        tf.equal(containerid, 'pagelike'): func_true,
                        tf.equal(containerid, 'pagerelated'): func_true
                    },
                    default=func_false, exclusive=True), [])
                return result
            return feature_filter


    def default_device_filter(self):
        run_config_tmp = tf.estimator.RunConfig()
        device_filters = None
        if run_config_tmp.task_type == 'master':
            device_filters = ['/job:ps', '/job:master']
        elif run_config_tmp.task_type == 'chief':
            device_filters = ['/job:ps', '/job:chief']
        elif run_config_tmp.task_type == 'worker':
            device_filters = [
                '/job:ps',
                '/job:worker/task:%d' % run_config_tmp.task_id
            ]
        elif run_config_tmp.task_type == 'ps':
            device_filters = [
                '/job:ps', '/job:worker', '/job:chief', '/job:master'
            ]
        else:
            print(
                "If the task_type is `EVALUATOR` or something other than the ones in TaskType then don't set any "
                "device filters. "
            )
        return device_filters

    def train_and_eval(self):
        hooks = []

        chief_last_exit_hook = xbd.ChiefLastExitHook(self.run_config.num_worker_replicas, self.params['is_chief'])
        hooks.append(chief_last_exit_hook)
        if self.params['is_chief'] is True:
            chief_saver_hook = xbd.CheckpointSaverHook(
                checkpoint_dir=self.params['model_dir'],
                save_secs=900,
                checkpoint_basename='model.ckpt',
                need_checkpoint_loader=True,
                keep_weips_checkpoint_max=2)
            hooks.append(chief_saver_hook)

        train_spec = tf.estimator.TrainSpec(input_fn=lambda: self.source.get_input_fn(),hooks=hooks)

        exporter = tf.estimator.LatestExporter(
            'exporter_ulike_ple',
            self.model.get_serving_input_fn(),
            as_text=False,
            exports_to_keep=3)

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: self.source.get_input_fn(
            ),
            steps=10,
            start_delay_secs=20,  # start evaluating after N seconds
            throttle_secs=60,  # evaluate every N seconds
            exporters=exporter,
        )
        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)

    def predict(self):
        from xembedding.model.BaseModel import XBDPredictor

        out_fields = self.params['predict_columns']

        for task_name in self.params['task_names']:
            out_fields.append(task_name + '_pred')
            out_fields.append(task_name + '_label')

        predictor = XBDPredictor(self.run_config, self.estimator, model_id=self.params['model_name'],
                                 total_num=100000,
                                 batch=self.params['runtime_batchsize'])
        predictor.predict(input_fn=lambda: self.source.get_input_fn(), predict_version=None,
                          predict_keys=out_fields, predict_result_path=self.params['predict_path'],func_type='json')

    def run(self):
        if self.mode == "train":
            if 'runtime_evaluator' in self.params and self.params['runtime_evaluator'] is not None:
                my_evaluator = self.params['runtime_evaluator']
                exporter = xbd.XBDExporter(
                    estimator=self.estimator,
                    model_dir=self.params['model_dir'],
                    export_dir=self.params['model_export_dir'],
                    serving_input_receiver_fn=self.model.get_serving_input_fn(),
                    feature_spec=None if self.params['no_example'] else self.model.get_serving_input_fn(serving=False)(),
                    receiver_tensor=self.model.get_serving_input_fn(serving=False, raw_feature_tensor=True)() if self.params['no_example'] else None,
                    keep_max_savedmodel_num=1,)
                if self.params['model_deploy'] is True:
                    exporter.set_deploy(ps_list=self.params["servingps_list"], model_name=self.params["model_name"])
                    my_evaluator.set_export(exporter)
                my_evaluator.start()

            if self.params['train']:
                self.train_and_eval()

        if self.mode == "predict":
            self.predict()
