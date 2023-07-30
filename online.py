# coding=utf-8
# time: 2022/11/17 4:12 下午
# author: shaojun7

import sys
sys.path.append("../../")

import os
import xembedding as xbd
from xembedding.source.BaseSource import *
import datetime
from entry import xRuntime
import dataschema
import feature

class XBDProcess(object):

    def __init__(self, args):
        self.params = {}

        self.params['mode'] = args.mode
        self.params['model_type'] = args.model_type
        self.params['model_id'] = args.model_id
        self.params['model_dropout_keep_deep'] = args.model_dropout_keep_deep
        self.params['learning_rate'] = args.learning_rate
        self.params['predict_offline'] = args.predict_offline
        self.params['model_l2'] = args.model_l2
        self.params['model_embedding_dimension'] = args.model_embedding_dimension
        self.params['num_tasks'] = args.num_tasks
        self.params['num_levels'] = args.num_levels
        self.params['specific_expert_num'] = args.specific_expert_num
        self.params['shared_expert_num'] = args.shared_expert_num
        self.params['expert_hidden_units'] = args.expert_hidden_units
        self.params['gate_hidden_units'] = args.gate_hidden_units
        self.params['tower_hidden_units'] = args.tower_hidden_units
        self.params['task_names'] = args.task_names.split(',')
        self.params['steps_to_live'] = args.steps_to_live
        self.params['model_deploy'] = args.model_deploy
        self.params['runtime_batchsize'] = args.runtime_batchsize
        self.params['model_version'] = args.model_version
        self.params['model_name'] = args.model_id

        self.params['add_feature_filter'] = args.add_feature_filter
        self.params['scene_id'] = args.scene_id

        self.params['sample_label_column'] = 'is_realread'
        self.params["feature_columns"], self.params["group_column"] = feature.feature_process(self.params)
        self.params['sample_data_schema'] = dataschema.get_data_schema()
        self.params['predict_columns'] = dataschema.get_prediction_schema()

        self.base_dir = 'hdfs://ns-fed/wbml/wb_oprd_supertopic_algo/model/zoo/model_id=' + self.params['model_name']
        self.params['model_base_dir'] = self.base_dir
        self.params['model_root_dir'] = self.base_dir
        self.params['model_dir'] = self.base_dir + '/checkpoint'
        self.params['model_export_dir'] = self.base_dir + '/version'
        self.params['model_profile_dir'] = self.base_dir + '/profile'

        self.params['no_example'] = True

        self.params['runtime_stream_eval_dir'] = self.base_dir + '/eval'
        self.params['runtime_is_sync'] = False
        self.params['runtime_stream_eval'] = True
        self.params['train'] = True

        # sample config
        self.params["sample_topic"] = args.sample_topic
        self.params["topic_partitions"] = args.topic_partitions
        self.params['sample_groupid'] = args.sample_groupid
        self.params["sample_group_user"] = 'wb_oprd_supertopic_algo'
        self.params["sample_group_password"] = 'KVw2eKSU7YUiFe'
        self.params["bootstrap_server"] = "kfk60.c12.al.sina.com.cn:9110,kfk11.c12.al.sina.com.cn:9110,kfk47.c12.al.sina.com.cn:9110,kfk22.c12.al.sina.com.cn:9110,kfk58.c12.al.sina.com.cn:9110"
        self.params['servingps_list'] = [
            {
                "cluster": "zk://uf3001.zks.sina.com.cn:12185,uf3002.zks.sina.com.cn:12185,uf3003.zks.sina.com.cn:12185,uf3004.zks.sina.com.cn:12185,uf3005.zks.sina.com.cn:12185",
                "repo": "weips-v6.3/supertopic-xbd-serving/youfu"
            },
            {
                "cluster": "zk://uf3001.zks.sina.com.cn:12185,uf3002.zks.sina.com.cn:12185,uf3003.zks.sina.com.cn:12185,uf3004.zks.sina.com.cn:12185,uf3005.zks.sina.com.cn:12185",
                "repo": "weips-v6.3/supertopic-xbd-serving2/youfu"
            }
        ]

    def process(self):
        mode = self.params['mode']
        print(mode)

        xbd.init_v2(self.params['model_name'], 'train', async_push_pull=True)

        if mode == 'train':
            if 'TF_CONFIG' in os.environ or self.params['runtime_stream_eval'] is True:
                self.params['runtime_stream_eval'] = True
                self.params['runtime_stream_eval_dir'] = self.params['model_base_dir'] + '/eval'
                self.params['runtime_stream_eval_device'] = '/job:xevaluator/task:0'

            print(self.params)

            self.params["sample_addr"] = DataAddr(
                source_type=SourceType.KAFKA,
                bootstrap_server=self.params['bootstrap_server'],
                topics=self.params['sample_topic'],
                partitions=self.params["topic_partitions"],
                offset="-1",
                groupid=self.params['sample_groupid'],
                username=self.params['sample_group_user'],
                password=self.params['sample_group_password'])

        elif mode == 'predict':
            self.params['runtime_batchsize'] = 15
            self.params['predict_records'] = 10240
            columns, _ = feature.get_feature_conf()
            self.params['predict_columns'] = columns
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            filename = 'predict_file.txt.' + nowtime
            self.params['predict_file'] = os.path.join(self.params['model_dir'], filename)
            self.params['predict_path'] = os.path.join(self.params['model_dir'], 'ple_predict')
            self.params["sample_addr"] = DataAddr(
                source_type=SourceType.FILE,
                file_path= 'hdfs://ns-fed/wbml/kafka2hdfs/super-topic-ulike-realtime-sample-flat',
                compression_type='GZIP',
                dirs_pattern='dt=20230701/hour=12/.*',
            )
            print("+++++++++++++++")
            print(self.params)
            print("+++++++++++++++")

        runtime = xRuntime(self.params, mode)
        runtime.run()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='ple')
    parser.add_argument('--model_id', type=str, default='d_m_16502')
    parser.add_argument('--model_version', type=str, default='001')
    parser.add_argument('--sample_groupid', type=str, default='wb_oprd_supertopic_algo_YY7Pbg')
    parser.add_argument('--sample_topic', type=str, default='super-topic-ulike-realtime-sample-flat')
    parser.add_argument('--model_dropout_keep_deep', type=float, default=0.5)
    parser.add_argument('--predict_offline', type=bool, default=False)
    parser.add_argument('--model_l2', type=float, default=0.001)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--model_embedding_dimension', type=int, default=32)
    parser.add_argument('--num_tasks', type=int, default=2)
    parser.add_argument('--num_levels', type=int, default=2)
    parser.add_argument('--specific_expert_num', type=int, default=1)
    parser.add_argument('--shared_expert_num', type=int, default=1)
    parser.add_argument('--expert_hidden_units', type=int, default=128)
    parser.add_argument('--gate_hidden_units', type=int, default=128)
    parser.add_argument('--tower_hidden_units', type=int, default=64)
    parser.add_argument('--task_names', type=str, default='is_click,is_addatten')
    parser.add_argument('--steps_to_live', type=int, default=2*3600*24*2)
    parser.add_argument('--model_deploy', type=bool, default=False)
    parser.add_argument('--add_feature_filter', type=bool, default=False)
    parser.add_argument('--scene_id', type=str, default='other')
    parser.add_argument('--topic_partitions', type=int, default=75)
    parser.add_argument('--runtime_batchsize', type=int, default=1024)
    args = parser.parse_args()
    print(args)
    XBDProcess(args).process()
