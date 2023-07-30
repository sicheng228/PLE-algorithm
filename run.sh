#!/bin/bash

export XBD_ENV=1
export XBD_HOOK_FOR_EMBEDDING_COLUMN=1
export XBD_CPP_OPS_LOG_LEVEL=0
export TF_CPP_MIN_VLOG_LEVEL=0


tf_config=`python -c "import json,os; print json.loads(os.getenv('TF_CONFIG')).get('cluster').get('ps')"`
if [ "$tf_config" = "None" ] ; then
unset TF_CONFIG
fi

cd ple

python online.py $* \
--model_id d_m_17814 \
--add_feature_filter True \
--scene_id notfeed \
--sample_groupid wb_oprd_supertopic_algo_dswCEi \
--sample_topic super-topic-ulike-realtime-sample-flat \
--topic_partitions 50 \
--model_deploy True \
--model_type ple \
#--mode predict \
#--predict_offline True \
