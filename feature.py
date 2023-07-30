# coding=utf-8
# time: 2022/11/17 4:12 下午
# author: shaojun7

import xembedding as xbd
import tensorflow as tf

MAX_HASH_SIZE = 5000000

def feature_process(params):
    columns_ = []
    column_limit_duration = []
    # ============================= USER FEATURE START  16=============================
    # wu211 用户自填性别
    u_wu211 = xbd.feature_column.PickcatsWithVocabularyList(key='u_wu211', vocabulary_list=['0','1','2'])
    columns_.append(u_wu211)
    # wu212 用户年龄
    u_wu212 = xbd.feature_column.BucketizedColumn(key='u_wu212', boundaries=[0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 200])
    columns_.append(u_wu212)
    # wu215 用户挖掘年龄
    u_wu215 = xbd.feature_column.PickcatsWithVocabularyList(key='u_wu215', vocabulary_list=['00s','10s','90s','80s','70s','60s','50s','40s'])
    columns_.append(u_wu215)
    # wu217 阅读者v类型
    u_wu217 = xbd.feature_column.PickcatsWithVocabularyList(key='u_wu217', vocabulary_list=['0','1','2','3','4','5','6','7','8','9','10'])
    columns_.append(u_wu217)
    # ou101 uid自填省份
    u_ou101 = xbd.feature_column.CategoricalColumnWithHashBucket( key = 'u_ou101', hash_bucket_size=50)
    columns_.append(u_ou101)
    # ou102 uid自填城市
    u_ou102 = xbd.feature_column.CategoricalColumnWithHashBucket( key = 'u_ou102', hash_bucket_size=2000 )
    columns_.append(u_ou102)
    
    # wu2117 阅读者登录频次等级
    u_wu2117 = xbd.feature_column.PickcatsWithVocabularyList(key='u_wu2117', vocabulary_list=['0','1','2','3','4','5','6','7','8','9','10'])
    columns_.append(u_wu2117)
    # wu2043 用户手机型号
    u_wu2043 = xbd.feature_column.CategoricalColumnWithHashBucket( key='u_wu2043', hash_bucket_size=100)
    columns_.append(u_wu2043)
    # wu2123 新标签体系用户三级标签长期兴趣
    u_wu2123 = xbd.feature_column.PickcatsWithHashBucket( key = 'u_wu2123',
                                                          inter_delimiter='|',
                                                          intra_delimiter='@',
                                                          hash_bucket_size=1000 * 10000)
    columns_.append(u_wu2123)
    # wu21136 新标签体系用户二级标签长期兴趣
    u_wu21136 = xbd.feature_column.PickcatsWithHashBucket( key = 'u_wu21136',
                                                           inter_delimiter='|',
                                                           intra_delimiter='@',
                                                           hash_bucket_size=50 * 10000)
    columns_.append(u_wu21136)

    # 新特征体系用户一级特征长期兴趣
    u_wu21135 = xbd.feature_column.PickcatsWithHashBucket( key = 'u_wu21135',
                                                           inter_delimiter='|',
                                                           intra_delimiter='@',
                                                           hash_bucket_size=10 * 10000)
    columns_.append(u_wu21135)

    # 最近访问超话
    u_so1102 = xbd.feature_column.PickcatsWithHashBucket( key = 'u_so1102',
                                                          inter_delimiter='|',
                                                          hash_bucket_size=MAX_HASH_SIZE)
    column_limit_duration.append(u_so1102)
    columns_.append(u_so1102)

    # 用户关注uid列表
    u_ff1001 = xbd.feature_column.PickcatsWithHashBucket( key = 'u_ff1001',
                                                          inter_delimiter='|',
                                                          hash_bucket_size=MAX_HASH_SIZE*10)
    columns_.append(u_ff1001)
    column_limit_duration.append(u_ff1001)
    # 用户常用超话
    u_so1101 = xbd.feature_column.PickcatsWithHashBucket( key = 'u_so1101',
                                                          inter_delimiter='|',
                                                          hash_bucket_size=MAX_HASH_SIZE)
                                                        
    columns_.append(u_so1101)
    column_limit_duration.append(u_so1101)

    # 实时用户最近关注Top50
    u_wu261079 = xbd.feature_column.PickcatsWithHashBucket( key = 'u_wu261079',
                                                          inter_delimiter='|',
                                                          hash_bucket_size=MAX_HASH_SIZE)
    columns_.append(u_wu261079)
    column_limit_duration.append(u_wu261079)
    # 离线用户最近关注Top50
    u_wu261080 = xbd.feature_column.PickcatsWithHashBucket( key = 'u_wu261080',
                                                          inter_delimiter='|',
                                                          hash_bucket_size=MAX_HASH_SIZE)
    columns_.append(u_wu261080)
    column_limit_duration.append(u_wu261080)


    #  ============================= USER FEATURE END =============================

    #  ============================= ITEM FEATURE START 20=============================

    # ost101 超话帖子数
    r_ost101 = xbd.feature_column.BucketizedColumn(key='r_ost101', boundaries=[0.0, 100, 500, 1000, 2500, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000 ])
    columns_.append(r_ost101)
    # ost102 超话粉丝数
    r_ost102 = xbd.feature_column.BucketizedColumn(key='r_ost102', boundaries=[0.0, 100, 500, 1000, 2000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000 ])
    columns_.append(r_ost102)
    # ost108 超话属性
    r_ost108 = xbd.feature_column.CategoricalColumnWithHashBucket( key = 'r_ost108', hash_bucket_size=10 )
    columns_.append(r_ost108)
    # 超话对应大v的uid
    r_ost116 = xbd.feature_column.CategoricalColumnWithHashBucket(key ='r_ost116', hash_bucket_size=500000)
    columns_.append(r_ost116)
    column_limit_duration.append(r_ost116)
    # ost119 超话日增帖子数
    r_ost119 = xbd.feature_column.BucketizedColumn(key='r_ost119', boundaries=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 101.0, 201.0, 301.0, 401.0, 501.0, 601.0, 701.0, 801.0, 901.0, 1001.0, 2001.0, 3001.0, 4001.0, 5001.0, 6001.0, 7001.0, 8001.0, 9001.0, 10001.0, 20001.0, 30001.0, 40001.0, 50001.0, 60001.0, 70001.0, 80001.0, 90001.0, 100001.0, 200001.0, 300001.0, 400001.0, 500001.0, 600001.0, 700001.0, 800001.0, 900001.0, 1000001.0, 2000001.0, 3000001.0, 4000001.0, 5000000])
    columns_.append(r_ost119)
    # ost121 超话日增粉丝数
    r_ost121 = xbd.feature_column.BucketizedColumn(key='r_ost121', boundaries=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 101.0, 201.0, 301.0, 401.0, 501.0, 601.0, 701.0, 801.0, 901.0, 1001.0, 2001.0, 3001.0, 4001.0, 5001.0, 6001.0, 7001.0, 8001.0, 9001.0, 10001.0, 20001.0, 30001.0, 40001.0, 50001.0, 60001.0, 70001.0, 80001.0, 90001.0, 100001.0, 200001.0, 300001.0, 400001.0, 500001.0, 600001.0, 700001.0, 800001.0, 900001.0, 1000000 ])
    columns_.append(r_ost121)
    # ost123 超话今日明星空降数
    r_ost123 = xbd.feature_column.BucketizedColumn(key='r_ost123', boundaries=[1.0, 3, 5, 8, 10, 15, 20, 30, 50, 70, 100, 130, 160, 200 ])
    columns_.append(r_ost123)
    # ost124 超话最近3天明星空降数
    r_ost124 = xbd.feature_column.BucketizedColumn(key='r_ost124', boundaries=[0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 24, 39, 50, 100, 150, 200])
    columns_.append(r_ost124)
    # ost125 超话最近7天明星空降数
    r_ost125 = xbd.feature_column.BucketizedColumn(key='r_ost125', boundaries=[0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300 ])
    columns_.append(r_ost125)
    # ost126 超话阅读人数
    r_ost126 = xbd.feature_column.BucketizedColumn(key='r_ost126', boundaries=[0.0, 10, 50, 100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 75000, 100000, 500000, 1000000, 5000000, 10000000, 20000000, 30000000, 40000000, 50000000, 60000000, 70000000, 80000000, 90000000, 100000000, 500000000, 1000000000 ])
    columns_.append(r_ost126)
    # ost127 超话被提及数
    r_ost127 = xbd.feature_column.BucketizedColumn(key='r_ost127', boundaries=[0.0, 1, 2, 3, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 3000000, 5000000, 7000000, 9000000, 11000000, 13000000, 15000000, 17000000, 20000000, 50000000, 100000000])
    columns_.append(r_ost127)
    # ost128 超话被提及人数
    r_ost128 = xbd.feature_column.BucketizedColumn(key='r_ost128', boundaries=[0.0, 1, 2, 3, 5, 10, 50, 100, 500, 1000, 2500, 5000, 7500, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 2500000, 5000000, 7500000, 10000000, 50000000])
    columns_.append(r_ost128)
    # 超话L1类别
    r_ost129 = xbd.feature_column.PickcatsWithHashBucket( key = 'r_ost129',
                                                           inter_delimiter='|',
                                                           intra_delimiter='@',
                                                           hash_bucket_size=1000)
    columns_.append(r_ost129)
    # 超话L2类别
    r_ost130 = xbd.feature_column.PickcatsWithHashBucket( key = 'r_ost130',
                                                           inter_delimiter='|',
                                                           intra_delimiter='@',
                                                           hash_bucket_size=10000)
    columns_.append(r_ost130)
    # ost131 超话每小时活跃数
    r_ost131 = xbd.feature_column.BucketizedColumn(key='r_ost131', boundaries=[-1.0, 0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100, 150, 200, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 5000, 10000 ])
    columns_.append(r_ost131)
    # zst2000 item所有召回来源+score
    r_zst2000 = xbd.feature_column.PickcatsWithHashBucket( key = 'r_zst2000',
                                                           inter_delimiter='|',
                                                           intra_delimiter='@',
                                                           hash_bucket_size=10000)
    columns_.append(r_zst2000)
    # zst3000 item所有召回来源+召回位置
    r_zst3000 = xbd.feature_column.PickcatsWithHashBucket( key = 'r_zst3000',
                                                           inter_delimiter='|',
                                                           intra_delimiter='@',
                                                           hash_bucket_size=10000)
    columns_.append(r_zst3000)
    # 超话的人均刷新(7天)
    r_wst620006 = xbd.feature_column.BucketizedColumn(key='r_wst620006', boundaries=[1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 100, 500, 1000 ])
    columns_.append(r_wst620006)
    # 超话的人均发博(7天)
    r_wst620007 = xbd.feature_column.BucketizedColumn(key='r_wst620007', boundaries=[0.0, 1.5, 2.0, 2.15, 2.3, 2.5, 2.65, 2.75, 2.9, 3.3, 3.5, 4.0, 4.5, 5.0, 6.5, 8.0, 9.0, 10, 50, 100, 500 ])
    columns_.append(r_wst620007)
    # 超话的次留率(7天)
    r_wst620008 = xbd.feature_column.BucketizedColumn(key='r_wst620008', boundaries=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.72, 0.75, 0.77, 0.8, 0.82, 0.85, 0.87, 0.9, 1.0])
    columns_.append(r_wst620008)

    column_no_limit_duration = [i for i in columns_ if i not in column_limit_duration]
    print("originalFeatureLen: {0} , limitDurationFeatureLen: {1}, noLimitDurationFeatureLen: {2}".format(len(columns_), len(column_limit_duration), len(column_no_limit_duration)))
    
    no_limit_duration_feature = xbd.feature_column.GroupColumn(key='group_no_limit', columns=column_no_limit_duration, sparse=True, use_conf=False)
    limit_duration_feature = xbd.feature_column.GroupColumn(key='group_limit', columns=column_limit_duration, sparse=True, use_conf=False)

    # 36 , [6个序列特征,30]
    return columns_, [no_limit_duration_feature, limit_duration_feature]

def get_feature_conf():
    columns = [
                "u_wu211","u_wu212", "u_wu215", "u_wu217", "u_ou101", "u_ou102", "u_wu2117", "u_wu2043", "u_wu2123", "u_wu21136", 
                "u_wu21135", 
                "u_so1102",
                "u_ff1001", 
                "u_so1101", "u_wu261079", "u_wu261080",
                "r_ost101", "r_ost102", "r_ost116", "r_ost119", "r_ost121", "r_ost129", "r_ost130", "r_zst2000", "r_zst3000", "r_ost108", "r_ost123",
                "r_ost124", "r_ost125", "r_ost126", "r_ost127", "r_ost128", "r_ost131", "r_wst620006", "r_wst620007", "r_wst620008"
            ]
    default_values = [['0'] for _ in columns]
    return columns, default_values
