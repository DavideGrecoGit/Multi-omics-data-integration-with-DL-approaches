python ./mt_tuning.py -study_name "cls_mlp" -net_type "MLP" -n_trials 1 -timeout 120 -config_tuning ./tuning/tune_cls.json -gridsearch True
# python ./mt_tuning.py -study_name "cls_gcn" -net_type "GCN" -n_trials 324 -timeout 120 -config_tuning ./tuning/tune_cls.json -gridsearch True
# python ./mt_tuning.py -study_name "cls_gat" -net_type "GAT" -n_trials 324 -timeout 120 -config_tuning ./tuning/tune_cls.json -gridsearch True

# python ./mt_tuning.py -study_name "surv_mlp" -net_type "MLP" -n_trials 972 -timeout 120 -config_tuning ./tuning/tune_surv.json -gridsearch True
# python ./mt_tuning.py -study_name "surv_gcn" -net_type "GCN" -n_trials 972 -timeout 120 -config_tuning ./tuning/tune_surv.json -gridsearch True
# python ./mt_tuning.py -study_name "surv_gat" -net_type "GAT" -n_trials 972 -timeout 120 -config_tuning ./tuning/tune_surv.json -gridsearch True

# python ./mt_tuning.py -study_name "mt_mlp" -net_type "MLP" -n_trials 1000 -timeout 120 -config_tuning ./tuning/tune_mt.json
# python ./mt_tuning.py -study_name "mt_gcn" -net_type "GCN" -n_trials 1000 -timeout 120 -config_tuning ./tuning/tune_mt.json
# python ./mt_tuning.py -study_name "mt_gat" -net_type "GAT" -n_trials 1000 -timeout 120 -config_tuning ./tuning/tune_mt.json
