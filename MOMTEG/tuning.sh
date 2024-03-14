# python ./mt_tuning.py -study_name "cls_mlp" -net_type "MLP" -n_trials 324 -timeout 120 -config ./tuning/tune_cls_MLP.json -gridsearch True
# python ./mt_tuning.py -study_name "cls_gcn" -net_type "GCN" -n_trials 324 -timeout 120 -config ./tuning/tune_cls_GNN.json -gridsearch True
# python ./mt_tuning.py -study_name "cls_gat" -net_type "GAT" -n_trials 324 -timeout 120 -config ./tuning/tune_cls_GNN.json -gridsearch True

# python ./mt_tuning.py -study_name "surv_mlp" -net_type "MLP" -n_trials 972 -timeout 120 -config ./tuning/tune_surv_MLP.json -gridsearch True
# python ./mt_tuning.py -study_name "surv_gcn" -net_type "GCN" -n_trials 972 -timeout 120 -config ./tuning/tune_surv_GNN.json -gridsearch True
# python ./mt_tuning.py -study_name "surv_gat" -net_type "GAT" -n_trials 972 -timeout 120 -config ./tuning/tune_surv_GNN.json -gridsearch True

# python ./mt_tuning.py -study_name "mt_mlp" -net_type "MLP" -n_trials 1152 -timeout 240 -config ./tuning/tune_mt_MLP.json -gridsearch True
python ./mt_tuning.py -study_name "mt_gcn" -net_type "GCN" -n_trials 1152 -timeout 240 -config ./tuning/tune_mt_GNN.json -gridsearch True
python ./mt_tuning.py -study_name "mt_gat" -net_type "GAT" -n_trials 1152 -timeout 240 -config ./tuning/tune_mt_GNN.json -gridsearch True
