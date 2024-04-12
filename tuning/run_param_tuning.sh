n_trials=100
timeout=60
metric="f1+chi-square"

python ./cls_tuning.py -study_name "MLP" -net_type "MLP" -cls_type "MLP" -n_trials $n_trials -timeout $timeout -conf_tuning ./tuning/tune_cls.json -metric $metric

for net_type in "GCN" "GAT" "GATv2"
do

    for n_edges in 0 422 1000 2500 5000
    do
        python ./cls_tuning.py -study_name "${net_type}_${n_edges}_GNN" -net_type $net_type -cls_type $net_type -n_trials $n_trials -timeout $timeout -conf_tuning ./tuning/tune_cls.json -n_edges $n_edges -metric $metric
    done
done

for net_type in "GCN" "GAT" "GATv2"  
do

    for n_edges in 0 422 1000 2500 5000
    do
        python ./cls_tuning.py -study_name "${net_type}_${n_edges}_MLP" -net_type $net_type -cls_type "MLP" -n_trials $n_trials -timeout $timeout -conf_tuning ./tuning/tune_cls.json -n_edges $n_edges -metric $metric
    done
done