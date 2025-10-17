### Generates structures using EDM conditioned on specific values of a property

vals=(
    52.6
    56.1
    59.6
    63.1
    66.6
    70.1
    73.6
    77.1
    80.6
    84.1
    87.6
    91.1
    94.6
    98.1
    101.6
)

log_file="log.log"

for i in "${vals[@]}"; do
    echo $i >> $log_file
    echo "" >> $log_file
    # generate + save structures with edm
    CUDA_VISIBLE_DEVICES=0 python eval_conditional_qm9.py \
        --generators_path outputs/exp_35_conditional_nf192_9l_alpha \
        --classifiers_path qm9/property_prediction/outputs/exp_1_alpha/ \
        --property alpha --iterations 50 --context $i --batch_size 40 \
        --task edm >> $log_file 2>&1
    
    echo "" >> $log_file
done
