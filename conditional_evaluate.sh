### Evaluates pre-generated structures using a classifier

lst=(
/radish/songk/cG-SchNet/models/cgschnet-polarizability/generated/xyzs_0
)

log_file="log.log"

for i in "${lst[@]}"; do
    echo $i >> $log_file
    echo "" >> $log_file

    # evaluate pre-generated structures using classifier
    CUDA_VISIBLE_DEVICES=0 python eval_conditional_qm9.py \
        --generators_path outputs/exp_35_conditional_nf192_9l_alpha \
        --classifiers_path qm9/property_prediction/outputs/exp_1_alpha/ \
        --property alpha --iterations 100 --batch_size 100 --task xyz \
        --xyz_dir "$i" >> $log_file 2>&1
    
    echo "" >> $log_file
done
