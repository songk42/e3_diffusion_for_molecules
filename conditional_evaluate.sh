### Evaluates pre-generated structures using a classifier

lst=(
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_00-55-48/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_00-59-14/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_01-02-25/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_01-06-02/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_01-09-59/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_01-14-04/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_01-17-57/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_01-21-36/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_01-25-45/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_01-29-52/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_01-33-21/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_01-36-54/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_01-40-52/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_01-44-42/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_01-48-36/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_01-52-49/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_01-57-22/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_02-00-47/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_02-04-44/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_02-08-02/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_02-11-35/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_02-15-53/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_02-20-04/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_02-24-21/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_02-28-52/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_02-32-57/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_02-36-29/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_02-40-00/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_02-43-46/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_02-47-18/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_02-51-28/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_02-56-09/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_03-00-10/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_03-04-20/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_03-08-51/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_03-12-15/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_03-16-22/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_03-19-59/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_03-23-39/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_03-27-49/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_03-32-33/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_03-37-06/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_03-41-51/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_03-46-14/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_03-50-13/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_03-54-23/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_03-57-45/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_04-01-12/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_04-05-20/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_04-09-39/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_04-14-16/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_04-18-33/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_04-23-10/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_04-27-02/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_04-31-31/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_04-35-16/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_04-39-53/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_04-43-46/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_04-48-53/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_04-53-24/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_04-58-19/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_05-02-51/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_05-06-49/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_05-11-21/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_05-14-55/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_05-19-17/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_05-24-27/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_05-30-35/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_05-35-21/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_05-39-55/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_05-43-54/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_05-49-08/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_05-54-30/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_05-58-12/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_06-06-00/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_06-11-34/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_06-17-21/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_06-21-58/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_06-27-17/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_06-31-50/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_06-37-20/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_06-44-46/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_06-48-55/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_06-54-19/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_07-00-15/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_07-05-49/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_07-11-45/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_07-16-36/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_07-22-47/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_07-28-21/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_07-35-36/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_07-39-50/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_07-46-43/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_07-54-18/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_08-02-03/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_08-07-22/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_08-15-21/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_08-23-16/samples
/potato/songk/symphony-torch-data/outputs/sample/dev/runs/2026-04-28_08-29-41/samples
)

log_file="eval_multi_relenergy.log"

# evaluate all pre-generated structures with one Python process so the
# classifier and generator are loaded only once.
CUDA_VISIBLE_DEVICES=5 python eval_conditional_qm9.py \
    --generators_path outputs/exp_cond_relenergy \
    --classifiers_path qm9/property_prediction/outputs/exp_class_relenergy/ \
    --property relative_atomic_energy --iterations 20 --batch_size 100 --task xyz \
    --cond_keys gap relative_atomic_energy \
    --xyz_dir "${lst[@]}" >> $log_file 2>&1
