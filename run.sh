python train.py --config_file configs/default.yaml --experiment_name baseline --model.type ResNet_baseline --logging.save_dir logs/methodcomp --_gpu_assigned
python train.py --config_file configs/default.yaml --experiment_name baseline_group --model.type ResNet_baseline_group --logging.save_dir logs/methodcomp --_gpu_assigned
python train.py --config_file configs/default.yaml --experiment_name single --model.type ResNet_single --logging.save_dir logs/methodcomp --_gpu_assigned
python train.py --config_file configs/default.yaml --experiment_name multi1 --model.type ResNet_multi1 --logging.save_dir logs/methodcomp --_gpu_assigned
python train.py --config_file configs/default.yaml --experiment_name multi1group --model.type ResNet_multi1group --logging.save_dir logs/methodcomp --_gpu_assigned
python train.py --config_file configs/default.yaml --experiment_name multi1nogroup --model.type ResNet_multi1nogroup --logging.save_dir logs/methodcomp --_gpu_assigned
python train.py --config_file configs/default.yaml --experiment_name multi2 --model.type ResNet_multi2 --logging.save_dir logs/methodcomp --_gpu_assigned
python train.py --config_file configs/default.yaml --experiment_name multi3 --model.type ResNet_multi3 --logging.save_dir logs/methodcomp --_gpu_assigned
