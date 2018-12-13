mkdir work_dir
DATASET_PATH=~/dataset/scene_flow/RGB_finalpass/
ROUND1_WORK_DIR=./work_dir/stereo_pretrain_sceneflow_round1
ROUND2_WORK_DIR=./work_dir/stereo_pretrain_sceneflow

python main_stereo.py --mode train --dataset sceneflow --data_path $DATASET_PATH --train_list ./list/sceneflow_train_list.txt --val_list ./list/sceneflow_test_list.txt --batch_size 4 --height 384 --width 768 --num_epochs 50 --learning_rate 0.0001 --lrepochs "25,35,45" --work_dir $ROUND1_WORK_DIR
python main_stereo.py --mode train --dataset sceneflow --data_path $DATASET_PATH --train_list ./list/sceneflow_train_list.txt --val_list ./list/sceneflow_test_list.txt --batch_size 4 --height 384 --width 768 --num_epochs 50 --learning_rate 0.0001 --lrepochs "25,35,45" --work_dir $ROUND2_WORK_DIR --load_ckpt $ROUND1_WORK_DIR/checkpoint_000049.ckpt

