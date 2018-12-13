mkdir work_dir
DATASET_PATH=~/dataset/kitti_raw/
PRETRAINED_WORK_DIR=./work_dir/stereo_pretrain_sceneflow
WORK_DIR=./work_dir/stereo_unsupft_kitti

python main_unsupft_stereo.py --mode train --dataset kitti --data_path $DATASET_PATH --train_list ./list/eigen_train_list.txt --val_list ./list/eigen_val_list.txt --batch_size 4 --height 256 --width 832 --num_epochs 10 --learning_rate 0.00002 --work_dir $WORK_DIR --pretrained_ckpt $PRETRAINED_WORK_DIR/checkpoint_000049.ckpt --load_ckpt $PRETRAINED_WORK_DIR/checkpoint_000049.ckpt $@
