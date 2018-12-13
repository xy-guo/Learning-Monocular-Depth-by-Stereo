mkdir work_dir
DATASET_PATH=~/dataset/kitti_raw/
WORK_DIR=./work_dir/mono_unsupft_kitti/
STEREO_CKPT=./work_dir/stereo_unsupft_kitti/checkpoint_000009.ckpt

python main_distill_mono.py --mode=train --dataset=kitti --data_path $DATASET_PATH --train_list ./list/eigen_train_list.txt --val_list ./list/eigen_test_list.txt --batch_size 4 --height 256 --width 512 --num_epochs 50 --learning_rate 0.0001 --work_dir $WORK_DIR --stereo_ckpt $STEREO_CKPT
