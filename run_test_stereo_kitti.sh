DATASET_PATH=~/dataset/kitti_raw/
TEST_WORK_DIR=./work_dir/stereo_unsupft_kitti

python main_stereo.py --mode test --dataset=kitti --data_path $DATASET_PATH --test_list ./list/eigen_test_list.txt --batch_size=1 --height=384 --width=1280 --work_dir $TEST_WORK_DIR --load_latest
python scripts/evaluate_kitti.py --split eigen --predicted_disp_path $TEST_WORK_DIR/disparities.npy --gt_path $DATASET_PATH --garg_crop
