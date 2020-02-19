
python ./exp/test/eval_show_cihp2pascal.py \
 --batch 1 --gpus 1 --classes 7 \
 --loadmodel './data/pretrained_model/cihp2pascal.pth' \
 --input_path ~/local/datasets/ClothCapture/weipeng_studio_40/openpose_images/ \
 --output_path ~/local/datasets/ClothCapture/weipeng_studio_40/GraphonomyPart/ \
 --txt_file ~/local/datasets/ClothCapture/weipeng_studio_40/list.txt
