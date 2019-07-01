NAME='8'
python exp/inference/inference.py \
    --loadmodel ./data/pretrained_model/universal_trained.pth \
    --img_path ./img/${NAME}.png \
    --output_path ./output/ \
    --output_name /${NAME}
