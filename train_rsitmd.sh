card=$1
port=$2
name=$3
CUDA_VISIBLE_DEVICES=$card torchrun --nproc_per_node 2 --master_port $port -m training.main \
    --train-data './dataset/train/ret2.csv' \
    --images-dir './images' \
    --csv-separator ',' --csv-img-key 'filename' --csv-caption-key 'title' \
    --retrieval-data './dataset/test/rsitmd_test.csv' \
    --retrieval-images-dir './images' \
    --retrieval-csv-separator ',' --retrieval-csv-img-key 'filename' --retrieval-csv-caption-key 'title' \
    --retrieval-frequency 1 \
    --epochs 7 --save-frequency 0 --batch-size 100 --workers 2 \
    --lr 5e-5 --warmup 100 --weight_decay 0.5 --max-grad-norm 50 \
    --image-model 'ViT-B-32' --image-model-builder 'openclip' \
    --text-model 'ViT-B-32' --text-model-builder 'openclip' \
    --pretrained-image-model --pretrained-text-model \
    --loss 'InfoNCE' \
    --report-to tensorboard --name $name \
    --cache-dir './cache/weights' \

CUDA_VISIBLE_DEVICES=$card torchrun --nproc_per_node 2   --master_port $port -m training.main \
    --retrieval-data './dataset/test/rsitmd_test.csv' --retrieval-images-dir './images' \
    --retrieval-csv-separator ',' --retrieval-csv-img-key 'filename' --retrieval-csv-caption-key 'title' \
    --retrieval-frequency 1 \
    --save-frequency 0 --batch-size 128 --workers 2 \
    --image-model 'ViT-B-32' --image-model-builder 'openclip' --text-model 'ViT-B-32' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'InfoNCE' \
    --resume logs/$name/checkpoints/best.pt \
    --cache-dir './cache/weights'