## mvtec experiments with many shot with different backbone & image size

export CUDA_VISIBLE_DEVICES=0
# efficientnet b4
python scripts/few_shot_training.py --backbone efficientnet_b4 --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 190
python scripts/few_shot_training.py --backbone efficientnet_b4 --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 380
python scripts/few_shot_training.py --backbone efficientnet_b4 --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 570
python scripts/few_shot_training.py --backbone efficientnet_b4 --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 760

# convnext small
python scripts/few_shot_training.py --backbone convnext_small_384_in22ft1k  --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 192
python scripts/few_shot_training.py --backbone convnext_small_384_in22ft1k  --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 384
python scripts/few_shot_training.py --backbone convnext_small_384_in22ft1k  --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 576
python scripts/few_shot_training.py --backbone convnext_small_384_in22ft1k  --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 768

# convnext base
python scripts/few_shot_training.py --backbone convnext_base_384_in22ft1k  --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 192
python scripts/few_shot_training.py --backbone convnext_base_384_in22ft1k  --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 384
python scripts/few_shot_training.py --backbone convnext_base_384_in22ft1k  --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 576
python scripts/few_shot_training.py --backbone convnext_base_384_in22ft1k  --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 768

# wide resnet
python scripts/few_shot_training.py --backbone wide_resnet50_2  --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 112
python scripts/few_shot_training.py --backbone wide_resnet50_2  --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 224
python scripts/few_shot_training.py --backbone wide_resnet50_2  --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 336
python scripts/few_shot_training.py --backbone wide_resnet50_2  --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 448

# anti aliased wide resnet
python scripts/few_shot_training.py --backbone antialiased_wide_resnet50_2  --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 112
python scripts/few_shot_training.py --backbone antialiased_wide_resnet50_2  --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 224
python scripts/few_shot_training.py --backbone antialiased_wide_resnet50_2  --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 336
python scripts/few_shot_training.py --backbone antialiased_wide_resnet50_2  --dataset mvtec --k-shots 1 5 10 25 50 --batch-size 16 4 4 1 1 --image-size 448
