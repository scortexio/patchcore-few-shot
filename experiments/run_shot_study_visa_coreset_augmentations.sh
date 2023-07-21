# Visa dataset

export CUDA_VISIBLE_DEVICES=1
python scripts/few_shot_training.py --backbone antialiased_wide_resnet50_2  --dataset visa --augment True --number-transforms 16 --k-shots 1 5 10 --batch-size 4 4 1 --coreset-ratio 25088
python scripts/few_shot_training.py --backbone antialiased_wide_resnet50_2  --dataset visa --augment True --number-transforms 16 --k-shots 1 5 10 --batch-size 4 4 1 --coreset-ratio 12544
python scripts/few_shot_training.py --backbone antialiased_wide_resnet50_2  --dataset visa --augment True --number-transforms 16 --k-shots 1 5 10 --batch-size 4 4 1 --coreset-ratio 3136

# python scripts/few_shot_training.py --backbone antialiased_wide_resnet50_2  --dataset visa --augment True --number-transforms 8 --k-shots 1 5 10 --batch-size 4 4 1 --coreset-ratio 25088
# python scripts/few_shot_training.py --backbone antialiased_wide_resnet50_2  --dataset visa --augment True --number-transforms 8 --k-shots 1 5 10 --batch-size 4 4 1 --coreset-ratio 12544
# python scripts/few_shot_training.py --backbone antialiased_wide_resnet50_2  --dataset visa --augment True --number-transforms 8 --k-shots 1 5 10 --batch-size 4 4 1 --coreset-ratio 3136

# python scripts/few_shot_training.py --backbone antialiased_wide_resnet50_2  --dataset visa --augment True --number-transforms 16 --k-shots 1 5 10 --batch-size 4 4 1 --coreset-ratio 47040
# python scripts/few_shot_training.py --backbone antialiased_wide_resnet50_2  --dataset visa --augment True --number-transforms 16 --k-shots 1 5 10 --batch-size 4 4 1 --coreset-ratio 31360
# python scripts/few_shot_training.py --backbone antialiased_wide_resnet50_2  --dataset visa --augment True --number-transforms 16 --k-shots 1 5 10 --batch-size 4 4 1 --coreset-ratio 15680
