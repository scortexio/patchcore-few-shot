# VisA dataset

# + Apply the same number of augmentations
# + Vary for different combinations of augmentations
# for different k-shot

export CUDA_VISIBLE_DEVICES=1

# anti aliased wide resnet
python scripts/few_shot_training_iterate_augmentations.py --backbone antialiased_wide_resnet50_2  --dataset visa --augment True --number-transforms 8 --k-shots 1 5 10 --batch-size 4 2 1

# python scripts/few_shot_training_iterate_augmentations.py --backbone antialiased_wide_resnet50_2  --dataset visa --augment True --number-transforms 6 --k-shots 1 5 10 --batch-size 4 2 1
# python scripts/few_shot_training_iterate_augmentations.py --backbone antialiased_wide_resnet50_2  --dataset visa --augment True --number-transforms 4 --k-shots 1 5 10 --batch-size 4 2 1
