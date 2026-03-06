#!/bin/bash
# CheXpert + Swin Base — 4 runs alternating: random, inet, random, inet
# Results go to: Ark_Plus/Finetuning/Outputs/Classification/CheXpert/
# Models go to:  Ark_Plus/Finetuning/Models/Classification/CheXpert/

module load mamba/latest
module load cuda-13.0.1-gcc-13.2.0

# Install all required Python packages
pip install numpy tqdm scikit-image scikit-learn SimpleITK scipy pydicom \
  yacs einops opencv-python timm==0.5.4 transformers>=4.37.0 \
  albumentations imgaug Pillow pretrainedmodels pyyaml

cd Ark_Plus/Finetuning/

echo "=========================================="
echo "Run 1: Swin Base — Random Init (trial 1)"
echo "=========================================="

python main_classification.py \
  --data_set CheXpert \
  --data_dir /scratch/ayerrams/CheXpert-v1.0/chexpertchestxrays-u20210408 \
  --train_list ../dataset/CheXpert/CheXpert_train_official.csv \
  --val_list ../dataset/CheXpert/CheXpert_valid_official.csv \
  --test_list ../dataset/CheXpert/CheXpert_test_official.csv \
  --num_class 14 \
  --model swin_base \
  --init random \
  --lr 0.01 --opt sgd --epochs 200 --warmup-epochs 0 --batch_size 64 \
  --img_size 256 --input_size 224 \
  --trial 1 \
  --exp_name _randomOne

echo "================================================"
echo "Run 2: Swin Base — ImageNet-21K (trial 1)"
echo "================================================"

python main_classification.py \
  --data_set CheXpert \
  --data_dir /scratch/ayerrams/CheXpert-v1.0/chexpertchestxrays-u20210408 \
  --train_list ../dataset/CheXpert/CheXpert_train_official.csv \
  --val_list ../dataset/CheXpert/CheXpert_valid_official.csv \
  --test_list ../dataset/CheXpert/CheXpert_test_official.csv \
  --num_class 14 \
  --model swin_base \
  --init imagenet_21k \
  --lr 0.01 --opt sgd --epochs 200 --warmup-epochs 0 --batch_size 64 \
  --img_size 256 --input_size 224 \
  --trial 1 \
  --exp_name _inetOne

echo "=========================================="
echo "Run 3: Swin Base — Random Init (trial 2)"
echo "=========================================="

python main_classification.py \
  --data_set CheXpert \
  --data_dir /scratch/ayerrams/CheXpert-v1.0/chexpertchestxrays-u20210408 \
  --train_list ../dataset/CheXpert/CheXpert_train_official.csv \
  --val_list ../dataset/CheXpert/CheXpert_valid_official.csv \
  --test_list ../dataset/CheXpert/CheXpert_test_official.csv \
  --num_class 14 \
  --model swin_base \
  --init random \
  --lr 0.01 --opt sgd --epochs 200 --warmup-epochs 0 --batch_size 64 \
  --img_size 256 --input_size 224 \
  --trial 1 \
  --exp_name _randomTwo

echo "================================================"
echo "Run 4: Swin Base — ImageNet-21K (trial 2)"
echo "================================================"

python main_classification.py \
  --data_set CheXpert \
  --data_dir /scratch/ayerrams/CheXpert-v1.0/chexpertchestxrays-u20210408 \
  --train_list ../dataset/CheXpert/CheXpert_train_official.csv \
  --val_list ../dataset/CheXpert/CheXpert_valid_official.csv \
  --test_list ../dataset/CheXpert/CheXpert_test_official.csv \
  --num_class 14 \
  --model swin_base \
  --init imagenet_21k \
  --lr 0.01 --opt sgd --epochs 200 --warmup-epochs 0 --batch_size 64 \
  --img_size 256 --input_size 224 \
  --trial 1 \
  --exp_name _inetTwo

echo "Done! Check results in Outputs/Classification/CheXpert/"
