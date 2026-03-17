#!/bin/bash
# CheXpert + Swin Base — Random Init (1 trial)
# Results go to: Ark_Plus/Finetuning/Outputs/Classification/CheXpert/
# Models go to:  Ark_Plus/Finetuning/Models/Classification/CheXpert/

module load mamba/latest
module load cuda-13.0.1-gcc-13.2.0

# Install all required Python packages
pip install numpy tqdm scikit-image scikit-learn SimpleITK scipy pydicom \
  yacs einops opencv-python timm==0.5.4 transformers>=4.37.0 \
  albumentations imgaug Pillow pretrainedmodels pyyaml

RUN_ID=$RANDOM

cd Ark_Plus/Finetuning/

echo "=========================================="
echo "Swin Base — Random Init"
echo "=========================================="

python main_classification.py \
  --data_set CheXpert \
  --data_dir /scratch/ayerrams/CheXpert-v1.0/chexpertchestxrays-u20210408 \
  --train_list ../dataset/CheXpert/CheXpert_train_official.csv \
  --val_list ../dataset/CheXpert/CheXpert_test_official.csv \
  --test_list ../dataset/CheXpert/CheXpert_test_official.csv \
  --num_class 14 \
  --model swin_base \
  --init random \
  --lr 0.02 --opt sgd --epochs 200 --warmup-epochs 0 --batch_size 64 \
  --img_size 256 --input_size 224 \
  --trial 1 \
  --test_every_epoch \
  --exp_name "_random_${RUN_ID}"

echo "Done! Check results in Outputs/Classification/CheXpert/"