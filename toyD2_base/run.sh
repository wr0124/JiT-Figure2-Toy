rm -r *.png  *.gif
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 python3 -B train_base.py
