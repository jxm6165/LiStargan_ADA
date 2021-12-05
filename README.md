# LiStargan_ADA

## 1. Install the dependencies
```bash
pip install munch kornia ffmpeg-python opencv-python ninja torchprofile
```

## 2. Download Pretrained StarGAN v2 models and Datasets

<b>CelebA.</b> To download the [CelebA](https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs) dataset and the pre-trained network, run the following commands:
```bash
bash download.sh celeba-hq-dataset
bash download.sh pretrained-network-celeba-hq
bash download.sh wing
```

<b>AFHQ.</b> To download the [AFHQ](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) dataset and the pre-trained network, run the following commands:
```bash
bash download.sh afhq-dataset
bash download.sh pretrained-network-afhq
```
## 3. Train Lightweight StarGAN with ADA

<b>CelebA.</b> Two Domain Image-to-Image Translation.
```bash
python main.py --mode train --num_domains 2 --w_hpf 1 \
               --alpha 128 --efficient 1 --f_lr 1e-4 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 \
               --batch_size 8 --val_batch_size 8 \
               --teacher_checkpoint_dir expr/checkpoints/celeba_hq --teacher_resume_iter 100000 \
               --train_img_dir data/celeba_hq/train \
               --val_img_dir data/celeba_hq/val \
               --sample_every 500 --save_every 500 --print_every 50 --eval_every 10000 \
               --total_iter 20000
```

<b>AFHQ.</b> Multi-Domain Image-to-Image Translation.
```bash
python main.py --mode train --num_domains 3 --w_hpf 0 \
               --alpha 128 --efficient 1 --f_lr 1e-4 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1 \
               --batch_size 8 --val_batch_size 8 \
               --teacher_checkpoint_dir expr/checkpoints/afhq --teacher_resume_iter 100000 \
               --train_img_dir data/afhq/train \
               --val_img_dir data/afhq/val \
               --sample_every 500 --save_every 500 --print_every 50 --eval_every 10000 \
               --total_iter 30000
```

To resume from previously saved checkpoint with iteration number, e.g., 10000, set `--resume_iter 10000`.



## 4. Repository Structure

<pre>
LiStargan_ADA/
|-- assets
|   `-- paper
|       |-- afhq
|       |   |-- ref
|       |   |   |-- cat
|       |   |   |   |-- flickr_cat_000557.jpg
|       |   |   |   |-- pixabay_cat_000730.jpg
|       |   |   |   `-- pixabay_cat_001699.jpg
|       |   |   |-- dog
|       |   |   |   |-- pixabay_dog_000322.jpg
|       |   |   |   `-- pixabay_dog_000409.jpg
|       |   |   `-- wild
|       |   |       |-- flickr_wild_002020.jpg
|       |   |       |-- flickr_wild_002092.jpg
|       |   |       `-- flickr_wild_003355.jpg
|       |   `-- src
|       |       |-- cat
|       |       |   |-- flickr_cat_000253.jpg
|       |       |   `-- pixabay_cat_000181.jpg
|       |       |-- dog
|       |       |   |-- flickr_dog_000094.jpg
|       |       |   `-- pixabay_dog_001082.jpg
|       |       `-- wild
|       |           |-- flickr_wild_002036.jpg
|       |           |-- flickr_wild_002159.jpg
|       |           |-- pixabay_wild_000558.jpg
|       |           `-- pixabay_wild_000637.jpg
|       `-- celeba_hq
|           |-- ref
|           |   |-- female
|           |   |   |-- 064119.jpg
|           |   |   `-- 113393.jpg
|           |   `-- male
|           |       |-- 037023.jpg
|           |       `-- 060259.jpg
|           `-- src
|               |-- female
|               |   |-- 021443.jpg
|               |   `-- 051340.jpg
|               `-- male
|                   |-- 006930.jpg
|                   `-- 016387.jpg
|-- core
|   |-- torch_utils
|   |   |-- ops
|   |   |   |-- bias_act.cpp
|   |   |   |-- bias_act.cu
|   |   |   |-- bias_act.h
|   |   |   |-- bias_act.py
|   |   |   |-- conv2d_gradfix.py
|   |   |   |-- conv2d_resample.py
|   |   |   |-- fma.py
|   |   |   |-- grid_sample_gradfix.py
|   |   |   |-- upfirdn2d.cpp
|   |   |   |-- upfirdn2d.cu
|   |   |   |-- upfirdn2d.h
|   |   |   `-- upfirdn2d.py
|   |   |-- custom_ops.py
|   |   |-- misc.py
|   |   |-- persistence.py
|   |   `-- training_stats.py
|   |-- augment.py
|   |-- checkpoint.py
|   |-- data_loader.py
|   |-- distiller.py
|   |-- model.py
|   |-- utils.py
|   `-- wing.py
|-- dnnlib
|   |-- __pycache__
|   |   |-- __init__.cpython-37.pyc
|   |   `-- util.cpython-37.pyc
|   |-- __init__.py
|   `-- util.py
|-- metrics
|   |-- eval.py
|   |-- fid.py
|   |-- lpips.py
|   `-- lpips_weights.ckpt
|-- download.sh
`-- main.py
</pre>

## References

This project is largely dependent on previous works. Below are citations of important works we refer to.


<a id="1">[1]</a> 
Y. Choi, Y. Uh, J. Yoo, and J.-W. Ha.
Stargan v2: Diverse image synthesis for multiple domains.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2020.

<a id="2">[2]</a> 
P. Kapoor.
StarGAN-v2 compression using knowledge distillation.
PhD thesis, Concordia University, 2021.

<a id="3">[3]</a> 
T. Karras, M. Aittala, J. Hellsten, S. Laine, J. Lehtinen, and T. Aila. 
Training generative adversarial networks with limited data. arXiv preprint arXiv:2006.06676, 2020.

