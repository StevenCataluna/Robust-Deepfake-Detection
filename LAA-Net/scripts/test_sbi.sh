#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python scripts/test.py --cfg configs/efn4_fpn_sbi_adv.yaml \
                                    -i Test_images/Test_df.png
