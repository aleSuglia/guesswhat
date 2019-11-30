#!/bin/bash

python src/extract_dialogue_features.py \
      -features /scratch/ale_models/guesswhat_devries/dialogue_states/sl \
      -data_dir data/original/ \
      -exp_dir /scratch/ale_models/guesswhat_devries/out/qgen/ \
      -exp_identifier 867d59b933a89f4525b189da9d67f17b \
      -img_dir /scratch/ale_models/guesswhat_devries/ft_vgg_img \
      -config config/qgen/config.json \
      -dict_file data/original/dict.json \
      -no_thread 2

python src/extract_dialogue_features.py \
      -features /scratch/ale_models/guesswhat_devries/dialogue_states/rl \
      -data_dir data/original/ \
      -config config/qgen/config.json \
      -img_dir /scratch/ale_models/guesswhat_devries/ft_vgg_img \
      -exp_dir /scratch/ale_models/guesswhat_devries/out/loop/ \
      -dict_file data/original/dict.json \
      -exp_identifier e3a6b91687fa1589696f5969b1433a47 \
      --rl_model \
      -no_thread 2
