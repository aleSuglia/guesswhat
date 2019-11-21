#!/bin/bash

#python src/extract_dialogue_features.py \
#      -features /scratch/ale_models/guesswhat_devries/dialogue_states/sl \
#      -data_dir data/original/ \
#      -config config/guesser/config.json \
#      -img_dir /scratch/ale_models/guesswhat_devries/ft_vgg_img \
#      -exp_dir /scratch/ale_models/guesswhat_devries/out/guesser/ \
#      -no_thread 2

python src/extract_dialogue_features.py \
      -features /scratch/ale_models/guesswhat_devries/dialogue_states/rl \
      -data_dir data/original/ \
      -config config/guesser/config.json \
      -img_dir /scratch/ale_models/guesswhat_devries/ft_vgg_img \
      -exp_dir /scratch/ale_models/guesswhat_devries/out/loop/ \
      -exp_identifier e3a6b91687fa1589696f5969b1433a47 \
      -no_thread 2
