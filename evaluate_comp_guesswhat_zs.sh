#!/bin/bash

# evaluate the supervised learning baseline
python src/evaluate_gameplay.py
    -data_dir data/comp_guesswhat_zs/ \
    -exp_dir /scratch/ale_models/guesswhat_devries/out/loop/ \
    -config config/looper/config.json \
    -img_dir /scratch/ale_models/nocaps/vggnet \
    -networks_dir /scratch/ale_models/guesswhat_devries/out/ \
    -oracle_identifier 156cb3d352b97ba12ffd6cf547281ae2 \
    -qgen_identifier 867d59b933a89f4525b189da9d67f17b \
    -guesser_identifier e2c11b1757337d7969dc223c334756a9 \
    -evaluate_all true \
    -store_games true \
    -no_thread 2;

# evaluate the RL optimised model
python src/evaluate_gameplay.py
    -data_dir data/comp_guesswhat_zs/ \
    -exp_dir /scratch/ale_models/guesswhat_devries/out/loop/ \
    -config config/looper/config.json \
    -img_dir /scratch/ale_models/nocaps/vggnet \
    -networks_dir /scratch/ale_models/guesswhat_devries/out/ \
    -oracle_identifier 156cb3d352b97ba12ffd6cf547281ae2 \
    -qgen_identifier 867d59b933a89f4525b189da9d67f17b \
    -guesser_identifier e2c11b1757337d7969dc223c334756a9 \
    -evaluate_all true \
    -load_rl true \
    -store_games true \
    -no_thread 2;