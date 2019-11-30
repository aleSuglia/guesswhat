import argparse
import logging
import os
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from generic.data_provider.image_loader import get_img_builder
from generic.data_provider.iterator import Iterator
from generic.tf_utils.ckpt_loader import create_resnet_saver
from generic.tf_utils.evaluator import Evaluator
from generic.utils.config import load_config
from guesswhat.data_provider.guesswhat_dataset import Dataset
from guesswhat.data_provider.guesswhat_tokenizer import GWTokenizer
from guesswhat.data_provider.questioner_batchifier import QuestionerBatchifier
from guesswhat.models.guesser.guesser_network import GuesserNetwork
from guesswhat.models.qgen.qgen_lstm_network import QGenNetworkLSTM


def save_dialogue_states(output_path, split_name, dialogue_state_features, dialogue_state_ids):
    split_output_path = os.path.join(output_path, split_name)
    if not os.path.exists(split_output_path):
        os.makedirs(split_output_path)

    print("Saving dialogue state features for split {}".format(split_name))
    for i, dialogue_id in enumerate(tqdm(dialogue_state_ids)):
        output_file = os.path.join(split_output_path, "{}.npy".format(dialogue_id))

        np.save(output_file, dialogue_state_features[i])


if __name__ == '__main__':

    ###############################
    #  LOAD CONFIG
    #############################

    parser = argparse.ArgumentParser('Guesser network baseline!')
    parser.add_argument("-features", type=str, help="Output directory that will contain the output features")
    parser.add_argument("-data_dir", type=str, help="Directory with data")
    parser.add_argument("-exp_dir", type=str, help="Directory in which experiments are stored")
    parser.add_argument("-img_dir", type=str, help='Directory with images')
    parser.add_argument("-config", type=str, help="Configuration file")
    parser.add_argument("-exp_identifier", type=str, help="Experiment identifier")
    parser.add_argument("--rl_model", action="store_true")
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Dictionary file name")
    parser.add_argument("-gpu_ratio", type=float, default=1., help="How many GPU ram is required? (ratio)")
    parser.add_argument("-no_thread", type=int, default=1, help="No thread to load batch")

    args = parser.parse_args()
    config, exp_identifier, save_path = load_config(args.config, args.exp_dir, args.exp_identifier)
    logger = logging.getLogger()

    ###############################
    #  LOAD DATA
    #############################

    image_builder, crop_builder = None, None

    # Load image
    logger.info('Loading images..')
    use_resnet = False
    if 'image' in config['model']:
        logger.info('Loading images..')
        image_builder = get_img_builder(config['model']['image'], args.img_dir)

    # Load data
    logger.info('Loading data..')
    trainset = Dataset(args.data_dir, "train", "guesswhat", image_builder, crop_builder)
    validset = Dataset(args.data_dir, "valid", "guesswhat", image_builder, crop_builder)
    testset = Dataset(args.data_dir, "test", "guesswhat", image_builder, crop_builder)

    # Load dictionary
    logger.info('Loading dictionary..')
    tokenizer = GWTokenizer(args.dict_file)

    # Build Network
    logger.info('Building network..')
    if args.rl_model:
        network = QGenNetworkLSTM(config["model"], num_words=tokenizer.no_words, policy_gradient=True)
        qgen_var = [v for v in tf.global_variables() if "qgen" in v.name]  # and 'rl_baseline' not in v.name
    else:
        network = QGenNetworkLSTM(config["model"], num_words=tokenizer.no_words, policy_gradient=False)
        qgen_var = [v for v in tf.global_variables() if "qgen" in v.name and 'rl_baseline' not in v.name]
    saver = tf.train.Saver(var_list=qgen_var)

    # Build Optimizer
    logger.info('Building optimizer..')
    # optimizer, outputs = create_optimizer(network, config)
    outputs = [network.decoder_state]

    ###############################
    #  START  TRAINING
    #############################

    # Load config
    batch_size = config['optimizer']['batch_size']
    no_epoch = config["optimizer"]["no_epoch"]

    # CPU/GPU option
    cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

        sources = network.get_sources(sess)
        logger.info("Sources: " + ', '.join(sources))
        saver.restore(sess, save_path.format('params.ckpt'))

        if not os.path.exists(args.features):
            os.makedirs(args.features)
        # create training tools
        evaluator = Evaluator(sources, network.scope_name, network=network, tokenizer=tokenizer)
        batchifier = QuestionerBatchifier(tokenizer, sources, status=('success',))

        train_iterator = Iterator(trainset,
                                  batch_size=batch_size * 2, pool=cpu_pool,
                                  batchifier=batchifier,
                                  shuffle=False)
        _, train_states = evaluator.process(sess, train_iterator, outputs=outputs, output_dialogue_states=True)

        save_dialogue_states(args.features, "train", *train_states)

        valid_iterator = Iterator(validset, pool=cpu_pool,
                                  batch_size=batch_size * 2,
                                  batchifier=batchifier,
                                  shuffle=False)
        _, valid_states = evaluator.process(sess, valid_iterator, outputs=outputs, output_dialogue_states=True)

        save_dialogue_states(args.features, "valid", *valid_states)

        test_iterator = Iterator(testset, pool=cpu_pool,
                                 batch_size=batch_size * 2,
                                 batchifier=batchifier,
                                 shuffle=False)
        _, test_states = evaluator.process(sess, test_iterator, outputs, output_dialogue_states=True)

        save_dialogue_states(args.features, "test", *test_states)
