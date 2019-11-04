import argparse
import logging
import os
from distutils.util import strtobool
from multiprocessing import Pool

import tensorflow as tf

from generic.data_provider.image_loader import get_img_builder
from generic.tf_utils.evaluator import Evaluator
from generic.utils.config import load_config, get_config_from_xp
from guesswhat.data_provider.guesswhat_dataset import Dataset
from guesswhat.data_provider.guesswhat_tokenizer import GWTokenizer
from guesswhat.data_provider.looper_batchifier import LooperBatchifier
from guesswhat.models.guesser.guesser_network import GuesserNetwork
from guesswhat.models.guesser.guesser_wrapper import GuesserWrapper
from guesswhat.models.looper.basic_looper import BasicLooper
from guesswhat.models.oracle.oracle_network import OracleNetwork
from guesswhat.models.oracle.oracle_wrapper import OracleWrapper
from guesswhat.models.qgen.qgen_lstm_network import QGenNetworkLSTM
from guesswhat.models.qgen.qgen_wrapper import QGenWrapper
from guesswhat.train.utils import test_model, compute_qgen_accuracy


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Question generator (policy gradient baseline))')

    parser.add_argument("-data_dir", type=str, required=True, help="Directory with data")
    parser.add_argument("-exp_dir", type=str, required=True, help="Directory in which experiments are stored")
    parser.add_argument("-img_dir", type=str, help='Directory with images')
    parser.add_argument("-crop_dir", type=str, help='Directory with images')
    parser.add_argument("-config", type=str, required=True, help='Config file')
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Dictionary file name")

    parser.add_argument("-networks_dir", type=str, help="Directory with pretrained networks")
    parser.add_argument("-oracle_identifier", type=str, required=True,
                        help='Oracle identifier')  # Use checkpoint id instead?
    parser.add_argument("-qgen_identifier", type=str, required=True, help='Qgen identifier')
    parser.add_argument("-guesser_identifier", type=str, required=True, help='Guesser identifier')
    parser.add_argument("-load_rl", type=bool, default=False, help="Load RL model weights")
    parser.add_argument("-evaluate_all", type=lambda x: bool(strtobool(x)), default="False",
                        help="Evaluate sampling, greedy and BeamSearch?")  # TODO use an input list
    parser.add_argument("-store_games", type=lambda x: bool(strtobool(x)), default="True",
                        help="Should we dump the game at evaluation times")
    parser.add_argument("-gpu_ratio", type=float, default=0.95, help="How muany GPU ram is required? (ratio)")
    parser.add_argument("-no_thread", type=int, default=1, help="No thread to load batch")

    args = parser.parse_args()

    loop_config, exp_identifier, save_path = load_config(args.config, args.exp_dir)

    # Load all  networks configs
    oracle_config = get_config_from_xp(os.path.join(args.networks_dir, "oracle"), args.oracle_identifier)
    guesser_config = get_config_from_xp(os.path.join(args.networks_dir, "guesser"), args.guesser_identifier)
    qgen_config = get_config_from_xp(os.path.join(args.networks_dir, "qgen"), args.qgen_identifier)

    logger = logging.getLogger()

    ###############################
    #  LOAD DATA
    #############################

    # Load image
    logger.info('Loading images..')
    image_builder = get_img_builder(qgen_config['model']['image'], args.img_dir, custom_features=True)

    crop_builder = None
    if oracle_config['inputs'].get('crop', False):
        logger.info('Loading crops..')
        crop_builder = get_img_builder(oracle_config['model']['crop'], args.crop_dir, is_crop=True)

    # Load dictionary
    logger.info('Loading dictionary..')
    tokenizer = GWTokenizer(os.path.join(args.data_dir, args.dict_file))

    ###############################
    #  LOAD NETWORKS
    #############################

    logger.info('Building networks..')

    qgen_network = QGenNetworkLSTM(qgen_config["model"], num_words=tokenizer.no_words, policy_gradient=True)
    qgen_var = [v for v in tf.global_variables() if "qgen" in v.name]  # and 'rl_baseline' not in v.name
    qgen_saver = tf.train.Saver(var_list=qgen_var)

    oracle_network = OracleNetwork(oracle_config, num_words=tokenizer.no_words)
    oracle_var = [v for v in tf.global_variables() if "oracle" in v.name]
    oracle_saver = tf.train.Saver(var_list=oracle_var)

    guesser_network = GuesserNetwork(guesser_config["model"], num_words=tokenizer.no_words)
    guesser_var = [v for v in tf.global_variables() if "guesser" in v.name]
    guesser_saver = tf.train.Saver(var_list=guesser_var)

    loop_saver = tf.train.Saver(allow_empty=False)

    ###############################
    #  START TRAINING
    #############################

    # Load config
    batch_size = loop_config['optimizer']['batch_size']
    no_epoch = loop_config["optimizer"]["no_epoch"]

    mode_to_evaluate = ["greedy"]
    if args.evaluate_all:
        mode_to_evaluate = ["greedy", "sampling", "beam_search"]

    # create a saver to store/load checkpoint
    saver = tf.train.Saver()

    # CPU/GPU option
    cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        ###############################
        #  LOAD PRE-TRAINED NETWORK
        #############################

        sess.run(tf.global_variables_initializer())
        if args.load_rl:  # TODO only reload qgen ckpt
            # use RL model for evaluation
            qgen_saver.restore(sess, save_path.format('params.ckpt'))
        else:
            # use SL qgen model for evaluation
            qgen_var_supervized = [v for v in tf.global_variables() if "qgen" in v.name and 'rl_baseline' not in v.name]
            qgen_loader_supervized = tf.train.Saver(var_list=qgen_var_supervized)
            qgen_loader_supervized.restore(sess,
                                           os.path.join(args.networks_dir, 'qgen', args.qgen_identifier, 'params.ckpt'))

        oracle_saver.restore(sess, os.path.join(args.networks_dir, 'oracle', args.oracle_identifier, 'params.ckpt'))
        guesser_saver.restore(sess, os.path.join(args.networks_dir, 'guesser', args.guesser_identifier, 'params.ckpt'))

        # create training tools
        loop_sources = qgen_network.get_sources(sess)
        logger.info("Sources: " + ', '.join(loop_sources))

        evaluator = Evaluator(loop_sources, qgen_network.scope_name, network=qgen_network, tokenizer=tokenizer)

        eval_batchifier = LooperBatchifier(tokenizer, generate_new_games=False)

        # Initialize the looper to eval/train the game-simulation

        oracle_wrapper = OracleWrapper(oracle_network, tokenizer)
        guesser_wrapper = GuesserWrapper(guesser_network)
        qgen_network.build_sampling_graph(qgen_config["model"], tokenizer=tokenizer,
                                          max_length=loop_config['loop']['max_depth'])
        qgen_wrapper = QGenWrapper(qgen_network, tokenizer,
                                   max_length=loop_config['loop']['max_depth'],
                                   k_best=loop_config['loop']['beam_k_best'])

        looper_evaluator = BasicLooper(loop_config,
                                       oracle_wrapper=oracle_wrapper,
                                       guesser_wrapper=guesser_wrapper,
                                       qgen_wrapper=qgen_wrapper,
                                       tokenizer=tokenizer,
                                       batch_size=loop_config["optimizer"]["batch_size"])

        # Compute the initial scores
        logger.info(">>>-------------- INITIAL SCORE ---------------------<<<")

        for split in ["nd_test", "nd_valid", "od_test", "od_valid"]:
            logger.info("Loading dataset split {}".format(split))
            testset = Dataset(args.data_dir, split, "guesswhat_nocaps", image_builder, crop_builder)

            logger.info(">>>  New Games  <<<")
            compute_qgen_accuracy(sess, testset, batchifier=eval_batchifier, evaluator=looper_evaluator,
                                  tokenizer=tokenizer,
                                  mode=mode_to_evaluate, save_path=save_path, cpu_pool=cpu_pool, batch_size=batch_size,
                                  store_games=args.store_games, dump_suffix="init.new_games")
            logger.info(">>>------------------------------------------------<<<")
