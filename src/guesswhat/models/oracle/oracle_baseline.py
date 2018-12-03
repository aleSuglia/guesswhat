import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers

from neural_toolbox import rnn

from generic.tf_utils.abstract_network import ResnetModel
from generic.tf_factory.image_factory import get_image_features, get_cbn
from generic.tf_factory.attention_factory import get_attention


class OracleNetwork(ResnetModel):

    def __init__(self, config, num_words, num_answers, device='', reuse=False):
        ResnetModel.__init__(self, "oracle", device=device)

        with tf.variable_scope(self.scope_name, reuse=reuse):
            embeddings = []

            batch_size = None
            self._is_training = tf.placeholder(tf.bool, name="is_training")

            dropout_keep_scalar = float(config["dropout_keep_prob"])
            dropout_keep = tf.cond(self._is_training,
                                   lambda: tf.constant(dropout_keep_scalar),
                                   lambda: tf.constant(1.0))

            # QUESTION
            self._question = tf.placeholder(tf.int32, [batch_size, None], name='question')
            self._seq_length = tf.placeholder(tf.int32, [batch_size], name='seq_length')

            word_emb = tfc_layers.embed_sequence(ids=self._question,
                                                 vocab_size=num_words,
                                                 embed_dim=config["question"]["word_embedding_dim"],
                                                 scope="word_embedding",
                                                 reuse=reuse)

            _, last_rnn_state = rnn.rnn_factory(inputs=word_emb,
                                                seq_length=self._seq_length,
                                                cell="lstm",
                                                num_hidden=config['question']["rnn_units"])

            # Prepare conditional batchnorm
            embeddings.append(last_rnn_state)

            # CATEGORY
            if config['inputs']['category']:
                self._category = tf.placeholder(tf.int32, [batch_size], name='category')

                cat_emb = tfc_layers.embed_sequence(ids=self._category,
                                                    vocab_size=config['category']["n_categories"] + 1,  # we add the unkwown category
                                                    embed_dim=config['category']["embedding_dim"],
                                                    scope="cat_embedding",
                                                    reuse=reuse)

                embeddings.append(cat_emb)
                print("Input: Category")

            # SPATIAL
            if config['inputs']['spatial']:
                self._spatial = tf.placeholder(tf.float32, [batch_size, 8], name='spatial')
                embeddings.append(self._spatial)
                print("Input: Spatial")

            # IMAGE
            if config['inputs']['image']:

                self._image = tf.placeholder(tf.float32, [batch_size] + config['image']["dim"], name='image')

                cbn = None
                if "cbn" in config:
                    cbn = get_cbn(config["cbn"], last_rnn_state, dropout_keep, self._is_training)

                self.image_out = get_image_features(image=self._image,
                                                    is_training=self._is_training,
                                                    config=config['image'],
                                                    cbn=cbn)

                if len(self.image_out.get_shape()) > 2:
                    with tf.variable_scope("image_pooling"):
                        self.image_out = get_attention(self.image_out, last_rnn_state,
                                                       is_training=self._is_training,
                                                       config=config["pooling"],
                                                       dropout_keep=dropout_keep,
                                                       reuse=reuse)

                embeddings.append(self.image_out)
                print("Input: Image")

            # CROP
            if config['inputs']['crop']:

                self._crop = tf.placeholder(tf.float32, [batch_size] + config['crop']["dim"], name='crop')

                cbn = None
                if "cbn" in config:
                    cbn = get_cbn(config["cbn"], last_rnn_state, dropout_keep, self._is_training)

                self.crop_out = get_image_features(image=self._crop,
                                                   is_training=self._is_training,
                                                   config=config['crop'],
                                                   cbn=cbn)

                if len(self.image_out.get_shape()) > 2:
                    with tf.variable_scope("crop_pooling"):
                        self.image_out = get_attention(self.crop_out, last_rnn_state,
                                                       is_training=self._is_training,
                                                       config=config["pooling"],
                                                       dropout_keep=dropout_keep,
                                                       reuse=reuse)

                embeddings.append(self.crop_out)
                print("Input: Crop")

            # Compute the final embedding
            emb = tf.concat(embeddings, axis=1)

            # OUTPUT
            self._answer = tf.placeholder(tf.float32, [batch_size, num_answers], name='answer')

            with tf.variable_scope('mlp'):
                num_hiddens = config['MLP']['num_hiddens']

                self.out = tfc_layers.fully_connected(emb, num_hiddens, activation_fn=tf.nn.relu, scope='l1')
                self.out = tfc_layers.fully_connected(self.out, num_answers, activation_fn=None, scope='None')

                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self._answer, name='cross_entropy')
                self.loss = tf.reduce_mean(self.cross_entropy)

                self.softmax = tf.nn.softmax(self.out, name='answer_prob')
                self.prediction = tf.argmax(self.out, axis=1, name='predicted_answer')  # no need to compute the softmax

                self.success = tf.equal(self.prediction, tf.argmax(self._answer, axis=1))  # no need to compute the softmax

        with tf.variable_scope('accuracy'):
            self.accuracy = tf.equal(self.prediction, tf.argmax(self._answer, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

            print('Model... Oracle build!')

    def get_loss(self):
        return self.loss

    def get_error(self):
        return 1 - self.accuracy

    def get_accuracy(self):
        return self.accuracy
