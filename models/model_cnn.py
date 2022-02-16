import os
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle, resample
from dataset import pad_sequences
from utils import Timer, Log
from data_utils import countNumRelation, countNumPos, countNumSynset, countVocab
import constants
from sklearn.metrics import f1_score

tf.random.Generator = None

seed = 1234
np.random.seed(seed)

tf.compat.v1.disable_eager_execution()


class CnnModel:
    def __init__(self, model_name, embeddings, triples, wordnet, batch_size):
        self.model_name = model_name
        self.embeddings = embeddings
        self.triples = triples
        self.batch_size = batch_size
        self.wordnet_emb = wordnet

        self.max_length = constants.MAX_LENGTH
        # Num of dependency relations
        self.num_of_depend = countNumRelation()
        # Num of pos tags
        self.num_of_pos = countNumPos()
        self.num_of_synset = countNumSynset()
        self.num_of_siblings = countVocab()
        self.num_of_class = len(constants.ALL_LABELS)
        self.trained_models = constants.TRAINED_MODELS
        self.initializer = tf.initializers.glorot_normal()

    def _add_placeholders(self):
        """
        Adds placeholders to self
        """
        self.labels = tf.compat.v1.placeholder(name="labels", shape=[None], dtype='int32')
        # Indexes of first channel (word + dependency relations)
        self.word_ids = tf.compat.v1.placeholder(name='word_ids', shape=[None, None], dtype='int32')
        # Indexes of channel (sibling + dependency relations)
        self.sibling_ids = tf.compat.v1.placeholder(name='sibling_ids', shape=[None, None, None], dtype='int32')
        # Indexes of third channel (position + dependency relations)
        self.positions_1 = tf.compat.v1.placeholder(name='positions_1', shape=[None, None], dtype='int32')
        # Indexes of third channel (position + dependency relations)
        self.positions_2 = tf.compat.v1.placeholder(name='positions_2', shape=[None, None], dtype='int32')
        # Indexes of second channel (pos tags + dependency relations)
        self.pos_ids = tf.compat.v1.placeholder(name='pos_ids', shape=[None, None], dtype='int32')
        # Indexes of fourth channel (synset + dependency relations)
        self.synset_ids = tf.compat.v1.placeholder(name='synset_ids', shape=[None, None], dtype='int32')

        # self.triple_ids = tf.compat.v1.placeholder(name='triple_ids', shape=[None, None], dtype='int32')

        self.relations = tf.compat.v1.placeholder(name='relations', shape=[None, None], dtype='int32')
        self.dropout_embedding = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="dropout_embedding")
        self.dropout = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.is_training = tf.compat.v1.placeholder(tf.bool, name='phase')

    def _add_word_embeddings_op(self):
        """
        Adds word embeddings to self
        """
        with tf.compat.v1.variable_scope("embedding"):
            # Create dummy embedding vector for index 0 (for padding)
            dummy_eb = tf.Variable(np.zeros((1, constants.INPUT_W2V_DIM)), name="dummy", dtype=tf.float32,
                                   trainable=False)
            # Create dependency relations randomly
            embeddings_re = tf.Variable(self.initializer(shape=[self.num_of_depend + 1, constants.INPUT_W2V_DIM],
                                                         dtype=tf.float32), name="re_lut")
            # create direction vectors randomly
            embedding_dir = tf.Variable(self.initializer(shape=[3, constants.INPUT_W2V_DIM], dtype=tf.float32),
                                        name="dir_lut")
            # Concat dummy vector and relations vectors
            embeddings_re = tf.concat([dummy_eb, embeddings_re], axis=0)
            # Concat relation vectors and direction vectors
            embeddings_re = tf.concat([embeddings_re, embedding_dir], axis=0)

            embeddings_sb = tf.Variable(self.initializer(shape=[self.num_of_siblings + 1, 15], dtype=tf.float32),
                                        name="sb_lut")
            dummy_eb5 = tf.Variable(np.zeros((1, 15)), name="dummy5", dtype=tf.float32, trainable=False)

            embeddings_sb = tf.concat([dummy_eb5, embeddings_sb], axis=0)
            dummy_eb_ex = tf.Variable(np.zeros((self.num_of_siblings + 2, constants.INPUT_W2V_DIM - 15)),
                                      name="dummy_ex", dtype=tf.float32,
                                      trainable=False)
            embeddings_sb = tf.concat([embeddings_sb, dummy_eb_ex], axis=-1)

            all_sb_rel_table = tf.concat([embeddings_re, embeddings_sb], axis=0)
            all_sb_rel_lookup = tf.nn.embedding_lookup(params=all_sb_rel_table, ids=self.sibling_ids)

            weights = tf.Variable(self.initializer(shape=[1, constants.INPUT_W2V_DIM], dtype=tf.float32),
                                  name="weights", trainable=True)

            all_sb_mean = all_sb_rel_lookup * weights

            # p_sibling = tf.reduce_max(p_sibling, axis=2)
            p_sibling = tf.reduce_mean(input_tensor=all_sb_mean, axis=2)
            self.sibling_embeddings = tf.nn.dropout(p_sibling, 1 - self.dropout_embedding)

            # Create word embedding tf variable
            embedding_wd = tf.Variable(self.embeddings, name="lut", dtype=tf.float32, trainable=False)
            # embedding_wd = tf.concat([embedding_wd, embeddings_sb], axis=0)
            embedding_wd = tf.concat([embedding_wd, embeddings_re], axis=0)
            # Lookup from indexs to vectors of words and dependency relations
            self.word_embeddings = tf.nn.embedding_lookup(params=embedding_wd, ids=self.word_ids)
            self.word_embeddings = tf.nn.dropout(self.word_embeddings, 1 - self.dropout_embedding)

            # embedding_tr = tf.Variable(self.triples, name='triple_lut', dtype=tf.float32, trainable=False)
            # embedding_tr = tf.concat([dummy_eb, embedding_tr], axis=0)
            # self.triple_embeddings = tf.nn.embedding_lookup(params=embedding_tr, ids=self.triple_ids)
            # self.triple_embeddings = tf.nn.dropout(self.triple_embeddings, 1 - self.dropout_embedding)

            # Create pos tag embeddings randomly
            dummy_eb2 = tf.Variable(np.zeros((1, 6)), name="dummy2", dtype=tf.float32, trainable=False)
            embeddings_re2 = tf.Variable(self.initializer(shape=[self.num_of_depend + 1, 6], dtype=tf.float32),
                                         name="re_lut2")
            embeddings_re2 = tf.concat([dummy_eb2, embeddings_re2], axis=0)
            embedding_dir2 = tf.Variable(self.initializer(shape=[3, 6], dtype=tf.float32), name="dir2_lut")
            embeddings_re2 = tf.concat([embeddings_re2, embedding_dir2], axis=0)
            embeddings_pos = tf.Variable(self.initializer(shape=[self.num_of_pos + 1, 6], dtype=tf.float32),
                                         name='pos_lut')
            embeddings_pos = tf.concat([dummy_eb2, embeddings_pos], axis=0)
            embeddings_pos = tf.concat([embeddings_pos, embeddings_re2], axis=0)
            self.pos_embeddings = tf.nn.embedding_lookup(params=embeddings_pos, ids=self.pos_ids)
            self.pos_embeddings = tf.nn.dropout(self.pos_embeddings, 1 - self.dropout_embedding)

            # Create synset embeddings randomly
            dummy_eb4 = tf.Variable(np.zeros((1, 18)), name="dummy4", dtype=tf.float32,
                                    trainable=False)
            embeddings_re4 = tf.Variable(self.initializer(shape=[self.num_of_depend + 1, 18],
                                                          dtype=tf.float32), name="re_lut4")
            # embeddings_re4 = tf.random.uniform(name="re_lut4", shape=[self.num_of_depend + 1, 12], minval=0,
            #                                    maxval=1e-4)
            embeddings_re4 = tf.concat([dummy_eb4, embeddings_re4], axis=0)
            embedding_dir4 = tf.Variable(self.initializer(shape=[3, 18], dtype=tf.float32),
                                         name="dir4_lut")
            embeddings_re4 = tf.concat([embeddings_re4, embedding_dir4], axis=0)
            embeddings_synset = tf.Variable(self.wordnet_emb, name='syn_lut', dtype=tf.float32, trainable=False)
            embeddings_synset = tf.concat([dummy_eb4, embeddings_synset], axis=0)
            embeddings_synset = tf.concat([embeddings_synset, embeddings_re4], axis=0)
            self.synset_embeddings = tf.nn.embedding_lookup(params=embeddings_synset, ids=self.synset_ids)
            self.synset_embeddings = tf.nn.dropout(self.synset_embeddings, 1 - self.dropout_embedding)

            # Create position embeddings randomly, each vector has length of WORD EMBEDDINGS / 2
            embeddings_position = tf.Variable(self.initializer(shape=[self.max_length * 2, 25], dtype=tf.float32),
                                              name='position_lut', trainable=True)
            dummy_posi_emb = tf.Variable(np.zeros((1, 25)),
                                         dtype=tf.float32)  # constants.INPUT_W2V_DIM // 2)), dtype=tf.float32)
            embeddings_position = tf.concat([dummy_posi_emb, embeddings_position], axis=0)

            dummy_eb3 = tf.Variable(np.zeros((1, 50)), name="dummy3", dtype=tf.float32, trainable=False)
            embeddings_re3 = tf.Variable(self.initializer(shape=[self.num_of_depend + 1, 50], dtype=tf.float32),
                                         name="re_lut3")
            embeddings_re3 = tf.concat([dummy_eb3, embeddings_re3], axis=0)
            embedding_dir3 = tf.Variable(self.initializer(shape=[3, 50], dtype=tf.float32), name="dir3_lut")
            embeddings_re3 = tf.concat([embeddings_re3, embedding_dir3], axis=0)
            # Concat each position vector with half of each dependency relation vector
            embeddings_position1 = tf.concat([embeddings_position, embeddings_re3[:, :25]],
                                             axis=0)  # :int(constants.INPUT_W2V_DIM / 2)]], axis=0)
            embeddings_position2 = tf.concat([embeddings_position, embeddings_re3[:, 25:]],
                                             axis=0)  # int(constants.INPUT_W2V_DIM / 2):]], axis=0)
            # Lookup concatenated indexes vectors to create concatenated embedding vectors
            self.position_embeddings_1 = tf.nn.embedding_lookup(params=embeddings_position1, ids=self.positions_1)
            self.position_embeddings_1 = tf.nn.dropout(self.position_embeddings_1, 1 - self.dropout_embedding)
            self.position_embeddings_2 = tf.nn.embedding_lookup(params=embeddings_position2, ids=self.positions_2)
            self.position_embeddings_2 = tf.nn.dropout(self.position_embeddings_2, 1 - self.dropout_embedding)

            # Concat 2 position feature into single feature (third channel)
            self.position_embeddings = tf.concat([self.position_embeddings_1, self.position_embeddings_2], axis=-1)

    def _multiple_input_cnn_layers(self):
        # Create 5-channel features
        self.word_embeddings = tf.expand_dims(self.word_embeddings, -1)
        self.sibling_embeddings = tf.expand_dims(self.sibling_embeddings, -1)
        self.pos_embeddings = tf.expand_dims(self.pos_embeddings, -1)
        self.synset_embeddings = tf.expand_dims(self.synset_embeddings, -1)
        self.position_embeddings = tf.expand_dims(self.position_embeddings, -1)
        # self.triple_embeddings = tf.expand_dims(self.triple_embeddings, -1)

        # create CNN model
        cnn_outputs = []
        for k in constants.CNN_FILTERS:
            filters = constants.CNN_FILTERS[k]
            cnn_output_w = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(k, constants.INPUT_W2V_DIM),
                strides=(1, 1),
                activation='tanh',
                use_bias=False, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(self.word_embeddings)

            # cnn_output_tr = tf.keras.layers.Conv2D(
            #     filters=filters,
            #     kernel_size=(k, constants.INPUT_W2V_DIM),
            #     strides=(1, 1),
            #     activation='tanh',
            #     use_bias=False, padding="valid",
            #     kernel_initializer=tf.keras.initializers.GlorotNormal(),
            #     kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            # )(self.triple_embeddings)

            cnn_output_sb = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(k, constants.INPUT_W2V_DIM),
                strides=(1, 1),
                activation='tanh',
                use_bias=False, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(self.sibling_embeddings)

            cnn_output_pos = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(k, 6),
                strides=(1, 1),
                activation='tanh',
                use_bias=False, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(self.pos_embeddings)

            cnn_output_synset = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(k, 18),
                strides=(1, 1),
                activation='tanh',
                use_bias=False, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(self.synset_embeddings)

            cnn_output_position = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(k, 50),
                strides=(1, 1),
                activation='tanh',
                use_bias=False, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(self.position_embeddings)

            cnn_output = tf.concat(
                [cnn_output_w, cnn_output_sb, cnn_output_pos, cnn_output_synset, cnn_output_position],
                axis=1)
            # cnn_output = tf.concat(
            #     [cnn_output_w, cnn_output_tr, cnn_output_sb, cnn_output_pos, cnn_output_synset, cnn_output_position],
            #     axis=1)
            # cnn_output = tf.concat([cnn_output_w, cnn_output_pos, cnn_output_synset, cnn_output_position], axis=1)
            # cnn_output = cnn_output_w
            cnn_output = tf.reduce_max(input_tensor=cnn_output, axis=1)
            cnn_output = tf.reshape(cnn_output, [-1, filters])
            cnn_outputs.append(cnn_output)

        final_cnn_output = tf.concat(cnn_outputs, axis=-1)
        final_cnn_output = tf.nn.dropout(final_cnn_output, 1 - self.dropout)

        return final_cnn_output

    def _add_logits_op(self):
        """
        Adds logits to self
        """
        final_cnn_output = self._multiple_input_cnn_layers()
        hidden_1 = tf.keras.layers.Dense(
            units=128, name="hidden_1",
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(final_cnn_output)
        hidden_2 = tf.keras.layers.Dense(
            units=128, name="hidden_2",
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(hidden_1)
        self.outputs = tf.keras.layers.Dense(
            units=self.num_of_class,
            activation=tf.nn.softmax,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(hidden_2)
        self.logits = tf.nn.softmax(self.outputs)

    def _add_loss_op(self):
        with tf.compat.v1.variable_scope('loss_layers'):
            log_likelihood = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            regularizer = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.reduce_mean(input_tensor=log_likelihood)
            self.loss += tf.reduce_sum(input_tensor=regularizer)

    def _add_train_op(self):
        self.extra_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

        with tf.compat.v1.variable_scope("train_step"):
            tvars = tf.compat.v1.trainable_variables()
            grad, _ = tf.clip_by_global_norm(tf.gradients(ys=self.loss, xs=tvars), 100.0)
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
            self.train_op = optimizer.apply_gradients(zip(grad, tvars))

    def build(self):
        timer = Timer()
        timer.start("Building model...")

        self._add_placeholders()
        self._add_word_embeddings_op()
        self._add_logits_op()
        self._add_loss_op()
        self._add_train_op()

        timer.stop()

    def _next_batch(self, data, num_batch):
        start = 0
        idx = 0
        while idx < num_batch:
            # Get BATCH_SIZE samples each batch
            word_ids = data['words'][start:start + self.batch_size]
            sibling_ids = data['siblings'][start:start + self.batch_size]
            positions_1 = data['positions_1'][start:start + self.batch_size]
            positions_2 = data['positions_2'][start:start + self.batch_size]
            pos_ids = data['poses'][start:start + self.batch_size]
            synset_ids = data['synsets'][start:start + self.batch_size]
            relation_ids = data['relations'][start:start + self.batch_size]
            directions = data['directions'][start:start + self.batch_size]
            labels = data['labels'][start:start + self.batch_size]
            # triple_ids = data['triples'][start: start + self.batch_size]

            # Padding sentences to the length of longest one
            word_ids, _ = pad_sequences(word_ids, pad_tok=0, max_sent_length=self.max_length)
            sibling_ids, _ = pad_sequences(sibling_ids, pad_tok=0, max_sent_length=self.max_length, nlevels=2)
            positions_1, _ = pad_sequences(positions_1, pad_tok=0, max_sent_length=self.max_length)
            positions_2, _ = pad_sequences(positions_2, pad_tok=0, max_sent_length=self.max_length)
            pos_ids, _ = pad_sequences(pos_ids, pad_tok=0, max_sent_length=self.max_length)
            synset_ids, _ = pad_sequences(synset_ids, pad_tok=0, max_sent_length=self.max_length)
            relation_ids, _ = pad_sequences(relation_ids, pad_tok=0, max_sent_length=self.max_length)
            directions, _ = pad_sequences(directions, pad_tok=0, max_sent_length=self.max_length)
            # triple_ids, _ = pad_sequences(triple_ids, pad_tok=0, max_sent_length=self.max_length)

            # Create index matrix with words and dependency relations between words
            new_relation_ids = self.embeddings.shape[0] + relation_ids + directions
            word_relation_ids = np.zeros((word_ids.shape[0], word_ids.shape[1] + new_relation_ids.shape[1]))
            w_ids, rel_idxs = [], []
            for j in range(word_ids.shape[1] + new_relation_ids.shape[1]):
                if j % 2 == 0:
                    w_ids.append(j)
                else:
                    rel_idxs.append(j)
            word_relation_ids[:, w_ids] = word_ids
            word_relation_ids[:, rel_idxs] = new_relation_ids

            # Create index matrix with pos tags and dependency relations between pos tags
            new_relation_ids = self.num_of_siblings + 1 + relation_ids + directions
            sb_rels = []

            for i in range(sibling_ids.shape[2]):
                sb_rel = np.zeros([new_relation_ids.shape[0], sibling_ids.shape[1] + new_relation_ids.shape[1]])
                sb_rel[:, rel_idxs] = new_relation_ids
                sb_rels.append(sb_rel)

            sibling_relation_ids = np.dstack(tuple(sb_rels))
            sibling_relation_ids[:, w_ids, :] = sibling_ids
            #
            # Create index matrix with pos tags and dependency relations between pos tags
            new_relation_ids = self.num_of_pos + 1 + relation_ids + directions
            pos_relation_ids = np.zeros((pos_ids.shape[0], pos_ids.shape[1] + new_relation_ids.shape[1]))
            pos_relation_ids[:, w_ids] = pos_ids
            pos_relation_ids[:, rel_idxs] = new_relation_ids

            # Create index matrix with synsets and dependency relations between synsets
            new_relation_ids = self.num_of_synset + 1 + relation_ids + directions
            synset_relation_ids = np.zeros((synset_ids.shape[0], synset_ids.shape[1] + new_relation_ids.shape[1]))
            synset_relation_ids[:, w_ids] = synset_ids
            synset_relation_ids[:, rel_idxs] = new_relation_ids

            # Create index matrix with positions and dependency relations between positions
            new_relation_ids = self.max_length + 1 + relation_ids + directions
            positions_1_relation_ids = np.zeros(
                (positions_1.shape[0], positions_1.shape[1] + new_relation_ids.shape[1]))
            positions_1_relation_ids[:, w_ids] = positions_1
            positions_1_relation_ids[:, rel_idxs] = new_relation_ids

            # Create index matrix with positions and dependency relations between positions
            positions_2_relation_ids = np.zeros(
                (positions_2.shape[0], positions_2.shape[1] + new_relation_ids.shape[1]))
            positions_2_relation_ids[:, w_ids] = positions_2
            positions_2_relation_ids[:, rel_idxs] = new_relation_ids

            start += self.batch_size
            idx += 1
            yield positions_1_relation_ids, positions_2_relation_ids, word_relation_ids, sibling_relation_ids, \
                  pos_relation_ids, synset_relation_ids, relation_ids, labels
            # yield positions_1_relation_ids, positions_2_relation_ids, word_relation_ids, pos_relation_ids, \
            #     synset_relation_ids, relation_ids, labels

    def _train(self, epochs, early_stopping=True, patience=10, verbose=True):
        Log.verbose = verbose
        if not os.path.exists(self.trained_models):
            os.makedirs(self.trained_models)

        saver = tf.compat.v1.train.Saver(max_to_keep=2)
        best_f1 = 0
        n_epoch_no_improvement = 0
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            num_batch_train = len(self.dataset_train.labels) // self.batch_size + 1
            for e in range(epochs):
                # print(len(self.dataset_train.siblings))
                words_shuffled, siblings_shuffled, positions_1_shuffle, positions_2_shuffle, poses_shuffled, \
                synset_shuffled, relations_shuffled, directions_shuffled, labels_shuffled, triple_shuffled = shuffle(
                    # words_shuffled, positions_1_shuffle, positions_2_shuffle, poses_shuffled, synset_shuffled, \
                    #     relations_shuffled, directions_shuffled, labels_shuffled=shuffle(
                    self.dataset_train.words,
                    self.dataset_train.siblings,
                    self.dataset_train.positions_1,
                    self.dataset_train.positions_2,
                    self.dataset_train.poses,
                    self.dataset_train.synsets,
                    self.dataset_train.relations,
                    self.dataset_train.directions,
                    self.dataset_train.labels,
                    self.dataset_train.triples
                )

                data = {
                    'words': words_shuffled,
                    'siblings': siblings_shuffled,
                    'positions_1': positions_1_shuffle,
                    'positions_2': positions_2_shuffle,
                    'poses': poses_shuffled,
                    'synsets': synset_shuffled,
                    'relations': relations_shuffled,
                    'directions': directions_shuffled,
                    'labels': labels_shuffled,
                    # 'triples': triple_shuffled
                }

                for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch_train)):
                    positions_1, positions_2, word_ids, sibling_ids, pos_ids, synset_ids, relation_ids, labels, \
                         = batch
                    # positions_1, positions_2, word_ids, pos_ids, synset_ids, relation_ids, labels = batch
                    feed_dict = {
                        self.positions_1: positions_1,
                        self.positions_2: positions_2,
                        self.word_ids: word_ids,
                        self.sibling_ids: sibling_ids,
                        self.pos_ids: pos_ids,
                        self.synset_ids: synset_ids,
                        self.relations: relation_ids,
                        self.labels: labels,
                        # self.triple_ids: triple_ids,
                        self.dropout_embedding: 0.5,
                        self.dropout: 0.5,
                        self.is_training: True
                    }
                    _, _, loss_train = sess.run([self.train_op, self.extra_update_ops, self.loss], feed_dict=feed_dict)
                    if idx % 10 == 0:
                        Log.log("Iter {}, Loss: {} ".format(idx, loss_train))

                # stop by validation loss
                if early_stopping:
                    num_batch_val = len(self.dataset_validation.labels) // self.batch_size + 1
                    total_f1 = []

                    data = {
                        'words': self.dataset_validation.words,
                        'siblings': self.dataset_validation.siblings,
                        'positions_1': self.dataset_validation.positions_1,
                        'positions_2': self.dataset_validation.positions_2,
                        'poses': self.dataset_validation.poses,
                        'synsets': self.dataset_validation.synsets,
                        'relations': self.dataset_validation.relations,
                        'directions': self.dataset_validation.directions,
                        'labels': self.dataset_validation.labels,
                        # 'triples': self.dataset_validation.triples
                    }

                    for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch_val)):
                        positions_1, positions_2, word_ids, sibling_ids, pos_ids, synset_ids, relation_ids, labels, \
                             = batch
                        # positions_1, positions_2, word_ids, pos_ids, synset_ids, relation_ids, labels = batch
                        acc, f1 = self._accuracy(sess, feed_dict={
                            self.positions_1: positions_1,
                            self.positions_2: positions_2,
                            self.word_ids: word_ids,
                            self.sibling_ids: sibling_ids,
                            self.pos_ids: pos_ids,
                            self.synset_ids: synset_ids,
                            self.relations: relation_ids,
                            self.labels: labels,
                            # self.triple_ids: triple_ids,
                            self.dropout_embedding: 0.5,
                            self.dropout: 0.5,
                            self.is_training: True
                        })
                        total_f1.append(f1)

                    val_f1 = np.mean(total_f1)
                    Log.log("F1: {}".format(val_f1))
                    print("Best F1: ", best_f1)
                    print("F1 for epoch number {}: {}".format(e + 1, val_f1))
                    if val_f1 > best_f1:
                        saver.save(sess, self.model_name)
                        Log.log('Save the model at epoch {}'.format(e + 1))
                        best_f1 = val_f1
                        n_epoch_no_improvement = 0
                    else:
                        n_epoch_no_improvement += 1
                        Log.log("Number of epochs with no improvement: {}".format(n_epoch_no_improvement))
                        if n_epoch_no_improvement >= patience:
                            print("Best F1: {}".format(best_f1))
                            break

            if not early_stopping:
                saver.save(sess, self.model_name)

    def _accuracy(self, sess, feed_dict):
        feed_dict = feed_dict
        feed_dict[self.dropout_embedding] = 1.0
        feed_dict[self.dropout] = 1.0
        feed_dict[self.is_training] = False

        logits = sess.run(self.logits, feed_dict=feed_dict)
        accuracy = []
        f1 = []
        predict = []
        exclude_label = []
        for logit, label in zip(logits, feed_dict[self.labels]):
            logit = np.argmax(logit)
            exclude_label.append(label)
            predict.append(logit)
            accuracy += [logit == label]

        f1.append(f1_score(predict, exclude_label, average='macro'))
        return accuracy, np.mean(f1)

    def load_data(self, train, validation):
        timer = Timer()
        timer.start("Loading data")

        self.dataset_train = train
        self.dataset_validation = validation

        print("Number of training examples:", len(self.dataset_train.labels))
        print("Number of validation examples:", len(self.dataset_validation.labels))
        timer.stop()

    def run_train(self, epochs, early_stopping=True, patience=10):
        timer = Timer()
        timer.start("Training model...")
        self._train(epochs=epochs, early_stopping=early_stopping, patience=patience)
        timer.stop()

    def predict(self, test):
        saver = tf.compat.v1.train.Saver()
        with tf.compat.v1.Session() as sess:
            Log.log("Testing model over test set")
            saver.restore(sess, self.model_name)

            y_pred = []
            num_batch = len(test.labels) // self.batch_size + 1

            data = {
                'words': test.words,
                'siblings': test.siblings,
                'positions_1': test.positions_1,
                'positions_2': test.positions_2,
                'poses': test.poses,
                'synsets': test.synsets,
                'relations': test.relations,
                'directions': test.directions,
                'labels': test.labels,
                # 'triples': test.triples
            }

            for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch)):
                positions_1, positions_2, word_ids, sibling_ids, pos_ids, synset_ids, relation_ids, labels,  \
                    = batch
                # positions_1, positions_2, word_ids, pos_ids, synset_ids, relation_ids, labels = batch
                feed_dict = {
                    self.positions_1: positions_1,
                    self.positions_2: positions_2,
                    self.word_ids: word_ids,
                    self.sibling_ids: sibling_ids,
                    self.pos_ids: pos_ids,
                    self.synset_ids: synset_ids,
                    self.relations: relation_ids,
                    self.labels: labels,
                    # self.triple_ids: triple_ids,
                    self.dropout_embedding: 1,
                    self.dropout: 1,
                    self.is_training: False
                }
                logits = sess.run(self.logits, feed_dict=feed_dict)

                for logit in logits:
                    decode_sequence = np.argmax(logit)
                    y_pred.append(decode_sequence)

        return y_pred
