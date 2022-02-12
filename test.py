from models.model_cnn import CnnModel
import constants
from data_utils import get_trimmed_w2v_vectors
import tensorflow as tf
import numpy as np
from data_utils import load_vocab, make_triple_vocab
from dataset import Dataset
import pickle
from sklearn.utils import shuffle

seed = 1234
np.random.seed(seed)

# embeddings = get_trimmed_w2v_vectors(constants.TRIMMED_W2V)
# print(embeddings)
# emb = EmbeddingModule(embeddings)
# _ = emb([])
#
# dummy_eb2 = tf.Variable(np.zeros((1, 6)), name="dummy2", dtype=tf.float32, trainable=False)
# print([var for var in emb.variables])
# print(emb.embeddings_pos)
#
# emb.embeddings_pos = tf.concat([dummy_eb2, emb.embeddings_pos], axis=0)
#
# print(emb.embeddings_pos)

if constants.IS_REBUILD == 1:
    print('Build data')
    # Load vocabularies
    vocab_words = load_vocab(constants.ALL_WORDS)
    vocab_poses = load_vocab(constants.ALL_POSES)
    vocab_synsets = load_vocab(constants.ALL_SYNSETS)
    vocab_depends = load_vocab(constants.ALL_DEPENDS)

    vocab_chems = make_triple_vocab(constants.DATA + 'chemical2id.txt')
    vocab_dis = make_triple_vocab(constants.DATA + 'disease2id.txt')
    vocab_rel = make_triple_vocab(constants.DATA + 'relation2id.txt')

    # Create Dataset objects and dump into files
    train = Dataset('data/raw_data/sdp_data_acentors_hypernyms.train.txt', 'data/raw_data/sdp_triple.train.txt',
                    vocab_words=vocab_words, vocab_poses=vocab_poses, vocab_synset=vocab_synsets,
                    vocab_chems=vocab_chems, vocab_dis=vocab_dis, vocab_rel=vocab_rel,
                    vocab_depends=vocab_depends)
    # pickle.dump(train, open(constants.PICKLE_DATA + 'train.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
    dev = Dataset('data/raw_data/sdp_data_acentors_hypernyms.dev.txt', 'data/raw_data/sdp_triple.dev.txt',
                  vocab_words=vocab_words, vocab_poses=vocab_poses, vocab_synset=vocab_synsets,
                  vocab_chems=vocab_chems, vocab_dis=vocab_dis, vocab_rel=vocab_rel,
                  vocab_depends=vocab_depends)
    # pickle.dump(dev, open(constants.PICKLE_DATA + 'dev.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
    test = Dataset('data/raw_data/sdp_data_acentors_hypernyms.test.txt', 'data/raw_data/sdp_triple.test.txt',
                   vocab_words=vocab_words, vocab_poses=vocab_poses, vocab_synset=vocab_synsets,
                   vocab_chems=vocab_chems, vocab_dis=vocab_dis, vocab_rel=vocab_rel,
                   vocab_depends=vocab_depends)
    pickle.dump(test, open(constants.PICKLE_DATA + 'test.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
else:
    print('Load data')
    train = pickle.load(open(constants.PICKLE_DATA + 'train.pickle', 'rb'))
    dev = pickle.load(open(constants.PICKLE_DATA + 'dev.pickle', 'rb'))
    test = pickle.load(open(constants.PICKLE_DATA + 'test.pickle', 'rb'))

# print(train.words)
# print(train.triples)
validation = Dataset('', '', process_data=False)
train_ratio = 0.85
n_sample = int(len(dev.words) * (2 * train_ratio - 1))

props = ['words', 'siblings', 'positions_1', 'positions_2', 'labels', 'poses', 'synsets', 'relations', 'directions', 
         'identities', 'triples']
for prop in props:
    train.__dict__[prop].extend(dev.__dict__[prop][:n_sample])
    validation.__dict__[prop] = dev.__dict__[prop][n_sample:]

print("Train shape: ", len(train.words))
print("Test shape: ", len(test.words))
print("Validation shape: ", len(validation.words))

for i in range(20):
    state = np.random.randint(1, 10000)

    train.words, train.siblings, train.positions_1, train.positions_2, train.poses, train.synsets, train.relations, \
    train.directions, train.labels, train.triples = shuffle(
        train.words,
        train.siblings,
        train.positions_1,
        train.positions_2,
        train.poses,
        train.synsets,
        train.relations,
        train.directions,
        train.labels,
        train.triples,
        random_state=state
    )

    print(train.words[-1])

