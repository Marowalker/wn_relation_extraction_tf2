import pickle

import numpy as np
from data_utils import load_vocab
import constants
import os
from collections import defaultdict


# NLPLAB_W2V = 'data/w2v_model/wikipedia-pubmed-and-PMC-w2v.bin'
# NLPLAB_W2V = 'data/w2v_model/BioWordVec_PubMed_MIMICIII_d200.vec.bin'
NLPLAB_W2V = 'data/w2v_model/w2v_retrain.bin'
os.chdir("..")
os.chdir("..")


def export_trimmed_nlplab_vectors(vocab, trimmed_filename, dim=200, bin=NLPLAB_W2V):
    """
    Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
        :param bin:
    """
    # embeddings contains embedding for the pad_tok as well
    embeddings = np.zeros([len(vocab) + 1, dim])
    with open(bin, 'rb') as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        print('nlplab vocab size', vocab_size)
        binary_len = np.dtype('float32').itemsize * layer1_size

        count = 0
        m_size = len(vocab)
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = b''.join(word)
                    break
                if ch != b'\n':
                    word.append(ch)
            word = word.decode("utf-8")

            if word in vocab:
                count += 1
                embedding = np.fromstring(f.read(binary_len), dtype='float32')
                word_idx = vocab[word]
                embeddings[word_idx] = embedding
            else:
                f.read(binary_len)

    print('Missing rate {}'.format(1.0 * (m_size - count)/m_size))
    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_all_offsets():
    file = open('data/wordnet-entities.txt')
    lines = file.readlines()
    temp = defaultdict()
    for (idx, line) in enumerate(lines):
        l = line.split()
        temp[l[0]] = idx + 1
    return temp


def trim_wordnet(filename):
    with open('data/w2v_model/wordnet_embeddings.pkl', 'rb') as f:
        emb = pickle.load(f)
    hyp_vocab = load_vocab(constants.ALL_SYNSETS)
    count = 0
    m_size = len(hyp_vocab)
    print("vobcab length: ", m_size)
    embeddings = np.zeros([len(hyp_vocab) + 1, emb.shape[-1]])
    all_offsets = get_all_offsets()

    for off in hyp_vocab:
        # print(off)
        if off in all_offsets:
            # print(off)
            count += 1
            idx = hyp_vocab[off]
            emb_idx = all_offsets[off]
            embeddings[idx] = emb[emb_idx]
    print('Missing rate {}'.format(1.0 * (m_size - count) / m_size))
    print("Embeddings: ")
    print(embeddings)
    np.savez_compressed(filename, embeddings=embeddings)

# vocab_words = load_vocab(constants.ALL_WORDS)
# export_trimmed_nlplab_vectors(vocab_words, 'w2v_retrain_nlplab.npz')


trim_wordnet('data/w2v_model/wordnet_embeddings.npz')


