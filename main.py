import pickle

import constants
from data_utils import *
from dataset import Dataset
from evaluate.bc5 import evaluate_bc5
from models.model_cnn import CnnModel
from sklearn.utils import shuffle

seed = 1234
np.random.seed(seed)


def main():
    result_file = open('data/results.txt', 'a')

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

        train = Dataset('data/raw_data/sdp_data_acentors_hypernyms.train.txt', 'data/raw_data/sdp_triple.train.txt',
                        vocab_words=vocab_words, vocab_poses=vocab_poses, vocab_synset=vocab_synsets,
                        vocab_chems=vocab_chems, vocab_dis=vocab_dis, vocab_rel=vocab_rel,
                        vocab_depends=vocab_depends)
        pickle.dump(train, open(constants.PICKLE_DATA + 'train.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
        dev = Dataset('data/raw_data/sdp_data_acentors_hypernyms.dev.txt', 'data/raw_data/sdp_triple.dev.txt',
                      vocab_words=vocab_words, vocab_poses=vocab_poses, vocab_synset=vocab_synsets,
                      vocab_chems=vocab_chems, vocab_dis=vocab_dis, vocab_rel=vocab_rel,
                      vocab_depends=vocab_depends)
        pickle.dump(dev, open(constants.PICKLE_DATA + 'dev.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
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

    # Train, Validation Split
    validation = Dataset('', '', process_data=False)
    train_ratio = 0.85
    n_sample = int(len(dev.words) * (2 * train_ratio - 1))
    props = ['words', 'siblings', 'positions_1', 'positions_2', 'labels', 'poses', 'synsets', 'relations', 'directions',
             'identities', 'triples']
    # props = ['words', 'positions_1', 'positions_2', 'labels', 'poses', 'synsets', 'relations', 'directions',
    #          'identities']
    for prop in props:
        train.__dict__[prop].extend(dev.__dict__[prop][:n_sample])
        validation.__dict__[prop] = dev.__dict__[prop][n_sample:]

    print("Train shape: ", len(train.words))
    print("Test shape: ", len(test.words))
    print("Validation shape: ", len(validation.words))

    # Get word embeddings
    embeddings = get_trimmed_w2v_vectors(constants.TRIMMED_W2V)
    wn_emb = get_trimmed_w2v_vectors('data/w2v_model/wordnet_embeddings.npz')
    # with open('data/w2v_model/wordnet_embeddings.pkl', 'rb') as f:
    #     wn_emb = pickle.load(f)
    with open('data/w2v_model/triple_embeddings.pkl', 'rb') as f:
        triple_emb = pickle.load(f)
    model = CnnModel(
        model_name=constants.MODEL_NAMES.format('cnn', constants.JOB_IDENTITY),
        embeddings=embeddings,
        triples=triple_emb,
        wordnet=wn_emb,
        batch_size=256
    )

    # Build model
    model.build()

    for i in range(1):
        model.load_data(train=train, validation=validation)
        model.run_train(epochs=constants.EPOCHS, early_stopping=constants.EARLY_STOPPING, patience=constants.PATIENCE)

        # Test on abstract
        answer = {}
        identities = test.identities
        # print(identities)
        y_pred = model.predict(test)
        for i in range(len(y_pred)):
            if y_pred[i] == 0:
                if identities[i][0] not in answer:
                    answer[identities[i][0]] = []

                if identities[i][1] not in answer[identities[i][0]]:
                    answer[identities[i][0]].append(identities[i][1])

        print(
            'result: abstract: ', evaluate_bc5(answer)
        )

        result_file.write(str(evaluate_bc5(answer)))
        result_file.write('\n')
    result_file.close()


if __name__ == '__main__':
    main()
