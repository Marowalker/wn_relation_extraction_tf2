import argparse

ALL_LABELS = ['CID', 'NONE']

UNK = '$UNK$'

parser = argparse.ArgumentParser(description='Multi-region size CNN for re')
parser.add_argument('-i', help='Job identity', type=int, default=0)
parser.add_argument('-rb', help='Rebuild data', type=int, default=1)
parser.add_argument('-e', help='Number of epochs', type=int, default=80)
parser.add_argument('-p', help='Patience of early stop (0 for ignore early stop)', type=int, default=10)

parser.add_argument('-config', help='CNN configurations default \'1:128\'', type=str, default='2:32,3:96,4:32,5:64')
parser.add_argument('-len', help='Max sentence or document length', type=int, default=100)


opt = parser.parse_args()
print('Running opt: {}'.format(opt))

JOB_IDENTITY = opt.i
IS_REBUILD = opt.rb
EPOCHS = opt.e
EARLY_STOPPING = False if opt.p == 0 else True
PATIENCE = opt.p

# INPUT_W2V_DIM = 300
INPUT_W2V_DIM = 200
# INPUT_W2V_DIM = 768

MAX_LENGTH = opt.len

CNN_FILTERS = {}
if opt.config:
    print('Use model CNN with config', opt.config)
    USE_CNN = True
    CNN_FILTERS = {
        int(k): int(f) for k, f in [i.split(':') for i in opt.config.split(',')]
    }
else:
    raise ValueError('Configure CNN model to start')

DATA = 'data/'
RAW_DATA = DATA + 'raw_data/'
PICKLE_DATA = DATA + 'pickle/'
W2V_DATA = DATA + 'w2v_model/'

# ALL_WORDS = DATA + 'vocab.txt'
ALL_WORDS = DATA + 'vocab_lower.txt'
ALL_POSES = DATA + 'all_pos.txt'
ALL_SYNSETS = DATA + 'all_hypernyms.txt'
# ALL_DEPENDS = DATA + 'all_depend.txt'
ALL_DEPENDS = DATA + 'no_dir_depend.txt'

# TRIMMED_W2V = W2V_DATA + 'biobert_nlplab.npz'
TRIMMED_W2V = W2V_DATA + 'biowordvec_nlplab.npz'
# TRIMMED_W2V = W2V_DATA + 'trimmed_w2v.npz'

TRAINED_MODELS = DATA + 'trained_models/'
MODEL_NAMES = TRAINED_MODELS + '{}_{}'
