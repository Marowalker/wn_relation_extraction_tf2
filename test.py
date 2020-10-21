from models.model_cnn import CnnModel, EmbeddingModule
import constants
from data_utils import get_trimmed_w2v_vectors
import tensorflow as tf
import numpy as np


embeddings = get_trimmed_w2v_vectors(constants.TRIMMED_W2V)
emb = EmbeddingModule(embeddings)
_ = emb([])

dummy_eb2 = tf.Variable(np.zeros((1, 6)), name="dummy2", dtype=tf.float32, trainable=False)
print([var for var in emb.variables])
# print(emb.embeddings_pos)
#
# emb.embeddings_pos = tf.concat([dummy_eb2, emb.embeddings_pos], axis=0)
#
# print(emb.embeddings_pos)
