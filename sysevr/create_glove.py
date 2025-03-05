import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = "/home/ao_ding/expand/SySeVR/glove/vectors.txt"
tmp_file = "glove.txt"
glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)
vector1 = model['(']
print(vector1.size)