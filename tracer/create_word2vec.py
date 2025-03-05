from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

corpus_file = '/home/ao_ding/expand/trace/code/corpus.txt'
# 生成器函数，按行读取文件并进行分词
def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield word_tokenize(line.strip())

sentences = [sent for sent in read_corpus(corpus_file)]
print(sentences[:10])
# 训练Word2Vec模型
model = Word2Vec(sentences=sentences, vector_size=300, window=5, min_count=3, workers=4)

# 保存模型
model.save("word2vec_model.model")

# 测试模型
print(model.wv.most_similar("COMMAND_ARG1"))  # 替换成你感兴趣的词
