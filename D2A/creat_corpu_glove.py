import get_dataset as gds
from nltk.tokenize import word_tokenize
corpus_file = '/home/ao_ding/expand/D2A/code/corpus.txt'
def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield word_tokenize(line.strip())

sentences = [sent for sent in read_corpus(corpus_file)]
new_corpus_file = '/home/ao_ding/expand/D2A/code/corpus_glove.txt'          
with open(new_corpus_file, 'w', encoding='utf-8') as f:
    for sentence in sentences:
        for word in sentence:
            f.write(word+" ")