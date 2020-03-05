import re

def read_file(filepath):
    with open(filepath) as fin:
        for line in fin:
            pass

class Dictionary():
    def __init__(self):
        self.word2idx = dict()
        self.idx2word = list()


    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1


class Corpus():
    def __init__(self, train_path):
        self.dictionary = Dictionary()
        self.train = self.import_from_file(train_path)


    def import_from_file(self, path):
        ids, sentences, labels = list(), list(), list()
        e1_poss, e2_poss = list(), list()
        idx, sentence, label = str(), str(), str()
        e1_pos, e2_pos = -1, -1
        with open(path) as fin:
            for line in [line.strip() for line in fin]:
                if len(line) == 0:
                    ids.append(idx)
                    sentences.append(sentence)
                    labels.append(label)
                    ids, sentences, labels = list(), list(), list()
                    e1_poss, e2_poss = list(), list()
                    e1_pos, e2_pos = -1, -1
                else:
                    splitted = line.split('\t')
                    if len(splitted) > 1:
                        idx = splitted[0]
                        sentence = splitted[1]
                        sentence, e1_pos, e2_pos = self.normalize_single_sentence(sentence)
                    else:
                        label = splitted[0]
        return sentences, labels, e1_poss, e2_poss


    def normalize_single_sentence(self, sentence):
        e1_pos = -1
        e1_pos = -1
        e1_regex = re.compile('<e1>(.*)</e1>')
        e2_regex = re.compile('<e2>(.*)</e2>')
        normalized_ =  list()
        for pos, word in enumerate(re.split(' ', sentence)):
            e1_matched = e1_regex.search(word)
            e2_matched = e2_regex.search(word)
            if e1_matched:
                normalized_word = e1_matched.group(1)
                e1_pos = pos
            elif e2_matched:
                normalized_word = e2_matched.group(1)
                e2_pos = pos
            else:
                normalized_word = word
            normalized_.append(normalized_word)
        return ' '.join(normalized_), e1_pos, e2_pos
    
if __name__ == '__main__':
    corpus = Corpus('../data/train.txt')
    # print(corpus.train)
    # print(len(corpus.train))
    train_sent, train_label = corpus.train
    ts = train_sent
    tl = train_label
    for m, n in zip(ts, tl):
        print(m, n, sep='\t', end='\n\n')

            