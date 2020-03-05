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
        self.label_dictionary = Dictionary()
        self.train = self.import_from_file(train_path)


    def import_from_file(self, path):
        ids, sentences, labels = list(), list(), list()
        e1_dists, e2_dists = list(), list()

        idx, sentence, label = str(), str(), str()
        e1_dist, e2_dist = list(), list()

        with open(path) as fin:
            for line in [line.strip() for line in fin]:
                if len(line) == 0:
                    ids.append(idx)
                    sentences.append(sentence)
                    labels.append(label)
                    e1_dists.append(e1_dist)
                    e2_dists.append(e2_dist)
                    idx, sentence, label = str(), str(), str()
                    e1_dist, e2_dist = list(), list()
                else:
                    splitted = line.split('\t')
                    if len(splitted) > 1:
                        idx = splitted[0]
                        sentence = splitted[1]
                        sentence, e1_pos, e2_pos = self.normalize_single_sentence(sentence)
                        e1_dist, e2_dist = self.get_relative_distance(len(sentence), e1_pos, e2_pos)
                    else:
                        label = splitted[0]
                        self.label_dictionary.add_word(label)
        return sentences, labels, e1_dists, e2_dists


    def normalize_single_sentence(self, sentence):
        e1_pos = -1
        e2_pos = -1
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
            self.dictionary.add_word(normalized_word)
            normalized_.append(normalized_word)
        return normalized_, e1_pos, e2_pos
    

    def get_relative_distance(self, length_of_sentence, e1_pos, e2_pos):
        e1_dist = [0] * length_of_sentence
        e2_dist = [1] * length_of_sentence

        for pos in range(length_of_sentence):
            e1_dist[pos] = pos - e1_pos
            e2_dist[pos] = pos - e2_pos

        return e1_dist, e2_dist


if __name__ == '__main__':
    corpus = Corpus('../data/train.txt')
 
    # train_sent, train_label = corpus.train
    # ts = train_sent
    # tl = train_label
    # for m, n in zip(ts, tl):
    #     print(m, n, sep='\t', end='\n\n')

            
