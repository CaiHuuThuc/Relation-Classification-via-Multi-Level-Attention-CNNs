import os
import pandas as pd
import numpy as np
import re 
from collections import Counter
from gensim.models import KeyedVectors

#self.train_sentence_np mean numpy

class Dictionary():
    def __init__(self):
        self.word2idx = dict()
        self.idx2word = list()

        self.word2idx[''] = 0 # out of vocab
        self.idx2word.append('') # out of vocab

    def  add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

    def __len__(self):
        return len(self.idx2word)

    def to_idx(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return 0
    
    def write_to_file(self, out_fp):
        with open(out_fp, 'w') as fout:
            for word in sorted(self.idx2word):
                fout.write('%s\n' % word)
    
    def build_embedding_matrix(self, pretrained_fp):
        n_words = len(self.idx2word)
        dims = 300
        embedd_matrix = np.random.uniform(-.1, .1, size=(n_words, dims))
        pretrained_matrix = KeyedVectors.load_word2vec_format(pretrained_fp, binary=True)

        for idx, word in enumerate(self.word2idx):
            if word in pretrained_matrix:
                embedd_matrix[idx, :] = pretrained_matrix[word]
        
        return embedd_matrix


class Corpus():
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.label_dict = Dictionary()
        self.train_sentence, self.train_label  = self.tokenize(os.path.join(path, 'train_set.csv'))
        self.dev_sentence, self.dev_label  = self.tokenize(os.path.join(path, 'test_set.csv'))
        
        self.vocab_size = len(self.dictionary)
        self.num_classes = len(self.label_dict)

        self.train_sentence = self.whole_sentence_to_ids(self.train_sentence)
        self.train_label = self.whole_label_to_onehot_ids(self.train_label)
        
        self.n_train_sample = len(self.train_sentence)
        self.n_dev_sample = len(self.dev_sentence)

        self.dev_sentence = self.whole_sentence_to_ids(self.dev_sentence)
        self.dev_label = self.whole_label_to_onehot_ids(self.dev_label)
        self.n_dev_sample = len(self.dev_sentence)

        self.max_sentence_length = 5800
        
        self.train_max_sentence_length = max([len(sent) for sent in self.train_sentence])
        self.dev_max_sentence_length = max([len(sent) for sent in self.dev_sentence])

        self.train_sentence_np = np.zeros(shape=(self.n_train_sample), dtype=object)
        for idx, sent in enumerate(self.train_sentence):

            self.train_sentence_np[idx] = np.asarray(sent)

        self.dev_sentence_np = np.zeros(shape=(self.n_dev_sample), dtype=object)
        for idx, sent in enumerate(self.dev_sentence):
            self.dev_sentence_np[idx] = np.asarray(sent)

        print(self.train_max_sentence_length)
        print(self.dev_max_sentence_length)

        self.dictionary.write_to_file('../all_vocabs.txt')
        # print(self.train_sentence_np.shape)
        # print(self.train_label.shape)
        # print(self.dev_sentence_np.shape)
        # print(self.dev_label.shape)

    
    def tokenize(self, path):
        csv_data = pd.read_csv(path)
        sentences = csv_data['1']
        labels = csv_data['0']
        labels = [l if l != 'User interface' else 'User Interface' for l in labels]
        sentences = [re.sub(r'[\r\n\t\=\- ]+', ' ', sentence) for sentence in sentences]
        sentences = [re.split(r'[\r\n\t\.\(\)\,\;\'\"\$\:\&\/\@ ]', sentence) for sentence in sentences]

        for sentence in sentences:
            for word in sentence:
                self.dictionary.add_word(word)

        for label in labels:
            self.label_dict.add_word(label)     

        return sentences, labels


    def single_sentence_to_ids(self, sentence):
        sentence_ids = list()
        for word in sentence:
            idx = self.dictionary.to_idx(word)
            sentence_ids.append(idx)
        return sentence_ids

    def whole_sentence_to_ids(self, sentences):
        sentences_ids = list()
        for sentence in sentences:
            sentences_ids.append(self.single_sentence_to_ids(sentence))
        return sentences_ids
    
    def whole_label_to_onehot_ids(self, labels):
        labels_ids = list()
        for label in labels:
            labels_ids.append(self.label_dict.to_idx(label))
        labels_ids = np.asarray(labels_ids)
        labels_ids_onehot = np.zeros(shape=(labels_ids.shape[0], self.num_classes), dtype=np.int32)
        labels_ids_onehot[np.arange(labels_ids.size), labels_ids] = 1
        return labels_ids_onehot

    def batch_iter(self, n_epoches=50, batch_size=100):
        n_batches = self.n_train_sample // batch_size + 1

        for _ in range(n_epoches):
            shuffled_idx = np.random.permutation(self.n_train_sample)
            shuffled_sentences = self.train_sentence_np[shuffled_idx]
            shuffled_labels = self.train_label[shuffled_idx]
            for batch_idx in range(n_batches):
                start_idx = batch_size * batch_idx
                end_idx = min(start_idx + batch_size, self.n_train_sample)
                actual_batch_size = end_idx - start_idx + 1
                max_length_in_batch = max([sent.shape[0] for sent in shuffled_sentences[start_idx:end_idx]])
                unnormal_sent_batch = shuffled_sentences[start_idx:end_idx]
                sentences_batch = np.zeros((actual_batch_size, max_length_in_batch), dtype=np.int32)
                length_batch = np.asarray([len(sent) for sent in unnormal_sent_batch])
                for idx, sent in enumerate(unnormal_sent_batch):
                    leng = sent.shape[0]
                    sentences_batch[idx, :leng] = sent
                yield sentences_batch, shuffled_labels[start_idx:end_idx], length_batch, max_length_in_batch

if __name__ == '__main__':
    corpus = Corpus('../data_set/')
    
        
