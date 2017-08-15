#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

class SkipGramModel():
    def __init__(self, input_file, vec_size=200, window_size=5, epochs=10, noise=5, learning_rate=0.1, anneal=0.998, t=0.00001):
        with open(input_file, 'r') as train_file:
            self.corpus = train_file.read().split()
        self.word_counts = {}
        index = 0
        print("Read corpus")
        for word in self.corpus:
            if word not in self.word_counts:
                self.word_counts[word] = [1, index]
                index += 1
            else:
                self.word_counts[word][0] += 1
        print("Created counter")
        count = len(self.corpus)
        print("Corpus size before subsample: %s" % len(self.corpus))
        self.corpus = [word for word in self.corpus if np.random.uniform(0, 1) > 1 - np.sqrt(t / (self.word_counts[word][0] / count))]
        print("Corpus size after subsample: %s" % len(self.corpus))
        print("Subsampled corpus")
        vocab_size = len(self.word_counts)
        self.syn0 = np.random.normal(0, 0.15, (vocab_size, vec_size))
        self.syn1 = np.zeros((vocab_size, vec_size))
        self._input, self._output = self.create_pairs(window_size)
        print("Created input output pairs")
        self.negative_samples_table = self.negative_samples()
        del self.corpus
        print("Created negative samples table, deleted corpus")
        #pickle.dump(self, open("preprocessed_skipgram_%d.p" % int(time.time()), 'wb'))
        self.train(vec_size=vec_size, learning_rate=learning_rate, epochs=epochs, noise=noise, anneal=anneal)

    def negative_samples(self):
        return [self.word_counts[word][1] for word in self.word_counts for i in range(int(np.power(self.word_counts[word][0], 3/4)))]

    def index(self, word):
        return self.word_counts[word][1]

    def create_pairs(self, window_size):
        corpus_size = len(self.corpus)
        X = [word for i, word in enumerate(self.corpus) for _ in range(window_size*2) if i >= window_size and i <= corpus_size - window_size - 1]
        Y = [self.corpus[i+j] for i in range(corpus_size) for j in range(-window_size, window_size + 1) if j != 0 and i >= window_size and i <= corpus_size - window_size - 1]
        return X, Y

    def train(self, vec_size, noise, learning_rate, epochs, anneal):
        X = self._input
        Y = self._output
        j = 0
        adagrad_syn0 = np.ones((len(self.word_counts), vec_size))
        adagrad_syn1 = np.ones((len(self.word_counts), vec_size))

        for _ in range(epochs):
            for center_word, target_word in zip(X, Y):
                center_index = self.index(center_word)
                target_index = self.index(target_word)
                input_vec = self.syn0[center_index]
                output_vec = self.syn1[target_index]
                neg_samples = [0] * noise

                for i in range(noise):
                    index = np.random.randint(0, len(self.negative_samples_table))
                    while self.negative_samples_table[index] == target_index:
                        index = np.random.randint(0, len(self.negative_samples_table))
                    neg_samples[i] = self.negative_samples_table[index]

                if j % 10000 == 0:
                    #learning_rate *= anneal
                    print("iteration: %d, learning_rate: %f" % (j, learning_rate))
                    loss_target = np.log(self.sigmoid(np.dot(input_vec, output_vec)))
                    loss_neg = sum([np.log(self.sigmoid(np.dot(-input_vec, self.syn1[index]))) for index in neg_samples])
                    loss = -loss_target - loss_neg
                    print("loss: %f, word: %s" % (loss, center_word))
                j += 1

                syn1_target_grad = self.gradient_output(input_vec, output_vec)
                syn0_center_grad = self.gradient_input(input_vec, output_vec)

                self.syn1[target_index] -= learning_rate * syn1_target_grad / np.sqrt(adagrad_syn1[target_index])
                
                for i in neg_samples:
                    syn1_neg_grad = self.gradient_output(input_vec, self.syn1[i], neg=True)
                    syn0_neg_grad = self.gradient_input(input_vec, self.syn1[i], neg=True)
                    adagrad_syn1[i] += np.square(syn1_neg_grad)
                    syn0_center_grad += syn0_neg_grad
                    self.syn1[i] -= learning_rate * syn1_neg_grad / np.sqrt(adagrad_syn1[i])

                self.syn0[center_index] -= learning_rate * syn0_center_grad / np.sqrt(adagrad_syn0[center_index])

                adagrad_syn1[target_index] += np.square(syn1_target_grad)
                adagrad_syn0[center_index] += np.square(syn0_center_grad)

        pickle.dump(self, open("trained_skipgram_%d" % int(time.time()), 'wb'))


    def get(self, words):
        return dict([(word, self.syn0[self.index(word)]) for word in words if word in self.word_counts])

    def get_all(self):
        return dict([(word, self.syn0[self.index(word)]) for word in self.word_counts])

    def gradient_output(self, input_vec, output_vec, neg=False):
        if neg:
            return (self.sigmoid(np.dot(input_vec, output_vec)) - 0) * input_vec
        return (self.sigmoid(np.dot(input_vec, output_vec)) - 1) * input_vec

    def gradient_input(self, input_vec, output_vec, neg=False):
        if neg:
            return (self.sigmoid(np.dot(input_vec, output_vec)) - 0) * output_vec
        return (self.sigmoid(np.dot(input_vec, output_vec)) - 1) * output_vec

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))


def run(input_file, vocab_file, output_file):
    vec_size=300
    window_size=5
    epochs=1
    noise=5
    learning_rate=0.05
    anneal=0.998
    t=10**-5
    load_file = False

    if load_file:
            skipgram = pickle.load(open("preprocessed_skipgram_1493369420.p", 'rb'))
            skipgram.train(learning_rate=learning_rate, epochs=epochs, noise=noise)
    else:
        skipgram = SkipGramModel(input_file, vec_size=vec_size, window_size=window_size, epochs=epochs, noise=noise, learning_rate=learning_rate, anneal=anneal, t=t)
    with open(vocab_file, 'r') as test_file:
        vocab = [word.strip() for word in test_file.readlines()]
    vecs = skipgram.get(vocab)
    with open(output_file, 'w') as output_file:
        for word, vec in vecs.items():
            print(word + ' ' + ' '.join(str("%.3f" % x) for x in vec), file=output_file)

if __name__ == "__main__":
    run('train.txt', 'test.txt', 'vectors.txt')
