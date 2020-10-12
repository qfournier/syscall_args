import pickle


class Dictionary(object):
    def __init__(self, path=None):
        if path is None:
            self.word2idx = {'[MASK]': 0}
            self.idx2word = ['[MASK]']
        else:
            with open('{}'.format(path), 'rb') as file:
                self.word2idx, self.idx2word = pickle.load(file)

    def add_word(self, word):
        """Add a word to the dictionary

        Args:
            word (str): word to add

        Returns:
            int: index of the word
        """
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def save(self, path):
        with open('{}'.format(path), 'wb') as file:
            pickle.dump([self.word2idx, self.idx2word], file)
