from torch.utils.data import Dataset


class LMDataset(Dataset):
    """Language modeling dataset."""
    def __init__(self, corpus):
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        return self.corpus.call[idx][:-1], self.corpus.entry[
            idx][:-1], self.corpus.ret[idx][:-1], self.corpus.time[
                idx][:-1], self.corpus.proc[idx][:-1], self.corpus.pid[
                    idx][:-1], self.corpus.tid[idx][:-1], self.corpus.call[
                        idx][1:]
