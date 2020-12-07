import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, n_syscall, n_process, args):
        super(LSTM, self).__init__()

        # Get which arguments are disabled
        self.disable_entry = args.disable_entry
        self.disable_ret = args.disable_ret
        self.disable_time = args.disable_time
        self.disable_proc = args.disable_proc
        self.disable_pid = args.disable_pid
        self.disable_tid = args.disable_tid

        # Get arguments
        self.emb_sys = args.emb_sys
        self.emb_proc = args.emb_proc
        self.emb_pid = args.emb_pid
        self.emb_tid = args.emb_tid
        self.emb_time = args.emb_time

        # Compute the embedding size
        self.emb_dim = self.emb_sys
        if not self.disable_time:
            self.emb_dim += +self.emb_time
        if not self.disable_proc:
            self.emb_dim += self.emb_proc
        if not self.disable_pid:
            self.emb_dim += self.emb_pid
        if not self.disable_tid:
            self.emb_dim += self.emb_tid

        # Embeddings
        self.embedding_call = nn.Embedding(n_syscall,
                                           self.emb_sys,
                                           padding_idx=0)
        if not self.disable_entry:
            self.embedding_entry = nn.Embedding(3, self.emb_sys, padding_idx=0)
        if not self.disable_ret:
            self.embedding_ret = nn.Embedding(3, self.emb_sys, padding_idx=0)
        if not self.disable_proc:
            self.embedding_proc = nn.Embedding(n_process,
                                               self.emb_proc,
                                               padding_idx=0)
        self.dropout = nn.Dropout(args.dropout)

        # LSTM
        self.hidden_dim = args.hiddens
        self.layers = args.layers
        self.lstm = nn.LSTM(input_size=self.emb_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.layers,
                            batch_first=True,
                            dropout=args.dropout)

        # Decoder
        self.decoder = nn.Linear(args.hiddens, n_syscall)

        self.init_weights()

    # see https://discuss.pytorch.org/t/correct-way-to-declare
    # -hidden-and-cell-states-of-lstm/15745/3
    def init_hidden(self, batch_size):
        hidden = Variable(next(self.parameters()).data.new(
            self.layers, batch_size, self.hidden_dim),
                          requires_grad=False)
        cell = Variable(next(self.parameters()).data.new(
            self.layers, batch_size, self.hidden_dim),
                        requires_grad=False)
        return hidden.zero_(), cell.zero_()

    def init_weights(self):
        """Initialize weights using the uniform distribution proposed by Xavier
        & Bengio."""
        nn.init.xavier_uniform_(self.embedding_call.weight)
        if not self.disable_entry:
            nn.init.xavier_uniform_(self.embedding_entry.weight)
        if not self.disable_ret:
            nn.init.xavier_uniform_(self.embedding_ret.weight)
        if not self.disable_proc:
            nn.init.xavier_uniform_(self.embedding_proc.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        self.decoder.bias.data.zero_()

    def embedding(self, call, entry, ret, time, proc, pid, tid):
        size = call.shape

        # System call embedding
        emb = self.embedding_call(call)
        if not self.disable_entry:
            emb += self.embedding_entry(entry)
        if not self.disable_ret:
            emb += self.embedding_ret(ret)

        # Process embedding
        if not self.disable_proc:
            emb = torch.cat((emb, self.embedding_proc(proc)), -1)
        if not self.disable_pid:
            pid = pid.unsqueeze(2)
            denominator = torch.exp(
                torch.arange(0, self.emb_pid, 2).float() *
                (-math.log(1000.0) / self.emb_pid)).to(call.device)
            pid_enc = torch.zeros(size[0], size[1],
                                  self.emb_pid).to(call.device)
            pid_enc[:, :, 0::2] = torch.sin(pid * denominator)
            pid_enc[:, :, 1::2] = torch.cos(pid * denominator)
            emb = torch.cat((emb, pid_enc), -1)
        if not self.disable_tid:
            tid = tid.unsqueeze(2)
            denominator = torch.exp(
                torch.arange(0, self.emb_tid, 2).float() *
                (-math.log(1000.0) / self.emb_tid)).to(call.device)
            tid_enc = torch.zeros(size[0], size[1],
                                  self.emb_tid).to(call.device)
            tid_enc[:, :, 0::2] = torch.sin(tid * denominator)
            tid_enc[:, :, 1::2] = torch.cos(tid * denominator)
            emb = torch.cat((emb, tid_enc), -1)

        # Positional encoding using the timestamp
        if not self.disable_time:
            position = time.type(torch.float64).unsqueeze(2) / 1000
            denominator = torch.exp(
                torch.arange(0, self.emb_time, 2).float() *
                (-math.log(10000.0) / self.emb_time)).to(call.device)
            pos_enc = torch.zeros(size[0], size[1],
                                  self.emb_time).to(call.device)
            pos_enc[:, :, 0::2] = torch.sin(position * denominator)
            pos_enc[:, :, 1::2] = torch.cos(position * denominator)
            emb = torch.cat((emb, pos_enc), -1)

        return self.dropout(emb)

    def forward(self, x, entry, ret, time, proc, pid, tid, pad_mask, mlm, chk):
        emb = self.embedding(x, entry, ret, time, proc, pid, tid)

        # LSTM
        h_t, c_t = self.init_hidden(emb.size(0))
        h_t, c_t = self.lstm(emb, (h_t, c_t))

        # Classifier
        out = self.decoder(h_t)

        return out
