import math
import copy
import torch
import torch.nn as nn
from torch.nn.modules.container import ModuleList
from torch.utils.checkpoint import checkpoint

import logging
logger = logging.getLogger('logger')


class Transformer(nn.Module):
    """Transformer with the Masked Language Model."""
    def __init__(self, n_syscall, n_process, args):
        """Constructor.

        Args:
            n_syscall (int): number of word in the vocabulary
            args (argparse.Namespace): arguments
        """
        super(Transformer, self).__init__()

        # Get which arguments are disabled
        self.disable_entry = args.disable_entry
        self.disable_ret = args.disable_ret
        self.disable_time = args.disable_time
        self.disable_proc = args.disable_proc
        self.disable_pid = args.disable_pid
        self.disable_tid = args.disable_tid
        self.disable_order = args.disable_order

        # Get arguments
        self.emb_sys = args.emb_sys
        self.emb_proc = args.emb_proc
        self.emb_pid = args.emb_pid
        self.emb_tid = args.emb_tid
        self.emb_order = args.emb_order
        self.emb_time = args.emb_time

        # Compute the embedding size
        self.emb_dim = self.emb_sys
        if not self.disable_proc:
            self.emb_dim += self.emb_proc
        if not self.disable_pid:
            self.emb_dim += self.emb_pid
        if not self.disable_tid:
            self.emb_dim += self.emb_tid
        if not self.disable_order:
            self.emb_dim += self.emb_order
        if not self.disable_time:
            self.emb_dim += self.emb_time

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

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(self.emb_dim, args.heads,
                                                   args.hiddens, args.dropout)
        self.encoder_layers = ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(args.layers)])
        self.encoder_norm = nn.modules.normalization.LayerNorm(self.emb_dim)

        # Decoder
        self.decoder = nn.Linear(self.emb_dim, n_syscall)

        self.init_weights()

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
            self.embedding_ret(ret)

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

        # Positional encoding of the ordering = [1, 2, 3, ...]
        if not self.disable_order:
            ordering = torch.arange(
                0, size[1], dtype=torch.float).unsqueeze(1).to(call.device)
            denominator = torch.exp(
                torch.arange(0, self.emb_order, 2).float() *
                (-math.log(10000.0) / self.emb_order)).to(call.device)
            pos_enc = torch.zeros(size[0], size[1], self.emb_order).to(call.device)
            pos_enc[:, :, 0::2] = torch.sin(ordering * denominator)
            pos_enc[:, :, 1::2] = torch.cos(ordering * denominator)
            emb = torch.cat((emb, pos_enc), -1)

        # Positional encoding of the timestamp
        if not self.disable_time:
            position = time.type(torch.float64).unsqueeze(2) / 1000
            denominator = torch.exp(
            torch.arange(0, self.emb_time, 2).float() *
            (-math.log(10000.0) / self.emb_time)).to(call.device)
            pos_enc = torch.zeros(size[0], size[1], self.emb_time).to(call.device)
            pos_enc[:, :, 0::2] = torch.sin(position * denominator)
            pos_enc[:, :, 1::2] = torch.cos(position * denominator)
            emb = torch.cat((emb, pos_enc), -1)

        return self.dropout(emb)

    def _generate_square_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def forward(self, x, entry, ret, time, proc, pid, tid, pad_mask, mlm, chk):
        # Embedding
        emb = self.embedding(x, entry, ret, time, proc, pid,
                             tid).permute(1, 0, 2)
        if mlm:
            # Transformer encoder
            for layer in self.encoder_layers:
                emb = checkpoint(layer, emb, None, pad_mask) if chk else layer(
                    emb, src_key_padding_mask=pad_mask)
        else:
            # Source mask
            src_mask = self._generate_square_mask(x.size(1)).to(x.device)
            # Transformer decoder
            for layer in self.encoder_layers:
                emb = checkpoint(layer, emb, src_mask,
                                 pad_mask) if chk else layer(
                                     emb, src_mask, pad_mask)

        return self.decoder(emb.permute(1, 0, 2))
