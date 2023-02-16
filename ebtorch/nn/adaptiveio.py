#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# A bare copy-paste from: https://github.com/taufique74/AdaptiveIO/blob/master/model.py
# (c) 2020 Taufiquzzaman Peyash (https://github.com/taufique74) - All Rights Reserved.
#
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import Sequential


stop_iteration_literal = (
    "Warning! Call to exhausted 'self.parameters()' iterator."
    "\nExplicit exception has been silenced, but no further action has been performed."
)


class VariationalDropout(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.0:
            return x

        batch_size = x.shape[1]
        mask = x.new_empty(1, batch_size, x.shape[2], requires_grad=False).bernoulli_(
            1 - self.dropout
        )
        x = x.masked_fill(mask == 0, 0) / (1 - self.dropout)

        return x


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False
    ):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    "When using the tied flag, nhid must be equal to emsize"
                )
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, myinput, hidden):
        emb = self.drop(self.encoder(myinput))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        try:
            weight = next(self.parameters())
        except StopIteration:
            print(stop_iteration_literal)
        if self.rnn_type == "LSTM":
            return (
                weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid),
            )
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class AwdLstm(nn.LSTM):
    def __init__(
        self,
        *args,
        dropouti: float = 0.5,
        dropoutw: float = 0.5,
        dropouto: float = 0.5,
        unit_forget_bias=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti)
        self.output_drop = VariationalDropout(dropouto)
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size : 2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                getattr(self, name).data = torch.nn.functional.dropout(
                    param.data, p=self.dropoutw, training=self.training
                ).contiguous()

    def forward(self, myinput, hx=None):
        self._drop_weights()
        self.flatten_parameters()
        myinput = self.input_drop(myinput)
        seq, state = super().forward(myinput, hx=hx)
        return self.output_drop(seq), state


class AdaptiveSoftmaxRNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        ntoken,
        ninp,
        nhid,
        nlayers,
        emb_dropout=0.0,
        rnn_dropout=0.2,
        tail_dropout=0.5,
        cutoffs=(20000, 50000),
        tie_weights=False,
        adaptive_input=False,
    ):
        super(AdaptiveSoftmaxRNN, self).__init__()

        cutoffs = list(cutoffs)

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.out_dropout = nn.Dropout(0.5)
        if adaptive_input:
            self.encoder = AdaptiveInput(ninp, ntoken, cutoffs, tail_drop=tail_dropout)
        else:
            self.encoder = nn.Embedding(ntoken, ninp)

        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=rnn_dropout)
        self.decoder = AdaptiveLogSoftmaxWithLoss(
            nhid, ntoken, cutoffs=cutoffs, div_value=2.0, tail_drop=tail_dropout
        )
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

        # weight sharing as described in the paper
        if tie_weights and adaptive_input:
            # for i in range(len(cutoffs)):
            for i in enumerate(cutoffs):
                i = i[0]
                self.encoder.tail[i][0].weight = self.decoder.tail[i][1].weight

                # sharing the projection layers
                self.encoder.tail[i][1].weight = torch.nn.Parameter(
                    self.decoder.tail[i][0].weight.transpose(0, 1)
                )

    # Very ugly workaround to save some memory until this is "fixed"!
    @staticmethod
    def init_weights():
        print(
            "You called the mathod 'init_weights()' which is currently not implemented. Nothing has been done, but execution has not been halted for backward-compatibility."
        )

    def forward(self, myinput, hidden, targets):
        emb = self.emb_dropout(self.encoder(myinput))  # (seq_len, bsz, ninp)
        output, hidden = self.rnn(emb, hidden)  # (seq_len, bsz, ninp)
        output = self.out_dropout(output)
        output = output.view(-1, output.size(2))  # (seq_len*bsz, ninp)
        output, loss = self.decoder(output, targets)
        return output, hidden, loss

    def init_hidden(self, bsz):
        try:
            weight = next(self.parameters())
        except StopIteration:
            print(stop_iteration_literal)
        return (
            weight.new_zeros(self.nlayers, bsz, self.nhid),
            weight.new_zeros(self.nlayers, bsz, self.nhid),
        )


class AdaptiveSoftmaxRNNImproved(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        ntoken,
        ninp,
        nhid,
        nlayers,
        emb_dropout=0.1,
        out_dropout=0.4,
        rnn_dropout=0.3,
        tail_dropout=0.3,
        cutoffs=(20000, 50000),
        tie_weights=True,
    ):
        super().__init__()

        cutoffs = list(cutoffs)

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.out_dropout = nn.Dropout(out_dropout)

        self.encoder = AdaptiveInput(ninp, ntoken, cutoffs, tail_drop=tail_dropout)
        self.rnn = AwdLstm(
            ninp,
            nhid,
            num_layers=nlayers,
            dropouti=0.1,
            dropouto=0.1,
            dropoutw=rnn_dropout,
        )

        self.decoder = AdaptiveLogSoftmaxWithLoss(
            nhid, ntoken, cutoffs=cutoffs, div_value=2.0, tail_drop=tail_dropout
        )
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

        # weight sharing as described in the paper
        if tie_weights:
            # for i in range(len(cutoffs)):
            for i in enumerate(cutoffs):
                i = i[0]
                self.encoder.tail[i][0].weight = self.decoder.tail[i][1].weight

                # sharing the projection layers
                self.encoder.tail[i][1].weight = torch.nn.Parameter(
                    self.decoder.tail[i][0].weight.transpose(0, 1)
                )

    # Very ugly workaround to save some memory until this is "fixed"!
    @staticmethod
    def init_weights():
        print(
            "You called the mathod 'init_weights()' which is currently not implemented. Nothing has been done, but execution has not been halted for backward-compatibility."
        )

    def forward(self, myinput, hidden, targets):
        emb = self.emb_dropout(self.encoder(myinput))  # (seq_len, bsz, ninp)
        output, hidden = self.rnn(emb, hidden)  # (seq_len, bsz, ninp)
        output = self.out_dropout(output)
        output = output.view(-1, output.size(2))  # (seq_len*bsz, ninp)
        output, loss = self.decoder(output, targets)
        return output, hidden, loss

    def init_hidden(self, bsz):
        try:
            weight = next(self.parameters())
        except StopIteration:
            print(stop_iteration_literal)
        return (
            weight.new_zeros(self.nlayers, bsz, self.nhid),
            weight.new_zeros(self.nlayers, bsz, self.nhid),
        )


class AdaptiveInput(nn.Module):
    def __init__(
        self,
        in_features,
        n_classes,
        cutoffs=None,
        div_value=2.0,
        head_bias=False,
        tail_drop=0.5,
    ):
        super(AdaptiveInput, self).__init__()
        if not cutoffs:
            cutoffs = [5000, 10000]
        cutoffs = list(cutoffs)

        if (
            (cutoffs != sorted(cutoffs))
            or (min(cutoffs) <= 0)
            or (max(cutoffs) >= (n_classes - 1))
            or (len(set(cutoffs)) != len(cutoffs))
            or any(int(c) != c for c in cutoffs)
        ):
            raise ValueError(
                "cutoffs should be a sequence of unique, positive "
                "integers sorted in an increasing order, where "
                "each value is between 1 and n_classes-1"
            )

        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias
        self.tail_drop = tail_drop

        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.cutoffs[0]

        self.head = nn.Embedding(self.head_size, self.in_features)
        #                                   nn.Linear(self.in_features, self.in_features, bias=self.head_bias))

        self.tail = nn.ModuleList()

        for i in range(self.n_clusters):
            hsz = int(self.in_features // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            projection = nn.Sequential(
                nn.Embedding(osz, hsz),
                nn.Linear(hsz, self.in_features, bias=False),
                nn.Dropout(self.tail_drop),
            )

            self.tail.append(projection)

    def forward(self, myinput):
        input_size = list(myinput.size())

        output = myinput.new_zeros(
            [myinput.size(0) * myinput.size(1)] + [self.in_features]
        ).float()
        myinput = myinput.view(-1)

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            input_mask = (myinput >= low_idx) & (myinput < high_idx)
            row_indices = input_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue
            out = (
                self.head(myinput[input_mask] - low_idx)
                if i == 0
                else self.tail[i - 1](myinput[input_mask] - low_idx)
            )
            output.index_copy_(0, row_indices, out)

        return output.view(input_size[0], input_size[1], -1)


_ASMoutput = namedtuple("ASMoutput", ["output", "loss"])


class AdaptiveLogSoftmaxWithLoss(Module):
    def __init__(
        self,
        in_features,
        n_classes,
        cutoffs,
        div_value=4.0,
        head_bias=False,
        tail_drop=0.5,
    ):
        super(AdaptiveLogSoftmaxWithLoss, self).__init__()

        cutoffs = list(cutoffs)

        if (
            (cutoffs != sorted(cutoffs))
            or (min(cutoffs) <= 0)
            or (max(cutoffs) > (n_classes - 1))
            or (len(set(cutoffs)) != len(cutoffs))
            or any(int(c) != c for c in cutoffs)
        ):
            raise ValueError(
                "cutoffs should be a sequence of unique, positive "
                "integers sorted in an increasing order, where "
                "each value is between 1 and n_classes-1"
            )

        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias
        self.tail_drop = tail_drop

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.head = Linear(self.in_features, self.head_size, bias=self.head_bias)
        self.tail = ModuleList()

        for i in range(self.n_clusters):
            hsz = int(self.in_features // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            projection = Sequential(
                Linear(self.in_features, hsz, bias=False),
                Linear(hsz, osz, bias=False),
                nn.Dropout(self.tail_drop),
            )

            self.tail.append(projection)

    def reset_parameters(self):
        self.head.reset_parameters()
        for i2h, h2o in self.tail:
            i2h.reset_parameters()
            h2o.reset_parameters()

    def forward(self, myinput, target):
        if myinput.size(0) != target.size(0):
            raise RuntimeError(
                "Input and target should have the same size in the batch dimension."
            )

        used_rows = 0
        batch_size = target.size(0)

        output = myinput.new_zeros(batch_size)
        gather_inds = target.new_empty(batch_size)

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            target_mask = (target >= low_idx) & (target < high_idx)
            row_indices = target_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue

            if i == 0:
                gather_inds.index_copy_(0, row_indices, target[target_mask])

            else:
                relative_target = target[target_mask] - low_idx
                input_subset = myinput.index_select(0, row_indices)

                cluster_output = self.tail[i - 1](input_subset)
                cluster_index = self.shortlist_size + i - 1

                gather_inds.index_fill_(0, row_indices, cluster_index)

                cluster_logprob = F.log_softmax(cluster_output, dim=1)
                local_logprob = cluster_logprob.gather(1, relative_target.unsqueeze(1))
                output.index_copy_(0, row_indices, local_logprob.squeeze(1))

            used_rows += row_indices.numel()

        if used_rows != batch_size:
            raise RuntimeError(
                f"Target values should be in [0, {self.n_classes - 1}], "
                "but values in range [{target.min().item()}, {target.max().item()}] "
                "were found. "
            )

        head_output = self.head(myinput)
        head_logprob = F.log_softmax(head_output, dim=1)
        output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze()
        loss = (-output).mean()

        return _ASMoutput(output, loss)

    def _get_full_log_prob(self, myinput, head_output):
        """Given input tensor, and output of `self.head`,
        compute the log of the full distribution"""
        out = myinput.new_empty((head_output.size(0), self.n_classes))
        head_logprob = F.log_softmax(head_output, dim=1)

        out[:, : self.shortlist_size] = head_logprob[:, : self.shortlist_size]

        for i, (start_idx, stop_idx) in enumerate(zip(self.cutoffs, self.cutoffs[1:])):
            cluster_output = self.tail[i](myinput)
            cluster_logprob = F.log_softmax(cluster_output, dim=1)
            output_logprob = cluster_logprob + head_logprob[
                :, self.shortlist_size + i
            ].unsqueeze(1)

            out[:, start_idx:stop_idx] = output_logprob

        return out

    def log_prob(self, myinput):
        r"""Computes log probabilities for all :math:`\texttt{n\_classes}`

        Args:
            myinput (Tensor): a minibatch of examples

        Returns:
            log-probabilities of for each class :math:`c`
            in range :math:`0 <= c <= \texttt{n\_classes}`, where :math:`\texttt{n\_classes}` is a
            parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor.

        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N, \texttt{n\_classes})`

        """
        head_output = self.head(myinput)
        return self._get_full_log_prob(myinput, head_output)

    def predict(self, myinput):
        r"""This is equivalent to `self.log_pob(myinput).argmax(dim=1)`,
        but is more efficient in some cases.

        Args:
            myinput (Tensor): a minibatch of examples

        Returns:
            output (Tensor): a class with the highest probability for each example

        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N)`
        """
        head_output = self.head(myinput)
        output = torch.argmax(head_output, dim=1)
        not_in_shortlist = output >= self.shortlist_size
        all_in_shortlist = not not_in_shortlist.any()

        if all_in_shortlist:
            return output

        elif not_in_shortlist.all():
            log_prob = self._get_full_log_prob(myinput, head_output)
            return torch.argmax(log_prob, dim=1)

        else:
            log_prob = self._get_full_log_prob(
                myinput[not_in_shortlist], head_output[not_in_shortlist]
            )
            output[not_in_shortlist] = torch.argmax(log_prob, dim=1)
            return output
