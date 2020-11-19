import torch
import torch.nn as nn
import torch.nn.functional as F


class LocationBlock(nn.Module):
    def __init__(self, out_channels=128, kernel_size=31):
        super().__init__()
        self.conv1d = nn.Conv1d(2, 32, kernel_size=kernel_size, padding=kernel_size//2)
        self.linear = nn.Linear(32, out_channels)

    def forward(self, x):
        """
        x: [B, 2, T]
        return: [B, T, 128]
        """
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.linear(x)
        return x


class PreNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_channels, out_channels),
            nn.Linear(out_channels, out_channels),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, .5, self.train)
        return x


class PostNet(nn.Module):
    def __init__(self, mel_channels):
        super().__init__()
        conv_layers = []
        mapping = [mel_channels, 512, 512, 512, 512, mel_channels]
        for idx in range(len(mapping) - 1):
            block = nn.Sequential(nn.Conv1d(mapping[idx], mapping[idx + 1], kernel_size=5, padding=5//2),
                                  nn.BatchNorm1d(mapping[idx + 1]))
            conv_layers.append(block)
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            x = F.tanh(x)
            x = F.dropout(x, .5, self.train)
        return x


class Decoder(nn.Module):
    def __init__(self, mel_channels, mel_scale):
        super().__init__()
        self.m_scale = mel_scale
        self.mel_channels = mel_channels

        self.prenet = PreNet(mel_channels * self.mel_scale, 256)

        self.att_query_rnn = nn.LSTMCell(256 + 512, 1024)
        self.to_query = nn.Linear(1024, 128)

        self.memory_key = nn.Linear(512, 128)

        self.location_block = LocationBlock(128)

        self.attention = nn.Linear(128, 1)

        self.dec_rnn = nn.LSTMCell(512 + 1024, 1024)

        self.to_mel = nn.Linear(1024 + 512, self.mel_channels * self.mel_scale)
        self.to_gate = nn.Linear(1024 + 512, 1)

        self.post_net = PostNet(self.mel_channels)

    def forward(self, encoder_out, mels, text_mask, mel_mask):
        """
        args:
            encoder_out: [B, T, C]
            mels: [B, T, C]
            mask: [B, 1, T]
        -----
        return:
            mels_out: [B, T, C]
            gates_out: [B, T, 1]
            attentions_out: [B, T, T]
        """
        batch_size = encoder_out.size(0)
        time_size = encoder_out.size(1)
        mels_out, gates_out, attentions_out = [], [], []

        att_context = torch.zeros(batch_size, 512)

        att_hidden = torch.zeros(batch_size, 1024)
        att_cell = torch.zeros(batch_size, 1024)

        att_weight = torch.zeros(batch_size, 1, time_size)
        att_weight_total = torch.zeros(batch_size, 1, time_size)

        dec_hidden = torch.zeros(batch_size, 1024)
        dec_cell = torch.zeros(batch_size, 1024)

        # [B, T, C] -> [B, T/mel_scale, mel_scale * C]
        mels = mels.view(mels.size(0), -1, self.mel_channels * self.mel_scale)

        # [B, T, C] -> [T, B, C]
        mel_init = torch.zeros(batch_size, 1, self.mel_channels * self.mel_scale)
        mels = torch.cat([mel_init, mels], 1).transpose(0, 1)
        # [T, B, n_mel_dim] -> [T, B, 128]
        mels = self.prenet(mels)

        # ------------------------------------------------------------------------------
        # [B, T, 512] -> [B, T, 128]
        memory_keys = self.memory_key(encoder_out)
        # ------------------------------------------------------------------------------

        # [T, B, C] -> T * [B, C]
        for mel in mels:
            # [B, 256 + 512] -> [B, 1024]
            x = torch.cat([mel, att_context], -1)
            att_hidden, att_cell = self.att_query_rnn(x, [att_hidden, att_cell])
            att_hidden = F.dropout(att_hidden, .5, self.train)
            # [B, 1024] -> [B, 1, 128]
            query = self.to_query(att_hidden.unsqueeze(1))

            # ------------------------------------------------------------------------------
            att_weight_ = torch.cat([att_weight, att_weight_total], 1)
            # [B, 2, T] -> [B, T, 128]
            location = self.location_block(att_weight_)

            # ------------------------------------------------------------------------------
            # [B, 1, 128] + [B, T, 128] + [B, T, 128]
            attention_total = F.tanh(query + memory_keys + location)
            # [B, T, 128] -> [B, T, 1] -> [B, 1, T]
            attention_total = self.attention(attention_total).transpose(1, 2)

            # ------------------------------------------------------------------------------
            # [B, 1, T]
            att_weight = torch.masked_fill(attention_total, text_mask, -float('Inf'))
            att_weight = F.softmax(att_weight, -1)
            att_weight_total += att_weight

            # ------------------------------------------------------------------------------
            # [B, 1, T] * [B, T, 512] -> [B, 1, 512] -> [B, 512]
            att_context = torch.bmm(att_weight, encoder_out).squeeze(1)

            # ------------------------------------------------------------------------------
            # [B, 512 + 1024] -> [B, 1024]
            att_context_hidden = torch.cat([att_context, att_hidden], -1)
            dec_hidden, dec_cell = self.dec_rnn(att_context_hidden, [dec_hidden, dec_cell])
            dec_hidden = F.dropout(dec_hidden, .5, self.train)

            # [B, 1024 + 512] -> [B, n_mel_dim], [B, 1]
            dec_hidden = torch.cat([dec_hidden, att_context], -1)
            mel_out = self.to_mel(dec_hidden)
            gate_out = self.to_gate(dec_hidden)

            # MEL: T * [B, C], GATE: T * [B, 1], ATT: T * [B, T]
            mels_out.append(mel_out)
            gates_out.append(gate_out)
            attentions_out.append(att_weight.squeeze(1))

        # T * [B, C] -> [T, B, C] -> [B, T, C]
        mels_out = torch.stack(mels_out).transpose(0, 1)
        attentions_out = torch.stack(attentions_out).transpose(0, 1)
        gates_out = torch.stack(gates_out).transpose(0, 1)

        # [B, T, C] -> [B, T, self.mel_channels]
        mels_out = mels_out.view(mels_out.size(0), -1, self.mel_channels)
        mels_out += self.post_net(mels_out)

        mels_out = mels_out.masked_fill(mel_mask, 0.0)
        attentions_out = attentions_out.masked_fill(mel_mask, 0.0)
        gates_out = gates_out.masked_fill(mel_mask, 1e3)

        return mels_out, gates_out, attentions_out
