from __future__ import print_function
import torch
import torch.nn as nn


# AWGN channel simulation function
def AWGN_channel(x, snr):
    print(f"Current SNR: {snr} dB")
    [batch_size, length] = x.shape
    x_power = torch.sum(torch.abs(x)) / (batch_size * length)
    n_power = x_power / (10 ** (snr / 10.0))
    noise = torch.rand(batch_size, length, device="cuda") * n_power
    return x + noise

# definition of channel codec
class channel_net(nn.Module):
    def __init__(self, in_dims=800, mid_dims=128, snr=25):
        super(channel_net, self).__init__()
        self.enc_fc = nn.Linear(in_dims, mid_dims)
        self.dec_fc = nn.Linear(mid_dims, in_dims)
        self.snr = snr
        self.total_transferred_size = 0

    def forward(self, x):
        ch_code = self.enc_fc(x)
        ch_code_with_n = AWGN_channel(ch_code, self.snr)
        x = self.dec_fc(ch_code_with_n)

        return ch_code, ch_code_with_n, x