from torchsummary import summary
import torch
from torch import nn
from semantic_codec_mask import semantic_net
from channel_codec import channel_net


# merge semantic model and channel model as a transmission model
class transmission_net(nn.Module):
    def __init__(self, sc_model, channel_model):
        super(transmission_net, self).__init__()
        self.sc_model = sc_model  # semantic codec model
        self.ch_model = channel_model  # channel codec model

    def forward(self, x):
        s_code = self.isc_model(x)
        c_code, c_code_, s_code_ = self.ch_model(s_code)
        im_decoding = self.isc_model(latent=s_code_)
        # return raw and recovered channel code, raw and recovered semantic code, and the reconstructed image array
        return c_code, c_code_, s_code, s_code_, im_decoding


if __name__ == '__main__':
    SC_model = semantic_net()
    channel_model = channel_net(in_dims=5408, snr=25)

    model = transmission_net(SC_model, channel_model).to("cuda")
    summary(model, (3, 64, 64), device="cuda")

