import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# definition of semantic codec
class semantic_net(nn.Module):
    def __init__(self, input_dim=3, MASK=False, masknet_path=None):
        super(semantic_net, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 128, kernel_size=5,
                               bias=False)
        self.pool = nn.MaxPool2d((2, 2),
                                 return_indices=True)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=5, bias=False)
        self.use_MASK = MASK
        self.Mask = MaskNet(32)

        if masknet_path:
            self.Mask.load_state_dict(torch.load(masknet_path, map_location="cpu"))
            print(f"Loaded MaskNet weights from {masknet_path}")

        self.convt1 = nn.ConvTranspose2d(32, 128, kernel_size=5)
        self.convt2 = nn.ConvTranspose2d(128, input_dim, kernel_size=5)
        self.uppool = nn.MaxUnpool2d(2, 2)


    def forward(self, x=None, latent=None):
        if latent == None:
            x = F.leaky_relu(self.conv1(x))
            x, self.indices1 = self.pool(x)
            x = F.leaky_relu(self.conv2(x))
            x, self.indices2 = self.pool(x)
            self.x_shape = x.shape
            if self.use_MASK:
                x = self.Mask(x)
            latent = x.view(x.size(0), -1)

            return latent
        else:
            x = latent.view(self.x_shape)
            x = self.uppool(x, self.indices2)
            x = F.leaky_relu(self.convt1(x))
            x = self.uppool(x, self.indices1)
            x = F.tanh(self.convt2(x))
            return x


# definition of the mask network
class MaskNet(nn.Module):
    def __init__(self, input_dim=32):
        super(MaskNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, input_dim, kernel_size=3, padding=1)
        self.total_mask_percentage = 0
    def forward(self, x):
        y = self.conv1(x)
        y = F.relu(y)
        mask = self.conv2(y) + torch.abs(x)
        mask = torch.sign(mask)
        mask = F.relu(mask)
        x = torch.mul(x, mask)
        # Calculate the percentage of the masked area in the image
        total_elements = x.numel()
        masked_elements = (mask == 0).sum().item()
        mask_percentage = masked_elements / total_elements * 100
        print(f"Proportion of masking area:: {mask_percentage:.2f}%")
        self.total_mask_percentage +=mask_percentage
        return x


if __name__ == '__main__':
    # Test the Semantic_net model
     net = semantic_net()
     net.to("cuda")
     summary(net,(3,64,64),device="cuda")
    ## Test the MaskNet model
    # net = MaskNet()
    # net.to("cuda")
    # summary(net, (32, 64, 64), device="cuda")

