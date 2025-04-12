import json
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import os
import warnings
import torchvision.datasets as dset
from PIL import Image

warnings.filterwarnings("ignore")
from transmission_model import transmission_net
from channel_codec import channel_net
from semantic_codec_mask import semantic_net
import time
import numpy as np
import torchvision
import random
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

import torchmetrics

imagenet_mean = np.array([0.485, 0.456, 0.406])  # 这两个数组存储了ImageNet数据集的平均值和标准差，用于图像的标准化处理
imagenet_std = np.array([0.229, 0.224, 0.225])
torch.cuda.set_device(0)


class params():
    checkpoint_path = "checkpoints"
    device = "cuda"
    dataset = r"data_voc2012/semantic_feature_maps" # or "data_voc2012/original images"
    log_path = "logs"
    epoch = 100  # training epoch
    lr = 1e-4  # learning rate 1e-4
    batchsize = 128
    snr = 5
    weight_delay = 1e-6
    use_MASK = True
    save_model_name = "LAM-SE" # or "Original image"


# fix random seed
def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# show the constructed images
def show_images(pred_images, filename):
    imgs_sample = (pred_images.data + 1) / 2.0
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)


# construction of dataset
class custom_datasets(Dataset):
    def __init__(self, data):
        self.data = data.imgs
        self.img_transform = self.transform()

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        img = Image.open(self.data[item][0]).convert('RGB')
        img = self.img_transform(img)
        return img, self.data[item][0]

    def transform(self):
        compose = [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ]
        return transforms.Compose(compose)

# Train semantic codec and channel codec networks
def train_SC_net(model, train_dataloader, arg: params):
    # load model
    #weights_path = os.path.join(arg.checkpoint_path, f"{arg.save_model_name}_snr{arg.snr}.pth")
    weights_path = os.path.join(arg.checkpoint_path, f"{arg.save_model_name}_snr{arg.snr}_add_mask.pth")
    model = model.to(arg.device)

    # Load MaskNet pre-training parameters
    masknet_path = os.path.join(arg.checkpoint_path, "LAM-SE_snr5_pure_mask.pth")
    model.isc_model.Mask.load_state_dict(torch.load(masknet_path, map_location=arg.device))
    print(f"Loaded MaskNet weights from {masknet_path}")

    # Freeze MaskNet parameters
    for param in model.isc_model.Mask.parameters():
        param.requires_grad = False

    optimizer_SC = torch.optim.Adam(model.isc_model.parameters(), lr=arg.lr,
                                    weight_decay=arg.weight_delay)  # 定义了两个 Adam 优化器
    optimizer_Ch = torch.optim.Adam(model.ch_model.parameters(), lr=arg.lr,
                                    weight_decay=arg.weight_delay)

    # Use the SSIM from the torchmetrics library
    ssim_metric = torchmetrics.functional.structural_similarity_index_measure
    model.train()
    loss_record = []
    for epoch in range(arg.epoch):
        start = time.time()
        losses = []
        # training channel codec
        for i, (x, y) in enumerate(train_dataloader):
            optimizer_Ch.zero_grad()
            x = x.to(arg.device)
            c_code, c_code_, s_code, s_code_, im_decoding = model(x)

            # Calculate the cosine similarity loss
            cos_sim = F.cosine_similarity(s_code, s_code_, dim=1)
            loss_ch = 1 - cos_sim.mean()

            loss_ch.backward()
            optimizer_Ch.step()
            losses.append(loss_ch.item())
        # training semantic codec
        for i, (x, y) in enumerate(train_dataloader):
            optimizer_SC.zero_grad()
            x = x.to(arg.device)
            c_code, c_code_, s_code, s_code_, im_decoding = model(x)

            # Calculate the SSIM loss
            loss_SC = 1 - ssim_metric(im_decoding, x)

            loss_SC.backward()
            optimizer_SC.step()
            losses.append(loss_SC.item())
        losses = np.mean(losses)
        loss_record.append(losses)
        print(f"epoch {epoch} | loss: {losses} | waste time: {time.time() - start}")
        if epoch % 5 == 0:
            os.makedirs(os.path.join(arg.log_path, f"{arg.snr}"), exist_ok=True)
            # show raw and reconstructed images
            show_images(x.detach().cpu(),
                        os.path.join(arg.log_path, f"{arg.snr}", f"{arg.save_model_name}_imgs.jpg"))
            show_images(im_decoding.detach().cpu(),
                        os.path.join(arg.log_path, f"{arg.snr}", f"{arg.save_model_name}_rec_imgs.jpg"))
        #with open(os.path.join(arg.log_path, f"{arg.save_model_name}_snr{arg.snr}_loss.json"), "w",
        #          encoding="utf-8") as f:
        with open(os.path.join(arg.log_path, f"{arg.save_model_name}_snr{arg.snr}_loss_add_mask.json"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps(loss_record, indent=4, ensure_ascii=False))

        # save weights
        torch.save(model.state_dict(), weights_path)

        # Save loss visualization plot
        plt.figure(figsize=(10, 6))
        plt.plot(loss_record, label='Loss', color='b', linewidth=2)
        #plt.title('Loss vs Epoch for "LAM-SE" (SNR=5dB)')
        plt.title('Loss vs Epoch for "LAM-SE with mask" (SNR=5dB)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend(loc='upper right')
        os.makedirs(os.path.join(arg.log_path), exist_ok=True)
        #plt.savefig(os.path.join(arg.log_path, f"{arg.save_model_name}_snr{arg.snr}_loss_plot.png"))
        plt.savefig(os.path.join(arg.log_path, f"{arg.save_model_name}_snr{arg.snr}_loss_plot_add_mask.png"))
        plt.close()


# training mask network
def train_MaskNet(model, train_dataloader, arg: params):
    # laod weights
    weights_path = os.path.join(arg.checkpoint_path,
                                f"{arg.save_model_name}_snr{arg.snr}.pth")
    weights = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(weights)
    model = model.to(arg.device)
    # Train the MASKNet and frozen the semantic_net
    for param in model.parameters():
        param.requires_grad = False
    for param in model.isc_model.Mask.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.isc_model.Mask.parameters(), lr=arg.lr,
                                 weight_decay=arg.weight_delay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.1, patience=100,
                                                           verbose=True, threshold=0.0001,
                                                           threshold_mode='rel',
                                                           cooldown=0, min_lr=0,
                                                           eps=1e-08)

    ssim_metric = torchmetrics.functional.structural_similarity_index_measure
    model.train()
    loss_record = []
    for epoch in range(10):
        start = time.time()
        losses = []
        for i, (x, y) in enumerate(train_dataloader):
            model.zero_grad()
            x = x.to(arg.device)
            c_code, c_code_, s_code, s_code_, im_decoding = model(x)

            cos_sim = F.cosine_similarity(s_code, s_code_, dim=1)
            loss_ch = (1 - cos_sim.mean()) / 2

            # compute SSIM loss
            loss_SC = 1 - ssim_metric(im_decoding, x)
            loss = loss_SC + loss_ch

            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            losses.append(loss.item())
        losses = np.mean(losses)
        loss_record.append(losses)
        print(
            f"epoch {epoch} | loss: {loss.item()} | waste time: {time.time() - start}")  # 打印当前 epoch 的损失和训练时间
        if epoch % 5 == 0:
            os.makedirs(os.path.join(arg.log_path, f"{arg.snr}"), exist_ok=True)
            # show raw and reconstructed images
            show_images(x.detach().cpu(),
                        os.path.join(arg.log_path, f"{arg.snr}", f"{arg.save_model_name}_imgs.jpg"))
            show_images(im_decoding.detach().cpu(), os.path.join(arg.log_path, f"{arg.snr}",
                                                                 f"{arg.save_model_name}_rec_imgs.jpg"))
        with open(os.path.join(arg.log_path, f"{arg.save_model_name}_snr{arg.snr}_loss.json"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps(loss_record, indent=4, ensure_ascii=False))
        # save weights
        weights_path = os.path.join(arg.checkpoint_path,
                                    f"{arg.save_model_name}_snr{arg.snr}.pth")
        torch.save(model.state_dict(), weights_path)

         # Save MaskNet weights separately
        masknet_weights_path = os.path.join(arg.checkpoint_path,
                                    f"{arg.save_model_name}_snr{arg.snr}_pure_mask.pth")
        torch.save(model.isc_model.Mask.state_dict(), masknet_weights_path)

        # Save loss visualization plot
        plt.figure(figsize=(10, 6))
        plt.plot(loss_record, label='Loss', color='b', linewidth=2)
        plt.title('Loss vs Epoch for mask network (SNR=5dB)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend(loc='upper right')
        os.makedirs(os.path.join(arg.log_path), exist_ok=True)
        plt.savefig(os.path.join(arg.log_path, f"{arg.save_model_name}_snr{arg.snr}_loss_plot_mask.png"))
        plt.close()

# data transmission
@torch.no_grad()
def data_transmission(img_path, dataset="example"):
    Img_data = dset.ImageFolder(root=img_path)
    datasets = custom_datasets(Img_data)
    dataloader = DataLoader(dataset=datasets, batch_size=1, shuffle=False, num_workers=0,
                            drop_last=False)
    # load SC model
    SC_model = semantic_net(input_dim=3, MASK=True)
    channel_model = channel_net(in_dims=5408, snr=arg.snr)
    model = transmission_net(SC_model, channel_model)
    weights_path = os.path.join(arg.checkpoint_path, f"{arg.save_model_name}_snr5_add_mask.pth")
    weight = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(weight)
    model.to(arg.device)
    model.eval()
    save_dir = f"data_voc2012/rec_images/{dataset}"
    os.makedirs(save_dir, exist_ok=True)
    for i, (x, y) in enumerate(dataloader):
        x = x.to(arg.device)
        c_code, c_code_, s_code, s_code_, im_decoding = model(x)
        show_images(im_decoding.cpu(),
                    os.path.join(save_dir, f"rec_img_{i}.jpg"))
    avg_mask_percentage = SC_model.Mask.total_mask_percentage / 3 # 3 is the number of images transferred
    print(f"Average masking area ratio: {avg_mask_percentage:.2f}%")

arg = params()
if __name__ == '__main__':
    same_seeds(1024)

    Img_data = dset.ImageFolder(root=arg.dataset)
    datasets = custom_datasets(Img_data)
    train_dataloader = DataLoader(dataset=datasets, batch_size=arg.batchsize, shuffle=True, num_workers=0,
                                  drop_last=False)

    masknet_path = os.path.join(arg.checkpoint_path, "LAM-SE_snr5_pure_mask.pth")

    # training semantic codec and channel codec
    # SC_model = SCNet(input_dim=3, MASK=False)
    semantic_model = semantic_net(input_dim=3, MASK=True, masknet_path=masknet_path)
    channel_model = channel_net(in_dims=5408, snr=arg.snr)
    model = transmission_net(semantic_model, channel_model)
    train_SC_net(model, train_dataloader, arg)

    ## training MaskNet
    # semantic_model = semantic_net(input_dim=3, MASK=True)
    # channel_model = channel_net(in_dims=5408, snr=arg.snr)
    # model = base_net(semantic_model, channel_model)
    # train_MaskNet(model, train_dataloader, arg)
