# data_url : https://www.kaggle.com/c/carvana-image-masking-challenge/data
import torch 
import numpy as np 
from SETR.transformer_seg import SETRModel
from PIL import Image
import glob 
import torch.nn as nn 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

img_url = sorted(glob.glob("./segmentation_car/imgs/*"))
mask_url = sorted(glob.glob("./segmentation_car/masks/*"))
# print(img_url)
train_size = int(len(img_url) * 0.8)
train_img_url = img_url[:train_size]
train_mask_url = mask_url[:train_size]
val_img_url = img_url[train_size:]
val_mask_url = mask_url[train_size:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is " + str(device))
epoches = 100
out_channels = 1

def build_model():
    model = SETRModel(img_size=(16, 16), 
                    in_channels=3, 
                    out_channels=1, 
                    hidden_size=1024, 
                    num_hidden_layers=6, 
                    num_attention_heads=16, 
                    decode_features=[512, 256, 128, 64])
    return model

class CarDataset(Dataset):
    def __init__(self, img_url, mask_url):
        super(CarDataset, self).__init__()
        self.img_url = img_url
        self.mask_url = mask_url

    def __getitem__(self, idx):
        img = Image.open(self.img_url[idx])
        img = img.resize((256, 256))
        img_array = np.array(img, dtype=np.float32) / 255
        mask = Image.open(self.mask_url[idx])
        mask = mask.resize((256, 256))
        mask = np.array(mask, dtype=np.float32)
        img_array = img_array.transpose(2, 0, 1)
    
        return torch.tensor(img_array.copy()), torch.tensor(mask.copy())

    def __len__(self):
        return len(self.img_url)

def compute_dice(input, target):
    eps = 0.0001
    # input 是经过了sigmoid 之后的输出。
    input = (input > 0.5).float()
    target = (target > 0.5).float()

    # inter = torch.dot(input.view(-1), target.view(-1)) + eps
    inter = torch.sum(target.view(-1) * input.view(-1)) + eps

    # print(self.inter)
    union = torch.sum(input) + torch.sum(target) + eps

    t = (2 * inter.float()) / union.float()
    return t

def predict():
    model = build_model()
    model.load_state_dict(torch.load("./checkpoints/SETR_car.pkl", map_location="cpu"))
    print(model)

    import matplotlib.pyplot as plt
    val_dataset = CarDataset(val_img_url, val_mask_url)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for img, mask in val_loader:
            pred = torch.sigmoid(model(img))
            pred = (pred > 0.5).int()
            plt.subplot(1, 3, 1)
            print(img.shape)
            img = img.permute(0, 2, 3, 1)
            plt.imshow(img[0])
            plt.subplot(1, 3, 2)
            plt.imshow(pred[0].squeeze(0), cmap="gray")
            plt.subplot(1, 3, 3)
            plt.imshow(mask[0], cmap="gray")
            plt.show()

if __name__ == "__main__":

    model = build_model()
    model.to(device)

    train_dataset = CarDataset(train_img_url, train_mask_url)
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)

    val_dataset = CarDataset(val_img_url, val_mask_url)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    step = 0
    report_loss = 0.0
    for epoch in range(epoches):
        print("epoch is " + str(epoch))

        for img, mask in tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()
            step += 1
            img = img.to(device)
            mask = mask.to(device)

            pred_img = model(img) ## pred_img (batch, len, channel, W, H)
            if out_channels == 1:
                pred_img = pred_img.squeeze(1) # 去掉通道维度

            loss = loss_func(pred_img, mask)
            report_loss += loss.item()
            loss.backward()
            optimizer.step()

            if step % 1000 == 0:
                dice = 0.0
                n = 0
                model.eval()
                with torch.no_grad():
                    print("report_loss is " + str(report_loss))
                    report_loss = 0.0
                    for val_img, val_mask in tqdm(val_loader, total=len(val_loader)):
                        n += 1
                        val_img = val_img.to(device)
                        val_mask = val_mask.to(device)
                        pred_img = torch.sigmoid(model(val_img))
                        if out_channels == 1:
                            pred_img = pred_img.squeeze(1)
                        cur_dice = compute_dice(pred_img, val_mask)
                        dice += cur_dice
                    dice = dice / n
                    print("mean dice is " + str(dice))
                    torch.save(model.state_dict(), "./checkpoints/SETR_car.pkl")
                    model.train()
