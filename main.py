from SETR.transformer_seg import SETRModel
from SETR.transformer_model import TransConfig
import torch 

if __name__ == "__main__":
    config = TransConfig(img_size=(32, 32), in_channels=3, out_channels=1, hidden_size=1024, num_hidden_layers=8, num_attention_heads=16)

    t1 = torch.rand(1, 3, 256, 256)
    print("input: " + str(t1.shape))
    net = SETRModel(config)
    # print(net)
    print("output: " + str(net(t1).shape))