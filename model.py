import torchvision.models as models
import torch
import torch.nn as nn





class ImageNet(nn.Module):
    """Constructs a (ResNet-18+Hashing ) model.
    """
    def __init__(self, hash_length):
        super(ImageNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, hash_length)
        self.tanh=torch.nn.Tanh()
        # torch.nn.init.xavier_uniform_(resnet.fc.weight)
        # torch.nn.init.constant_(resnet.fc.bias, 0.0)

    def forward(self, x):
        resnet_feature=self.resnet(x)
        image_feature=self.tanh(resnet_feature)
        return image_feature


class TextHashNet(nn.Module):
    def __init__(self, input_dim, code_length):
        super(TextHashNet, self).__init__()
        self.fc = nn.Linear(input_dim, code_length)
        self.tanh = torch.nn.Tanh()
        # torch.nn.init.xavier_uniform_(self.fc.weight)
        # torch.nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        hash_features = self.fc(x)
        hash_features=self.tanh(hash_features)
        return hash_features