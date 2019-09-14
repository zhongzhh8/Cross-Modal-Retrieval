import torchvision.models as models
import torch
import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig


class ImageNet(nn.Module):
    def __init__(self, hash_length):
        super(ImageNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, hash_length)
        self.tanh=torch.nn.Tanh()

    def forward(self, x):
        resnet_feature=self.resnet(x)
        image_feature=self.tanh(resnet_feature)
        return image_feature


class TextNet(nn.Module):
    def __init__(self,  code_length):
        super(TextNet, self).__init__()

        modelConfig = BertConfig.from_pretrained('/home/disk1/zhaoyuying/models/modeling_bert/bert-base-uncased-config.json')
        self.textExtractor = BertModel.from_pretrained('/home/disk1/zhaoyuying/models/modeling_bert/bert-base-uncased-pytorch_model.bin', config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size

        self.fc = nn.Linear(embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        output=self.textExtractor(tokens, token_type_ids=segments, attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]

        hash_features = self.fc(text_embeddings)
        hash_features=self.tanh(hash_features)
        return hash_features