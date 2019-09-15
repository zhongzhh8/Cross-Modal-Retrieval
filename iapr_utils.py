import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as torchdata
import os
import re


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(torchdata.Dataset):
    def __init__(self, args,txt,transform=None, loader=default_loader):
        self.transform = transform
        self.loader = loader

        name_label = []
        for line in open(txt):
            line = line.strip('\n').split()
            label = list(map(int, np.array(line[len(line)-255:]))) #后255个二进制码是label的，前2912个是单词在词袋中的二进制编码
            tem = re.split('[/.]', line[0])
            file_name, sample_name = tem[0], tem[1]
            name_label.append([file_name, sample_name, label])
            # # print('label = ', label)
            # print('file_name = %s,  sample_name = %s' %(file_name, sample_name))
            # label_list = np.where(label=='1')
            # print('label_list = ', label_list)
        self.name_label = name_label
        self.image_dir=args.image_dir
        self.text_dir = args.text_dir


    def __getitem__(self, index):
        words = self.name_label[index]  # words = [file_name, sample_name, label]
        # print('words = ', words[0:2])

        img_path = os.path.join(self.image_dir, words[0], words[1]+'.jpg')
        text_path = os.path.join(self.text_dir, words[0], words[1]+'.txt')
        # img
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        # text
        text = 'None'
        for line in open(text_path):
            text = '[CLS]' + line + '[SEP]'
            
        # label
        label = torch.LongTensor(words[2])

        return img, text, label
        

    def __len__(self):
        return len(self.name_label)


def IAPR_dataloader(args):

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


    root = args.root_dir
    train_file = os.path.join(root, 'iapr_train')
    test_file = os.path.join(root, 'iapr_test')
    retrieval_file = os.path.join(root, 'iapr_retrieval')

    train_set = MyDataset(args,txt=train_file, transform=transform_train)
    train_loader = torchdata.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_set = MyDataset(args,txt=test_file, transform=transform_test)
    test_loader = torchdata.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    db_set = MyDataset(args,txt=retrieval_file, transform=transform_test)
    db_loader = torchdata.DataLoader(db_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, db_loader



# root = '/home/disk1/zhaoyuying/dataset/iapr-tc12_255labels'
# train_file = os.path.join(root, 'iapr_train')
# test_file = os.path.join(root, 'iapr_test')
# retrieval_file = os.path.join(root, 'iapr_retrieval')
#
# mean = (0.4914, 0.4822, 0.4465)
# std = (0.2023, 0.1994, 0.2010)
# transform_train = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=mean, std=std)
# ])
# train_set = MyDataset(txt=train_file, transform=transform_train)
# train_loader = torchdata.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
#
# iii = 0
# for batch_idx, (images, texts, labels) in enumerate(train_loader):
#     # print('batch_idx = ', batch_idx)
#     iii += 1


# # words =  ['32', '32289']
# # words =  ['30', '30583']
# # words =  ['22', '22116']
# # words =  ['12', '12660']
# words =  ['03', '3580']

# text_path = os.path.join('../dataset/iapr-tc12_255labels/annotations', words[0], words[1]+'.txt')

# tokenizer = BertTokenizer.from_pretrained('../models/tokenization_bert/bert-base-uncased-vocab.txt')

# # text
# tokenized_text = None
# for line in open(text_path):
#     text = "[CLS]" + line + "[SEP]"
#     tokenized_text = tokenizer.tokenize(text)

# print('len(tokenized_text) = ', len(tokenized_text))