import torch
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as torchdata
import torch.utils.data as Data
from collections import namedtuple
import os
import copy
import numpy as np
from torch import autograd
import time
from torch.autograd import Function
import random
import itertools
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

from iapr_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
parser = argparse.ArgumentParser(description='BlockDrop Training')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--margin', type=float, default=12, help='margin of triplet loss')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
# parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--max_epochs', type=int, default=250, help='total epochs to run')
parser.add_argument('--hashbits', type=int, default=32)
parser.add_argument('--cv_dir', default='iapr_2')
# -----------------------------------------------------------------------------------
args = parser.parse_args()
# define the Pytorch Tensor
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def get_tokens(texts):
    tokens, segments, input_masks = [], [], []
    for text in texts: 
        tokenized_text = tokenizer.tokenize(text)
        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens.append(indexed_tokens)
        segments.append([0]*len(indexed_tokens))
        input_masks.append([1]*len(indexed_tokens))
    max_len = max([len(single) for single in tokens])
    # get padding and mask
    for j in range(len(tokens)):
        padding = [0]*(max_len-len(tokens[j]))
        tokens[j] += padding
        segments[j] += padding
        input_masks[j] += padding
    tokens = torch.tensor(tokens)
    segments = torch.tensor(segments)
    input_masks = torch.tensor(input_masks)
    return tokens.cuda(), segments.cuda(), input_masks.cuda()

def compute_result_image(dataloader, net):

    bs, clses = [], []

    time_start = time.time()
    # for batch_idx, (img_names, images, labels) in enumerate(dataloader):
    for batch_idx, (images, labels) in enumerate(dataloader):

        clses.append(labels.data.cpu())
        with torch.no_grad():
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
        
        hashCodes = net.forward(images)
        # hashCodes = net_2.forward(hashFeatures)
        hashCodes = torch.tanh(hashCodes)
        bs.append(hashCodes.data.cpu())

    total_time = time.time() - time_start

    return torch.sign(torch.cat(bs)), torch.cat(clses), total_time
    # return torch.cat(img_name), torch.cat(img),  torch.sign(torch.cat(bs)), torch.cat(clses), total_time



def triplet_loss(Ihash, labels, margin):
    triplet_loss = torch.tensor(0.0).cuda()
    labels_ = labels.cpu().data.numpy()
    triplets = []
    for label in labels_:
        label_mask = np.matmul(labels_, np.transpose(label)) > 0  # multi-labels
        label_indices = np.where(label_mask)[0]
        if len(label_indices) < 2:
            continue
        negative_indices = np.where(np.logical_not(label_mask))[0]
        if len(negative_indices) < 1:
            continue
        anchor_positives = list(itertools.combinations(label_indices, 2))
        temp = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                for neg_ind in negative_indices]
        triplets += temp

    length = len(triplets)
    
    if triplets:
        triplets = np.array(triplets)
        # print('triplet', triplets.shape)
        # intra triplet loss
        I_ap = (Ihash[triplets[:, 0]] - Ihash[triplets[:, 1]]).pow(2).sum(1)
        I_an = (Ihash[triplets[:, 0]] - Ihash[triplets[:, 2]]).pow(2).sum(1)
        # print('I_ap = ', I_ap)
        # print('I_an = ', I_an)
        triplet_loss = F.relu(margin + I_ap - I_an).mean()

    return triplet_loss, length

def CrossModel_triplet_loss(imgae_Ihash, text_Ihash, labels, margin):
    triplet_loss = torch.tensor(0.0).cuda()
    labels_ = labels.cpu().data.numpy()
    triplets = []
    for label in labels_:
        label_mask = np.matmul(labels_, np.transpose(label)) > 0  # multi-labels
        label_indices = np.where(label_mask)[0]
        if len(label_indices) < 2:
            continue
        negative_indices = np.where(np.logical_not(label_mask))[0]
        if len(negative_indices) < 1:
            continue
        anchor_positives = list(itertools.combinations(label_indices, 2))
        temp = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                for neg_ind in negative_indices]
        triplets += temp

    length = len(triplets)
    
    if triplets:
        triplets = np.array(triplets)
        # print('triplet', triplets.shape)

        # intra triplet loss
        # image
        imgae_I_ap = (imgae_Ihash[triplets[:, 0]] - imgae_Ihash[triplets[:, 1]]).pow(2).sum(1)
        imgae_I_an = (imgae_Ihash[triplets[:, 0]] - imgae_Ihash[triplets[:, 2]]).pow(2).sum(1)
        imgae_triplet_loss = F.relu(margin + imgae_I_ap - imgae_I_an).mean()
        # text
        text_I_ap = (text_Ihash[triplets[:, 0]] - text_Ihash[triplets[:, 1]]).pow(2).sum(1)
        text_I_an = (text_Ihash[triplets[:, 0]] - text_Ihash[triplets[:, 2]]).pow(2).sum(1)
        text_triplet_loss = F.relu(margin + text_I_ap - text_I_an).mean()

        # cross model triplet loss
        # anchor-image    negative,positive-text
        imgae_text_I_ap = (imgae_Ihash[triplets[:, 0]] - text_Ihash[triplets[:, 1]]).pow(2).sum(1)
        imgae_text_I_an = (imgae_Ihash[triplets[:, 0]] - text_Ihash[triplets[:, 2]]).pow(2).sum(1)
        imgae_text_triplet_loss = F.relu(margin + imgae_text_I_ap - imgae_text_I_an).mean()
        # anchor-text     negative,positive-image
        text_image_I_ap = (text_Ihash[triplets[:, 0]] - imgae_Ihash[triplets[:, 1]]).pow(2).sum(1)
        text_image_I_an = (text_Ihash[triplets[:, 0]] - imgae_Ihash[triplets[:, 2]]).pow(2).sum(1)
        text_image_triplet_loss = F.relu(margin + text_image_I_ap - text_image_I_an).mean()


    return imgae_triplet_loss, text_triplet_loss, imgae_text_triplet_loss, text_image_triplet_loss, length



def train(epoch):

    imageNet.train()
    textExtractor.train()
    textHashNet.train()

    accum_loss = 0
    
    for batch_idx, (images, texts, labels) in enumerate(train_loader):

        # image
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        image_hashCodes = imageNet.forward(images)
        image_hashCodes = torch.tanh(image_hashCodes)

        # text
        tokens, segments, input_masks = get_tokens(texts)
        output = textExtractor(tokens, token_type_ids=segments, attention_mask=input_masks)
        text_embeddings = output[0][:,0,:]
        text_hashCodes = textHashNet.forward(text_embeddings)
        text_hashCodes = torch.tanh(text_hashCodes)

        imgae_triplet_loss, text_triplet_loss, \
        imgae_text_triplet_loss, text_image_triplet_loss, \
        len_triplets = CrossModel_triplet_loss(image_hashCodes, text_hashCodes, labels, args.margin)

        loss = imgae_triplet_loss + text_triplet_loss + imgae_text_triplet_loss + text_image_triplet_loss


        if len_triplets > 0:
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            accum_loss += loss.data.item()

    # -------------
    print("epoch: %d, accum_loss: %.6f " % (epoch, accum_loss))
    # s =  'epoch = ' + str(epoch) + ',  accum_loss = ' + str(accum_loss)
    # torch.save(s, args.cv_dir+'/'+str(epoch)+'.txt')

def compute_result_CrossModel(dataloader, imageNet, textExtractor, textHashNet):
    bs_image, bs_text, clses = [], [], []

    time_start = time.time()

    for batch_idx, (images, texts, labels) in enumerate(dataloader):

        clses.append(labels.data.cpu())
        
        # image
        with torch.no_grad():
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
        image_hashCodes = imageNet.forward(images)
        # image_hashCodes = torch.tanh(image_hashCodes)

        # text
        tokens, segments, input_masks = get_tokens(texts)
        output = textExtractor(tokens, token_type_ids=segments, attention_mask=input_masks)
        text_embeddings = output[0][:,0,:]
        text_hashCodes = textHashNet.forward(text_embeddings)
        # text_hashCodes = torch.tanh(text_hashCodes)
        
        bs_image.append(image_hashCodes.data.cpu())
        bs_text.append(text_hashCodes.data.cpu())

        

    total_time = time.time() - time_start

    return torch.sign(torch.cat(bs_image)), torch.sign(torch.cat(bs_text)), torch.cat(clses), total_time


def compute_mAP_MultiLabels(trn_binary, tst_binary, trn_label, tst_label):
    """
    compute mAP by searching testset from trainset
    https://github.com/flyingpot/pytorch_deephash
    """
    for x in trn_binary, tst_binary, trn_label, tst_label: x.long()

    AP = []
    Ns = torch.arange(1, trn_binary.size(0) + 1)
    Ns = Ns.type(torch.FloatTensor)
    # cnt = 0
    # total = 0.0
    # print('Ns = ', Ns)
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        # print('query_binary = ', query_binary)
        # print('trn_binary = ', trn_binary)
        # 计算汉明距离，并将距离从小到大排序(query_result是索引)
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        # print('query_result = ', query_result)
        # correct = (query_label == trn_label[query_result]).float()
        # 与 query label 相同的
        correct = ((trn_label[query_result]*query_label).sum(1) > 0).float()
        # print('correct = ', correct)
        P = torch.cumsum(correct, dim=0) / Ns
        # print('P = ', P)
        AP.append(torch.sum(P * correct) / torch.sum(correct))
        
        # if query_label[0] == 1:
        #     print('tst'+str(i)+' AP = ', torch.sum(P * correct) / torch.sum(correct))
            # cnt += 1
            # total += torch.sum(P * correct) / torch.sum(correct)
        # print('append to AP = ', torch.sum(P * correct) / torch.sum(correct))
        # print(torch.sum(correct))
        # print(trn_binary.size(0))
    # print('AP = ', AP)
    mAP = torch.mean(torch.Tensor(AP))
    # print('cnt = ', cnt)
    # print('total = ', total)
    # print('mAP = ', mAP)
    return mAP

def test(epoch):

    imageNet.eval()
    textExtractor.eval()
    textHashNet.eval()
   
    tst_image_binary, tst_text_binary, tst_label, tst_time = compute_result_CrossModel(test_loader, imageNet, textExtractor, textHashNet)
  
    db_image_binary, db_text_binary, db_label, db_time = compute_result_CrossModel(db_loader, imageNet, textExtractor, textHashNet)
   
    # print('test_codes_time = %.6f, db_codes_time = %.6f'%(tst_time ,db_time))

    it_mAP = compute_mAP_MultiLabels(db_text_binary, tst_image_binary, db_label, tst_label)
    ti_mAP = compute_mAP_MultiLabels(db_image_binary, tst_text_binary, db_label, tst_label)
    print("epoch: %d, retrieval it_mAP: %.6f, retrieval ti_mAP: %.6f" %(epoch, it_mAP, ti_mAP))
    # logger.add_scalar('retrieval_mAP', mAP, epoch)

    f = open('result/' + args.cv_dir + 'mAP.txt', 'a') 
    f.write('Epoch:'+str(epoch)+':  it_mAP = '+str(it_mAP)+', ti_mAP = '+str(ti_mAP)+'\n')
    f.close()
    
    
    torch.save(imageNet.state_dict(), args.cv_dir+'/ckpt_E_%d_it_mAP_%.5f_ti_mAP_%.5f_imageNet.t7'%(epoch, it_mAP, ti_mAP))
    torch.save(textExtractor.state_dict(), args.cv_dir+'/ckpt_E_%d_it_mAP_%.5f_ti_mAP_%.5f_textExtractor.t7'%(epoch, it_mAP, ti_mAP))
    torch.save(textHashNet.state_dict(), args.cv_dir+'/ckpt_E_%d_it_mAP_%.5f_ti_mAP_%.5f_textHashNet.t7'%(epoch, it_mAP, ti_mAP))
    




class TextHashNet(nn.Module):
    def __init__(self, input_dim, code_length):
        super(TextHashNet, self).__init__()
        self.fc = nn.Linear(input_dim, code_length)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        hash_features = self.fc(x)

        return hash_features

start_epoch = 0
total_tst_time = 0
test_cnt = 0
loss_print = 0
MODEL_UPDATE_ITER = 0

train_loader, test_loader, db_loader = IAPR_dataloader(args)
# image net
imageNet = models.resnet18(pretrained=True)
imageNet.fc = nn.Linear(512, args.hashbits)
torch.nn.init.xavier_uniform_(imageNet.fc.weight)
imageNet.cuda()
# text net
tokenizer = BertTokenizer.from_pretrained('../models/tokenization_bert/bert-base-uncased-vocab.txt')

modelConfig = BertConfig.from_pretrained('../models/modeling_bert/bert-base-uncased-config.json')
textExtractor = BertModel.from_pretrained('../models/modeling_bert/bert-base-uncased-pytorch_model.bin', config=modelConfig)
textHashNet = TextHashNet(input_dim=textExtractor.config.hidden_size, code_length=args.hashbits)
textExtractor.cuda()
textHashNet.cuda()

# optimizer_image = optim.Adam(imageNet.parameters(), lr=args.image_lr, weight_decay=args.weight_decay)
# optimizer_text = optim.Adam(list(textExtractor.parameters())+list(textHashNet.parameters()), lr=args.text_lr, weight_decay=args.weight_decay)
optimizer = optim.Adam(list(imageNet.parameters())+list(textExtractor.parameters())+list(textHashNet.parameters()), lr=args.lr, weight_decay=args.weight_decay)



# train 1
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):

    # lr_scheduler_image.adjust_learning_rate(epoch)
    # net = train(epoch, net)
    train(epoch)
    if epoch % 10 == 0:
        test(epoch)
# epoch = 0
# train(epoch)





# 55556  tmux0 GPU2