import torch.optim as optim
from torch.autograd import Variable
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
import argparse
from iapr_utils import *
from utils import *
from model import ImageNet,TextNet

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
use_cuda = torch.cuda.is_available()

def get_args():
    parser = argparse.ArgumentParser(description='BlockDrop Training')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--image_lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--text_lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--margin', type=float, default=12, help='margin of triplet loss')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    # parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--max_epochs', type=int, default=500, help='total epochs to run')
    parser.add_argument('--hashbits', type=int, default=32)
    parser.add_argument('--cv_dir', default='checkpoints')
    # -----------------------------------------------------------------------------------
    args = parser.parse_args()

    return args


def train(args, epoch):
    imageNet.train()
    textNet.train()

    accum_loss = 0
    for batch_idx, (images, texts, labels) in enumerate(train_loader):
        # image
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        image_hashCodes = imageNet.forward(images)
        # text
        tokens, segments, input_masks = get_tokens(texts,tokenizer)
        text_hashCodes = textNet.forward(tokens, segments, input_masks)
        #计算triplet loss
        imgae_triplet_loss, text_triplet_loss, \
        imgae_text_triplet_loss, text_image_triplet_loss, \
        len_triplets = CrossModel_triplet_loss(image_hashCodes, text_hashCodes, labels, args.margin)

        loss = imgae_triplet_loss + text_triplet_loss + imgae_text_triplet_loss + text_image_triplet_loss

        if len_triplets > 0:
            # 计算网络的梯度，先更新imageNet部分，此时梯度已经用掉了。然后再backward计算一次梯度，然后更新textNet
            # optimizer_image.zero_grad()
            # loss.backward(retain_graph=True)  # 保留计算图, 接下面的backward，不然他直接就把图释放了
            # optimizer_image.step()
            #
            # optimizer_text.zero_grad()
            # loss.backward()
            # optimizer_text.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accum_loss += loss.data.item()

    print("epoch: %d, accum_loss: %.6f " % (epoch, accum_loss))
    # s =  'epoch = ' + str(epoch) + ',  accum_loss = ' + str(accum_loss)
    # torch.save(s, args.cv_dir+'/'+str(epoch)+'.txt')


def test(args,epoch):
    imageNet.eval()
    textNet.eval()

    tst_image_binary, tst_text_binary, tst_label, tst_time = compute_result_CrossModel(test_loader, imageNet, textNet,tokenizer)
    db_image_binary, db_text_binary, db_label, db_time = compute_result_CrossModel(db_loader, imageNet,  textNet,tokenizer)
    # print('test_codes_time = %.6f, db_codes_time = %.6f'%(tst_time ,db_time))

    it_mAP = compute_mAP_MultiLabels(db_text_binary, tst_image_binary, db_label, tst_label)
    ti_mAP = compute_mAP_MultiLabels(db_image_binary, tst_text_binary, db_label, tst_label)
    print("epoch: %d, retrieval it_mAP: %.6f, retrieval ti_mAP: %.6f" %(epoch, it_mAP, ti_mAP))

    f = open('result/' + args.cv_dir + 'mAP.txt', 'a')
    f.write('Epoch:'+str(epoch)+':  it_mAP = '+str(it_mAP)+', ti_mAP = '+str(ti_mAP)+'\n')
    f.close()

    if epoch%50 == 0:
        torch.save(imageNet.state_dict(), args.cv_dir+'/ckpt_E%d_it_mAP_%.5f_ti_mAP_%.5f_imageNet.t7'%(epoch, it_mAP, ti_mAP))
        torch.save(textNet.state_dict(), args.cv_dir+'/ckpt_E%d_it_mAP_%.5f_ti_mAP_%.5f_textHashNet.t7'%(epoch, it_mAP, ti_mAP))

if __name__ == '__main__':
    args=get_args()
    start_epoch = 0
    total_tst_time = 0
    test_cnt = 0
    loss_print = 0
    MODEL_UPDATE_ITER = 0

    train_loader, test_loader, db_loader = IAPR_dataloader(args)
    # image net
    imageNet=ImageNet(args.hashbits)
    imageNet.cuda()
    # text net
    tokenizer = BertTokenizer.from_pretrained('/home/disk1/zhaoyuying/models/tokenization_bert/bert-base-uncased-vocab.txt')
    textNet = TextNet(code_length=args.hashbits)
    textNet.cuda()

    # optimizer_image = optim.Adam(imageNet.parameters(), lr=args.image_lr, weight_decay=args.weight_decay)
    # optimizer_text = optim.Adam(list(textExtractor.parameters())+list(textHashNet.parameters()), lr=args.text_lr, weight_decay=args.weight_decay)
    optimizer = optim.Adam(list(imageNet.parameters())+list(textNet.parameters()), lr=args.lr, weight_decay=args.weight_decay)  #+list(textExtractor.parameters())

    for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
        train(args,epoch)
        if epoch % 10 == 0:
            test(args,epoch)


