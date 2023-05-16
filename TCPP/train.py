import pandas as pd
import datetime
from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib as plt
# from logger import ProtLogger
import copy
import os



class Config(object):
    '''
    配置参数
    '''
    def __init__(self,args):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_path = 'data/train.csv'
        self.test_path = 'data/test.csv'
        self.vocab_path = 'data/vocab.txt'
        self.train_batch_size = 1024
        self.test_batch_size = 1024

        self.num_epochs = 50  # epoch数

        self.learning_rate = 5e-4  # 学习率
        self.log_path = 'log_dir/L4_aas_jiahuoxing_lenth25_padding_09_20cu_qchuoxinghao_lable10'    #tensorboard path
        self.checkpoint = 'L4_aas_jiahuoxing_lenth25_padding_09_20cu_qchuoxinghao_jiauniport_lable10'     #checkpoint/的文件名
        self.logger = 'L4_aas_jiahuoxing_lenth25_padding_09_20cu_qchuoxinghao_jiauniport_lable10.txt'

        #模型参数
        self.dropout = 0.5  # 随机失活
        self.pad_size = 15  # 每句话处理成的长度(短填长切)
        self.n_vocab = 25  # 词表大小

        self.embed = 300

        self.dim_model = 300     #默认300
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 2         #默认5
        self.num_encoder = 2      #默认2

        self.num_filters=256
        self.filter_sizes=(2, 3, 4)




def read_data(path):
    df = pd.read_csv(path)

    features = list(df['seq'])
    labels = list(df['lable'])

    return features, labels

def create_feature(features, labels, config):  #序列===>数字
    #vocab==>vocab词典
    with open(config.vocab_path, 'r', encoding='utf8') as f:
        vocab = []
        for i in f.readlines():
            vocab.append(i.replace('\n', ''))
    word2id = dict(zip(vocab, [i for i in range(len(vocab))]))  # 分词字典word2id: {'PAD': 0, 'Q': 1, 'S': 2, 'C': 3, 'N': 4, 'F': 5, 'R': 6, 'D': 7, 'I': 8, 'V': 9, 'H': 10, 'E': 11, 'M': 12, 'A': 13, 'L': 14, 'P': 15, 'G': 16, 'W': 17, 'K': 18, 'T': 19, 'Y': 20}

    print('word2id:', word2id)
    features_id = []
    for i in features:
        arr = [word2id[j] for j in i]  # arr:[2,3,4,5,7,9............]
        if len(arr) < 20:
            arr += [0] * (20 - len(arr))  # padding成29位
        features_id.append(arr)
    features_id = np.array(features_id)
    print('fea:',type(features_id))
    print('fea:',features_id.shape)
    features_id = features_id[:, :15]  # 每个序列取15位,  size: [bz,15]

    return torch.LongTensor(features_id), torch.LongTensor(labels)


##============================Transformer=====================================
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # if config.embedding_pretrained is not None:
        #     self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        # else:
        #     self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=0)  #embed等于d_model
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout, config.device)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(config.num_encoder)])

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes]) #1 256 （k,embading=300）;k qu (2,3,4)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), 2)

        # self.fc1 = nn.Linear(config.pad_size * config.dim_model, 1) # TODO change this
        # self.fc1 = nn.Linear(config.pad_size * config.dim_model, 2)  # TODO change this erfenlei
        # self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        # out = self.embedding(x[0])
        out = self.postion_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        ##
        out = out.unsqueeze(dim=1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

        # out = out.view(out.size(0), -1)     #3维度变为2维。  第一维是out.size(0)
        # out = self.fc1(out)
        # return out  #logit           [b,class]


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)]) #pos*embed的矩阵
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out
##===========================================================================


# ##_________________________________________________________________________________
if __name__ == '__main__':
    config = Config()

    # 准备数据集
    features, labels = read_data(config.train_path)
    test_features, test_labels = read_data(config.test_path)
    # create_vocab(features)

    # 序列处理成数字
    features_id, labels = create_feature(features, labels, config)
    test_features_id, test_labels = create_feature(test_features, test_labels, config)

    # 利用DataLoader 来加载数据
    train_dataset=Data.TensorDataset(features_id,labels)
    test_dataset=Data.TensorDataset(test_features_id,test_labels)
    train_loader=Data.DataLoader(dataset=train_dataset,batch_size=config.train_batch_size,shuffle=True,num_workers=1)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=config.test_batch_size,shuffle=False, num_workers=1)
    # a =1

    if config.device:
        print('GPU is available!')

    model = Model(config).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    ## train
    epochs = config.num_epochs

    # 添加tensorboard
    writer = SummaryWriter(logdir=config.log_path) #  自动生成tensorboard文件存放地址
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print('参数量：',total_params)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    ##logger
    logger_path = 'logger/'+config.logger
    f1 = open(logger_path,'a')

    total_batch = 0  # 记录进行到多少batch
    for epoch in range(epochs):
        print('------第 [{}/{}]轮训练开始------'.format(epoch + 1, config.num_epochs))
        # print("------第 {} 轮训练开始------".format(i + 1))
        # 训练步骤开始
        model.train()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            # print()
            # model.train()
            ouput = model(batch_x.to(config.device))
            optimizer.zero_grad()
            loss = F.cross_entropy(ouput, batch_y.to(config.device))
            loss.backward()
            optimizer.step()

        #train_acc
        true = batch_y.data.cpu()
        #torch.max此函数输出两个tensor，第一个tensor是每行的最大概率(或者logit)，第二个tensor是每行最大概率的索引

        predict = torch.max(ouput.data,1)[1].cpu()  #output = torch.max(input, dim),1行；在计算准确率时第一个tensor values是不需要的，所以我们只需提取第二个tensor 索引
        train_acc = metrics.accuracy_score(true, predict)

        ## test
        predict_f = []
        true_f = []
        model.eval()

        loss_total = 0
        for x, y in test_loader:
            pred = model(x.to(config.device))
            dev_loss = F.cross_entropy(pred,y.to(config.device))
            # dev_loss = criterion(pred.float(), y.to(config.device).float())
            loss_total += dev_loss.item()
            pred = torch.max(pred.data,1)[1]
            # pred_s = np.array(pred.cpu().detach()).flatten().tolist()
            # true_s = np.array(batch_y.detach()).flatten().tolist()

            predict_f +=pred.cpu()# bz个pred
            true_f +=y.cpu()
        dev_acc = metrics.accuracy_score(true_f, predict_f)

        dev_loss = loss_total / len(test_loader)

        # print("train time：{}, Loss: {}, lr:{}".format(train_step, loss.item(),optimizer.param_groups[0]['lr']))
        print("epoch：{0:}, train_Loss: {1:.4f}, train_acc: {2:.4f}, dev_loss: {3:.4f}, dev_acc: {4:.4f}".format(epoch+1, loss.item(), train_acc,dev_loss, dev_acc))

        writer.add_scalar("train_step_loss", loss.item(), total_batch)
        writer.add_scalar("train_acc", train_acc, total_batch)
        writer.add_scalar("dev_loss", dev_loss, total_batch)
        writer.add_scalar("dev_acc", dev_acc, total_batch)
        # writer.add_scalar("dev_recall", n, train_step)
        # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], train_step)

        f1.write('epoch:'+str(epoch+1)+
                 ', train_step_loss:'+str(loss.item())+
                 ', train_acc:'+str(train_acc)+
                 ', dev_loss:'+str(dev_loss)+
                 ', dev_acc:'+str(dev_acc)+'\n')


        checkpoint = config.checkpoint
        if not os.path.exists('checkpoint/'+checkpoint): #如果checkpoint/下没有checkpoint
            os.mkdir('checkpoint/'+checkpoint)             #创建

        torch.save(model,'checkpoint/'+checkpoint+'/checkpoint'+str(epoch+1)+'.tar')

    print('success save model')
    writer.close()
