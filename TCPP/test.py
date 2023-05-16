import numpy as np
import pandas as pd
import pandas as pd
import datetime
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
import torch
import torch.utils.data as Data
# from logger import ProtLogger


class Config(object):
    '''
    配置参数
    '''

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.train_path = 'data/train.csv'
        self.test_path = 'data/test/result_ck115_iter100_seed0_jiauniport.csv'
        self.vocab_path = 'data/vocab.txt'
        self.train_batch_size = 256
        self.test_batch_size = 100

        self.load_checkpoint = 'checkpoint/checkpoint8.tar'   #checkpoint文件下的所有 ck18


def read_data(path):
    df = pd.read_csv(path)

    features = list(df['seq'])
    labels = list(df['lable'])
    return features, labels

def create_feature(features, labels, config):

    with open(config.vocab_path, 'r', encoding='utf8') as f:
        vocab = []
        for i in f.readlines():
            vocab.append(i.replace('\n', ''))

    word2id = dict(zip(vocab, [i for i in range(len(vocab))]))  # 分词字典word2id: {'PAD': 0, 'Q': 1, 'S': 2, 'C': 3, 'N': 4, 'F': 5, 'R': 6, 'D': 7, 'I': 8, 'V': 9, 'H': 10, 'E': 11, 'M': 12, 'A': 13, 'L': 14, 'P': 15, 'G': 16, 'W': 17, 'K': 18, 'T': 19, 'Y': 20}

    features_id = []
    for i in features:
        arr = [word2id[j] for j in i] #把序列按照字典编成数字； arr:[2,3,4,5,7,9............]
        if len(arr) < 20:
            arr += [0] * (20 - len(arr))  # padding成29位
        features_id.append(arr)
    features_id = np.array(features_id)
    features_id = features_id[:, :15]  # 每个序列取前15位,  size: [bz,15]

    # labels = np.array(labels).reshape(-1, 1)  # 变维为n行，1列。 size是二维的！
    # labels = np.log(labels) + 1

    # return torch.from_numpy(features_id), torch.from_numpy(np.array(labels))
    return torch.LongTensor(features_id), torch.LongTensor(labels)




##============================Transformer=====================================
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)  #embed等于d_model
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout, config.device)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(config.num_encoder)])

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), 2)



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


if __name__ == '__main__':
    config = Config()
    # 准备数据集
    features, labels = read_data(config.train_path)
    test_features, test_labels = read_data(config.test_path)

    # 序列处理成数字
    features_id, labels = create_feature(features, labels,config)
    test_features_id, test_labels = create_feature(test_features, test_labels,config)

    # 利用DataLoader 来加载数据
    train_dataset = Data.TensorDataset(features_id, labels)
    test_dataset = Data.TensorDataset(test_features_id, test_labels)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=1)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('GPU is available!')

    ## test
    predict_probs = []
    predict_lables = []
    predict_logits=[]
    batch_xs = []
    # model.eval()
    model = torch.load(config.load_checkpoint)
    model.eval()
    for batch_x, batch_y in test_loader:
        pred = model(batch_x.to(device))     #[bz,2] 2个logit
        pred_softmax=F.softmax(pred,dim=1)   #[bz,2] 2个概率

#torch.max(input,dim=1)反回两个值，第一个是最大概率，第二个是最大概率的索引
        pred_prob = torch.max(pred_softmax.data, 1)[0] #prob 最大概率
        pred_logit = torch.max(pred.data, 1)[0]     #logit 最大logit
        pred_lable=torch.max(pred.data, 1)[1]        #lable 最大概率的索引
        #
        predict_probs += pred_prob.cpu().tolist()
        predict_logits += pred_logit.cpu().tolist()
        predict_lables += pred_lable.cpu().tolist()
        batch_xs+=batch_x.cpu().tolist()
        # a = 0
    print(predict_probs)
    # print(predict_logits)
    # print(predict_lables)
    #
    # f1 = open('data/L4_ass/test/result_ck95_iter36_seed0.txt','r')
    # xulie = f1.readlines()
    df_test = pd.read_csv(config.test_path)


    # print(len(predict_probs))
    # print(len(predict_lables))
    # print(len(xulie))


    df = pd.DataFrame(columns=['seq','prob','logit','lable'])
    for i in range(len(batch_xs)):
        df.loc[i]={'seq':df_test['seq'][i],'prob':predict_probs[i],'logit':predict_logits[i],'lable':predict_lables[i]}

    df.to_csv('test_result/all_result/all_result.csv',index=False) #ck18

    df_result_analyze = pd.DataFrame(columns=['seq', 'prob', 'logit', 'lable'])

    for i in range(len(df['seq'])):
        if str(df['lable'][i]) == str(1):

            df_result_analyze.loc[i] = {'seq': df['seq'][i], 'prob': df['prob'][i],'logit': df['logit'][i], 'lable': df['lable'][i]}
    df_result_analyze.to_csv('test_result/pos_result/pos_result.csv',index=False)







