import pandas as pd
import numpy as np



def read_data(path):
    df = pd.read_csv(path)

    features = list(df['comb'])
    labels = list(df['label'])

    return features, labels


def create_vocab(features=None):  # features(序列)    生成vocab
    vocab = list()
    for i in features:                #abbccdd
        vocab.extend(list(i))
    vocab = ["PAD"] + list(set(vocab)) #PADabcd
    print('vocab:', vocab)
    with open('vocab.txt', 'w', encoding='utf8') as f:
        for i in vocab:
            f.write(i + '\n')

features, labels = read_data('train.csv')
create_vocab(features)

# print('features_exmple:',features[:3])
# print('lables_exmple:',labels[:3])
# a = np.array(labels[:3]).reshape(-1,1)
# print('a:',a)
# b = labels = np.log(a) + 1
# print('b:',b)

with open('vocab.txt', 'r', encoding='utf8') as f:
    vocab = []
    for i in f.readlines():
        vocab.append(i.replace('\n', ''))
print(vocab)
word2id = dict(zip(vocab, [i for i in range(len(vocab))]))  # 分词字典
print('word2id:', word2id)
print(len(vocab))
