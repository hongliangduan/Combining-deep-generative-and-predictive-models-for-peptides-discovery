from model import CVAE
from utils import *
import numpy as np
import os
import tensorflow as tf

import time
import argparse
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help='batch_size', type=int, default=128) #
parser.add_argument('--num_iteration', help='num_iteration', default=100)
parser.add_argument('--latent_size', help='latent_size', type=int, default=200)
parser.add_argument('--unit_size', help='unit_size of rnn cell', type=int, default=512)
parser.add_argument('--n_rnn_layer', help='number of rnn layer', type=int, default=3)
parser.add_argument('--seq_length', help='max_seq_length', type=int, default=15)  ##
parser.add_argument('--mean', help='mean of VAE', type=float, default=0.0)
parser.add_argument('--stddev', help='stddev of VAE', type=float, default=1.0)
parser.add_argument('--num_prop', help='number of propertoes', type=int, default=1) ##
# parser.add_argument('--save_file', help='save file', type=str,default='save/peptide/lable_only-1/model_293.ckpt-293') ##
parser.add_argument('--save_file', help='save file', type=str,default='save/model_115.ckpt-115')##ckpt
parser.add_argument('--target_prop', help='target properties', type=str,default='1') ##  1
parser.add_argument('--prop_file', help='name of property file', type=str, default='data/L4_aas.txt') ## data
parser.add_argument('--result_filename', help='name of result filename', type=str, default='result/random_ck115_iter100_try.txt')  ##result
parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)
args = parser.parse_args()

#convert smiles to numpy array
_, _, char, vocab, _, _ = load_data(args.prop_file, args.seq_length)
vocab_size = len(char)

#model and restore model parapmeters
model = CVAE(vocab_size,
             args
             )
model.restore(args.save_file)

print ('Number of parameters : ', np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))

#target property to numpy array
target_prop = np.array([[float(p) for p in args.target_prop.split()] for _ in range(args.batch_size)])
start_codon = np.array([np.array(list(map(vocab.get, 'X')))for _ in range(args.batch_size)])

#generate smiles
smiles = []
for _ in range(args.num_iteration):
    np.random.seed(_)
    latent_vector = s = np.random.normal(args.mean, args.stddev, (args.batch_size, args.latent_size))
    generated = model.sample(latent_vector, target_prop, start_codon, args.seq_length)

    smiles += [convert_to_smiles(generated[i], char) for i in range(len(generated))]

##wzp新添加
print ('number of trial : ', len(smiles))
smiles = list(set([s.split('E')[0] for s in smiles]    ))
with open(args.result_filename, 'w') as w:
    for s in smiles:
        if s:
            w.write(s+'\n')
#########去掉rdkit测smiles性质的代码部分

#write smiles and calcualte properties of them    

# print ('number of trial : ', len(smiles))
# smiles = list(set([s.split('E')[0] for s in smiles]    ))
# print ('number of generate smiles (after remove duplicated ones) : ', len(smiles))
# ms = [Chem.MolFromSmiles(s) for s in smiles]
# ms = [m for m in ms if m is not None]
# print ('number of valid smiles : ', len(ms))
# with open(args.result_filename, 'w') as w:
#     w.write('smiles\tMW\tLogP\tTPSA\n')
#     for m in ms:
#         try:
#             w.write('%s\t%.3f\t%.3f\t%.3f\n' %(Chem.MolToSmiles(m), ExactMolWt(m), MolLogP(m), CalcTPSA(m)))
#         except:
#             continue.
