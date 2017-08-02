import os
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from src import models
import argparse
import time
import math
from src.data_utils import *
import pickle

parser = argparse.ArgumentParser(description='lm.py')

parser.add_argument('-save_model', default='lm',
                    help="""Model filename to save""")
parser.add_argument('-load_model', default='',
                    help="""Model filename to load""")
parser.add_argument('-train', default='/tera/JD_content.txt',
                    help="""Text filename for training""")
parser.add_argument('-valid', default='data/valid.txt',
                    help="""Text filename for validation""")
parser.add_argument('-rnn_type', default='mlstm',
                    help='Number of layers in the encoder/decoder')
parser.add_argument('-layers', type=int, default=1,
                    help='Number of layers in the encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=4096,
                    help='Size of hidden states')
parser.add_argument('-embed_size', type=int, default=256,
                    help='Size of embeddings')
parser.add_argument('-seq_length', type=int, default=15,
                    help="Maximum sequence length")
parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-learning_rate', type=float, default=0.5,
                    help="""Starting learning rate.""")
parser.add_argument('-dropout', type=float, default=0.1,
                    help='Dropout probability.')
parser.add_argument('-param_init', type=float, default=0.05,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-clip', type=float, default=5,
                    help="""Clip gradients at this value.""")
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')
# GPU
parser.add_argument('-cuda', action='store_true', default='True',
                    help="Use CUDA")


parser.add_argument('-ns', action='store_true',
                    help="Load open AI mlstm weights from numpy files")

opt = parser.parse_args()

path = opt.train
torch.manual_seed(opt.seed)
if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

# load data
# with open(path) as f:
#    train_data = f.readlines()
# remove data contains less than 5 words and sort by length
# train_data = [line.replace('\n', '') for
#                   line in train_data if len(line) >= 6]
# train_data.sort(key = len)
# word2id, id2word = construct_vocab(train_data,100000)

# pickle.dump(train_data, open('train_data.p', 'wb'))
# pickle.dump(word2id, open('word2id.p', 'wb'))
# pickle.dump(id2word, open('id2word.p', 'wb'))
train_data = pickle.load(open('dat/train_data.p', 'rb'))
word2id = pickle.load(open('dat/word2id.p', 'rb'))
id2word = pickle.load(open('dat/id2word.p', 'rb'))

batch_size = opt.batch_size
hidden_size = opt.rnn_size
input_size = opt.embed_size
output_size = len(id2word)
TIMESTEPS = opt.seq_length
vocab_size = len(word2id)
learning_rate = opt.learning_rate

if len(opt.load_model) > 0:
    checkpoint = torch.load(opt.load_model)
    embed = checkpoint['embed']
    rnn = checkpoint['rnn']
else:
    embed = nn.Embedding(vocab_size, opt.embed_size)
    if opt.rnn_type == 'gru':
        rnn = models.StackedLSTM(nn.GRUCell,
                                 opt.layers, input_size, hidden_size,
                                 output_size, opt.dropout)
    elif opt.rnn_type == 'mlstm':
        rnn = models.StackedLSTM(models.mLSTM,
                                 opt.layers, input_size, hidden_size,
                                 output_size, opt.dropout)
    else:
        # default to lstm
        rnn = models.StackedLSTM(nn.LSTMCell,
                                 opt.layers, input_size, hidden_size,
                                 output_size, opt.dropout)

loss_fn = nn.CrossEntropyLoss()

nParams = sum([p.nelement() for p in rnn.parameters()])
print('* number of parameters: %d' % nParams)
n_batch = len(train_data)//batch_size

print(n_batch)
embed_optimizer = optim.SGD(embed.parameters(), lr=learning_rate)
rnn_optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)


def clip_gradient_coeff(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, clip / (totalnorm + 1e-6))


def calc_grad_norm(model):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    return math.sqrt(totalnorm)


def calc_grad_norms(model):
    """Computes a gradient clipping coefficient based on gradient norm."""
    norms = []
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        norms += [modulenorm]
    return norms


def update_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr
    return


def clip_gradient(model, clip):
    """Clip the gradient."""
    totalnorm = 0
    for p in model.parameters():
        p.grad.data = p.grad.data.clamp(-clip, clip)


def make_cuda(state):
    if isinstance(state, tuple):
        return (state[0].cuda(), state[1].cuda())
    else:
        return state.cuda()


def copy_state(state):
    if isinstance(state, tuple):
        return (Variable(state[0].data), Variable(state[1].data))
    else:
        return Variable(state.data)


def evaluate():
    rnn = nn.DataParallel(rnn, output_device=[1])
    embed = nn.DataParallel(embed, output_device=[1])
    hidden_init = rnn.state0(opt.batch_size)
    if opt.cuda:
            embed.cuda()
            rnn.cuda()
            hidden_init = make_cuda(hidden_init)

    loss_avg = 0
    for s in range(nv_batch-1):
        batch = Variable(valid.narrow(0, s*TIMESTEPS, TIMESTEPS+1).long())
        start = time.time()
        hidden = hidden_init
        if opt.cuda:
            batch = batch.cuda()

        loss = 0
        for t in range(TIMESTEPS):
            emb = embed(batch[t])
            hidden, output = rnn(emb, hidden)
            loss += loss_fn(output, batch[t+1])

        hidden_init = copy_state(hidden)
        loss_avg = loss_avg + loss.data[0]/TIMESTEPS
        if s % 10 == 0:
            print('v %s / %s loss %.4f loss avg %.4f time %.4f' %
                  (s, nv_batch, loss.data[0]/TIMESTEPS, loss_avg/(s+1),
                   time.time()-start))
    return loss_avg/nv_batch


def train_epoch(epoch, rnn, embed):
        # hidden_init = rnn.state0(opt.batch_size)
        if opt.cuda:
            embed.cuda()
            rnn.cuda()
            # hidden_init = make_cuda(hidden_init)

        epoch_loss = 0

        for s in range(n_batch-1):
            embed_optimizer.zero_grad()
            rnn_optimizer.zero_grad()
            batch = train_data[s*opt.batch_size:(s+1)*opt.batch_size]
            tensor_batch, max_len = convert_to_tensor(batch, word2id)
            start = time.time()
            hidden = rnn.state0(len(batch))
            hidden = make_cuda(hidden)
            if opt.cuda:
                tensor_batch = tensor_batch.cuda()

            n_step = max_len//TIMESTEPS + 1
            batch_loss = 0
            for step in range(n_step):
                step_base = step*TIMESTEPS
                try:
                    curr_batch = tensor_batch[:,
                                              step_base:step_base + TIMESTEPS]
                except ValueError:
                    print(step_base, curr_batch.size(), TIMESTEPS)
                    continue
                loss = 0
                step_lengths = curr_batch.size()[0]
                for t in range(step_lengths):
                    if step_base + t + 1 < max_len:
                        emb = embed(curr_batch[:, t])
                        hidden, output = rnn(emb, hidden)
                        loss += loss_fn(output,
                                        tensor_batch[:, step_base + t + 1])

                loss.backward()

                gn = calc_grad_norm(rnn)
                clip_gradient(rnn, opt.clip)
                clip_gradient(embed, opt.clip)
                embed_optimizer.step()
                rnn_optimizer.step()
                batch_loss += loss.data[0]/step_lengths
            epoch_loss += batch_loss/n_step
            if s % 100 == 0:
                print('e%s %s / %sloss %.4floss avg %.4f time %.4f grad_norm %.4f' %
                      (epoch, s, n_batch,
                       loss.data[0]/step_lengths,
                       epoch_loss/(s+1), time.time()-start, gn))


for e in range(10):
        try:
                train_epoch(e, rnn, embed)
        except KeyboardInterrupt:
                print('Exiting from training early')
        # loss_avg = evaluate()
        checkpoint = {
            'rnn': rnn,
            'embed': embed,
            'opt': opt,
            'epoch': e
        }
        save_file = ('%s_e %s.pt' % (opt.save_model, e))
        print('Saving to '+ save_file)
        torch.save(checkpoint, save_file)
        learning_rate *= 0.7
        update_lr(rnn_optimizer, learning_rate)
        update_lr(embed_optimizer, learning_rate)
