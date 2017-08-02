"""Data utilities."""
import torch
from torch.autograd import Variable
import operator
import json



def construct_vocab(lines, vocab_size):
    """Construct a vocabulary from tokenized lines."""
    vocab = {}
    for line in lines:
        for word in line:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
    
    word2id = {}
    id2word = {}
    word2id['<pad>'] = 0
    word2id['<unk>'] = 1
    id2word[0] = '<pad>'
    id2word[1] = '<pad>'
    
    sorted_word2id = sorted(
        vocab.items(),
        key=operator.itemgetter(1),
        reverse=True
    )

    sorted_words = [x[0] for x in sorted_word2id[:vocab_size]]

    for ind, word in enumerate(sorted_words):
        word2id[word] = ind + 2

    for ind, word in enumerate(sorted_words):
        id2word[ind + 2] = word

    return word2id, id2word






def convert_to_tensor(batch, word2ind):
    """Prepare minibatch."""
    lens = [len(line) for line in batch]
    max_len = lens[-1]
    input_lines = [
        [word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line] +
        [word2ind['<pad>']] * (max_len - len(line))
        for line in batch
    ]

    #mask = [
    #    ([1] * (l - 1)) + ([0] * (max_len - l))
    #    for l in lens
    #]

    tensor_batch = Variable(torch.LongTensor(input_lines))
    #mask = Variable(torch.FloatTensor(mask))

    return tensor_batch, max_len
