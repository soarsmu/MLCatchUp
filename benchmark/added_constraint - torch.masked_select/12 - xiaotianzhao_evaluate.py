#!/usr/bin/python
#-*- coding:utf8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
from util import get_minibatch

def evaluate_model(
    model,
    test_word_seqs,
    test_tag_seqs,
    word_lang,
    tag_lang,
    config
):
    model.eval()
    # predict = Variable(torch.LongTensor([]))
    # reference = Variable(torch.LongTensor([]))
    right_count = 0.
    total_item_count = 0.

    batch_size = config['data']['batch_size']
    use_cuda = config['training']['use_cuda']
    for j in xrange(0,len(test_word_seqs),len(test_word_seqs)):
        input_lines,input_mask = get_minibatch(
            test_word_seqs,
            word_lang,
            j,
            batch_size,
            max_len=config['data']['max_length'],
            add_start=False,
            add_end=False,
            use_cuda=use_cuda
        )

        output_lines,output_mask = get_minibatch(
            test_tag_seqs,
            tag_lang,
            j,
            batch_size,
            max_len=config['data']['max_length'],
            add_start=False,
            add_end=False,
            use_cuda=use_cuda
        )

        tag_logit = model(input_lines)

        tag_logit = torch.max(tag_logit,dim=2)[-1]

        # print(tag_logit.size())
        # target_file = open('test.pos','w')
        for i in xrange(batch_size):
            # print('Batch %d'%i)
            reference = torch.masked_select(output_lines[i],output_mask[i].ge(0.5))
            predict = torch.masked_select(tag_logit[i],output_mask[i].ge(0.5))
            right_count += predict.eq(reference).sum().data[0]
            total_item_count += reference.size(0)
            
            # reference = [tag_lang.id2word[z] for z in reference.cpu().data]
            # target_file.write(' '.join(reference)+'\n')
        # print(tag_logit)
        # print(output_lines)
        # predict = torch.masked_select(tag_logit,output_mask.ge(0.5))
        # reference = torch.masked_select(output_lines,output_mask.ge(0.5))
        # precision += predict.eq(reference).sum().data[0]
        # total_item_count += predict.size(0)
    # print(right_count,total_item_count)
    return right_count/total_item_count

# def caculate_precision(predict,reference):
#     """Caculate precision

#     Args:
#         predict:a list of predict tag sequence
#         reference:a list of regference tag sequence

#     Return:
#         precision number
#     """
#     if len(predict) != len(reference):
#         return -1.

#     total_count = 0
#     right_count = 0
#     for i in xrange(len(predict)):
#         if len(predict[i]) == len(reference[i]):
#             for j in xrange(predict[i]):
#                 total_count += 0

#                 if predict[i][j] == reference[i][j]:
#                     right_count += 0
#                 else:
#                     print('predict sequence and reference sequence is unequal')
    
#     return 1.*right_count / total_count            