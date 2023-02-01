import sys
import os
# import idr_torch
import torch
import torch.distributed as dist

sys.path.append(os.getcwd())
sys.path.append('/gpfsdswork/projects/rech/imi/uez56by/code/ELLA/babyai')
sys.path.append('/gpfsdswork/projects/rech/imi/uez56by/code/ELLA/gym-minigrid')

# print(idr_torch.rank)

import babyai.utils as utils
from attrdict import AttrDict
from babyai.QA_simple import Model
from babyai.l_class import Model as Model_l

import time
from torch.nn.parallel import DistributedDataParallel as DDP



dist.init_process_group(backend='nccl',
                        init_method='env://',
                        world_size=idr_torch.size,
                        rank=idr_torch.rank)
torch.cuda.set_device(idr_torch.local_rank)
gpu = torch.device("cuda")

if idr_torch.rank==0:
    print('world_size: {}'.format(idr_torch.size))
print("rank: {}, local_rank: {}, gpu: {}".format(idr_torch.rank, idr_torch.local_rank, torch.cuda.current_device()))

def load_model(no_answer, debiased, train_env, model_QA, epoch_QA, model_qa_l=None, epoch_qa_l=None):
    # Load voc
    demo_voc = utils.get_demos_QG_voc_path('{}_agent_done'.format(train_env), train_env, None,
                                           valid=False)
    if no_answer == True:
        demo_voc = demo_voc.replace("QG_vocab.pkl", "QG_no_answer_vocab.pkl")
    if debiased == True:
        demo_voc = demo_voc.replace("vocab.pkl", "biased_vocab.pkl")
    print(demo_voc)
    vocab = utils.load_voc(demo_voc)
    # values for the model
    print(vocab['answer'])
    emb_size = len(vocab['question'])
    numb_action = 8

    attr = AttrDict()
    # TRANSFORMER settings
    # size of transformer embeddings
    attr['demb'] = 768
    # number of heads in multi-head attention
    attr['encoder_heads'] = 12
    # number of layers in transformer encoder
    attr['encoder_layers'] = 2
    # how many previous actions to use as input
    attr['num_input_actions'] = 1
    # which encoder to use for language encoder (by default no encoder)
    attr['encoder_lang'] = {
        'shared': True,
        'layers': 2,
        'pos_enc': True,
        'instr_enc': False,
    }
    # which decoder to use for the speaker model
    attr['decoder_lang'] = {
        'layers': 2,
        'heads': 12,
        'demb': 768,
        'dropout': 0.1,
        'pos_enc': True,
    }

    attr['detach_lang_emb'] = False

    # DROPOUT
    attr['dropout'] = {
        # dropout rate for language (goal + instr)
        'lang': 0.0,
        # dropout rate for Resnet feats
        'vis': 0.3,
        # dropout rate for processed lang and visual embeddings
        'emb': 0.0,
        # transformer model specific dropouts
        'transformer': {
            # dropout for transformer encoder
            'encoder': 0.1,
            # remove previous actions
            'action': 0.0,
        },
    }

    # ENCODINGS
    attr['enc'] = {
        # use positional encoding
        'pos': True,
        # use learned positional encoding
        'pos_learn': False,
        # use learned token ([WORD] or [IMG]) encoding
        'token': False,
        # dataset id learned encoding
        'dataset': False,
    }
    if no_answer:
        attr['vocab_path'] = demo_voc
        et_qa = Model(attr, emb_size, numb_action, pad=0)
        if debiased == True:
            et_qa.load_state_dict(torch.load('storage/models/{}_no_answer_biased/model_{}/et_qa_{}.pt'.format(train_env,
                                                                                                              model_QA,
                                                                                                              epoch_QA)))
            qa_l = Model_l(attr, emb_size, 0, pad=0)
            qa_l.load_state_dict(torch.load('storage/models/{}_no_answer_l_class/model_{}/et_qa_{}.pt'.format(train_env,
                                                                                                              model_qa_l,
                                                                                                              epoch_qa_l)))
            qa_l.cuda()
            qa_l.eval()
        else:
            et_qa.load_state_dict(torch.load('storage/models/{}_no_answer/model_{}/et_qa_{}.pt'.format(train_env,
                                                                                                       model_QA,
                                                                                                       epoch_QA)))
    else:
        attr['vocab_path'] = demo_voc
        et_qa = Model(attr, emb_size, numb_action, pad=0)
        if debiased == True:
            et_qa.load_state_dict(torch.load('storage/models/{}_biased/model_{}/et_qa_{}.pt'.format(train_env,
                                                                                                    model_QA,
                                                                                                    epoch_QA)))
            qa_l = Model_l(attr, emb_size, 0, pad=0)
            qa_l.load_state_dict(torch.load('storage/models/{}_l_class/model_{}/et_qa_{}.pt'.format(train_env,
                                                                                                    model_qa_l,
                                                                                                    epoch_qa_l)))
            qa_l.cuda()
            qa_l.eval()
        else:
            print('storage/models/{}/model_{}/et_qa_{}.pt'.format(train_env,
                                                                  model_QA,
                                                                  epoch_QA))
            et_qa.load_state_dict(torch.load('storage/models/{}/model_{}/et_qa_{}.pt'.format(train_env,
                                                                                             model_QA,
                                                                                             epoch_QA)))
    print('===vocab_path===')
    print(attr['vocab_path'])
    et_qa.cuda()
    et_qa.eval()

    if debiased:
        return et_qa, vocab, qa_l
    else:
        return et_qa, vocab


QA, vocab = load_model(no_answer=True,
                       debiased=False,
                       train_env='BabyAI-PutNextLocal-v0',
                       model_QA=2,
                       epoch_QA=10)

lenght_seq = 512
number_quest = 38

if idr_torch.rank == 0:
    slices = []
    for i in range(idr_torch.size):

        demo_batch = {'questions': torch.randint(0, 8, (number_quest, 9), dtype=torch.int64),
                      'answers': torch.randint(0, 8, size=(number_quest,), dtype=torch.int64),
                      'frames': torch.randint(0, 6, (number_quest, lenght_seq, 3, 7, 7), dtype=torch.uint8),
                      'length_frames': torch.ones((number_quest, 1), dtype=torch.int64)*lenght_seq,
                      'length_frames_max': torch.ones(1, dtype=torch.int64)*lenght_seq,
                      'actions': torch.randint(0, 7, (number_quest, lenght_seq), dtype=torch.int64)}

        slices.append(demo_batch)
    torch.distributed.broadcast_object_list(slices, src=idr_torch.rank)
    dict_s = {k: slices[idr_torch.rank][k].to(gpu) for k in slices[idr_torch.rank].keys() if k != 'answers'}
    with torch.no_grad():
        answer = QA.forward(vocab['question'], **dict_s)['answers'].contiguous()
    # print("answer for local process {}: {}".format(idr_torch.rank, answer.dtype))
    # print("answer size: {},  answer:{}".format(answer.shape, answer))


else:
    slices = [dict() for _ in range(idr_torch.size)]
    torch.distributed.broadcast_object_list(slices, src=0)
    dict_s = {k: slices[idr_torch.rank][k].to(gpu) for k in slices[idr_torch.rank].keys() if k != 'answers'}
    with torch.no_grad():
        answer = QA.forward(vocab['question'], **dict_s)['answers'].contiguous()
    # print("answer for local process {}: {}".format(idr_torch.rank, answer.dtype))
    # print("answer size: {},  answer:{}".format(answer.shape, answer))


# print("answer size: {},  answer:{}".format(answer.shape, answer))
tensor_list = [torch.zeros((number_quest, 10), dtype=torch.float32).cuda() for _ in range(idr_torch.size)]
dist.all_gather(tensor_list, answer)
if idr_torch.rank == 0:
    print(" ==========list==========")
    answer_pred = torch.cat(tensor_list)
    answer = torch.cat([slices[i]['answers'] for i in range(idr_torch.size)])
    print(answer_pred.shape)
    print(answer.shape)
