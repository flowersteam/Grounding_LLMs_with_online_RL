import os
import torch
import json
import collections
import copy
import numpy as np

from importlib import import_module
from pathlib import Path
from PIL import Image
from torch.nn import functional as F

# from alfred.utils import metric_util
# from alfred.gen import constants


def adjust_lr(optimizer, args, epoch, schedulers):
    '''
    adjust optimizer learning rate w.r.t the schedulers
    '''
    if epoch >= args.lr['warmup_epoch']:
        schedulers['base'].step()
    else:
        schedulers['warmup'].step()


def create_optimizer_and_schedulers(first_epoch, args, parameters, optimizer=None):
    '''
    create a scheduler for the learning rate
    '''
    # create an optimizer if it was not provided
    init_lr = args.lr['init'] * args.lr['warmup_scale']
    if args.lr['warmup_scale'] != 1:
        assert args.lr['warmup_epoch'] > 0
    if optimizer is None:
        assert args.optimizer in ('adam', 'adamw')
        OptimizerClass = torch.optim.Adam if args.optimizer == 'adam' else torch.optim.AdamW
        optimizer = OptimizerClass(parameters, lr=init_lr, weight_decay=args.weight_decay)
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = init_lr

    # create a learning rate scheduler
    assert args.lr['profile'] in ('linear', 'cosine', 'triangular', 'triangular2')
    if args.lr['profile'] == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, gamma=args.lr['decay_scale'], step_size=args.lr['decay_epoch'])
    elif args.lr['profile'] == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(args.epochs - args.lr['warmup_epoch'] - 1), eta_min=args.lr['final'])
    else:
        assert min(args.lr['cycle_epoch_up'], args.lr['cycle_epoch_down']) > 0
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=args.lr['init'], max_lr=args.lr['final'],
            step_size_up=args.lr['cycle_epoch_up'], step_size_down=args.lr['cycle_epoch_down'],
            mode=args.lr['profile'], cycle_momentum=False)

    # create a learning rate scheduler for the warmup period
    warmup_scheduler = None
    if args.lr['warmup_epoch']:
        warmup_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=(1 / args.lr['warmup_scale'] ** (1 / args.lr['warmup_epoch'])))

    # in case if we start not from the first epoch, fastforward the scheduler
    for epoch in range(first_epoch):
        if epoch >= args.lr['warmup_epoch']:
            lr_scheduler.step()
        else:
            warmup_scheduler.step()
    return optimizer, {'base': lr_scheduler, 'warmup': warmup_scheduler}


def load_model(fsave, device, check_epoch=None):
    '''
    load pth model from disk
    '''
    print('Loading from {} to {}'.format(fsave, device))
    save = torch.load(fsave, map_location=device)
    LearnedModel = import_module('alfred.model.learned').LearnedModel
    model = LearnedModel(save['args'], save['embs_ann'], save['vocab_out'])
    model.load_state_dict(save['model'])
    OptimizerClass = torch.optim.Adam if save['args'].optimizer == 'adam' else torch.optim.AdamW
    optimizer = OptimizerClass(model.parameters(), lr=1e-3, weight_decay=save['args'].weight_decay)
    optimizer.load_state_dict(save['optim'])
    if check_epoch:
        assert save['metric']['epoch'] == check_epoch, 'Epochs in info.json and latest.pth do not match'
    model = model.to(torch.device(device))
    optimizer_to(optimizer, torch.device(device))
    return model, optimizer


def load_model_args(fsave):
    '''
    load model's args from disk
    '''
    save = torch.load(fsave, map_location=lambda storage, loc: storage)
    return save['args']


def save_model(model, model_name, stats, optimizer=None, symlink=False):
    '''
    save the model to args.dout/model_name or create a symlink from the latest model to args.dout/model_name
    '''
    save_path = os.path.join(model.args.dout, model_name)
    if not symlink:
        # nn.DaraParallel related renaming
        state_dict = {key.replace('model.module.', 'model.'): value
                      for key, value in model.state_dict().items()}
        assert optimizer is not None
        torch.save({
            'metric': stats,
            'model': state_dict,
            'optim': optimizer.state_dict(),
            'args': model.args,
            'vocab_out': model.vocab_out,
            'embs_ann': model.embs_ann,
        }, save_path)
    else:
        # create symlink to last saved model
        model_path = os.path.join(
            model.args.dout, 'model_{:02d}.pth'.format(stats['epoch']))
        if os.path.islink(save_path):
            os.unlink(save_path)
        os.symlink(model_path, save_path)


def has_interaction(action):
    '''
    check if low-level action is interactive
    '''
    non_interact_actions = ['MoveAhead', 'Rotate', 'Look', '<<stop>>', '<<pad>>', '<<seg>>']
    if any(a in action for a in non_interact_actions):
        return False
    else:
        return True


def get_task_and_ann_id(ex):
    '''
    single string for task_id and annotation repeat idx
    '''
    return "%s_%s" % (ex['task_id'], str(ex['repeat_idx']))


def tensorboard(writer, metrics, split, iter, frequency, batch_size):
    if (iter // batch_size) % frequency == 0:
        for metric_name, metric_value_list in metrics.items():
            metric_value = np.mean(metric_value_list[-frequency:])
            writer.add_scalar('{}/{}'.format(split, metric_name), metric_value, iter)


def save_log(dout, progress, total, stage, **kwargs):
    '''
    logging a method json for besteffort mode and jobs monitoring on Alex's machine
    '''
    info_path = os.path.join(dout, 'info.json')
    info_dicts = []
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info_dicts = json.load(f)
    info_dict = {'stage': stage, 'progress': progress, 'total': total}
    info_dict.update(kwargs)
    info_dicts.append(info_dict)
    with open(info_path, 'w') as f:
        json.dump(info_dicts, f)


def load_log(dout, stage):
    '''
    loading a method json to continue training from the correct place
    '''
    info_path = os.path.join(dout, 'info.json')
    if os.path.exists(info_path):
        with open(info_path) as f:
            info_dicts = json.load(f)
        info_dict = [el for el in info_dicts if el['stage'] == stage][-1]
    else:
        info_dict = {'progress': 0, 'best_loss': {}, 'iters': {}}
    if isinstance(info_dict['best_loss'], dict):
        info_dict['best_loss'] = collections.defaultdict(
            lambda: 1e10, info_dict['best_loss'])
    if isinstance(info_dict['iters'], dict):
        info_dict['iters'] = collections.defaultdict(
            lambda: 0, info_dict['iters'])
    return info_dict

def load_log_choose_progress(dout, stage, epoch):
    '''
    loading a method json to continue training from the correct place
    '''
    info_path = os.path.join(dout, 'info.json')
    if os.path.exists(info_path):
        with open(info_path) as f:
            info_dicts = json.load(f)
        info_dict = [el for el in info_dicts if el['stage'] == stage][epoch]
    return info_dict

def update_log(dout, stage, update, **kwargs):
    '''
    updating a method json for monitoring on Alex's machine
    '''
    assert update in ('increase', 'rewrite')
    info_path = os.path.join(dout, 'info.json')
    assert os.path.exists(info_path)
    with open(info_path) as f:
        info_dicts = json.load(f)
    info_dict = copy.deepcopy([el for el in info_dicts if el['stage'] == stage][-1])
    # update the values
    for key, value in kwargs.items():
        assert key in info_dict
        new_value = value + info_dict[key] if update == 'increase' else value
        info_dict[key] = new_value
    # decide what to do with the list with updated values
    if info_dicts[-1]['stage'] == stage:
        # rewrite the values
        info_dicts[-1] = info_dict
    else:
        # append a new list element
        info_dicts.append(info_dict)
    # dump to the disk
    with open(info_path, 'w') as f:
        json.dump(info_dicts, f)


def test_extractor(orig_json_path, extractor, feats_orig):
    images_root = Path(orig_json_path).parents[0] / 'raw_images'
    if not images_root.is_dir():
        print('WARNING: feature extraction unit check can not be performed, directory does not exist')
        return
    first_image = Image.open(images_root / '000000000.png')
    feat_extracted = extractor.featurize([first_image], batch=1)
    assert torch.isclose(feat_extracted.mean(), feats_orig[0].mean()), \
        'feature extraction is not the same for training and evaluation'


def triangular_mask(size, device, diagonal_shift=1):
    '''
    generate upper triangular matrix filled with ones
    '''
    square = torch.triu(torch.ones(size, size, device=device), diagonal=diagonal_shift)
    square = square.masked_fill(square == 1., float('-inf'))
    return square


def generate_attention_mask(len_lang, len_frames, device, num_input_actions=0):
    '''
    generate mask for attention (a timestep at t does not attend to timesteps after t)'''
    # 1.1 language should attend only to language
    lang_to_lang = torch.zeros((len_lang, len_lang), device=device).float()
    #lang_to_rest = torch.ones((len_lang, len_frames * 2), device=device).float() * float('-inf')
    # 1.2 language should attend to all frames and all actions
    lang_to_rest = torch.zeros((len_lang, len_frames * 2), device=device).float()
    lang_to_all = torch.cat((lang_to_lang, lang_to_rest), dim=1)
    # 2.1 frames should attend to all language tokens
    frames_to_lang = torch.zeros((len_frames, len_lang), device=device).float()
    # 2.2 frames should attend to frames with timestep <= t
    frames_to_frames = triangular_mask(len_frames, device)
    # 2.3 frames should attend to actions with timestep < t. first make all actions invisible
    frames_to_actions = torch.ones((len_frames, len_frames), device=device).float() * float('-inf')
    # 2.3 then unmask `num_input_actions` previous actions for each frame (excluding index t)
    for a_idx in range(num_input_actions):
        for f_idx in range(len_frames):
            if f_idx - 1 - a_idx < 0:
                # the index is out of bound
                continue
            frames_to_actions[f_idx, f_idx - 1 - a_idx] = 0.
    frames_to_all = torch.cat((frames_to_lang, frames_to_frames, frames_to_actions), dim=1)
    # 3. actions should attend to the same indices as frames
    actions_to_all = frames_to_all.clone()
    # 4. concatenate all the masks
    all_to_all = torch.cat((lang_to_all, frames_to_all, actions_to_all), dim=0)
    return all_to_all


def process_prediction(
        action, objects, pad, vocab_action, clean_special_tokens, predict_object=True):
    '''
    process a single trajectory, return it as a dict
    '''
    # remove padding tokens
    if pad in action:
        pad_start_idx = action.index(pad)
        action = action[:pad_start_idx]
        objects = objects[:pad_start_idx]
    if clean_special_tokens:
        # remove <<stop>> tokens
        stop_token = vocab_action.word2index('<<stop>>')
        if stop_token in action:
            stop_start_idx = action.index(stop_token)
            action = action[:stop_start_idx]
            objects = objects[:stop_start_idx]
    # index to API actions
    words = vocab_action.index2word(action)

    if predict_object:
        pred_object = objects[None].max(2)[1].cpu().numpy()
    else:
        pred_object = None
    pred_processed = {
        'action': ' '.join(words),
        'object': pred_object,
    }
    return pred_processed


def extract_action_preds(
        model_out, pad, vocab_action, clean_special_tokens=True, lang_only=False):
    '''
    output processing for a VLN agent
    '''
    zipped_data = zip(model_out['action'].max(2)[1].tolist(), model_out['object'])
    predict_object = not lang_only
    preds_list = [
        process_prediction(
            action, objects, pad, vocab_action, clean_special_tokens, predict_object)
        for action, objects in zipped_data]
    return preds_list


"""def compute_f1_and_exact(metrics, preds, labels, loss_key):
    '''
    compute f1 and extract match scores for agent output
    '''
    m = collections.defaultdict(list)
    for pred_str, label_str in zip(preds, labels):
        pred_list, label_list = pred_str.lower().split(' '), label_str.lower().split(' ')
        # compute f1 score for the full sequence of actions
        m['{}/f1'.format(loss_key)].append(
            metric_util.compute_f1(label_str, pred_str))
        # compute exact matching for each timestep individually
        for pred_action, label_action in zip(pred_list, label_list):
            m['{}/exact'.format(loss_key)].append(
                metric_util.compute_exact(label_action, pred_action))
    m_averaged = {k: sum(v)/len(v) for k, v in m.items()}
    for k, v in m_averaged.items():
        metrics[k].append(v)"""


def compute_obj_class_precision(metrics, gt_dict, classes_out):
    '''
    compute precision of predictions for interaction object classes
    '''
    interact_idxs = torch.nonzero(gt_dict['action_valid_interact'])
    obj_classes_prob = classes_out[tuple(interact_idxs.T)]
    obj_classes_pred = obj_classes_prob.max(1)[1]
    obj_classes_gt = torch.cat(gt_dict['object'], dim=0)
    precision = torch.sum(
        obj_classes_pred == obj_classes_gt) / len(obj_classes_gt)
    metrics['action/object'].append(precision.item())


def obj_classes_loss(pred_obj_cls, gt_obj_cls, interact_idxs):
    '''
    Compute a cross-entropy loss for the object class predictions.
    '''
    pred_obj_cls_inter = pred_obj_cls[interact_idxs]
    # the interaction objects should be non zeros
    assert not (gt_obj_cls == 0).any()
    # compute the loss for interaction objects
    obj_cls_loss = F.cross_entropy(
        pred_obj_cls_inter, gt_obj_cls, reduction='mean')
    return obj_cls_loss


def tokens_to_lang(tokens, vocab, skip_tokens=None, join=True):
    '''
    convert tokens into human-readable words
    '''
    if skip_tokens is None:
        skip_tokens = {}
    def _tokens_to_lang(seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()
        lang = [vocab.index2word(t) for t in seq if t not in skip_tokens]
        lang = ' '.join(lang) if join else lang
        return lang
    if isinstance(tokens[0], int):
        # a list of ints is provided, only one sequence
        output = _tokens_to_lang(tokens)
    else:
        # a list of lists is provided, several sequences
        output = [_tokens_to_lang(seq) for seq in tokens]
    return output


def translate_to_vocab(tokens, vocab, vocab_translate, skip_new_tokens=False):
    '''
    translate tokens from orig vocab to translate vocab
    '''
    if vocab_translate.contains_same_content(vocab):
        return tokens
    lang_orig = tokens_to_lang(tokens, vocab, join=False)
    tokens_new = []
    for word in lang_orig:
        if skip_new_tokens and word not in vocab_translate.counts:
            word = '<<pad>>'
        tokens_new.append(vocab_translate.word2index(word))
    if not skip_new_tokens:
        lang_new = tokens_to_lang(tokens_new, vocab_translate, join=False)
        assert lang_orig == lang_new
    return tokens_new


"""def last_model_path(exp_name):
    '''
    get path of the last model in the exp
    '''
    model_path = os.path.join(constants.ET_LOGS, exp_name, 'latest.pth')
    assert os.path.islink(model_path)
    return model_path"""


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
