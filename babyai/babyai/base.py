from torch import nn

# from alfred.utils import data_util


class Model(nn.Module):
    def __init__(self, args, emb_ann_size, numb_action, pad):
        '''
        Abstract model
        '''
        nn.Module.__init__(self)
        self.args = args
        self.numb_action = numb_action
        self.pad = pad
        # shape manually given TO IMPROVE as in ET
        # self.visual_tensor_shape = data_util.read_dataset_info(
        #   args.data['train'][0])['feat_shape'][1:]
        self.visual_tensor_shape = [128, 2, 2]
        # self.visual_tensor_shape = [512, 7, 7]
        # create language and action embeddings

        self.emb_ann = nn.Embedding(emb_ann_size, args.demb)

        # dropouts
        self.dropout_vis = nn.Dropout(args.dropout['vis'], inplace=True)
        self.dropout_lang = nn.Dropout2d(args.dropout['lang'])

    def init_weights(self, init_range=0.1):
        '''
        init linear layers in embeddings
        '''
        self.emb_ann.weight.data.uniform_(-init_range, init_range)

    def compute_metrics(self, model_out, gt_dict, metrics_dict, verbose):
        '''
        compute model-specific metrics and put it to metrics dict
        '''
        raise NotImplementedError

    def forward(self, vocab, **inputs):
        '''
        forward the model for multiple time-steps (used for training)
        '''
        raise NotImplementedError()

    def compute_batch_loss(self, model_out, gt_dict):
        '''
        compute the loss function for a single batch
        '''
        raise NotImplementedError()

    def compute_loss(self, model_outs, gt_dicts):
        '''
        compute the loss function for several batches
        '''
        # compute losses for each batch
        losses = {}
        for dataset_key in model_outs.keys():
            losses[dataset_key] = self.compute_batch_loss(
                model_outs[dataset_key], gt_dicts[dataset_key])
        return losses

    def compute_batch_DOE(self, model_out, gt_dict):
        '''
        compute the DOE for a single batch
        '''
        raise NotImplementedError()

    def compute_DOE(self, model_outs):
        '''
        compute the DOE for several batches
        '''
        # compute losses for each batch
        DOE= {}
        for dataset_key in model_outs.keys():
            DOE[dataset_key] = self.compute_batch_DOE(model_outs[dataset_key])
        return DOE
