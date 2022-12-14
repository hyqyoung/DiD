# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import ml_collections
# from skimage import measure
from .obj_loc import obj_loc
import cv2

# import model_config as configs

# logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])
        if config.split == 'non-overlap':
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        elif config.split == 'overlap':
            n_patches = ((img_size[0] - patch_size[0]) // config.slide_step + 1) * ((img_size[1] - patch_size[1]) // config.slide_step + 1)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                        out_channels=config.hidden_size,
                                        kernel_size=patch_size,
                                        stride=(config.slide_step, config.slide_step))
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # self.cls_token2 = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # self.cls_token3 = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens1 = self.cls_token1.expand(B, -1, -1)
        # cls_tokens2 = self.cls_token2.expand(B, -1, -1)
        # cls_tokens3 = self.cls_token3.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens1, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    def forward(self, x):
        length = len(x)
        last_map = x[0]
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)
        last_map = last_map[:,:,0,1:]

        _, max_inx = last_map.max(2)
        return _, max_inx

def feats_pooling(x, method='avg', sh=8, sw=8):
    if method == 'avg':
       x = F.avg_pool2d(x, (sh, sw))
    if method == 'max':
       x = F.max_pool2d(x, (sh, sw))
    if method == 'gwp':
       x1 = F.max_pool2d(x, (sh, sw))
       x2 = F.avg_pool2d(x, (sh, sw))
       x = (x1 + x2)/2
    return x

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))
        self.part_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        attn_weights = []
        for layer in self.layer:
            hidden_states, weights = layer(hidden_states)
            attn_weights.append(weights)           
        part_encoded = self.part_norm(hidden_states)  

        return attn_weights, part_encoded#[:, 0]#, part_encoded[:, 1]#,part_encoded[:, 2]

class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        attn_weights, part_encoded = self.encoder(embedding_output)
        return attn_weights, part_encoded

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, smoothing_value=0, zero_head=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.smoothing_value = smoothing_value
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size)
        # self.part_head = Linear(config.hidden_size, num_classes)
        self.convclass = nn.Conv2d(config.hidden_size, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.num_classes = num_classes

    def forward(self, x, device):
        
        attn_weights1, part_encoded = self.transformer(x)
        # return attn_weights1
        b,c,h,w = x.size()
        # print(part_encoded.shape)
        ga = part_encoded[:, 1:]
        h_w, in_features = ga.shape[1], ga.shape[2]
        h_,w_ = int(math.sqrt(h_w)), int(math.sqrt(h_w))
        ga = ga.transpose(1,2).contiguous().view(b, in_features, h_, w_)
        ga = self.convclass(ga)             #b x c x h_ x w_
        gs = feats_pooling(ga, method='avg', sh=int(h/16), sw=int(w/16))                
        gs = gs.view(gs.size(0), -1)        #bxc

        feature_map = ga.detach()
        # cams = feature_map
        cams = get_cams(attn_weights1, feature_map, device)
        

        topN = 4
        linputs = torch.zeros([b, topN, 3, h, w]).cuda()
        for i in range(b):
            gs_inv, gs_ind = gs[i].sort(descending=True)
            # gs_inv, gs_ind = gs[i].sort(descending=False)

            # perm = torch.randperm(gs[i].size(0))
            # gs_inv = gs[i][perm]
            # gs_ind = perm

            for j in range(topN):
                # cam = cams[i, gs_ind[j], :]
                
                cam_b = cams[i, [gs_ind[j]]]
                cam_b = torch.mean(cam_b, dim=0, keepdim=True)
                cam_b = cam_b.detach().cpu().numpy().transpose(1,2,0)
                cam_b = resize_cam(cam_b, size=(h, w))
                [x0, y0, x1, y1] = get_bboxes(cam_b, cam_thr=0.6)
                # [x0, y0, x1, y1] = get_random_coordi(h)
                # print('hey',[x0, y0, x1, y1])
                # linputs[i:i+1, j ] = F.interpolate(x[i:i + 1, :, y0:(y1+1), x0:(x1+1)], size=(h, w),
                #                             mode='bilinear', align_corners=True)
                linputs[i:i+1, j] = F.interpolate(x[i:i + 1, :, y0:y1, x0:x1], size=(h, w),
                                            mode='bilinear', align_corners=True)
        linputs = linputs.view(b * topN, 3, h, w)
        _, part_encoded2 = self.transformer(linputs.detach())
        la = part_encoded2[:, 1:]
        h_w, in_features = la.shape[1], la.shape[2]
        h_,w_ = int(math.sqrt(h_w)), int(math.sqrt(h_w))
        la = la.transpose(1,2).contiguous().view(b*topN, in_features, h_, w_)


        lf = feats_pooling(la, method='avg', sh=int(h/16), sw=int(w/16))
        ls = self.convclass(lf)
        ls = F.max_pool2d(ls.reshape(b, topN, self.num_classes, 1).permute(0,3,1,2), (topN, 1))
        ls = ls.view(ls.size(0), -1)
        # cls2 = self.part_head(part_encoded2[:, 0])
        # cls2 = cls2.reshape(b, topN, self.num_classes).unsqueeze(1)
        # cls2 = ls+F.max_pool2d(cls2, (topN,1)).squeeze(1).squeeze(1)
        # print(ls.shape, cls2.shape, 'hhhhhh')

        return gs, ls#cls1, cls2
        

        #return self.part_head(cls1_1)#, self.part_head(cls1_1), self.part_head(cls1_1), self.part_head(cls1_1)

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.transformer.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.transformer.parameters())
        return [
                {'params': self.transformer.parameters(), 'lr': lr * lrp},
                {'params': large_lr_layers, 'lr': lr},
                ]

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token1.copy_(np2th(weights["cls"]))
            # self.transformer.embeddings.cls_token2.copy_(np2th(weights["cls"]))
            # self.transformer.embeddings.cls_token3.copy_(np2th(weights["cls"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                print("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                if bname.startswith('part') == False:
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname) 

def get_cams(attn_weights, feature_map, device):
    # M = torch.randn(attn_weights[0].shape[0], 19*19)
    mask_matrix = []
    for item in range(attn_weights[0].shape[0]):
        attn_weights_new = []
        for i in range(len(attn_weights)):
            attn_weights_new.append(attn_weights[i][item])
        att_mat = torch.stack(attn_weights_new).squeeze(1)#.cpu()
        att_mat = torch.mean(att_mat, dim=1)
        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat + residual_att.to(device)
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
        joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
        joint_attentions[0] = aug_att_mat[0]
        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
        v = joint_attentions[-1]
        # patch = torch.mean(v[:, 1:],dim=0)/(torch.mean(v[:, 1:],dim=0).max())
        grid_size = int(np.sqrt(v.size(-1)))
        # mask = torch.mean(v[:, 1:],dim=0)/(torch.mean(v[:, 1:],dim=0).max())
        mask = v[0, 1:].reshape(grid_size, grid_size)
        # mask = v[0, 1:]/v[0, 1:].max()
        # mask = mask.reshape(grid_size, grid_size)
        mask_matrix.append(mask)
    mask_matrix = torch.stack(mask_matrix).unsqueeze(1)

    cams = mask_matrix * feature_map
    # cams = mask_matrix + feature_map
    return cams
    # a = torch.mean(cams)*1.5 #1.5
    # M = (cams>a).float()
    # return M
    

def resize_cam(cam, size=(448, 448)):
    cam = cv2.resize(cam , (size[0], size[1]))
    #cam = cam - cam.min()
    #cam = cam / cam.max()
    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min)
    return cam

def get_bboxes(cam, cam_thr=0.2):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)
    #thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [x, y, x + w, y + h]
    else:
        estimated_bbox = [0, 0, 1, 1]

    return estimated_bbox  #, thr_gray_heatmap, len(contours)

def get_coordi_attention(cam):#(80,80)
    # coordinates = []
    mask_np = cam.cpu().detach().numpy()
    component_labels = measure.label(mask_np)

    properties = measure.regionprops(component_labels)
    areas = []
    for prop in properties:
        areas.append(prop.area)
    max_idx = areas.index(max(areas))

    bbox = properties[max_idx]['bbox']
    x_lefttop = bbox[0] * 16 - 1
    y_lefttop = bbox[1] * 16 - 1
    x_rightlow = bbox[2] * 16 - 1
    y_rightlow = bbox[3] * 16 - 1
    # for image
    if x_lefttop < 0:
        x_lefttop = 0
    if y_lefttop < 0:
        y_lefttop = 0
    coordinate = [int(x_lefttop), int(y_lefttop), int(x_rightlow), int(y_rightlow)]
    # coordinates.append(coordinate)
    return coordinate


def get_random_coordi(size):
    left_seed = np.random.randint(0, size, 2)
    x_lefttop, y_lefttop = left_seed[0], left_seed[1]
    x_rest = size - x_lefttop
    y_rest = size - y_lefttop
    x_rightlow = x_lefttop + np.random.randint(0, x_rest, 1)[0]
    y_rightlow = y_lefttop + np.random.randint(0, y_rest, 1)[0]
    if x_lefttop < 0:
        x_lefttop = 0
    if y_lefttop < 0:
        y_lefttop = 0
    if x_rightlow < 0:
        x_rightlow = 0
    if y_rightlow < 0:
        y_rightlow = 0
    coordinate = [int(x_lefttop), int(y_lefttop), int(x_rightlow), int(y_rightlow)]
    return coordinate


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.split = 'non-overlap'
    # config.split = 'overlap'
    config.slide_step = 12
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_l16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.split = 'non-overlap'
    # config.split = 'overlap'
    config.slide_step = 12
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

CONFIGS = {
    'ViT-B_16': get_b16_config(),
    # 'ViT-B_32': get_b32_config(),
    'ViT-L_16': get_l16_config(),
    # 'ViT-L_32': get_l32_config(),
    # 'ViT-H_14': get_h14_config(),
    # 'testing': get_testing(),
}


