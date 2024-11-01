import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
# from knn_cuda import KNN
# from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2   
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points
from .pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from .curvenet_util import *
from collections import OrderedDict




class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel, group_norm=False):
        super().__init__()
        self.encoder_channel = encoder_channel
        if group_norm:
            first_norm = nn.GroupNorm(8, 128)
            second_norm = nn.GroupNorm(8, 512)
        else:
            first_norm = nn.BatchNorm1d(128)
            second_norm = nn.BatchNorm1d(512)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            first_norm,
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            second_norm,
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        #self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3    N  number of points ,  M is number of centers (number of groups )
            ---------------------------
            output: B G M 3     G is group size 32
            center : B G 3
        '''
        # print(xyz.shape)
        # if len(xyz.shape) == 2:
        #     xyz = torch.unsqueeze(xyz, dim=0)

        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3    sample 128 center points from 2048 points
        # knn to get the neighborhood
        ##_, idx = self.knn(xyz, center)  # B G M,   kNN samples for each center  idx (B, M, G)   every center has G (group size) NN points

        idx = knn_points(center.cuda(), xyz.cuda(), K=self.group_size, return_sorted=False)
        idx = idx.idx
        idx = idx.long()


        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points  # idx_base  (8, 1, 1)
        idx = idx + idx_base  # for  batch 0 offset 0,   batch 1 ~7,  offset  1*2048
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx,
                       :]  # (8, 2048, 3) -> (8*2048, 3)   # todo sampling the neighborhoold points for each center in each batch
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size,
                                         3).contiguous()  # (8, 128, 32, 3)  128 groups, each group has 32 points,
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center



class Group_DINO(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        #self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        _, batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz[0], self.num_group) # B G 3 

        neighborhood_list = []
        for i in range (6):    
            idx = knn_points(center.cuda(), xyz[i].cuda(), K=self.group_size, return_sorted=False)
            idx = idx.idx
            idx = idx.long()

            # Gather the points corresponding to the indices

            assert idx.size(1) == self.num_group
            assert idx.size(2) == self.group_size
            idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
            idx = idx + idx_base
            idx = idx.view(-1)
            neighborhood = xyz[i].view(batch_size * num_points, -1)[idx, :]
            neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

            neighborhood_list.append(neighborhood)
        
        neighborhood = torch.stack(neighborhood_list, 0).permute(1,0,2,3,4).reshape(-1, self.num_group, 32, 3)

        # normalize
        neighborhood_org = neighborhood
        center = torch.cat((center[:, None, :, :], center[:, None, :, :], center[:, None, :, :], center[:, None, :, :], center[:, None, :, :], center[:, None, :, :]), 1)
        neighborhood = neighborhood - center.reshape(-1, self.num_group, 3).unsqueeze(2)
        return neighborhood, center.reshape(-1, self.num_group, 3), neighborhood_org 


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# todo now it will return the features and feature list for part-segmentation classification head
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return x, feature_list


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.group_norm = config.group_norm
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Transformer')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims, group_norm=self.group_norm)

        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool() 

        self.num_mask = int(self.mask_ratio * G)  # 115, 13 masked

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def forward(self, neighborhood, center, noaug=False, only_unmasked=True):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        if self.mask_ratio == 0:
            only_unmasked = False

        if only_unmasked:
            batch_size, seq_len, C = group_input_tokens.size()

            x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)

            masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
            pos = self.pos_embed(masked_center)
            x_vis = torch.cat((cls_tokens, x_vis), dim=1)
        else:
            pos = self.pos_embed(center)
            x_vis = torch.cat((cls_tokens, group_input_tokens), dim=1)

        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x_vis, x_vis_feature_list = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos, x_vis_feature_list, group_input_tokens


@MODELS.register_module()
class Point_MAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger='Point_MAE')
        self.config = config
        self.cls_dim = config.cls_dim
        self.group_norm = config.group_norm
        self.num_hid_cls_layers = config.num_hid_cls_layers
        self.trans_dim = config.transformer_config.trans_dim

        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.regularize = config.regularize
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        last_dim = 2 * self.trans_dim
        class_blocks = []

        for cls_block in range(0, self.num_hid_cls_layers):
            if self.group_norm:
                norm_layer = nn.GroupNorm(8, 256)
            else:
                norm_layer = nn.BatchNorm1d(256)       

            class_blocks.extend((nn.Linear(last_dim, 256), norm_layer, nn.ReLU(inplace=True), nn.Dropout(0.5)))
            last_dim = 256
        self.class_head = nn.Sequential(*class_blocks, nn.Linear(last_dim, self.cls_dim))        ### uncomment for shapenet dataset     
        
        # Define the last layer separately    
        # last_layer = nn.Linear(last_dim, self.cls_dim)          

        # self.class_head = nn.Sequential(
        #     *class_blocks,  # Unnamed layers
        #     nn.Sequential(OrderedDict([("custom_last_layer_name", last_layer)]))  # Named layer  
        # )    
        
        
        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')

        self.method = config.transformer_config.method
        self.loss_encoder = config.transformer_config.loss_encoder

        if (self.method == "MATE"):  
            
            self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)  

        
        elif (self.method == "MATE_joint_MSE" or self.method == "MATE_only_cls_KLD" or self.method == "MATE_only_cls_CE" or self.method == "MATE_joint_MSE_Downsample" or self.method == "MATE_our_joint_MSE_Downsample"
              or self.method == "Tent_Dino"):   
 
            self.group_divider = Group_DINO(num_group=self.num_group, group_size=self.group_size) 


        self.group_divider_test = Group(num_group=self.num_group, group_size=self.group_size)          

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)
        self.l1_consistency_loss = torch.nn.L1Loss(reduction='mean')

        ################### CE loss for DINO
        # self.student_temp = 0.1
        # self.warmup_teacher_temp = 0.04
        # self.teacher_temp = 0.04
        # self.warmup_teacher_temp_epochs = 0
        # self.nepochs = 300
        # self.register_buffer("center", torch.zeros(1, 64, 384))  
        # #self.register_buffer("center", torch.zeros(1, 26, 384)) 

        # self.teacher_temp_schedule = np.concatenate((
        #     np.linspace(self.warmup_teacher_temp,
        #                 self.teacher_temp, self.warmup_teacher_temp_epochs),
        #     np.ones(self.nepochs - self.warmup_teacher_temp_epochs) * self.teacher_temp
        # ))

        # self.center_momentum = 0.9
        ####################################

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            #elf.loss_func = ChamferDistanceL1().cuda()  
            self.loss_func = chamfer_distance
        elif loss_type == 'cdl2':
            #self.loss_func = ChamferDistanceL2().cuda()
            self.loss_func = chamfer_distance
        else:
            raise NotImplementedError

        self.loss_ce = nn.CrossEntropyLoss()

        # self.loss_func = emd().cuda()

    def get_loss_acc(self, ret, gt):

        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path, load_part_seg=None):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def classification_only(self, pts, only_unmasked=True):
        
        neighborhood, center = self.group_divider_test(pts)
        x_vis_w_token = self.MAE_encoder(neighborhood, center, only_unmasked=only_unmasked)[0]  
        feat = torch.cat([x_vis_w_token[:, 0], x_vis_w_token[:, 1:].max(1)[0]], dim=-1)
        class_ret = self.class_head(feat)
        return class_ret
    
    def classification_only_train(self, pts, only_unmasked=True, epoch = 0):
        
        neighborhood, center, neighborhood_org = self.group_divider(pts)  
        x_vis_w_token = self.MAE_encoder(neighborhood, center, only_unmasked=only_unmasked)[0]  
        feat = torch.cat([x_vis_w_token[:, 0], x_vis_w_token[:, 1:].max(1)[0]], dim=-1)
        class_ret = self.class_head(feat)
        if self.method == "MATE_only_cls_KLD":
                kld_loss = self.DINO_KLD_loss(x_vis_w_token[:, 1:])
                return class_ret, kld_loss  
        
        if self.method == "MATE_only_cls_CE":
                ce_loss = self.DINO_CE_loss(x_vis_w_token[:, 1:], epoch)    
                return class_ret, ce_loss  
            
        if (self.method == "Tent_Dino"):    
            mse_loss = self.DINO_MSE_loss(x_vis_w_token[:, 1:])   
            return class_ret, mse_loss 
            

    def forward(self, pts, vis=False, cyclic=False, **kwargs):

        if (self.method == "MATE"):
            neighborhood, center = self.group_divider(pts)

        elif (self.method == "MATE_joint_MSE" or self.method == "MATE_joint_MSE_Downsample" or self.method == "MATE_our_joint_MSE_Downsample"):    
            neighborhood, center, neighborhood_org = self.group_divider(pts) 

        if (self.method == "MATE_our_joint_MSE_Downsample"):
            x_vis_w_token, mask, _, _ = self.MAE_encoder(neighborhood, center, noaug = True, only_unmasked = False)      
        else:
            x_vis_w_token, mask, _, _ = self.MAE_encoder(neighborhood, center)  
        x_vis = x_vis_w_token[:, 1:]
        B, _, C = x_vis.shape  # B VIS C
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        
        x_rec = self.MAE_decoder(x_full, pos_full, N)    
        
        feat = torch.cat([x_vis_w_token[:, 0], x_vis_w_token[:, 1:].max(1)[0]], dim=-1)      

        if (self.method == "MATE_joint_MSE" or self.method == "MATE_joint_MSE_Downsample" or self.method == "MATE_our_joint_MSE_Downsample"):        
            if (self.loss_encoder == "MSE"):
                #mse_loss = self.DINO_MSE_loss(x_rec[:, 1:])   
                mse_loss = self.DINO_MSE_loss(x_vis_w_token[:, 1:])       

        if not cyclic:
            class_ret = self.class_head(feat)
        else:
            class_ret = self.classification_only(pts, only_unmasked=False)  # return logits from 100% of tokens  

        B, M, C = x_rec.shape
        #rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
        # if self.MAE_encoder.mask_ratio == 0:  
        #     gt_points = neighborhood.reshape(B * M, -1, 3)   
        # else:
        if (self.method == "MATE_our_joint_MSE_Downsample"):    
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(-1, 6, 64, 32, 3)[:, 2:].reshape(-1, 32, 3)                       # B M 1024
            gt_points_1 = neighborhood[~mask].reshape(-1, 6, 64, 32, 3)[:, :2]
            gt_points = torch.cat((gt_points_1, gt_points_1), 1).reshape(-1, 32, 3)   
            loss1 = self.loss_func(rebuild_points, gt_points)   

        else:
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
            gt_points = neighborhood[mask].reshape(B * M, -1, 3)
            loss1 = self.loss_func(rebuild_points, gt_points)    

        if self.regularize:
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)

            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0).reshape(B, self.num_group, 32, 3)

            mean_rebuild = torch.mean(full, dim=0)

            regularization_loss = torch.tensor(0, dtype=torch.float).cuda()

            for bs in range((full.shape[0])):
                regularization_loss += self.loss_func(full[bs, :, :, :].squeeze(), mean_rebuild)
            regularization_loss = regularization_loss / full.shape[0]

            mean_class_ret = class_ret.mean(dim=0)
            ce_pred_consitency = torch.tensor(0, dtype=torch.float).cuda()

            for bs in range((class_ret.shape[0])):
                ce_pred_consitency += self.l1_consistency_loss(class_ret[bs, :].squeeze(), mean_class_ret.squeeze())
            class_ret = ce_pred_consitency / class_ret.shape[0]

        else:
            regularization_loss = torch.tensor(0, dtype=torch.float).cuda()       
            class_ret = class_ret

        # print(self.loss_func)
        # vis = True
        if vis:
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            return ret1, ret2, full_center
        else:
            if (self.method == "MATE" or self.method == "MATE_joint_MSE"):
                return loss1, class_ret, regularization_loss
            elif (self.method == "MATE_joint_MSE" or self.method == "MATE_joint_MSE_Downsample" or self.method == "MATE_our_joint_MSE_Downsample"):      
                return loss1, class_ret, mse_loss 


@MODELS.register_module()
class PointNet(nn.Module):
    def __init__(self, config, normal_channel=False):
        #super(get_model, self).__init__()
        super().__init__()
        if normal_channel:
            channel = 6
        else: 
            channel = 3  
                        
        self.cls_dim = config.cls_dim    
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)   
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        #self.fc3 = nn.Linear(256, self.cls_dim)   
        self.fc3_ = nn.Linear(256, self.cls_dim)              #### Fo loading shapenet  
        #self.dropout = nn.Dropout(p=0.4)  
        self.dropout = nn.Dropout(p=0.3)    
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.loss_ce = nn.CrossEntropyLoss()     
        
        ################ Added by Ali  
        # self.group_size = config.group_size
        # self.group_norm = config.group_norm
        # self.num_group = config.num_group
        # self.encoder_dims = config.transformer_config.encoder_dims
        
        # self.encoder = Encoder(encoder_channel=self.encoder_dims, group_norm=self.group_norm)
        # self.group_divider_test = Group(num_group=self.num_group, group_size=self.group_size) 
        ################ Added by Ali 
        
    def load_model_from_ckpt(self, bert_ckpt_path, load_part_seg=None):
        if bert_ckpt_path is not None:  
            ckpt = torch.load(bert_ckpt_path)    
            print(ckpt.keys())
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}      
            #base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['model_state'].items()}  
            
            ################# CVPR paper
            # New state dictionary with modified keys
            """new_state_dict = {}

            # Remove the prefix 'model.' from each key
            for key in base_ckpt.keys():
                new_key = key.replace('model.', '')  # This removes the 'model.' prefix from the keys
                new_state_dict[new_key] = base_ckpt[key]    
                
            base_ckpt = new_state_dict  """
            ################# CVPR paper      

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)    

    def get_loss_acc(self, ret, gt):

        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def forward(self, x):  
        
        # neighborhood, center = self.group_divider_test(x)
        # x = self.encoder(neighborhood)
        # x = torch.cat((center, x), -1)
        
        x, trans, trans_feat = self.feat(x.permute(0, 2, 1))  
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))                 
        #x = self.fc3(x)
        x = self.fc3_(x)
        #x = F.log_softmax(x, dim=1)         
        return x


@MODELS.register_module()
class DGCNN_cls(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.k
        
        self.cls_dim = config.cls_dim  
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(config.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), 
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, config.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(config.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=config.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256) 
        self.dp2 = nn.Dropout(p=config.dropout)
        self.linear3 = nn.Linear(256, self.cls_dim)
        #self.linear3_ = nn.Linear(256, self.cls_dim)                        #### for loading shapenet 
         
        self.loss_ce = nn.CrossEntropyLoss() 
        
        
    def load_model_from_ckpt(self, bert_ckpt_path, load_part_seg=None):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)    

    def get_loss_acc(self, ret, gt):

        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x.permute(0, 2, 1), k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points) 

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        #x = self.linear3_(x)                                                # (batch_size, 256) -> (batch_size, output_channels)        
        
        return x              
 

@MODELS.register_module()   
class PointNet_ssg_Plus_Plus(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.hparams = True   
        self.cls_dim = config.cls_dim  

        self._build_model()
        
        self.loss_ce = nn.CrossEntropyLoss() 

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[0, 64, 64, 128],  
                use_xyz=self.hparams,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=self.hparams,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], use_xyz=self.hparams
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim),
        )

    def load_model_from_ckpt(self, bert_ckpt_path, load_part_seg=None):  
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)    

    def get_loss_acc(self, ret, gt):

        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100



    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features) 

        return self.fc_layer(features.squeeze(-1))   
    
        
curve_config = {
    'default': [[100, 5], [100, 5], None, None],
    'long':  [[10, 30], None,  None,  None]}       

@MODELS.register_module()        
class CurveNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        k = 20
        setting='default'
        self.cls_dim = config.cls_dim  

        additional_channel = 32
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # encoder
        self.cic11 = CIC(npoint=1024, radius=0.05, k=k, in_channels=additional_channel, output_channels=64, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][0])
        self.cic12 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][0])
        
        self.cic21 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][1])
        self.cic22 = CIC(npoint=1024, radius=0.1, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][1])

        self.cic31 = CIC(npoint=256, radius=0.1, k=k, in_channels=128, output_channels=256, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][2])
        self.cic32 = CIC(npoint=256, radius=0.2, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][2])

        self.cic41 = CIC(npoint=64, radius=0.2, k=k, in_channels=256, output_channels=512, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][3])
        self.cic42 = CIC(npoint=64, radius=0.4, k=k, in_channels=512, output_channels=512, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][3])

        self.conv0 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Linear(1024 * 2, 512, bias=False)
        self.conv2 = nn.Linear(512, self.cls_dim) 
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        
        self.loss_ce = nn.CrossEntropyLoss()   
        
        
    def load_model_from_ckpt(self, bert_ckpt_path, load_part_seg=None):  
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)    
            

    def get_loss_acc(self, ret, gt):

        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100    

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)  
        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)
 
        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)

        x = self.conv0(l4_points)
        x_max = F.adaptive_max_pool1d(x, 1)
        x_avg = F.adaptive_avg_pool1d(x, 1)
        
        x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)
        x = F.relu(self.bn1(self.conv1(x).unsqueeze(-1)), inplace=True).squeeze(-1)
        x = self.dp1(x)
        x = self.conv2(x)
        return x     
        

@MODELS.register_module()
class PointNet_rotnet(nn.Module):
    def __init__(self, config, normal_channel=False):
        #super(get_model, self).__init__()
        super().__init__()
        if normal_channel:
            channel = 6
        else: 
            channel = 3  
                        
        self.cls_dim = config.cls_dim    
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)   
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.cls_dim)   
        #self.fc3_ = nn.Linear(256, self.cls_dim)              #### Fo loading shapenet  
        self.dropout = nn.Dropout(p=0.4)  
        #self.dropout = nn.Dropout(p=0.3)    
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.loss_ce = nn.CrossEntropyLoss()        
        
        ################ Added by Ali  
        # self.group_size = config.group_size
        # self.group_norm = config.group_norm
        # self.num_group = config.num_group
        # self.encoder_dims = config.transformer_config.encoder_dims
        
        # self.encoder = Encoder(encoder_channel=self.encoder_dims, group_norm=self.group_norm)
        # self.group_divider_test = Group(num_group=self.num_group, group_size=self.group_size) 
        ################ Added by Ali 
        
    def load_model_from_ckpt(self, bert_ckpt_path, load_part_seg=None):
        if bert_ckpt_path is not None:  
            ckpt = torch.load(bert_ckpt_path)    
            print(ckpt.keys())
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}      
            #base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['model_state'].items()}
            
            ################# CVPR paper
            # New state dictionary with modified keys
            """new_state_dict = {}

            # Remove the prefix 'model.' from each key
            for key in base_ckpt.keys():
                new_key = key.replace('model.', '')  # This removes the 'model.' prefix from the keys
                new_state_dict[new_key] = base_ckpt[key]    
                
            base_ckpt = new_state_dict    """
            ################# CVPR paper    

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)    

    def get_loss_acc(self, ret, gt):

        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def forward_rotnet(self, pts, pts_rot, gt, gt_rot, tta=False, **kwargs):
        
        # neighborhood, center = self.group_divider_test(x)
        # x = self.encoder(neighborhood)
        # x = torch.cat((center, x), -1)
        
        x, trans, trans_feat = self.feat(pts_rot.permute(0, 2, 1))  
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))             
        x = self.fc3(x)
        #x = self.fc3_(x)
        loss_rot = self.loss_ce(x, gt_rot.long())  
        #x = F.log_softmax(x, dim=1)     
        return loss_rot
    
    
    def forward(self, pts, pts_rot, gt, gt_rot, tta=False, **kwargs):
        
        # neighborhood, center = self.group_divider_test(x)
        # x = self.encoder(neighborhood)
        # x = torch.cat((center, x), -1)
        
        x, trans, trans_feat = self.feat(pts.permute(0, 2, 1))  
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))               
        x = self.fc3(x)
        #x = self.fc3_(x)
        #x = F.log_softmax(x, dim=1)     
        return x


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx 

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

# todo pointmae model for joint-training of RotNet (sun et al. TTT)
@MODELS.register_module()
class Point_MAE_rotnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger='Point_MAE')
        self.config = config
        self.cls_dim = config.cls_dim
        self.cls_dim_rotation = config.cls_dim_rotation
        self.group_norm = config.group_norm
        self.num_hid_cls_layers = config.num_hid_cls_layers
        self.trans_dim = config.transformer_config.trans_dim

        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        last_dim = 2 * self.trans_dim
        class_blocks = []

        for cls_block in range(0, self.num_hid_cls_layers):
            if self.group_norm:
                norm_layer = nn.GroupNorm(8, 256)
            else:
                norm_layer = nn.BatchNorm1d(256)

            class_blocks.extend((nn.Linear(last_dim, 256), norm_layer, nn.ReLU(inplace=True), nn.Dropout(0.5)))
            last_dim = 256
        self.class_head = nn.Sequential(*class_blocks, nn.Linear(last_dim, self.cls_dim))  # outputs == num of classes
        self.class_head_rotnet = nn.Sequential(*class_blocks, nn.Linear(last_dim, self.cls_dim_rotation))  # 4 outputs

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        trunc_normal_(self.mask_token, std=.02)
        # loss
        self.loss_ce = nn.CrossEntropyLoss()

    def load_model_from_ckpt(self, bert_ckpt_path, load_part_seg=None):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def classification_only(self, pts, pts_rot, gt, gt_rot, tta=False): 
        if not tta:
            neighborhood, center = self.group_divider(pts)
            neighborhood_rot, center_rot = self.group_divider(pts_rot)

            x_vis_w_token = self.MAE_encoder(neighborhood, center, only_unmasked=False)[0]
            x_vis_w_token_rot = self.MAE_encoder(neighborhood_rot, center_rot, only_unmasked=False)[0]

            feat = torch.cat([x_vis_w_token[:, 0], x_vis_w_token[:, 1:].max(1)[0]], dim=-1)
            feat_rot = torch.cat([x_vis_w_token_rot[:, 0], x_vis_w_token_rot[:, 1:].max(1)[0]], dim=-1)

            class_ret = self.class_head(feat)
            class_ret_rot = self.class_head_rotnet(feat_rot)

            pred_rot = class_ret_rot.argmax(-1)
            acc_cls_rot = (pred_rot == gt_rot).sum() / float(gt.size(0))
            pred = class_ret.argmax(-1)
            acc_cls = (pred == gt).sum() / float(gt.size(0))

            return acc_cls * 100, acc_cls_rot * 100
        else:
            neighborhood, center = self.group_divider(pts)
            x_vis_w_token = self.MAE_encoder(neighborhood, center, only_unmasked=False)[0]
            feat = torch.cat([x_vis_w_token[:, 0], x_vis_w_token[:, 1:].max(1)[0]], dim=-1)
            class_ret = self.class_head(feat)

            return class_ret

    def forward(self, pts, pts_rot, gt, gt_rot, tta=False, **kwargs):
        if not tta:
            neighborhood, center = self.group_divider(pts)
            neighborhood_rot, center_rot = self.group_divider(pts_rot)

            x_vis_w_token = self.MAE_encoder(neighborhood, center, only_unmasked=False)[0]
            x_vis_w_token_rot = self.MAE_encoder(neighborhood_rot, center_rot, only_unmasked=False)[0]

            feat = torch.cat([x_vis_w_token[:, 0], x_vis_w_token[:, 1:].max(1)[0]], dim=-1)
            feat_rot = torch.cat([x_vis_w_token_rot[:, 0], x_vis_w_token_rot[:, 1:].max(1)[0]], dim=-1)

            class_ret = self.class_head(feat)
            class_ret_rot = self.class_head_rotnet(feat_rot)

            loss_cls = self.loss_ce(class_ret, gt.long())
            loss_rot = self.loss_ce(class_ret_rot, gt_rot.long())
            pred_rot = class_ret_rot.argmax(-1)
            acc_cls_rot = (pred_rot == gt_rot).sum() / float(gt.size(0))
            pred = class_ret.argmax(-1)
            acc_cls = (pred == gt).sum() / float(gt.size(0))
            return loss_cls, loss_rot, acc_cls * 100, acc_cls_rot * 100
        else:
            neighborhood_rot, center_rot = self.group_divider(pts_rot)
            x_vis_w_token_rot = self.MAE_encoder(neighborhood_rot, center_rot, only_unmasked=False)[0]
            feat_rot = torch.cat([x_vis_w_token_rot[:, 0], x_vis_w_token_rot[:, 1:].max(1)[0]], dim=-1)
            class_ret_rot = self.class_head_rotnet(feat_rot)
            loss_rot = self.loss_ce(class_ret_rot, gt_rot.long())
            return loss_rot


##### PointMAE for Part Segmentation #####
@MODELS.register_module()
class Point_MAE_PartSegmentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE_Segmentation] ', logger='Point_MAE_Segmentation')
        self.config = config
        self.npoint = config.npoint
        self.cls_dim = config.cls_dim
        self.num_classes = config.num_classes
        self.group_norm = config.group_norm
        self.num_hid_cls_layers = config.num_hid_cls_layers
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        convs1 = nn.Conv1d(3392, 512, 1)
        dp1 = nn.Dropout(0.5)
        convs2 = nn.Conv1d(512, 256, 1)
        convs3 = nn.Conv1d(256, self.cls_dim, 1)
        bns1 = nn.BatchNorm1d(512)
        bns2 = nn.BatchNorm1d(256)

        relu = nn.ReLU()

        class_blocks = [convs1, bns1, relu, dp1, convs2, bns2, relu, convs3]

        self.class_head = nn.Sequential(*class_blocks)

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE_Segmentation')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3,
                                                        mlp=[self.trans_dim * 4, 1024])

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(0.2))
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            #self.loss_func = ChamferDistanceL1().cuda()
            self.loss_func = chamfer_distance
        elif loss_type == 'cdl2':
            #self.loss_func = ChamferDistanceL2().cuda()
            self.loss_func = chamfer_distance
        else:
            raise NotImplementedError

        self.loss_seg = nn.NLLLoss()

    def get_acc(self, args, seg_pred, target):
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        acc = correct.item() / (args.batch_size * self.npoint)
        return acc

    def load_model_from_ckpt(self, bert_ckpt_path, load_part_seg=False):
        if load_part_seg:
            ckpt = torch.load(bert_ckpt_path)

            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['model_state_dict'].items()}

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')


        else:
            if bert_ckpt_path is not None:
                ckpt = torch.load(bert_ckpt_path)
                base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

                incompatible = self.load_state_dict(base_ckpt, strict=False)

                if incompatible.missing_keys:
                    print_log('missing_keys', logger='Transformer')
                    print_log(
                        get_missing_parameters_message(incompatible.missing_keys),
                        logger='Transformer'
                    )
                if incompatible.unexpected_keys:
                    print_log('unexpected_keys', logger='Transformer')
                    print_log(
                        get_unexpected_parameters_message(incompatible.unexpected_keys),
                        logger='Transformer'
                    )

                print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
            else:
                print_log('Training from scratch!!!', logger='Transformer')
                self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def classification_only(self, pts, cls_label, only_unmasked=True):
        B, N, C = pts.shape
        neighborhood, center = self.group_divider(pts)
        x_vis_w_token, mask, feature_list, group_input_tokens = self.MAE_encoder(neighborhood, center,
                                                                                 only_unmasked=only_unmasked)
        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]

        x = torch.cat((feature_list[0], feature_list[1], feature_list[2]), dim=1)  # 1152

        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)

        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)

        cls_label_one_hot = cls_label.view(B, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)

        x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), 1)  # 1152*2 + 64
        f_level_0 = self.propagation_0(pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x, mask_ratio=self.MAE_encoder.mask_ratio)

        x = torch.cat((f_level_0, x_global_feature), 1)

        class_ret = self.class_head(x)
        class_ret = F.log_softmax(class_ret, dim=1)
        class_ret = class_ret.permute(0, 2, 1)
        return class_ret

    def forward(self, pts, cls_label, cls_loss_masked=True, tta=False, vis=False, **kwargs):
        B_, N_, _ = pts.shape  # pts (8, 2048, 3),  cls_label (8,1,16) one-hot ,   partnet_cls

        neighborhood, center = self.group_divider(pts)  # normalized neighborhood  (8, 128, 32, 3) 128 groups, each group has 32 points,   center (8, 128, 3)  128 group centers
        x_vis_w_token, mask, feature_list, group_input_tokens = self.MAE_encoder(neighborhood, center)
        #  todo x_vis_w_token (8, 14, 384), mask (8,128) feature_list 3-level features:  a list of (8,14,384),  group_input_tokens (8,128,384)
        x_vis = x_vis_w_token[:, 1:]
        B, _, C = x_vis.shape  # B VIS C
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)  # positional embedding for visible tokens  13
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)  # positional embedding for masked tokens   115

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)  # (8, 115, 384)
        x_full = torch.cat([x_vis, mask_token], dim=1)  # (8, 128, 384)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)  # (8, 128, 384)

        x_rec = self.MAE_decoder(x_full, pos_full, N)  # #  todo only the masked token are reconstructed  (8, 115, 384)
        if not tta:
            feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in
                            feature_list]  # feature_list  a list of (8,384, 14)
            x = torch.cat((feature_list[0], feature_list[1], feature_list[2]), dim=1)  # (8,1152,14)    384x3 = 1152
            x_max = torch.max(x, 2)[0]  # (8, 1152)
            x_avg = torch.mean(x, 2)  # (8, 1152)
            # todo 3 types of features: maxpoing global feature, avgpooling global feature, object label feature,   duplicate to N=2048
            x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N_)  # todo  duplicate the feature on 3rd dimension for N=2048 (8, 1152, 2048)
            x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N_)  # (8, 1152, 2048)
            # todo  cls_label is object category label, it is considered as a data source, which is used to compute features
            cls_label_one_hot = cls_label.view(B, self.num_classes, 1)
            cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N_)  # (8, 64, 2048)

            x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), 1)  # (8, 2368, 2048)    1152*2 + 64 = 2368, feature dim

            # todo  the problem is
            #   x is the concatenation of 3-level featurse only for the 14 visible tokens (note that only visible tokens have features from encoder)
            #    but here the center is still  all 128 centers
            #  todo ############ suggested correction ################################################
            #   ############################################################

            n_visible_tokens = x_vis.size(1)
            center_visible = center[~mask].reshape(B, n_visible_tokens, 3)
            f_level_0 = self.propagation_0(pts.transpose(-1, -2), center_visible.transpose(-1, -2), pts.transpose(-1, -2), x, mask_ratio=self.MAE_encoder.mask_ratio)
            # todo instead of
            # f_level_0 = self.propagation_0(pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x)
            #  todo ############################################################
            #     ############################################################

            x = torch.cat((f_level_0, x_global_feature), 1)

            # todo - if we do not want to pass the tokens through the cls head set 'cls_loss_masked' to False
            # todo - if this is false, cls outputs are taken from the method - 'classification_only'
            # todo - the advantage of taking the outputs from 'classification_only' is that cls loss can be computed from
            # todo - 100% of the tokens!!!
            if cls_loss_masked:
                class_ret = self.class_head(x)
                class_ret = F.log_softmax(class_ret, dim=1)
                class_ret = class_ret.permute(0, 2, 1)
            else:
                class_ret = self.classification_only(pts, cls_label, only_unmasked=False)
        else:
            class_ret = 0

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        if vis:  # visualization
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            return ret1, ret2, full_center


        # dummy = neighborhood[mask]
        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        loss1 = self.loss_func(rebuild_points, gt_points)
        return loss1, class_ret


##### PointMAE for Semantic Segmentation #####
@MODELS.register_module()
class Point_MAE_SemSegmentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE_Segmentation] ', logger='Point_MAE_Segmentation')
        self.config = config
        self.npoint = config.npoint
        self.cls_dim = config.cls_dim
        self.group_norm = config.group_norm
        self.num_hid_cls_layers = config.num_hid_cls_layers
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        convs1 = nn.Conv1d(3328, 512, 1)
        dp1 = nn.Dropout(0.5)
        convs2 = nn.Conv1d(512, 256, 1)
        convs3 = nn.Conv1d(256, self.cls_dim, 1)
        bns1 = nn.BatchNorm1d(512)
        bns2 = nn.BatchNorm1d(256)

        relu = nn.ReLU()

        class_blocks = [convs1, bns1, relu, dp1, convs2, bns2, relu, convs3]

        self.class_head = nn.Sequential(*class_blocks)

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE_Segmentation')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3,
                                                        mlp=[self.trans_dim * 4, 1024])

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(0.2))
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            #self.loss_func = ChamferDistanceL1().cuda()
            self.loss_func = chamfer_distance
        elif loss_type == 'cdl2':
            #self.loss_func = ChamferDistanceL2().cuda()  
            self.loss_func = chamfer_distance
        else:
            raise NotImplementedError

        self.loss_seg = nn.NLLLoss()

    def get_acc(self, args, seg_pred, target):
        pred_choice = seg_pred.data.max(1)[1]
        # import pdb
        # pdb.set_trace()
        correct = pred_choice.eq(target.data).cpu().sum()
        acc = correct.item() / (args.batch_size * self.npoint)
        return acc

    def load_model_from_ckpt(self, bert_ckpt_path, load_part_seg=False):
        if load_part_seg:
            ckpt = torch.load(bert_ckpt_path)

            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['model_state_dict'].items()}

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')


        else:
            if bert_ckpt_path is not None:
                ckpt = torch.load(bert_ckpt_path)
                base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

                incompatible = self.load_state_dict(base_ckpt, strict=False)

                if incompatible.missing_keys:
                    print_log('missing_keys', logger='Transformer')
                    print_log(
                        get_missing_parameters_message(incompatible.missing_keys),
                        logger='Transformer'
                    )
                if incompatible.unexpected_keys:
                    print_log('unexpected_keys', logger='Transformer')
                    print_log(
                        get_unexpected_parameters_message(incompatible.unexpected_keys),
                        logger='Transformer'
                    )

                print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
            else:
                print_log('Training from scratch!!!', logger='Transformer')
                self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # todo # now it should be segmentation # todo
    def classification_only(self, pts, only_unmasked=True):
        B, N, C = pts.shape
        neighborhood, center = self.group_divider(pts)
        x_vis_w_token, mask, feature_list, group_input_tokens = self.MAE_encoder(neighborhood, center,
                                                                                 only_unmasked=only_unmasked)
        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]

        x = torch.cat((feature_list[0], feature_list[1], feature_list[2]), dim=1)  # 1152

        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)

        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)

        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1)  # 1152*2 + 64
        f_level_0 = self.propagation_0(pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x)

        x = torch.cat((f_level_0, x_global_feature), 1)

        class_ret = self.class_head(x)
        class_ret = F.log_softmax(class_ret, dim=1)
        class_ret = class_ret.permute(0, 2, 1)
        return class_ret

    def forward(self, pts, cls_loss_masked=True, tta=False, **kwargs):

        B_, N_, _ = pts.shape  # pts (8, 2048, 3),  cls_label (8,1,16) one-hot ,   partnet_cls

        neighborhood, center = self.group_divider(
            pts)  # normalized neighborhood  (8, 128, 32, 3) 128 groups, each group has 32 points,   center (8, 128, 3)  128 group centers
        x_vis_w_token, mask, feature_list, group_input_tokens = self.MAE_encoder(neighborhood, center)
        #  todo x_vis_w_token (8, 14, 384), mask (8,128) feature_list 3-level features:  a list of (8,14,384),  group_input_tokens (8,128,384)
        x_vis = x_vis_w_token[:, 1:]
        B, _, C = x_vis.shape  # B VIS C
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1,
                                                                    C)  # positional embedding for visible tokens  13
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1,
                                                                    C)  # positional embedding for masked tokens   115
        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)  # (8, 115, 384)
        x_full = torch.cat([x_vis, mask_token], dim=1)  # (8, 128, 384)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)  # (8, 128, 384)

        x_rec = self.MAE_decoder(x_full, pos_full, N)  # #  todo only the masked token are reconstructed  (8, 115, 384)
        if not tta:
            feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in
                            feature_list]  # feature_list  a list of (8,384, 14)
            # todo concatenation of 3-level features only for 14 visible tokens
            x = torch.cat((feature_list[0], feature_list[1], feature_list[2]), dim=1)  # (8,1152,14)    384x3 = 1152
            x_max = torch.max(x, 2)[0]  # (8, 1152)
            x_avg = torch.mean(x, 2)  # (8, 1152)
            # todo 3 types of features: maxpoing global feature, avgpooling global feature, object label feature,   duplicate to N=2048
            x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1,
                                                                   N_)  # todo  duplicate the feature on 3rd dimension for N=2048 (8, 1152, 2048)
            x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N_)  # (8, 1152, 2048)

            x_global_feature = torch.cat((x_max_feature, x_avg_feature),
                                         1)  # (8, 2368, 2048)    1152*2 + 64 = 2368, feature dim

            # todo  the problem is
            #   x is the concatenation of 3-level featurse only for the 14 visible tokens (note that only visible tokens have features from encoder)
            #    but here the center is still  all 128 centers
            #  todo ############ suggested correction ################################################
            #   ############################################################

            n_visible_tokens = x_vis.size(1)
            center_visible = center[~mask].reshape(B, n_visible_tokens, 3)
            f_level_0 = self.propagation_0(pts.transpose(-1, -2), center_visible.transpose(-1, -2),
                                           pts.transpose(-1, -2), x)
            # todo instead of
            # f_level_0 = self.propagation_0(pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x)
            #  todo ############################################################
            #     ############################################################

            x = torch.cat((f_level_0, x_global_feature), 1)

            # todo - if we do not want to pass the tokens through the cls head set 'cls_loss_masked' to False
            # todo - if this is false, cls outputs are taken from the method - 'classification_only'
            # todo - the advantage of taking the outputs from 'classification_only' is that cls loss can be computed from
            # todo - 100% of the tokens!!!
            if cls_loss_masked:
                class_ret = self.class_head(x)
                class_ret = F.log_softmax(class_ret, dim=1)
                class_ret = class_ret.permute(0, 2, 1)
            else:
                class_ret = self.classification_only(pts, only_unmasked=False)
        else:
            class_ret = 0

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        loss1 = self.loss_func(rebuild_points, gt_points)
        return loss1, class_ret


# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.num_hid_cls_layers = config.num_hid_cls_layers
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        last_dim = self.trans_dim * 2
        class_blocks = []
        for cls_block in range(0, self.num_hid_cls_layers):
            class_blocks.extend((nn.Linear(last_dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5)))
            last_dim = 256
        self.class_head = nn.Sequential(*class_blocks, nn.Linear(last_dim, self.cls_dim))

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)[0]
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.class_head(concat_f)
        return ret














########################## plot point clouds
import numpy as np
import torch
import open3d as o3d

def save_colored_point_cloud(centers, neighborhoods, filename="colored_patches.ply"):
    """
    Save a point cloud where each patch and its neighbors are assigned a unique color.

    Parameters:
    - centers (torch.Tensor): Tensor of shape (64, 3) representing the center points.
    - neighborhoods (torch.Tensor): Tensor of shape (64, 32, 3) representing the neighborhood points.
    - filename (str): The name of the output PLY file.
    """
    # Convert tensors to numpy arrays
    centers_np = centers.cpu().numpy()
    neighborhoods_np = neighborhoods.cpu().numpy()

    # Generate random colors for each patch
    colors = np.random.rand(centers_np.shape[0], 3)

    # Initialize an empty list to hold all points and their corresponding colors
    all_points = []
    all_colors = []

    # Loop through each patch and its neighborhood
    for i in range(centers_np.shape[0]):
        patch_points = neighborhoods_np[i]  # (32, 3)
        patch_color = np.tile(colors[i], (patch_points.shape[0], 1))  # Repeat the color 32 times

        all_points.append(patch_points)
        all_colors.append(patch_color)

    # Convert the lists to numpy arrays
    all_points = np.vstack(all_points)  # (64*32, 3)
    all_colors = np.vstack(all_colors)  # (64*32, 3)

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    # Save the point cloud to a PLY file
    o3d.io.write_point_cloud(filename, pcd)

    print(f"Point cloud saved as {filename} with each patch and its neighbors colored uniquely.")

# # Example usage:
# # Assuming you have centers and neighborhoods as PyTorch tensors
# centers = torch.rand(64, 3)  # Example tensor for centers
# neighborhoods = torch.rand(64, 32, 3)  # Example tensor for neighborhoods

# save_colored_point_cloud(centers, neighborhoods)


################################3
import numpy as np
import torch
import open3d as o3d
from scipy.spatial import cKDTree

def assign_colors_to_centers(centers, existing_centers=None, existing_colors=None):
    """
    Assign colors to centers, either generating new colors or matching to existing centers.

    Parameters:
    - centers (numpy.ndarray): Array of shape (64, 3) representing the center points.
    - existing_centers (numpy.ndarray): Optional. Existing centers to match colors with.
    - existing_colors (numpy.ndarray): Optional. Colors corresponding to existing centers.

    Returns:
    - colors (numpy.ndarray): Array of shape (64, 3) with colors for each center.
    """

    centers = centers.cpu().numpy()

    if existing_centers is None:
        # Generate random colors for the first set of centers
        return np.random.rand(centers.shape[0], 3)
    else:
        # Match current centers to existing centers and assign corresponding colors
        tree = cKDTree(existing_centers)
        _, indices = tree.query(centers)
        return existing_colors[indices]

def save_colored_point_cloud(centers, neighborhoods, colors, filename="colored_patches.ply"):
    """
    Save a point cloud where each patch and its neighbors are assigned a unique color.

    Parameters:
    - centers (torch.Tensor): Tensor of shape (64, 3) representing the center points.
    - neighborhoods (torch.Tensor): Tensor of shape (64, 32, 3) representing the neighborhood points.
    - colors (numpy.ndarray): Array of shape (64, 3) representing the colors for each patch.
    - filename (str): The name of the output PLY file.
    """
    # Convert tensors to numpy arrays
    neighborhoods_np = neighborhoods.cpu().numpy()

    # Initialize an empty list to hold all points and their corresponding colors
    all_points = []
    all_colors = []

    # Loop through each patch and its neighborhood
    for i in range(centers.shape[0]):
        patch_points = neighborhoods_np[i]  # (32, 3)
        patch_color = np.tile(colors[i], (patch_points.shape[0], 1))  # Repeat the color 32 times

        all_points.append(patch_points)
        all_colors.append(patch_color)

    # Convert the lists to numpy arrays
    all_points = np.vstack(all_points)  # (64*32, 3)
    all_colors = np.vstack(all_colors)  # (64*32, 3)

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    # Save the point cloud to a PLY file
    o3d.io.write_point_cloud(filename, pcd)

    print(f"Point cloud saved as {filename} with each patch and its neighbors colored uniquely.")

