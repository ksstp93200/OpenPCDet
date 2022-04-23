import torch
import torch.nn as nn
from torch.nn import functional as F
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
import math


class MultiFrameROIHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = self.model_cfg.ROI_GRID_POOL.IN_CHANNEL * GRID_SIZE * GRID_SIZE
        
        representation_size = 1024
        self.embed_dim = 64
        self.groups = 16
        self.feat_dim = representation_size

        self.stage = 3
        
        # attention
        # self.base_num = cfg.MODEL.VID.RPN.REF_POST_NMS_TOP_N
        # self.advanced_num = int(self.base_num * cfg.MODEL.VID.MEGA.RATIO)

        fcs, Wgs, Wqs, Wks, Wvs, us = [], [], [], [], [], []

        for i in range(self.stage):
            r_size = pre_channel if i == 0 else representation_size
            fcs.append(self.make_fc(r_size, representation_size))
            Wgs.append(nn.Conv2d(self.embed_dim, self.groups, kernel_size=1, stride=1, padding=0))
            Wqs.append(self.make_fc(self.feat_dim, self.feat_dim))
            Wks.append(self.make_fc(self.feat_dim, self.feat_dim))
            Wvs.append(nn.Conv2d(self.feat_dim * self.groups, self.feat_dim, kernel_size=1, stride=1, padding=0, groups=self.groups))
            us.append(nn.Parameter(torch.Tensor(self.groups, 1, self.embed_dim)))
            for l in [Wgs[i], Wvs[i]]:
                torch.nn.init.normal_(l.weight, std=0.01)
                torch.nn.init.constant_(l.bias, 0)
            for weight in [us[i]]:
                torch.nn.init.normal_(weight, std=0.01)

        self.l_fcs = nn.ModuleList(fcs)
        self.l_Wgs = nn.ModuleList(Wgs)
        self.l_Wqs = nn.ModuleList(Wqs)
        self.l_Wks = nn.ModuleList(Wks)
        self.l_Wvs = nn.ModuleList(Wvs)
        self.l_us = nn.ParameterList(us)
        
        pre_channel = self.feat_dim
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        
        self.init_weights()
    
    def make_fc(self, dim_in, hidden_dim):
        fc = nn.Linear(dim_in, hidden_dim)
        nn.init.kaiming_uniform_(fc.weight, a=1)
        nn.init.constant_(fc.bias, 0)
        return fc

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def roi_grid_pool(self, batch_dict, key_frame=False):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                spatial_features_2d: (B, C, H, W)
        Returns:

        """
        batch_size = batch_dict['batch_size'] if key_frame else batch_dict['batxh_frame_size']
        rois = batch_dict['rois_keyframe'].detach() if key_frame else batch_dict['rois'].detach()
        spatial_features_2d = batch_dict['spatial_features_2d'].detach()[batch_dict['key_frame_id']] if key_frame else batch_dict['spatial_features_2d'].detach()
        height, width = spatial_features_2d.size(2), spatial_features_2d.size(3)

        dataset_cfg = batch_dict['dataset_cfg']
        min_x = dataset_cfg.POINT_CLOUD_RANGE[0]
        min_y = dataset_cfg.POINT_CLOUD_RANGE[1]
        voxel_size_x = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[0]
        voxel_size_y = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[1]
        down_sample_ratio = self.model_cfg.ROI_GRID_POOL.DOWNSAMPLE_RATIO

        pooled_features_list = []
        torch.backends.cudnn.enabled = False
        for b_id in range(batch_size):
            # Map global boxes coordinates to feature map coordinates
            x1 = (rois[b_id, :, 0] - rois[b_id, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            x2 = (rois[b_id, :, 0] + rois[b_id, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            y1 = (rois[b_id, :, 1] - rois[b_id, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
            y2 = (rois[b_id, :, 1] + rois[b_id, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)

            angle, _ = common_utils.check_numpy_to_torch(rois[b_id, :, 6])

            cosa = torch.cos(angle)
            sina = torch.sin(angle)

            theta = torch.stack((
                (x2 - x1) / (width - 1) * cosa, (x2 - x1) / (width - 1) * (-sina), (x1 + x2 - width + 1) / (width - 1),
                (y2 - y1) / (height - 1) * sina, (y2 - y1) / (height - 1) * cosa, (y1 + y2 - height + 1) / (height - 1)
            ), dim=1).view(-1, 2, 3).float()

            grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
            grid = nn.functional.affine_grid(
                theta,
                torch.Size((rois.size(1), spatial_features_2d.size(1), grid_size, grid_size))
            )

            pooled_features = nn.functional.grid_sample(
                spatial_features_2d[b_id].unsqueeze(0).expand(rois.size(1), spatial_features_2d.size(1), height, width),
                grid
            )

            pooled_features_list.append(pooled_features)

        torch.backends.cudnn.enabled = True
        pooled_features = torch.cat(pooled_features_list, dim=0)

        return pooled_features
    
    @staticmethod
    def extract_position_embedding(position_mat, feat_dim, wave_length=1000.0):
        device = position_mat.device
        # position_mat, [num_rois, num_nongt_rois, 4]
        feat_range = torch.arange(0, feat_dim / 8, device=device)
        dim_mat = torch.full((len(feat_range),), wave_length, device=device).pow(8.0 / feat_dim * feat_range)
        dim_mat = dim_mat.view(1, 1, 1, -1).expand(*position_mat.shape, -1)

        position_mat = position_mat.unsqueeze(3).expand(-1, -1, -1, dim_mat.shape[3])
        position_mat = position_mat * 100.0

        div_mat = position_mat / dim_mat
        sin_mat, cos_mat = div_mat.sin(), div_mat.cos()

        # [num_rois, num_nongt_rois, 4, feat_dim / 4]
        embedding = torch.cat([sin_mat, cos_mat], dim=3)
        # [num_rois, num_nongt_rois, feat_dim]
        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], embedding.shape[2] * embedding.shape[3])

        return embedding

    @staticmethod
    def extract_position_matrix(bbox, ref_bbox):
        bbox_width_ref = ref_bbox[..., 3]
        bbox_height_ref = ref_bbox[..., 4]
        center_x_ref = ref_bbox[..., 0]
        center_y_ref = ref_bbox[..., 1]

        bbox_width = bbox[..., 3]
        bbox_height = bbox[..., 4]
        center_x = bbox[..., 0]
        center_y = bbox[..., 1]

        delta_x = center_x - center_x_ref.transpose(0, 1)
        delta_x = delta_x / bbox_width
        delta_x = (delta_x.abs() + 1e-3).log()

        delta_y = center_y - center_y_ref.transpose(0, 1)
        delta_y = delta_y / bbox_height
        delta_y = (delta_y.abs() + 1e-3).log()

        delta_width = bbox_width / bbox_width_ref.transpose(0, 1)
        delta_width = delta_width.log()

        delta_height = bbox_height / bbox_height_ref.transpose(0, 1)
        delta_height = delta_height.log()

        position_matrix = torch.stack([delta_x, delta_y, delta_width, delta_height], dim=2)

        return position_matrix
    
    def multihead_attention(self, roi_feat, ref_feat, position_embedding,
                            feat_dim=1024, dim=(1024, 1024, 1024), group=16,
                            index=0, ver="local"):
        if ver in ("local", "memory"):
            Wgs, Wqs, Wks, Wvs, us = self.l_Wgs, self.l_Wqs, self.l_Wks, self.l_Wvs, self.l_us
        # else:
        #     assert position_embedding is None
        #     Wqs, Wks, Wvs, us = self.g_Wqs, self.g_Wks, self.g_Wvs, self.g_us

        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)

        # position_embedding, [1, emb_dim, num_rois, num_nongt_rois]
        # -> position_feat_1, [1, group, num_rois, nongt_dim]
        if position_embedding is not None:
            position_feat_1 = F.relu(Wgs[index](position_embedding))
            # aff_weight, [num_rois, group, num_nongt_rois, 1]
            aff_weight = position_feat_1.permute(2, 1, 3, 0)
            # aff_weight, [num_rois, group, num_nongt_rois]
            aff_weight = aff_weight.squeeze(3)

        # multi head
        assert dim[0] == dim[1]

        q_data = Wqs[index](roi_feat)
        q_data_batch = q_data.reshape(-1, group, int(dim_group[0]))
        # q_data_batch, [group, num_rois, dim_group[0]]
        q_data_batch = q_data_batch.permute(1, 0, 2)

        k_data = Wks[index](ref_feat)
        k_data_batch = k_data.reshape(-1, group, int(dim_group[1]))
        # k_data_batch, [group, num_nongt_rois, dim_group[1]]
        k_data_batch = k_data_batch.permute(1, 0, 2)

        # v_data, [num_nongt_rois, feat_dim]
        v_data = ref_feat

        # aff_a, [group, num_rois, num_nongt_rois]
        aff_a = torch.bmm(q_data_batch, k_data_batch.transpose(1, 2))

        # aff_c, [group, 1, num_nongt_rois]
        aff_c = torch.bmm(us[index], k_data_batch.transpose(1, 2))

        # aff = aff_a + aff_b + aff_c + aff_d
        aff = aff_a + aff_c

        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        # aff_scale, [num_rois, group, num_nongt_rois]
        aff_scale = aff_scale.permute(1, 0, 2)

        # weighted_aff, [num_rois, group, num_nongt_rois]
        if position_embedding is not None:
            weighted_aff = (aff_weight + 1e-6).log() + aff_scale
        else:
            weighted_aff = aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=2)

        aff_softmax_reshape = aff_softmax.reshape(aff_softmax.shape[0] * aff_softmax.shape[1], aff_softmax.shape[2])

        # output_t, [num_rois * group, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)
        # output_t, [num_rois, group * feat_dim, 1, 1]
        output_t = output_t.reshape(-1, group * feat_dim, 1, 1)
        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = Wvs[index](output_t)

        output = linear_out.squeeze(3).squeeze(2)
        
        roi_feat = roi_feat + output

        return output
    
    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois_keyframe'] = targets_dict['rois']
            batch_dict['roi_labels_keyframe'] = targets_dict['roi_labels']

        # ROI: B, N, C
        # RoI aware pooling
        B, N = batch_dict['rois'].shape[:2]
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, C, 7, 7)
        pooled_features = pooled_features.view(B, N, -1)  # (B, N, C*7*7)
        keyframe_features = self.roi_grid_pool(batch_dict, keyframe=True)
        keyframe_features = keyframe_features.view(-1, N, pooled_features.shape[-1])
        pooled_features = F.relu(self.l_fcs[0](pooled_features))
        keyframe_features = F.relu(self.l_fcs[0](keyframe_features))
        batch_rois = torch.split(batch_dict['rois'], batch_dict['local_frame_count'] + 1, dim=0)
        batch_rois_feat = torch.split(pooled_features, batch_dict['local_frame_count'] + 1, dim=0)
        batch_key_rois_feat = []
        for b in range(B):
            cur_roi, local_roi = batch_dict['rois_keyframe'][b], batch_rois[b].reshape(batch_rois[b].shape[0]*batch_rois[b].shape[1], -1)
            cur_feat, local_feat = keyframe_features[b], batch_rois_feat[b].reshape(batch_rois_feat[b].shape[0]*batch_rois_feat[b].shape[1], -1)
            
            local_cache = []
            
            local_cache.append({"rois_cur": torch.cat([cur_roi, local_roi], dim=0),
                                "rois_ref": local_roi,
                                "feats_cur": torch.cat([cur_feat, local_feat], dim=0),
                                "feats_ref": local_feat})
            for _ in range(self.stage - 2):
                local_cache.append({"rois_cur": torch.cat([cur_roi, local_roi], dim=0),
                                    "rois_ref": local_roi})
            local_cache.append({"rois_cur": cur_roi,
                                "rois_ref": local_roi})
            
        
            for i in range(self.stage):
                rois_cur = local_cache[i].pop("rois_cur")
                rois_ref = local_cache[i].pop("rois_ref")
                feats_cur = local_cache[i].pop("feats_cur")
                feats_ref = local_cache[i].pop("feats_ref")

                position_embedding = self.cal_position_embedding(rois_cur, rois_ref)
                attention = self.multihead_attention(feats_cur, feats_ref, position_embedding, index=i, ver="local")
                feats_cur  = feats_cur + attention
                if i != self.stage - 1:
                    feats_cur = F.relu(self.l_fcs[i + 1](feats_cur))
                if i == self.stage - 1:
                    batch_key_rois_feat.append(feats_cur)         
                elif i == self.stage - 2:
                    local_cache[i + 1]["feats_cur"] = feats_cur[:cur_roi.shape[0]]
                    local_cache[i + 1]["feats_ref"] = feats_cur[cur_roi.shape[0]:]
                else:
                    local_cache[i + 1]["feats_cur"] = feats_cur
                    local_cache[i + 1]["feats_ref"] = feats_cur[cur_roi.shape[0]:]
            # for i in range(self.global_res_stage):
            #     attention = self.multihead_attention(x, global_feat, None, index=i, ver="global")
            #     x = x + attention
        batch_key_rois_feat = torch.cat(batch_key_rois_feat, dim=0)
        # Box Refinement
        batch_size_rcnn = batch_key_rois_feat.shape[0] * batch_key_rois_feat.shape[1]
        shared_features = self.shared_fc_layer(batch_key_rois_feat.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1) 
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1) 

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
