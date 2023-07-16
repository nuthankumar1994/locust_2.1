import torch.nn as nn

from ..registry import HEADS
from ..utils import ConvModule
# from .bbox_head_count1 import BBoxHead
from .bbox_head import BBoxHead


@HEADS.register_module
class ConvFCBBoxHeadImgLevel(BBoxHead):
    """More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_count_convs=0,                       #Fixme: +
                 num_count_fcs=0,                         #Fixme: +
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHeadImgLevel, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs > 0)
        if num_cls_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_count:                                          #Fixme: +
            assert num_count_convs == 0 and num_count_fcs == 0           #Fixme: +

        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_count_convs = num_count_convs                       #Fixme: +
        self.num_count_fcs = num_count_fcs                           #Fixme: +
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add count_score specific branch                           # Fixme: +
        self.count_convs, self.count_fcs, self.count_last_dim = \
            self._add_conv_fc_branch(
                self.num_count_convs, self.num_count_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
                self.count_last_dim *= self.roi_feat_area          #Fixme: +


        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
            self.fc_count = nn.Linear(self.count_last_dim, self.num_counts)        #Fixme: +  Stage1 has 4 Classes; Stage2 has 16 Classes; Stage3 has 64 Classes;
            print("convf_bbox_head.py/ConvFCBBoxHeadImgLevel()", self.num_classes,self.num_counts)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= 1 # Fixme: image level distribution; self.roi_feat_area (7*7)  ==> 1         last_layer_dim==256*7*7
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHeadImgLevel, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.count_fcs]:  #Fixme: +  Resized binary count +self.count_fcs
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        #print("forward()==>",self.num_shared_convs, self.num_shared_fcs, self.in_channels)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = nn.functional.adaptive_avg_pool2d(x, (1,1))
            #print("forward()==>",x.shape)
            x = x.view(x.size(0), -1)
            #print("forward()==>",x.shape)
            for fc in self.shared_fcs:
                #print(fc)
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_count = x               #Fixme: +

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.count_convs:                                     # Fixme: +
            x_count = conv(x_count)
        if x_count.dim() > 2:
            if self.with_avg_pool:
                x_count = self.avg_pool(x_count)
            x_count = x_count.view(x_count.size(0), -1)
        for fc in self.count_fcs:
            x_count = self.relu(fc(x_count))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        count_score = self.fc_count(x_count) if self.with_cls else None   #Fixme: +
        return cls_score, count_score                        # Fixme: +



@HEADS.register_module
class SharedFCBBoxHeadImgLevel(ConvFCBBoxHeadImgLevel):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCBBoxHeadImgLevel, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_count_convs=0,
            num_count_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)