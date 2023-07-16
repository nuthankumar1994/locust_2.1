from .bbox_head import BBoxHead
from .convfc_bbox_head import ConvFCBBoxHead, SharedFCBBoxHead
from .convfc_image_head import ConvFCBBoxHeadImgLevel, SharedFCBBoxHeadImgLevel   # Fixme: + image level guided
from .double_bbox_head import DoubleConvFCBBoxHead

__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'SharedFCBBoxHead', 'DoubleConvFCBBoxHead', 'ConvFCBBoxHeadImgLevel', 'SharedFCBBoxHeadImgLevel'
]

# Add ConvFCBBoxHeadImgLevel in __init__.py  2020/03/27
