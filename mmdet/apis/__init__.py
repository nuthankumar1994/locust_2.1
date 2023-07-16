from .env import get_root_logger, init_dist, set_random_seed
#这里的all变量包含了所有引入的变量，下面两句比较重要：
from .inference import (inference_detector, init_detector, show_result,
                        show_result_pyplot)
from .train import train_detector

__all__ = [
    'init_dist', 'get_root_logger', 'set_random_seed', 'train_detector',
    'init_detector', 'inference_detector', 'show_result', 'show_result_pyplot'
]
