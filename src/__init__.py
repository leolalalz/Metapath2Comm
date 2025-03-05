"""
SpatialNet - 空间转录组异构网络分析工具包。

此包提供了预处理、训练和结果生成的完整工作流程。
"""

__version__ = "0.1.0"
__author__ = "Xu Chenle"
__email__ = "xuchenle@big.ac.cn"

from .preprocessing import preprocess_data
from .training import train_model
from .resulting import generate_results