# spatialnet/cli.py
"""命令行接口模块"""

import argparse
import os
from .preprocessing import preprocess_data
from .training import train_model
from .resulting import generate_results

def main():
    """主命令行入口点"""
    # 创建主解析器
    parser = argparse.ArgumentParser(
        description="SpatialNet: 空间转录组异构网络分析工具包",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 添加全局选项
    parser.add_argument("--id", help="项目标识符")
    parser.add_argument("--source-dir", help="源数据目录或文件")
    parser.add_argument("--data-dir", help="处理数据存放目录")
    parser.add_argument("--deg", help="是否取差异表达基因", default=True)
    parser.add_argument("--prop", help="基因表达比例阈值", default=0.1)
    parser.add_argument("--celltype", help="细胞类型列名")
    parser.add_argument("--model_dir", help="模型存放目录") 
    parser.add_argument("--result_dir", help="结果输出目录")
    parser.add_argument("--spatial", type=int, default=1, help="是否包含空间信息 (0=否, 1=是)")
    parser.add_argument("--TF", type=int, default=0, help="是否使用转录因子 (0=否, 1=是)")
    parser.add_argument("--train_count", type=int, default=0, help="训练次数")
    parser.add_argument("--device", type=str, default='cuda:0', help="GPU设备ID")
    parser.add_argument("--start_step", type=int, default=1, choices=[1, 2, 3], 
                        help="开始步骤 (1=预处理, 2=训练, 3=结果)")
    
    args = parser.parse_args()
    
    # 检查必要参数
    required_args = ["id", "source_dir", "data_dir", "celltype", "model_dir", "result_dir"]
    missing_args = [arg for arg in required_args if getattr(args, arg) is None]
    
    if missing_args:
        parser.error(f"以下必需参数缺失: {', '.join('--' + arg.replace('_', '-') for arg in missing_args)}")
    
    # 设置LR参数
    LR = 0 if args.TF == 1 else 1
    
    # 创建必要的目录
    for dir_path in [args.data_dir, args.model_dir, args.result_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 执行流程
    if args.start_step <= 1:
        print("执行预处理...")
        preprocess_data(
            id=args.id,
            source_dir=args.source_dir, 
            data_dir=args.data_dir,
            deg=args.deg,
            prop=args.prop,
            celltype=args.celltype,
            LR=LR,
            spatial=args.spatial,
            TF=args.TF
        )
    
    if args.start_step <= 2:
        print("执行训练...")
        train_model(
            id=args.id,
            preprocess_dir=args.data_dir,
            model_path=args.model_dir,
            spatial=args.spatial,
            TF=args.TF,
            train_count=args.train_count,
            device=args.device
        )
    
    if args.start_step <= 3:
        print("生成结果...")
        generate_results(
            id=args.id,
            model_dir=args.model_dir,
            data_dir=args.data_dir,
            result_dir=args.result_dir,
            train_count=args.train_count,
            TF=args.TF,
            cutoff=args.cutoff
        )
        
    print("✅ 所有步骤完成!")

if __name__ == "__main__":
    main()