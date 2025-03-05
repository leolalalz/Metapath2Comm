#!/bin/bash
# filepath: enhanced-pipeline.sh

export PYTHONWARNINGS="ignore::UserWarning:torch_geometric.typing"

# 设置默认参数
DEFAULT_SPATIAL=1
DEFAULT_TF=0
DEFAULT_TRAIN_COUNT=0
DEFAULT_START_STEP=1
DEFAULT_GPU_ID=0
DEFAULT_LOGGING_DIR="./logs"

# 显示帮助信息
show_help() {
  cat << EOF
使用方法: $(basename $0) [选项]

必需参数:
  --id ID                 项目标识符
  --source_dir DIR        源数据目录或文件
  --data_dir DIR          处理数据存放目录
  --celltype COL          细胞类型列名
  --model_dir DIR         模型存放目录
  --result_dir DIR        结果输出目录

可选参数:
  --spatial INT           是否包含空间信息 (0=否, 1=是) [默认: ${DEFAULT_SPATIAL}]
  --tf INT                是否使用转录因子 (0=否, 1=是) [默认: ${DEFAULT_TF}]
  --train_count INT       训练次数 [默认: ${DEFAULT_TRAIN_COUNT}]
  --start_step INT        开始步骤 (1=预处理, 2=训练, 3=结果) [默认: ${DEFAULT_START_STEP}]
  --gpu INT               GPU设备ID [默认: ${DEFAULT_GPU_ID}]
  --logging_dir DIR       日志目录 [默认: ${DEFAULT_LOGGING_DIR}]
  --log                   启用日志记录 (自动创建日志文件)
  -h, --help              显示此帮助信息

示例:
  $(basename $0) --id lung --source_dir ../data/adata/lung_st_adata.h5ad --data_dir ../case/preprocess/226/  \\
                --celltype celltype --model_dir ../case/model/226/ --result_dir ../case/result/226/ \\
                --spatial 1 --tf 0 --train_count 1 --start_step 1 --log

EOF
}

# 解析命令行参数
parse_arguments() {
  # 检查是否提供了任何参数
  if [ $# -eq 0 ]; then
    show_help
    exit 1
  fi

  # 解析参数
  while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
      --id)
        ID="$2"
        shift 2
        ;;
      --source_dir)
        SOURCE_DIR="$2"
        shift 2
        ;;
      --data_dir)
        DATA_DIR="$2"
        shift 2
        ;;
      --celltype)
        CELLTYPE="$2"
        shift 2
        ;;
      --model_dir)
        MODEL_DIR="$2"
        shift 2
        ;;
      --result_dir)
        RESULT_DIR="$2"
        shift 2
        ;;
      --spatial)
        SPATIAL="$2"
        shift 2
        ;;
      --tf)
        TF="$2"
        shift 2
        ;;
      --train_count)
        TRAIN_COUNT="$2"
        shift 2
        ;;
      --start_step)
        START_STEP="$2"
        shift 2
        ;;
      --gpu)
        GPU_ID="$2"
        shift 2
        ;;
      --logging_dir)
        LOGGING_DIR="$2"
        shift 2
        ;;
      --log)
        ENABLE_LOGGING=1
        shift
        ;;
      -h|--help)
        show_help
        exit 0
        ;;
      *)
        echo "错误: 未知参数 '$1'"
        show_help
        exit 1
        ;;
    esac
  done
}

# 验证参数
validate_parameters() {
  # 检查必需参数
  MISSING_PARAMS=()
  [ -z "$ID" ] && MISSING_PARAMS+=("--id")
  [ -z "$SOURCE_DIR" ] && MISSING_PARAMS+=("--source_dir")
  [ -z "$DATA_DIR" ] && MISSING_PARAMS+=("--data_dir")
  [ -z "$CELLTYPE" ] && MISSING_PARAMS+=("--celltype")
  [ -z "$MODEL_DIR" ] && MISSING_PARAMS+=("--model_dir")
  [ -z "$RESULT_DIR" ] && MISSING_PARAMS+=("--result_dir")
  
  if [ ${#MISSING_PARAMS[@]} -gt 0 ]; then
    echo "错误: 缺少以下必需参数:"
    for param in "${MISSING_PARAMS[@]}"; do
      echo "  $param"
    done
    exit 1
  fi
  
  # 设置默认值
  SPATIAL=${SPATIAL:-$DEFAULT_SPATIAL}
  TF=${TF:-$DEFAULT_TF}
  TRAIN_COUNT=${TRAIN_COUNT:-$DEFAULT_TRAIN_COUNT}
  START_STEP=${START_STEP:-$DEFAULT_START_STEP}
  GPU_ID=${GPU_ID:-$DEFAULT_GPU_ID}
  LOGGING_DIR=${LOGGING_DIR:-$DEFAULT_LOGGING_DIR}
  
  # 检查数值参数的有效性
  is_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
  }
  
  is_valid_boolean() {
    [[ "$1" -eq 0 || "$1" -eq 1 ]]
  }
  
  if ! is_integer "$START_STEP" || [ "$START_STEP" -lt 1 ] || [ "$START_STEP" -gt 3 ]; then
    echo "错误: START_STEP 必须是 1、2 或 3 的整数。"
    exit 1
  fi
  
  if ! is_integer "$TRAIN_COUNT" || [ "$TRAIN_COUNT" -lt 0 ]; then
    echo "错误: TRAIN_COUNT 必须是非负整数。"
    exit 1
  fi
  
  if ! is_valid_boolean "$SPATIAL"; then
    echo "错误: SPATIAL 必须是 0 或 1。"
    exit 1
  fi
  
  if ! is_valid_boolean "$TF"; then
    echo "错误: TF 必须是 0 或 1。"
    exit 1
  fi
  
  # 如果设置了TF，为LR赋予适当的值
  if [ "$TF" -eq 1 ]; then
    LR=0
  else
    LR=1
  fi
  
  # 检查目录是否存在，如果不存在则创建
  create_directory_if_not_exists() {
    if [ ! -d "$1" ]; then
      echo "警告: 目录 '$1' 不存在，正在创建..."
      mkdir -p "$1"
    fi
  }
  
  create_directory_if_not_exists "$DATA_DIR"
  create_directory_if_not_exists "$MODEL_DIR"
  create_directory_if_not_exists "$RESULT_DIR"
  
  # 如果启用了日志，则创建日志目录
  if [ "${ENABLE_LOGGING:-0}" -eq 1 ]; then
    create_directory_if_not_exists "$LOGGING_DIR"
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$LOGGING_DIR/${ID}_${TIMESTAMP}.log"
  fi
}

# 执行命令并处理错误
run_command() {
  local cmd="$1"
  local step_name="$2"
  
  echo "执行 $step_name..."
  echo "$ $cmd"
  
  # 执行命令
  eval "$cmd"
  local exit_code=$?
  
  if [ $exit_code -eq 0 ]; then
    echo "✓ $step_name 成功完成"
    return 0
  else
    echo "✗ $step_name 失败 (退出代码: $exit_code)"
    exit $exit_code
  fi
}

# 主函数
main() {
  # 解析命令行参数
  parse_arguments "$@"
  
  # 验证参数
  validate_parameters
  
  # 选择是否启用日志记录
  if [ "${ENABLE_LOGGING:-0}" -eq 1 ]; then
    echo "日志将写入: $LOG_FILE"
    # 重定向所有输出到日志文件，同时显示在终端
    exec > >(tee -a "$LOG_FILE") 2>&1
  fi
  
  # 显示运行配置
  echo "========== 运行配置 =========="
  echo "ID: $ID"
  echo "源目录: $SOURCE_DIR"
  echo "数据目录: $DATA_DIR"
  echo "细胞类型: $CELLTYPE"
  echo "模型目录: $MODEL_DIR"
  echo "结果目录: $RESULT_DIR"
  echo "空间信息: $SPATIAL"
  echo "转录因子: $TF"
  echo "训练次数: $TRAIN_COUNT"
  echo "开始步骤: $START_STEP"
  echo "GPU 设备: cuda:$GPU_ID"
  echo "=============================="
  
  # 记录开始时间
  START_TIME=$(date +%s)
  
  # 运行预处理
  if [ "$START_STEP" -le 1 ]; then
    CMD="python preprocessing.py --id $ID --source_dir $SOURCE_DIR --data_dir $DATA_DIR --celltype $CELLTYPE --LR $LR --spatial $SPATIAL --TF $TF"
    run_command "$CMD" "预处理"
  else
    echo "跳过预处理步骤"
  fi
  
  # 运行训练
  if [ "$START_STEP" -le 2 ]; then
    CMD="python training.py --id $ID --data_dir $DATA_DIR --model_dir $MODEL_DIR --spatial $SPATIAL --TF $TF --train_count $TRAIN_COUNT --device cuda:$GPU_ID"
    run_command "$CMD" "训练"
  else
    echo "跳过训练步骤"
  fi
  
  # 运行结果生成
  if [ "$START_STEP" -le 3 ]; then
    CMD="python resulting.py --id $ID --model_dir $MODEL_DIR --data_dir $DATA_DIR --result_dir $RESULT_DIR --train_count $TRAIN_COUNT --TF $TF"
    run_command "$CMD" "结果生成"
  else
    echo "跳过结果生成步骤"
  fi
  
  # 计算运行时间
  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))
  HOURS=$((DURATION / 3600))
  MINUTES=$(( (DURATION % 3600) / 60 ))
  SECONDS=$((DURATION % 60))
  
  echo "=============================="
  echo "✅ 所有步骤完成!"
  echo "总运行时间: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
  echo "=============================="
}

# 执行主函数
main "$@"