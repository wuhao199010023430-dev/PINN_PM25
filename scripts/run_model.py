import argparse
import yaml
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.trainer import run_all_models
from utils.visualization import Visualizer

def main():
    """命令行接口主函数"""
    parser = argparse.ArgumentParser(description='PM2.5预测模型 - 自适应多尺度PINN框架')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, required=False,
                        help='数据文件路径（CSV格式），如果配置文件中定义了数据源则此参数可选')
    parser.add_argument('--config_path', type=str, default='config.yaml',
                        help='配置文件路径（默认：config.yaml）')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['lstm', 'gru', 'transformer', 'cnn_lstm', 'adaptive_ms_pinn', 'all'],
                        help='模型类型（使用配置文件值）')
    
    # 训练参数 - 默认值为None，表示使用配置文件中的值
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮次（使用配置文件值）')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批量大小（使用配置文件值）')
    parser.add_argument('--seq_len', type=int, default=None,
                        help='序列长度（使用配置文件值）')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率（使用配置文件值）')
    
    # 可视化参数
    parser.add_argument('--visualize', action='store_true',
                        help='是否生成可视化结果（默认：False）')
    parser.add_argument('--save_figures', action='store_true',
                        help='是否保存可视化图片（默认：False）')
    parser.add_argument('--generate_report', action='store_true',
                        help='是否生成性能报告（默认：False）')
    
    args = parser.parse_args()
    
    # 检查数据源配置
    if not args.data_path and not os.path.exists(args.config_path):
        parser.error("必须提供 --data_path 参数或有效的 --config_path 配置文件")
    
    # 加载配置文件
    config = {}
    if os.path.exists(args.config_path):
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # 扁平化配置参数（将层级化的配置合并到顶层）
    if 'data' in config:
        config.update(config['data'])
    if 'model' in config:
        config.update(config['model'])
    if 'training' in config:
        config.update(config['training'])
    
    # 如果配置文件中定义了数据源，则不需要data_path参数
    if 'data_sources' in config and config['data_sources']:
        print(f"使用多数据源配置: {list(config['data_sources'].keys())}")
        # 设置一个虚拟数据路径，因为trainer需要这个参数
        if not args.data_path:
            args.data_path = 'multi_source_config'
    elif not args.data_path:
        parser.error("必须提供 --data_path 参数，或在配置文件中定义 data_sources")
    
    # 更新配置 - 只有在命令行中明确指定参数时才覆盖配置文件中的值
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.seq_len is not None:
        config['seq_len'] = args.seq_len
    if args.lr is not None:
        config['lr'] = args.lr
    if args.model_type is not None:
        config['model_type'] = args.model_type
    else:
        # 如果命令行未指定模型类型，使用配置文件中的值
        config['model_type'] = config.get('model_type', 'all')
    
    # 获取最终使用的参数值（用于显示）
    final_epochs = config.get('epochs', 100)
    final_batch_size = config.get('batch_size', 32)
    final_seq_len = config.get('seq_len', 24)
    final_lr = config.get('lr', 0.001)
    final_model_type = config.get('model_type', 'all')
    
    print("="*60)
    print("PM2.5预测模型 - 自适应多尺度PINN框架")
    print("="*60)
    print(f"数据路径: {args.data_path}")
    print(f"模型类型: {final_model_type}")
    print(f"训练轮次: {final_epochs}")
    print(f"批量大小: {final_batch_size}")
    print(f"序列长度: {final_seq_len}")
    print(f"学习率: {final_lr}")
    print("="*60)
    
    # 运行模型
    if final_model_type == 'all':
        # 运行所有模型
        results = run_all_models(config, args.data_path, final_epochs)
    else:
        # 运行单个模型
        from utils.trainer import ModelTrainer
        trainer = ModelTrainer(config.copy())
        
        # 根据配置决定数据源
        if 'data_sources' in config and config['data_sources']:
            # 使用多数据源配置
            metrics, preds, labels, history = trainer.run_experiment(
                config['data_sources'], final_model_type, final_epochs
            )
        else:
            # 使用单数据源
            metrics, preds, labels, history = trainer.run_experiment(
                args.data_path, final_model_type, final_epochs
            )
        
        results = {
            final_model_type: {
                'metrics': metrics,
                'predictions': preds,
                'labels': labels,
                'history': history
            }
        }
    
    # 可视化结果
    if args.visualize or args.save_figures or args.generate_report:
        visualizer = Visualizer()
        
        # 绘制每个模型的训练曲线和预测结果
        for model_name, result in results.items():
            # 绘制训练曲线
            visualizer.plot_training_curve(result['history'], model_name)
            
            # 绘制预测结果对比
            visualizer.plot_prediction_comparison(
                result['labels'], result['predictions'], model_name
            )
            
            # 绘制散点图
            visualizer.plot_scatter_comparison(
                result['labels'], result['predictions'], model_name
            )
            
            # 绘制误差分布
            visualizer.plot_error_distribution(
                result['labels'], result['predictions'], model_name
            )
        
        # 绘制模型对比图
        if len(results) > 1:
            visualizer.plot_model_comparison(results)
            visualizer.plot_metric_radar(results)
        
        # 生成性能报告 - 每当可视化被启用时都生成报告
        visualizer.generate_report(results)
    
    print("="*60)
    print("模型运行完成！")
    print("结果文件已保存到:")
    print(f"  - 模型文件: models/saved/")
    print(f"  - 可视化结果: figures/")
    print("="*60)

if __name__ == "__main__":
    main()
