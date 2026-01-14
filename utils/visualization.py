import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

class Visualizer:
    """可视化工具类"""
    def __init__(self, save_dir='figures'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 设置中文字体 - 增强兼容性，添加更多系统通用字体
        plt.rcParams['font.sans-serif'] = [
            'Microsoft YaHei',      # Windows 常用中文字体
            'SimHei',               # 黑体
            'Arial Unicode MS',     # Mac 常用中文字体
            'DejaVu Sans',          # 回退字体
            'WenQuanYi Micro Hei',  # Linux 常用中文字体
            'Heiti TC'              # 繁体中文支持
        ]
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置样式
        sns.set_style('whitegrid')
        sns.set_palette('husl')
    
    def plot_training_curve(self, history, model_name='model'):
        """绘制训练和验证损失曲线"""
        plt.figure(figsize=(12, 6))
        
        # 绘制训练和验证损失
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        
        # 如果是PINN模型，还绘制数据损失和物理损失
        if history['train_data_loss'] and history['train_physics_loss']:
            plt.plot(epochs, history['train_data_loss'], 'g--', label='Data Loss')
            plt.plot(epochs, history['train_physics_loss'], 'y--', label='Physics Loss')
        
        plt.title(f'{model_name} Training Convergence Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图片
        save_path = os.path.join(self.save_dir, f'{model_name}_training_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线已保存到: {save_path}")
    
    def plot_prediction_comparison(self, y_true, y_pred, model_name='model', samples=100):
        """绘制预测结果与真实值对比"""
        plt.figure(figsize=(12, 6))
        
        # 只绘制部分样本，避免过于密集
        if len(y_true) > samples:
            indices = np.linspace(0, len(y_true) - 1, samples, dtype=int)
            y_true_plot = y_true[indices]
            y_pred_plot = y_pred[indices]
        else:
            y_true_plot = y_true
            y_pred_plot = y_pred
        
        # 绘制真实值和预测值
        plt.plot(y_true_plot, 'b-', label='True Values', alpha=0.8)
        plt.plot(y_pred_plot, 'r-', label='Predicted Values', alpha=0.8)
        
        plt.title(f'{model_name} Prediction Comparison')
        plt.xlabel('Sample Index')
        plt.ylabel('PM2.5 Concentration')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图片
        save_path = os.path.join(self.save_dir, f'{model_name}_prediction_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"预测对比图已保存到: {save_path}")
    
    def plot_scatter_comparison(self, y_true, y_pred, model_name='model'):
        """绘制真实值与预测值的散点图"""
        plt.figure(figsize=(8, 8))
        
        # 计算R²值
        from sklearn.metrics import r2_score
        if len(y_true) > 0 and len(y_pred) > 0:
            r2 = r2_score(y_true, y_pred)
        else:
            r2 = float('nan')
            print("警告: 没有真实值和预测值，无法计算R²")
        
        # 绘制散点图
        plt.scatter(y_true, y_pred, alpha=0.6, s=50)
        
        # 绘制对角线
        if len(y_true) > 0 and len(y_pred) > 0:
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
        else:
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
            print("警告: 没有真实值和预测值，使用默认对角线")
        
        plt.title(f'{model_name} True vs Predicted (R² = {r2:.4f})')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图片
        save_path = os.path.join(self.save_dir, f'{model_name}_scatter_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"散点对比图已保存到: {save_path}")
    
    def plot_error_distribution(self, y_true, y_pred, model_name='model'):
        """绘制预测误差分布"""
        plt.figure(figsize=(10, 6))
        
        # 计算误差
        errors = y_pred - y_true
        
        # 绘制直方图
        sns.histplot(errors, bins=50, kde=True, alpha=0.7)
        
        plt.title(f'{model_name} Prediction Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图片
        save_path = os.path.join(self.save_dir, f'{model_name}_error_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"误差分布图已保存到: {save_path}")
    
    def plot_model_comparison(self, results):
        """绘制不同模型的性能对比"""
        # 准备数据
        model_names = list(results.keys())
        metrics = ['mse', 'rmse', 'mae', 'r2']
        metric_names = ['MSE', 'RMSE', 'MAE', 'R²']
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            values = [results[model]['metrics'][metric] for model in model_names]
            
            # 绘制柱状图
            bars = axes[i].bar(model_names, values, alpha=0.8)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height, 
                            f'{height:.4f}', ha='center', va='bottom')
            
            axes[i].set_title(f'不同模型 {metric_name} 对比')
            axes[i].set_xlabel('模型')
            axes[i].set_ylabel(metric_name)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.save_dir, 'model_performance_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"模型性能对比图已保存到: {save_path}")
    
    def plot_metric_radar(self, results):
        """绘制模型性能雷达图"""
        # 准备数据
        model_names = list(results.keys())
        metrics = ['mse', 'rmse', 'mae', 'r2']
        metric_names = ['MSE', 'RMSE', 'MAE', 'R²']
        
        # 归一化数据（R²需要特殊处理，因为它的范围是[-∞, 1]）
        data = []
        for model in model_names:
            m = results[model]['metrics']
            # 对于R²，我们使用 (r2 + 1) / 2 进行归一化到[0, 1]
            normalized = [
                1 / (1 + m['mse']),  # MSE越小越好，取倒数归一化
                1 / (1 + m['rmse']),  # RMSE越小越好，取倒数归一化
                1 / (1 + m['mae']),  # MAE越小越好，取倒数归一化
                (m['r2'] + 1) / 2  # R²归一化到[0, 1]
            ]
            data.append(normalized)
        
        # 绘制雷达图
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        # 绘制每个模型
        for i, (model_name, values) in enumerate(zip(model_names, data)):
            values += values[:1]  # 闭合
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
            ax.fill(angles, values, alpha=0.25)
        
        # 添加标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1)
        ax.set_title('模型性能雷达图', size=15, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # 保存图片
        save_path = os.path.join(self.save_dir, 'model_performance_radar.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"模型性能雷达图已保存到: {save_path}")
    
    def create_animation(self, y_true, y_preds, model_names, save_path='prediction_animation.gif'):
        """创建预测结果动画"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 准备数据
        frames = len(model_names)
        samples = min(100, len(y_true))
        indices = np.linspace(0, len(y_true) - 1, samples, dtype=int)
        y_true_plot = y_true[indices]
        
        # 初始化绘图
        line_true, = ax.plot(y_true_plot, 'b-', label='真实值')
        line_pred, = ax.plot([], [], 'r-', label='预测值')
        title = ax.set_title('')
        
        ax.set_xlabel('样本索引')
        ax.set_ylabel('PM2.5浓度')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        def init():
            line_pred.set_data([], [])
            return line_pred, title
        
        def animate(i):
            model_name = model_names[i]
            y_pred = y_preds[i]
            y_pred_plot = y_pred[indices]
            
            line_pred.set_data(range(len(y_pred_plot)), y_pred_plot)
            title.set_text(f'{model_name} 预测结果')
            
            # 调整y轴范围
            ax.set_ylim(
                min(np.min(y_true_plot), np.min(y_pred_plot)) * 0.9,
                max(np.max(y_true_plot), np.max(y_pred_plot)) * 1.1
            )
            
            return line_pred, title
        
        # 创建动画
        anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=1000, blit=True)
        
        # 保存动画
        save_path = os.path.join(self.save_dir, save_path)
        anim.save(save_path, writer='pillow', fps=1)
        plt.close()
        
        print(f"预测动画已保存到: {save_path}")
    
    def generate_report(self, results):
        """生成模型性能报告"""
        # 准备数据
        data = []
        for model_name, result in results.items():
            metrics = result['metrics']
            data.append({
                '模型': model_name,
                'MSE': metrics['mse'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2']
            })
        
        df = pd.DataFrame(data)
        
        # 保存为CSV
        save_path = os.path.join(self.save_dir, 'model_performance_report.csv')
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        # 生成HTML报告
        html_content = f"""
        <html>
        <head>
            <title>PM2.5预测模型性能报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .figure {{ margin: 30px 0; text-align: center; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; padding: 10px; }}
            </style>
        </head>
        <body>
            <h1>PM2.5预测模型性能报告</h1>
            
            <h2>性能指标对比</h2>
            {df.to_html(index=False, classes='table table-striped')}
            
            <h2>可视化结果</h2>
            
            <div class="figure">
                <h3>模型性能对比</h3>
                <img src="model_performance_comparison.png" alt="模型性能对比">
            </div>
            
            <div class="figure">
                <h3>模型性能雷达图</h3>
                <img src="model_performance_radar.png" alt="模型性能雷达图">
            </div>
        </body>
        </html>
        """
        
        # 保存HTML报告
        html_path = os.path.join(self.save_dir, 'model_performance_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"性能报告已保存到: {save_path} 和 {html_path}")
        
        return df

if __name__ == "__main__":
    # 测试可视化工具
    visualizer = Visualizer()
    print("可视化工具初始化完成")
