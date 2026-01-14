import os
import time
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.data_processor import DataProcessor
from models.adaptive_ms_pinn import AdaptiveMSPINN, PINNTrainer
from models.baseline_models import LSTMModel, GRUModel, TransformerModel, CNNLSTMModel, BaselineTrainer

class ModelTrainer:
    """模型训练和评估主类"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None
        self.trainer = None
        self.model_type = None  # 添加模型类型属性
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_data_loss': [],
            'train_physics_loss': []
        }
    
    def prepare_data(self, data_path):
        """准备数据"""
        self.processor = DataProcessor(self.config)
        data = self.processor.load_data(data_path)
        features, target = self.processor.preprocess_data(data)
        
        # 提取时空特征
        X, y = self.processor.extract_spatiotemporal_features(
            features, target, self.config['seq_len']
        )
        
        # 划分数据集
        X_train, X_val, X_test, y_train, y_val, y_test = self.processor.split_dataset(X, y)
        
        # 创建数据加载器
        self.train_loader, self.val_loader, self.test_loader = self.processor.create_dataloaders(
            X_train, X_val, X_test, y_train, y_val, y_test, self.config['batch_size']
        )
        
        # 更新输入维度
        self.config['input_dim'] = X.shape[-1]
        
        return X_train.shape[-1]
    
    def build_model(self, model_type):
        """构建模型"""
        self.model_type = model_type  # 保存模型类型
        
        if model_type == 'adaptive_ms_pinn':
            self.model = AdaptiveMSPINN(self.config)
            self.trainer = PINNTrainer(self.model, self.config)
        elif model_type == 'lstm':
            self.model = LSTMModel(self.config)
            self.trainer = BaselineTrainer(self.model, self.config)
        elif model_type == 'gru':
            self.model = GRUModel(self.config)
            self.trainer = BaselineTrainer(self.model, self.config)
        elif model_type == 'transformer':
            self.model = TransformerModel(self.config)
            self.trainer = BaselineTrainer(self.model, self.config)
        elif model_type == 'cnn_lstm':
            self.model = CNNLSTMModel(self.config)
            self.trainer = BaselineTrainer(self.model, self.config)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        print(f"{model_type}模型构建完成，参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def train(self, epochs, patience=10):
        """训练模型"""
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练阶段
            total_train_loss = 0
            total_data_loss = 0
            total_physics_loss = 0
            
            for batch_idx, (x, y) in enumerate(self.train_loader):
                if isinstance(self.trainer, PINNTrainer):
                    train_loss, data_loss, physics_loss = self.trainer.train_step(x, y)
                    total_data_loss += data_loss
                    total_physics_loss += physics_loss
                else:
                    train_loss = self.trainer.train_step(x, y)
                    data_loss = train_loss
                    physics_loss = 0
                
                total_train_loss += train_loss
            
            avg_train_loss = total_train_loss / len(self.train_loader)
            avg_data_loss = total_data_loss / len(self.train_loader)
            avg_physics_loss = total_physics_loss / len(self.train_loader)
            
            # 验证阶段
            total_val_loss = 0
            if len(self.val_loader) > 0:
                for batch_idx, (x, y) in enumerate(self.val_loader):
                    val_loss = self.trainer.validate_step(x, y)
                    total_val_loss += val_loss
                
                avg_val_loss = total_val_loss / len(self.val_loader)
            else:
                avg_val_loss = avg_train_loss  # 如果没有验证集，使用训练损失作为验证损失
            
            # 记录历史
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['train_data_loss'].append(avg_data_loss)
            self.history['train_physics_loss'].append(avg_physics_loss)
            
            # 打印训练信息
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}, "
                  f"Time: {epoch_time:.2f}s")
            
            # 早停机制（仅在有验证集时生效）
            if len(self.val_loader) > 0:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    early_stop_counter = 0
                    # 保存最佳模型
                    self.save_model()
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print(f"早停机制触发，在第{epoch+1}轮停止训练")
                        break
    
    def evaluate(self):
        """评估模型"""
        # 加载最佳模型
        self.load_model()
        
        # 测试模型
        total_test_loss = 0
        all_preds = []
        all_labels = []
        
        if len(self.test_loader) > 0:
            for batch_idx, (x, y) in enumerate(self.test_loader):
                test_loss, y_pred, y_true = self.trainer.test_step(x, y)
                total_test_loss += test_loss
                
                # 收集预测结果和真实标签
                all_preds.append(y_pred.cpu().numpy())
                all_labels.append(y_true.cpu().numpy())
            
            avg_test_loss = total_test_loss / len(self.test_loader)
        else:
            avg_test_loss = 0.0
            print("警告: 测试集为空，跳过测试评估")
        
        # 合并结果
        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
        else:
            all_preds = np.array([])
            all_labels = np.array([])
            print("警告: 没有预测结果，返回空数组")
        
        # 反归一化
        if len(all_preds) > 0:
            all_preds = self.processor.inverse_transform(all_preds)
            all_labels = self.processor.inverse_transform(all_labels)
            # 计算评估指标
            mae = mean_absolute_error(all_labels, all_preds)
            mse = mean_squared_error(all_labels, all_preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(all_labels, all_preds)
        else:
            mae = float('nan')
            mse = float('nan')
            rmse = float('nan')
            r2 = float('nan')
            print("警告: 没有预测结果，返回空指标")
        
        print(f"测试结果:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        
        # 保存评估结果
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'test_loss': avg_test_loss
        }
        
        return metrics, all_preds, all_labels
    
    def save_model(self):
        """保存模型"""
        model_dir = os.path.join('models', 'saved')
        os.makedirs(model_dir, exist_ok=True)
        
        # 使用模型类型作为文件名的一部分，确保每个模型保存为独立文件
        model_path = os.path.join(model_dir, f"best_model_{self.model_type}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'history': self.history
        }, model_path)
        
        print(f"模型已保存到: {model_path}")
    
    def load_model(self):
        """加载模型"""
        model_dir = os.path.join('models', 'saved')
        # 使用模型类型加载对应的模型文件
        model_path = os.path.join(model_dir, f"best_model_{self.model_type}.pt")
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.history = checkpoint['history']
                print(f"模型已从: {model_path} 加载")
            except RuntimeError as e:
                print(f"模型状态字典不匹配，跳过加载: {e}")
                print("将使用新初始化的模型继续训练")
        else:
            print(f"警告: 未找到模型文件: {model_path}")
    
    def get_history(self):
        """获取训练历史"""
        return self.history
    
    def run_experiment(self, data_path, model_type, epochs=100, patience=10):
        """运行完整实验"""
        print(f"开始实验: {model_type}")
        
        # 准备数据
        self.prepare_data(data_path)
        
        # 构建模型
        self.build_model(model_type)
        
        # 训练模型
        self.train(epochs, patience)
        
        # 评估模型
        metrics, preds, labels = self.evaluate()
        
        return metrics, preds, labels, self.history

def run_all_models(config, data_path, epochs=100):
    """运行所有模型进行对比"""
    model_types = ['lstm', 'gru', 'transformer', 'cnn_lstm', 'adaptive_ms_pinn']
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"运行模型: {model_type}")
        print(f"{'='*50}")
        
        trainer = ModelTrainer(config.copy())
        
        # 根据配置决定数据源
        if 'data_sources' in config and config['data_sources']:
            # 使用多数据源配置
            metrics, preds, labels, history = trainer.run_experiment(
                config['data_sources'], model_type, epochs
            )
        else:
            # 使用单数据源
            metrics, preds, labels, history = trainer.run_experiment(
                data_path, model_type, epochs
            )
        
        results[model_type] = {
            'metrics': metrics,
            'predictions': preds,
            'labels': labels,
            'history': history
        }
    
    # 比较结果
    print(f"\n{'='*50}")
    print("模型性能对比")
    print(f"{'='*50}")
    print(f"{'模型':<20} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print(f"{'='*60}")
    
    for model_type, result in results.items():
        m = result['metrics']
        print(f"{model_type:<20} {m['mse']:<10.6f} {m['rmse']:<10.6f} {m['mae']:<10.6f} {m['r2']:<10.6f}")
    
    return results

if __name__ == "__main__":
    # 测试训练器
    config = {
        'seq_len': 24,
        'batch_size': 32,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'd_model': 128,
        'nhead': 4,
        'num_encoder_layers': 2,
        'num_adaptive_layers': 3,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'alpha': 0.1
    }
    
    trainer = ModelTrainer(config)
    print("模型训练器初始化完成")
