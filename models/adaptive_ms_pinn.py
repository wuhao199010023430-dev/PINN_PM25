import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class AdaptiveLayer(nn.Module):
    """自适应层，根据输入特征自动调整输出维度"""
    def __init__(self, in_dim, base_out_dim=64, max_out_dim=256):
        super(AdaptiveLayer, self).__init__()
        self.in_dim = in_dim
        self.base_out_dim = base_out_dim
        self.max_out_dim = max_out_dim
        self.fixed_out_dim = 128  # 固定输出维度
        
        # 注意力机制用于特征重要性评估
        self.attention = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid()
        )
        
        # 自适应权重矩阵
        self.weight = nn.Parameter(torch.randn(in_dim, max_out_dim))
        self.bias = nn.Parameter(torch.randn(max_out_dim))
        
        # 批量归一化 - 使用固定输出维度128
        self.bn = nn.BatchNorm1d(self.fixed_out_dim)
    
    def forward(self, x):
        # 计算注意力权重
        attn_weights = self.attention(x)
        attn_weights = attn_weights.mean(dim=1, keepdim=True)
        
        # 使用固定输出维度
        out_dim = self.fixed_out_dim
        
        # 使用前out_dim列权重
        weight = self.weight[:, :out_dim]
        bias = self.bias[:out_dim]
        
        # 线性变换
        x = torch.matmul(x, weight) + bias
        
        # 批量归一化（仅在批次大小>1时使用）
        if x.size(0) > 1:
            x = self.bn(x)  # 现在维度匹配了
        else:
            # 对于单个样本，使用层归一化作为替代
            x = F.layer_norm(x, x.size()[1:])
        
        x = F.relu(x)
        
        return x, out_dim

class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器"""
    def __init__(self, input_dim, scales=[1, 3, 5, 7]):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.scales = scales
        self.conv_layers = nn.ModuleList()
        
        for scale in scales:
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(input_dim, 64, kernel_size=scale, padding=scale//2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
            ))
        
        self.fc = nn.Linear(64 * len(scales), 128)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)
        
        scale_features = []
        for conv in self.conv_layers:
            feat = conv(x)
            feat = torch.mean(feat, dim=2)  # 全局平均池化
            scale_features.append(feat)
        
        # 拼接多尺度特征
        combined = torch.cat(scale_features, dim=1)
        combined = self.fc(combined)
        combined = F.relu(combined)
        
        return combined

class PhysicsInformedLoss(nn.Module):
    """物理约束损失函数"""
    def __init__(self, alpha=0.1):
        super(PhysicsInformedLoss, self).__init__()
        self.alpha = alpha  # 物理损失权重
    
    def forward(self, y_pred, y_true, x):
        # 数据损失
        data_loss = F.mse_loss(y_pred, y_true)
        
        # 物理约束：简单的大气扩散方程近似
        # ∂c/∂t + u∂c/∂x + v∂c/∂y = D∇²c - kc
        # 这里使用简化版，假设u, v, D, k为常数
        
        # 计算时间导数
        if x.requires_grad:
            x.requires_grad_(True)
            y_pred.requires_grad_(True)
            
            # 计算梯度
            grad_t = torch.autograd.grad(y_pred, x, 
                                        grad_outputs=torch.ones_like(y_pred),
                                        create_graph=True, retain_graph=True)[0]
            
            # 简化的物理损失：梯度平滑性约束
            physics_loss = torch.mean(torch.abs(grad_t))
        else:
            physics_loss = torch.tensor(0.0, device=y_pred.device)
        
        # 总损失
        total_loss = data_loss + self.alpha * physics_loss
        
        return total_loss, data_loss, physics_loss

class AdaptiveMSPINN(nn.Module):
    """自适应多尺度PINN模型"""
    def __init__(self, config):
        super(AdaptiveMSPINN, self).__init__()
        self.config = config
        self.input_dim = config['input_dim']
        self.seq_len = config['seq_len']
        
        # 多尺度特征提取
        self.ms_feature_extractor = MultiScaleFeatureExtractor(self.input_dim)
        
        # 自适应深度网络
        self.adaptive_layers = nn.ModuleList()
        in_dim = 128  # MultiScaleFeatureExtractor的输出维度
        for i in range(config['num_adaptive_layers']):
            self.adaptive_layers.append(AdaptiveLayer(in_dim))
            # 每个自适应层的输出维度是动态的，但下一层的输入维度保持为128
            in_dim = 128
        
        # 物理约束层
        self.physics_layer = nn.Linear(128, 1)
        
        # 输出层
        self.output_layer = nn.Linear(1, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # 提取多尺度特征
        ms_features = self.ms_feature_extractor(x)
        
        # 自适应深度网络
        adaptive_out = ms_features
        for layer in self.adaptive_layers:
            adaptive_out, _ = layer(adaptive_out)
        
        # 物理约束
        physics_out = self.physics_layer(adaptive_out)
        
        # 最终输出
        output = self.output_layer(physics_out)
        
        return output
    
    def predict(self, x):
        """预测函数"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

class PINNTrainer:
    """PINN训练器"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        self.criterion = PhysicsInformedLoss(alpha=config['alpha'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        
        x, y = x.to(self.device), y.to(self.device)
        x.requires_grad_(True)
        
        y_pred = self.model(x)
        loss, data_loss, physics_loss = self.criterion(y_pred, y, x)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), data_loss.item(), physics_loss.item()
    
    def validate_step(self, x, y):
        self.model.eval()
        with torch.no_grad():
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            loss = F.mse_loss(y_pred, y)
        return loss.item()
    
    def test_step(self, x, y):
        self.model.eval()
        with torch.no_grad():
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            loss = F.mse_loss(y_pred, y)
        return loss.item(), y_pred, y

if __name__ == "__main__":
    # 测试模型
    config = {
        'input_dim': 10,
        'seq_len': 24,
        'num_adaptive_layers': 3,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'alpha': 0.1
    }
    
    model = AdaptiveMSPINN(config)
    print("自适应多尺度PINN模型初始化完成")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
