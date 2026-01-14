import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureInteractionLayer(nn.Module):
    """特征交互层
    
    用于捕捉不同特征之间的交互关系，支持多种交互方式
    
    Args:
        input_dim (int): 输入特征维度
        hidden_dim (int, optional): 隐藏层维度，默认为64
        interaction_type (str, optional): 交互类型，可选值：'cross'（交叉特征）、'polynomial'（多项式特征）、'attention'（注意力机制）
    """
    
    def __init__(self, input_dim, hidden_dim=64, interaction_type='cross'):
        """初始化特征交互层
        
        Args:
            input_dim (int): 输入特征维度
            hidden_dim (int, optional): 隐藏层维度
            interaction_type (str, optional): 交互类型
        """
        super(FeatureInteractionLayer, self).__init__()
        self.interaction_type = interaction_type
        self.input_dim = input_dim
        
        if interaction_type == 'cross':
            # 交叉特征层
            self.cross_weights = nn.Parameter(torch.randn(input_dim, input_dim))
            self.cross_bias = nn.Parameter(torch.zeros(input_dim))
        elif interaction_type == 'polynomial':
            # 多项式特征层
            self.poly_weights = nn.Linear(input_dim, hidden_dim)
            self.poly_activation = nn.ReLU()
            self.poly_out = nn.Linear(hidden_dim, input_dim)
        elif interaction_type == 'attention':
            # 注意力机制
            self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(input_dim)
        else:
            raise ValueError(f"不支持的交互类型: {interaction_type}")
        
        logger.info(f"特征交互层初始化完成，类型：{interaction_type}，输入维度：{input_dim}")
    
    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入特征，形状为 (batch_size, seq_len, feature_dim) 或 (batch_size, feature_dim)
        
        Returns:
            torch.Tensor: 交互后的特征
        """
        original_x = x
        
        if self.interaction_type == 'cross':
            # 交叉特征计算: x * (x * W + b)
            if x.dim() == 3:  # 时序数据
                # (batch_size, seq_len, feature_dim) * (feature_dim, feature_dim) -> (batch_size, seq_len, feature_dim)
                cross_out = torch.matmul(x, self.cross_weights) + self.cross_bias
                cross_out = x * cross_out
            else:  # 非时序数据
                cross_out = torch.matmul(x, self.cross_weights) + self.cross_bias
                cross_out = x * cross_out
            return cross_out + original_x  # 残差连接
        
        elif self.interaction_type == 'polynomial':
            # 多项式特征计算: relu(x * W1 + b1) * W2 + b2
            if x.dim() == 3:
                # 时序数据：先展平，再恢复形状
                batch_size, seq_len, feature_dim = x.shape
                x_flat = x.view(-1, feature_dim)
                poly_out = self.poly_out(self.poly_activation(self.poly_weights(x_flat)))
                poly_out = poly_out.view(batch_size, seq_len, feature_dim)
            else:
                poly_out = self.poly_out(self.poly_activation(self.poly_weights(x)))
            return poly_out + original_x  # 残差连接
        
        elif self.interaction_type == 'attention':
            # 注意力机制
            if x.dim() == 2:
                # 添加序列维度
                x = x.unsqueeze(1)
            
            attn_out, _ = self.attention(x, x, x)
            attn_out = self.norm(attn_out + x)  # 残差连接 + 层归一化
            
            if original_x.dim() == 2:
                # 恢复原始形状
                attn_out = attn_out.squeeze(1)
            
            return attn_out


class MultiSourceFusionLayer(nn.Module):
    """多源特征融合层
    
    用于融合来自不同数据源的特征，支持灵活的数据来源数量
    
    Args:
        source_dims (dict): 各数据源的特征维度映射
            例如：{'meteorological': 10, 'spatial': 5, 'social': 8}
        fusion_dim (int, optional): 融合后的特征维度，默认为128
        fusion_type (str, optional): 融合类型，可选值：'concat'（拼接）、'attention'（自注意力）、'cross_attention'（交叉注意力）
    """
    
    def __init__(self, source_dims, fusion_dim=128, fusion_type='attention'):
        """初始化多源特征融合层
        
        Args:
            source_dims (dict): 各数据源的特征维度映射
            fusion_dim (int, optional): 融合后的特征维度
            fusion_type (str, optional): 融合类型
        """
        super(MultiSourceFusionLayer, self).__init__()
        self.source_dims = source_dims
        self.fusion_dim = fusion_dim
        self.fusion_type = fusion_type
        self.num_sources = len(source_dims)
        
        # 为每个数据源创建投影层
        self.source_projections = nn.ModuleDict()
        for source_name, dim in source_dims.items():
            self.source_projections[source_name] = nn.Linear(dim, fusion_dim)
        
        if fusion_type == 'attention':
            # 自注意力融合
            self.self_attention = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=4, batch_first=True)
            self.attention_norm = nn.LayerNorm(fusion_dim)
        elif fusion_type == 'cross_attention':
            # 交叉注意力融合
            self.cross_attention = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=4, batch_first=True)
            self.cross_norm = nn.LayerNorm(fusion_dim)
        elif fusion_type == 'concat':
            # 拼接融合
            total_dim = fusion_dim * self.num_sources
            self.concat_linear = nn.Linear(total_dim, fusion_dim)
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")
        
        self.output_linear = nn.Linear(fusion_dim, fusion_dim)
        self.activation = nn.ReLU()
        
        logger.info(f"多源特征融合层初始化完成，类型：{fusion_type}，数据源数量：{self.num_sources}")
    
    def forward(self, source_features):
        """前向传播
        
        Args:
            source_features (dict): 各数据源的特征映射
                例如：{
                    'meteorological': tensor(batch_size, seq_len, feature_dim),
                    'spatial': tensor(batch_size, seq_len, feature_dim)
                }
        
        Returns:
            torch.Tensor: 融合后的特征，形状为 (batch_size, seq_len, fusion_dim)
        """
        # 确保所有数据源特征维度一致
        projected_features = []
        for source_name, features in source_features.items():
            # 投影到统一维度
            projection = self.source_projections[source_name]
            projected = projection(features)
            projected_features.append(projected)
        
        if self.fusion_type == 'concat':
            # 拼接所有投影后的特征
            concatenated = torch.cat(projected_features, dim=-1)
            fused = self.concat_linear(concatenated)
        elif self.fusion_type == 'attention':
            # 将所有投影特征堆叠成一个序列
            stacked = torch.stack(projected_features, dim=1)  # (batch_size, num_sources, seq_len, fusion_dim)
            batch_size, num_sources, seq_len, fusion_dim = stacked.shape
            
            # 调整形状以适应注意力机制
            reshaped = stacked.view(batch_size * seq_len, num_sources, fusion_dim)
            
            # 自注意力融合
            attn_out, _ = self.self_attention(reshaped, reshaped, reshaped)
            attn_out = self.attention_norm(attn_out + reshaped)  # 残差连接
            
            # 取平均或最大池化作为融合结果
            fused = torch.mean(attn_out, dim=1).view(batch_size, seq_len, fusion_dim)
        elif self.fusion_type == 'cross_attention':
            # 使用第一个数据源作为查询，其他作为键值
            query = projected_features[0]
            key_value = torch.stack(projected_features[1:], dim=1)
            batch_size, num_sources_minus_1, seq_len, fusion_dim = key_value.shape
            
            # 调整形状
            query = query.view(batch_size * seq_len, 1, fusion_dim)
            key_value_reshaped = key_value.view(batch_size * seq_len, num_sources_minus_1, fusion_dim)
            
            # 交叉注意力
            attn_out, _ = self.cross_attention(query, key_value_reshaped, key_value_reshaped)
            attn_out = self.cross_norm(attn_out + query)
            
            # 恢复形状
            fused = attn_out.view(batch_size, seq_len, fusion_dim)
            
            # 与查询特征融合
            fused = fused + projected_features[0]
        
        # 输出层
        output = self.activation(self.output_linear(fused))
        return output


class SpatiotemporalFeatureExtractor(nn.Module):
    """时空特征提取器
    
    结合空间特征和时间特征，使用CNN+LSTM或Transformer提取时空特征
    
    Args:
        input_dim (int): 输入特征维度
        hidden_dim (int, optional): 隐藏层维度，默认为64
        extractor_type (str, optional): 提取器类型，可选值：'cnn_lstm'（CNN+LSTM）、'transformer'（Transformer）
        seq_len (int, optional): 序列长度，默认为24
    """
    
    def __init__(self, input_dim, hidden_dim=64, extractor_type='cnn_lstm', seq_len=24):
        """初始化时空特征提取器
        
        Args:
            input_dim (int): 输入特征维度
            hidden_dim (int, optional): 隐藏层维度
            extractor_type (str, optional): 提取器类型
            seq_len (int, optional): 序列长度
        """
        super(SpatiotemporalFeatureExtractor, self).__init__()
        self.extractor_type = extractor_type
        self.seq_len = seq_len
        
        if extractor_type == 'cnn_lstm':
            # CNN用于提取空间特征
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, padding=1),
                nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
                nn.ReLU()
            )
            
            # LSTM用于提取时间特征
            self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, 
                               batch_first=True, bidirectional=True)
            
            # 输出层
            self.fc = nn.Linear(hidden_dim * 2, hidden_dim)  # 双向LSTM输出维度翻倍
        elif extractor_type == 'transformer':
            # Transformer用于同时提取时空特征
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True),
                num_layers=2
            )
            self.transformer_norm = nn.LayerNorm(input_dim)
            self.fc = nn.Linear(input_dim, hidden_dim)
        else:
            raise ValueError(f"不支持的提取器类型: {extractor_type}")
        
        logger.info(f"时空特征提取器初始化完成，类型：{extractor_type}，输入维度：{input_dim}")
    
    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入时空特征，形状为 (batch_size, seq_len, feature_dim)
        
        Returns:
            torch.Tensor: 提取后的时空特征
        """
        if self.extractor_type == 'cnn_lstm':
            # CNN需要调整输入形状为 (batch_size, feature_dim, seq_len)
            x_cnn = x.permute(0, 2, 1)  # 调整维度
            cnn_out = self.cnn(x_cnn)
            
            # 调整回LSTM需要的形状 (batch_size, seq_len, feature_dim)
            cnn_out = cnn_out.permute(0, 2, 1)
            
            # LSTM前向传播
            lstm_out, _ = self.lstm(cnn_out)
            
            # 使用最后一个时间步的输出
            lstm_out = lstm_out[:, -1, :]
            
            # 全连接层
            out = self.fc(lstm_out)
        elif self.extractor_type == 'transformer':
            # Transformer前向传播
            transformer_out = self.transformer(x)
            transformer_out = self.transformer_norm(transformer_out)
            
            # 使用最后一个时间步的输出
            transformer_out = transformer_out[:, -1, :]
            
            # 全连接层
            out = self.fc(transformer_out)
        
        return out


class FeatureFusionNetwork(nn.Module):
    """特征融合网络
    
    集成特征交互、多源融合和时空特征提取的完整网络
    
    Args:
        config (dict): 配置字典
            必须包含：
            - source_dims (dict): 各数据源的特征维度
            - interaction_type (str): 特征交互类型
            - fusion_type (str): 融合类型
            - extractor_type (str): 时空特征提取器类型
            - hidden_dim (int): 隐藏层维度
            - seq_len (int): 序列长度
    """
    
    def __init__(self, config):
        """初始化特征融合网络
        
        Args:
            config (dict): 配置字典
        """
        super(FeatureFusionNetwork, self).__init__()
        
        # 从配置中获取参数
        self.source_dims = config['source_dims']
        self.interaction_type = config.get('interaction_type', 'cross')
        self.fusion_type = config.get('fusion_type', 'attention')
        self.extractor_type = config.get('extractor_type', 'cnn_lstm')
        self.hidden_dim = config.get('hidden_dim', 64)
        self.seq_len = config.get('seq_len', 24)
        
        # 为每个数据源创建独立的特征交互层
        self.feature_interaction_layers = nn.ModuleDict()
        for source_name, dim in self.source_dims.items():
            self.feature_interaction_layers[source_name] = FeatureInteractionLayer(
                input_dim=dim,  # 使用每个数据源的实际特征维度
                hidden_dim=self.hidden_dim,
                interaction_type=self.interaction_type
            )
        
        # 多源特征融合层
        self.source_fusion = MultiSourceFusionLayer(
            source_dims=self.source_dims,
            fusion_dim=self.hidden_dim,
            fusion_type=self.fusion_type
        )
        
        # 时空特征提取器
        self.spatiotemporal_extractor = SpatiotemporalFeatureExtractor(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            extractor_type=self.extractor_type,
            seq_len=self.seq_len
        )
        
        logger.info("特征融合网络初始化完成")
    
    def forward(self, source_features):
        """前向传播
        
        Args:
            source_features (dict): 各数据源的特征映射
        
        Returns:
            torch.Tensor: 最终融合后的特征
        """
        # 1. 对每个数据源进行特征交互
        interacted_features = {}
        for source_name, features in source_features.items():
            # 使用对应数据源的特征交互层
            interaction_layer = self.feature_interaction_layers[source_name]
            interacted = interaction_layer(features)
            interacted_features[source_name] = interacted
        
        # 2. 多源特征融合
        fused_features = self.source_fusion(interacted_features)
        
        # 3. 时空特征提取
        final_features = self.spatiotemporal_extractor(fused_features)
        
        return final_features


if __name__ == "__main__":
    # 测试特征融合模块
    config = {
        'source_dims': {
            'meteorological': 10,
            'spatial': 5
        },
        'interaction_type': 'cross',
        'fusion_type': 'attention',
        'extractor_type': 'cnn_lstm',
        'hidden_dim': 64,
        'seq_len': 24
    }
    
    # 创建测试数据
    batch_size = 32
    seq_len = 24
    
    test_data = {
        'meteorological': torch.randn(batch_size, seq_len, 10),
        'spatial': torch.randn(batch_size, seq_len, 5)
    }
    
    # 初始化网络
    fusion_net = FeatureFusionNetwork(config)
    
    # 前向传播
    output = fusion_net(test_data)
    logger.info(f"测试完成，输出形状：{output.shape}")
