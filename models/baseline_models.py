import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class LSTMModel(nn.Module):
    """LSTM模型"""
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc(out)
        return out
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

class GRUModel(nn.Module):
    """GRU模型"""
    def __init__(self, config):
        super(GRUModel, self).__init__()
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc(out)
        return out
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

class TransformerModel(nn.Module):
    """Transformer模型"""
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.input_dim = config['input_dim']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_encoder_layers = config['num_encoder_layers']
        self.dropout = config['dropout']
        
        # 输入嵌入
        self.embedding = nn.Linear(self.input_dim, self.d_model)
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, self.d_model))
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_encoder_layers)
        
        # 输出层
        self.fc = nn.Linear(self.d_model, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        seq_len = x.size(1)
        
        # 嵌入层
        x = self.embedding(x)
        
        # 添加位置编码
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer编码
        out = self.transformer_encoder(x)
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 输出层
        out = self.fc(out)
        
        return out
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

class CNNLSTMModel(nn.Module):
    """CNN-LSTM混合模型"""
    def __init__(self, config):
        super(CNNLSTMModel, self).__init__()
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # CNN层用于特征提取
        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1)
        )
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # 输出层
        self.fc = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        
        # CNN特征提取
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # (batch_size, new_seq_len, 128)
        
        # LSTM处理
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        
        # 输出层
        out = self.fc(out)
        
        return out
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

class BaselineTrainer:
    """基线模型训练器"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        
        x, y = x.to(self.device), y.to(self.device)
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate_step(self, x, y):
        self.model.eval()
        with torch.no_grad():
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
        return loss.item()
    
    def test_step(self, x, y):
        self.model.eval()
        with torch.no_grad():
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
        return loss.item(), y_pred, y

if __name__ == "__main__":
    # 测试基线模型
    config = {
        'input_dim': 10,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'd_model': 128,
        'nhead': 4,
        'num_encoder_layers': 2,
        'lr': 0.001,
        'weight_decay': 0.0001
    }
    
    models = {
        'LSTM': LSTMModel(config),
        'GRU': GRUModel(config),
        'Transformer': TransformerModel(config),
        'CNN-LSTM': CNNLSTMModel(config)
    }
    
    for name, model in models.items():
        print(f"{name}模型初始化完成")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
