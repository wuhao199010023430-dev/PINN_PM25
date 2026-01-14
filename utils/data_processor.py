import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PM25Dataset(Dataset):
    """PM2.5预测数据集类
    
    继承自PyTorch的Dataset类，用于将numpy数组转换为PyTorch张量数据集
    
    Args:
        x (numpy.ndarray): 输入特征数据，形状为 (样本数, 序列长度, 特征数) 或 (样本数, 特征数)
        y (numpy.ndarray): 目标数据，形状为 (样本数, 1)
    """
    def __init__(self, x, y):
        """初始化数据集
        
        Args:
            x (numpy.ndarray): 输入特征数据
            y (numpy.ndarray): 目标数据
        """
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        """获取数据集长度
        
        Returns:
            int: 数据集中的样本数量
        """
        return len(self.x)
    
    def __getitem__(self, idx):
        """根据索引获取样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (输入特征张量, 目标张量)
        """
        return self.x[idx], self.y[idx]

class DataProcessor:
    """PM2.5预测数据处理器
    
    负责多源数据融合、特征工程、数据预处理、数据集划分和数据加载器创建
    支持移动走航数据、激光雷达数据、小微站数据等多种数据源
    
    Args:
        config (dict): 配置字典，包含以下关键字：
            - seq_len (int): 序列长度
            - batch_size (int): 批量大小
            - feature_scaler (str): 特征归一化方法，'minmax'或'standard'
            - target_scaler (str): 目标归一化方法，'minmax'或'standard'
            - time_resolution (str): 目标时间分辨率，如'1H'（1小时）、'30T'（30分钟）
            - target_latitude (float): 目标位置纬度
            - target_longitude (float): 目标位置经度
    """
    
    def __init__(self, config):
        """初始化数据处理器
        
        Args:
            config (dict): 配置字典
        """
        self.config = config
        self.seq_len = config.get('seq_len', 24)
        self.batch_size = config.get('batch_size', 32)
        
        # 时间和空间配置
        self.time_resolution = config.get('time_resolution', '1h')  # 目标时间分辨率，使用'h'而非'H'避免警告
        # 转换 deprecated 的时间格式
        if 'H' in self.time_resolution:
            self.time_resolution = self.time_resolution.replace('H', 'h')
        self.target_latitude = config.get('target_latitude', 39.9042)  # 默认目标位置（北京）
        self.target_longitude = config.get('target_longitude', 116.4074)
        
        # 选择归一化方法
        scaler_type = {'minmax': MinMaxScaler, 'standard': StandardScaler}
        self.scaler_x = scaler_type.get(config.get('feature_scaler', 'minmax'))()
        self.scaler_y = scaler_type.get(config.get('target_scaler', 'minmax'))()
        
        # 特征工程配置
        self.feature_engineering = config.get('feature_engineering', {})
        self.auto_scale = self.feature_engineering.get('auto_scale', True)
        self.feature_selection = self.feature_engineering.get('feature_selection', True)
        self.temporal_features = self.feature_engineering.get('temporal_features', True)
        self.spatial_features = self.feature_engineering.get('spatial_features', True)
        self.interaction_features = self.feature_engineering.get('interaction_features', True)
        
        # 自适应反馈机制配置
        self.adaptive_feedback = config.get('training', {}).get('adaptive_feedback', True)
        
        # 存储数据和状态
        self.features = None
        self.target = None
        self.data_sources = {}
        self.feature_columns = None
        self.raw_data = None  # 存储原始合并数据
        self.source_count = 0  # 数据源数量
        self.source_types = []  # 数据源类型
        
        logger.info(f"数据处理器初始化完成，序列长度：{self.seq_len}，批量大小：{self.batch_size}")
        logger.info(f"目标时间分辨率：{self.time_resolution}，目标位置：({self.target_latitude}, {self.target_longitude})")
        logger.info(f"特征工程配置：自动缩放={self.auto_scale}，特征选择={self.feature_selection}")
        logger.info(f"自适应反馈机制：{self.adaptive_feedback}")
    
    def load_data(self, file_paths_dict):
        """加载多源数据
        
        支持从多个数据源加载数据，每个数据源对应一个文件路径
        支持的数据源类型：
        - meteorological: 常规气象站数据
        - mobile: 移动走航数据
        - lidar: 激光雷达数据
        - microsite: 小微站数据
        
        Args:
            file_paths_dict (dict or str): 数据源名称到文件路径的映射，或单个文件路径
                例如：{
                    'meteorological': 'data/meteorological.csv',
                    'mobile': 'data/pm25_mobile.csv',
                    'lidar': 'data/pm25_lidar.csv',
                    'microsite': 'data/pm25_microsite.csv'
                }
                或单个文件路径: 'data/pm25_data_template.csv'
        
        Returns:
            pd.DataFrame: 合并后的数据
        """
        # 重置数据源计数和类型
        self.source_count = 0
        self.source_types = []
        
        # 处理字符串路径的情况（单数据源）
        if isinstance(file_paths_dict, str):
            logger.info(f"加载单数据源文件：{file_paths_dict}")
            data = pd.read_csv(file_paths_dict)
            self.raw_data = data.copy()
            self.source_count = 1
            self.source_types = ['meteorological']  # 默认单数据源为气象数据
            logger.info(f"单数据源加载完成，数据形状：{data.shape}")
            return data
        
        # 处理多数据源的情况
        processed_data_sources = {}
        
        # 处理每个数据源
        for source_name, file_path in file_paths_dict.items():
            logger.info(f"加载数据源：{source_name}，文件路径：{file_path}")
            
            # 加载原始数据
            data = pd.read_csv(file_path)
            
            # 根据数据源类型进行预处理
            if source_name == 'mobile':
                processed_data = self._process_mobile_data(data)
            elif source_name == 'lidar':
                processed_data = self._process_lidar_data(data)
            elif source_name == 'microsite':
                processed_data = self._process_microsite_data(data)
            else:  # meteorological, spatial, social等其他数据源
                processed_data = data
            
            processed_data_sources[source_name] = processed_data
            self.data_sources[source_name] = data  # 保存原始数据
        
        # 合并所有处理后的数据
        merged_data = self._merge_data_sources(processed_data_sources)
        
        # 更新数据源计数和类型
        self.source_count = len(processed_data_sources)
        self.source_types = list(processed_data_sources.keys())
        
        logger.info(f"多源数据加载完成，共{self.source_count}个数据源，类型：{self.source_types}")
        logger.info(f"合并后数据形状：{merged_data.shape}")
        
        # 自适应反馈：根据数据源数量调整处理策略
        if self.adaptive_feedback:
            self._adjust_processing_strategy()
        
        self.raw_data = merged_data.copy()
        return merged_data
    
    def _process_mobile_data(self, data):
        """处理移动走航数据
        
        Args:
            data (pd.DataFrame): 原始移动走航数据
            
        Returns:
            pd.DataFrame: 处理后的移动走航数据
        """
        logger.info("处理移动走航数据")
        
        # 转换datetime列
        data['datetime'] = pd.to_datetime(data['datetime'])
        
        # 检查是否包含经纬度信息
        if 'latitude' in data.columns and 'longitude' in data.columns:
            # 计算距离目标位置的距离（用于空间过滤）
            data['distance_to_target'] = self._calculate_distance(
                data['latitude'], data['longitude'],
                self.target_latitude, self.target_longitude
            )
            
            # 过滤距离目标位置较近的数据（5公里以内）
            data = data[data['distance_to_target'] < 5.0]
        else:
            # 如果没有经纬度信息，使用默认距离0（表示在目标位置）
            data['distance_to_target'] = 0.0
            logger.warning("移动走航数据缺少经纬度信息，跳过空间过滤")
        
        return data
    
    def _process_lidar_data(self, data):
        """处理激光雷达数据
        
        Args:
            data (pd.DataFrame): 原始激光雷达数据
            
        Returns:
            pd.DataFrame: 处理后的激光雷达数据
        """
        logger.info("处理激光雷达数据")
        
        # 转换datetime列
        data['datetime'] = pd.to_datetime(data['datetime'])
        
        # 检查是否包含高度信息
        if 'height' in data.columns:
            # 激光雷达数据包含高度剖面，这里只保留地面高度(0米)或近地面数据
            data = data[data['height'] <= 100]  # 只保留100米以下的数据
        else:
            logger.warning("激光雷达数据缺少高度信息，跳过高度过滤")
        
        # 按时间分组，计算垂直方向的平均值
        # 只对存在的列进行聚合
        agg_columns = {}
        if 'backscatter_coeff' in data.columns:
            agg_columns['backscatter_coeff'] = 'mean'
        if 'extinction_coeff' in data.columns:
            agg_columns['extinction_coeff'] = 'mean'
        if 'aod_550' in data.columns:
            agg_columns['aod_550'] = 'mean'
        if 'pm25_est' in data.columns:
            agg_columns['pm25_est'] = 'mean'
        if 'depolarization_ratio' in data.columns:
            agg_columns['depolarization_ratio'] = 'mean'
        if 'signal_to_noise' in data.columns:
            agg_columns['signal_to_noise'] = 'mean'
        
        if agg_columns:
            data = data.groupby('datetime').agg(agg_columns).reset_index()
        else:
            logger.warning("激光雷达数据缺少特定列，跳过聚合处理")
        
        return data
    
    def _process_microsite_data(self, data):
        """处理小微站数据
        
        Args:
            data (pd.DataFrame): 原始小微站数据
            
        Returns:
            pd.DataFrame: 处理后的小微站数据
        """
        logger.info("处理小微站数据")
        
        # 转换datetime列
        data['datetime'] = pd.to_datetime(data['datetime'])
        
        # 检查是否包含站点信息和经纬度
        if 'site_id' in data.columns and 'latitude' in data.columns and 'longitude' in data.columns:
            # 计算每个站点到目标位置的距离
            data['distance_to_target'] = self._calculate_distance(
                data['latitude'], data['longitude'],
                self.target_latitude, self.target_longitude
            )
            
            # 选择距离目标位置最近的3个站点
            closest_sites = data.groupby('site_id')['distance_to_target'].mean().nsmallest(3).index
            data = data[data['site_id'].isin(closest_sites)]
        else:
            # 如果没有站点信息，使用默认距离0
            data['distance_to_target'] = 0.0
            logger.warning("小微站数据缺少站点信息，跳过站点过滤")
        
        # 按时间分组，计算多个站点的平均值
        # 只对存在的列进行聚合
        agg_columns = {}
        if 'pm25' in data.columns:
            agg_columns['pm25'] = 'mean'
        if 'temperature' in data.columns:
            agg_columns['temperature'] = 'mean'
        if 'humidity' in data.columns:
            agg_columns['humidity'] = 'mean'
        if 'pressure' in data.columns:
            agg_columns['pressure'] = 'mean'
        if 'wind_speed' in data.columns:
            agg_columns['wind_speed'] = 'mean'
        if 'wind_direction' in data.columns:
            agg_columns['wind_direction'] = 'mean'
        
        if agg_columns:
            data = data.groupby('datetime').agg(agg_columns).reset_index()
        else:
            logger.warning("小微站数据缺少特定列，跳过聚合处理")
        
        return data
    
    def _adjust_processing_strategy(self):
        """自适应调整数据处理策略
        
        根据数据源数量和类型自动调整：
        1. 特征工程策略
        2. 数据处理参数
        3. 模型配置建议
        """
        logger.info(f"根据{self.source_count}个数据源自动调整处理策略")
        
        # 1. 根据数据源数量调整特征工程策略
        if self.source_count == 1:
            # 单数据源场景：简化特征工程，避免过拟合
            logger.info("单数据源场景：简化特征工程")
            # 保留基本特征
            self.interaction_features = False
            self.rolling_features = False
        elif self.source_count == 2:
            # 双数据源场景：适度增加特征工程
            logger.info("双数据源场景：适度增加特征工程")
            self.interaction_features = True
            self.rolling_features = False
        else:  # 3个及以上数据源
            # 多数据源场景：完整特征工程
            logger.info("多数据源场景：完整特征工程")
            self.interaction_features = True
            self.rolling_features = True
        
        # 2. 根据数据源类型调整处理策略
        if 'mobile' in self.source_types:
            logger.info("包含移动走航数据：增强空间特征处理")
            self.spatial_features = True
        
        if 'lidar' in self.source_types:
            logger.info("包含激光雷达数据：增强垂直剖面特征处理")
            # 这里可以添加激光雷达特有特征处理逻辑
            pass
        
        if 'microsite' in self.source_types:
            logger.info("包含小微站数据：增强多站点融合特征处理")
            # 这里可以添加小微站特有特征处理逻辑
            pass
        
        # 3. 调整时间序列参数
        if self.source_count > 2:
            # 多数据源：增加序列长度，充分利用时间信息
            new_seq_len = min(self.seq_len * 2, 48)  # 最大48小时
            if new_seq_len != self.seq_len:
                logger.info(f"调整序列长度：{self.seq_len} → {new_seq_len}")
                self.seq_len = new_seq_len
        
        logger.info(f"自适应调整完成：")
        logger.info(f"  - 交互特征：{self.interaction_features}")
        logger.info(f"  - 滚动特征：{getattr(self, 'rolling_features', False)}")
        logger.info(f"  - 空间特征：{self.spatial_features}")
        logger.info(f"  - 序列长度：{self.seq_len}")

    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """计算两点之间的距离（单位：公里）
        
        使用Haversine公式计算两点之间的球面距离
        
        Args:
            lat1, lon1: 第一个点的纬度和经度
            lat2, lon2: 第二个点的纬度和经度
            
        Returns:
            float: 两点之间的距离（公里）
        """
        # 转换为弧度
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine公式
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # 地球平均半径（公里）
        
        return c * r
    
    def _merge_data_sources(self, processed_data_sources):
        """合并多个数据源
        
        Args:
            processed_data_sources (dict): 处理后的数据源字典
            
        Returns:
            pd.DataFrame: 合并后的数据
        """
        logger.info("合并多个数据源")
        
        # 获取所有数据源名称和数据
        source_names = list(processed_data_sources.keys())
        data_list = list(processed_data_sources.values())
        
        if not data_list:
            return pd.DataFrame()
        
        # 为每个数据源添加后缀，避免列名冲突
        for i in range(len(data_list)):
            source_name = source_names[i]
            if i > 0:  # 第一个数据源不添加后缀
                data = data_list[i].copy()
                # 保留datetime列不添加后缀
                non_datetime_cols = [col for col in data.columns if col != 'datetime']
                data = data.rename(columns={col: f"{col}_{source_name}" for col in non_datetime_cols})
                data_list[i] = data
        
        # 以第一个数据源为基础
        merged_data = data_list[0].copy()
        merged_data['datetime'] = pd.to_datetime(merged_data['datetime'])
        merged_data.set_index('datetime', inplace=True)
        
        # 合并其他数据源
        for i in range(1, len(data_list)):
            data = data_list[i].copy()
            data['datetime'] = pd.to_datetime(data['datetime'])
            data.set_index('datetime', inplace=True)
            
            # 使用左连接合并，保留基础数据源的时间索引
            merged_data = merged_data.join(data, how='left')
        
        # 重置索引，保留datetime列
        merged_data.reset_index(inplace=True)
        
        return merged_data
    
    def create_domain_interaction_features(self, data):
        """创建基于领域知识的交互特征
        
        基于气象学、环境科学和时空数据分析，创建有意义的交互特征
        
        Args:
            data (pd.DataFrame): 原始数据
            
        Returns:
            pd.DataFrame: 添加了交互特征的数据
        """
        logger.info("开始创建领域知识交互特征")
        
        # 气象交互特征
        if all(col in data.columns for col in ['temperature', 'humidity']):
            # 温湿指数：用于衡量体感舒适度，影响大气扩散条件
            data['temp_humidity_index'] = data['temperature'] * data['humidity'] / 100
            logger.info("创建温湿指数特征")
        
        if all(col in data.columns for col in ['wind_speed', 'pressure']):
            # 风压交互：影响大气扩散能力
            data['wind_pressure_ratio'] = data['wind_speed'] / (data['pressure'] / 1000)
            logger.info("创建风压比值特征")
        
        if all(col in data.columns for col in ['dew_point', 'temperature']):
            # 露点温度差：反映空气中水汽含量，影响污染物形成
            data['dew_temp_diff'] = data['temperature'] - data['dew_point']
            logger.info("创建露点温度差特征")
        
        # 时空交互特征
        if all(col in data.columns for col in ['hour', 'temperature']):
            # 小时-温度交互：反映不同时段的温度变化
            data['hour_temp_interaction'] = data['hour'] * data['temperature']
            logger.info("创建小时-温度交互特征")
        
        if all(col in data.columns for col in ['month', 'pressure']):
            # 月份-气压交互：反映季节变化对气压的影响
            data['month_pressure_interaction'] = data['month'] * data['pressure']
            logger.info("创建月份-气压交互特征")
        
        # 地理交互特征（如果有地理数据）
        if all(col in data.columns for col in ['latitude', 'longitude', 'elevation']):
            # 地理-海拔交互：反映地形对污染物扩散的影响
            data['geo_elevation_factor'] = (data['latitude'] + data['longitude']) * data['elevation'] / 10000
            logger.info("创建地理-海拔交互特征")
        
        # 社会经济交互特征（如果有社会数据）
        if all(col in data.columns for col in ['population_density', 'traffic_flow']):
            # 人口密度-交通流量交互：反映人类活动强度
            data['human_activity_index'] = data['population_density'] * data['traffic_flow'] / 1000
            logger.info("创建人类活动指数特征")
        
        # 移动走航数据交互特征
        if all(col in data.columns for col in ['pm25', 'speed']):
            # PM2.5与移动速度的交互：反映移动走航时的污染物分布
            data['pm25_speed_interaction'] = data['pm25'] / (data['speed'] + 1)  # +1避免除以0
            logger.info("创建PM2.5-速度交互特征")
        
        # 激光雷达数据交互特征
        if all(col in data.columns for col in ['aod_550', 'extinction_coeff']):
            # AOD与消光系数的交互：反映大气垂直方向的污染分布
            data['aod_extinction_interaction'] = data['aod_550'] * data['extinction_coeff']
            logger.info("创建AOD-消光系数交互特征")
        
        if all(col in data.columns for col in ['backscatter_coeff', 'signal_to_noise']):
            # 后向散射系数与信噪比的交互：反映激光雷达信号质量对污染监测的影响
            data['backscatter_snr_interaction'] = data['backscatter_coeff'] * data['signal_to_noise']
            logger.info("创建后向散射-信噪比交互特征")
        
        # 小微站数据交互特征
        if all(col in data.columns for col in ['pm25', 'pressure']):
            # PM2.5与气压的交互：反映气压对地面污染物浓度的影响
            data['pm25_pressure_interaction'] = data['pm25'] * (data['pressure'] / 1000)
            logger.info("创建PM2.5-气压交互特征")
        
        # 多源数据融合交互特征
        if all(col in data.columns for col in ['pm25', 'pm25_est']):
            # 实测PM2.5与激光雷达估算PM2.5的交互：结合实测与遥感数据
            data['pm25_lidar_interaction'] = (data['pm25'] + data['pm25_est']) / 2
            logger.info("创建PM2.5-激光雷达估算交互特征")
        
        logger.info("领域知识交互特征创建完成")
        return data
    
    def preprocess_data(self, data):
        """预处理数据，支持多源数据融合
        
        包括：时间分辨率对齐、缺失值处理、时间特征提取、领域交互特征创建、数据归一化
        
        Args:
            data (pd.DataFrame): 原始数据
            
        Returns:
            tuple: (归一化后的特征, 归一化后的目标)
        """
        logger.info("开始数据预处理")
        
        # 1. 对齐时间分辨率
        logger.info(f"对齐时间分辨率到：{self.time_resolution}")
        data = self._align_time_resolution(data)
        
        # 2. 处理缺失值
        logger.info("处理缺失值")
        data = data.ffill().bfill()
        
        # 3. 提取时间特征
        if 'datetime' in data.columns:
            logger.info("提取时间特征")
            data['datetime'] = pd.to_datetime(data['datetime'])
            data['hour'] = data['datetime'].dt.hour
            data['day'] = data['datetime'].dt.day
            data['month'] = data['datetime'].dt.month
            data['dayofweek'] = data['datetime'].dt.dayofweek
        
        # 4. 创建领域交互特征
        data = self.create_domain_interaction_features(data)
        
        # 5. 选择特征和目标
        logger.info("选择特征和目标变量")
        
        # 确保目标变量存在，处理不同数据源的pm25列名
        target_col = None
        if 'pm25' in data.columns:
            target_col = 'pm25'
        elif 'pm25_est' in data.columns:
            target_col = 'pm25_est'
            # 将pm25_est重命名为pm25，统一目标变量名称
            data = data.rename(columns={'pm25_est': 'pm25'})
        else:
            raise ValueError("数据中缺少pm25或pm25_est目标变量")
        
        # 确定特征列：排除datetime、pm25和非数值列
        all_cols = [col for col in data.columns if col not in ['datetime', 'pm25']]
        # 只保留数值列作为特征
        numeric_cols = data[all_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = numeric_cols
        
        logger.info(f"特征列：{numeric_cols}")
        features = data[numeric_cols].values
        target = data['pm25'].values.reshape(-1, 1)
        
        # 6. 归一化
        logger.info("执行数据归一化")
        self.features = self.scaler_x.fit_transform(features)
        self.target = self.scaler_y.fit_transform(target)
        
        logger.info(f"数据预处理完成，特征形状：{self.features.shape}，目标形状：{self.target.shape}")
        return self.features, self.target
    
    def _align_time_resolution(self, data):
        """对齐数据到目标时间分辨率
        
        Args:
            data (pd.DataFrame): 原始数据
            
        Returns:
            pd.DataFrame: 时间分辨率对齐后的数据
        """
        if 'datetime' not in data.columns:
            logger.warning("数据中缺少datetime列，跳过时间分辨率对齐")
            return data
        
        # 转换datetime列并设置为索引
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)
        
        # 对数值列进行重采样
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("数据中没有数值列，跳过时间分辨率对齐")
            data.reset_index(inplace=True)
            return data
        
        # 对数值列进行重采样（平均值）
        resampled_numeric = data[numeric_cols].resample(self.time_resolution).mean()
        
        # 如果有非数值列，尝试进行重采样（取第一个值）
        if len(non_numeric_cols) > 0:
            resampled_non_numeric = data[non_numeric_cols].resample(self.time_resolution).first()
            resampled_data = resampled_numeric.join(resampled_non_numeric, how='left')
        else:
            resampled_data = resampled_numeric
        
        # 重置索引，保留datetime列
        resampled_data.reset_index(inplace=True)
        
        logger.info(f"时间分辨率对齐完成，原始数据形状：{data.shape}，对齐后数据形状：{resampled_data.shape}")
        
        return resampled_data
    
    def extract_spatiotemporal_features(self, features=None, target=None, seq_len=None):
        """提取时空特征序列
        
        将数据转换为序列格式，用于时序模型训练
        
        Args:
            features (numpy.ndarray, optional): 特征数据，默认为None（使用内部存储的features）
            target (numpy.ndarray, optional): 目标数据，默认为None（使用内部存储的target）
            seq_len (int, optional): 序列长度，默认为None（使用配置中的seq_len）
            
        Returns:
            tuple: (序列特征, 序列目标)
        """
        logger.info("开始提取时空特征序列")
        
        # 使用默认值
        if features is None:
            features = self.features
        if target is None:
            target = self.target
        if seq_len is None:
            seq_len = self.seq_len
        
        # 构建序列数据
        X, y = [], []
        
        # 当数据量小于等于序列长度时，至少生成一个序列
        if len(features) <= seq_len:
            logger.warning(f"数据量({len(features)})小于等于序列长度({seq_len})，仅生成一个序列")
            if len(features) > 0:
                # 使用所有可用数据作为一个序列
                X.append(features[0:len(features)])
                y.append(target[-1])  # 使用最后一个数据作为目标
        else:
            # 正常生成多个序列
            for i in range(len(features) - seq_len + 1):
                X.append(features[i:i+seq_len])
                y.append(target[i+seq_len-1])  # 使用序列最后一个数据作为目标
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"时空特征提取完成，序列形状：{X.shape}，目标形状：{y.shape}")
        return X, y
    
    def split_dataset(self, X, y, test_size=0.2, val_size=0.1):
        """划分训练集、验证集和测试集
        
        Args:
            X (numpy.ndarray): 特征数据
            y (numpy.ndarray): 目标数据
            test_size (float, optional): 测试集比例，默认为0.2
            val_size (float, optional): 验证集比例，默认为0.1
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"划分数据集，测试集比例：{test_size}，验证集比例：{val_size}")
        
        n_samples = len(X)
        
        # 处理样本量不足的情况
        if n_samples <= 1:
            logger.warning(f"样本量不足({n_samples})，无法按比例划分数据集，所有数据作为训练集")
            return X, np.array([]), np.array([]), y, np.array([]), np.array([])
        elif n_samples == 2:
            logger.warning(f"样本量较少({n_samples})，调整划分比例")
            # 1个训练集，1个测试集，无验证集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, random_state=42
            )
            return X_train, np.array([]), X_test, y_train, np.array([]), y_test
        
        try:
            # 正常划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # 处理验证集划分
            if len(X_train) > 1:
                # 从训练集中划分验证集
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=val_size/(1-test_size), random_state=42
                )
            else:
                logger.warning("训练集样本量不足，无法划分验证集")
                X_val, y_val = np.array([]), np.array([])
        except ValueError:
            logger.warning("按比例划分失败，调整为固定划分")
            # 调整划分比例，确保至少有一个样本
            test_size = min(0.5, test_size)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            X_val, y_val = np.array([]), np.array([])
        
        logger.info(f"数据集划分完成：")
        logger.info(f"  训练集：{X_train.shape}, {y_train.shape}")
        logger.info(f"  验证集：{X_val.shape}, {y_val.shape}")
        logger.info(f"  测试集：{X_test.shape}, {y_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_dataloaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=None):
        """创建数据加载器
        
        Args:
            X_train, X_val, X_test (numpy.ndarray): 训练、验证、测试特征数据
            y_train, y_val, y_test (numpy.ndarray): 训练、验证、测试目标数据
            batch_size (int, optional): 批量大小，默认为None（使用配置中的batch_size）
            
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        logger.info(f"创建数据加载器，批量大小：{batch_size}")
        
        # 创建数据集
        train_dataset = PM25Dataset(X_train, y_train)
        val_dataset = PM25Dataset(X_val, y_val)
        test_dataset = PM25Dataset(X_test, y_test)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"数据加载器创建完成：")
        logger.info(f"  训练集加载器：{len(train_loader)}个批次")
        logger.info(f"  验证集加载器：{len(val_loader)}个批次")
        logger.info(f"  测试集加载器：{len(test_loader)}个批次")
        
        return train_loader, val_loader, test_loader
    
    def inverse_transform(self, y_pred):
        """反归一化预测结果
        
        将归一化后的预测结果转换回原始PM2.5浓度单位
        
        Args:
            y_pred (numpy.ndarray): 归一化后的预测结果
            
        Returns:
            numpy.ndarray: 反归一化后的真实PM2.5浓度
        """
        return self.scaler_y.inverse_transform(y_pred)
    
    def get_feature_importance(self):
        """获取特征重要性信息
        
        Returns:
            dict: 特征名称到归一化缩放比例的映射，用于粗略评估特征重要性
        """
        if self.feature_columns is None:
            raise ValueError("尚未执行预处理，无法获取特征重要性")
        
        # 使用归一化器的缩放比例作为特征重要性的粗略估计
        # 注意：这不是真正的特征重要性，仅用于参考
        feature_importance = {}
        for i, col in enumerate(self.feature_columns):
            if hasattr(self.scaler_x, 'scale_') and len(self.scaler_x.scale_) > i:
                # StandardScaler使用scale_属性
                importance = 1 / self.scaler_x.scale_[i] if self.scaler_x.scale_[i] != 0 else 0
            elif hasattr(self.scaler_x, 'data_range_') and len(self.scaler_x.data_range_) > i:
                # MinMaxScaler使用data_range_属性
                importance = self.scaler_x.data_range_[i]
            else:
                importance = 1.0
            feature_importance[col] = importance
        
        return feature_importance

if __name__ == "__main__":
    # 测试数据处理模块
    config = {
        'seq_len': 24,
        'batch_size': 32,
        'feature_scaler': 'minmax',
        'target_scaler': 'minmax'
    }
    
    processor = DataProcessor(config)
    logger.info("数据处理模块初始化完成")
    
    # 测试多源数据加载
    try:
        # 创建示例数据用于测试
        test_data = {
            'datetime': pd.date_range('2023-01-01 00:00:00', periods=100, freq='h'),
            'pm25': np.random.randint(10, 100, size=100),
            'temperature': np.random.randint(0, 35, size=100),
            'humidity': np.random.randint(30, 90, size=100),
            'pressure': np.random.uniform(1000, 1020, size=100),
            'wind_speed': np.random.uniform(0, 10, size=100)
        }
        
        test_df = pd.DataFrame(test_data)
        
        # 保存为临时CSV文件用于测试
        test_df.to_csv('test_meteorological.csv', index=False)
        
        # 测试单源数据加载和处理
        logger.info("\n=== 测试单源数据处理 ===")
        file_paths = {'meteorological': 'test_meteorological.csv'}
        data = processor.load_data(file_paths)
        features, target = processor.preprocess_data(data)
        logger.info(f"预处理完成，特征形状：{features.shape}，目标形状：{target.shape}")
        
        # 测试时空特征提取
        X, y = processor.extract_spatiotemporal_features(features, target)
        logger.info(f"时空特征提取完成，序列形状：{X.shape}，目标形状：{y.shape}")
        
        # 测试数据集划分
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_dataset(X, y)
        
        # 测试数据加载器创建
        train_loader, val_loader, test_loader = processor.create_dataloaders(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        logger.info("\n=== 测试完成 ===")
        logger.info("数据处理模块测试通过")
        
        # 清理临时文件
        import os
        if os.path.exists('test_meteorological.csv'):
            os.remove('test_meteorological.csv')
            logger.info("临时测试文件已清理")
            
    except Exception as e:
        logger.error(f"测试过程中出现错误：{e}")
        import traceback
        traceback.print_exc()
