#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""测试新添加的多源数据处理功能"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_mobile_data_processing():
    """测试移动走航数据处理"""
    logger.info("=== 测试移动走航数据处理 ===")
    
    # 初始化数据处理器
    config = {
        'seq_len': 24,
        'batch_size': 32,
        'time_resolution': '1H',
        'target_latitude': 39.9042,
        'target_longitude': 116.4074
    }
    processor = DataProcessor(config)
    
    # 加载移动走航数据
    file_paths = {
        'mobile': 'data/pm25_mobile_template.csv'
    }
    
    try:
        data = processor.load_data(file_paths)
        logger.info(f"移动走航数据加载完成，形状：{data.shape}")
        
        # 预处理数据
        features, target = processor.preprocess_data(data)
        logger.info(f"移动走航数据预处理完成，特征形状：{features.shape}，目标形状：{target.shape}")
        
        return True
    except Exception as e:
        logger.error(f"移动走航数据处理失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_lidar_data_processing():
    """测试激光雷达数据处理"""
    logger.info("\n=== 测试激光雷达数据处理 ===")
    
    # 初始化数据处理器
    config = {
        'seq_len': 24,
        'batch_size': 32,
        'time_resolution': '1H',
        'target_latitude': 39.9000,
        'target_longitude': 116.4000
    }
    processor = DataProcessor(config)
    
    # 加载激光雷达数据
    file_paths = {
        'lidar': 'data/pm25_lidar_template.csv'
    }
    
    try:
        data = processor.load_data(file_paths)
        logger.info(f"激光雷达数据加载完成，形状：{data.shape}")
        
        # 预处理数据
        features, target = processor.preprocess_data(data)
        logger.info(f"激光雷达数据预处理完成，特征形状：{features.shape}，目标形状：{target.shape}")
        
        return True
    except Exception as e:
        logger.error(f"激光雷达数据处理失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_microsite_data_processing():
    """测试小微站数据处理"""
    logger.info("\n=== 测试小微站数据处理 ===")
    
    # 初始化数据处理器
    config = {
        'seq_len': 24,
        'batch_size': 32,
        'time_resolution': '1H',
        'target_latitude': 39.9042,
        'target_longitude': 116.4074
    }
    processor = DataProcessor(config)
    
    # 加载小微站数据
    file_paths = {
        'microsite': 'data/pm25_microsite_template.csv'
    }
    
    try:
        data = processor.load_data(file_paths)
        logger.info(f"小微站数据加载完成，形状：{data.shape}")
        
        # 预处理数据
        features, target = processor.preprocess_data(data)
        logger.info(f"小微站数据预处理完成，特征形状：{features.shape}，目标形状：{target.shape}")
        
        return True
    except Exception as e:
        logger.error(f"小微站数据处理失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_source_fusion():
    """测试多源数据融合"""
    logger.info("\n=== 测试多源数据融合 ===")
    
    # 初始化数据处理器
    config = {
        'seq_len': 24,
        'batch_size': 32,
        'time_resolution': '1H',
        'target_latitude': 39.9042,
        'target_longitude': 116.4074
    }
    processor = DataProcessor(config)
    
    # 加载多种数据源
    file_paths = {
        'meteorological': 'data/pm25_data_template.csv',
        'mobile': 'data/pm25_mobile_template.csv',
        'lidar': 'data/pm25_lidar_template.csv',
        'microsite': 'data/pm25_microsite_template.csv'
    }
    
    try:
        data = processor.load_data(file_paths)
        logger.info(f"多源数据加载完成，形状：{data.shape}")
        logger.info(f"数据列：{list(data.columns)}")
        
        # 预处理数据
        features, target = processor.preprocess_data(data)
        logger.info(f"多源数据预处理完成，特征形状：{features.shape}，目标形状：{target.shape}")
        
        # 提取时空特征
        X, y = processor.extract_spatiotemporal_features(features, target)
        logger.info(f"时空特征提取完成，X形状：{X.shape}，y形状：{y.shape}")
        
        # 划分数据集
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_dataset(X, y)
        logger.info(f"数据集划分完成，训练集：{X_train.shape}, {y_train.shape}")
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = processor.create_dataloaders(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        logger.info(f"数据加载器创建完成，训练加载器批次：{len(train_loader)}")
        
        return True
    except Exception as e:
        logger.error(f"多源数据融合失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_time_resolution_alignment():
    """测试时间分辨率对齐"""
    logger.info("\n=== 测试时间分辨率对齐 ===")
    
    # 测试不同时间分辨率
    resolutions = ['30T', '1H', '2H']
    
    for resolution in resolutions:
        logger.info(f"\n测试时间分辨率：{resolution}")
        
        # 初始化数据处理器
        config = {
            'seq_len': 24,
            'batch_size': 32,
            'time_resolution': resolution,
            'target_latitude': 39.9042,
            'target_longitude': 116.4074
        }
        processor = DataProcessor(config)
        
        # 加载数据
        file_paths = {
            'meteorological': 'data/pm25_data_template.csv',
            'microsite': 'data/pm25_microsite_template.csv'
        }
        
        try:
            data = processor.load_data(file_paths)
            logger.info(f"数据加载完成，形状：{data.shape}")
            
            # 预处理数据（包含时间分辨率对齐）
            features, target = processor.preprocess_data(data)
            logger.info(f"数据预处理完成，特征形状：{features.shape}，目标形状：{target.shape}")
            
        except Exception as e:
            logger.error(f"时间分辨率对齐失败：{e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def main():
    """主测试函数"""
    logger.info("开始测试PM2.5预测新功能")
    
    # 运行所有测试
    tests = [
        test_mobile_data_processing,
        test_lidar_data_processing,
        test_microsite_data_processing,
        test_multi_source_fusion,
        test_time_resolution_alignment
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # 汇总测试结果
    passed = sum(results)
    total = len(results)
    
    logger.info(f"\n=== 测试结果汇总 ===")
    logger.info(f"通过测试：{passed}/{total}")
    
    if passed == total:
        logger.info("✅ 所有测试通过！")
        return 0
    else:
        logger.error(f"❌ 有 {total - passed} 个测试失败！")
        return 1


if __name__ == '__main__':
    sys.exit(main())
