import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import shap


class XGBGPUTrainer:
    def __init__(self, params=None):
        """
        初始化 XGBoost GPU 训练器
        :param params: 自定义参数字典
        """
        # 默认针对 CUDA 13.0 优化的 GPU 参数
        self.default_params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'verbosity': 1,
            'base_score': 0.5  # 显式设置避免 SHAP 兼容性问题
        }
        if params:
            self.default_params.update(params)
        
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=50):
        """
        执行训练
        """
        self.model = xgb.XGBRegressor(**self.default_params)
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=100
        )
        return self.model

    def evaluate(self, X_test, y_test, inverse_func=None):
        """
        评估模型性能
        :param X_test: 测试特征
        :param y_test: 真实标签
        :param inverse_func: 逆转函数，例如 np.expm1 或 scaler.inverse_transform
        """
        if self.model is None:
            raise ValueError("模型尚未训练！")

        # 预测
        preds = self.model.predict(X_test)

        # 处理尺度转换
        if inverse_func:
            y_true_final = inverse_func(y_test)
            y_pred_final = inverse_func(preds)
            scale_msg = "(已转换回原始尺度)"
        else:
            y_true_final = y_test
            y_pred_final = preds
            scale_msg = "(原始模型尺度)"

        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_true_final, y_pred_final))
        mae = mean_absolute_error(y_true_final, y_pred_final)
        r2 = r2_score(y_true_final, y_pred_final)

        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

        print(f"\n--- 评估结果 {scale_msg} ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        
        return metrics, y_pred_final

    def save_model(self, path="xgb_model.json"):
        if self.model:
            self.model.save_model(path)
            print(f"模型已保存至: {path}")

    def predict(self, X, inverse_func=None):
            """
            执行预测并支持尺度还原
            :param X: 待预测的特征数据
            :param inverse_func: 逆转函数，如 np.expm1 或 scaler.inverse_transform
            :return: 预测结果向量
            """
            if self.model is None:
                raise ValueError("模型尚未训练或加载！")

            # 执行基础预测
            preds = self.model.predict(X)

            # 如果提供了逆转函数，则处理尺度转换
            if inverse_func:
                preds = inverse_func(preds)
                
            return preds
