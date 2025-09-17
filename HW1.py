import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt


def training_data_preprocess(file_path):
    """
    training data 資料清洗與格式轉換
    """
    df = pd.read_csv(file_path, encoding='utf-8')
    df.columns = ['Location', 'Date', 'ItemName'] + [str(i) for i in range(24)] # 確保欄位名稱正確
    df['ItemName'] = df['ItemName'].str.strip()
    
    # 將'Date'欄位轉換為datetime物件，並抽取出月份作為分組依據
    # 假設年份不影響，只關注月份和日期
    df['Date_obj'] = pd.to_datetime(df['Date'].str.split(' ').str[0], format='%m/%d')
    df['Month'] = df['Date_obj'].dt.month

    # 按月份進行處理與內插
    all_months_processed = []
    # 使用groupby將數據按月份分組，對每個月的數據獨立進行處理
    for month, month_df in df.groupby('Month'):
        # 對單一月份的數據進行Melt和Pivot
        month_data = month_df.melt(id_vars=['Date', 'Location', 'ItemName'], 
                                   value_vars=[str(i) for i in range(24)],
                                   var_name='hour', value_name='value')
        month_data['value'] = month_data['value'].replace({'NR': 0, '#': '', '*': '', 'x': '', 'A': ''})
        month_data['value'] = pd.to_numeric(month_data['value'], errors='coerce') # 保留NaN以便內插
        month_data['hour'] = month_data['hour'].astype(int)
        # 建立連續的DateTime索引
        month_data['DateTime'] = pd.to_datetime('2025/' + month_data['Date'].str.split(' ').str[0] + ' ' + month_data['hour'].astype(str) + ':00', 
                                                format='%Y/%m/%d %H:%M')
        pivoted_month = month_data.pivot_table(values='value', index='DateTime', columns='ItemName')
        
        # 在月份內部進行內插
        pivoted_month.interpolate(inplace=True)
        all_months_processed.append(pivoted_month)

    # 合併所有月份的結果
    final_df = pd.concat(all_months_processed)
    
    # 處理第一筆數據可能存在的NaN (如果開頭就有無效值)
    final_df.fillna(0, inplace=True)
    return final_df

def select_core_features(corr_df, top_n=8):
    """
    根據與PM2.5的相關性計算相關係數，選出Top N個核心特徵
    """
    correlations = corr_df.corr()['PM2.5'].abs().sort_values(ascending=False)
    # 排序後的第一個是PM2.5本身，故從第二個開始選
    core_features = correlations[1:top_n+1].index.tolist()

    # 注意：此處選出前8名相關的特徵，鑒於輸入是9小時的data，因此實際特徵數量是：8*9 = 72個原始特徵
    # 換個想法，對於第10小時而言，前9、8、7.......小時的不同特徵濃度對其的影響本應就不同
    return core_features


def generate_features_from_samples(X_original, core_features):
    """
    從滑動窗口後的樣本中，生成組合特徵（原始特徵、二次方、交互項）
    Args:
        X_original (np.array): shape為 (n_samples, 9, n_core_features) 的Numpy array
        core_features (list): 核心特徵的名稱列表
    Returns:
        X_candidate (np.array): 包含原始、二次方、交互項的攤平後的候選特徵矩陣
        candidate_feature_names (list): 描述每個候選特徵組合的名稱列表
    """
    n_samples = X_original.shape[0]
    n_hours = X_original.shape[1]

    # 生成特徵名稱
    original_names = [f"h{h}_{feat}" for h in range(n_hours) for feat in core_features]
    squared_names = [f"h{h}_{feat}_sq" for h in range(n_hours) for feat in core_features]
    interaction_names = []
    
    for h in range(n_hours):
            for i, j in combinations(range(len(core_features)), 2):
                interaction_names.append(f"h{h}_{core_features[i]}_x_{core_features[j]}")

    candidate_feature_names = original_names + squared_names + interaction_names

    # 1. 二次方項
    X_squared = X_original ** 2
    
    # 2. 交互項 (兩兩相乘)
    X_interaction = []
    for i in range(n_samples):
        sample_interaction_per_hour = []
        for hour_slice in X_original[i]:
            interaction_features = []
            for f1, f2 in combinations(hour_slice, 2):
                interaction_features.append(f1 * f2)
            sample_interaction_per_hour.append(interaction_features)
        X_interaction.append(sample_interaction_per_hour)
    X_interaction = np.array(X_interaction)

    # 將所有特徵攤平並組合
    X_original_flat = X_original.reshape(n_samples, -1)
    X_squared_flat = X_squared.reshape(n_samples, -1)
    X_interaction_flat = X_interaction.reshape(n_samples, -1)
    
    X_candidate = np.concatenate([X_original_flat, X_squared_flat, X_interaction_flat], axis=1)
    return X_candidate, candidate_feature_names


def generate_candidate_features(raw_df, core_features):
    """
    根據核心特徵，建立滑動窗口，並呼叫特徵生成器。
    """
    core_df = raw_df[core_features]
    X_original, y = [], []

    # 使用滑動窗口方式建立原始特徵樣本和目標值
    for i in range(len(core_df) - 9):
        X_original.append(core_df.iloc[i:i+9].values)
        y.append(raw_df.iloc[i+9]['PM2.5'])
    X_original = np.array(X_original)
    y = np.array(y)
    

    # 呼叫共用函數來生成候選特徵和其名稱
    X_candidate, candidate_feature_names = generate_features_from_samples(X_original, core_features)
    
    return X_candidate, y , candidate_feature_names


def select_features_with_lasso(X_candidate, y_candidate, alpha=0.1):
    """
    用Lasso模型從候選特徵池中篩選特徵。
    """

    # Lasso對特徵尺度敏感，需要先進行標準化
    # 計算候選特徵池的平均值和標準差
    mean = np.mean(X_candidate, axis=0)
    std = np.std(X_candidate, axis=0)
    std[std == 0] = 1  # 避免除以零
    
    # 執行標準化
    X_scaled = (X_candidate - mean) / std
     
    # 加上偏置項
    X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

    # 初始化並訓練Lasso模型
    # 注意：特徵數量是 X_b.shape[1]
    lasso_model = NumpyLinearRegression(
        n_features=X_b.shape[1], 
        epochs=2000, 
        learning_rate=0.1, 
        lambda_strength=alpha, 
        norm='L1'  # 指定使用L1正規化
    )
    lasso_model.train(X_b, y_candidate)
    
    # 找出權重不為零的特徵的索引
    # 梯度下降法可能讓權重很接近0但不是剛好為0，所以在此設一個小閾值
    # 要找的是weights[1:]中的非零權重，因為weights[0]是偏置項
    selected_feature_indices = np.where(np.abs(lasso_model.weights[1:]) > 1e-5)[0]
    
    return selected_feature_indices

class NumpyLinearRegression:
    # L1 norm用於特徵的篩選，L2 norm用於最終訓練
    """
    僅使用Numpy實現的線性迴歸模型，可選L1/L2正規化
    """
    def __init__(self, n_features, learning_rate=0.1, epochs=1000, lambda_strength=0.01, norm='L2'):
        """
        初始化模型參數。
        Args:
            n_features (int): 輸入特徵的數量 (包含偏置項)。
            learning_rate (float): 學習率。
            epochs (int): 訓練的迭代次數。
            lambda_strength (float): 正規化強度。
            norm (str): 正規化類型, 'L1' 或 'L2'。
        """
        self.lr = learning_rate
        self.epochs = epochs
        self.lambda_strength = lambda_strength
        self.weights = np.zeros((n_features, 1))
        self.norm = norm

    def _compute_loss(self, y_true, y_pred):
        """計算包含正規化項的MSE損失。"""
        mse = np.mean((y_true - y_pred)**2)
        
        # 根據正規化類型計算懲罰項
        if self.norm == 'L2':
            penalty = self.lambda_strength * np.sum(self.weights[1:]**2)
        elif self.norm == 'L1':
            penalty = self.lambda_strength * np.sum(np.abs(self.weights[1:]))
        else:
            penalty = 0
            
        return mse + penalty

    def predict(self, X):
        """使用當前權重進行預測。"""
        return X @ self.weights

    def train(self, X, y):
        """
        使用Adagrad優化器訓練模型。
        Args:
            X (np.array): 訓練特徵矩陣 (已包含偏置項)。
            y (np.array): 訓練目標向量。
        """
        n_samples = len(y)
        train_rmses = []
        val_rmses = []

        y = y.reshape(-1, 1)
        
        adagrad_w = np.zeros_like(self.weights)
        eps = 1e-8

        for epoch in range(self.epochs):
            y_pred = self.predict(X)
            
            # 計算梯度
            # 1. 計算MSE的梯度
            dw = (-2/n_samples) * X.T @ (y - y_pred)
            
            # 2. 根據正規化類型加上對應的梯度
            if self.norm == 'L2':
                # L2正規化項的梯度
                dw[1:] += 2 * self.lambda_strength * self.weights[1:]
            elif self.norm == 'L1':
                # L1正規化項的梯度 (subgradient)
                dw[1:] += self.lambda_strength * np.sign(self.weights[1:])
            
            # Adagrad更新步驟
            adagrad_w += dw**2
            self.weights -= self.lr * dw / (np.sqrt(adagrad_w) + eps)
            
            # 定期記錄RMSE
            if (epoch + 1) % 10 == 0:
                y_pred_train = self.predict(X)
                current_train_rmse = np.sqrt(np.mean((y - y_pred_train)**2))
                train_rmses.append(current_train_rmse)
                
                if X is not None and y is not None:
                    y_pred_val = self.predict(X)
                    current_val_rmse = np.sqrt(np.mean((y.reshape(-1, 1) - y_pred_val)**2))
                    val_rmses.append(current_val_rmse)
    
        return train_rmses, val_rmses


def predict_from_test_file(test_file_path, core_features, all_feature_names):
    """
    讀取測試檔案，進行處理，並呼叫特徵生成器，最終做出預測。
    """
    # 讀取與清理測試資料
    test_df = pd.read_csv(test_file_path, header=None, encoding='big5')
    test_df.iloc[:, 2:] = test_df.iloc[:, 2:].replace({'NR': 0, '#': '', '*': '', 'x': '', 'A': ''})
    test_df.iloc[:, 2:] = test_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 重塑與提取核心特徵
    n_samples = test_df.shape[0] // 18
    test_data = test_df.iloc[:, 2:].values.reshape(n_samples, 18, 9)
    test_data = test_data.transpose(0, 2, 1)
    core_feature_indices = {name: i for i, name in enumerate(core_features)} # 建立名稱到索引的映射
    all_feature_indices = {name: i for i, name in enumerate(all_feature_names)}
    
    # 找出核心特徵在18個原始特徵中的絕對索引
    abs_core_indices = [all_feature_indices[f] for f in core_features]
    X_original_test = test_data[:, :, abs_core_indices]
    
    # 按先前篩選出的特徵組合配方直接生成最終特徵
    final_feature_names = np.load('final_feature_names.npy') # 讀取配方
    final_feature_columns = []
    for feature_name in final_feature_names:
        parts = feature_name.split('_')
        hour = int(parts[0][1:]) # e.g., 'h0' -> 0
        
        if 'sq' in parts: # 二次方項, e.g., 'h0_PM10_sq'
            feat_name = parts[1]
            feat_idx = core_feature_indices[feat_name]
            col = X_original_test[:, hour, feat_idx] ** 2
            final_feature_columns.append(col)
        elif 'x' in parts: # 交互項, e.g., 'h0_PM10_x_SO2'
            feat1_name = parts[1]
            feat2_name = parts[3]
            feat1_idx = core_feature_indices[feat1_name]
            feat2_idx = core_feature_indices[feat2_name]
            col = X_original_test[:, hour, feat1_idx] * X_original_test[:, hour, feat2_idx]
            final_feature_columns.append(col)
        else: # 原始特徵, e.g., 'h0_PM10'
            feat_name = parts[1]
            feat_idx = core_feature_indices[feat_name]
            col = X_original_test[:, hour, feat_idx]
            final_feature_columns.append(col)
            
    # 將生成的欄位堆疊成最終的特徵矩陣
    X_final_test = np.stack(final_feature_columns, axis=1)

    # 使用已保存的權重進行預測
    scaler_params = np.load('final_scaler_params.npz')
    train_mean = scaler_params['mean']
    train_std = scaler_params['std']
    X_test_scaled = (X_final_test - train_mean) / train_std
    
    X_test_b = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]
    weights = np.load('trained_model.npy')
    predictions = X_test_b @ weights
    
    return predictions.flatten()

def run_data_size_experiment(X_full, y_full, X_val, y_val):
    """比較不同訓練資料量對模型準確度的影響。"""
    print("\n--- 不同訓練資料量對準確度的影響 ---")
    sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
    rmses = []
    for size in sizes:
        # 根據比例切分訓練子集
        n_train = int(X_full.shape[0] * size)
        X_train_subset = X_full[:n_train]
        y_train_subset = y_full[:n_train]
        
        # 重新訓練模型
        model = NumpyLinearRegression(n_features=X_full.shape[1], epochs=1000, learning_rate=0.5, lambda_strength=0.001)
        model.train(X_train_subset, y_train_subset)
        
        # 在固定的驗證集上評估RMSE
        y_pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val.reshape(-1, 1) - y_pred)**2))
        rmses.append(rmse)
        print(f"訓練資料量: {size*100:.0f}%, RMSE: {rmse:.4f}")
    
    # 繪製結果圖
    plt.figure()
    plt.plot([s * 100 for s in sizes], rmses, marker='o')
    plt.title('Impact of Training Data Size on RMSE')
    plt.xlabel('Percentage of Training Data Used (%)')
    plt.ylabel('RMSE on Validation Set')
    plt.grid(True)
    plt.savefig('data_size_impact.png')
    print("data_size_impact.png 已保存")
    return rmses

def run_regularization_experiment(X_train, y_train, X_val, y_val):
    """比較不同L2正規化強度對模型準確度的影響"""
    print("\n--- 比較不同L2正規化強度對模型準確度的影響 ---")
    lambdas = [0, 0.0001, 0.001, 0.01, 0.1, 1]
    rmses = []
    for l2 in lambdas:
        # 使用不同的lambda重新訓練模型
        model = NumpyLinearRegression(n_features=X_train.shape[1], epochs=1000, learning_rate=0.5, lambda_strength=l2)
        model.train(X_train, y_train)
        
        # 在固定的驗證集上評估RMSE
        y_pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val.reshape(-1, 1) - y_pred)**2))
        rmses.append(rmse)
        print(f"Lambda (L2): {l2}, RMSE: {rmse:.4f}")
    
    # 繪製結果圖
    plt.figure()
    plt.plot(lambdas, rmses, marker='o')
    plt.xscale('log') # Lambda的影響通常在對數尺度上更明顯
    plt.title('Impact of L2 Regularization on RMSE')
    plt.xlabel('Lambda (L2 Regularization Strength)')
    plt.ylabel('RMSE on Validation Set')
    plt.grid(True)
    plt.savefig('regularization_impact.png')
    print("regularization_impact.png 已保存。")
    return rmses





if __name__ == "__main__":

    # ---資料清洗與前處理----
    print("---資料清洗與前處理---")
    # 原始檔案路徑
    original_file_path = 'train.csv'
    formatted_file_path = 'train_formatted.csv'

    # 清洗資料
    clean_timeseries_df = training_data_preprocess(original_file_path)

    # 獲取所有18個原始特徵的名稱列表
    all_feature_names = clean_timeseries_df.columns.tolist()

    # 產生一個格式化後的中繼檔案，方便後續函數統一讀取
    clean_timeseries_df.to_csv(formatted_file_path, index=False)
    
    # ---特徵篩選---
    print("\n---特徵篩選---")
    # 識別核心特徵
    core_features = select_core_features(clean_timeseries_df, top_n=8)
    print(f"選出的核心特徵: {core_features}")
 
    # 生成候選特徵組合及其名稱配方
    X_candidate, y_candidate, candidate_feature_names = generate_candidate_features(clean_timeseries_df, core_features)
    print(f"生成的候選特徵池維度: {X_candidate.shape}")

    # Lasso特徵篩選
    selected_indices = select_features_with_lasso(X_candidate, y_candidate, alpha=0.01)
    
    # 根據索引，建立並儲存最終的特徵組合配方
    final_feature_names = [candidate_feature_names[i] for i in selected_indices]
    np.save('final_feature_names.npy', final_feature_names)


    # 根據索引，建立最終的訓練特徵矩陣
    X_final = X_candidate[:, selected_indices]
    print(f"Lasso 篩選後，最終特徵數量: {X_final.shape[1]}")


    # ---準備訓練集與驗證集----
    # 按照80/20的比例切分資料
    val_split = int(len(X_final) * 0.8)
    X_train_full, y_train_full = X_final[:val_split], y_candidate[:val_split]
    X_val, y_val = X_final[val_split:], y_candidate[val_split:]

    # 標準化最終的特徵集
    train_mean = np.mean(X_train_full, axis=0)
    train_std = np.std(X_train_full, axis=0)
    # 處理標準差為0的情況，避免除以零
    train_std[train_std == 0] = 1 

    # 進行標準化
    X_train_scaled = (X_train_full - train_mean) / train_std
    X_val_scaled = (X_val - train_mean) / train_std
    # 保存標準化所需的mean和std
    np.savez('final_scaler_params.npz', mean=train_mean, std=train_std)

    # 為訓練集和驗證集加上偏置項 (bias term, x0=1)
    X_train_b = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
    X_val_b = np.c_[np.ones((X_val_scaled.shape[0], 1)), X_val_scaled]
    

    # ---實驗：資料量影響與正規化影響---
    # 資料量影響
    data_size_results = run_data_size_experiment(X_train_b, y_train_full, X_val_b, y_val)
    # 正規化影響
    reg_results = run_regularization_experiment(X_train_b, y_train_full, X_val_b, y_val)
    

    # ---用選出的特徵組合訓練最終模型---
    print("\n--- 用選出的特徵組合訓練最終模型 ---")
    # 此處可根據實驗結果調整超參數，完成最終模型的訓練
    final_model = NumpyLinearRegression(n_features=X_train_b.shape[1], epochs=2000, learning_rate=0.5, lambda_strength=0.01)
    train_rmses, val_rmses = final_model.train(X_train_b, y_train_full)
    # 保存訓練好的模型權重
    np.save('trained_model.npy', final_model.weights)
    print("最終模型權重已保存。")   


    # --- 評估最終模型在驗證集上的表現 ---
    print("\n--- 評估最終模型在驗證集上的表現 ---")
    # 使用訓練好的最終模型對驗證集進行預測
    y_val_pred = final_model.predict(X_val_b)
    
    # 計算預測結果與真實標籤y_val之間的RMSE
    final_val_rmse = np.sqrt(np.mean((y_val.reshape(-1, 1) - y_val_pred)**2))
    # --- 繪製學習曲線圖 ---
    epochs_range = range(10, final_model.epochs + 1, 10)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_rmses, marker='.', linestyle='-', label='Training RMSE')
    plt.plot(epochs_range, val_rmses, marker='.', linestyle='-', label='Validation RMSE')
    plt.title('Learning Curve: RMSE vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=3.0) # 設定y軸下限，讓曲線變化更明顯
    plt.savefig('learning_curve.png')
    print("學習曲線圖 'learning_curve.png' 已成功生成。")
    print(f"最終模型在 20% 驗證集上的 RMSE: {final_val_rmse:.4f}")





    # ---讀取測試集，進行預測---
    predictions = predict_from_test_file('test.csv', core_features, all_feature_names)
    # 生成提交檔案
    submission_df = pd.DataFrame({'index': [f'index_{i}' for i in range(len(predictions))],
                                  'answer': predictions})
    submission_df.to_csv('submission.csv', index=False)
    print("\nsubmission.csv 已生成。")