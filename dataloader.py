import os
import glob
import random
import pandas as pd

TRAIN_FILES = 30
VAL_FILES = 7
RANDOM_SEED = 42
TEST_RATIO = 0.15  # 全域測試集比例參數

def get_data(folder, k=None, num_k=None, train_val_ratio=0.8, train_files=TRAIN_FILES, val_files=VAL_FILES, random_seed=RANDOM_SEED):
    """
    載入並分割 CSV 檔案資料集，用於機器學習訓練。
    
    支援兩種模式：
    1. 傳統分割模式：直接按比例分割為訓練、驗證、測試集
    2. 完整交叉驗證模式：先分割測試集，再對剩餘資料進行K-Fold交叉驗證
    
    Parameters:
    -----------
    folder : str
        包含 CSV 檔案的資料夾路徑
    k : int, optional
        交叉驗證的折數。如果為 None，則使用傳統分割模式
    num_k : int, optional
        要取得的第幾個折（從1開始）。如果為 None，則使用傳統分割模式
    train_val_ratio : float, default=0.8
        訓練集在訓練驗證集中的比例（0.6-0.9之間）
    train_files : int, default=TRAIN_FILES (30)
        傳統模式下用於訓練的檔案數量
    val_files : int, default=VAL_FILES (7)
        傳統模式下用於驗證的檔案數量
    random_seed : int, default=RANDOM_SEED (42)
        隨機種子
        
    Returns:
    --------
    tuple : (train_df, val_df, test_df)
        train_df : pandas.DataFrame
            訓練資料集
        val_df : pandas.DataFrame
            驗證資料集
        test_df : pandas.DataFrame
            測試資料集（交叉驗證模式下為提前分割的測試集）
            
    Raises:
    -------
    ValueError
        當資料夾中的檔案數量不等於 43 時拋出錯誤
        當 k 或 num_k 參數不正確時拋出錯誤
        當 train_val_ratio 不在有效範圍時拋出錯誤
        
    Examples:
    ---------
    # 傳統分割模式
    >>> folder = "2025_dataset_0806/train"
    >>> train_df, val_df, test_df = get_data(folder)
    
    # 5折交叉驗證，取得第1折，訓練集佔80%
    >>> train_df, val_df, test_df = get_data(folder, k=5, num_k=1, train_val_ratio=0.8)
    
    # 10折交叉驗證，取得第3折，訓練集佔70%
    >>> train_df, val_df, test_df = get_data(folder, k=10, num_k=3, train_val_ratio=0.7)
    
    Notes:
    ------
    - 函數會自動為每個 DataFrame 添加 'filename' 欄位，記錄資料來源檔案
    - 使用 random.seed(random_seed) 確保相同隨機種子時的分割結果一致
    - 檔案會先按名稱排序，然後根據隨機種子打亂
    - 交叉驗證模式下，會先分割出測試集（TEST_RATIO），然後對剩餘資料進行K-Fold分割
    - 測試集比例固定為全域參數 TEST_RATIO
    """
    all_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    print(f"總檔案數: {len(all_files)}")
    if len(all_files) != 43:
        raise ValueError(f"Folder {folder} must contain 43 files")
    
    # 隨機打亂
    random.seed(random_seed)
    random.shuffle(all_files)
    
    # 檢查是否使用交叉驗證模式
    if k is not None and num_k is not None:
        # 完整交叉驗證模式
        if k <= 0 or num_k <= 0 or num_k > k:
            raise ValueError(f"Invalid k={k} or num_k={num_k}. k must be > 0 and num_k must be between 1 and k")
        
        if train_val_ratio < 0.6 or train_val_ratio > 0.9:
            raise ValueError(f"train_val_ratio must be between 0.6 and 0.9, got {train_val_ratio}")
        
        print(f"使用完整交叉驗證模式：{k} 折交叉驗證，測試集比例 {TEST_RATIO:.1%}")
        print(f"訓練集在訓練驗證集中的比例：{train_val_ratio:.1%}")
        
        # 1. 先分割測試集（使用全域參數）
        test_count = int(len(all_files) * TEST_RATIO)
        test_list = all_files[:test_count]
        train_val_list = all_files[test_count:]
        
        print(f"測試集檔案數: {len(test_list)}")
        print(f"訓練驗證集檔案數: {len(train_val_list)}")
        
        # 2. 對訓練驗證集進行K-Fold分割
        files_per_fold = len(train_val_list) // k
        remainder = len(train_val_list) % k
        
        # 計算當前折的起始和結束索引
        start_idx = (num_k - 1) * files_per_fold + min(num_k - 1, remainder)
        end_idx = start_idx + files_per_fold + (1 if num_k <= remainder else 0)
        
        # 分割檔案
        val_list = train_val_list[start_idx:end_idx]
        train_list = train_val_list[:start_idx] + train_val_list[end_idx:]
        
        # 3. 根據train_val_ratio調整訓練集和驗證集的比例
        total_train_val_files = len(train_list) + len(val_list)
        target_train_files = int(total_train_val_files * train_val_ratio)
        
        # 重新分配訓練集和驗證集
        if len(train_list) > target_train_files:
            # 訓練集太多，移動一些到驗證集
            move_to_val = len(train_list) - target_train_files
            val_list = train_list[-move_to_val:] + val_list
            train_list = train_list[:-move_to_val]
        elif len(train_list) < target_train_files:
            # 訓練集太少，從驗證集移動一些過來
            move_to_train = target_train_files - len(train_list)
            if move_to_train <= len(val_list):
                train_list = train_list + val_list[:move_to_train]
                val_list = val_list[move_to_train:]
        
        print(f"第{num_k}折 - 訓練集檔案數: {len(train_list)}")
        print(f"第{num_k}折 - 驗證集檔案數: {len(val_list)}")
        
    else:
        # 傳統分割模式
        print("使用傳統分割模式")
        train_list = all_files[:train_files]
        val_list = all_files[train_files:train_files+val_files]
        test_list = all_files[train_files+val_files:]
        
        print(f"訓練集檔案數: {len(train_list)}")
        print(f"驗證集檔案數: {len(val_list)}")
        print(f"測試集檔案數: {len(test_list)}")

    # 載入資料
    train_dfs = []
    val_dfs = []
    test_dfs = []

    for file in train_list:
        df = pd.read_csv(file)
        df['filename'] = os.path.basename(file)
        train_dfs.append(df)
    for file in val_list:
        df = pd.read_csv(file)
        df['filename'] = os.path.basename(file)
        val_dfs.append(df)
    for file in test_list:
        df = pd.read_csv(file)
        df['filename'] = os.path.basename(file)
        test_dfs.append(df)

    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()

    return train_df, val_df, test_df
        

if __name__ == "__main__":
    folder = "2025_dataset_0806/train"
    
    print("=== 測試傳統分割模式 ===")
    train_df, val_df, test_df = get_data(folder)
    print(f"訓練集大小: {len(train_df)}")
    print(f"驗證集大小: {len(val_df)}")
    print(f"測試集大小: {len(test_df)}")
    print()
    
    print("=== 測試5折交叉驗證（不同訓練驗證比例） ===")
    for train_val_ratio in [0.7, 0.8, 0.9]:
        train_df, val_df, test_df = get_data(folder, k=5, num_k=1, train_val_ratio=train_val_ratio)
        print(f"訓練驗證比例 {train_val_ratio:.1%} - 訓練集: {len(train_df)}, 驗證集: {len(val_df)}, 測試集: {len(test_df)}")
    print()
    
    print("=== 測試10折交叉驗證 ===")
    train_df, val_df, test_df = get_data(folder, k=10, num_k=3, train_val_ratio=0.8)
    print(f"第3折 - 訓練集: {len(train_df)}, 驗證集: {len(val_df)}, 測試集: {len(test_df)}")
    print()
    
    print("=== 測試不同折數的交叉驗證 ===")
    for k in [3, 5, 10]:
        train_df, val_df, test_df = get_data(folder, k=k, num_k=1, train_val_ratio=0.8)
        print(f"{k}折交叉驗證 - 訓練集: {len(train_df)}, 驗證集: {len(val_df)}, 測試集: {len(test_df)}")
