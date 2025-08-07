import os
import glob
import random
import pandas as pd

TRAIN_FILES = 30
VAL_FILES = 7

def get_data(folder, k=1, train_files=TRAIN_FILES, val_files=VAL_FILES):
    """
    載入並分割 CSV 檔案資料集，用於機器學習訓練。
    
    此函數會從指定資料夾中載入所有 CSV 檔案，並根據指定的分割比例
    將資料分為訓練集、驗證集和預測集。檔案會根據隨機種子進行打亂，
    確保每次呼叫時的分割結果一致。
    
    Parameters:
    -----------
    folder : str
        包含 CSV 檔案的資料夾路徑
    k : int, default=1
        隨機種子倍數，用於產生不同的資料分割
    train_files : int, default=TRAIN_FILES (30)
        用於訓練的檔案數量
    val_files : int, default=VAL_FILES (7)
        用於驗證的檔案數量
        
    Returns:
    --------
    tuple : (train_df, val_df, pred_df)
        train_df : pandas.DataFrame
            訓練資料集，包含所有訓練檔案的資料
        val_df : pandas.DataFrame
            驗證資料集，包含所有驗證檔案的資料
        pred_df : pandas.DataFrame
            預測資料集，包含剩餘檔案的資料
            
    Raises:
    -------
    ValueError
        當資料夾中的檔案數量不等於 43 時拋出錯誤
        
    Examples:
    ---------
    >>> folder = "2025_dataset_0806/train"
    >>> train_df, val_df, pred_df = get_data(folder, k=1)
    >>> print(f"訓練集大小: {len(train_df)}")
    >>> print(f"驗證集大小: {len(val_df)}")
    >>> print(f"預測集大小: {len(pred_df)}")
    
    Notes:
    ------
    - 函數會自動為每個 DataFrame 添加 'filename' 欄位，記錄資料來源檔案
    - 使用 random.seed(42*k) 確保相同 k 值時的分割結果一致
    - 檔案會先按名稱排序，然後根據隨機種子打亂
    """
    all_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    print(len(all_files))
    if len(all_files) != 43:
        raise ValueError(f"Folder {folder} must contain 43 files")
    
    # 隨機打亂
    random.seed(42*k)
    random.shuffle(all_files)
    print(all_files[0])
    train_list = all_files[:train_files]
    val_list = all_files[train_files:train_files+val_files]
    pred_list = all_files[train_files+val_files:]

    train_dfs = []
    val_dfs = []
    pred_dfs = []

    for file in train_list:
        df = pd.read_csv(file)
        df['filename'] = os.path.basename(file)
        train_dfs.append(df)
    for file in val_list:
        df = pd.read_csv(file)
        df['filename'] = os.path.basename(file)
        val_dfs.append(df)
    for file in pred_list:
        df = pd.read_csv(file)
        df['filename'] = os.path.basename(file)
        pred_dfs.append(df)

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    pred_df = pd.concat(pred_dfs, ignore_index=True)

    return train_df, val_df, pred_df
        


if __name__ == "__main__":
    folder = "2025_dataset_0806/train"
    for k in range(1, 11):
        train_df, val_df, pred_df = get_data(folder, k)
        print(train_df.iloc[0]['filename'])
