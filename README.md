# 資料載入器 (Data Loader)

這是一個用於機器學習資料集分割的 Python 工具，支援傳統分割模式和完整的 K-Fold 交叉驗證架構。

## 🎯 功能特色

### 雙模式支援
- **傳統分割模式**：直接按比例分割為訓練、驗證、測試集
- **完整交叉驗證模式**：先分割測試集，再對剩餘資料進行 K-Fold 交叉驗證

### 📊 完整交叉驗證架構
```
整體資料 = 原始資料集 (43個檔案)
        ├── 測試集（test set） ← 固定 15% (TEST_RATIO)
        └── 剩下的資料（train_val set）← 85%
               ↓
           K-Fold 分成：
           Fold 1：訓練(70-90%) + 驗證(10-30%)
           Fold 2：訓練(70-90%) + 驗證(10-30%)
           ...
           Fold K：訓練(70-90%) + 驗證(10-30%)
```

### ⚙️ 可配置參數
- **測試集比例**：固定為全域參數 `TEST_RATIO = 0.15`
- **訓練驗證比例**：可調整訓練集在訓練驗證集中的比例 (60%-90%)
- **交叉驗證折數**：支援任意 K 值
- **隨機種子**：確保結果可重現

## 📦 安裝需求

```bash
pip install pandas numpy
```

## 🚀 使用方法

### 基本使用

```python
from dataloader import get_data

# 傳統分割模式
train_df, val_df, test_df = get_data("2025_dataset_0806/train")

# 5折交叉驗證，取得第1折
train_df, val_df, test_df = get_data("2025_dataset_0806/train", k=5, num_k=1)

# 10折交叉驗證，取得第3折，訓練集佔80%
train_df, val_df, test_df = get_data("2025_dataset_0806/train", k=10, num_k=3, train_val_ratio=0.8)
```

### 📋 參數說明

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `folder` | str | - | 包含 CSV 檔案的資料夾路徑 |
| `k` | int | None | 交叉驗證的折數 |
| `num_k` | int | None | 要取得的第幾個折（從1開始） |
| `train_val_ratio` | float | 0.8 | 訓練集在訓練驗證集中的比例 (0.6-0.9) |
| `train_files` | int | 30 | 傳統模式下用於訓練的檔案數量 |
| `val_files` | int | 7 | 傳統模式下用於驗證的檔案數量 |
| `random_seed` | int | 42 | 隨機種子 |

## 💡 使用範例

### 1. 傳統分割模式
```python
# 使用預設參數
train_df, val_df, test_df = get_data("2025_dataset_0806/train")

print(f"訓練集大小: {len(train_df)}")
print(f"驗證集大小: {len(val_df)}")
print(f"測試集大小: {len(test_df)}")
```

### 2. 5折交叉驗證
```python
# 取得所有5折的資料
for fold in range(1, 6):
    train_df, val_df, test_df = get_data("2025_dataset_0806/train", k=5, num_k=fold)
    print(f"第{fold}折 - 訓練集: {len(train_df)}, 驗證集: {len(val_df)}, 測試集: {len(test_df)}")
```

### 3. 調整訓練驗證比例
```python
# 不同訓練驗證比例的比較
for ratio in [0.7, 0.8, 0.9]:
    train_df, val_df, test_df = get_data("2025_dataset_0806/train", k=5, num_k=1, train_val_ratio=ratio)
    print(f"訓練驗證比例 {ratio:.1%} - 訓練集: {len(train_df)}, 驗證集: {len(val_df)}")
```

### 4. 不同折數的交叉驗證
```python
# 比較不同折數的效果
for k in [3, 5, 10]:
    train_df, val_df, test_df = get_data("2025_dataset_0806/train", k=k, num_k=1)
    print(f"{k}折交叉驗證 - 訓練集: {len(train_df)}, 驗證集: {len(val_df)}")
```

## ⚙️ 全域參數

```python
TRAIN_FILES = 30      # 傳統模式下訓練檔案數
VAL_FILES = 7         # 傳統模式下驗證檔案數
RANDOM_SEED = 42      # 隨機種子
TEST_RATIO = 0.15     # 測試集比例（固定）
```

## 📊 輸出格式

每個返回的 DataFrame 都包含：
- 原始 CSV 檔案的所有欄位
- 新增的 `filename` 欄位，記錄資料來源檔案

## ⚠️ 錯誤處理

函數會檢查以下條件並拋出相應的錯誤：

- 資料夾中的檔案數量不等於 43
- `k` 或 `num_k` 參數不正確
- `train_val_ratio` 不在有效範圍 (0.6-0.9)

## 🧪 測試

執行以下命令來測試功能：

```bash
python dataloader.py
```

測試輸出包括：
- 傳統分割模式測試
- 不同訓練驗證比例的交叉驗證測試
- 不同折數的交叉驗證測試

## 📝 注意事項

1. **檔案要求**：資料夾必須包含恰好 43 個 CSV 檔案
2. **記憶體使用**：大型資料集可能需要較多記憶體
3. **隨機性**：使用固定隨機種子確保結果可重現
4. **測試集獨立性**：交叉驗證模式下，測試集完全獨立，不參與交叉驗證

## 🎯 最佳實踐

1. **交叉驗證**：建議使用 5-10 折交叉驗證
2. **訓練驗證比例**：通常使用 70%-80% 的訓練集比例
3. **測試集**：保持測試集獨立，僅用於最終評估
4. **隨機種子**：在實驗中保持隨機種子一致

## 📈 版本歷史

- **v1.0**：基本傳統分割功能
- **v2.0**：新增 K-Fold 交叉驗證支援
- **v3.0**：完整交叉驗證架構，固定測試集比例
- **v4.0**：新增訓練驗證比例調整功能

## 📄 授權

本專案採用 MIT 授權條款。

---

## 🔗 相關檔案

- `dataloader.py` - 主要資料載入器程式碼
- `README.md` - 完整專案說明
- `requirements.txt` - 依賴套件清單
