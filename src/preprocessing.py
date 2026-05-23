import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold # RUBRİK GÜNCELLEMESİ
from sklearn.feature_selection import VarianceThreshold
import data_loader
import warnings
warnings.filterwarnings('ignore')

def prep_batadal(df, config):
    target_col = config["batadal_params"]["target_col"]
    drop_cols = config["batadal_params"]["drop_time_cols"]
    
    df_features = df.drop(columns=drop_cols + [target_col], errors='ignore')
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.apply(pd.to_numeric, errors='coerce')
    df_features = df_features.ffill().bfill().fillna(0)
    
    X = df_features.values.astype(np.float64)
    y = df[target_col].values
    
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    selector = VarianceThreshold(threshold=0.0)
    X_train = selector.fit_transform(X_train)
    X_val = selector.transform(X_val)
    X_test = selector.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=1)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    return {
        "scaled": (X_train_scaled, X_val_scaled, X_test_scaled),
        "pca": (X_train_pca, X_val_pca, X_test_pca),
        "y": (y_train, y_val, y_test)
    }

def prep_skab(df, config):
    target_col = config["skab_params"]["target_col"]
    drop_cols = config["skab_params"]["drop_cols"]
    groups = df['source_file'].values
    
    df_features = df.drop(columns=drop_cols + [target_col], errors='ignore')
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.apply(pd.to_numeric, errors='coerce')
    df_features = df_features.ffill().bfill().fillna(0)
    
    X = df_features.values.astype(np.float64)
    y = df[target_col].values
    
    # RUBRİK İSTERİ: GroupKFold (5'e bölüp 60-20-20 oranını yakalama)
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(X, y, groups))
    
    # 5 Fold'un 3'ü Train (%60), 1'i Val (%20), 1'i Test (%20)
    train_idx = np.concatenate([folds[0][1], folds[1][1], folds[2][1]])
    val_idx = folds[3][1]
    test_idx = folds[4][1]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    selector = VarianceThreshold(threshold=0.0)
    X_train = selector.fit_transform(X_train)
    X_val = selector.transform(X_val)
    X_test = selector.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=1)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    return {
        "scaled": (X_train_scaled, X_val_scaled, X_test_scaled),
        "pca": (X_train_pca, X_val_pca, X_test_pca),
        "y": (y_train, y_val, y_test)
    }