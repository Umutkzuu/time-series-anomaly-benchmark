import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import VarianceThreshold # CLAUDE'UN ÖNERDİĞİ KÜTÜPHANE
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
    
    # --- YENİ EKLENEN: SIFIR VARYANS TEMİZLİĞİ ---
    # Değeri hiç değişmeyen (sabit) ölü sensörleri bulup atıyoruz
    selector = VarianceThreshold(threshold=0.0)
    X_train = selector.fit_transform(X_train)
    X_val = selector.transform(X_val)
    X_test = selector.transform(X_test)
    
    # Artık ölü sensör kalmadığı için StandardScaler sıfıra bölme hatası vermeyecek
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
    
    gss_train = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, temp_idx = next(gss_train.split(X, y, groups))
    
    X_train, y_train, groups_temp = X[train_idx], y[train_idx], groups[temp_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]
    
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss_val.split(X_temp, y_temp, groups_temp))
    
    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]
    
    # --- YENİ EKLENEN: SIFIR VARYANS TEMİZLİĞİ ---
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

if __name__ == "__main__":
    cfg = data_loader.load_config()
    
    batadal_df = data_loader.load_batadal_data(cfg)
    res_batadal = prep_batadal(batadal_df, cfg)
    
    print("--- BATADAL BÖLME İŞLEMİ ---")
    print(f"Eğitim (Train) Boyutu      : {res_batadal['scaled'][0].shape}")
    print(f"Doğrulama (Val) Boyutu     : {res_batadal['scaled'][1].shape}")
    print(f"Test Boyutu                : {res_batadal['scaled'][2].shape}")
    print(f"Otomata (PCA) Train Boyutu : {res_batadal['pca'][0].shape}\n")
    
    skab_df = data_loader.load_skab_data(cfg)
    res_skab = prep_skab(skab_df, cfg)
    
    print("--- SKAB DOSYA BAZLI BÖLME İŞLEMİ ---")
    print(f"Eğitim (Train) Boyutu      : {res_skab['scaled'][0].shape}")
    print(f"Doğrulama (Val) Boyutu     : {res_skab['scaled'][1].shape}")
    print(f"Test Boyutu                : {res_skab['scaled'][2].shape}")
    print(f"Otomata (PCA) Train Boyutu : {res_skab['pca'][0].shape}")