import os
import glob
import json
import pandas as pd

def load_config(config_file="config.json"):
    """Merkezi konfigürasyon dosyasını dinamik yolla okur."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, config_file)
    
    with open(config_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_skab_data(config):
    """SKAB veri setini yükler ve birleştirir."""
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.normpath(os.path.join(current_dir, config["data_paths"]["skab_base_dir"]))
    folders = config["skab_params"]["folders"]
    
    all_data = []
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        for file in csv_files:
            df = pd.read_csv(file, sep=';') 
            
            
            df['source_group'] = folder
            df['source_file'] = os.path.basename(file)
            
            all_data.append(df)
            
    return pd.concat(all_data, ignore_index=True)

def load_batadal_data(config):
    """BATADAL veri setini yükler."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.normpath(os.path.join(current_dir, config["data_paths"]["batadal_file"]))
    
    batadal_df = pd.read_csv(filepath, sep=',')
    batadal_df.columns = batadal_df.columns.str.strip() 
    return batadal_df

if __name__ == "__main__":
    cfg = load_config()
    
    print("SKAB verisi yükleniyor...")
    skab = load_skab_data(cfg)
    print(f"SKAB Boyutu: {skab.shape}")
    print(f"SKAB Örnek Sütunlar: {skab.columns.tolist()[:3]} ... {skab.columns.tolist()[-3:]}\n")
    
    print("BATADAL verisi yükleniyor...")
    batadal = load_batadal_data(cfg)
    print(f"BATADAL Boyutu: {batadal.shape}")