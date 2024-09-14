from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
import pandas as pd
import os

config = OmegaConf.load("config.yaml")

def get_data():
    loc = snapshot_download(repo_id='Ransaka/TTS_dataset_unzipped', repo_type='dataset', token=config.hf_api_key)
    df = pd.read_csv(f"{loc}/metadata.csv", header=None, names=['audio','text'], skiprows=1)
    df['audio'] = df['audio'].apply(lambda x:f"{loc}/data/{x}")
    train_idx = range(0, int(len(df)*0.9))
    test_idx = range(int(len(df)*.9), len(df))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[test_idx].reset_index(drop=True)
    train_fp = os.path.join(config.data_dir, config.training_files)
    test_fp = os.path.join(config.data_dir, config.validation_files)
    train_df.to_csv(train_fp, index=False)
    val_df.to_csv(test_fp, index=False)