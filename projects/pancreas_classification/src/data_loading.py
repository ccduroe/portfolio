import pandas as pd
import scanpy as sc
import numpy as np

def load_data(mouse_data_path, human_data_path, mouse_label_path, human_label_path):
    mouse_data = sc.read_h5ad(mouse_data_path)
    human_data = sc.read_h5ad(human_data_path)

    train_label_df = pd.read_csv(mouse_label_path, usecols=[1], skiprows=[0], names=["Label"])
    test_label_df = pd.read_csv(human_label_path, usecols=[1], skiprows=[0], names=["Label"])

    return mouse_data, human_data, train_label_df, test_label_df
