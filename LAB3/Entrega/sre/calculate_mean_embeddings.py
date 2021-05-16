import pandas as pd

# X-Vectors
df = pd.read_csv('voxceleb_pt/embeddings/train_xvector.csv')
df_mean = df.groupby(['id'])[df.columns[4:]].mean()
df_mean.to_csv(r'voxceleb_pt/embeddings/train_xvector_mean.csv')

# ECAPA-TDNN
df = pd.read_csv('voxceleb_pt/embeddings/train_ecapa.csv')
df_mean = df.groupby(['id'])[df.columns[4:]].mean()
df_mean.to_csv(r'voxceleb_pt/embeddings/train_ecapa_mean.csv')