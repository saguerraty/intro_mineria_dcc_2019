

#%%
from imblearn.over_sampling import SMOTENC 
import pandas as pd

#%%
tabla_imbalanced = pd.read_csv("C:/Users/sague/Documents/GitHub/intro_mineria_dcc_2019/proyecto/tabla_imbalanced.csv")

#%%
tabla_imbalanced.dtypes

#%%
columns = list(tabla_imbalanced)
categories = tabla_imbalanced['cat_mp10'].drop_duplicates().tolist()

#%%
balanced_df = SMOTENC(['cat_mp10','cat_mp25'])