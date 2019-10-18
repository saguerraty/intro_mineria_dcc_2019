from collections import Counter
from imblearn.over_sampling import SMOTENC
import pandas as pd


tabla_imbalanced = pd.read_csv("./tabla_imbalanced.csv").drop(columns=['Fecha'])

# we drop SO2, MP25, and NO2 due to much NaN
#tabla_imbalanced = tabla_imbalanced.drop(columns=['SO2', 'MP25', 'NO2'])
# %%
tabla_imbalanced.dtypes

#%%  
tabla_imbalanced.isna().sum()

# %%
columns = list(tabla_imbalanced)
categories = tabla_imbalanced['cat_mp10'].drop_duplicates().tolist()
nan_count = tabla_imbalanced.isna().sum()

# %%
smotenc = SMOTENC(categorical_features=[10])

X = tabla_imbalanced.iloc[:, 0:11]
y = tabla_imbalanced.iloc[:, 9:10]

X_res, y_res = smotenc.fit_resample(X, y)
print(f'Resampled dataset samples per class{Counter(y_res)}')


#%%[markdown]
Esta es una celda de markdown

#%%
