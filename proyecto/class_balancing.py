

#%%
from collections import Counter
from imblearn.over_sampling import SMOTENC
import pandas as pd

#%%
tabla_imbalanced = pd.read_csv("./tabla_imbalanced.csv")

#%%
tabla_imbalanced.dtypes

#%%
columns = list(tabla_imbalanced)
categories = tabla_imbalanced['cat_mp10'].drop_duplicates().tolist()

#%%
smotenc = SMOTENC(['cat_mp10','cat_mp25'])

X = tabla_imbalanced.iloc[:, 0:13]
y = tabla_imbalanced.loc[:, 'cat_mp10']

X_res, y_res = smotenc.fit_resample(X, y)
print(f'Resampled dataset samples per class{Counter(y_res)}')

