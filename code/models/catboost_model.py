import numpy as np
import pandas as pd
import glob
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import joblib


current_directory = Path.cwd().parent.parent
current_directory = str(current_directory)
current_directory = current_directory.replace("\\", "/")

file_path_pattern = current_directory + "/data/tradestats_202*/tradestats_202*.csv"

file_list = glob.glob(file_path_pattern)
file_list = sorted(file_list)
print(file_list)

dataframes = [pd.read_csv(file, sep=';') for file in file_list]
combined_df = pd.concat(dataframes, ignore_index=True)

print(combined_df.head())

del dataframes

combined_df['datetime'] = combined_df['tradedate'].astype(str) + ' ' + combined_df['tradetime']
combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
combined_df = combined_df.drop(columns=['tradedate', 'tradetime', 'SYSTIME'])

combined_df['pr_std'] = combined_df.groupby('secid')['pr_std'].transform(lambda x: x.fillna(x.mean()))
combined_df['pr_vwap_b'] = combined_df.groupby('secid')['pr_vwap_b'].transform(lambda x: x.fillna(x.mean()))
combined_df['pr_vwap_s'] = combined_df.groupby('secid')['pr_vwap_s'].transform(lambda x: x.fillna(x.mean()))

mean_pr_std = combined_df['pr_std'].mean()  # Calculate mean of pr_std
combined_df['pr_std'].fillna(mean_pr_std, inplace=True)  # Fill nulls with mean

mean_pr_vwap_b = combined_df['pr_vwap_b'].mean()  # Calculate mean of pr_vwap_b
combined_df['pr_vwap_b'].fillna(mean_pr_vwap_b, inplace=True)  # Fill nulls with mean

combined_df['year'] = combined_df['datetime'].dt.year
combined_df['month'] = combined_df['datetime'].dt.month
combined_df['day'] = combined_df['datetime'].dt.day
combined_df['hour'] = combined_df['datetime'].dt.hour
combined_df['minute'] = combined_df['datetime'].dt.minute

combined_df = combined_df.drop(columns=['datetime'])

label_encoder = LabelEncoder()
combined_df['secid_encoded'] = label_encoder.fit_transform(combined_df['secid'])

combined_df = combined_df.drop(columns=['secid'])

df = combined_df.copy()
del combined_df

df['target'] = df.groupby('secid_encoded')['pr_close'].shift(-1) > df['pr_close']

df['target'] = df['target'].astype(int)

df['target'].fillna(0, inplace=True)

catboost_model = CatBoostClassifier(
    iterations=1000,
    l2_leaf_reg=3,
    random_strength=0.1,
    bagging_temperature=0.5,
    eval_metric="PRAUC",
    custom_loss=["PRAUC"],
    random_seed=17,
    od_type="Iter",
    od_wait=30,
    task_type='GPU',
)

data = df
X = data.drop('target', axis=1)
y = data['target']
smote = SMOTE(sampling_strategy='minority')

batch_size = 100000  # Size of each batch
X_resampled, y_resampled = [], []

for i in range(0, len(X), batch_size):
    print(i)
    X_batch = X[i:i + batch_size]
    y_batch = y[i:i + batch_size]

    smote = SMOTE(sampling_strategy='minority')
    X_resampled_batch, y_resampled_batch = smote.fit_resample(X_batch, y_batch)

    X_resampled.append(X_resampled_batch)
    y_resampled.append(y_resampled_batch)

# Concatenate the batches
X_resampled = np.vstack(X_resampled)
y_resampled = np.concatenate(y_resampled)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

catboost_model.fit(X_train, y_train)

joblib.dump(catboost_model, 'catboost.pkl')
