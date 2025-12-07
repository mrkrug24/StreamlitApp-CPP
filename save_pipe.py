import io, re, pickle, requests, warnings
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, TransformerMixin
warnings.filterwarnings('ignore')

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_medians = {}
        self.categorical_modes = {}
        self.torque_pat = re.compile(r'(\d+\.?\d*)\s*(Nm|kgm)', re.I)
    
    def _ext_num(self, v):
        if pd.isna(v): return np.nan
        nums = re.findall(r'\d+\.?\d*', str(v))
        return float(nums[0]) if nums else np.nan
    
    def _ext_torque(self, t):
        if pd.isna(t): return np.nan
        m = self.torque_pat.search(str(t))
        return float(m.group(1)) * (9.80665 if m.group(2).lower().startswith('kg') else 1) if m else np.nan
    
    def fit(self, X, y=None):
        Xc = X.copy()
        for c in ['mileage','engine','max_power']: Xc[c] = Xc[c].apply(self._ext_num)
        if 'torque' in Xc.columns: Xc['torque'] = Xc['torque'].apply(self._ext_torque)
        Xc['seats'] = Xc['seats'].astype(str)
        
        num_cols = ['mileage','engine','max_power','torque','year','km_driven']
        cat_cols = ['fuel','seller_type','transmission','owner','seats']
        
        for c in num_cols:
            if c in Xc.columns:
                Xc[c] = pd.to_numeric(Xc[c], errors='coerce')
                self.numeric_medians[c] = Xc[c].median()
        
        for c in cat_cols:
            if c in Xc.columns:
                mode = Xc[c].mode()
                self.categorical_modes[c] = mode.iloc[0] if not mode.empty else Xc[c].iloc[0]
        
        return self
    
    def transform(self, X):
        Xc = X.copy()
        for c in ['mileage','engine','max_power']: Xc[c] = Xc[c].apply(self._ext_num)
        if 'torque' in Xc.columns: Xc['torque'] = Xc['torque'].apply(self._ext_torque)
        
        for c, v in self.numeric_medians.items():
            if c in Xc.columns:
                Xc[c] = pd.to_numeric(Xc[c], errors='coerce').fillna(v)
        
        for c, v in self.categorical_modes.items():
            if c in Xc.columns:
                if c == 'seats': Xc[c] = Xc[c].astype(str)
                Xc[c] = Xc[c].fillna(v)
        
        return Xc

df_train = pd.read_csv(io.StringIO(requests.get('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv', verify=False).text))
df_test = pd.read_csv(io.StringIO(requests.get('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv', verify=False).text))

X_train, X_test = df_train.drop('selling_price', axis=1), df_test.drop('selling_price', axis=1)
y_train, y_test = np.log(df_train['selling_price']), np.log(df_test['selling_price'])

num_feat = ['year','km_driven','mileage','engine','max_power','torque']
cat_feat = ['fuel','seller_type','transmission','owner','seats']

prep = ColumnTransformer([
    ('num', StandardScaler(), num_feat),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feat)
])

pipe = Pipeline([
    ('fe', FeatureExtractor()),
    ('prep', prep),
    ('model', Ridge(alpha=10, random_state=42))
])

pipe.fit(X_train, y_train)

y_train_pred, y_test_pred = pipe.predict(X_train), pipe.predict(X_test)
train_r2, test_r2 = r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)

print(f"R2 на train: {train_r2:.4f}\nR2 на test: {test_r2:.4f}")

with open('pipe.pkl', 'wb') as f: pickle.dump(pipe, f)

fe, prep = pipe['fe'], pipe['prep']
cat_enc = prep.named_transformers_['cat']
cat_feat_out = list(cat_enc.get_feature_names_out(cat_feat)) if hasattr(cat_enc, 'get_feature_names_out') else [f"cat_{i}" for i in range(cat_enc.n_features_in_)]

pipe_info = {
    'pipeline': pipe,
    'numeric_features': num_feat,
    'categorical_features': cat_feat,
    'all_features': num_feat + cat_feat_out,
    'numeric_medians': fe.numeric_medians,
    'categorical_modes': fe.categorical_modes,
    'model_metrics': {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'model_type': 'Ridge Regression',
        'alpha': 10
    }
}

with open('pipe_info.pkl', 'wb') as f: pickle.dump(pipe_info, f)