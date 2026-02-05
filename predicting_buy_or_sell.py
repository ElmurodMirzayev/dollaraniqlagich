import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('USD_UZS_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

df['close_lag1'] = df['close'].shift(1)
df['close_lag7'] = df['close'].shift(7)
df['close_ma7'] = df['close'].rolling(7).mean().shift(1)
df['change_lag1'] = df['change'].shift(1)
df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek


df['tomorrow'] = df['close'].shift(-4)
df = df.dropna().reset_index(drop=True)

###########################
df['target'] = (df['tomorrow'] > df['close']).astype(int)

df = df[df['date'] >= '2017-09-18']

###################################
features = ['close_lag1', 'close_lag7', 'close_ma7', 'change_lag1', 'change', 'dayofweek', 'month']

X = df[features]
y = df['target']
########################

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

#################################

model = XGBClassifier(
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.01
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
####################################



import joblib
joblib.dump(model, 'usduzs_buysell.pkl')


