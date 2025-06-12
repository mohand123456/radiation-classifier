import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = {
    'radiation_type': ['WiFi', '4G', '5G', 'Bluetooth', 'IR'] * 20,
    'frequency_GHz': [2.4, 2.6, 3.5, 2.45, 1.0] * 20,
    'power_dBm': [-20, -10, 0, -15, -30] * 20,
    'duration_minutes': [10, 60, 30, 120, 5] * 20,
    'SAR_W_per_kg': [0.2, 1.5, 2.0, 0.1, 0.05] * 20,
    'label': ['Non-Harmful', 'Harmful', 'Harmful', 'Non-Harmful', 'Non-Harmful'] * 20
}

df = pd.DataFrame(data)
df = pd.get_dummies(df, columns=['radiation_type'])
X = df.drop('label', axis=1)
y = df['label']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump((model, X.columns), 'radiation_model.pkl')
