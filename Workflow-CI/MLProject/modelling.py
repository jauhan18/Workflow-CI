import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn
import os
import shutil

# 1. Load Data secara dinamis menggunakan os.path
SML_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(SML_dir, 'data_bersih.csv'))

# 2. Pemisahan Fitur dan Target
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Training Model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 4. Simpan model sebagai artefak lokal
model_dir = "titanic_model_artifact"
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

mlflow.sklearn.save_model(model, model_dir)
print(f"Model berhasil dilatih dan disimpan di direktori: {model_dir}")
