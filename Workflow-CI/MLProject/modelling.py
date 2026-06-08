import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn
import os
import shutil

# Load Data
df = pd.read_csv('C:\\Users\\Jauhanunu\\Documents\\Pijak\\SMSML_Jauhan-Ahmad\\Eksperimen_SML_Jauhan\\Membangun_model\\dataset_preprocessing\\data_bersih.csv')
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Simpan model sebagai artefak lokal
model_dir = "titanic_model_artifact"
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

mlflow.sklearn.save_model(model, model_dir)
print(f"Model berhasil dilatih dan disimpan di direktori: {model_dir}")