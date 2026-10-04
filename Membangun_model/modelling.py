# modelling.py
# training model churn, logging diserahkan ke mlflow autolog.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn

# autolog yang mencatat parameter, metrik, dan artefak model ke mlflow
mlflow.sklearn.autolog()

mlflow.set_experiment("Eksperimen_Telco_Churn")

# dataset hasil preprocessing (satu folder dengan file ini)
data = pd.read_csv("telco_churn_preprocessing.csv")

X = data.drop("Churn", axis=1).astype(float)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("data latih:", X_train.shape, "| data uji:", X_test.shape)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # score() dipanggil supaya autolog ikut mencatat metriknya
    akurasi = model.score(X_test, y_test)
    print("akurasi di data uji:", round(akurasi, 4))

print("selesai. cek hasilnya lewat: mlflow ui  (lalu buka http://127.0.0.1:5000)")
