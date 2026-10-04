# modelling_tuning.py
# training + tuning GridSearchCV, logging ke mlflow dilakukan manual.
# untuk logging online ke DagsHub: ubah PAKAI_DAGSHUB jadi True (repo DagsHub harus sudah ada).

import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

import mlflow
import mlflow.sklearn

# pengaturan dagshub
PAKAI_DAGSHUB = False
DAGSHUB_REPO_OWNER = "anggarabudiyantooo-hub"
DAGSHUB_REPO_NAME = "Eksperimen_SML_Anggara-Budiyanto"

if PAKAI_DAGSHUB:
    import dagshub
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
else:
    mlflow.set_tracking_uri("mlruns")

mlflow.set_experiment("Eksperimen_Telco_Churn_Tuning")

# ---------------- siapkan data ----------------
data = pd.read_csv("telco_churn_preprocessing.csv")
X = data.drop("Churn", axis=1).astype(float)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- tuning hyperparameter ----------------
# kombinasi sengaja dibuat sedikit supaya tidak lama
pilihan_param = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}

print("mulai tuning... (mencoba", 3 * 3 * 2, "kombinasi, cv=3)")
mulai = time.time()

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    pilihan_param,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
)
grid.fit(X_train, y_train)

waktu_latih = time.time() - mulai
print("tuning selesai dalam {:.1f} detik".format(waktu_latih))
print("param terbaik:", grid.best_params_)

model = grid.best_estimator_
y_pred = model.predict(X_test)

# ---------------- hitung metrik ----------------
akurasi = accuracy_score(y_test, y_pred)
presisi = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("akurasi :", round(akurasi, 4))
print("presisi :", round(presisi, 4))
print("recall  :", round(recall, 4))
print("f1      :", round(f1, 4))

# ---------------- logging manual ke mlflow ----------------
with mlflow.start_run(run_name="random_forest_tuning"):
    # parameter ikut dicatat
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("tuning", "GridSearchCV cv=3")
    mlflow.log_params(grid.best_params_)

    # metrik dicatat manual (menggantikan autolog)
    mlflow.log_metric("accuracy", akurasi)
    mlflow.log_metric("precision", presisi)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("waktu_training_detik", waktu_latih)

    # artefak tambahan 1: confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["No churn", "Churn"]).plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix (data uji)")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png")
    plt.close(fig)

    # artefak tambahan 2: feature importance 10 teratas
    penting = pd.Series(model.feature_importances_, index=X.columns).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    penting.plot(kind="barh", ax=ax, color="#E0A94C")
    plt.title("10 fitur paling berpengaruh")
    fig.tight_layout()
    fig.savefig("feature_importance.png")
    plt.close(fig)

    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("feature_importance.png")

    # model ikut disimpan
    mlflow.sklearn.log_model(model, "model", input_example=X_test.head(3))

    # id run disimpan ke file, berguna kalau mau serving/docker otomatis
    try:
        with open("run_id_terakhir.txt", "w") as f:
            f.write(mlflow.active_run().info.run_id)
    except PermissionError:
        pass  # file sedang dikunci windows, tidak apa-apa

print("logging selesai.")
if PAKAI_DAGSHUB:
    print("cek hasilnya di halaman DagsHub (menu Experiments)")
else:
    print("cek hasilnya lewat: mlflow ui  (lalu buka http://127.0.0.1:5000)")
