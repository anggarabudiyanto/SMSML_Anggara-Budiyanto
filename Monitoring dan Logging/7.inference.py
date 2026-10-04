# 7.inference.py
# kirim beberapa contoh data ke endpoint model dan tampilkan prediksinya.
# model harus sudah di-serve dulu (mlflow models serve -p 5000, atau docker run -p 5000:8080).

import json
import requests

ALAMAT = "http://127.0.0.1:5000/invocations"

with open("sample_payload.json") as f:
    semua_contoh = json.load(f)

print("mengirim", len(semua_contoh), "contoh ke model di", ALAMAT)
print("-" * 60)

for i, contoh in enumerate(semua_contoh, start=1):
    payload = {"dataframe_records": [contoh["fitur"]]}
    r = requests.post(ALAMAT, json=payload, timeout=10)

    if r.status_code == 200:
        prediksi = r.json()["predictions"][0]
        tegas_churn = "CHURN" if prediksi >= 0.5 else "TIDAK CHURN"
        label_asli = "churn" if contoh["churn_sebenarnya"] == 1 else "tidak churn"
        cocok = "<- cocok" if (prediksi >= 0.5) == (contoh["churn_sebenarnya"] == 1) else "<- meleset"
        print("contoh %d: prediksi = %-12s | label asli: %-11s %s" % (i, tegas_churn, label_asli, cocok))
    else:
        print("contoh %d: request gagal, status %d" % (i, r.status_code))
        print(r.text[:300])

print("-" * 60)
print("selesai.")
