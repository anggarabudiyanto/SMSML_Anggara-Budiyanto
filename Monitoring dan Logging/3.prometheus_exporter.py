# 3.prometheus_exporter.py
# exporter prometheus untuk model churn yang sedang di-serve.
# tiap beberapa detik dikirim contoh data ke endpoint model, hasilnya dicatat
# jadi metrik, ditambah metrik sistem. total 13 metrik.
# metrik tampil di http://localhost:8000/metrics

import os
import json
import time
import threading

import requests
import psutil
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# pengaturan (bisa diubah lewat environment variable kalau perlu)
ALAMAT_MODEL = os.environ.get("MODEL_URL", "http://127.0.0.1:5000/invocations")
PORT_EXPORTER = int(os.environ.get("EXPORTER_PORT", "8000"))
JEDA_DETIK = 5  # jeda antar request test ke model

# ----------------- definisi metrik (13 buah) -----------------
requests_total = Counter("model_requests_total", "jumlah request prediksi yang dikirim ke model")
errors_total = Counter("model_request_errors_total", "jumlah request yang gagal")
response_time_hist = Histogram("model_response_time_seconds", "lama respon model (histogram)")
last_response_time = Gauge("model_last_response_time_seconds", "lama respon model yang terakhir")
last_prediction = Gauge("model_last_prediction", "prediksi terakhir (0 = tidak churn, 1 = churn)")
churn_ratio = Gauge("model_churn_prediction_ratio", "rasio prediksi churn dari 50 prediksi terakhir")
health_status = Gauge("model_health_status", "1 kalau endpoint model hidup, 0 kalau mati")
uptime_seconds = Gauge("model_exporter_uptime_seconds", "lama exporter ini berjalan")
cpu_usage = Gauge("system_cpu_usage_percent", "pemakaian cpu sistem (persen)")
memory_usage = Gauge("system_memory_usage_percent", "pemakaian ram sistem (persen)")
disk_usage = Gauge("system_disk_usage_percent", "pemakaian disk sistem (persen)")
tenure_mean = Gauge("input_tenure_mean", "nilai tenure dari input terakhir (indikator drift)")
monthly_charges_mean = Gauge("input_monthly_charges_mean", "nilai MonthlyCharges dari input terakhir (indikator drift)")

riwayat_prediksi = []  # untuk menghitung rasio churn berjalan
waktu_mulai = time.time()

# contoh data untuk request test (dari data uji asli)
with open("sample_payload.json") as f:
    semua_contoh = json.load(f)


def kirim_request_test():
    """tiap JEDA_DETIK kirim satu contoh ke model, catat hasilnya jadi metrik"""
    nomor = 0
    while True:
        contoh = semua_contoh[nomor % len(semua_contoh)]
        payload = {"dataframe_records": [contoh["fitur"]]}

        t0 = time.time()
        try:
            r = requests.post(ALAMAT_MODEL, json=payload, timeout=5)
            lama = time.time() - t0
            requests_total.inc()
            response_time_hist.observe(lama)
            last_response_time.set(lama)

            if r.status_code == 200:
                health_status.set(1)
                prediksi = r.json()["predictions"][0]
                prediksi = int(prediksi >= 0.5) if isinstance(prediksi, float) else int(prediksi)
                last_prediction.set(prediksi)

                riwayat_prediksi.append(prediksi)
                if len(riwayat_prediksi) > 50:
                    riwayat_prediksi.pop(0)
                churn_ratio.set(sum(riwayat_prediksi) / len(riwayat_prediksi))
            else:
                health_status.set(0)
                errors_total.inc()
        except Exception:
            # model mati / belum di-serve, tetap jalan supaya grafana kelihatan turun
            health_status.set(0)
            errors_total.inc()

        # simpan nilai input terakhir sebagai pembanding distribusi data (drift sederhana)
        tenure_mean.set(contoh["fitur"]["tenure"])
        monthly_charges_mean.set(contoh["fitur"]["MonthlyCharges"])

        nomor += 1
        time.sleep(JEDA_DETIK)


def update_metrik_sistem():
    """update metrik cpu/ram/disk dan uptime secara berkala"""
    while True:
        cpu_usage.set(psutil.cpu_percent())
        memory_usage.set(psutil.virtual_memory().percent)
        # os.sep biar jalan di windows (C:\) maupun linux (/)
        disk_usage.set(psutil.disk_usage(os.path.abspath(os.sep)).percent)
        uptime_seconds.set(time.time() - waktu_mulai)
        time.sleep(2)


if __name__ == "__main__":
    threading.Thread(target=kirim_request_test, daemon=True).start()
    threading.Thread(target=update_metrik_sistem, daemon=True).start()

    print("exporter prometheus jalan di: http://localhost:%d/metrics" % PORT_EXPORTER)
    print("memantau model di:", ALAMAT_MODEL)
    print("tekan ctrl+c untuk berhenti")

    start_http_server(PORT_EXPORTER)
    while True:
        time.sleep(1)
