Cara menjalankan monitoring dan logging (kriteria 4)

Persiapan, install dulu yang dibutuhkan:
pip install prometheus-client psutil requests mlflow

1. Serving model
Dari folder Membangun_model jalankan:
mlflow models serve -m "model" -p 5000 --env-manager=local
Setelah hidup, coba kirim request dengan: python 7.inference.py
Buktinya ada di folder 1.bukti_serving
File sample_payload.json dipakai oleh 7.inference.py sebagai contoh input.

2. Exporter dan Prometheus
Keduanya dijalankan di terminal yang terpisah:
python 3.prometheus_exporter.py
prometheus --config.file=2.prometheus.yml
Halaman metrik ada di http://localhost:8000/metrics
dan halaman target ada di http://localhost:9090/targets
Buktinya ada di folder 4.bukti monitoring Prometheus

Metrik yang diekspor ada 13:
model_requests_total, model_request_errors_total,
model_response_time_seconds, model_last_response_time_seconds,
model_last_prediction, model_churn_prediction_ratio,
model_health_status, model_exporter_uptime_seconds,
system_cpu_usage_percent, system_memory_usage_percent,
system_disk_usage_percent, input_tenure_mean,
input_monthly_charges_mean

3. Grafana
Buka http://localhost:3000, tambahkan data source Prometheus
dengan url http://localhost:9090, lalu buat dashboard bernama
anggarabudiyantoo dan tambahkan panel untuk metrik di atas.
Buktinya ada di folder 5.bukti monitoring Grafana

4. Alerting
Di menu Alerting dibuat 6 rule, yaitu model mati, ada request
gagal, respon lambat, cpu tinggi, ram tinggi, dan rasio churn
tinggi. Semua diarahkan ke contact point bukti-alert.
Supaya terlihat menyala, serving dihentikan sebentar lalu alert
model mati dan request gagal menyala merah.
Buktinya ada di folder 6.bukti alerting Grafana
