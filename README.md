# Klasifikasi Berita — Skip-gram + Naive Bayes (UTS PPW)

Aplikasi web Streamlit dengan 4 tahap: Crawling → Preprocessing → Modeling (Word2Vec skip-gram) → Klasifikasi (Gaussian Naive Bayes).

## Jalankan lokal
```
pip install -r requirements.txt
streamlit run app.py
```
Buka tab 1, klik **Mulai crawling**, lalu tab 3 **Latih model**, lalu tab 4 untuk prediksi.

## Deploy gratis: Streamlit Community Cloud
1. Buat repo GitHub, upload isi folder ini (app.py, pipeline.py, requirements.txt).
2. Buka https://share.streamlit.io → New app → pilih repo, branch `main`, file `app.py` → Deploy.
3. Setelah online, jalankan Crawling dan Latih model dari dalam aplikasi.
   (Disk server bersifat sementara; model hilang saat app restart, latih ulang atau commit folder `model/` dan `data/`.)

## Alternatif: Hugging Face Spaces
Buat Space bertipe Streamlit, upload file yang sama.

## Jika crawling gagal
Struktur HTML/proteksi situs bisa berubah. Ubah selector di `pipeline.py` (`_artikel`, `_link_kategori`) atau upload CSV (kolom `judul, isi, kategori`) lewat tab Crawling.
