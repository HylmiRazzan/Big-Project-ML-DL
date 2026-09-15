# [Brazilian E Commerce Public Dataset]
<img width="2486" height="1496" alt="image" src="https://github.com/user-attachments/assets/e8990e5d-caaa-46c7-9e3d-acf39aad9fea" />


## 1. Deskripsi Dataset
Proyek ini menggunakan **[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)** yang tersedia di Kaggle. 

Dataset ini berisi data pesanan e-commerce komersial nyata di Brasil dari tahun 2016 hingga 2018. Data ini sangat lengkap, memungkinkan analisis pesanan dari berbagai dimensi (waktu, lokasi, produk, ulasan, dll).

### Pemilihan Dataset untuk Prediksi Review Score
Meskipun dataset asli terdiri dari banyak tabel relasional, tidak semua tabel digunakan dalam pembuatan model prediksi ini. Untuk menjaga efisiensi dan fokus pada fitur-fitur yang paling relevan dengan kepuasan pelanggan, berikut adalah rincian dataset yang dipilih beserta hubungannya dengan target prediksi (`review_score`):

*   **`olist_order_reviews_dataset.csv`**: Ini adalah dataset utama karena berisi variabel target prediksi kita, yaitu `review_score` (skala 1 sampai 5).
*   **`olist_orders_dataset.csv`**: Memuat status pesanan dan semua detail waktu (tanggal pembelian, pengiriman, hingga barang diterima). Durasi pengiriman atau keterlambatan merupakan salah satu faktor paling krusial yang menentukan ulasan pembeli.
*   **`olist_order_items_dataset.csv`**: Menyediakan detail harga produk dan biaya ongkos kirim. Rasio antara ongkos kirim dibandingkan dengan harga barang sering kali memengaruhi ekspektasi dan tingkat kepuasan pelanggan.
*   **`olist_products_dataset.csv`**: Menambahkan konteks kategori produk. Pelanggan biasanya memiliki standar ekspektasi dan pengalaman berbelanja yang berbeda-beda tergantung pada jenis kategori barang yang dibeli (misalnya elektronik vs. perabotan rumah tangga).
*   **`olist_order_payments_dataset.csv`**: Menunjukkan informasi metode pembayaran (kartu kredit, boleto, dll) dan jumlah cicilan. Ini membantu model melihat pola kepuasan berdasarkan perilaku finansial pelanggan.
*   **`olist_customers_dataset.csv`**: Memberikan informasi demografi wilayah pelanggan (kota dan provinsi). Data tingkat wilayah ini sudah cukup untuk menangkap pola kepuasan yang mungkin berbeda di setiap area.

### Dataset yang Tidak Digunakan (Excluded Data)
Untuk mencegah model mempelajari data yang kurang relevan (noise) dan mengurangi beban komputasi, dua dataset berikut dikeluarkan dari proses analisis:

1.  **`olist_geolocation_dataset.csv`**: Tabel ini berisi titik koordinat peta (garis lintang dan garis bujur) dengan ukuran data yang sangat besar. Informasi tingkat koordinat terlalu mendetail dan berlebihan, karena data kota dan provinsi dari dataset pelanggan sudah lebih dari cukup untuk menangkap pola lokasi.
2.  **`olist_sellers_dataset.csv`**: Tabel ini mencatat lokasi spesifik penjual. Pada platform e-commerce, pembeli pada umumnya tidak mempedulikan dari mana penjual berasal. Fokus utama pembeli adalah kecepatan pengiriman barang dan kesesuaian produk. Variabel kecepatan pengiriman sudah tertutupi sepenuhnya oleh tabel `orders` yang mencatat detail waktu.

## 2. Workflow (Alur Kerja)

Proyek ini dikerjakan secara sistematis melalui beberapa tahapan utama, mulai dari pemrosesan data mentah hingga pembuatan model klasifikasi yang kuat. Berikut adalah rincian langkah-langkahnya:

*   **Import Library:** Menggunakan pustaka standar manipulasi data (Pandas, Numpy), visualisasi (Seaborn, Matplotlib), serta pustaka *Machine Learning* tingkat lanjut (Scikit-Learn, XGBoost, LightGBM, Optuna).
*   **Penggabungan Data (Data Merging):** Menggabungkan 6 tabel CSV yang berbeda ke dalam satu tabel utama (`df_master`) menggunakan metode *left join*. Tujuannya agar seluruh konteks informasi (pesanan, produk, pelanggan, pembayaran, dan ulasan) berada dalam satu tempat untuk dianalisis.
*   **Pembersihan Data (Data Cleaning):** Menghapus kolom ID yang tidak berguna untuk prediksi (seperti `order_id`, `customer_id`) agar model tidak mempelajari pola yang salah. Selain itu, baris yang memiliki nilai kosong (*missing values*) pada kolom esensial juga dibuang.
*   **Rekayasa Fitur (Feature Engineering):** Mengekstrak dan membuat fitur baru dari data yang sudah ada untuk memberikan informasi lebih kaya kepada model. Beberapa fitur yang dibuat antara lain:
    *   **Fitur Waktu:** Menghitung durasi pengiriman, selisih hari keterlambatan pengiriman, hingga deteksi apakah pembelian dilakukan di akhir pekan.
    *   **Fitur Harga:** Membuat rasio perbandingan ongkos kirim terhadap harga barang (*freight to price ratio*).
    *   **Fitur Dimensi:** Menghitung total volume produk dalam sentimeter kubik.
    *   **Fitur Teks:** Menghitung panjang karakter dan jumlah kata dari komentar ulasan pembeli.
*   **Penentuan Variabel Target:** Mengubah skor ulasan asli (skala 1 sampai 5) menjadi 3 kategori sentimen utama, yaitu:
    *   **Kelas 0:** Kecewa (Skor 1 dan 2)
    *   **Kelas 1:** Netral (Skor 3)
    *   **Kelas 2:** Puas (Skor 4 dan 5)
*   **Pembagian Data (Train-Test Split):** Membagi data menjadi 80% data latih (*training*) dan 20% data uji (*testing*). Pembagian dilakukan secara *stratified* agar proporsi kelas Kecewa, Netral, dan Puas terdistribusi secara seimbang di kedua data.
*   **Transformasi Data (Preprocessing Pipeline):** Menggunakan `ColumnTransformer` untuk memproses berbagai jenis tipe data secara otomatis:
    *   **Numerik Diskrit:** Distandardisasi menggunakan `RobustScaler` untuk menangani *outlier*.
    *   **Numerik Kontinu:** Dinormalkan distribusinya menggunakan `PowerTransformer` (metode Yeo-Johnson).
    *   **Kategorik:** Diubah menjadi angka biner menggunakan `OneHotEncoder`.
    *   **Teks:** Diubah menjadi representasi matriks angka menggunakan `TfidfVectorizer` (dibatasi 1000 kata penting).
*   **Penanganan Data Tidak Seimbang:** Menerapkan pembobotan kelas (`class_weight='balanced'` dan `compute_sample_weight`) agar model memberikan perhatian ekstra pada kategori minoritas (misalnya kategori Netral atau Kecewa) dan tidak bias pada mayoritas.
*   **Evaluasi Model Baseline (Linear vs Non-Linear):** Sebelum melakukan optimasi penuh, dilakukan pengujian awal untuk membandingkan karakteristik dataset.
    *   F1 Macro (Logistic Regression / Linear) : 0.6643
    *   F1 Macro (XGBoost / Non-Linear) : 0.6795
    *   *Insight:* Meskipun margin performanya relatif tipis, evaluasi *baseline* ini menunjukkan dataset lebih condong berkarakter non-linear. Oleh karena itu, pendekatan pemodelan selanjutnya diprioritaskan menggunakan algoritma *tree-based* untuk menangkap kompleksitas pola yang tidak dapat dipetakan oleh model linear.
*   **Pemodelan Lanjutan (Stacking Ensemble & Optuna Tuning):** Model dibangun dalam dua level (Stacking) untuk memaksimalkan performa:
    *   **Level 1 (Base Models):** Melatih dua algoritma *tree-based* yang kuat yaitu **LightGBM** dan **XGBoost**. Parameter terbaik dari kedua model ini dicari secara otomatis menggunakan **Optuna**. Prediksi dari model level 1 ini dibuat menggunakan metode *Out-of-Fold* (OOF).
    *   **Level 2 (Meta-Learner):** Menggunakan **Logistic Regression** untuk mempelajari hasil prediksi dari LightGBM dan XGBoost, lalu mengambil keputusan akhir. Nilai parameter Meta-Learner juga dioptimasi dengan Optuna.
*   **Penyimpanan Model:** Mengekspor gabungan *pipeline preprocessing* dan model *stacking* ke dalam file `.pkl` menggunakan `joblib` agar siap untuk tahap *deployment*.

## 3. Metrik Penilaian

Karena dataset ulasan pelanggan pada umumnya tidak seimbang (pembeli yang puas biasanya jauh lebih banyak daripada yang kecewa atau netral), menggunakan metrik Akurasi saja bisa sangat menyesatkan.

Oleh karena itu, proyek ini menggunakan metrik **F1-Score (Macro Average)** sebagai tolak ukur utama.
*   **F1-Score** memberikan keseimbangan antara *Precision* (kemampuan model untuk tidak salah menebak) dan *Recall* (kemampuan model untuk menemukan seluruh data pada kelas tersebut).
*   **Macro Average** memastikan bahwa setiap kelas (Kecewa, Netral, Puas) dianggap sama pentingnya. Model dipaksa untuk harus pintar menebak kategori minoritas, bukan hanya menebak mayoritas saja.
*   **Classification Report** juga dievaluasi untuk melihat rincian performa presisi dan recall secara spesifik di tiap-tiap kategori.

## 4. Hasil Akhir

Setelah melewati proses *Hyperparameter Tuning* dengan Optuna dan menggunakan arsitektur Stacking, model berhasil dilatih dengan performa yang sangat stabil:

*   **Skor F1 Macro (Data Train):** 0.7332
*   **Skor F1 Macro (Data Test):** 0.7491
*   **Status Overfitting (Gap):** 0.0158 (Tidak ada *overfitting*)
*   **Akurasi Keseluruhan:** 85% (0.85)

## 5. Kesimpulan dan Kelayakan Model

Sistem *Machine Learning* yang telah dibangun ini berada dalam kondisi **layak untuk langsung diterapkan (Production-Ready)** ke dalam lingkungan operasional.

Penilaian kelayakan ini bertumpu pada ketajaman performa model dalam mendeteksi **dua kelas bisnis yang paling krusial**, yaitu pelanggan yang kecewa dan pelanggan yang puas.

### 1. Deteksi Pelanggan Kecewa (Bintang 1 dan 2)
*   **Recall:** 87% - 88%
*   **Precision:** 85% - 86%

Model memiliki daya tangkap yang sangat tinggi terhadap pelanggan yang berpotensi memberikan ulasan buruk. Dengan nilai *Recall* mencapai 88%, hampir **9 dari 10 keluhan kritis** dapat terdeteksi secara otomatis. Hal ini menjadikan model sebagai instrumen **pencegahan churn** (kehilangan pelanggan) yang sangat efektif. Tim layanan pelanggan dapat langsung melakukan intervensi (seperti meminta maaf atau memberi kompensasi) sebelum kekecewaan tersebut berkembang menjadi masalah yang lebih besar.

### 2. Deteksi Pelanggan Puas (Bintang 4 dan 5)
*   **Precision:** 96%
*   **Recall:** 88% - 89%

Pola kepuasan pelanggan juga berhasil dipetakan dengan sangat akurat. Tingkat *Precision* yang mencapai 96% menunjukkan bahwa ketika model memprediksi seorang pelanggan merasa puas, tebakannya nyaris selalu tepat sasaran. Informasi ini sangat berharga untuk membantu perusahaan mengidentifikasi dan mempertahankan strategi, layanan, atau produk yang sudah terbukti memiliki performa unggul di mata pembeli.

### Konklusi Akhir

Secara keseluruhan, proses kalibrasi akhir menggunakan algoritma pencarian **Optuna** pada arsitektur **Stacking Ensemble** telah menghasilkan fondasi prediksi yang sangat solid dengan:

> **Akurasi Global: 85% - 86%**

Berbekal kombinasi metrik teknis yang stabil dan nilai bisnis yang sangat terarah, model klasifikasi sentimen ini dinilai sukses dan **sangat siap untuk di-deploy ke tahap produksi (Production-Ready)**.

Berikut adalah rincian *Classification Report* dari performa model pada data uji (*test set*):

```text
                 precision    recall  f1-score   support

0 (Kecewa: 1-2)       0.86      0.88      0.87      2568
1 (Netral: 3)         0.39      0.58      0.46       872
2 (Puas: 4-5)         0.96      0.88      0.92      6034

       accuracy                           0.85      9474
      macro avg       0.73      0.78      0.75      9474
   weighted avg       0.88      0.85      0.86      9474
