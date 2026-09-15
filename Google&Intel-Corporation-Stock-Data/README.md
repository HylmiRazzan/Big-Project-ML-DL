# [Google & Intel Corp Stock Data Long Short-Term Memory (LSTM)]

## 1. Deskripsi Dataset dan Lingkungan Proyek

Proyek ini berfokus pada pemodelan deret waktu (*Time Series Forecasting*) untuk memprediksi pergerakan harga saham dari dua perusahaan teknologi, yaitu **Google** dan **Intel Corp**. Seluruh komputasi dan pemodelan dalam proyek ini dieksekusi menggunakan platform **Kaggle Notebook**.

Dalam pengerjaannya, proyek ini menggunakan dua dataset historis saham yang berbeda (satu untuk Google dan satu untuk Intel). Meskipun dikerjakan di dalam satu *notebook* yang sama, **kedua dataset ini diolah secara sepenuhnya terpisah dan independen**. Pendekatan ini dilakukan agar model dapat mempelajari tren, volatilitas, dan pola spesifik dari masing-masing saham tanpa ada percampuran informasi.

Untuk menjaga model tetap fokus pada pergerakan harga, proses pemuatan data difilter dengan menjadikan kolom `Date` sebagai indeks waktu historis, dan hanya mengambil kolom `Close` (yang diubah namanya menjadi `Price`) sebagai variabel target prediksi.

## 2. Workflow (Alur Kerja) dan Penjelasan Logika Kode

Alur kerja proyek ini dirancang secara khusus untuk mempertahankan kronologi waktu dan dieksekusi secara berurutan. Pertama, seluruh proses diterapkan pada saham Google (GOOGL), lalu proses yang sama persis diulang untuk saham Intel (INTC). 

Berikut adalah rincian tahapan dan logika dari kode yang dijalankan:

*   **Inisialisasi Lingkungan (Setup):** 
    *   Modul dasar diimpor (Numpy, Pandas, Matplotlib, TensorFlow). 
    *   Nilai *seed* (42) ditetapkan pada Numpy, TensorFlow, dan Random agar hasil eksperimen tetap konsisten dan dapat direproduksi di masa depan. 
    *   Peringatan (warning) dari sistem TensorFlow disembunyikan menggunakan perintah `tf.get_logger().setLevel(logging.ERROR)` agar tampilan output di dalam *notebook* tetap bersih.
*   **Pemuatan dan Pembersihan Data:** 
    *   Fungsi `pd.read_csv` digunakan untuk membaca data. Argumen `parse_dates=["Date"]` dan `index_col=["Date"]` secara otomatis mengubah format tanggal menjadi format waktu baku dan menjadikannya sebagai indeks baris tabel.
    *   Kolom yang tidak diperlukan dibuang, menyisakan kolom `Close` yang kemudian diubah namanya menjadi `Price`.
*   **Pembagian Data Berdasarkan Waktu (Time-based Split):**
    *   Berbeda dengan klasifikasi biasa, data saham tidak boleh diacak. Kodingan akan mendeteksi tanggal paling akhir di dalam dataset, lalu mundur tepat 1 tahun menggunakan perintah `pd.DateOffset(years=1)`.
    *   Data sebelum batas 1 tahun tersebut disimpan sebagai data latih (`data_train_full`), sedangkan data 1 tahun terakhir disimpan sebagai data uji (`data_test`).
*   **Pembuatan Jendela Waktu (Windowing):**
    *   Kodingan membuat fungsi `create_windows` untuk membentuk struktur sekuensial (berurutan) yang wajib dimiliki oleh model LSTM.
    *   Menggunakan ukuran jendela (`window_size`) = 5 dan horison (`horizon`) = 1. Logikanya, model akan mengamati pola harga selama **5 hari terakhir** untuk memprediksi harga pada **1 hari ke depan**. Data diiris terus-menerus bergeser satu hari ke depan hingga akhir baris.
*   **Pemisahan Data Validasi (Train-Val Split):**
    *   Data historis (`data_train_full`) dipecah kembali. Sebesar 90% pertama dijadikan data latih sesungguhnya (`x_train`, `y_train`), dan 10% sisa ujungnya dijadikan data validasi (`x_val`, `y_val`). 
*   **Penyesuaian Dimensi (Reshaping):**
    *   Perintah `np.expand_dims(x, axis=-1)` digunakan untuk menambahkan satu dimensi baru pada susunan array data. Hal ini dilakukan karena arsitektur jaringan saraf LSTM mewajibkan bentuk input 3D, yaitu `(jumlah_sampel, langkah_waktu, jumlah_fitur)`.
*   **Skenario Pemodelan 1 (Baseline Model):**
    *   Model sederhana dibuat menggunakan struktur linear (`Sequential`). Terdiri dari satu lapis LSTM dengan 50 unit *neuron* (fungsi aktivasi ReLU), dan langsung dihubungkan ke 1 lapis Dense output untuk memprediksi angka akhir.
    *   Model dikompilasi menggunakan fungsi pengoptimal `adam`, fungsi hitung kerugian `mse` (Mean Squared Error), dan metrik evaluasi `mae` (Mean Absolute Error). 
    *   Dilatih sebanyak 20 putaran (epoch) dengan ukuran *batch* 32.
*   **Skenario Pemodelan 2 (Modified/Tuned Model):**
    *   Untuk meningkatkan akurasi dan mencegah hafalan buta (*overfitting*), arsitektur model dimodifikasi. 
    *   Sebuah lapisan `Dropout(0.1)` disisipkan. Lapisan ini akan mematikan 10% *neuron* secara acak saat belajar, sehingga model dipaksa untuk mencari pola yang lebih umum (general).
    *   Kecepatan belajar (*learning rate*) dari optimizer Adam diturunkan secara manual menjadi `0.0005` agar model belajar lebih hati-hati.
*   **Penerapan Callbacks:**
    *   `EarlyStopping`: Sistem akan menghentikan proses pelatihan secara otomatis jika nilai kesalahan pada data validasi (`val_loss`) tidak membaik selama 15 putaran berturut-turut. Ini menghemat waktu dan mencegah model menjadi terlalu bias pada data latih.
    *   `ReduceLROnPlateau`: Sistem akan memotong kecepatan belajar (*learning rate*) menjadi setengahnya (faktor 0.5) jika performa stagnan selama 5 putaran. Ini membantu model mencapai titik akurasi optimal dengan sangat halus.
    *   Model modifikasi ini kemudian dilatih dengan batas maksimal 150 putaran (epoch) dan ukuran *batch* yang lebih besar (64).

  ## 3. Metrik Penilaian

Karena proyek ini adalah kasus regresi (memprediksi angka pasti berupa harga), metrik yang digunakan bertujuan untuk mengukur seberapa besar selisih antara tebakan model dengan harga asli di pasar saham. Tiga metrik yang digunakan adalah:

*   **RMSE (Root Mean Squared Error):** Menghitung rata-rata dari kuadrat kesalahan. Metrik ini sangat sensitif dan memberikan penalti (hukuman) yang besar jika model melakukan tebakan yang meleset sangat jauh pada hari tertentu.
*   **MAE (Mean Absolute Error):** Menghitung rata-rata selisih absolut antara harga tebakan dan harga asli. Satuan MAE sama dengan satuan harga saham (Dolar). Metrik ini sangat mudah dipahami karena langsung menunjukkan rata-rata nominal uang yang meleset dari prediksi model.
*   **MAPE (Mean Absolute Percentage Error):** Menghitung persentase rata-rata kesalahan tebakan model. Metrik ini adalah tolok ukur yang paling adil untuk membandingkan performa model pada dua saham yang berbeda rentang harganya.

## 4. Hasil Akhir dan Analisis

Setelah melakukan pelatihan dan optimasi (menggunakan model yang telah dimodifikasi dengan *Dropout* dan *Callbacks*), prediksi model diuji pada data uji (1 tahun terakhir transaksi).

Berikut adalah hasil evaluasi performa model untuk masing-masing saham:

### Evaluasi Model Saham Google (GOOGL)
*   **Nilai RMSE:** 30.3023
*   **Nilai MAE:** 20.7450
*   **Nilai MAPE:** 1.69% (0.0169)

### Evaluasi Model Saham Intel Corp (INTC)
*   **Nilai RMSE:** 1.5269
*   **Nilai MAE:** 0.9597
*   **Nilai MAPE:** 1.81% (0.0181)

### Insight Evaluasi:
Jika hanya melihat angka secara umum, persentase kesalahan (MAPE) di bawah 2% (Google 1.69% dan Intel 1.81%) mungkin terlihat kecil dan stabil. Namun, jika dinilai dari kacamata praktisi atau *trader* di bursa saham, performa model ini sebenarnya **sangat berisiko dan tergolong buruk**. 

Dalam skenario perdagangan nyata (*real case*), target margin keuntungan harian seorang *trader* biasanya sangatlah tipis, yakni hanya berada di kisaran 0.5% hingga 1%. Jika tingkat melesetnya model (*error*) sudah melebihi 1%, maka penggunaan model ini menjadi sangat berbahaya. Selisih tebakan yang tidak akurat tersebut dapat dengan mudah menyapu bersih seluruh margin keuntungan dan justru menimbulkan kerugian finansial yang fatal.

---

## 5. Kesimpulan dan Kelayakan Model

Berdasarkan realitas bisnis dan toleransi risiko di pasar saham, model regresi LSTM yang dibangun pada proyek ini dinyatakan **tidak layak untuk di-deploy (Not Production-Ready)** ke dalam sistem perdagangan otomatis (*live trading*). 

Dunia perdagangan saham menuntut tingkat presisi yang jauh lebih tajam dan absolut. Model ini belum mampu memenuhi standar akurasi yang dibutuhkan untuk mengamankan profit harian secara konsisten.

### Saran Improvisasi dan Pengembangan Lanjutan
Untuk memperbaiki kelemahan model dan meningkatkan akurasinya hingga mencapai standar operasional di masa depan, berikut adalah beberapa langkah perbaikan teknis yang wajib dilakukan:

1.  **Penerapan Analisis Multivariat (Multivariate):** Model saat ini hanya belajar dari satu variabel historis (Univariate), yaitu harga penutupan (*Close*). Pergerakan pasar saham sangat kompleks, sehingga model membutuhkan lebih banyak fitur pendukung (Multivariate), seperti volume transaksi, pergerakan harga pembukaan (*Open*), harga tertinggi/terendah (*High/Low*), atau bahkan sentimen berita ekonomi eksternal.
2.  **Penanganan Distribusi Data Ekstrem:** Harga saham rentan terhadap volatilitas yang liar dan kejutan pasar. Diperlukan metode pengolahan data atau transformasi lanjutan untuk menjinakkan distribusi harga yang terlalu ekstrem, agar jaringan saraf pada model tidak kebingungan saat mencari pola normal (*baseline*).
3.  **Optimalisasi Penskalaan Data (Scaling):** Proses standardisasi atau normalisasi angka (*scaling*) pada data masukan harus dilakukan dengan perhitungan yang lebih presisi dan ketat. Penskalaan yang buruk pada data deret waktu dapat menyebabkan perhitungan matematis di dalam algoritma LSTM menjadi tidak stabil.
