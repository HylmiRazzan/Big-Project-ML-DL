# [DANA Sentiment Analysis from Playstore Indonesia]

## 1. Deskripsi Dataset

Proyek ini menggunakan dataset ulasan pengguna dari aplikasi dompet digital DANA yang diambil dari Google Play Store (`review_dana_labelled.csv`). Dataset ini secara spesifik berfokus pada pemrosesan bahasa alami (Natural Language Processing) dan terdiri dari dua informasi utama:

*   **`content`**: Berisi teks mentah dari ulasan atau komentar pengguna.
*   **`sentimen`**: Berisi label kategori dari ulasan tersebut (contoh: POSITIVE atau NEGATIVE).

Tantangan utama dari dataset teks ulasan aplikasi adalah tingginya tingkat "noise" atau gangguan linguistik. Pengguna sering kali menyingkat kata, menggunakan bahasa gaul (slang), salah ketik (typo), atau menggunakan pola perulangan kata dengan angka (contoh: "bagus2" atau "mantap2"). Oleh karena itu, diperlukan teknik pembersihan teks tingkat lanjut sebelum mesin dapat memahami pola dari kata-kata tersebut.

## 2. Workflow (Alur Kerja)

Proyek ini dieksekusi melalui beberapa tahapan yang logis, mulai dari eksplorasi data hingga pemodelan tingkat lanjut menggunakan arsitektur *Stacking*. Berikut adalah penjelasan cara kerja dari setiap tahapan pemrosesannya:

*   **Eksplorasi Data (EDA):** Menganalisis distribusi panjang karakter dari setiap teks ulasan dan menghitung frekuensi kata yang paling sering muncul. Langkah ini berguna untuk memetakan kata ganti atau singkatan (slang) apa saja yang paling sering dipakai oleh pengguna DANA.
*   **Pembersihan Teks (Text Preprocessing):** Membuat sebuah fungsi kustom bernama `clean_text_dana` yang bekerja baris demi baris untuk membersihkan teks:
    *   Mengubah seluruh huruf menjadi kecil (*lowercase*).
    *   Menghapus tautan (URL) dan format HTML.
    *   Menjabarkan perulangan kata yang menggunakan angka (misalnya "bagus2" diubah secara otomatis menjadi "bagus bagus").
    *   Menghapus karakter non-alfanumerik (seperti emoji dan tanda baca ekstrem).
    *   Menerjemahkan singkatan atau bahasa gaul ke dalam Bahasa Indonesia baku menggunakan kamus kustom (misalnya "tf" menjadi "transfer", "wd" menjadi "tarik", "apk" menjadi "aplikasi").
*   **Pembagian Dataset:** Membagi keseluruhan data menjadi tiga bagian, yaitu 80% Data Latih (*Train*), 10% Data Validasi (*Validation*), dan 10% Data Uji (*Test*). Pembagian menggunakan metode *stratified* agar proporsi sentimen tetap seimbang di ketiga bagian data tersebut.
*   **Ekstraksi Fitur Teks (Sentence Embeddings):** Teks yang sudah bersih tidak lagi diproses menggunakan TF-IDF konvensional, melainkan menggunakan **Sentence Transformers** (`paraphrase-multilingual-MiniLM-L12-v2`). Teknologi ini mengubah teks menjadi vektor angka (embedding) dengan memperhatikan konteks makna kata dalam berbagai bahasa, sehingga pemahaman model terhadap sentimen jauh lebih dalam.
*   **Pembuatan Kelas Model Kustom (BalancedXGB):** Untuk mengatasi ketidakseimbangan kelas dan tingginya urgensi bisnis dalam mendeteksi ulasan buruk, sebuah algoritma *XGBoost* kustom diciptakan. Model ini dimodifikasi dengan penambahan parameter `faktor_negatif` agar penalti kesalahan saat menebak sentimen "Negatif" dihitung jauh lebih berat dibandingkan sentimen lainnya.
*   **Optimasi Hyperparameter (Optuna Level 1):** Melakukan pencarian kombinasi parameter terbaik secara otomatis untuk dua model dasar, yaitu XGBoost Kustom dan Logistic Regression. Pencarian ini divalidasi menggunakan metode *Stratified K-Fold* (3 lipatan) agar hasilnya konsisten.
*   **Pemodelan Akhir (Stacking Ensemble):** Prediksi dari XGBoost dan Logistic Regression pada Level 1 digabungkan. Kemudian, sebuah model *Meta-Learner* (Logistic Regression) dilatih pada Level 2 menggunakan Optuna untuk mengambil keputusan akhir yang paling optimal berdasarkan bobot dari kedua model dasar tersebut.

## 3. Metrik Penilaian

Karena tujuan utama bisnis dalam analisis ulasan aplikasi adalah menemukan dan merespons keluhan pengguna, evaluasi tidak hanya berfokus pada **F1-Score (Macro Average)**. 

Proyek ini juga menyoroti metrik **Recall untuk kelas Negatif**. Hal ini sangat krusial karena perusahaan lebih baik salah mendeteksi ulasan netral/positif sebagai keluhan (False Positive), daripada melewatkan keluhan kritis dari pengguna yang kecewa (False Negative).

Selain itu, evaluasi kesehatan model diukur melalui **Recall Negative Gap**, yaitu selisih nilai Recall kelas negatif antara Data Latih dan Data Uji. Jika selisihnya terlalu jauh (di atas 0.05), model akan otomatis dilabeli sebagai *Overfit*. 

## 4. Hasil Akhir

Setelah melalui proses kalibrasi tingkat lanjut, model *Stacking Classifier* diuji menggunakan data yang belum pernah dilihat sebelumnya (Data Uji). Berikut adalah performa akhirnya:

**Distribusi Bobot Meta-Learner:**
*   **XGBoost:** 18.49%
*   **Logistic Regression:** 14.85%
*(Catatan: Angka ini merepresentasikan penyerapan bobot secara langsung oleh model).*

## 5. Kesimpulan dan Nilai Bisnis

Berdasarkan hasil pengujian akhir, sistem *Machine Learning* ini menunjukkan performa yang sangat memuaskan dan berstatus **layak pakai (Production-Ready)**. 

Satu hal krusial yang ditekankan dalam evaluasi proyek ini adalah kehati-hatian terhadap metrik **Akurasi**. Meskipun akurasi global model mencapai 81%, metrik ini bisa sangat menipu pada dataset ulasan aplikasi. Jika sebuah model bertindak malas dan hanya menebak "Positif" pada semua ulasan karena kelas tersebut dominan, akurasinya akan tetap terlihat tinggi. Namun, model tersebut akan cacat secara bisnis karena sepenuhnya gagal mendeteksi pengguna yang mengalami masalah teknis.

Oleh karena itu, kesimpulan keberhasilan model ini dititikberatkan pada metrik **Recall untuk kelas Negatif (Keluhan)**:

*   **Keberhasilan Deteksi Keluhan (Fokus Recall):** Dengan modifikasi penalti pada algoritma *BalancedXGB*, model terbukti sangat sensitif dan berhasil mencapai tingkat **Recall 74%** pada kelas negatif. Artinya, sebagian besar pengguna yang melayangkan komplain berhasil dideteksi dan tidak terlewatkan oleh sistem. Dari keluhan yang ditangkap tersebut, tingkat ketepatan tebakannya (*Precision*) juga sangat solid di angka 80%.
*   **Kesehatan Model (Good Fit):** Jarak selisih (*gap*) performa Recall kelas negatif antara data latih (0.7927) dan data uji (0.7446) hanya sebesar 0.0481. Angka selisih yang sangat kecil ini berada di bawah batas kritis, membuktikan bahwa model tidak *overfit*. Model benar-benar belajar mengenali makna konteks bahasa dan singkatan khas pengguna DANA berkat *Sentence Transformers*, bukan sekadar menghafal.
*   **Keandalan Sentimen Positif:** Kelas positif berhasil diprediksi dengan tingkat presisi yang luar biasa, yaitu 94%. Sistem memiliki keyakinan dan kepastian yang sangat tinggi saat menyortir ulasan yang murni berupa pujian tanpa ragu.
*   **Tantangan Sentimen Netral:** Seperti karakteristik analisis bahasa pada umumnya, sentimen netral adalah yang paling sulit diidentifikasi (F1-Score 59%). Ulasan netral sering kali sangat ambigu, terpotong, atau merupakan campuran antara kepuasan dan keluhan teknis di saat yang bersamaan.

**Konklusi Akhir:**
Nilai jual utama dari model ini bukanlah pada seberapa tinggi akurasi keseluruhannya, melainkan pada seberapa tajam kemampuannya menyaring keluhan pelanggan di tengah lautan ulasan positif. Dengan sensitivitas *Recall* yang tinggi dan bebas dari gejala *overfitting*, sistem ini adalah instrumen yang sangat bernilai bagi tim *Customer Service*. Implementasi model ini memungkinkan perusahaan untuk mendeteksi dan merespons keluhan pengguna DANA secara otomatis, terarah, dan jauh lebih cepat.

**Performa Evaluasi pada Data Uji (Test Data):**
```text
                 precision    recall  f1-score   support

    NEGATIVE       0.80      0.74      0.77      1707
     NEUTRAL       0.48      0.76      0.59       637
    POSITIVE       0.94      0.86      0.90      2656

    accuracy                           0.81      5000
   macro avg       0.74      0.79      0.75      5000
weighted avg       0.84      0.81      0.82      5000
