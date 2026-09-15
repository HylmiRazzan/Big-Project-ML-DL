#[Natural Disaster Image Classification Using Convolal-Neural-Networks(CNN)]
## 1. Deskripsi Dataset

Proyek ini menggunakan dataset berupa kumpulan citra (gambar) yang mendokumentasikan berbagai kejadian bencana alam. Tujuan utama dari proyek ini adalah membangun model Convolutional Neural Network (CNN) yang mampu mengklasifikasikan gambar secara otomatis ke dalam empat kategori bencana.

Karakteristik utama dan sekaligus tantangan terbesar dari dataset ini adalah masalah ketidakseimbangan data (*class imbalance*) yang sangat signifikan. Berikut adalah rincian distribusi jumlah gambar untuk setiap kelas:

| Kelas Data | Jumlah Gambar |
| :--- | :--- |
| Water Disaster | 1035 |
| Landslide | 456 |
| Urban Fire | 419 |
| Earthquake | 36 |

Ketimpangan data yang sangat ekstrem terlihat pada kelas *Earthquake* yang hanya memiliki 36 sampel gambar, berbanding jauh dengan *Water Disaster* yang mendominasi dengan lebih dari 1000 gambar. Kondisi ini menuntut penerapan teknik augmentasi atau penyeimbangan data khusus pada alur kerja agar model tidak bias hanya pada kelas mayoritas.

## 2. Workflow (Alur Kerja)

Proyek ini dieksekusi melalui serangkaian tahapan pemrosesan citra dan pemodelan yang sangat terstruktur. Mengingat adanya tantangan data yang tidak seimbang, berikut adalah rincian langkah kerja yang dilakukan:

*   **Pembersihan Data (Data Cleaning):** Proses pembersihan dilakukan langsung pada penyimpanan asal di Google Drive. Langkah ini mencakup penghapusan data duplikat, file rusak (korup), gambar kosong, serta pembuangan 19 file pencilan (outlier) warna RGB yang terdeteksi saat tahap eksplorasi data (EDA). Pembersihan kelas *Earthquake* dilakukan dengan sangat hati-hati agar tidak mengurangi jumlah sampelnya yang sudah sangat sedikit.
*   **Injeksi Dataset (Data Loading):** Gambar ditarik ke dalam memori Google Colab menggunakan modul pembaca data dari TensorFlow. Pada tahap ini, fitur `crop_to_aspect_ratio` diaktifkan untuk memotong gambar menjadi bentuk persegi dengan resolusi 224x224 piksel. Fitur ini memastikan ukuran gambar seragam tanpa merusak proporsi objek aslinya.
*   **Pembagian Data (Train-Test Split):** Total gambar dipilah menjadi tiga bagian, yaitu 70% Data Latih (*Training*), 15% Data Validasi (*Validation*), dan 15% Data Uji (*Testing*). Khusus untuk Data Latih, urutan gambar diacak ulang (*reshuffle*) setiap kali pergantian *epoch* agar model benar-benar belajar mengenali pola visual, bukan sekadar menghafal urutan data yang masuk.
*   **Normalisasi dan Augmentasi:** Nilai kecerahan piksel pada gambar diperkecil ukurannya (*rescaling*) agar berada di rentang angka 0.0 hingga 1.0 untuk memperingan komputasi. Data Latih kemudian diberikan variasi visual buatan (augmentasi) seperti pembalikan gambar secara horizontal, rotasi acak sebesar 10%, dan perbesaran (*zoom*) 10%. Proses ini dipercepat menggunakan instruksi paralelisasi `AUTOTUNE` dan *prefetching* dari TensorFlow.
*   **Strategi Penanganan Data Tidak Seimbang:** Untuk mencegah model menjadi bias dan hanya menebak kelas mayoritas (*Water Disaster*), dua skenario diuji secara bergantian:
    *   **Class Weighting:** Menggunakan fungsi dari scikit-learn untuk memodifikasi penalti kesalahan. Kelas minoritas (seperti gempa bumi) diberikan bobot penalti yang jauh lebih berat sehingga model dipaksa untuk lebih memperhatikannya saat proses belajar.
    *   **Oversampling:** Menggandakan dan menyusun ulang gambar dari kelas minoritas ke dalam *batch* baru, sehingga jumlah gambar keempat kelas bencana bernilai setara saat dimasukkan ke dalam model.
*   **Arsitektur Model (Modelling):** Pelatihan diuji coba melalui dua pendekatan komputasi yang berbeda:
    *   **Scratch Model (CNN Murni):** Model dirakit dari awal menggunakan 4 lapisan tersembunyi (*hidden layer*) dengan jumlah neuron 32, 64, 128, dan 64. Model ini didukung oleh fungsi aktivasi ReLU, *Max Pooling*, dan Dropout 0.2 untuk mencegah hafalan buta. Total parameter yang dilatih mencapai sekitar 5,6 juta.
    *   **Transfer Learning (EfficientNetB1):** Menggunakan model yang sudah cerdas karena pernah dilatih pada dataset raksasa ImageNet. Bobot kecerdasan bawaannya dikunci (*frozen*), lalu ditambahkan lapisan *Global Average Pooling*, *Batch Normalization*, dan lapisan klasifikasi baru di bagian akhir. Total parameternya dijaga agar tetap efisien di bawah 10 juta.
*   **Optimasi Pelatihan:** Kedua model tersebut dilatih menggunakan algoritma optimasi Adam dengan *Learning Rate* 0.001 dan metrik evaluasi *Loss Sparse Categorical Crossentropy*. Untuk menjaga agar model tidak mengalami *overfitting* atau jalan di tempat, mekanisme *callback* diaktifkan. Fitur `ReduceLROnPlateau` digunakan untuk menurunkan *learning rate* saat performa stagnan, dan `EarlyStopping` digunakan untuk menghentikan pelatihan otomatis jika tidak ada perbaikan performa.

## 3. Metrik Penilaian

Dalam konteks tanggap darurat bencana alam, pemilihan metrik evaluasi klasifikasi gambar sangatlah krusial. Proyek ini tidak menggunakan metrik **Accuracy** karena hasilnya akan sangat menipu pada kondisi persebaran data yang sangat timpang (*imbalanced data*). 

Fokus evaluasi juga tidak dijatuhkan secara tunggal pada metrik berikut:
*   **Recall:** Jika hanya berfokus pada *Recall*, model akan menjadi terlalu sensitif dan berpotensi besar mengklasifikasikan foto kejadian biasa sebagai bencana (*False Positive*). Hal ini akan membuang waktu dan sumber daya tim penyelamat yang terbatas.
*   **Precision:** Sebaliknya, jika hanya berfokus pada *Precision*, model menjadi terlalu berhati-hati. Foto bencana sungguhan yang visualnya kurang jelas (abu-abu) berpotensi diabaikan (*False Negative*), sehingga penyaluran bantuan kepada korban akan terlambat.

Oleh karena itu, proyek ini menggunakan **F1-Score (Macro Average)** untuk mendapatkan keseimbangan prediksi yang optimal:
*   **F1-Score:** Memberikan titik keseimbangan yang adil antara *Precision* dan *Recall*.
*   **Macro Average:** Pendekatan *Weighted Average* dihindari karena metrik tersebut akan didominasi oleh performa kelas mayoritas (*Water Disaster*), yang berisiko menutupi kelemahan fatal pada kelas minoritas (*Earthquake*). *Macro Average* memastikan setiap jenis bencana dianggap sama pentingnya, menuntut model untuk benar-benar andal di seluruh kategori.

## 4. Hasil Akhir dan Analisis

Penelitian ini menguji empat pendekatan pemodelan (*Scratch* vs *Transfer Learning*, dikombinasikan dengan *Class Weight* vs *Oversampling*). Berdasarkan evaluasi akhir, **Transfer Learning (EfficientNetB1) menggunakan metode Class Weight** terbukti menjadi model yang paling stabil dan dapat diandalkan.

Metode *Oversampling* pada kasus ini terbukti hanya menghasilkan ilusi performa karena *overfitting*. Meskipun kurva pelatihannya terlihat stabil, model gagal melakukan generalisasi dan hanya menghafal data duplikat pada kelas minoritas. Hal ini terlihat dari anjloknya nilai *Precision* kelas *Earthquake* menjadi 0.28 pada metode *Oversampling* akibat tingginya prediksi salah (*False Positive*).

## 5. Kesimpulan dan Rekomendasi

Secara keseluruhan, model **Transfer Learning (EfficientNetB1) dengan metode Class Weight** dinilai sudah cukup solid dan **layak untuk diimplementasikan (Production-Ready)**. 

Namun, evaluasi mendalam pada penelitian ini menyoroti beberapa catatan penting terkait karakteristik dataset dan proses pembelajaran model:

*   **Kendala Ketimpangan Data:** Tantangan paling mendasar dari proyek ini adalah ketidakseimbangan kelas yang sangat ekstrem. Meskipun metode pembobotan kelas (*class weighting*) berhasil menstabilkan performa, jurang jumlah data yang terlalu besar tetap menjadi batasan nyata. Model masih harus berjuang keras untuk mengekstrak pola visual secara maksimal dari kelas minoritas (terutama *Earthquake*).
*   **Fenomena Pelandaian (Plateau):** Proses pelatihan model terpantau dengan cepat mencapai fase *plateau* atau stagnan. Hal ini sangat wajar terjadi karena *EfficientNetB1* merupakan model *pre-trained* dari *ImageNet* yang sudah menguasai deteksi fitur visual dasar (seperti garis tepi, tekstur, dan transisi warna). Model ini belajar dengan sangat cepat, menyerap seluruh fitur relevan dari dataset bencana yang kita miliki, lalu mencapai batas maksimalnya karena kehabisan variasi pola baru untuk dipelajari.

### Rekomendasi Pengembangan Lanjutan

Untuk menekan bias prediksi antar kelas dan meningkatkan akurasi model di masa depan, sangat disarankan untuk melakukan beberapa langkah perbaikan berikut:

*   **Pengumpulan Data yang Seimbang:** Mengumpulkan lebih banyak sampel gambar otentik, khususnya untuk kelas minoritas yang tertinggal jauh. Model perlu dilatih ulang (*retrain*) dengan komposisi dataset yang lebih berimbang untuk menghilangkan dominasi bias dari kelas mayoritas.
*   **Pemanfaatan Teknologi GAN:** Selain mengandalkan augmentasi gambar konvensional (seperti rotasi atau pembalikan), teknologi *Generative Adversarial Networks* (GAN) dapat dimanfaatkan untuk menciptakan gambar sintesis bencana yang sangat realistis. Pendekatan ini merupakan solusi cerdas untuk memperkaya variasi pola visual secara buatan, sehingga dapat menutupi kekurangan jumlah sampel pada kelas minoritas tanpa harus mencari foto kejadian asli di lapangan.

Berikut adalah rincian *Classification Report* dari performa model terbaik (Transfer Learning + Class Weight) pada data uji:

```text
                 precision    recall  f1-score   support

    Earthquake       0.50      0.57      0.53         7
    Land_Slide       0.67      0.81      0.73        57
    Urban_Fire       1.00      0.78      0.88        65
Water_Disaster       0.93      0.93      0.93       165

      accuracy                           0.87       294
     macro avg       0.77      0.77      0.77       294
  weighted avg       0.88      0.87      0.87       294
