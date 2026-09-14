## 1. Deskripsi Dataset

Proyek ini menggunakan kumpulan citra bencana alam yang bersumber dari Kaggle, yaitu **[Disaster Images Dataset](https://www.kaggle.com/datasets/varpit94/disaster-images-dataset)**. 

Meskipun sumber dataset tersebut menyediakan berbagai macam kategori gambar bencana secara keseluruhan, proses eksperimen pada proyek ini difokuskan secara khusus dengan hanya mengambil **4 kelas kategori yaitu Earthquake, Urban Fire, Landslide, Water Disaster**.

Berikut adalah rincian 4 kelas data yang dipilih beserta jumlah sampel gambarnya:

| Kelas Data | Jumlah Gambar |
| :--- | :--- |
| Water Disaster | 1035 |
| Landslide | 456 |
| Urban Fire | 419 |
| Earthquake | 36 |

Karakteristik utama dan tantangan terbesar dari dataset yang digunakan ini adalah masalah ketidakseimbangan data (*class imbalance*) yang sangat ekstrem. Ketimpangan terlihat jelas pada kelas *Earthquake* yang hanya memiliki 36 sampel gambar, berbanding sangat jauh dengan *Water Disaster* yang mendominasi hingga lebih dari 1000 gambar. Kondisi ini mendasari perlunya penerapan strategi khusus seperti pembobotan kelas (*class weighting*) agar model tidak bias terhadap kelas mayoritas.
