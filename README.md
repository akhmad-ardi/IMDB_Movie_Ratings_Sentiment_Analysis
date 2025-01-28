# Laporan Proyek IMDB Movie Ratings Sentiment Analysis - Akhmad Ardiansyah Amnur

## Domain Proyek

Proyek ini berada dalam domain analisis sentimen, yang merupakan cabang dari pemrosesan bahasa alami (Natural Language Processing/NLP). Analisis sentimen bertujuan untuk mengidentifikasi dan mengkategorikan opini yang diekspresikan dalam sebuah teks, khususnya untuk menentukan apakah sikap penulis terhadap topik tertentu bersifat positif atau negatif. Dalam konteks proyek ini, kita akan menganalisis ulasan film dari IMDB untuk menentukan sentimen penonton terhadap film tersebut. Analisis ini dapat memberikan wawasan berharga bagi pembuat film, studio, dan pemasar untuk memahami persepsi publik dan mengarahkan strategi mereka berdasarkan umpan balik penonton.
  
Referensi:
-  [IMDB Movie Ratings Sentiment Analysis - Kaggle](https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis) 

## Business Understanding

IMDb (Internet Movie Database) adalah platform online yang sangat populer untuk mendapatkan informasi tentang film, acara TV, dan selebritas. Pengguna IMDb dapat memberikan rating dan menulis ulasan tentang film yang mereka tonton. Data ulasan ini sangat berharga karena dapat memberikan wawasan yang mendalam tentang persepsi publik terhadap suatu film.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana cara mengidentifikasi sentimen dari ulasan film di IMDB?
- Bagaimana tingkat akurasi model dalam mengklasifikasikan sentimen ulasan film sebagai positif atau negatif?
- Bagaimana model dapat membantu pembuat film dan pemasar dalam memahami persepsi publik terhadap film?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengembangkan model Deep Learning yang dapat mengidentifikasi sentimen dari ulasan film di IMDB.
- Mencapai tingkat akurasi yang tinggi dalam klasifikasi sentimen ulasan film.
- Memberikan wawasan yang dapat digunakan oleh pembuat film dan pemasar untuk memahami persepsi publik dan mengarahkan strategi mereka berdasarkan umpan balik penonton.

### Solution Statements

- Melakukan fine-tuning pada model bert-base-uncased untuk meningkatkan akurasi dalam mengklasifikasikan sentimen ulasan film.
- Menggunakan metrik evaluasi seperti akurasi, precision, recall, dan F1 score untuk mengukur kinerja model.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah dataset ulasan film dari IMDB yang tersedia di [Dataset IMDB movie ratigns sentiment anlysis Kaggle](https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis). Dataset ini berisi ulasan film yang telah diberi label sentimen positif atau negatif. Data ini sangat berguna untuk melatih model analisis sentimen karena mencakup berbagai opini penonton terhadap berbagai film.

### Variabel-variabel pada dataset IMDB adalah sebagai berikut:
- `text`: teks ulasan film yang diberikan oleh pengguna.
- `label`: label sentimen dari ulasan, yang dapat berupa 'positive' atau 'negative'.

## Data Preparation

Pada tahap data preparation, langkah pertama yang dilakukan adalah memuat dataset ulasan film dari IMDB. Setelah data dimuat, langkah berikutnya adalah melakukan pembersihan data (data cleaning) dengan menghapus ulasan yang kosong atau duplikat untuk memastikan kualitas data yang baik. Selanjutnya, dilakukan tokenisasi teks ulasan untuk memecah teks menjadi kata-kata atau token. Kemudian, dilakukan penghapusan stop words untuk menghilangkan kata-kata umum yang tidak memiliki makna signifikan dalam analisis sentimen. Teknik-teknik ini diterapkan secara berurutan dalam notebook dan dijelaskan secara rinci dalam laporan untuk memastikan replikasi dan pemahaman yang jelas.

## Modeling

Pada proyek ini, model machine learning yang digunakan adalah `bert-base-uncased`, sebuah model transformer yang telah dilatih sebelumnya oleh Google. Model ini dipilih karena performanya yang sangat baik dalam berbagai tugas NLP, termasuk analisis sentimen.

### Tahapan dan Parameter Pemodelan

1. **Fine-Tuning**: 
    - Model `bert-base-uncased` di-fine-tune menggunakan dataset ulasan film dari IMDB. Fine-tuning dilakukan untuk menyesuaikan model dengan karakteristik data ulasan film yang spesifik.
    - Proses fine-tuning melibatkan pelatihan ulang model pada dataset ulasan film dengan menggunakan label sentimen sebagai target. 

2. **Parameter yang Digunakan**:
    - **Learning Rate**: 3e-6
    - **Batch Size**: 32
    - **Epochs**: 3
    - **Optimizer**: Adam

### Kelebihan dan Kekurangan

- **Kelebihan**:
  - Model `bert-base-uncased` memiliki kemampuan untuk memahami konteks dari kata-kata dalam sebuah kalimat, sehingga dapat menghasilkan prediksi yang lebih akurat.
  - Model ini juga dapat menangani kata-kata yang jarang muncul atau tidak ada dalam dataset pelatihan awal.

- **Kekurangan**:
  - Memerlukan sumber daya komputasi yang besar untuk pelatihan dan inferensi.
  - Waktu pelatihan yang relatif lama dibandingkan dengan model machine learning yang lebih sederhana.

### Proses Improvement

Untuk meningkatkan performa model, dilakukan hyperparameter tuning dengan mencoba berbagai kombinasi parameter seperti learning rate, batch size, dan jumlah epochs.

### Pemilihan Model Terbaik

Setelah melakukan berbagai eksperimen, model `bert-base-uncased` yang di-fine-tune dengan parameter di atas dipilih sebagai model terbaik karena memberikan hasil evaluasi yang paling baik berdasarkan metrik akurasi, precision, recall, dan F1 score.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.
