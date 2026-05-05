# Evaluasi Model Data Mining: Ensemble Learning & Confusion Matrix

Proyek ini bertujuan untuk mengimplementasikan dan membandingkan berbagai metode **Ensemble Learning** menggunakan Python. Berdasarkan materi akademik "Klasifikasi - Ensemble Methods", proyek ini mengevaluasi bagaimana penggabungan beberapa model (*base learners*) dapat meningkatkan akurasi prediksi dibandingkan dengan model tunggal[cite: 1].

## 🚀 Ringkasan Metode
Proyek ini mencakup tiga pendekatan utama dalam Ensemble Learning:

1. **Bagging (Bootstrap Aggregating)**
   - **Prinsip**: Melatih beberapa model secara paralel menggunakan sampel bootstrap (sampling dengan pengembalian)[cite: 1].
   - **Tujuan**: Mengurangi *variance* dan mengatasi *overfitting*[cite: 1].
   - **Contoh**: Random Forest[cite: 1].

2. **Boosting**
   - **Prinsip**: Melatih model secara sekuensial, di mana model berikutnya fokus memperbaiki kesalahan model sebelumnya[cite: 1].
   - **Tujuan**: Mengurangi *bias* dan mengatasi *underfitting*[cite: 1].
   - **Contoh**: AdaBoost dan Gradient Boosting[cite: 1].

3. **Stacking (Stacked Generalization)**
   - **Prinsip**: Menggunakan model *meta-learner* untuk menggabungkan prediksi dari berbagai model dasar (SVM, k-NN, Decision Tree)[cite: 1].
   - **Tujuan**: Mendapatkan akurasi maksimal dengan mengombinasikan keunggulan berbagai arsitektur model[cite: 1].

## 📊 Dataset
Dataset yang digunakan adalah **Breast Cancer Wisconsin (Diagnostic)** dari Scikit-Learn[cite: 1]. Dataset ini terdiri dari fitur teknis hasil pemindaian medis untuk mengklasifikasikan tumor sebagai ganas (*malignant*) atau jinak (*benign*).

## 🛠️ Implementasi Kode
Kode utama (`main.py`) melakukan langkah-langkah berikut:
- Pre-processing data dan *split* data (70% Train, 30% Test)[cite: 1].
- Pelatihan model tunggal (Decision Tree) sebagai baseline[cite: 1].
- Pelatihan model Ensemble (Random Forest, AdaBoost, Gradient Boosting, Stacking)[cite: 1].
- Evaluasi menggunakan **Confusion Matrix** untuk melihat detail performa klasifikasi.
- Analisis **Feature Importance** untuk mengidentifikasi variabel paling berpengaruh[cite: 1].

## 📈 Hasil Eksperimen (Contoh)
| Model | Accuracy | Keterangan |
| :--- | :--- | :--- |
| Single Decision Tree | 0.9415 | Baseline |
| Random Forest | 0.9708 | Mengurangi Variance |
| Gradient Boosting | 0.9591 | Mengurangi Bias |
| **Stacking** | **0.9825** | **Performa Terbaik** |

*(Catatan: Nilai akurasi dapat bervariasi tergantung pada random seed)*

## 📂 Struktur Repositori
- `main.py`: Script utama implementasi model Scikit-Learn.
- `requirements.txt`: Daftar pustaka yang diperlukan (pandas, sklearn, seaborn).
- `DM06.pdf`: Referensi materi kuliah mengenai Ensemble Methods.

## 🎓 Referensi
- Materi Kuliah Data Mining: Klasifikasi - Ensemble Methods oleh Dr. Arya Adhyaksa Waskita[cite: 1].
- Breiman, L. (2001). *Random Forests. Machine Learning*[cite: 1].
- Wolpert, D. H. (1992). *Stacked Generalization. Neural Networks*[cite: 1].
