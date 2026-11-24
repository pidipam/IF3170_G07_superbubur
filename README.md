# IF3170_G07_superbubur

Implementasi algoritma **K-Nearest Neighbors (KNN)**, **Decision Tree Learning (C4.5)**, **Logistic Regression**, **Support Vector Machine (SVM)**, dan **Naive Bayes** dari awal (_from scratch_) sebagai pemenuhan Tugas Besar 2 IF3170 Inteligensi Artifisial 2026/2027.

Repository ini berisi notebook eksperimen, implementasi modul algoritma pada folder `src/`, serta artefak hasil eksekusi pada folder `output/`.

## Anggota Kelompok

| NIM      | Nama                         |
| -------- | ---------------------------- |
| 13524073 | Keisha Rizka Syofyani        |
| 13524077 | Yavie Azka Putra Araly       |
| 13524081 | Alya Nur Rahmah              |
| 13524091 | Vara Azzara Ramli Pulukadang |
| 13524109 | Helena Kristela Sarhawa      |

## Deskripsi Singkat

Tugas besar ini mengimplementasikan pipeline pembelajaran mesin untuk permasalahan klasifikasi pada dataset yang diberikan. Proses yang dilakukan meliputi _data understanding_, _data cleaning_, _preprocessing_, _feature engineering/selection_, pemodelan menggunakan algoritma dari awal, evaluasi, perbandingan dengan pustaka `scikit-learn`, serta pembuatan file submission.

Algoritma yang diimplementasikan:

- **KNN**: klasifikasi berbasis jarak dengan dukungan bobot _uniform/distance_, pemilihan _k_, dan tipe fitur numerik/kategorikal.
- **Decision Tree (C4.5)**: pohon keputusan dengan _information gain ratio_, penanganan fitur numerik/kategorikal, _pruning_/kontrol kompleksitas, dan visualisasi pohon.
- **Logistic Regression**: klasifikasi biner dengan _gradient descent_, regularisasi L1/L2/Elastic Net, _class weight_, dan pencatatan riwayat loss.
- **SVM**: _Support Vector Machine_ dengan kernel linear/polynomial/Gaussian/RBF, optimasi berbasis QP/SMO, serta utilitas visualisasi _decision boundary_.
- **Naive Bayes**: wrapper `NaiveBayes` dengan mode `discrete` (`DiscreteNaiveBayes`) dan `gaussian` (`GaussianNaiveBayes`).

## Arsitektur Program

```text
IF3170_G07_superbubur/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── docs/                          # Laporan PDF (deliverable)
├── output/                        # Hasil run algoritma, model, grafik, submission
├── src/
│   ├── __init__.py                # Ekspor modul utama
│   ├── preprocessing.py           # Cleaning, split, encoding, scaling, outlier, pipeline
│   ├── feature_selection.py       # ANOVA, point-biserial, MI, chi-square, selector
│   ├── pca.py                     # Implementasi PCA from scratch
│   ├── knn.py                     # Implementasi KNN from scratch
│   ├── decision_tree.py           # Implementasi Decision Tree C4.5 from scratch
│   ├── logistic_regression.py     # Implementasi Logistic Regression from scratch
│   ├── svm.py                     # Implementasi SVM from scratch
│   ├── naive_bayes.py             # Discrete/Gaussian Naive Bayes + wrapper NaiveBayes
│   └── tree_plot.py               # Visualisasi pohon keputusan
├── notebook.ipynb                 # Notebook utama eksperimen dan analisis
├── Makefile                       # Shortcut setup/run/clean
├── requirements.txt               # Dependensi Python
└── README.md
```

### Deskripsi Modul `src/`

| Modul                    | Tanggung Jawab Utama                                                                                                                     | Kelas/Fungsi Penting                                                                                                                                                     |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `preprocessing.py`       | Pembersihan data, _stratified split_, imputasi, encoding, scaling, clipping outlier, transformasi log, resampling, dan pipeline dataset. | `remove_duplicates`, `stratified_split`, `Preprocessor`, `random_oversample`, `random_undersample`, `smote`, `prepare_datasets`                                          |
| `feature_selection.py`   | Seleksi fitur berbasis skor statistik dan korelasi.                                                                                      | `compute_correlation_matrix`, `find_collinear_features`, `anova_f_score`, `point_biserial_corr`, `mutual_information_score`, `chi2_contingency_score`, `FeatureSelector` |
| `pca.py`                 | Reduksi dimensi menggunakan Principal Component Analysis.                                                                                | `PCA.fit`, `transform`, `fit_transform`, `explained_variance_ratio_`                                                                                                     |
| `knn.py`                 | Klasifikasi KNN dengan perhitungan jarak dan voting.                                                                                     | `KNN.fit`, `kneighbors`, `predict`, `predict_proba`, `predict_k_range`, `save/load`                                                                                      |
| `decision_tree.py`       | Decision Tree C4.5 dengan _gain ratio_, split numerik/kategorikal, dan prediksi probabilitas.                                            | `Node`, `DecisionTree.entropy`, `split_information`, `gain_ratio`, `best_split`, `fit`, `predict`, `save/load`                                                           |
| `logistic_regression.py` | Logistic Regression biner dengan regularisasi dan optimasi gradient descent.                                                             | `LogisticRegression.fit`, `decision_function`, `predict_probability`, `predict`, `score`, `save/load`                                                                    |
| `svm.py`                 | SVM biner dengan beberapa kernel dan utilitas visualisasi.                                                                               | `SVM.fit`, `predict`, `decision_function`, `plot_decision_boundary`, `plot_test_points`, `save/load`                                                                     |
| `naive_bayes.py`         | Naive Bayes diskrit dan Gaussian, serta wrapper antarmuka umum.                                                                          | `DiscreteNaiveBayes`, `GaussianNaiveBayes`, `NaiveBayes`, `predict_proba`, `save/load`                                                                                   |
| `tree_plot.py`           | Visualisasi struktur pohon keputusan hasil model DTL.                                                                                    | `plot_tree`                                                                                                                                                              |

## Cara Setup

### Prasyarat

- Python 3.13 (atau versi kompatibel dengan dependensi pada `requirements.txt`)
- `pip`
- Opsional: `make`, `jupyterlab`

### Instalasi

1. Clone repository:

   ```bash
   git clone <url-repository>
   cd IF3170_G07_superbubur
   ```

2. (Opsional) buat dan aktifkan virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependensi:

   ```bash
   pip install -r requirements.txt
   ```

   Atau menggunakan Makefile:

   ```bash
   make setup
   ```

## Cara Run Program

### Menjalankan Notebook

Notebook utama berada pada `notebook.ipynb`. Jalankan melalui Jupyter:

```bash
jupyter lab
```

atau

```bash
jupyter notebook
```

Lalu buka `notebook.ipynb` dan jalankan sel secara berurutan.

### Menjalankan via Makefile

Makefile menyediakan shortcut untuk mengeksekusi notebook:

```bash
make run
```

Perintah ini akan menjalankan:

```bash
jupyter nbconvert --to notebook --execute notebook.ipynb --output output/notebook_executed.ipynb
```

### Membersihkan Artefak

```bash
make clean
```

## Output

Hasil eksekusi disimpan di folder `output/`, antara lain:

- Model tersimpan (`.pkl`, `.json`)
- Grafik EDA, preprocessing, evaluasi, dan visualisasi model (`.png`, `.gif`)
- File submission (`submission.csv`, `submission_dtl.csv`)

## Pembagian Tugas

| NIM      | Nama                         | Tugas                                                                                                                                             |
| -------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 13524073 | Keisha Rizka Syofyani        | Implementasi Decision Tree C4.5, visualisasi pohon, evaluasi DTL, pengujian reproducibility/pipeline, integrasi DTL pada notebook.                |
| 13524077 | Yavie Azka Putra Araly       | Implementasi SVM (linear/polynomial/Gaussian), optimasi SMO/QP, visualisasi decision boundary, evaluasi dan perbandingan SVM dengan scikit-learn. |
| 13524081 | Alya Nur Rahmah              | EDA, analisis fitur numerik/kategorikal, preprocessing awal, dan dukungan analisis data pada notebook.                                            |
| 13524091 | Vara Azzara Ramli Pulukadang | Implementasi Naive Bayes, preprocessing pipeline, PCA, feature selection, serta integrasi komponen terkait pada notebook.                         |
| 13524109 | Helena Kristela Sarhawa      | Implementasi Logistic Regression, evaluasi model, perbaikan sel notebook terkait logistic regression, dan analisis hasil.                         |
