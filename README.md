# ECG Signal Classification using Feature Engineering and Machine Learning

## Project Overview 📖

This project focuses on the binary classification of 1D ECG signals into **normal (0)** and **abnormal (1)** categories. The primary goal is to leverage feature engineering and machine learning models to build a robust classifier for ECG signals. The project uses the Kaggle ECG Dataset and evaluates two different models: **Random Forest** and **Support Vector Machine (SVM)**.

---

## Dataset 📊

The dataset used in this project is the **Kaggle ECG Dataset**. It contains a total of **4,998 samples** of 1D ECG signals, each represented by a time series of **140 voltage measurements**. The class distribution is as follows:

* **Normal (Label 0):** 2,079 samples (41.6%)
* **Abnormal (Label 1):** 2,919 samples (58.4%)

---

## Methodology 💡

The project follows a comprehensive workflow that includes the following key stages:

### 1. Data Exploration and Visualization
* The raw ECG data is initially visualized using a **heatmap** to identify any discernible patterns in the signals.
* **16 statistical visualizations**, such as histograms and scatter plots of mean, median, mode, variance, skewness, and kurtosis, are generated to gain a deeper understanding of the data's underlying characteristics.

### 2. Feature Engineering
A total of **24 features** are extracted from the raw ECG signals, covering statistical, time-domain, and frequency-domain properties. These features include:
* **Statistical Features:** Mean, Standard Deviation, Skewness, Kurtosis, Range
* **Time-Domain Features:** RMS, Zero-Crossing Rate, Peak Count, RR Mean, HRV (SDNN), RR Median, QRS Duration, QRS Amplitude, P-Wave Count & Amplitude, T-Wave Count & Amplitude
* **Frequency-Domain Features:** PSD Mean, Dominant Frequency, PSD Total, FFT Max, Band Energy Ratio
* **Wavelet Features:** Wavelet Energy and Wavelet Variance

### 3. Feature Selection and Transformation
* **Mutual Information (MI)** is employed as a filter method to rank the features based on their relevance.
* A correlation-based selection process is then used to choose the top **15 features** with low inter-feature correlation (<0.9).
* The selected features are then transformed using **Standardization (Z-score Normalization)** to have zero mean and unit variance.

### 4. Model Building
Two machine learning models are trained and evaluated for the classification task:
* **Random Forest**
* **Support Vector Machine (SVM)**

The performance of both models is assessed using **5-fold cross-validation**.

---

## Results 🏆

After a thorough evaluation, the **SVM model was selected as the best-performing model** for this classification task. The performance of the two models is summarized below:

| Metric | Random Forest | SVM |
| :--- | :--- | :--- |
| **Precision** | 0.9533 | 0.9669 |
| **Recall** | 0.9510 | 0.9801 |
| **F1-Score** | 0.9522 | **0.9735** |
| **AUC-ROC**| 0.9428 | 0.9665 |
| **False Negatives** | 143 | **58** |

The SVM model was chosen due to its higher **F1-Score (0.9735)**, indicating a better balance between precision and recall. Furthermore, the SVM model exhibited a higher recall and a significantly lower number of false negatives (58), which is critical in medical diagnostic applications where minimizing missed abnormalities is a top priority.

---

## Usage 🚀

To run this project, you will need to use the `ECG.ipynb` Jupyter Notebook. The notebook is well-commented and divided into cells for each step of the process.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ADVAYA1/ecg-classification.git](https://github.com/ADVAYA1/ecg-classification.git)
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook ECG.ipynb
    ```

---

## Dependencies 📦

The following Python libraries are required to run the project:

* pandas
* requests
* io
* seaborn
* matplotlib
* numpy
* scikit-learn
