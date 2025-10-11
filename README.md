# ⚡ High-Performance Credit Risk Analysis

## 🚀 Project Overview
This project applies **High-Performance Computing (HPC)** and **parallel machine learning** to accelerate credit-risk modeling and financial stress-testing.  
The goal is to identify factors influencing **credit card payment default** while leveraging **multiprocessing, distributed data pipelines, and GPU acceleration (CUDA)** to significantly reduce computation time.

Using a dataset of **30,000+ Taiwanese credit-card clients**, the project combines demographic and behavioral data to predict **customer default probability** with high accuracy.  
By parallelizing model training and optimizing preprocessing workflows, the project achieved up to **5× training speedup** and **40% faster stress-test simulations** on **Tesla P100 GPUs**.

---

## 🎯 Objectives
- Identify **key demographic and financial factors** affecting customer defaults.  
- Develop a **scalable credit-risk scoring model** for large datasets.  
- Utilize **HPC parallelism (CPU & GPU)** for faster model training and evaluation.  
- Improve **financial stress-testing simulations** using CUDA acceleration.  

---

## 🧠 Methodology

### 1. Data Preprocessing
- Dataset: **Taiwanese Credit Card Default Dataset** (30,000 observations, 23 variables).  
- Handled missing values, normalized numerical fields, and encoded categorical variables.  
- Parallel preprocessing using **Dask DataFrame** for multi-core acceleration.  
- Memory-efficient batching to handle large-scale computations on the HPC cluster.

### 2. Analytical Modeling
- **Logistic Regression** and **Linear Discriminant Analysis (LDA)** for risk classification.  
- **Clustering (K-Means)** to identify demographic segments and behavioral risk profiles.  
- Model evaluation via **accuracy, precision, recall, and AUC** metrics.

### 3. HPC & Parallelization Techniques
| Technique | Technology | Performance Impact |
|------------|-------------|--------------------|
| CPU Parallelism | `multiprocessing`, `joblib` | 3–5× faster model training |
| Distributed Data Handling | `Dask` on HPC cluster | 4× faster preprocessing |
| GPU Acceleration | **CUDA on Tesla P100 GPUs** | 40% faster simulation time |
| Parallel Visualization | Matplotlib + Dask threads | Efficient large-data plotting |

### 4. Tools and Environment
- **HPC Discovery Cluster (Northeastern University)**  
- **Python Libraries:** `pandas`, `dask`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn`  
- **Hardware:** Multi-core CPUs, Tesla P100 GPUs  
- **Software Stack:** CUDA, Python 3.10, HPC job scheduler (SLURM)

---

## 📊 Results and Insights

### Model Performance
| Model | Accuracy | Key Predictors | Notes |
|--------|-----------|----------------|-------|
| Logistic Regression | 80.82% | Payment history, credit balance, marital status | Strong baseline |
| LDA | 79.6% | Demographics, education level | Good class separation |
| K-Means Clustering | N/A | 3 major risk segments | High-risk group = young, single, low education |

### Demographic Patterns
- **Younger and single individuals** showed higher default rates.  
- **Males** had slightly higher default tendencies than females.  
- **Lower education** levels correlated with increased risk.  

### Parallelization Outcomes
- ✅ **5× speedup** in model training using multiprocessing.  
- ✅ **40% faster stress-test simulation** with CUDA GPU kernels.  
- ✅ Improved resource utilization across multi-node HPC cluster.

---

## 💡 Recommendations
- Incorporate **demographic weighting** into credit scoring for improved fairness.  
- Deploy **dynamic risk models** updated with real-time data streams.  
- Expand to **deep learning models (PyTorch, TensorFlow)** for large-scale risk prediction.  
- Integrate **GPU-accelerated XGBoost or LightGBM** for next-generation credit scoring.

---

## 🧩 Key Learnings
- HPC parallelization can dramatically reduce **training and inference latency** in financial ML pipelines.  
- CUDA-enabled GPU simulation boosts the scalability of stress-testing workflows.  
- Dask and multiprocessing provide **cost-effective CPU acceleration** without compromising accuracy.  
- The project demonstrates how **AI and HPC** can jointly enhance financial risk management.

---

## 🏁 Conclusion
The **High-Performance Credit Risk Analysis** project showcases how combining **data science** with **HPC and GPU computing** enables real-time, scalable financial analytics.  
By optimizing both **data preprocessing** and **model computation**, the project achieved **5× training acceleration** and improved **simulation throughput** — demonstrating the practical impact of **HPC in financial AI systems**.

---

## 📚 References
- UCI Machine Learning Repository – *Taiwanese Credit Card Default Dataset*  
- Northeastern University Discovery HPC Documentation  
- NVIDIA CUDA Toolkit Developer Guide  
