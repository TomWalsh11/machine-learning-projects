# 📝 Quote Prediction for Approval

In this project, I build a machine learning model to predict whether a quote is approved or not based on a dataset containing information on various sales quotes. The goal is to classify quotes into two categories: "approved" or "not_accepted". The project involves data cleaning, exploratory data analysis (EDA), feature engineering, and model development.

### 🔍 Data Exploration
- **Data Analysis**: The dataset is initially explored to understand the data structure, identify missing values, and examine statistics like means, medians, and standard deviations.
- **Visualizations**: Created using Seaborn and Matplotlib to analyze:
  - Correlations between features like `Buyout`, `Consignment`, `Retail Price`, and `Store Credit`.
  - Distributions of `Retail Price` and `Quote Status`.
  - Relationships between features using scatterplots and boxplots.

### 🧹 Data Preprocessing
- **Cleaning**: Unwanted columns (e.g., `Product Type`, `Brand`, and `Estimated MSRP`) are dropped. Some columns like `Retail Price` are cleaned by replacing invalid values.
- **Encoding**: Categorical variables like `Contract Type` are one-hot encoded for model compatibility.
- **Feature Engineering**: Features like `Quote Status` are transformed into binary values (approved = 1, not accepted = 0), and other irrelevant features are removed.

### 🤖 Machine Learning Model
- **Model Development**: A deep learning model is built using Keras with TensorFlow backend:
  - The model consists of 3 hidden layers with ReLU activation functions and a dropout rate of 20% to prevent overfitting.
  - Binary cross-entropy loss function is used for binary classification.
- **Training**: The model is trained on 70% of the data with early stopping to avoid overfitting, monitored on the validation loss.

### 📝 Results
- **Performance Evaluation**: Model performance is evaluated using classification metrics such as accuracy, precision, recall, and confusion matrix. The model achieves good results with an optimized architecture.

### ⚙️ Techniques and Tools
- **Libraries**: Pandas, Numpy, Seaborn, Matplotlib, TensorFlow, Scikit-learn
- **Algorithms Used**: Neural Networks, Dropout Regularization, Early Stopping
- 
