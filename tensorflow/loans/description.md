# 💳 Lending Club Loan Prediction

In this project, I use machine learning techniques to predict whether a borrower will repay their loan in full based on a publicly available dataset from **Lending Club** (2007-2010). The goal is to classify loans as either "Fully Paid" or "Charged Off" by applying exploratory data analysis (EDA), data preprocessing, and machine learning models.

### 🔍 Data Exploration
- **Data Analysis**: Exploratory analysis includes distribution of loan status, loan amount, correlations, and visualizations to understand the data better.
- **Visualizations**: Created using Seaborn and Matplotlib to explore relationships between variables, including:
  - Loan amount vs installment amounts
  - Loan status counts by grade and sub-grade
  - Correlation heatmaps

### 🧹 Data Preprocessing
- **Handling Missing Values**: Null values are treated through methods like imputation and removal of irrelevant columns.
- **Categorical Encoding**: Categorical variables like `term`, `purpose`, `home_ownership`, and `zip_code` are transformed using one-hot encoding.
- **Feature Engineering**: Additional features are derived from existing columns like the `earliest_cr_line`, which is converted into a year.

### 🤖 Machine Learning Model
- **Model Selection**: A deep learning model built using Keras with TensorFlow backend:
  - Neural network with 3 hidden layers, ReLU activation, and dropout for regularization.
- **Model Training**: The model is trained on 10% of the dataset, using MinMax scaling for normalization and a binary classification loss function.
- **Performance Evaluation**: The model is evaluated using confusion matrices and classification reports to assess accuracy, precision, and recall.

### 📝 Results
- The model's performance is visualized with loss curves and classification metrics. It predicts whether new customers will repay their loan based on the trained features.

### ⚙️ Techniques and Tools
- **Libraries**: Pandas, NumPy, Seaborn, Matplotlib, Keras, Scikit-learn
- **Algorithms Used**: Neural Network (Deep Learning), Logistic Regression, Decision Trees, Random Forests
