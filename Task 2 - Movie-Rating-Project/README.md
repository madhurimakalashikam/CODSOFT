🎬 Movie Rating Prediction with Python

Project Overview

This project aims to predict the rating of a movie based on features such as genre, director, runtime, year, and votes using machine learning regression techniques.
By analyzing historical IMDb India movie data, this model helps estimate how well a movie might be rated by users or critics.

Objective

=>Analyze historical movie data.

=>Perform data preprocessing and feature engineering.

=>Build and train a regression model.

=>Evaluate model performance.

=>Predict ratings for new or unseen movies.

How to Run the Project

1️⃣ Install Dependencies

Run the following command in your terminal:

pip install -r requirements.txt

2️⃣ Data Preprocessing

Clean the IMDb India movie dataset:

python src/data_preprocessing.py

3️⃣ Feature Engineering

Generate new features like title length, genre encoding, etc.:

python src/feature_engineering.py

4️⃣ Model Training

Train and evaluate the model:

python src/train.py

5️⃣ Run Streamlit App (optional)

If you built a UI for predictions:

streamlit run src/streamlit_app.py

📊 Machine Learning Model

Algorithm Used: Random Forest Regressor

Preprocessing: Standard Scaling & One-Hot Encoding

Evaluation Metrics: RMSE, MAE, R² Score

🧾 Results

After training, the model achieved:

RMSE: ~0.5

MAE: ~0.37

R² Score: ~0.84

(These values may vary depending on random splits and preprocessing steps.)

 Insights:

Movie ratings are influenced by multiple factors such as genre popularity, director reputation, and number of votes.

Ensemble methods like Random Forest provide strong performance for regression problems with mixed features.

Requirements:

pandas

numpy

scikit-learn

joblib

streamlit

Author:

Name: Madhurima Kalashikam

Organization: CODSOFT Internship (Task 2)

Topic: Movie Rating Prediction with Python
