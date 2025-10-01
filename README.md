Multiple Disease Prediction Using Machine Learning
Project Overview:
The Multiple Disease Prediction System utilizes machine learning algorithms to predict the risk of five different diseases based on patient input data. It features a user-friendly Streamlit interface for early detection and efficient diagnostics.
Implementation step:
1. Dataset Selection 5 multiple disease 
o	Parkinson’s → parkinsons.csv
o	Kidney Disease → kidney_disease.csv
o	Liver Disease → indian_liver_patient.csv
o	Diabetes → diabetes.csv
o	Heart Disease → heart.csv
2. Dataset Pre-processing 
o	Missing Value 
o	Label encoding 
o	Normalization 
3. Feature Selection  
o	Apply Principal Component Analysis (PCA) to reduce dimensionality and enhance model performance.
4. Dataset Splitting 
o	80% Training 
o	20% Testing 
5. Model Training  
o	Xgboost  Algorithm
o	SVM Algorithm
o	Random forest  Algorithm
o	Decision tree Algorithm
o	KNN Algorithm
6. Prediction: 
o	Save trained models using Pickle.
o	Load each model and make predictions based on user input.
o	Predict individual risk for: Parkinson’s, Kidney, Liver, Diabetes, and Heart Diseases.
7. Performance Estimation and Analysis
The system’s performance will be assessed using the following metrics:
o	Accuracy
o	Precision
o	Recall
o	F1-Score
o	ROC-AUC
o	Confusion Matrix
o	Model Confidence / Probability Scores
o	Cross-Validation Scores
o	Training Time
8. Webpage 
o	Home Page 
o	Login & Register Page 
o	Prediction Page 
--------------------------------------------------------------------------------------------------------
Notes:
This project is a simulation-based software project and does not involve hardware implementation. It is a non-real-time Python simulation designed for modeling purposes.
Graphical user interface (GUI) will be developed.
The system will be implemented using:
Language: Python
Platform: Anaconda Navigator Python Frontend: Streamlit
