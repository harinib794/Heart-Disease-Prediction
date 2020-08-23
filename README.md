# Heart-Disease-Prediction   

## Introduction
Coronary Heart Disease (CHD) is the most common type of heart disease in the United States accounting for more than four hundred thousand deaths per year. The Coronary arteries form the network of blood vessels on the surface of the heart that supply it blood and oxygen. CHD develops when the coronary arteries become too narrow, reducing the blood flow to the heart.      
Prediction of CHD is regarded as one of the most important subject of clinical data analysis.Many of the factors that determine one's risk of developing coronary heart disease are, to a large extent, under ones’ control. Hence early prognosis of this disease can aid in making decisions on lifestyle changes in high risk patients and in turn mitigate the risk. This project intends to pinpoint the most relevant risk factors of heart disease as well as predict the overall risk.     

## The Data     
The dataset is from a cardiovascular study on residents of the town of Framingham,Massachusetts. The classification goal is to predict whether the patient has 10-year risk of future coronary heart disease (CHD). The dataset provides the patients’ demographic,behavioural and medical information. It includes over 4,000 records and 15 attributes.
Dataset Source : https://www.kaggle.com/amanajmera1/framingham-heart-study-dataset        

## Attributes:     
Each attribute is a potential risk factor.    
• sex: male or female.   
• age: age of the patient.    
• currentSmoker: whether or not the patient is a current smoker.        
• cigsPerDay: the number of cigarettes that the person smoked on average in one day.    
• BPMeds: whether or not the patient was on blood pressure medication.    
• prevalentStroke: whether or not the patient had previously had a stroke.     
• prevalentHyp: whether or not the patient was hypertensive.     
• diabetes: whether or not the patient had diabetes.     
• totChol: total cholesterol level.     
• sysBP: systolic blood pressure.    
• diaBP: diastolic blood pressure.    
• BMI: Body Mass Index.    
• heartRate: heart rate.   
• glucose: glucose level.    
• 10 year risk of coronary heart disease CHD (binary: “1”, means “Yes”, “0” means “No”).    

## Approach   
The data is divided into test and train set. The data is divided into 80:20 ratio. The training dataset is highly unbalanced, the positive Ten-year CHD cases account for only 15% of total cases. Three additional datasets are created to treat the imbalance in the dataset:       
• Under sampled dataset: Reduce the observations in the majority class to make the dataset balanced.      
• Over sampled dataset: Replicate observations in minority class to balance the dataset.      
• Under sampled and Over sampled dataset: The minority class is oversampled with replacement and majority class is under sampled without replacement.     
The following classification models are used for classification:    
• SVM   
• Logistic Regression   
• Naïve Bayes   
• Random Forest    
All models are trained on 4 datasets which are training dataset as it is, under sampled dataset, oversampled dataset and dataset created by over and under sampling. Model with the highest accuracy and specificity from confusion matrix is selected as the best model.      

## Conclusion     
The best models among the four models are:     
• SVM model with test accuracy of 59%. Sensitivity of the model is 60% and specificity of the model is 70%.AUC is 64%.     
• Random Forest model with test accuracy of 74%. Sensitivity of the model is 77% and specificity of the model is 60%.AUC is 68%.    
• Our primary goal is to identify patients who are at risk of developing CHD). The class of interest in our classification models is 0(not at risk of developing CHD). Hence, we want a model with higher specificity.           
• The important predictors influencing the development of ten-year CHD are age, cigarettes per day, systolic BP, BMI, gender and prevalent hypertension.         
