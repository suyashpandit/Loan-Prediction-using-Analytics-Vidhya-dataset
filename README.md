**Feature Forge: Enhancing & Evaluating ML Models - Loan Prediction using Analytics Vidhya dataset**

*This project was originally completed as part of a Data Analytics & AI Bootcamp and later curated for portfolio presentation.*

**Introduction:**

CSV dataset was obtained from Analytics Vidhya website which was aimed at implementing feature engineering techniques and evaluating a machine learning model. The goal is to extract and select features, apply evaluation metrics, and refine the models to improve performance.

**Problem Statement:**

Dream Housing Finance company deals in all kinds of home loans. They have presence across all urban, semi urban and rural areas. Customer first applies for home loan and after that company validates the customer eligibility for loan.

Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have provided a dataset to identify the customers segments that are eligible for loan amount so that they can specifically target these customers. 

**Objectives:**

1. Explore the dataset, preprocess and clean it.
2. Create visualizations for univariate and bivariate analysis.
3. Apply feature engineering methods to choose relevant features.
4. Train and implement a logistic regression model to predict the loan approval status.
5. Use K-Fold cross validation.
6. Use appropriate model evaluation metrics to evaluate the model.
7. Document the process and results.

**Methodology:**

1. **Tools used:**
   
   a. Python programming language
   
   b. Pandas for data manipulation and exploration

   c. Matplotlib and seaborn to visualize data and plot visualizations

   d. Scikitlearn to build model, perform model evaluation using metrics and preprocess using scaler.
   
2. **Data collection, cleaning and preprocessing:**

   a. Dataset was collected from Analytics Vidhya website. Only the train dataset was downloaded as it is planned to split the train dataset into train and test sets.

   b. Panda and numpy was used to inspect, manipulate and clean the data.

   c. Matplotlib and seaborn was used to visualize the data to see relevant relationships between features and target variable.

3. **Feature engineering:**

   a. Categorical features were encoded using one hot encoding and changed to numeric value for model use.

   b. Feature selection was done based on EDA and filter method, by only selecting a portion of the available features. Feature transformation was done by creating relevant new features from raw features.

   
4.**Modeling and Evaluation:**

   a. Logistic Regression was used because the target variable is binary and also because of my beginner level. Sample Toy Dataset for Diabetes Prediction was used as the reference to apply similar algorithms.

   b. K-fold cross validation was done.

   c. Classification report of the final model was generated with precision,recall, f-1 score and accuracy.


**Results**

1. Loan prediction dataset from analyticsvidhya was collected, cleaned and used for model development after feature engineering.
2. K-fold cross validation results: Accuracy 86% , Precision 0.84 , Recall 1.00 , F-1 Score 0.91
3. Model evaluation metric: 79 percent accuracy.

**Challenges and Solutions**

1. The dataset was a mix of categorical and numerical values. It was solved by doing one hot encoding of the categorical values using this documentation: https://pbpython.com/categorical-encoding.html
2. Difficulty plotting categorical values in seaborn which was solved by using this documentation: https://seaborn.pydata.org/tutorial/categorical.html
3. Null values in the dataset within various features. Numerical values were filled using median as the data was not normally distributed and categorical values were filled with mode.
4. Convergence warnings and iteration limits being reached which was solved by using StandardScaler referring to this documentation: https://scikit-learn.org/stable/modules/preprocessing.html

**Conclusion:**

This project ws done as a part of learning to build and evaluate a logistic regression model to predict loan approval chances using the Analytics Vidhya Loan Prediction dataset.

The dataset was evaluated, visualized and feature engineering was done to extract relevant features and use them in the model to be built. 

K-fold cross validation was done with an accuracy of 86 percent.

The model was tested on unseen test set which was split in the beginning and obtained accuracy of 79 percent.

This was a good learning experience with a mixed and imbalanced dataset with different types of variables. Using standardscaler was another learning experience. The only downside was not balancing the dataset during logistic regression which would probably have improved the recall of rejections of the model. 
