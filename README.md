# Loan Prediction Based on Cusmtomers Behaviour
Develop a machine learning model to predict loan approval based on customer behavior data. The project involves data preprocessing, exploratory data analysis, feature selection, model development using classification algorithms, evaluation of model performance, hyperparameter optimization, and providing actionable insights.

## Problem Statement:
Understanding the behavior and preferences of good borrowers (those with a low probability of default) and bad borrowers (those with a high probability of default) involves analyzing their demographic, financial, and behavioral data to identify patterns and characteristics that distinguish them. Reducing credit risk due to high numbers of defaulters is crucial to maintaining financial stability and profitability. Additionally, there is a need to accelerate the loan evaluation and approval process to enhance operational efficiency and customer satisfaction.

## Solution Approch:
To reduce credit risk, implement stricter credit assessment criteria, use predictive analytics to identify high-risk borrowers, and offer tailored financial products that encourage timely repayments. Accelerate the loan evaluation and approval process by leveraging automated decision-making systems, integrating advanced data analytics, and streamlining internal workflows. These measures will help to ensure more accurate assessments, quicker approvals, and better overall management of the loan portfolio.

## Observation:
**Exploratory Data Analysis and Data Description:**

The dataset used for predicting loan default based on customer behavior includes various features that describe demographic, financial, and behavioral aspects of customers. Each feature has an appropriate data type, and for machine learning purposes, categorical features will be converted into numerical features. There are no missing values, so there is no need for dropping or imputing values. The dataset contains no duplicate entries and includes 252,000 rows of data.

**Feature Descriptions:**

- **Income:** The average income of customers is Rs5,000,000, with a range from Rs10,000 to Rs9,900,000. The median income is close to the average, indicating no significant outliers.
- **Age:** The average age of customers is around 49 years, with an age range between 21 and 79 years.
- **Experience:** Customers have a range of 0 to 20 years of experience, with an average of 10 years.
- **Current Job Years:** The range of current job experience is from 0 to 14 years, with an average of 6 years.
- **Current House Years:** Customers tend to stay in their current residence for 10 to 14 years.
- **Risk Flag:** This feature has two values: 0 for customers who have not defaulted and 1 for those who have defaulted.
- **Marital Status:** The majority of customers are single.
- **House Ownership:** Most customers rent their homes.
- **Car Ownership:** The majority of customers do not own a car.
- **Profession:** The most common profession among customers is physicist.
- **City:** Most customers are from the city of Vijayanagaram.
- **State:** Most customers are from the state of Uttar Pradesh.

**Feature Distributions and Insights:**

- **Income Distribution:** The income distribution is even, with the highest income reaching Rs10,000,000. The interquartile range (IQR) for customers who have defaulted is larger than for those who have not, indicating greater income variation among defaulters. No outliers are present.
- **Age Distribution:** The age distribution is even between 20 and 80 years, with no detected outliers. Customers who have defaulted tend to be younger, as indicated by smaller Q1, Q2, and Q3 values.
- **Experience Distribution:** The work experience ranges from 0 to 20 years without outliers. Customers who have defaulted tend to have less work experience.
- **Current Job Years Distribution:** Most customers have 4 to 5 years of current job experience, with no outliers. Customers who have not defaulted tend to have longer current job experiences.
- **Current House Years Distribution:** Customers typically stay in their homes for 10 to 14 years, with no outliers. No significant differences were observed between defaulters and non-defaulters.
- **Marital Status:** Married customers are more likely to be non-defaulters compared to single customers.
- **House Ownership:** Customers who own their homes are more likely to be non-defaulters compared to those who rent or do not own homes.
- **Car Ownership:** Customers who own cars are more likely to be non-defaulters compared to those who do not own cars.
- **Profession Distribution:** Customers come from various professions, with doctors and statisticians being the most common. For analysis, these professions will be grouped into broader categories.
- **State Distribution:** Customers are primarily from Uttar Pradesh and Maharashtra. There is a typo in Uttar_Pradesh[5], which should be corrected to Uttar_Pradesh. For modeling purposes, these states will be grouped into smaller regions.

**Statistical Analysis Results:**

Based on chi-squared tests and t-tests, several features have statistically significant relationships with the target feature, risk_flag. These features include experience, state, house_ownership, car_ownership, age, marital status (married/single), profession, current_job_yrs, current_house_yrs, and income_category.

## Findings:
We developed a machine learning model using a base Decision Tree (default parameters) and evaluated its performance, achieving an accuracy of 87%. After hyperparameter tuning, the model showed a repayment rate of 96.4%. The feature importance analysis revealed that age, profession, and experience are the top three factors influencing the target variable.

## Insights:
Analysis reveals that certain professions such as Police Officer, Chartered Accountant, Army Officer, Surveyor, and Software Developer have the highest default rates. Younger customers are more likely to default, while longer work experience correlates with reduced default risk. Statistical tests confirm significant relationships between various features and default risk, indicated by very small p-values. Implementing the predictive model has improved the repayment rate from 87% to 96%, surpassing the 5% target increase. The model also enhances credit assessment efficiency by automating approvals for most categories, reducing approval time from 7 days to less than 24 hours. Consequently, faster and more accurate loan approvals are projected to increase customer satisfaction by 10% within six months.

## Access the Application:
[Click here to access the loan prediction application
](https://loan-prediction-based-on-behaviour.streamlit.app/)

**Preview 1:**

<img src="https://github.com/mfathul21/loan-prediction/blob/main/assets/Deploy%20-%201.png?raw=true" alt="Loan Prediction Application - 1" width="800">

**Preview 2:**

<img src="https://github.com/mfathul21/loan-prediction/blob/main/assets/Deploy%20-%202.png?raw=true" alt="Loan Prediction Application - 1" width="800">
