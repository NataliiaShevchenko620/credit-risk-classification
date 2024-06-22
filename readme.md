# Module 20 Report Template

## Overview of the Analysis

### Purpose of the Analysis

The primary objective of this analysis was to evaluate and refine machine learning models to accurately predict credit default. This prediction is crucial for financial institutions to manage risks associated with lending.

### Financial Information Analyzed

The dataset comprised various financial metrics reflective of clients' creditworthiness. Key indicators included income levels, credit history, debt-to-income ratios, and past loan repayment records. The ultimate goal was to predict whether an individual would default on a loan repayment.

### Variables Analyzed

The target variable was `loan_status`, which indicates whether a loan was repaid or defaulted. The value_counts method revealed an imbalance between the two classes, with a significantly higher number of loans being repaid than defaulted, which presented a challenge for the predictive modeling.

### Stages of Machine Learning Process

The machine learning process encompassed several key stages:

#### Data Preprocessing: 

##### Data Loading:

The dataset was read from a CSV file into a Pandas DataFrame using pd.read_csv(). This initial step is crucial for loading the raw data into an appropriate format for analysis and manipulation.

##### Feature and Label Separation:
The data was split into features and labels for the purpose of machine learning. The 'loan_status' column, indicating whether a loan was paid off or defaulted, was separated into the target variable y. The remaining columns, representing various attributes related to the borrowers and their loans, were designated as features X.

##### Data Splitting:
The dataset was divided into training and testing subsets using the train_test_split method from sklearn.model_selection. This method allocated 80% of the data to the training set and 20% to the testing set, which helps in validating the model’s performance on unseen data. A random_state was set to ensure reproducibility of the results.

This structured approach to data preprocessing ensures that the data is correctly formatted and divided for training and testing, which is essential for building reliable predictive models.

#### Model training and evaluation

I train Logistic Regression on train dataset. Logistic regression was chosen for the analysis due to its simplicity and interpretability, which makes this method ideal for binary classification tasks such as predicting loan defaults. This algorithm easily interprets how various factors influence the probability of an event and scales well even when working with large data sets, making it particularly suitable for credit risk tasks.

Model Training:
The model was trained using the training data set. During this phase, the logistic regression algorithm learns the relationship between the features and the target variable by adjusting its weights to minimize errors in predictions.

Prediction Making:
Once the model is trained, it is used to make predictions on the test data. This involves feeding the features of the test dataset into the model to obtain predicted values for the target variable.

Performance Evaluation:
The predictions are compared with the actual outcomes in the test set. Key performance metrics such as accuracy, precision, recall, and F1-score are calculated to evaluate how well the model is performing. A confusion matrix is also generated to visually assess the model’s performance in classifying the outcomes correctly.

## Results

```Confusion Matrix:
 [[14926    75]
 [   46   461]]
``` 

The main diagonal shows the correct answers. The confusion matrix showed that the model gave the correct result in cases where people returned the credit. There were `14926` such cases. The model made incorrect predictions `75` times in cases where it predicted that people would not return the credit, but in fact, they did. And `45` times the opposite happened. 

![Classification report](credit-risk-classification/Report.png)

The model demonstrates high accuracy with a significant number of correct predictions (`14926` `True Negatives` and `461` `True Positives`), and a low rate of `False Positives` (`75`), minimizing the risk of denying credit to potentially reliable clients. The small number of False Negatives (`46`) suggests effective risk management in credit issuance. Overall, the model maintains a good balance between precision and recall, ensuring reliability in predicting credit returns, which enhances both economic efficiency and the safety of lending practices. 

The classification report shows excellent model performance with an `overall accuracy` of `0.99`. For class 0, the model achieved perfect `precision`, `recall`, and `F1-score` of `1.00` across `15001` instances, indicating flawless prediction for this class. Class 1, with 507 instances, showed good precision (0.86) and better `recall` (`0.91`), leading to an `F1-score` of `0.88`. The macro average scores highlight a robust model with `0.93` `precision`, `0.95` `recall`, and `0.94` `F1-score`, while the weighted average underscores consistent performance across different classes, each reflecting a `0.99` score in `precision`, `recall`, and `F1-score`.

Due to the class imbalance (between people returning and not returning credit), it makes sense to analyze additional metrics such as precision and recall. 

`Precision` (accuracy) for class 0 is perfect (`1.00`), meaning that the model did not produce any false positives for this class. For class 1, the precision is `0.86`, indicating that when the model predicts that a credit will not be returned, it is correct about 86% of the time. This is a good indicator, but it can be improved, as the remaining 14% of false positives could negatively affect clients who would have been able to return the credit.

`Recall` (completeness) for class 0 is also perfect (`1.00`), which means that all real cases of credit return have been correctly identified by the model. For class 1, the `recall` is `0.91`, which is higher than the precision, and indicates that the model successfully identifies most cases of credit non-return. However, some cases are still missed, which could lead to financial losses.

Regarding the drawbacks of the model, it does not reflect the actual financial benefit from the prediction results. Additional analysis:

## Summary

The analysis conducted using logistic regression has proven to be highly effective in predicting loan defaults, demonstrating robust accuracy and reliability. The model performed well in distinguishing between defaulters and non-defaulters, which is essential for financial institutions looking to minimize risk and optimize lending practices.

Throughout the data preprocessing stage, we prepared and refined the dataset for analysis, ensuring clean and relevant input for the model. The feature selection process was meticulous, aiming to incorporate only the most impactful variables that influence loan default probability.

During model training and evaluation, the logistic regression model was systematically trained and tested, yielding high precision and recall for non-default predictions and decent metrics for default predictions. This indicates that the model is exceptionally adept at identifying reliable borrowers, thereby reducing potential financial losses from defaulted loans.

Our model allows us to save money on loans that were issued but not repaid. However, at the same time, we can lose money by not issuing loans to good clients. Based on the result of this function, we see that our model allows us to save 548.53 per client. Note that this is an upper estimation based on the assumption that the client does not return a whole loan and did not pay any interest. For a more accurate analysis, we will need additional information about loan issuance terms and other details.

However, there are opportunities for improvement, particularly in enhancing the model's ability to detect actual defaulters more accurately. Future work could explore more sophisticated algorithms, integrate additional data points, or apply more complex feature engineering techniques to further refine the model’s predictive capabilities.

Overall, this project underscores the utility of logistic regression in financial risk assessment, providing a valuable tool for credit risk management that can lead to more informed, data-driven decision-making in lending processes. 


## Requirements
Python Libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Environment:
Jupyter Notebook or any other Python development environment.

## Installation
1. Clone the repository to your local machine:

   ```
   [git clone https://github.com/NataliiaShevchenko620/credit-risk-classification.git](https://github.com/NataliiaShevchenko620/credit-risk-classification.git)
   ```

2. Install the required Python libraries
3. Run the notebook

## License

This project is licensed under the MIT License.