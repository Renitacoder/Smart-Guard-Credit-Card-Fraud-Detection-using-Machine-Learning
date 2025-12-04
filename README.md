# Smart-Guard-Credit-Card-Fraud-Detection-using-Machine-Learning

A simple machine learning project that detects whether a credit card transaction is fraudulent or genuine.
It uses basic transaction features like amount, time, risk level, and past fraud history, and applies ML algorithms.
The Random Forest model gave the best accuracy in this project.

## Features

1. Loads and preprocesses transaction data
2. Trains ML models (Decision Tree & Random Forest)
3. Detects fraudulent transactions
4. Shows accuracy and classification report

## Tech Stack

1. Python
2. Pandas
3. NumPy
4. Scikit-learn
5. Matplotlib (optional)

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/Renitacoder/Smart-Guard-Credit-Card-Fraud-Detection-using-Machine-Learning.git
```
2. Navigate to the project folder:

```bash
cd Smart-Guard-Credit-Card-Fraud-Detection-using-Machine-Learning
```
3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Run the code:

```bash
python fraud_detection.py
```

## How It Works

1. Data is cleaned and preprocessed

2. Features and labels are separated

3. Models are trained

4. Accuracy is calculated

5. Random Forest is used for final prediction

## Project Structure

SmartGuard/
│── fraud_detection.py
│── README.md
│── requirements.txt


## Model Used

1. Random Forest Classifier

2. Decision Tree (for comparison)

3. Logistic Regression (for comparison)

## Output

1. Accuracy score

2. Precision, recall, and F1-score

3. Fraud predicted as 1

4. Genuine transaction predicted as 0

5. Confusion matrix and ROC curve visualizations
