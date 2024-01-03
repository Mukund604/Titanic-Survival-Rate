# Titanic Survival Prediction using Naive Bayes Classifier

This repository contains a Python script for predicting Titanic passenger survival using the Naive Bayes classifier.

## Overview

- [Introduction](#introduction)
- [File Description](#file-description)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Training and Prediction](#training-and-prediction)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The script utilizes the Naive Bayes classifier to predict passenger survival on the Titanic based on various features available in the dataset. It performs data preprocessing, model training, and prediction to determine survival outcomes.

## File Description

- `titanic_survival_prediction.py`: Python script for predicting Titanic passenger survival using the Naive Bayes classifier.

## Dataset

The dataset used is 'titanic.csv', containing information about passengers onboard the Titanic, including features such as age, sex, fare, and survival status.

## Preprocessing

1. Dropping irrelevant columns such as 'PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', and 'Embarked'.
2. Handling missing values by filling NaN values in the 'Age' column with the mean age.
3. Encoding categorical variables like 'Sex' using dummy variables and one-hot encoding.

## Training and Prediction

1. Splitting the dataset into training and testing sets using a 80:20 ratio.
2. Initializing a Gaussian Naive Bayes classifier and fitting the training data into the model.
3. Making predictions on the test set.

## Evaluation

The model's accuracy is evaluated using the `Accuracy` function, measuring the accuracy of predicted survival outcomes compared to the actual survival status in the test set.

## Usage

To use this script:

1. Clone the repository:

    ```
    git clone https://github.com/Mukund604/Titanic-Survival-Prediction.git
    ```

2. Ensure Python is installed along with necessary libraries like numpy, pandas, and scikit-learn.

3. Execute the `titanic_survival_prediction.py` script to perform Titanic survival prediction.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is not licensed. The code is open-source and can be used, modified, and distributed freely.
