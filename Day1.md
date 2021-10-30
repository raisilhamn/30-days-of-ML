# Day 1

- [Day 1](#day-1)
  - [How Models Work](#how-models-work)
    - [keypoints](#keypoints)
  - [Basic Data Exploration](#basic-data-exploration)
  - [Your First Machine Learning Model](#your-first-machine-learning-model)
    - [Selecting The Prediction Target](#selecting-the-prediction-target)
    - [Choosing "Features"](#choosing-features)
    - [Building Your Model](#building-your-model)
  - [Model Validation](#model-validation)
    - [What is Model Validation ?](#what-is-model-validation-)
    - [Latihan](#latihan)
      - [The Importance of Data Splitting](#the-importance-of-data-splitting)

## How Models Work

- [How Models Work](https://www.kaggle.com/dansbecker/how-models-work)
  > The first step if you're new to machine learning.

### keypoints

- step of capturing patterns from data is called **fitting** or **training** the model.
- The data used to **fit** the model is called the **training data**.
- After the model has been fit, you can apply it to new data to **predict**.
![example](http://i.imgur.com/R3ywQsR.png)
- The point at the bottom where we make a prediction is called a **leaf**.

## Basic Data Exploration

using pandas

```python
import pandas as pd
```

you can assign file (data source) to variable

```python
# save filepath to variable for easier access
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
melbourne_data.describe()
```

## Your First Machine Learning Model

- [Your First Machine Learning Model](https://www.kaggle.com/dansbecker/your-first-machine-learning-model/tutorial)
  > Building your first model. Hurray!

We'll start by picking a few variables using our intuition. Later courses will show you statistical techniques to automatically prioritize variables.

To choose variables/columns, we'll need to see a list of all columns in the dataset. That is done with the columns property of the DataFrame (the bottom line of code below).

```python
import pandas as pd

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
melbourne_data.columns
```

```markdown
# output
Index(['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG',
       'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude',
       'Longtitude', 'Regionname', 'Propertycount'],
      dtype='object')
```

sementara ini kita drop dulu data yang hilang
> dropna drops missing values (think of na as "not available")

```python
melbourne_data = melbourne_data.dropna(axis=0)
# menghapus baris yang memiliki NA
```

> axis=0 (or axis='rows' is horizontal axis. axis=1 (or axis='columns') is vertical axis. To take it further, if you use pandas method drop, to remove columns or rows, if you specify axis=1 you will be removing columns. If you specify axis=0 you will be removing rows from dataset.

### Selecting The Prediction Target

We'll use the dot notation to select the column we want to predict, which is called the prediction target. By convention, the prediction target is called y. So the code we need to save the house prices in the Melbourne data is

```python
y = melbourne_data.Price
```

by convention this data called **y**

### Choosing "Features"  

The columns that are inputted into our model (and later used to make predictions) are called "**features**." In our case, those would be the columns used to determine the home price. Sometimes, you will use all columns except the target as features. Other times you'll be better off with fewer features.

For now, we'll build a model with only a few features. Later on you'll see how to iterate and compare models built with different features.

We select multiple features by providing a list of column names inside brackets. Each item in that list should be a string (with quotes).

```python
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
```

By convention, this data is called **X**.

### Building Your Model

You will use the **scikit-learn** library to create your models. When coding, this library is written as **sklearn**, as you will see in the sample code. Scikit-learn is easily the most popular library for modeling the types of data typically stored in DataFrames.
The steps to building and using a model are:

- **Define**: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
- **Fit**: Capture patterns from provided data. This is the heart of modeling.
- **Predict**: Just what it sounds like
- **Evaluate**: Determine how accurate the model's predictions are.
  
Here is an example of defining a decision tree model with scikit-learn and fitting it with the features and target variable.

```python
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)
```

```markdown
# output
DecisionTreeRegressor(random_state=1)
```

Many machine learning models allow some randomness in model training. Specifying a number for random_state ensures you get the same results in each run. This is considered a good practice. You use any number, and model quality won't depend meaningfully on exactly what value you choose.

We now have a fitted model that we can use to make predictions.

In practice, you'll want to make predictions for new houses coming on the market rather than the houses we already have prices for. But we'll make predictions for the first few rows of the training data to see how the predict function works.

```python
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
```

```markdown
# output
Making predictions for the following 5 houses:
   Rooms  Bathroom  Landsize  Lattitude  Longtitude
1      2       1.0     156.0   -37.8079    144.9934
2      3       2.0     134.0   -37.8093    144.9944
4      4       1.0     120.0   -37.8072    144.9941
6      3       2.0     245.0   -37.8024    144.9993
7      2       1.0     256.0   -37.8060    144.9954
The predictions are
[1035000. 1465000. 1600000. 1876000. 1636000.]
```

## Model Validation

Measure the performance of your model, so you can test and compare alternatives.

### What is Model Validation ?

> penting

Many people make a huge mistake when measuring predictive accuracy. **They make predictions with their _training data_ and compare those predictions to the target values in the _training data_.** You'll see the problem with this approach and how to solve it in a moment, but let's think about how we'd do this first.

You'd first need to summarize the model quality into an understandable way. If you compare predicted and actual home values for 10,000 houses, you'll likely find mix of good and bad predictions. Looking through a list of 10,000 predicted and actual values would be pointless. We need to summarize this into a single metric.

There are many metrics for summarizing model quality, but we'll start with one called **Mean Absolute Error** (also called MAE). Let's break down this metric starting with the last word, error.

The prediction error for each house is:

```markdown
error = actual − predicted
```

So, if a house cost 150,000 and you predicted it would cost 100,000 the error is 50,000.

With the MAE metric, we take the absolute value of each error. This converts each error to a positive number. We then take the average of those absolute errors. This is our measure of model quality. In plain English, it can be said as

> On average, our predictions are off by about X.

code :

```python
import pandas as pd

# Load data
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 

# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(X, y)
```

```markdown
# output
DecisionTreeRegressor()
```

Once we have a model, here is how we calculate the mean absolute error:

```python
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
```

```markdown
# output
434.71594577146544
```

### Latihan

Menggunakan _train_test_split_ dari sklearn
> [Dokumentasi](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
> [Demo](https://realpython.com/train-test-split-python-data/)

train test split digunakan agar prediksi tidak bias.
> you can split your dataset into subsets that minimize the potential for bias in your evaluation and validation process.

#### The Importance of Data Splitting

Supervised machine learning is about creating models that precisely map the given inputs (independent variables, or predictors) to the given outputs (dependent variables, or responses).

What’s most important to understand is that you usually need unbiased evaluation to properly use these measures, assess the predictive performance of your model, and validate the model.

This means that you can’t evaluate the predictive performance of a model with the same data you used for training. You need evaluate the model with fresh data that hasn’t been seen by the model before. You can accomplish that by splitting your dataset before you use it.

```python
>>> x_train, x_test, y_train, y_test = train_test_split(x, y)
>>> x_train
array([[15, 16],
       [21, 22],
       [11, 12],
       [17, 18],
       [13, 14],
       [ 9, 10],
       [ 1,  2],
       [ 3,  4],
       [19, 20]])
>>> x_test
array([[ 5,  6],
       [ 7,  8],
       [23, 24]])
>>> y_train
array([1, 1, 0, 1, 0, 1, 0, 1, 0])
>>> y_test
array([1, 0, 0])
```

Given two sequences, like x and y here, train_test_split() performs the split and returns four sequences (in this case NumPy arrays) in this order:

- x_train: The training part of the first sequence (x)
- x_test: The test part of the first sequence (x)
- y_train: The training part of the second sequence (y)
- y_test: The test part of the second sequence (y)

You probably got different results from what you see here. This is because dataset splitting is random by default. The result differs each time you run the function. However, this often isn’t what you want.

The figure below shows what’s going on when you call `train_test_split()`:
.
<img src="https://files.realpython.com/media/fig-1.c489adc748c8.png" width=35% height=35%>

You can see that y has six zeros and six ones. However, the test set has three zeros out of four items. If you want to (approximately) keep the proportion of y values through the training and test sets, then pass stratify=y. This will enable stratified splitting:

```python
>>> x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.33,  
      random_state=4, stratify=y)

>>> x_train
array([[21, 22],
       [ 1,  2],
       [15, 16],
       [13, 14],
       [17, 18],
       [19, 20],
       [23, 24],
       [ 3,  4]])
>>> x_test
array([[11, 12],
       [ 7,  8],
       [ 5,  6],
       [ 9, 10]])
>>> y_train
array([1, 0, 1, 0, 1, 0, 0, 1])
>>> y_test
array([0, 0, 1, 1])
```

Now y_train and y_test have the same ratio of zeros and ones as the original y array.

Stratified splits are desirable in some cases, like when you’re classifying an imbalanced dataset, a dataset with a significant difference in the number of samples that belong to distinct classes.
