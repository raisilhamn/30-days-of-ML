# Day 1

## Sumber

- [How Models Work](https://www.kaggle.com/dansbecker/how-models-work)
  > The first step if you're new to machine learning.

### keypoints

- step of capturing patterns from data is called **fitting** or **training** the model.
- The data used to **fit** the model is called the **training data**.
- After the model has been fit, you can apply it to new data to **predict**.
![example](http://i.imgur.com/R3ywQsR.png)
- The point at the bottom where we make a prediction is called a leaf.

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

To choose variables/columns, we'll need to see a list of all columns in the dataset. That is done with the columns property of the DataFrame (the bottom line of code below).

<iframe src="https://www.kaggle.com/embed/dansbecker/your-first-machine-learning-model?cellIds=2&kernelSessionId=74685890" height="300" style="margin: 0 auto; width: 100%; max-width: 950px;" frameborder="0" scrolling="auto" title="Your First Machine Learning Model"></iframe>

