# Import packages

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# Define functions 

def explore(x):
    """
    This function takes a dataframe as an input and
    return shape and info about the columns in the dataframe.

    Args:
    x (dataframe): The input dataframe to be explored.
    """
    print(x.info(), '\n\nShape:', x.shape)


def nulls_percentage(x):
    """
    This function returns the nulls total count for each column
    and it's percentage

    Args:
        x (dataframe): The input dataframe to calculate nulls count for each column.

    Returns:
        dataframe: nulls count and percentage for each column in the input dataframe.

    Example:
    * nulls_percentage(df)

    .
    """
    y = x.isnull().sum().sort_values(ascending=False)
    y = y.to_frame(name='Missing_Values')
    y['Percentage %'] = y['Missing_Values']/x.shape[0]*100
    return y


def nulls_drop(data, x):
    """
    This function drops the columns with null values greater than x percentage.

    Args:
        data (dataframe): The input dataframe.
        x (float): The threshold value (the null percentage value in decimal of which you want to drop null percentages above).

    Returns:
        data (dataframe): Cleaned dataframe (after dropping columns having a null percentage above the desired threshold value).

    Example:
    * nulls_drop(df , 0.7)   ---> drop columns with nulls percentage above 70%.

    .
    """

    # calculate the threshold number of non-null values needed (x% of the column values)
    threshold = int((1-x) * data.shape[0])
    # drop columns above certain threshold
    data.dropna(axis=1, thresh=threshold, inplace= True)
    # show the cleaned dataframe
    return data


def repeat_percentage(data, x=''):
    """
    This function determines the percentage of occurrence for each unique value within a specific column.

    Args:
        data (dataframe): The input dataframe.
        x (str): The specific column name you want to check repeated values in it. Defaults to ''.

    Returns:
        Pandas Series: Pandas series indicating the percentage of occurrence for each unique value within a specific column.

    Example:
    * repeat_percentage(df , 'NAME_CONTRACT_TYPE')

    .
    """
    return (data[x].value_counts()*100)/data.shape[0]


def correlation(data, x=[], y=''):
    """
    This function calculates the correlation between numerical variables (features) in a dataset.

    Args:
        data (dataframe): The input dataframe.
        x (list): The list of columns you want to measure the correlation between. Defaults to [].
        y (str): The target column in which you want to measure correlation within (must exist in the x list also). Defaults to ''.

    Returns:
        dataframe: A dataframe containing the percentage of correlation, indicating whether it is a negative or positive correlation. (from -1 to 1).

    Example:
    * correlation(df , [ 'EXT_SOURCE_3' , 'EXT_SOURCE_2' , 'EXT_SOURCE_1' , 'TARGET' ] , 'TARGET')

    .
    """
    return data.loc[:,x].corr().sort_values(by= y, axis=0, ascending=False)


def relation(data, x='', y=None):
    """
    This function evaluates the relation between a value or a class in a specific column with the different classes in the "TARGET" column for analysis purpose.


    Args:
        data (dataframe): The input dataframe.
        x (str): The desired column (feature). Defaults to ''.
        y (str, int or float): The desired value you want to evaluate with the target column whether it is a string, integer or a float. Defaults to None.

    Example:
    * relation(df, x='feature1', y=1)

    * relation(df_application_train, x='ORGANIZATION_TYPE', y='Postal')

    .
    """
    print('relation between ' , y , ' and TARGET\n' ,data[data[x] == y]["TARGET"].value_counts(normalize=True),'\n')


def corr_heatmap(data, threshold):
  """
  This function takes your numerical dataframe as an input, and plot a correlation heatmap figure for analysis purposes.


  Args:
      data (dataframe): The input dataframe (must contain numerical features only).
      threshold (float): enter the threshold value and tune it as needed.

  Example:
      corr_heatmap(data= df, threshold= 0.5)


  .
  """
  # create correlation matrix with abs values
  corr_matrix = data.corr().abs()
  # filter the matrix due to threshold value
  filtered_corr_df = corr_matrix[(corr_matrix >= threshold) & (corr_matrix != 1.000)]

  plt.figure(figsize=(18,10))
  sns.heatmap(filtered_corr_df, annot=True, cmap="Reds")
  plt.show()