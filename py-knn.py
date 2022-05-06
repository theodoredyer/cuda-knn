import pandas as pd
import numpy as np
import time
import math

# Theodore Dyer
# Introduction to GPU Programming Spring 2022 (EN605.617.81)
# Chance Pascale
# 4/21/2022


#%% 

def calculate_euclidian(entry_one, entry_two, cols):
    """
    Generates the euclidian distance between two points, to be utilized
    within my kNN Implementation

    Parameters
    ----------
    entry_one : Pandas dataframe row, represents the point from which
        we are seeking to calculate the distance to neighbors within kNN.
    entry_two : Pandas dataframe row, represents the point to which we
        are seeking to calculate distance (as a potential neighbor)
    cols : list[str], contains the names of the pandas columns to look at
        when considering our distance calculation. 

    Returns
    -------
    double
        Euclidian distance between two data points

    """
    distance = 0

    for i in range(len(cols)):
        colname = cols[i]
        distance += ((float(entry_one[colname]) - float(entry_two[colname])) ** 2)
        
    return math.sqrt(distance)

#%%

def calculate_weights(neighbors, verbose):
    """
    Parameters
    ----------
    neighbors : list[(pandas row, int)]
        list containing each neighbor calculated for the input point, as well as
        the distance associated with that point.
    verbose : Boolean
        indicates whether or not to produce verbose output text to console,
        in this case printing each neighbor's weight'

    Returns
    -------
    neighbor_weights : list
        weights for 'k' neighbors, used to select appropriate class

    """
    
    k = len(neighbors)
    total_neighbor_dist = 0
    for i in range(k):
        total_neighbor_dist += neighbors[i][1]
    
    neighbor_weights = []
    for i in range(k):
        neighbor_weights.append(neighbors[i][1] / total_neighbor_dist)
        # if(verbose):
            # print('Neighbor ' + str(i+1) + ' weight: ' + str(neighbor_weights[i]))
    if(verbose):
        print(sum(neighbor_weights))
        print()
    return neighbor_weights

#%%

def knn_init(train, typeflag, target, cols, verbose):
    """
    Closure function fo inner knn() providing a number of variables for it's
    implementation specifics.    


    Parameters
    ----------
    train : Pandas Dataframe
        training set upon which we query points for distances to generate neighbors
    typeflag : String
        either 'classification' or 'regression', dictating the behavior of kNN
    target : String
        Column name of target feature.
    cols : list[String]
        List of names for columns to consider in kNN distance genration
    verbose : Boolean
        Flag which dictates console output behavior. 

    Returns
    -------
    function
        knn() utilized for generating prediction results. 

    """
    def knn(k, query, typeflag):
        """     
        Inner knn algorithm implementation. Given a point, determine it's neighbors
        from an underlying training dataset and return corresponding prediction from 
        the determined neighbors.
        
        Parameters
        ----------
        k : integer
            hyperparameter dictating how many neighbors we consider for each iteration
            of the algorithm's execution.
        query : Pandas Row
            The point for which we would like to generate a prediction.
        typeflag : String
            Type of desired prediction, either classification or regression

        Returns
        -------
        variable
            Prediction of our knn algorithm. 

        """
        
        distances_list = []
        if(verbose):
            start_calc = time.time()
        for i in range(len(train)):
        
            dist = calculate_euclidian(query, train.iloc[i], cols)
            distances_list.append((train.iloc[i], dist))

        if(verbose):
            end_calc = time.time()
            print('Calculating distances took: ' + str(end_calc - start_calc) + ' ms.')
            print()
            start_sort = time.time()
        return 0
        
        nearest_neighbors = []
        distances_list.sort(key=lambda index: index[1])
        for i in range(k):
            nearest_neighbors.append(distances_list[i][0])
            if(verbose):
                print('Neighbor ' + str(i+1) + ': (Distance = ' + str(distances_list[i][1]) + ')')
                print(distances_list[i][0])
                print()

        if(verbose):
            end_time = time.time()
            print('Sorting neighbors took: ' + str(end_time - start_sort) + ' ms.')

            
        neighbor_weights = calculate_weights(nearest_neighbors, verbose)
        
        if(typeflag == 'regression'):
            regression_result = 0
            for i in range(k):
                regression_result += nearest_neighbors[i][target]
            regression_result = (regression_result / k)

            if(verbose):
                print('Regression Result: ' + str(regression_result))
                print('For point: ')
                print(query)

            return regression_result
            
        else:
            classification_options = []
            for i in range(k):
                classification_options.append(nearest_neighbors[i][target])
                
            classification_result = max(set(classification_options), key=classification_options.count)
            if(verbose):
                print('Classification Result: ' + str(classification_result))
                print('For point: ')
                print(query)
            return classification_result
            
    return knn

#%%

def k_fold(dataframe, target, k, typeflag, verbose):
    """
    Performs k-fold cross validation, which is done by dividing up proper
    index segments and then evaluating each fold agains the rest, rotating
    which fold is used as the test set each iteration.

    Parameters
    ----------
    dataframe : dataframe on which to run our k-fold cross validation
    target : column title of the target variable
    k : number of folds
    typeflag : string specifying execution type

    Returns
    -------
    None.

    """
    
    fold_size = int(len(dataframe) / k)
    
    pred_results = []
    
    for i in range(k):
            
        test_indices = [i for i in range((i*fold_size), (i+1)*fold_size)]
        train_indices = [i for i in range(len(dataframe))]    
        
        train_indices = [x for x in train_indices if x not in test_indices]
        
        if(i == 0):
            test_ret = test_indices
            train_ret = train_indices
        
        if(verbose):
            
            print(
                'test indices for fold ' + str(i + 1) + ': [' + str(min(test_indices)) + ' - ' + str(max(test_indices)) + ']'
            )
            if(i==0):
                print(
                'train indices for fold ' + str(i + 1) + ': [' + str(min(train_indices)) + ' - ' + str(max(train_indices)) + ']')
            else:
                print(
                    'train indices for fold ' + str(i + 1) + ': [' + str(min(train_indices)) + ' - ' + str(min(test_indices)) + '] & ['
                    + str(max(test_indices)) + ' - ' + str(max(train_indices)) + ']'
                )
        
    # Train = 183-917
    # Test = 0-182
    train = dataframe.iloc[train_ret]
    test = dataframe.iloc[test_ret]
        
    return train, test

#%%

def evaluate_performance(predicted, actual, typeflag):
    """
    Evaluates the performance of a machine learning algorithm's predictions

    Parameters
    ----------
    predicted : column of predicted values
    actual : column of actual values
    typeflag : string flag used to signify the type of algorithm we ran

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    if(typeflag == 'classification'):
        true_res = 0
        total = len(predicted)
        
        for i in range(total):
            if(predicted[i] == actual[i]):
                true_res += 1
                
        print('Classification Accuracy: ' + str(true_res / total))
        return (true_res / total)
        
    elif(typeflag == 'regression'):
        n = len(predicted)
        builder = 0
        
        for i in range(n):
            builder += ((predicted[i] - actual[i]) **2)
            
        print('Mean Squared Error: ' + str(builder / n))
        return builder / n
        
    else:
        print('specify performance metric')
        
#%%

def z_score_standardize(column):
    """
    Standardizes the values of a training set

    Parameters
    ----------
    column : column of values to standardize

    Returns
    -------
    column : column of standardized input values

    """
    
    feature_mean = column.mean()
    feature_std_dev = column.std()
    
    for i in range(len(column)):
        column[i] = (column[i] - feature_mean) / feature_std_dev
        
    return column

#%%

def discretize_data(column, bins, typeflag):
    """
    Given a column of values and a specified number of bins, replace original
    values with a corresponding bin number. This function does result in a loss
    of data but may be useful later on for certain algorithms. 

    Parameters
    ----------
    column : input values to discretize
    bins : number of desired bins
    typeflag : desired type of discretization (equal width or equal frequency)

    Returns
    -------
    column : column of discretized values

    """
    
    if(typeflag == 'equalwidth'):
        colmin = column.min()
        colmax = column.max()
        colrange = colmax-colmin
        
        binwidth = colrange/bins
        
        for i in range(len(column)):
            bin_num = column[i] // binwidth
            column[i] = bin_num
        
        return column
        
    elif(typeflag == 'equalfrequency'):
        
        "(NYI) Not Yet Implemented"
    
    else:
        print("specify discretization type")

# In[8]:

def fill_missing_values(column):
    """
    Fills in a dataset's missing values with an appropriate value according to 
    the type of data that we are working with (categorical receives the most 
    common value of that category, and numeric receives the mean of it's column')

    Parameters
    ----------
    column : input values with missing values

    Returns
    -------
    column : column representation of missing-value-filled input

    """
    
    if column.dtype == 'int64':
        for i in range(len(column)):
            if column[i] == '?' or column[i] == 'NaN':
                column[i] = ((int)(np.mean(column)))
                             
    if column.dtype == 'object':
        for i in range(len(column)):
            if column[i] == '?' or column[i] == 'NaN':
                column[i] = column.value_counts().idxmax()
                             
    return column


#%%

def read_data(data_path, column_names):
    """
    
    Processes a data file into Pandas.
    
    Params:
    data_path = file location for a dataset
    column_names = desired column names for the dataset
    
    Return:
    Pandas dataframe of specified file and column names. 
    
    """
    dataframe = pd.read_csv(data_path)
    dataframe.columns = column_names
    return dataframe



#%%

def time_test(point_num, verbose):
    """
    Record the execution time to generate euclidian distances
    for a select number of points, specified by parameter.

    Parameters
    ----------
    point_num : int
        The number of points for which to calculate euclidian distances

    Returns
    -------
    None.

    """
    knn = knn_init(train, 'classification', heart_failure_target, heart_failure_knn_columns, verbose)

    oneh_start = time.time()
    
    for i in range(point_num):
        if(verbose):
            print("Query Index: " + str(i))
        if(i == 1):
            i += 1
        knn(5, test.iloc[i], 'classification')
        
    oneh_end = time.time()
    print(str(point_num) + ' point execution time: ' + str(oneh_end - oneh_start) + ' seconds.')
    

#%%

# Setting up our data structure

heart_failure_data_path = 'datasets/clean_data.csv'
heart_failure_column_names = [
    'age',
    'sex',
    'chest_pain_type',
    'resting_bp',
    'cholesterol',
    'fasting_bp',
    'resting_ecg',
    'max_hr',
    'exercise_angina',
    'oldpeak',
    'st_slope',
    'heart_disease'
]

heart_failure_knn_columns = [
    'age',
    'sex',
    'chest_pain_type',
    'resting_bp',
    'cholesterol',
    'fasting_bp',
    'resting_ecg',
    'max_hr',
    'exercise_angina',
    'oldpeak',
    'st_slope',
]

heart_failure_target = "heart_disease"

# Reading in data and setting up our data structure
heart_failure_df = read_data(
    heart_failure_data_path,
    heart_failure_column_names
)


print(heart_failure_df)

#%%

# Setting up train and test split
train, test = k_fold(heart_failure_df, 'heart_disease', 5, 'classification', True)

#%%



time_test(10, False)


#%%
kval = 12

full_start = time.time()
knn = knn_init(train, 'classification', heart_failure_target, heart_failure_knn_columns, False)
knn_preds = []
for i in range(len(test)):
    knn_preds.append(knn(kval, test.iloc[i], 'classification'))
    
full_end = time.time()
print('Full execution time: ' + str(full_end - full_start) + ' seconds. (' + str(kval) + ' neighbors)')
    

#%%

evaluate_performance(knn_preds, test[heart_failure_target], 'classification')


