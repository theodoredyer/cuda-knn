# cuda-knn
 Implementation and analysis of the machine learning algorithm k-nearest neighbors, designed to leverage the GPU framework CUDA. 

## Note to Code Reviewers
Items to leave feedback on:
- data_prep.ipynb : is this complete enough or does it need more explanation?
- py-knn.py : this is my logic starting point for the later written CUDA version, any feedback on code structure would be appreciated. 
- Comments in py-knn.py might look a little unfamiliar, the #%% comments are used to break the python code up into "cells" that can be executed separately from the entire script in my IDE (Spyder), most other IDEs also support this.  


### Overview and Goals
This project will include the following:
- Python notebook for Exploratory Data Analysis of the input dataset
- Basic CPU based Python implementation of the k-nn algorithm
- Basic C based implementation of the k-nn algorithm
- CUDA based implementation of the k-nn algorithm
- Dataset to test each of these implementations

I'm including the Python version of this algorithm as I already have a fair amount of experience working with Python, and thought it would be a worthwhile exercise to first build out the algorithm in a platform I'm comfortable with, before then going to translate it into something I am less comfortable with (C/CUDA). 
  
After each of these implementations are finished, I am going to run a series of tests with one or multiple datasets to analyze the performance differences in each of their executions. 

### Hypothesis
Given that CUDA or GPU frameworks in general aim to provide performance benefit when solving problems of a SIMD (Single Instruction Multiple Data) model, I am expecting to see that the execution time of training/utilizing the CUDA based implementation of this algorithm to have at least 25% speedup over both the basic Python, and C implementations. 


### Development Process
1. Locating a dataset from https://www.kaggle.com/ (For now I'm thinking of using the Heart Failure Prediction Dataset found here: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
2. Performing EDA (Exploratory Data Analysis) on the input dataset to learn more about the features and value distributions, as well as making any necessary edits to dataset missing values. 
3. Provided I already have a fair amount of experience working with both Python and Machine Learning in general, I'm going to first begin by building out a Python based implementation of k-nn to familiarize myself with the algorithm, and just the process of working with a machine learning data pipeline in general. 
4. One of the biggest hurdles I'm going to face when developing the CUDA based version of this algorithm (or at least that I'm expecting) is going to be building the data pipeline in CPP/C to support the CUDA execution, given that my experience working with data pipelines so far has been using fairly advanced libraries in Python. With this in mind there will likely be some backstepping required to understand how to process the data without such helpful libraries which I'm assuming won't exist for what I need.
5. After I've developed the basic machine learning data pipeline for C/CPP (and by this I mean developing the ability to ingest the data, reformat it into an appropriate data structure, remove any null values and clean the data, and create functionality to be able to easily retrieve information), I'm going to try to translate my Python implementation into baseline C/CPP. 
6. Following this, I will analyze which parts of the training process for k-nn will be able to be replaced and sped up via kernel executions, and develop these components. This may be slightly complicated for k-nn, as this model strays from the traditional ML paradigm as it is non-parametric (I'll explain more about this in the final report). 
7. Lastly once all of these implementations are finished, I will run a series of tests for execution time on each. I will run a series of tests each utilizing a different amount of the dataset I choose, tentatively thinking 10%, 50%, 100%, and recording the execution time for each. Additionally, because the choice of 'k' for this algorithm does have an impact on computational complexity, I will try a few choices of this as well, for example k=3 (minimum), k=5 (middle), k=11 (maximum). 
8. The last step of developing this project will be to produce a final report based on the data generated from step 5. 

### Personal Notes
Given I'm not entirely sure how to fit kernel executions into the puzzle of the non parametric model, preliminary thoughts are the following: 
- Data parallelism inside calculate_euclidian function, this is going to be a relatively straightforward calculation made thousands of times, each with up to k internal calculations, should be relatively easy to model with a cuda solution. 
- Inside calculate_weights function, same as with calculate_euclidian, going to be a huge amount of data parallelism if I can figure out a way to model it. 