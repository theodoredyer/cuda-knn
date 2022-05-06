#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>
#include <time.h>
#include <unistd.h>

#define NUM_COLS 12
#define NUM_ROWS 918
#define TRAIN_POINTS 735
#define K 5


using namespace std;

/*
calculate_euclidian(dataframe, row_one, row_two)
- Function to compute the euclidian distance between two entries in our dataframe

- Params:
    dataframe = 2d array consisting of doubles, holding all input data
    row_one = row index for p1
    row_two = row index for p2

- Return:
    Euclidian distance between the two data points
*/
double calculate_euclidian(double dataframe[NUM_ROWS][NUM_COLS], int row_one, int row_two, bool verbose) {
    double distance = 0;

    // Using NUM_COLS - 1, beause we don't want to include the target column at index 12
    for(int i = 0; i < (NUM_COLS - 1); i++) {
        double p_dist = dataframe[row_one][i] - dataframe[row_two][i];
        p_dist *= p_dist;
        distance += p_dist;
    }

    distance = sqrt(distance);

    if(verbose) {
        cout << "Distance from entry (" << (row_one) << ") and entry (" << row_two << ") is: " << distance << endl;
    }

    return sqrt(distance);
}

/*
generate_neighbors(distances_list, verbose, k)
Sorts a list of neighbors to a query point to determine the k closest ones.

- Params:
    distances_list = array containing the following:
        [row ID][distance to query point]
    verbose = flag to indicate whether or not to print console output
    k = number of neighbors to return

- Return:
    void
*/
void sort_neighbors(double distances_list[TRAIN_POINTS][2], bool verbose, int k) {

}

/*
generate_neighbors(dataframe, fold_split, verbose, k)
- Computes the distances from a point to all others to determine closest points (neighbors)

- Params:
    dataframe = 2d array consisting of doubles, holding all input data
    fold_split = index indicating the start of the training portion of our data
    verbose = flag to indicate whether or not to print console output
    k = number of neighbors
    query = index of the row for which we want to calculate neighbor distances

- Return:
    void
*/
void generate_neighbors(double dataframe[NUM_ROWS][NUM_COLS], int fold_split,  bool verbose, int k, int query) {
    
    // Fold split should be 183 for basic version

    // If our test fold contains 182, start scanning at 183, etc.
    int num_train_points = NUM_ROWS - fold_split;
    double distances_list[num_train_points][2];

    clock_t start;
    start = clock();

    for(int i = fold_split; i < NUM_ROWS; i++) {
        double distance = calculate_euclidian(dataframe, query, i, false);
        distances_list[i - fold_split][0] = i;
        distances_list[i - fold_split][1] = distance;
        //cout << "Row (" << i << ") Distance = " << distances_list[i-fold_split][1] << endl;
    }

    clock_t elapsed;
    elapsed = clock() - start;

    if(verbose) {
        cout << "Query Index: " << query << endl;
        printf("Calculating distances took: %f seconds. \n", ((float)elapsed) / CLOCKS_PER_SEC);
    }

}




/*
calculate_weights(neighbors, verbose)

- Note:
    Because I don't have access to pandas columns containing tuples of (neighbor,dist)
    like in the python version of this function, I'm going to utilize the following logic:

    neighbors array looks like the following
    nb = neighbor, d = distance value
    (assuming k=5)
    [nb0, nb1, nb2, nb3, nb4, d0, d1, d2, d3, d4]

    So to retrieve nb0, access neighbors[0]
    to access d0, access neighbors[K + 0]

- Params:
    neighbors = array of integers containing neighbor indices and distances to target point
    verbose = boolean flag which determines whether or now we should print debugg output

- Return:
    neighbor_weights : array of doubles
        The weights for each of the k input neighbors, used to select the appropriate output class

- References:
    (Returning arrays in C++):
         https://www.tutorialspoint.com/cplusplus/cpp_return_arrays_from_functions.htm

         */
void calculate_weights(double neighbors[K], bool verbose) {
    int num_neighbors = K;
    double total_neighbor_dist = 0;

    for(int i = 0; i < num_neighbors; i++) {

        // Distances are held at (K + index) via above note.
        total_neighbor_dist += neighbors[K + i];
    }

    // The return array
    double neighbor_weights[K];

    for(int i = 0; i < num_neighbors; i++) {

        // Store scaled weights
        neighbor_weights[i] = (neighbors[K + i] /total_neighbor_dist);

        if(verbose) {
            cout << "Neighbor " << (i + 1) << " weight: " << neighbor_weights[i] << endl;
        }
    }

}

/*
evaluate_performance(predicted, actual)
    Evaluates the performance of the algorithm's predictions, in our case
    working with classification this will be done by producing an accuracy score

- Params:
    predicted = array of predicted values (order mattering)
    actual = array of actual values (order mattering)

- Return: double
    classification accuracy score

- Notes:
    the length of these input arrays might not be K, it could be fold size
    might need to cast to integers inside the function and take args as doubles

*/
void evaluate_performance(int predicted[NUM_ROWS], int actual[NUM_ROWS]) {
    int true_res = 0;
    int total = NUM_ROWS;

    for(int i = 0; i < total; i++) {
        if(predicted[i] == actual[i]) {
            true_res += 1;
        }
    }

    double score = (true_res / total);
    cout << "Classification Accuracy: " << score << endl;
}

/*
k_fold(dataframe, k, verbose)
    Performs k-fold cross validation, dividing up proper index segments and 
    evaluating each fold agains the rest to determine proper splits

- Params:
    dataframe = 2d array consisting of doubles, holding all input data
    folds = number of folds to split the data into
    verbose = boolean argument to signify whether or not we should print debug output

- Return:
    Still figuring this out, in the python version I returned two lists, but not entirely
    sure how to implement that in this version

- Note:
    on second though I might just hard code these values instead of determining them here, 
    I can just pull them from a test run in the equivalent python function, for now leaving it here. 

*/
void k_fold(double dataframe[NUM_ROWS][NUM_COLS], int folds, bool verbose) {

    // int fold_size = int(NUM_ROWS / folds);

    // ignoring for now.. 
}

/*
print_row(dataframe, row)
- Helper function used for debugging purposes to see contents of a row.

- Params:
    dataframe = 2d array consisting of doubles, holding all input data
    row = desired row of the dataframe to print
*/
void print_row(double dataframe[NUM_ROWS][NUM_COLS], int row) {

    cout << "Row: " << row << endl;
    for(int i = 0; i < NUM_COLS; i++) {
        cout << dataframe[row][i] << " ";
    }
    cout << endl;
}


int main() {

    double dataframe[918][12];

    // Opening our data file
    string line;
    string element;
    ifstream inputfile("datasets/clean_data.csv");

    // Processing input file line by line
    int row = 0;

    // Start file stream
    if(inputfile.is_open()) {
        
        // While there is a line to read (should happen 918 times)
        while(getline(inputfile,line)) {

            // Don't process the line of column names
            if(row == 0) {
                getline(inputfile,line);
            }

            // Tokenize the line
            stringstream X(line);
            int column = 0;
            while(getline(X, element, ',')) {

                // Convert input file elements to doubles and store them in the dataframe
                double double_element = atof(element.c_str());
                dataframe[row][column] = double_element;
                column += 1;
            }
            row += 1;
        }

        // Done Reading 
        inputfile.close();
    } else {
        cout << "unable to open data file";
    }

    // Test = 0-182
    // Train = 183-917

    //print_row(dataframe, 0);
    //print_row(dataframe, 200);

    generate_neighbors(dataframe, 183, true, 5, 12);

    return 0;
}
