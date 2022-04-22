#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>

#define NUM_COLS 12
#define NUM_ROWS 918
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
double calculate_euclidian(double dataframe[NUM_ROWS][NUM_COLS], int row_one, int row_two) {
    double distance = 0;

    // Using NUM_COLS - 1, beause we don't want to include the target column at index 12
    for(int i = 0; i < (NUM_COLS - 1); i++) {
        double p_dist = dataframe[row_one][i] - dataframe[row_two][i];
        p_dist *= p_dist;
        distance += p_dist;
    }

    return sqrt(distance);
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
double * calculate_weights(double neighbors[K], bool verbose) {
    int num_neighbors = K;
    total_neighbor_dist = 0;

    for(int i = 0; i < num_neighbors; i++) {

        // Distances are held at (K + index) via above note.
        total_neighbor_dist += neighbors[K + i];
    }

    // The return array
    double neighbor_weights[K];

    for(int i = 0; i < num_neighbors; i++) {

        // Store scaled weights
        neighbor_wights[i] = (neighbors[K + i] /total_neighbor_dist);

        if(verbose) {
            cout << "Neighbor " << (i + 1) << " weight: " << neighbor_weights[i] << endl;
        }
    }

    return neighbor_weights;
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

    print_row(dataframe, 918);

    return 0;
}
