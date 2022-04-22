#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#define NUM_COLS 12
#define NUM_ROWS 918


using namespace std;

/*

print_row(dataframe, row)
- Helper function used for debugging purposes to see contents of a row.

- Params:
    dataframe = 2d array consisting of doubles
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
