#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;


int main() {

    double dataframe[918][12];

    // Opening our data file
    string line;
    string element;
    ifstream inputfile("datasets/clean_data.csv");

    // Processing input file line by line
    int iterations = 0;

    // Start file stream
    if(inputfile.is_open()) {
        
        // While there is a line to read (should happen 917 times)
        while(getline(inputfile,line) && tracker < 1) {

            // cout << line << endl;

            // Tokenize the line
            stringstream X(line);
            while(getline(X, element, ',')) {
                cout << element << endl;
            }

            iterations += 1;
        }

        // Done Reading 
        inputfile.close();
    } else {
        cout << "unable to open data file";
    }

    return 0;
}
