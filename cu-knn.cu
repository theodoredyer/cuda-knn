#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;


int main() {

    // Opening our data file
    string file_loc = "/datasets/clean_data.csv";
    string line;
    ifstream inputfile(file_loc);

    // Processing input file line by line
    if(inputfile.is_open()) {
        while(getline(inputfile,line) && tracker < 10) {
            tracker += 1;
            cout << line << '\n';
        }
        inputfile.close();
    } else {
        cout << "unable to open file";
    }

    return 0;
}
