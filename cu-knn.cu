#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;


int main() {

    // Opening our data file
    string line;
    ifstream inputfile("datasets/clean_data.csv");


    // Processing input file line by line
    int tracker = 0;
    if(inputfile.is_open()) {
        while(getline(inputfile,line) && tracker < 1) {
            tracker += 1;
            cout << line << '\n';
        }
        inputfile.close();
    } else {
        cout << "unable to open file";
    }

    return 0;
}
