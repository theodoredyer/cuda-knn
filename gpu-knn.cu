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
#define TEST_FOLD 138
#define FOLD_SPLIT 183
#define K 5
#define KERNEL_LOOP 2048
#define KERNEL_SIZE 918


using namespace std;


/*
print_row_1d(dataframe, row)
- Helper function used for debugging purposes to see contents of a simulated row in a 1d array.

- Params:
    dataframe = array consisting of doubles, holding all input data
    row = desired row of the dataframe to print
*/
void print_row_1d(double dataframe[NUM_ROWS *NUM_COLS], int row) {

    cout << "Row: " << row << endl;
    for(int i = 0; i < NUM_COLS; i++) {
        int idx = (row * NUM_COLS) + i;
        cout << dataframe[idx] << " ";
    }
    cout << endl;
}

/*
fill_host_1d
- Processes dataset into a 1 dimensional array.

- Params:
    dataframe = pointer to array whic we are going to populate with data

*/
__host__ void fill_host_1d(double host_data_1d[NUM_COLS * NUM_ROWS]) {
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
                int idx = (row * NUM_COLS + column);
                host_data_1d[idx] = double_element;
                column += 1;
            }
            row += 1;
        }

        // Done Reading 
        inputfile.close();
    } else {
        cout << "unable to open data file";
    }
}


// Note - this actually computes distances to other points in the test set as well.
__global__ void execute_distance_kernel(double data[NUM_ROWS * NUM_COLS], const int num_elements) {
    
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(tid < NUM_ROWS) {

        // printf("Executing with tid = %d\n", tid);

        // For each point in the training set..
        for(int i = FOLD_SPLIT; i < NUM_ROWS; i++) {
            double distance = 0;

            // Calculate Euclidean
            for(int j = 0; j < (NUM_COLS - 1); j++) {
                double p_dist;
                int tid_idx = (tid * NUM_COLS) + j;
                int i_idx = (i * NUM_COLS) + j;
                p_dist = data[tid_idx] - data[i_idx];
                p_dist *= p_dist;
                distance += p_dist;
            }

            distance = sqrt(distance);

            int dest_idx = (tid * NUM_COLS) + 11;
            data[dest_idx] = distance;
        }
    }
}


__host__ void gpu_kernel(void) {

    // Basic constants setup
    const unsigned int num_elements = NUM_ROWS * NUM_COLS;
    const unsigned int num_threads = KERNEL_SIZE;
    const unsigned int num_blocks = (num_elements + num_threads - 1)/num_threads;
    const unsigned int num_bytes = num_elements * sizeof(double);

    double * data_gpu;

    // Initialize host arrays
    double host_dataframe[NUM_ROWS * NUM_COLS];
    double host_dataframe_output[NUM_ROWS * NUM_COLS];

    // Allocate gpu array
    cudaMalloc(&data_gpu, num_bytes);

    // Populate host array
    fill_host_1d(host_dataframe);

    // Populate device memory with host array
    cudaMemcpy(data_gpu, host_dataframe, num_bytes, cudaMemcpyHostToDevice);

    // Setup execution time 
    clock_t start;
    start = clock();

    // Execute kernel function
    execute_distance_kernel <<<num_blocks, num_threads>>>(data_gpu, num_elements);
    
    // Calculate and print execution time 
    clock_t elapsed;
    elapsed = clock() - start;
    printf("%d point execution time.. \n", NUM_ROWS);
    printf("Calculating distances took: %f seconds. \n", ((float)elapsed) / CLOCKS_PER_SEC);


    cudaThreadSynchronize();
    cudaGetLastError();

    cudaMemcpy(host_dataframe_output, data_gpu, num_bytes, cudaMemcpyDeviceToHost);

    //print_row_1d(host_dataframe, 10);
    //print_row_1d(host_dataframe_output, 10);


}

void execute_host_functions() {

}

void execute_gpu_functions() {
    gpu_kernel();
}

int main(void) {
    execute_host_functions();
    execute_gpu_functions();

    return EXIT_SUCCESS;
}
