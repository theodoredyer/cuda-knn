all: cu-knn.cu
	nvcc cu-knn.cu -o exe

clean:
	rm exe
