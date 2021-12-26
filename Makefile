CC = g++
CFLAGS = -g -Wall -fopenmp
TARGET = all

all: parallelmodels linearmodels parallelensemble linearensemble

parallelmodels: runmodels.cpp perceptron.cpp svm.cpp common.c 
	$(CC) $(CFLAGS) -o parallelmodels runmodels.cpp perceptron.cpp svm.cpp common.c 

linearmodels: runmodels.cpp linearperceptron.cpp linearsvm.cpp common.c 
	$(CC) $(CFLAGS) -o linearmodels runmodels.cpp linearperceptron.cpp linearsvm.cpp common.c 

parallelensemble: ensemble.cpp perceptron.cpp svm.cpp common.c 
	$(CC) $(CFLAGS) -o parallelensemble ensemble.cpp perceptron.cpp svm.cpp common.c 

linearensemble: linearensemble.cpp linearperceptron.cpp linearsvm.cpp common.c 
	$(CC) $(CFLAGS) -o linearensemble linearensemble.cpp linearperceptron.cpp linearsvm.cpp common.c 

clean:
	$(RM) parallelmodels linearmodels parallelensemble linearensemble