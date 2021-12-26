#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <omp.h>
#include "perceptron.h"

using namespace std;


//Initialize the Perceptron model with specified hyperparameters: learning_rate, epochs
perceptron::perceptron(float lr, int epochs) {
    p_epochs = epochs;
    p_lr = lr;
    model_bias = 0.00;
}


//Dot product utility function between two vector values X and Y
//Computes vector multiplication in parallel by reduction
float perceptron::dot(std::vector<float> X, std::vector<float> Y) {
    float res = 0.0;
    int y_len = Y.size(); 
    #pragma omp parallel for reduction (+:res)
    for (int i = 0; i <= y_len; i++) {
        res += (X[i] * Y[i]);
    }
    return res;
}


//Prediction function, utilizes dot product function to compute resultant value for prediction, method of measuring activation
float perceptron::predict(std::vector<float> X) {
    float activation = perceptron::dot(X, model_weights);
    activation += model_bias;
    return activation;
}


//Perceptron training function with the imported training data set
vector<float> perceptron::train() {
    //last element is answer, so ignore
    int col_len = train_data[0].size()-1;
    int row_len = train_data.size();

    std::vector<float> vec(col_len, 0.0);
    model_weights = vec;
    // for faster data reading
    std::vector<std::vector<float> > X = train_data;
    float b = model_bias;
    // lock to update
    omp_lock_t updatelock;
    omp_init_lock(&updatelock);


    // Iterate based upon number of epochs
    for (int i = 0; i < p_epochs; i++) {
        #pragma omp parallel for schedule(static)
        for(int row = 0; row < row_len; row++){
            std::vector<float> data = X[row];
            int Y = data.back();

            omp_set_lock(&updatelock);
            float pred = perceptron::predict(data);
            if ((Y * pred) <= 0) { //Check activation 
                #pragma omp parallel for schedule(static)
                for (int col = 0; col < col_len; col++){
                    model_weights[col] += (Y * p_lr * data[col]); //Update model weights
                }
                b += Y * p_lr; //Update bias
            }
            omp_unset_lock(&updatelock);
        }
    }
    omp_destroy_lock(&updatelock);

   vector<float> ret_weight = model_weights;
   return ret_weight;
}


//Test the trained Perceptron model against test data
float perceptron::testModel() {
    int row_len = test_data.size();
    int correct = 0;

    #pragma omp parallel for schedule(static) reduction(+:correct)
    for(int row = 0; row < row_len; row++){
        int ans = test_data[row].back();
        float pred = perceptron::predict(test_data[row]); //Predict
        if ((ans * pred) > 0){ //Check if prediction is valid, if so increment correct
            correct += 1.0;
        }
    }
    float acc = correct / test_data.size(); //Compute ratio of correct predictions over size of data
    cout << "Accuracy : " << acc << endl;
    return acc;
}


//Import data into the perceptron model based upon mode and specified filename to read from
void perceptron::importData(int mode, std::string fileName) {
    ifstream file (fileName);
    string line, val;
    getline(file, line);    
    while (getline(file, line)){
        vector<float> v;
        stringstream s(line);
        while (getline(s, val, ',')){
            v.push_back(stof(val));
        }
        //mode 1 for train, 0 for test
        if (mode){
            train_data.push_back(v);
        } else test_data.push_back(v);
    }
    file.close();
}


//Import data vector the perceptron model if the vector already exists, and based on specified mode
void perceptron::importDataVector(int mode, vector<vector<float>> input_data) {
    if (mode) {
        train_data = input_data;
    } else {
        test_data = input_data;
    }
}


//Helper function to return model weights for debug
vector<float>  perceptron::exportDataVector() {
    return model_weights;
}


//Helper function to print data for debug
void perceptron::printData(vector<vector<float> > data){
    for (vector<float>i : data) {
        for (float j : i){
            cout << j << ",";
        } 
        cout << "\n";
    }
}


//Helper function to print epochs for debug
void perceptron::printEpochs() {
    cout << "Model Epochs: " << p_epochs << endl;
}


//Helper function to print learning rate for debug
void perceptron::printLR() {
    cout << "Learning Rate: " << p_lr << endl;
}


//Helper function to print model weights for debug
void perceptron::printMWeight() {
    cout << "Model Weights" << endl;
    for (float j : model_weights){
            cout << j << "\t";
        } 
    cout << endl;   
}


//Helper function to print bias for debug
void perceptron::printBias() {
    cout << "Model Bias: " << float(model_bias) << endl;
}
