#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <omp.h>
#include "svm.h"

using namespace std;


//Initialize the SVM model with specified hyperparameters: learning_rate, epochs
svm::svm(float learning_rate, int epochs) {
    s_lr = learning_rate;
    s_epochs = epochs;
}


//Dot product utility function between two vector values X and Y
//Computes vector multiplication in parallel by reduction
float svm::dot(std::vector<float> X, std::vector<float> Y) {
    float res = 0.0;
    int y_len = Y.size(); 
    #pragma omp parallel for reduction (+:res)
    for (int i = 0; i <= y_len; i++) {
        res += (X[i] * Y[i]);
    }
    return res;
}


//Prediction function, utilizes dot product function to compute resultant value for prediction
float svm::predict(std::vector<float> X) {
        return svm::dot(X, model_weights);
}


//SVM training function with the imported training data set
void svm::train() {
    //last element is answer, so ignore
    int col_len = train_data[0].size()-2;
    int row_len = train_data.size();

    std::vector<float> vec(col_len, 0.0);
    model_weights = vec;
    // for faster data reading
    std::vector<std::vector<float> > X = train_data;

    // lock for weight vector to update
    omp_lock_t* weightlock = (omp_lock_t*) malloc(sizeof(omp_lock_t) * col_len);
    for(int i = 0; i <= col_len; i++) {
        omp_init_lock(&(weightlock[i]));
    }

    // Iterate based upon number of epochs
    for (int i = 1; i <= s_epochs; i++) {
        #pragma omp parallel for schedule(static)
        for(int row = 0; row < row_len; row++) {
            std::vector<float> data = X[row];
            float Y = data.back();
            float pred = svm::predict(data); //Form prediction
            float lambda = -2 / s_epochs;
            if ((Y * pred) >= 1){ //Check prediction and update model weight based upon update val
                #pragma omp parallel for 
                for (int col = 0; col <= col_len; col++){
                    float update = s_lr * (lambda * model_weights[col]);
                    omp_set_lock(&weightlock[col]); //Lock before modifying model weight
                    model_weights[col] += update;
                    omp_unset_lock(&weightlock[col]); //Unlock 
                }
            }
            else{
                #pragma omp parallel for 
                for (int col = 0; col <= col_len; col++){
                    float update = s_lr * ((data[col] * Y) + lambda * model_weights[col]);
                    omp_set_lock(&weightlock[col]); //Lock before modifying model weight
                    model_weights[col] += update;
                    omp_unset_lock(&weightlock[col]); //Unlock 
                }
            }
        }
    }

    for(int i = 0; i <= col_len; i++) {
        omp_destroy_lock(&(weightlock[i]));
    }
    free(weightlock);
}


//Import data files into the model based upon specified data mode and data
void svm::importDataVector(int mode, vector<vector<float>> input_data) {
    if (mode) {
        train_data = input_data;
    } else {
        test_data = input_data;
    }
}


//Helper function to return model weights for debugging
vector<float> svm::exportDataVector() {
    return model_weights;
}


//Import data files into the model based upon specified data mode and fileName 
void svm::importData(int mode, std::string fileName) {
    ifstream file (fileName);
    string line, val;
    getline(file, line);    
    while (getline(file, line)) { //Retrieve lines
        vector<float> v;
        stringstream s(line);
        while (getline(s, val, ',')) {
            v.push_back(stof(val));
        }
        //mode 1 for train, 0 for test
        if (mode) { //If mode is set we are inserting to training
            train_data.push_back(v);
        } else test_data.push_back(v); //Else we are inserting to test data
    }
    file.close();
}


//Test the trained model against test data
float svm::testModel() {
    float correct = 0;
    int row_len = test_data.size();

    //Compute the predictions and determine if the prediction is correct or not, if true increment ocrrect
    #pragma omp parallel for schedule(static) reduction (+:correct)
    for(int row = 0; row < row_len; row++){
        int ans = test_data[row].back();
        test_data[row].pop_back();
        float pred = svm::predict(test_data[row]);
        if ((ans * pred) > 0){
            correct += 1.0;
        }
    }
    float acc = correct / test_data.size(); //Compute ratio of correct answers across total data count
    cout << "Accuracy : " << acc << endl;
    return acc;
}


//Helper function to debug and print data vector
void svm::printData(vector<vector<float> > data){
    for (vector<float>i : data) {
        for (float j : i){
            cout << j << ",";
        } 
        cout << "\n";
    }
}


//Helper function to debug and print out LR 
void svm::printLR() {
    cout << "Learning Rate: " << s_lr << endl;
}


//Helper function to debug and print out epochs
void svm::printEpochs() {
    cout << "Model Epochs: " << s_epochs << endl;
}


//Helper function to debug and print out model weights
void svm::printMWeight() {
    cout << "Model Weights" << endl;
    for (float j : model_weights){
            cout << j << "\t";
        } 
    cout << endl;   
}