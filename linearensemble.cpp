//Linear version of the Stacking Ensemble Implementation
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <random>
#include <bits/stdc++.h>
#include "perceptron.h"
#include "svm.h"
#include "common.h"

#define NUM_MODELS 2
#define NUM_FOLDS 5

using namespace std;


void print_border(){
    fprintf(stdout, "=================================================================\n");
}


vector<vector<float>> importData(int mode, std::string fileName) {
    fprintf(stdout, "Importing data from %s\n", fileName.c_str());
    ifstream file (fileName);
    vector<vector<float> >train_data;
    vector<vector<float>> test_data;
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

    if (mode) {
        fprintf(stdout, "Returning train data\n");
        print_border();
        return train_data;
    } else {
        fprintf(stdout, "Returning test data\n");
        print_border();
        return test_data;
    }
}


int random(int min, int max) { //[min, max-1]]
    static bool first = true;
   if (first) 
   {  
      srand( time(NULL) ); 
      first = false;
   }
   return min + rand() % (( max) - min); //Add 1 to include max
}


vector<vector<vector<float>>> kFold_validation(vector<vector<float> > dataset) {
    fprintf(stdout, "%d Fold validation\n", NUM_FOLDS);

    unsigned int foldsize = dataset.size() / NUM_FOLDS;
    std::vector<std::vector<float>> copy_dataset = dataset;
    std::vector<std::vector<std::vector<float>>> split_dataset;
    
    for (int i = 0; i < NUM_FOLDS; i++) {
        vector<vector<float>> temp;
        while (temp.size() < foldsize) {
            int index = random(0, copy_dataset.size());
            vector<float> temp2 = copy_dataset[index];
            temp.push_back(temp2);
            copy_dataset.erase(copy_dataset.begin() + index);
        } 
        split_dataset.push_back(temp);
        
    }
    fprintf(stdout, "Done\n");
    print_border();
    return split_dataset;
}


float stacking (vector<vector<float> >training_data, vector<vector<float> >testing_data) {
    vector<vector<int>> test;  
    vector<perceptron> zero_models;

    for (int i = 0; i < NUM_MODELS; i++) {
        perceptron init = perceptron(0.1, 500);
        init.importDataVector(1, training_data);
        init.importDataVector(0, testing_data);
        init.train();
        zero_models.push_back(init);
    }

    vector<vector<float>> stacked_data;
    for (unsigned int i = 0; i < training_data.size(); i++) {
        vector<float> row_stack;
        for (int j = 0; j < NUM_MODELS; j++) {
            row_stack.push_back(zero_models[j].predict(training_data[i]));
        }
        row_stack.push_back(training_data[i].back());
        vector<float> row_dupe = training_data[i];
        row_dupe.pop_back();
        row_dupe.insert(row_dupe.end(), row_stack.begin(), row_stack.end());
        stacked_data.push_back(row_dupe);
    }

    svm level_one_svm = svm(0.001, 500);
    level_one_svm.importDataVector(1, stacked_data);
    level_one_svm.train();
    
       
    level_one_svm.importDataVector(0, testing_data);
    fprintf(stdout, "Each Fold Accuracy : \t");
    return level_one_svm.testModel();
}

void print_time(double timer[])
{
    fprintf(stdout, "Stack Module\t\tTime\n");
    for (int i = 1; i <= NUM_FOLDS; i++)
    {
        fprintf(stdout, "Fold %d Time : \t\t", i);
        fprintf(stdout, "%f\n", timer[i]);
    }
    fprintf(stdout, "Ensemble Time : \t\t");
    fprintf(stdout, "%f\n", timer[0]);
}



int main() {
    fprintf(stdout, "\n\n\n");

    uint64_t t0;
    uint64_t t1;
    double timer[NUM_FOLDS + 1];
    for(int i = 0; i <= NUM_FOLDS; i++) {
        timer[i] = 0.0;
    }
    InitTSC();
    
    t0 = ReadTSC();
    timer[0] += ElapsedTime(ReadTSC() - t0);

    fprintf(stdout, "Ensemble with Perceptron and SVM\n\n");
    vector<vector<float>> training_data = importData(1, "./data/mushtrain.csv");
    vector<vector<float>> test_data = importData(0, "./data/mushtest.csv");

    vector<vector<vector<float>>> folds = kFold_validation(training_data);
    
    float sum_acc = 0.0;
    for (int i = 0; i < NUM_FOLDS; i++) {
        fprintf(stdout, "Fold %d : Initiate\n", i+1);
        t1 = ReadTSC();
        sum_acc += stacking(folds[i], test_data);
        timer[i+1] += ElapsedTime(ReadTSC() - t1);
        fprintf(stdout, "Fold %d : Done\n", i+1);
    }
    timer[0] += ElapsedTime(ReadTSC() - t0);

    print_border();
    fprintf(stdout, "Ensemble Accuracy : %f\n",  sum_acc / NUM_FOLDS);
    fprintf(stdout, "Ensemble Done\n");
    print_border();

    print_time(timer);
    print_border();
    return 1;
}