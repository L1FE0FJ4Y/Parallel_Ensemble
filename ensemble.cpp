#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
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


//Utility function to print a border for the output to terminal.
void print_border(){
    fprintf(stdout, "=================================================================\n");
}


//Import data function to import training and testing data in .csv format
//Mode paramater indicates if it is importing a testing or training file
vector<vector<float>> importData(int mode, std::string fileName) {
    fprintf(stdout, "Importing data from %s\n", fileName.c_str());
    ifstream file (fileName);
    vector<vector<float> >train_data;
    vector<vector<float>> test_data;
    string line, val;
    getline(file, line);    
    while (getline(file, line)){ //Retrieve each line within the csv files
        vector<float> v;
        stringstream s(line);
        while (getline(s, val, ',')){ //Retrieve each value within the row
            v.push_back(stof(val));
        }
        //mode 1 for train, 0 for test
        if (mode){
            train_data.push_back(v);
        } else test_data.push_back(v);
    }
    file.close();

    if (mode) { //Check if mode is set
        fprintf(stdout, "Returning train data\n");
        print_border();
        return train_data;
    } else { // No mode set means it is test data
        fprintf(stdout, "Returning test data\n");
        print_border();
        return test_data;
    }
}


//Random function to return a value between minimum and maximum 
//Takes two parameters indicating the bounds desired for random value return
int random(int min, int max) { //[min, max-1]]
    static bool first = true;
    if (first) {  
      srand( time(NULL) ); 
      first = false;
    }
   return min + rand() % (( max) - min); //Add 1 to include max, otherwise it will be up until bounds mentioned above.
}


//KFoldValidation function to split the provided dataset into the specified folds defined at the top
//Dataset is the entire data set from which you want to split into folds.
vector<vector<vector<float>>> kFold_validation(vector<vector<float> > dataset) {
    fprintf(stdout, "%d Fold validation\n", NUM_FOLDS);

    unsigned int foldsize = dataset.size() / NUM_FOLDS; //Find the size of each fold
    std::vector<std::vector<float>> copy_dataset = dataset;
    std::vector<std::vector<std::vector<float>>> split_dataset;
    
    for (int i = 0; i < NUM_FOLDS; i++) { //Initialize each folds array and append each formed fold
        vector<vector<float>> temp;
        while (temp.size() < foldsize) { //While the fold size has not been met we continue to add into our data fold
            int index = random(0, copy_dataset.size());
            vector<float> temp2 = copy_dataset[index];
            temp.push_back(temp2);
            copy_dataset.erase(copy_dataset.begin() + index);
        } 
        split_dataset.push_back(temp); //Foldsize has been met incorporate this fold set into our total set of folds.
        
    }
    fprintf(stdout, "Done\n");
    print_border();
    return split_dataset; //Return the dataset split into the folds
}


//Stacking Ensemble Implementation with 2 Level 0 Perceptron Learners and 1 Level 1 SVM Learner
//Parameters of training_data and testing_data are provided from your training test split.
float stacking (vector<vector<float> >training_data, vector<vector<float> >testing_data) {
    vector<vector<int>> test;  
    vector<perceptron> zero_models; //Initialize an array of level 0 learners


    //Parallel training for level 0 learners, each model imports its data and trains itself, we push resultant trained model
    #pragma omp parallel for
    for (int i = 0; i < NUM_MODELS; i++) {
        perceptron init = perceptron(0.1, 500);
        init.importDataVector(1, training_data);
        init.importDataVector(0, testing_data);
        init.train();
        zero_models.push_back(init);
    }

    //Init stacked data set
    vector<vector<float>> stacked_data;
    for (unsigned int i = 0; i < training_data.size(); i++) { //Iterate through the entire training data
        vector<float> row_stack;
        for (int j = 0; j < NUM_MODELS; j++) {
            row_stack.push_back(zero_models[j].predict(training_data[i])); //Compute level 0 predictions
        }
        row_stack.push_back(training_data[i].back()); //Incorporate level 0 data predictions
        vector<float> row_dupe = training_data[i];
        row_dupe.pop_back();
        row_dupe.insert(row_dupe.end(), row_stack.begin(), row_stack.end()); //Compound the level 0 learning with the data set
        stacked_data.push_back(row_dupe);
    }

    svm level_one_svm = svm(0.001, 500); //Init SVM
    level_one_svm.importDataVector(1, stacked_data); //Use the new stacked data set for prediction
    level_one_svm.train(); //Train the SVM
    
       
    level_one_svm.importDataVector(0, testing_data); //Test upon test data
    fprintf(stdout, "Each Fold Accuracy : \t");
    return level_one_svm.testModel();
}


//Utility function to print timing for each fold 
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


//Primary program driver
int main() {
    fprintf(stdout, "\n\n\n");

    uint64_t t0;
    uint64_t t1;
    double timer[NUM_FOLDS + 1];
    for(int i = 0; i <= NUM_FOLDS; i++) { //Init timing
        timer[i] = 0.0;
    }
    InitTSC();
    
    t0 = ReadTSC();
    timer[0] += ElapsedTime(ReadTSC() - t0);

    fprintf(stdout, "Ensemble with Perceptron and SVM\n\n");

    vector<vector<float>> training_data = importData(1, "./data/mushtrain.csv"); //Import training data set (mushroom here)
    vector<vector<float>> test_data = importData(0, "./data/mushtest.csv"); //Import testing data set (mushroom here)

    vector<vector<vector<float>>> folds = kFold_validation(training_data);
    
    float sum_acc = 0.0;
    //Compute 5 fold average for the data to find average accuracy for ensemble stacking in parallel.
    #pragma omp parallel for schedule(static) private(t1) reduction(+:sum_acc)
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