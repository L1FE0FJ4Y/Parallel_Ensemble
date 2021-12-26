#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <string>


class svm {
    public:
        svm(float learning_rate, int epochs);
        float dot(std::vector<float> X, std::vector<float> Y);
        float predict(std::vector<float> X);
        void train();
        void importData(int mode, std::string fileName);
        std::vector<float> exportDataVector();
        void importDataVector(int mode, std::vector<std::vector<float>> input_data);
        float testModel();
        void printData(std::vector<std::vector<float> > data);
        void printLR();
        void printEpochs();
        void printMWeight();
        
    private:   
        float s_lr;
        int s_epochs;
        std::vector<std::vector<float> > train_data;
        std::vector<std::vector<float> > test_data;
        std::vector<float> model_weights;
};

