#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <string>


class perceptron {
    public:
        perceptron(float lr, int epochs);
        float dot(std::vector<float> X, std::vector<float> Y);
        float predict(std::vector<float> X);
        std::vector<float> train();
        float testModel();
        void importData(int mode, std::string fileName);
        void importDataVector(int mode, std::vector<std::vector<float>> input_data);
        std::vector<float> exportDataVector();
        void printData(std::vector<std::vector<float> > data);
        void printEpochs();
        void printLR();
        void printMWeight();
        void printBias();

    private:
        int p_epochs;
        float p_lr;
        std::vector<std::vector<float> > train_data;
        std::vector<std::vector<float> > test_data;
        std::vector<float> model_weights;
        float model_bias;
};

