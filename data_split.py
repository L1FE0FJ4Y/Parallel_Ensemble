import random
import sklearn.model_selection
import pandas 
import numpy


#Generic helper function to read out and parse data about .data file and read into a csv format
def read_iris(fileName, identifier, outputFile):
    names = open("./iris_data/attributes.names", "r")
    attributes = names.read() + "\n"
    fp = open(fileName, "r")
    ofp = open(outputFile, "w")
    ofp.write(attributes)
    for line in fp:
        line = line.strip()
        items = line.split(",")
        if items[-1] == identifier:
            items[-1] = '1\n'
        else:
            items[-1] = '0\n'

        new_items = ",".join(items)
        ofp.write(new_items) 
    fp.close() 
    ofp.close()

    remove_line = open(outputFile, "r")
    lines = remove_line.read()
    remove_line.close()
    all = lines.split("\n")
    s = "\n".join(all[:-2])
    remove_line = open(outputFile, "w+")
    for i in range(len(s)):
        remove_line.write(s[i])
    remove_line.close()


#Primary program driver
def main():
    read_iris('./iris_data/iris.data', "Iris-virginica", "./test2.csv")  #Read .data files, specify type to determine, write formatted file into .csv filename

    data_frame = pandas.read_csv('./test2.csv', index_col = 0) #Read the dataframe from the csv file we just wrote

    X_training, X_testing = sklearn.model_selection.train_test_split(data_frame, test_size=0.3, shuffle=True) #Use the SKLearn module to perform a 70/30 split with shuffle since data is in order
    X_testing.to_csv('./iris_data/test_data.csv') #Write testing dataframe to csv
    X_training.to_csv('./iris_data/training_data.csv') #Write training dataframe to csv
    return 1


if __name__ == '__main__':
    main()

