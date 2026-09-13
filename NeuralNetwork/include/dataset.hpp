#include <iostream>
#include <fstream>
#include <string>
#include "matrix.hpp"


using namespace std;

#pragma once


class DataSet {
    private:
        ifstream file; //file stream
        unsigned int batchSize; //fixed batch size
        vector<int> currentLabels; //labels of last batch
        bool endFlag; //true if end of set
        float scale; //scales all values


        vector<float> toVector(string & line) {
            vector<float> result;

            string value = "";
            for(char c : line) {
                if(c == ',') {
                    result.push_back(stof(value) / scale);
                    value = "";
                } else {
                    value += c;
                }
            }

            if (!value.empty()) {
                result.push_back(stof(value));
            }

            currentLabels.push_back(int((result[0]*scale +0.5)));
            result.erase(result.begin());

            return result;
        }

    public:

        DataSet() {batchSize = 1;}

        DataSet(string fileName, unsigned int const batch, float const scaling) {
            file.open(fileName);
            batchSize = batch;
            endFlag = false;
            scale = scaling;
        }


        Matrix<float> getNextBatch() {
            currentLabels.clear();

            Matrix<float> batch;

            string line = "";
            for(int i = 0; i < batchSize; i++) {
                if(!file.eof()) {
                    getline(file, line);
                    if(line.empty()) continue;
                    batch.addRow(toVector(line));
                } else {
                    endFlag = true;
                    break;
                }
            }

            return batch;
        }


        void resetSet() {
            file.seekg(0);
            endFlag = false;
        }


        vector<int> getCurrentLabels() {return currentLabels;}


        bool const reachedEnd() {return endFlag;}

        unsigned int const getBatchSize() {return batchSize;}
};