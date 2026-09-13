#include "layertypes.hpp"
#include <memory>
#include "dataset.hpp"


class ClassifierNetwork
{
private:
    vector<unique_ptr<NeuronLayer>> layers;
    ALSoftmaxCategoricalCrossEntropy loss;
    vector<int> netStruct;

    Matrix<float> lastForwardInput;



    int maxIndex(vector<float> & sample) {
        int index = 0;
        float maxValue = sample[0];

        for(int i = 0; i < sample.size(); i++) {
            if (sample[i] > maxValue) {
                maxValue = sample[i];
                index = i;
            }
        }

        return index;
    }


    unsigned int compare(vector<int> & targets, vector<int> & predictions) {
        if(targets.size() != predictions.size()) throw overflow_error("u stupid!");

        unsigned int counter = 0;

        for(int i = 0; i < targets.size(); i++) {
            if(targets[i] == predictions[i]) counter++;
        }
        return counter;
    }


public:
    ClassifierNetwork(vector<int> & structure) {
        netStruct = structure;

        if(netStruct.size() < 2) throw exception();

        for (int i = 1; i < netStruct.size(); i++) {
            layers.push_back(make_unique<LayerDense>(netStruct[i-1], netStruct[i]));
            layers.push_back(make_unique<ActivationReLU>());
        }

        layers.pop_back();                                             //remove last activation function
    }

    ClassifierNetwork(string filePath) {
        ifstream myFile;
        string line;
        string cell;
        myFile.open(filePath);

        getline(myFile, line);
        if(line != "ClassifierNetwork,") throw logic_error("Wrong file for wrong class, dude!");

        getline(myFile, line); //get net structure

        //read net structure
        for(char c : line) {
            if(c != ',') cell += c;
            else {
                netStruct.push_back(stoi(cell));
                cell.clear();
            }
        }


        //apply net structure
        if(netStruct.size() < 2) throw underflow_error("struct not correct");

        for (int i = 1; i < netStruct.size(); i++) {
            layers.push_back(make_unique<LayerDense>(netStruct[i-1], netStruct[i], false));
            layers.push_back(make_unique<ActivationReLU>());
        }

        layers.pop_back(); //remove last activation function


        //set all weights and biases to saved values
        int ic;
        for(int layerClass = 0; layerClass < layers.size(); layerClass++) {     //iterate through all layers
            if(layers[layerClass] -> identify() == "LayerDense") {              //look at all layer denses, ignore RelU and Softmax (if given)
                
                Matrix<float> & setWeights = layers[layerClass] -> getWeights();//get weights

                for(vector<float> & row : setWeights) {
                    getline(myFile, line);
                    ic = 0;

                    for(int column = 0; column < row.size(); column++) {
                        cell.clear();
                        while(line[ic] != ',') {
                            cell += line[ic];
                            ic++;
                        }
                        row[column] = stof(cell);
                        ic++;
                    }
                }

                getline(myFile, line);
                ic = 0;
                vector<float> & setBiases = layers[layerClass] -> getBiases();  //get weights
                for(int iBias = 0; iBias < setBiases.size(); iBias++) {
                    cell.clear();
                    while(line[ic] != ',') {
                        cell += line[ic];
                        ic++;
                    }
                    setBiases[iBias] = stof(cell);
                    ic++;
                }
            }
        }
    }


    //forward pass of network, generates output and returns loss
    float forward(Matrix<float> & input, vector<int> & targets) {

        lastForwardInput = input; // saves that for backward pass

        layers[0] -> forward(input);
        for(int i = 1; i < layers.size(); i++) {
            layers[i] ->forward(layers[i-1]->output);
        }

        loss.forward(layers.back() -> output, targets);
        return loss.output;
    }

    

    //backward pass of network, sets derivative matrixes of layer classes
    void backward(vector<int> & targets) {
        loss.backward(layers.back() -> output, targets);

        

        layers.back() -> backward(layers[layers.size()-2] -> output, loss.gradient);

        for(int i = layers.size() - 2; i > 0; i--) {
            layers[i] -> backward(layers[i-1] -> output, layers[i+1] ->gradient);
            //cout << i;
        }

        layers[0] -> backward(lastForwardInput, layers[1] ->gradient);
    }



    void gradientDescent(float const & learningRate) {
        for(int i = 0; i < layers.size(); i++) layers[i] -> gradientDescent(learningRate);
    }


    vector<int> predict(Matrix<float> & input) {
        Matrix<float> predictInput = input;


        for(int i = 0; i < layers.size(); i++) {
            layers[i] -> predict(predictInput);
        }

        loss.predict(predictInput);

        vector<int> predictions;
        for(vector<float> & sample : predictInput) {
            predictions.push_back(maxIndex(sample));
        }

        return predictions;
    }


    void printOutput() {
        for(int i = 0; i < layers.size(); i++) {
            cout << "Layer class " << i << " output:" << endl;
            cout << layers[i] -> output << endl;
        }

        cout << "Loss class output:" << endl;
            cout << loss.output << endl;
    }


    void printGradients() {
        //printing out all derivatives
        for(int i = 0; i < layers.size(); i++) {
            cout << "Layer class " << i << " derivative:" << endl;
            cout << layers[i] ->gradient << endl;
        }

        cout << "Loss class derivative:" << endl;
            cout << loss.gradient << endl;
    }



    void train(DataSet & set, float learningRate, unsigned int batchCycle = 1) {
        Matrix<float> batch;
        vector<int> targets;

        unsigned int batchCounter = 0;

        while (!set.reachedEnd())
        {
            batchCounter += set.getBatchSize();

            batch = set.getNextBatch();
            targets = set.getCurrentLabels();
            for(int i = 0; i < batchCycle; i++) {
                cout << batchCounter << ":   " << forward(batch, targets) << endl;
                backward(targets);
                gradientDescent(learningRate);
            }
        }
        
    }


    float testNet(DataSet & set) {
        Matrix<float> batch;
        vector<int> targets;
        vector<int> predictions;

        unsigned int batchCounter = 0;
        unsigned int correctCounter = 0;

        while (!set.reachedEnd())
        {
            batch = set.getNextBatch();
            targets = set.getCurrentLabels();
            predictions = predict(batch);

            batchCounter += batch.size();
            correctCounter += compare(targets, predictions);

            cout << batchCounter << ": " << (float)correctCounter / batchCounter * 100.0f << endl;
        }
        return (float)correctCounter / batchCounter * 100.0f;
    }


    unsigned int comp(vector<int> targets, vector<int> predictions) {
        return compare(targets, predictions);
    }


    void save(string path) {
        ofstream myFile;

        myFile.open(path);
        myFile << "ClassifierNetwork,\n";

        for(int neurons : netStruct) myFile << neurons << ",";
        myFile << "\n";


        for(int layerDense = 0; layerDense < layers.size(); layerDense += 2) {
            Matrix<float> & weights = layers[layerDense] -> getWeights();
            for(vector<float> & row : weights) {
                for(float & weight : row) {
                    myFile << weight << ",";
                }
                myFile << "\n";
            }

            vector<float> & biases = layers[layerDense] -> getBiases();
            for(float & bias : biases) myFile << bias << ",";
            myFile << "\n";
        }
    }
};
