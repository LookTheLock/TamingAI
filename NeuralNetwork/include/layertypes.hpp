#include <random>
#include "matrix.hpp"
#include <cmath>
#include <algorithm>

#pragma once

class NeuronLayer {
    public:
        Matrix<float> output;  // output of layer/function after last forward pass

        Matrix<float> gradient;  // calculated derivative after backward pass

        virtual
        ~NeuronLayer() = default;

        virtual
        void forward(Matrix<float> & input) = 0; // takes previous layer output and calculates output of layer

        virtual
        void predict(Matrix<float> & input) = 0; //models input to output

        virtual
        void backward(Matrix<float> & pLOutput, Matrix<float> & nLDerivatives) = 0; // takes previous layer output and calculates derivative of class

        virtual
        void gradientDescent(float const & learningRate) {}

        virtual
        Matrix<float> & getWeights() {throw logic_error("This class does not have weights.");}

        virtual
        vector<float> & getBiases() {throw logic_error("This class does not have biases.");}

        virtual
        string const identify() = 0; // now u can differ between layer types in a list

};


//layer dense
class LayerDense : public NeuronLayer {
private:
    Matrix<float> weights;
    vector<float> biases;

    vector<float> biasGradients;
    Matrix<float> weightGradients;


public:
    LayerDense() {}
    
    LayerDense(int nInputs, int nNeurons, bool randSetup = true) {

        if(randSetup) {   
            vector<float> nextRow;
            random_device rd;
            mt19937 gen(rd());


            normal_distribution<float> dist(0.0, 0.1);

            for (int i = 0; i < nInputs; i++) {
                for (int j = 0; j < nNeurons; j++) {
                    nextRow.push_back(dist(gen));
                }

                weights.addRow(nextRow);
                nextRow.clear();
            }
        } else {
            Matrix<float> zMatrix(nInputs, nNeurons);
            weights = zMatrix;
        }

        for (int i = 0; i < nNeurons; i++) {
            biases.push_back(1);
        }

        if(false) {
            cout << "Weights:" << endl << weights << endl;
            cout << "Biases:" << endl;
            for (float i : biases) cout << i << " ";
            cout << endl;
        }
    }


    string const identify() override {return "LayerDense";}


    void forward(Matrix<float> & inputs) override {output = inputs * weights + biases;}


    void predict(Matrix<float> & input) {input = input * weights + biases;}


    void backward(Matrix<float> & pLOutput, Matrix<float> & nLDerivatives) override {

        //calculation of weight gradients
        weightGradients = pLOutput.returnT() * nLDerivatives;

        
        //bias gradients
        biasGradients = nLDerivatives[0];

        for (int row = 1; row < nLDerivatives.rows(); row++) {
            for (int column = 0; column < nLDerivatives.columns(); column++) {
                biasGradients[column] += nLDerivatives[row][column];
            }
        }

        //pass gardient for next layer
        Matrix<float> weightsTransformed = weights.returnT();
        gradient = nLDerivatives * weightsTransformed;


        //print gradients
        if (false) {
            if (weightGradients.rows() < 30) {
            cout << "weight gradients: " << endl;
            cout << weightGradients;

            cout << "bias gradients: " << endl;
            for (float i : biasGradients) cout << i << " ";
            cout << endl;
            }
        }
    }


    void gradientDescent(float const & learningRate) {

        //adjusting biases
        for(int i = 0; i < biases.size(); i++) biases[i] -= biasGradients[i] * learningRate;


        //adjusting weights
        Matrix<float> mulWeights = weightGradients * learningRate;
        weights -= mulWeights;
    }

    Matrix<float> & getWeights() override {return weights;}

    vector<float> & getBiases() override {return biases;}
};


/*
    All activation classes are to be found here.

    They will be considered a layer in a network class.
*/

class ActivationReLU : public NeuronLayer {
    public:

        ActivationReLU() {}


        string const identify() override {return "ActivationReLU";}


        void forward(Matrix<float> & input) override {
            output = input;

            for (vector<float> & row : output) {
                for (float & value : row) {
                    if (value < 0) {
                        value = 0;
                    }
                }
            }
        }


        void predict(Matrix<float> & input) {
            for (vector<float> & row : input) {
                for (float & value : row) {
                    if (value < 0) {
                        value = 0;
                    }
                }
            }
        }
        

        void backward(Matrix<float> & pLOutput, Matrix<float> & nLGradient) override {
            gradient = pLOutput;

            for (int row = 0; row < gradient.rows(); row++) {
                for (int column = 0; column < gradient.columns(); column++) {
                    if (gradient[row][column] <= 0) {
                        gradient[row][column] = 0;
                    }
                    else {
                        gradient[row][column] = nLGradient[row][column];
                    }
                }
            }
        }
};


class ActivationSigmoid : public NeuronLayer {
public:
    ActivationSigmoid() {}

    string const identify() override {return "ActivationSigmoid";}

    void forward(Matrix<float> & input) override {
        output = input;

        for (vector<float> & sample : output) {
            for (float& value : sample) {
                value = 1.0f / (1.0f + exp(-value));
            }
        }
    }


    void predict(Matrix<float> & input) override {
        for (vector<float> & sample : input) {
            for (float & value : sample) {
                value = 1.0f / (1.0f + exp(-value));
            }
        }
    }


    void backward(Matrix<float>& pLOutput, Matrix<float>& nLGradient) override {
        gradient = nLGradient;

        float precalculatedSigmoid = 0;

        for (int sample = 0; sample < gradient.size(); sample++) {
            for (int value = 0; value < gradient.columns(); value++) {
                precalculatedSigmoid = output[sample][value];
                gradient[sample][value] *= precalculatedSigmoid * (1.0f - precalculatedSigmoid);
            }
        }
    }
};


class ActivationSoftmax : public NeuronLayer {
    public:

        ActivationSoftmax() {}


        string const identify() override {return "ActivationSoftmax";}


        void forward(Matrix<float> & input) override {
            output = input;
            float rowSum;
            float rowMax;

            for (vector<float> & row : output) {

                rowSum = 0;
                rowMax = max(row);
                
                for (float & value : row) {
                    value -= rowMax;
                    rowSum += exp(value);
                }

                for (float & value : row) {
                    value = exp(value) / rowSum;
                }
            }
        }

        void backward(Matrix<float> & pLOutput, Matrix<float> & nLGradient) override {}

    private:
        float max(vector<float> &vector) {
            float max = vector[0];
            for (float i : vector) {
                if (i > max) {
                    max = i;
                }
            }
            return max;
        }
};

/*
    combined last-layer and loss functions for efficiency reasons
*/

class ALSoftmaxCategoricalCrossEntropy {  //Actually this is a Parse Categorical Cross Entropy. Yes, I did it on accident, because I am simply too smart and found this optimization myself :D
    public:
        float output;
        Matrix<float> gradient;


        ALSoftmaxCategoricalCrossEntropy() {}

        void forward(Matrix<float> & input, vector<int> & targets) {

            float loss = 0;

            float targetValue; //targeted value of sample
            float expSumRow;  //summed exp values; with overflow protection (subtracting highest value in sample)
            float sampleMax;

            for (int row = 0; row < targets.size(); row++) {
                sampleMax = maxVector(input[row]);
                expSumRow = expSum(input[row], sampleMax);
                targetValue = input[row][targets[row]] - sampleMax;

                loss += log(expSumRow) - targetValue;
            }

            output = loss / targets.size();
        }


        void predict(Matrix<float> & input) {
            float expSumSample;
            float maxValue;
            for (vector<float> & row : input) {
                maxValue = maxVector(row);
                expSumSample = expSum(row, maxValue);
                
                for (float & column : row) {
                    column = exp(column - maxValue) / expSumSample;
                }
            }
        }


        void backward(Matrix<float> & pLOutput, vector<int> & targets) {
            gradient = pLOutput;
            //cout << "Loss pLOutput:" << endl << pLOutput;
            //cout << "Loss Derivative (pLoutput moved into variable):" << endl << derivatives;

            float vSum; //expSum of each sample
            float sampleMax;
            int batchSize = gradient.rows();

            for (int row = 0; row < batchSize; row++) {
                sampleMax = maxVector(gradient[row]);
                vSum = expSum(gradient[row], sampleMax);

                for (int column = 0; column < gradient.columns(); column++) {
                    gradient[row][column] = (exp(gradient[row][column] - sampleMax) / vSum) / batchSize;
                }
                gradient[row][targets[row]] -= 1.0f / batchSize;
            }

            //cout << "Loss Derivative (manipulated variable so it should be the real derivatives):" << endl << derivatives;
        }

    private:
        float mean(vector<float> & losses) {
            float sum = 0.0f;
            for(float i : losses) {
                sum += i;
            }
            return sum / losses.size();
        }

        float maxVector(vector<float> & vector) {
            float max = vector[0];
            for (float i : vector) {
                if (i > max) {
                    max = i;
                }
            }
            return max;
        }

        float expSum(vector<float> & values, float & sub) {
            float expSum = 0.0f;

            for(float & value : values) {
                expSum += exp(value - sub);
            }

            return expSum;
        }
};


/*
    Loss Functions
*/

class LossCategoricalCrossEntropy {

    public:
        float output;
        Matrix<float> gradient;

        LossCategoricalCrossEntropy() {};

        void forward(Matrix<float> & netOutput, vector<int> targets) {
            if (netOutput.size() != targets.size()) {
                throw exception();
            }
            vector<float> losses;

            for(int i = 0; i < targets.size(); i++) {
                losses.push_back(-1*log(clip(netOutput[i][targets[i]])));
            }
            
            output = mean(losses);
        }


        void backward(Matrix<float> & pLOutput, vector<int> & targets) { //missing because not used yet
            gradient = pLOutput;

        }

    private:
        float clip(float x) {
            return max(1e-7f, min(x, 1.0f - 1e-7f));
        }

        float mean(vector<float> & losses) {
            float sum = 0.0f;
            for(float i : losses) {
                sum += i;
            }
            return sum / losses.size();
        }
};


class LossMeanSquaredError {
public:
    float output;
    Matrix<float> gradient;

    LossMeanSquaredError() {}

    void forward(Matrix<float>& input, vector<int>& targets) {
        output = 0.0f;
        float sampleLoss = 0.0f;
        float squaredDifference = 0.0f;

        for (size_t sample = 0; sample < input.rows(); sample++) {
            for (size_t value = 0; value < input[sample].size(); value++) {
                if (targets[sample] == value) {
                    squaredDifference = pow(input[sample][value] - 1.0f, 2);
                }
                else {
                    squaredDifference = pow(input[sample][value] - 0.0f, 2);
                }
                
                sampleLoss += squaredDifference;
            }
            output += sampleLoss / input[sample].size();
        }

        output /= input.rows();
    }


    void backward(Matrix<float>& pLOutput, vector<int>& targets) {
       
    }
};