#include "converter.hpp"
#include "layertypes.hpp"
#include "classifiernetwork.hpp"
#include "dataset.hpp"

using namespace std;


int main(int argc, char* argv[]) {
    cout << "Content-Type: text/plain\n\n" << endl;

   /* DataSet testSet("../../../datasets/mnist/mnist_test.csv", 64, 255);
    DataSet trainSet("../../../datasets/mnist/mnist_train.csv", 64, 255);*/

//gets bytes of image
    vector<unsigned char> in;
    if (argc == 1) {
        in = Converter::getIn();
    }
    else if (argc == 2) {
        in = Converter::getIn(argv[1]);
    }
    else {
        cerr << "Wrong arguments." << endl;
        return -1;
    }

    //amount of bytes
    //cout << "Amount Bytes: " << in.size() << endl;


//convert to gray values
    Matrix<float> image = Converter::decodeAndConvert(in);

    //cout << "Rows: " << image.rows() << ", Columns: " << image.columns() << endl;


//test size
    try {
        if (image.size() != 1 || image.columns() != (28 * 28)) throw logic_error("Not the right image size.");
    }
    catch (logic_error le) {
        cerr << le.what() << endl;
        return -1;
    }

    //for (size_t i = 0; i < 28 * 28; i++) {
        //cout << image[0][i]*255 << " ";
       // if ((i + 1) % 28 == 0) cout << "\n";
   // }
//load network
    //cout << "The error is here" << endl;
    ClassifierNetwork net("../saves/nn.txt");

    /*ClassifierNetwork net("../../../saves/bnn.txt");

    for (size_t i = 0; i < 0; i++) {
        cout << "-------" << i << endl;
        net.train(trainSet, 0.0005, 1);
        trainSet.resetSet();
    }

    net.save("../../../saves/bnn.txt");*/


    //net.testNet(testSet);

//predict digit
    vector<int> prediction = net.predict(image);

//return to stdout
    cout << prediction[0];
    //cout << "The error is there" << endl;

    return 0;
}
