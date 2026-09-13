# Simple AI Model

This is a simple neural network that is made to classify handwritten digits using the MNIST dataset.

---

## Features
- Data preprocessing, training, and evaluation.

---

## Getting Started

### Prerequisites
Ensure you have the following installed on your system:
- CMake version 3.10 or higher
- A C++ compiler that supports C++20
- MNIST .csv files: mnist_test.csv, mnist_train.csv

### Get yourself the train and test data
The project is tested on handwritten digists provided by MNIST. You can find all the needed files on [this](https://github.com/phoebetronic/mnist) GitHub page.

---

### Build Instructions
To build the project on Linux, follow the steps below or just run the `build.sh` executale.

```bash
#!/bin/sh
git clone https://github.com/LookTheLock/NeuralNetwork
cd NeuralNetwork/datasets
git clone https://github.com/phoebetronic/mnist
cd mnist
unzip '*.zip'
cd ../../
mkdir -p build/build-x86
cd build/build-x86
cmake ../../
make
```

1. Clone the repository:
```bash
git clone https://github.com/MythStyxx/NeuralNetwork
cd NeuralNetwork
```

2. Navigate to the `datasets` directory:
```bash
cd datasets
```

3. Clone the [repository](https://github.com/phoebetronic/mnist) with MNIST data for training and testing:
```bash
git clone https://github.com/phoebetronic/mnist
cd mnist
```

4. Unzip the two zipfiles and return to the project directory:
```bash
unzip '*.zip'
cd ../../
```

5. Create a build directory:
```bash
mkdir -p build/build-x86
cd build/build-x86
```

6. Configure the project with CMake:
```bash
cmake ../../
```

7. Compile the project:
```bash
make
```

---

## Usage

1. Run the executable at `NeuralNetwork/build/src`
```bash
./NeuralNetwork
```

2. The script will:
   - Load the MNIST dataset of handwritten digits.
   - Normalize the data for better training performance.
   - Train a neural network to classify the digits.
   - Display the model's accuracy on test data.

---

## How It Works
1. **Data Preprocessing**:
   - The MNIST dataset is loaded and normalized to scale the pixel values to the range [0, 1].

2. **Model Architecture**:
   - Input Layer: Flattens 28x28 pixel images into a 1D array.
   - Hidden Layer: Dense layer with 128 neurons and ReLU activation.
   - Output Layer: Dense layer with 10 neurons and softmax activation.

3. **Training**:
   - The model is trained for 5 epochs using the Adam optimizer and sparse categorical crossentropy loss.

4. **Evaluation**:
   - The model is tested on unseen data, and its accuracy is displayed.

---

## Example Output
After running the script, you might see:
```
64:	2.4343
64:	2.21992
64:	2.13805
...
...
...
```

---

## Future Enhancements
- Implementing more complex neural network architectures.
- Testing the model on different datasets (e.g., CIFAR-10, Fashion MNIST).
- Deploying the model as a web application using Flask or Streamlit.

---

## Contributing
Contributions are welcome! To contribute:
1. Fork this repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add a new feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## Acknowledgments
- MNIST Dataset
