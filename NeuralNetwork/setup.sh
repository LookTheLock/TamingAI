#!/bin/sh

### Build Instructions
### To build the project, follow these steps:

# 1. Clone the repository:
# git clone https://github.com/LookTheLock/NeuralNetwork
# cd NeuralNetwork

# 2. Navigate to the `datasets` directory:
cd datasets

# 3. Clone the [repository](https://github.com/phoebetronic/mnist) with MNIST data for training and testing:
git clone https://github.com/phoebetronic/mnist
cd mnist

# 4. Unzip the two zipfiles and return to the project directory:
unzip '*.zip'
cd ../../

# 5. Create a build directory:
mkdir -p build/build-x86
cd build/build-x86

# 6. Configure the project with CMake:
cmake ../../

# 7. Compile the project:
make

echo "
   ___   ____     __              __
  / _ | / / / ___/ /__  ___  ___ / /
 / __ |/ / / / _  / _ \\/ _ \\/ -_)_/
/_/ |_/_/_/  \\_,_/\\___/_//_/\\__(_)
"
