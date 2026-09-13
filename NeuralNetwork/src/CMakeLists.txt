
# ausgeführtes Programm festlegen
add_executable(NeuralNetwork main.cpp)

# geht in den include Ordner um Header-Dateien zu finden
target_include_directories(NeuralNetwork PRIVATE ${PROJECT_SOURCE_DIR}/include)