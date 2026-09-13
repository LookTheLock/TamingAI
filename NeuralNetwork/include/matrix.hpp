#include <iostream>
#include <vector>
#include <numeric>
#include <exception>

#pragma once


using namespace std;


template <typename T>
class Matrix {

    public:

        Matrix() {}

        Matrix(int rows, int columns, T filler = T(0)) {

            vector<T> column;
            for (int i = 0; i < columns; i++)
                column.push_back(filler);
            
            for (int i = 0; i < rows; i++)
                myMatrix.push_back(column);
        }

        virtual
        ~Matrix() {}



        void print(ostream& os) const {  //prints out matrix in readable form

            os << "[" << endl;

            for (vector row : myMatrix) {
                os << "[";
                for (T value : row) {
                    os << value << " ";
                }
                os << "]" << endl;
            }

            os << "]" << endl;
        }


        int rows() { return int(myMatrix.size()); }

        int columns() { return int(myMatrix[0].size()); }

        int rows() const { return int(myMatrix.size()); }

        int columns() const { return int(myMatrix[0].size()); }


        size_t size() const {
            return myMatrix.size();
        }


        auto begin() { return myMatrix.begin(); }
        auto end() { return myMatrix.end(); }


        void addRow(vector<T> newRow) {
            myMatrix.push_back(newRow);
        }

// all operators
        vector<T>& operator[](size_t row) {
            if(row >= myMatrix.size()){throw overflow_error("index out of range");}
            return  myMatrix[row];
        }

        vector<T> const & operator[](size_t row) const {
            if(row >= myMatrix.size()){throw overflow_error("index out of range");}
            return  myMatrix[row];
        }


        Matrix<T> operator*(T const& other) const {
            Matrix<T> result = *this;
            for(vector<T>& row : result) {
                for(T& column : row) {
                    column *= other;
                }
            }
            return result;
        }

        vector<T> operator*(vector<T> const& other) {
            vector<T> result;
            for (vector<T> row : myMatrix) {
                result.push_back(inner_product(row.begin(), row.end(), other.begin(), 0.0));
            }
            return result;
        }


        Matrix<T> operator*(Matrix<T>& other) {
            Matrix<T> result(rows(), other.columns(), 0);
            T singleResult;

            for (size_t thisRow = 0; thisRow < rows(); thisRow++) {

                for (size_t otherColumn = 0; otherColumn < other.columns(); otherColumn++) {

                    singleResult = 0;
                    for (size_t crossIndex = 0; crossIndex < columns(); crossIndex++) {
                        singleResult += myMatrix[thisRow][crossIndex] * other[crossIndex][otherColumn];
                    }

                    result[thisRow][otherColumn] = singleResult;
                }
            }

            return result;
        }


        Matrix<T> operator+(vector<T> & other) {

            if (columns() != int(other.size())) {
                throw exception();
            }

            Matrix<T> result(rows(), columns(),0);
            

            for (size_t row = 0; row < rows(); row++) {
                for (int column = 0; column < columns(); column++) {
                    result[row][column] = myMatrix[row][column] + other[column]; 
                }
            }

            return result;
        }


        Matrix<T> & operator-=(Matrix<T> const & other) {

            for (size_t row = 0; row < (*this).size(); row++) {
                for (size_t column = 0; column < (*this)[row].size(); column++) {
                    (*this)[row][column] -= other[row][column];
                }
            }
            return *this;
        }

        //transposes the matrix
        void transpose() {
            Matrix<T> result(columns(), rows(), 0); //swapped dimensions

            for (int row = 0; row < rows(); row++) {
                for (int column = 0; column < columns(); column++) {
                    result[column][row] = myMatrix[row][column];
                }
            }

            myMatrix = move(result.myMatrix);
        }

        Matrix<T> returnT() const {
            Matrix<T> result(columns(), rows(), 0); //swapped dimensions

            for (size_t row = 0; row < rows(); row++) {
                for (size_t column = 0; column < columns(); column++) {
                    result[column][row] = myMatrix[row][column];
                }
            }

            return result;
        }


        //returns a vector with the sum of each row
        vector<T> sumRow() {
            vector<T> result;
            T sum;
            for (vector<T>& row : myMatrix) {
                sum = 0;
                for (T& value : row) {
                    sum += value;
                }
                result.push_back(sum);
            }
            return result;
        }


    private:
        vector<vector<T>> myMatrix;  //holds matrix using vector containing vector
};


//ostream operator
template <typename T>
ostream& operator<<(ostream& s, Matrix<T>& other) {
    other.print(s);

    return s;
}
