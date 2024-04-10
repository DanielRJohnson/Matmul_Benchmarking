#include <vector>
#include <random>
#include "matrix.h"

Matrix::Matrix(float* m, std::size_t innerSize) : _rowLB(0), _rowUB(innerSize), 
	_colLB(0), _colUB(innerSize), _matrix(std::move(m)), _innerSize(innerSize), _del(true) {}

Matrix::~Matrix() {
	if ( _del ) delete _matrix;
}

Matrix Matrix::subMatrix(std::size_t rowLB, std::size_t rowUB, std::size_t colLB, std::size_t colUB) {
	return Matrix(_matrix, _innerSize, rowLB, rowUB, colLB, colUB);
}

std::size_t Matrix::innerSize() const noexcept {
	return _innerSize;
}

std::size_t Matrix::rowSize() const noexcept {
	return _rowUB - _rowLB;
}

std::size_t Matrix::colSize() const noexcept {
	return _colUB - _colLB;
}

std::size_t Matrix::rowLowerBound() const noexcept {
	return _rowLB;
}

std::size_t Matrix::rowUpperBound() const noexcept {
	return _rowUB;
}

std::size_t Matrix::colLowerBound() const noexcept {
	return _colLB;
}

std::size_t Matrix::colUpperBound() const noexcept {
	return _colUB;
}

float* Matrix::operator[](std::size_t index) {
	return &(_matrix[index*_innerSize]);
}

float* Matrix::operator[](std::size_t index) const {
	return &(_matrix[index*_innerSize]);
}

Matrix::Matrix(float* m, std::size_t innerSize, std::size_t rowLB, std::size_t rowUB, 
	std::size_t colLB, std::size_t colUB) : _rowLB(rowLB), _rowUB(rowUB), _colLB(colLB), 
	_colUB(colUB), _matrix(m), _innerSize(innerSize) {}

Matrix generateMatrix(int N, bool fillRandomly) {
	/*
	* Generates and NxN matrix and optionally fills with random values
	*/
	float* matrix = (float*)calloc(N*N, sizeof(float));

	if (fillRandomly) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dist(0.0f, 10000.0f);
		for (int i = 0; i < N*N; i++) {
			matrix[i] = dist(gen);
		}
	}

	return Matrix(matrix, N);
}

bool matrixEquality(const Matrix& A, const Matrix& B) {
	/*
	* Checks the equality of two matrices, which is defined as follows:
	*	A must have the same size as B AND all A values must differ from B values by less than 1e-6
	*/
	if (A.rowSize() != B.rowSize() || A.colSize() != B.colSize()) {
		return false;
	}

	std::size_t i, j;
	row_iter(A, i) {
		row_iter(B, j) {
			if (std::abs(A[i][j] - B[i][j]) > 1e-6) {
				return false;
			}
		}
	}
	return true;
}
