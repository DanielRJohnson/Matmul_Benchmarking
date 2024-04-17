#include <vector>
#include <random>
#include <string>
#include <cassert>
#include "matrix.h"

std::string pad_right(std::string text, std::size_t width, std::string padding) {
	auto pad_diff = width - text.size();
	if (pad_diff < 1) return text;
	auto temp = text;
	for (int i = 0; i < pad_diff / padding.size(); i++)
		temp += padding;
	return temp;
}

std::string pad_left(std::string text, std::size_t width, std::string padding) {
	auto pad_diff = width - text.size();
	if (pad_diff < 1) return text;
	auto temp = text;
	for (int i = 0; i < pad_diff / padding.size(); i++)
		temp = padding + temp;
	return temp;
}

std::string pad_center(std::string text, std::size_t width, std::string padding) {
	auto lw = (width-text.size())/2 + text.size();
	return pad_right(pad_left(text,lw,padding),width,padding);
}

Matrix::Matrix(std::shared_ptr<double[]> m, std::size_t innerSize) : _rowLB(0), _rowUB(innerSize), 
	_colLB(0), _colUB(innerSize), _matrix(std::move(m)), _innerSize(innerSize) {}

Matrix Matrix::subMatrix(std::size_t rowLB, std::size_t rowUB, std::size_t colLB, std::size_t colUB) {
#ifdef DEBUG
	// bounds make valid submatrix of current view
	assert( rowLB >= _rowLB && rowUB <= _rowUB && colLB >= _colLB && colUB <= _colUB );
	// bounds are valid (>=0 and <=innersize are implicit, because the outer most matrix view has those bounds)
	assert( rowUB > rowLB && colUB > colLB );
#endif
	return Matrix(_matrix, _innerSize, rowLB, rowUB, colLB, colUB);
}

const Matrix Matrix::subMatrix(std::size_t rowLB, std::size_t rowUB, std::size_t colLB, std::size_t colUB) const {
#ifdef DEBUG
	// bounds make valid submatrix of current view
	assert( rowLB >= _rowLB && rowUB <= _rowUB && colLB >= _colLB && colUB <= _colUB );
	// bounds are valid (>=0 and <=innersize are implicit, because the outer most matrix view has those bounds)
	assert( rowUB > rowLB && colUB > colLB );
#endif
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

double* Matrix::operator[](int index) {
	return &(_matrix[index*_innerSize]);
}

double* Matrix::operator[](int index) const {
	return &(_matrix[index*_innerSize]);
}

Matrix::Matrix(std::shared_ptr<double[]> m, std::size_t innerSize, std::size_t rowLB, std::size_t rowUB, 
	std::size_t colLB, std::size_t colUB) : _rowLB(rowLB), _rowUB(rowUB), _colLB(colLB), 
	_colUB(colUB), _matrix(m), _innerSize(innerSize) {}

void Matrix::print(std::ostream& out) const {
	std::vector<std::size_t> colwidths;
	std::vector<std::vector<std::string>> values(rowSize(), std::vector<std::string>(colSize()));
	int i, j;
	col_iter((*this), j) {
		std::size_t maxw = 0;
		row_iter((*this), i) {
			std::string vstr = std::to_string((*this)[i][j]);
			if ( vstr.size() > maxw ) maxw = vstr.size();
			values.at(i-rowLowerBound()).at(j-colLowerBound()) = vstr;
		}
		colwidths.push_back(maxw);
	}

	//top
	out << "+";
	for ( auto i = 0; i < colwidths.size(); i++ ) {
		for ( auto j = 0; j < colwidths.at(i); j++ ) out << "-";
		if ( i == colwidths.size() - 1 ) out << "+";
		else out << "+";
	}
	out << std::endl;
	//cols
	row_iter((*this), i) {
		out << "|";
		col_iter((*this), j) {
			out << pad_center(values.at(i-rowLowerBound()).at(j-colLowerBound()), colwidths.at(j-colLowerBound()), " ");
			out << "|";
		}
		out << std::endl;
	}

	//bottom
	out << "+";
	for ( auto i = 0; i < colwidths.size(); i++ ) {
		for ( auto j = 0; j < colwidths.at(i); j++ ) out << "-";
		if ( i == colwidths.size() - 1 ) out << "+";
		else out << "+";
	}
	out << std::endl;
}

Matrix generateMatrix(int N, bool fillRandomly) {
	/*
	* Generates and NxN matrix and optionally fills with random values
	*/
	std::shared_ptr<double[]> matrix(new double[N*N]);

	if (fillRandomly) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dist(0.0f, 10.0f);
		for (int i = 0; i < N*N; i++) {
			matrix[i] = dist(gen);
		}
	} else {
		for (int i = 0; i < N*N; i++) {
			matrix[i] = 0;
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
		col_iter(B, j) {
			if (std::abs(A[i][j] - B[i][j]) > 1e-6) {
				return false;
			}
		}
	}
	return true;
}
