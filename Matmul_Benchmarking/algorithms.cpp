#include "algorithms.h"
#include <iostream>
#include <omp.h>

void vanillaMatmul(const Matrix& A, const Matrix& B, Matrix& C, int placeholder) {
	/*
	* Performs a standard matrix multiplication. 
	* The placeholder argument is to have the same type signature as other algorithms.
	*/
	std::size_t i, j;
	row_iter(A, i) {
		col_iter(B, j) {
			for ( auto k = 0; k < B.rowSize(); k++ ) {
				C[i][j] += A[i][k+A.colLowerBound()]*B[k+B.rowLowerBound()][j];
			}
		}
	}
}

void parforMatmul(const Matrix& A, const Matrix& B, Matrix& C, int placeholder) {
	/*
	* Performs a standard matrix multiplication with the outer two loops parallelized.
	* The placeholder argument is to have the same type signature as other algorithms.
	*/
	std::size_t i, j;
	#pragma omp parallel for
	row_iter(A, i) {
		#pragma omp parallel for
		col_iter(B, j) {
			for ( auto k = 0; k < B.rowSize(); k++ ) {
				C[i][j] += A[i][k+A.colLowerBound()]*B[k+B.rowLowerBound()][j];
			}
		}
	}
}

void tiledMatmul(const Matrix& A, const Matrix& B, Matrix& C, int tileSize) {
	
}

void tempDACMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff) {
	_tempDACMatmulInner(A, B, C, C.innerSize(), sizeCutoff);
}

void _tempDACMatmulInner(const Matrix& A, const Matrix& B, Matrix& C, int size, int sizeCutoff) {
	if ( sizeCutoff == size ) {
		vanillaMatmul(A, B, C, 0);
	} else {
		auto T = generateMatrix(C.innerSize(), false);

		std::size_t rowPivotA = A.rowLowerBound() + (A.rowSize()/2);
		std::size_t colPivotA = A.colLowerBound() + (A.colSize()/2);
		std::size_t rowPivotB = B.rowLowerBound() + (B.rowSize()/2);
		std::size_t colPivotB = B.colLowerBound() + (B.colSize()/2);
		std::size_t rowPivotC = C.rowLowerBound() + (C.rowSize()/2);
		std::size_t colPivotC = C.colLowerBound() + (C.colSize()/2);

		// submatrices
		auto a11 = A.subMatrix(A.rowLowerBound(), rowPivotA, A.colLowerBound(), colPivotA);
		auto a12 = A.subMatrix(A.rowLowerBound(), rowPivotA, colPivotA, A.colUpperBound());
		auto a21 = A.subMatrix(rowPivotA, A.rowUpperBound(), A.colLowerBound(), colPivotA);
		auto a22 = A.subMatrix(rowPivotA, A.rowUpperBound(), colPivotA, A.colUpperBound());
		auto b11 = B.subMatrix(B.rowLowerBound(), rowPivotB, B.colLowerBound(), colPivotB);
		auto b12 = B.subMatrix(B.rowLowerBound(), rowPivotB, colPivotB, B.colUpperBound());
		auto b21 = B.subMatrix(rowPivotB, B.rowUpperBound(), B.colLowerBound(), colPivotB);
		auto b22 = B.subMatrix(rowPivotB, B.rowUpperBound(), colPivotB, B.colUpperBound());
		auto c11 = C.subMatrix(C.rowLowerBound(), rowPivotC, C.colLowerBound(), colPivotC);
		auto c12 = C.subMatrix(C.rowLowerBound(), rowPivotC, colPivotC, C.colUpperBound());
		auto c21 = C.subMatrix(rowPivotC, C.rowUpperBound(), C.colLowerBound(), colPivotC);
		auto c22 = C.subMatrix(rowPivotC, C.rowUpperBound(), colPivotC, C.colUpperBound());
		auto t11 = T.subMatrix(C.rowLowerBound(), rowPivotC, C.colLowerBound(), colPivotC);
		auto t12 = T.subMatrix(C.rowLowerBound(), rowPivotC, colPivotC, C.colUpperBound());
		auto t21 = T.subMatrix(rowPivotC, C.rowUpperBound(), C.colLowerBound(), colPivotC);
		auto t22 = T.subMatrix(rowPivotC, C.rowUpperBound(), colPivotC, C.colUpperBound());

		// recurse
		#pragma omp parallel sections
		{
			#pragma omp section
			{ 
				_tempDACMatmulInner(a11, b11, c11, size/2, sizeCutoff);
			}
			#pragma omp section
			{ 
				_tempDACMatmulInner(a11, b12, c12, size/2, sizeCutoff);
			}
			#pragma omp section
			{ 
				_tempDACMatmulInner(a21, b11, c21, size/2, sizeCutoff);
			}
			#pragma omp section
			{ 
				_tempDACMatmulInner(a21, b12, c22, size/2, sizeCutoff);
			}
			#pragma omp section
			{ 
				_tempDACMatmulInner(a12, b21, t11, size/2, sizeCutoff);
			}
			#pragma omp section
			{ 
				_tempDACMatmulInner(a12, b22, t12, size/2, sizeCutoff);
			}
			#pragma omp section
			{ 
				_tempDACMatmulInner(a22, b21, t21, size/2, sizeCutoff);
			}
			#pragma omp section
			{ 
				_tempDACMatmulInner(a22, b22, t22, size/2, sizeCutoff);
			}
		}

		// reduce
		std::size_t i, j;
		#pragma omp parallel for
		row_iter(C, i) {
			#pragma omp parallel for
			col_iter(C, j) {
				C[i][j] += T[i][j];
			}
		}
	}
}

void noTempDACMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff) {
	
}

void strassensMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff) {
	//TODO
}