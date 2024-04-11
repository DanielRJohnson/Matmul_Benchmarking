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
	std::size_t i, j, k;
	#pragma omp parallel for
	for ( i = A.rowLowerBound() ; i < A.rowUpperBound() ; i += tileSize ) {
		#pragma omp parallel for
		for ( j = B.colLowerBound() ; j < B.colUpperBound() ; j += tileSize ) {
			for ( k = 0; k < C.rowSize(); k += tileSize ) {
				auto subA = A.subMatrix(i, i+tileSize, A.colLowerBound()+k, A.colLowerBound()+k+tileSize);
				auto subB = B.subMatrix(B.rowLowerBound()+k, B.rowLowerBound()+k+tileSize, j, j+tileSize);
				auto subC = C.subMatrix(i, i+tileSize, j, j+tileSize);
				vanillaMatmul(subA, subB, subC, 0);
			}
		}
	}
}

void tempDACMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff) {
	if ( sizeCutoff == C.rowSize() ) {
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
				tempDACMatmul(a11, b11, c11, sizeCutoff);
			}
			#pragma omp section
			{ 
				tempDACMatmul(a11, b12, c12, sizeCutoff);
			}
			#pragma omp section
			{ 
				tempDACMatmul(a21, b11, c21, sizeCutoff);
			}
			#pragma omp section
			{ 
				tempDACMatmul(a21, b12, c22, sizeCutoff);
			}
			#pragma omp section
			{ 
				tempDACMatmul(a12, b21, t11, sizeCutoff);
			}
			#pragma omp section
			{ 
				tempDACMatmul(a12, b22, t12, sizeCutoff);
			}
			#pragma omp section
			{ 
				tempDACMatmul(a22, b21, t21, sizeCutoff);
			}
			#pragma omp section
			{ 
				tempDACMatmul(a22, b22, t22, sizeCutoff);
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
	if ( C.rowSize() == sizeCutoff ) {
		vanillaMatmul(A, B, C, 0);
	} else {
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

		#pragma omp parallel sections
		{
			#pragma omp section
			{ 
				tempDACMatmul(a11, b11, c11, sizeCutoff);
			}
			#pragma omp section
			{ 
				tempDACMatmul(a11, b12, c12, sizeCutoff);
			}
			#pragma omp section
			{ 
				tempDACMatmul(a21, b11, c21, sizeCutoff);
			}
			#pragma omp section
			{ 
				tempDACMatmul(a21, b12, c22, sizeCutoff);
			}
		}

		#pragma omp parallel sections
		{
			#pragma omp section
			{ 
				tempDACMatmul(a12, b21, c11, sizeCutoff);
			}
			#pragma omp section
			{ 
				tempDACMatmul(a12, b22, c12, sizeCutoff);
			}
			#pragma omp section
			{ 
				tempDACMatmul(a22, b21, c21, sizeCutoff);
			}
			#pragma omp section
			{ 
				tempDACMatmul(a22, b22, c22, sizeCutoff);
			}
		}
	}
}

void strassensMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff) {
	//TODO
}