#include "algorithms.h"
#include <iostream>
#include <omp.h>

void vanillaMatmul(const Matrix& A, const Matrix& B, Matrix& C, int placeholder) {
	/*
	* Performs a standard matrix multiplication. 
	* The placeholder argument is to have the same type signature as other algorithms.
	*/
	std::size_t i, j, k;
	row_iter(A, i) {
		col_iter(B, j) {
			row_iter(B, k) {
				C[i][j] += A[i][k]*B[k][j];
			}
		}
	}
}

void parforMatmul(const Matrix& A, const Matrix& B, Matrix& C, int placeholder) {
	/*
	* Performs a standard matrix multiplication with the outer two loops parallelized.
	* The placeholder argument is to have the same type signature as other algorithms.
	*/
	std::size_t i, j, k;
	#pragma omp parallel for
	row_iter(A, i) {
		#pragma omp parallel for
		col_iter(B, j) {
			row_iter(B, k) {
				C[i][j] += A[i][k]*B[k][j];
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
		auto T = generateMatrix(C.rowSize(), false);

		std::size_t rowPivotA = A.rowLowerBound() + (A.rowSize()/2);
		std::size_t colPivotA = A.colLowerBound() + (A.colSize()/2);
		std::size_t rowPivotB = B.rowLowerBound() + (B.rowSize()/2);
		std::size_t colPivotB = B.colLowerBound() + (B.colSize()/2);
		std::size_t rowPivotC = C.rowLowerBound() + (C.rowSize()/2);
		std::size_t colPivotC = C.colLowerBound() + (C.colSize()/2);
		std::size_t rowPivotT = T.rowLowerBound() + (T.rowSize()/2);
		std::size_t colPivotT = T.colLowerBound() + (T.colSize()/2);

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
		auto t11 = T.subMatrix(T.rowLowerBound(), rowPivotT, T.colLowerBound(), colPivotT);
		auto t12 = T.subMatrix(T.rowLowerBound(), rowPivotT, colPivotT, T.colUpperBound());
		auto t21 = T.subMatrix(rowPivotT, T.rowUpperBound(), T.colLowerBound(), colPivotT);
		auto t22 = T.subMatrix(rowPivotT, T.rowUpperBound(), colPivotT, T.colUpperBound());

		// recurse
		//#pragma omp task
		_tempDACMatmulInner(a11, b11, c11, size/2, sizeCutoff);
		//#pragma omp task
		_tempDACMatmulInner(a11, b12, c12, size/2, sizeCutoff);
		//#pragma omp task
		_tempDACMatmulInner(a21, b11, c21, size/2, sizeCutoff);
		//#pragma omp task
		_tempDACMatmulInner(a21, b12, c22, size/2, sizeCutoff);
		//#pragma omp task
		_tempDACMatmulInner(a12, b21, t11, size/2, sizeCutoff);
		//#pragma omp task
		_tempDACMatmulInner(a12, b22, t12, size/2, sizeCutoff);
		//#pragma omp task
		_tempDACMatmulInner(a22, b21, t21, size/2, sizeCutoff);
		//#pragma omp task
		_tempDACMatmulInner(a22, b22, t22, size/2, sizeCutoff);

		c11.print(std::cout);
		t11.print(std::cout);

		#pragma omp taskwait
		// reduce
		std::size_t i, j;
		#pragma omp parallel for
		row_iter(C, i) {
			#pragma omp parallel for
			col_iter(C, j) {
				// `C` can be a window into a greater matrix (ergo, its indexed not starting at 0)
				// `T` is a matrix the same size as the window, but starting at index 0
				// ergo we do the substraction to adjust
				C[i][j] += T[i-C.rowLowerBound()][j-C.colLowerBound()];
			}
		}
	}
}

void noTempDACMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff) {
	
}

void strassensMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff) {
	//TODO
}