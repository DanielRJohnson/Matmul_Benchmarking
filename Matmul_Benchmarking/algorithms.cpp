#include "algorithms.h"
#include <iostream>
#include <omp.h>

void vanillaMatmul(const Matrix& A, const Matrix& B, Matrix& C, int placeholder) {
	/*
	* Performs a standard matrix multiplication. 
	* The placeholder argument is to have the same type signature as other algorithms.
	*/
	for ( auto i = 0; i < A.rowSize(); i++ ) {
		for ( auto j = 0; j < B.colSize(); j++ ) {
			for ( auto k = 0; k < B.rowSize(); k++ ) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

void _vanillaAdjusted(const Matrix& A, const Matrix& B, Matrix& C) {
	/*
	* Performs a standard matrix multiplication. 
	* tiled, DAC, and Strassens can have C appear in different offsets, which this version accounts for
	*/
	int i, j;
	row_iter(A, i) {
		col_iter(B, j) {
			for ( auto k = 0; k < B.rowSize(); k++ ) {
				C[i-A.rowLowerBound()+C.rowLowerBound()][j-B.colLowerBound()+C.colLowerBound()] 
					+= A[i][k+A.colLowerBound()]*B[k+B.rowLowerBound()][j];
			}
		}
	}
}

void parforMatmul(const Matrix& A, const Matrix& B, Matrix& C, int placeholder) {
	/*
	* Performs a standard matrix multiplication with the outer two loops parallelized.
	* The placeholder argument is to have the same type signature as other algorithms.
	*/
	int i, j;
	#pragma omp parallel for
	for ( i = 0; i < A.rowSize(); i++ ) {
		#pragma omp parallel for
		for ( j = 0; j < B.colSize(); j++ ) {
			for ( auto k = 0; k < B.rowSize(); k++ ) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

void tiledMatmul(const Matrix& A, const Matrix& B, Matrix& C, int tileSize) {
	int i, j;
	#pragma omp parallel for
	for ( i = A.rowLowerBound() ; i < A.rowUpperBound() ; i += tileSize ) {
		#pragma omp parallel for
		for ( j = B.colLowerBound() ; j < B.colUpperBound() ; j += tileSize ) {
			for ( int k = 0; k < C.rowSize(); k += tileSize ) {
				auto acol = A.colLowerBound() + k;
				auto subA = A.subMatrix(i, i+tileSize, acol, acol+tileSize);

				auto brow = B.rowLowerBound() + k;
				auto subB = B.subMatrix(brow, brow+tileSize, j, j+tileSize);

				auto subC = C.subMatrix(i, i+tileSize, j, j+tileSize);
				_vanillaAdjusted(subA, subB, subC);
			}
		}
	}
}

void tempDACMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff) {
	if ( sizeCutoff >= C.rowSize() ) {
		_vanillaAdjusted(A, B, C);
	} else {
		auto T = generateMatrix(C.rowSize(), false);

		std::size_t rowPivotA = A.rowLowerBound() + (A.rowSize()/2);
		std::size_t colPivotA = A.colLowerBound() + (A.colSize()/2);
		std::size_t rowPivotB = B.rowLowerBound() + (B.rowSize()/2);
		std::size_t colPivotB = B.colLowerBound() + (B.colSize()/2);
		std::size_t rowPivotC = C.rowLowerBound() + (C.rowSize()/2);
		std::size_t colPivotC = C.colLowerBound() + (C.colSize()/2);
		std::size_t rowPivotT = C.rowSize()/2;
		std::size_t colPivotT = C.colSize()/2;

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
		auto t11 = T.subMatrix(0, rowPivotT, 0, colPivotT);
		auto t12 = T.subMatrix(0, rowPivotT, colPivotT, C.colSize());
		auto t21 = T.subMatrix(rowPivotT, C.rowSize(), 0, colPivotT);
		auto t22 = T.subMatrix(rowPivotT, C.rowSize(), colPivotT, C.colSize());

		// recurse
		#pragma omp parallel
		{
			#pragma omp single nowait 
			{
				#pragma omp task
				{
					tempDACMatmul(a11, b11, c11, sizeCutoff);
				}
				#pragma omp task
				{
					tempDACMatmul(a11, b12, c12, sizeCutoff);
				}
				#pragma omp task
				{
					tempDACMatmul(a21, b11, c21, sizeCutoff);
				}
				#pragma omp task
				{
					tempDACMatmul(a21, b12, c22, sizeCutoff);
				}
				#pragma omp task
				{
					tempDACMatmul(a12, b21, t11, sizeCutoff);
				}
				#pragma omp task
				{
					tempDACMatmul(a12, b22, t12, sizeCutoff);
				}
				#pragma omp task
				{
					tempDACMatmul(a22, b21, t21, sizeCutoff);
				}
				#pragma omp task
				{
					tempDACMatmul(a22, b22, t22, sizeCutoff);
				}
				#pragma omp taskwait
			}
		}

		// reduce
		int i, j;
		#pragma omp parallel for
		row_iter(C, i) {
			#pragma omp parallel for
			col_iter(C, j) {
				C[i][j] += T[i-C.rowLowerBound()][j-C.colLowerBound()];
			}
		}
	}
}

void noTempDACMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff) {
	if ( sizeCutoff >= C.rowSize() ) {
		_vanillaAdjusted(A, B, C);
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

		#pragma omp parallel
		{
			#pragma omp single nowait
			{
				#pragma omp task
				{
					noTempDACMatmul(a11, b11, c11, sizeCutoff);
				}
				#pragma omp task
				{
					noTempDACMatmul(a11, b12, c12, sizeCutoff);
				}
				#pragma omp task
				{
					noTempDACMatmul(a21, b11, c21, sizeCutoff);
				}
				#pragma omp task
				{
					noTempDACMatmul(a21, b12, c22, sizeCutoff);
				}
				#pragma omp taskwait
			}
		}

		#pragma omp parallel
		{
			#pragma omp single nowait
			{
				#pragma omp task
				{
					noTempDACMatmul(a12, b21, c11, sizeCutoff);
				}
				#pragma omp task
				{
					noTempDACMatmul(a12, b22, c12, sizeCutoff);
				}
				#pragma omp task
				{
					noTempDACMatmul(a22, b21, c21, sizeCutoff);
				}
				#pragma omp task
				{
					noTempDACMatmul(a22, b22, c22, sizeCutoff);
				}
				#pragma omp taskwait
			}
		}
	}
}

void matAdd(const Matrix& A, const Matrix& B, Matrix& C, bool neg) {
	int i, j;
	#pragma omp parallel for
	row_iter(A, i) {
		#pragma omp parallel for
		col_iter(A, j) {
			if ( neg ) {
				C[i-A.rowLowerBound()+C.rowLowerBound()][j-A.colLowerBound()+C.colLowerBound()] =
					A[i][j] - B[i-A.rowLowerBound()+B.rowLowerBound()][j-A.colLowerBound()+B.colLowerBound()];
			} else {
				C[i-A.rowLowerBound()+C.rowLowerBound()][j-A.colLowerBound()+C.colLowerBound()] =
					A[i][j] + B[i-A.rowLowerBound()+B.rowLowerBound()][j-A.colLowerBound()+B.colLowerBound()];
			}
		}
	}
}

void strassensMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff) {
	if ( C.rowSize() <= sizeCutoff ) {
		_vanillaAdjusted(A, B, C);
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

		// temp matrices
		auto s0 = generateMatrix(c11.rowSize(), false);
		auto s1 = generateMatrix(c11.rowSize(), false);
		auto s2 = generateMatrix(c11.rowSize(), false);
		auto s3 = generateMatrix(c11.rowSize(), false);
		auto s4 = generateMatrix(c11.rowSize(), false);
		auto s5 = generateMatrix(c11.rowSize(), false);
		auto s6 = generateMatrix(c11.rowSize(), false);
		auto s7 = generateMatrix(c11.rowSize(), false);
		auto s8 = generateMatrix(c11.rowSize(), false);
		auto s9 = generateMatrix(c11.rowSize(), false);
		auto p1 = generateMatrix(c11.rowSize(), false);
		auto p2 = generateMatrix(c11.rowSize(), false);
		auto p3 = generateMatrix(c11.rowSize(), false);
		auto p4 = generateMatrix(c11.rowSize(), false);
		auto p5 = generateMatrix(c11.rowSize(), false);

		#pragma omp parallel
		{
			#pragma omp single nowait
			{
				#pragma omp task
				{
					matAdd(a11, a22, s0, false);
				}
				#pragma omp task
				{
					matAdd(b11, b22, s1, false);
				}
				#pragma omp task
				{
					matAdd(a21, a22, s2, false);
				}
				#pragma omp task
				{
					matAdd(b12, b22, s3, true);
				}
				#pragma omp task
				{
					matAdd(b21, b11, s4, true);
				}
				#pragma omp task
				{
					matAdd(a11, a12, s5, false);
				}
				#pragma omp task
				{
					matAdd(a21, a11, s6, true);
				}
				#pragma omp task
				{
					matAdd(b11, b12, s7, false);
				}
				#pragma omp task
				{
					matAdd(a12, a22, s8, true);
				}
				#pragma omp task
				{
					matAdd(b21, b22, s9, false);
				}
				#pragma omp taskwait
			}
		}

		#pragma omp parallel
		{
			#pragma omp single nowait
			{
				#pragma omp task
				{
					strassensMatmul(s0, s1, p1, sizeCutoff);
				}
				#pragma omp task
				{
					strassensMatmul(s2, b11, p2, sizeCutoff);
				}
				#pragma omp task
				{
					strassensMatmul(a11, s3, p3, sizeCutoff);
				}
				#pragma omp task
				{
					strassensMatmul(a22, s4, p4, sizeCutoff);
				}
				#pragma omp task
				{
					strassensMatmul(s5, b22, p5, sizeCutoff);
				}
				#pragma omp task
				{
					strassensMatmul(s6, s7, c22, sizeCutoff);
				}
				#pragma omp task
				{
					strassensMatmul(s8, s9, c11, sizeCutoff);
				}
				#pragma omp taskwait
			}
		}

		#pragma omp parallel
		{
			#pragma omp single nowait
			{
				#pragma omp task
				{
					matAdd(c11, p1, c11, false);
					matAdd(c11, p4, c11, false);
					matAdd(c11, p5, c11, true);
				}
				#pragma omp task
				{
					matAdd(c12, p3, c12, false);
					matAdd(c12, p5, c12, false);
				}
				#pragma omp task
				{
					matAdd(c21, p4, c21, false);
					matAdd(c21, p2, c21, false);
				}
				#pragma omp task
				{
					matAdd(c22, p1, c22, false);
					matAdd(c22, p2, c22, true);
					matAdd(c22, p3, c22, false);
				}
				#pragma omp taskwait
			}
		}
	}
}