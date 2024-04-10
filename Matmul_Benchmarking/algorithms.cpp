#include "algorithms.h"

void vanillaMatmul(const Matrix& A, const Matrix& B, Matrix& C, int placeholder) {
	/*
	* Performs a standard matrix multiplication. 
	* The placeholder argument is to have the same type signature as other algorithms.
	*/
	std::size_t i, j;
	row_iter(C, i) {
		col_iter(C, j) {
			for (int k = 0; k < C.rowSize(); k++) {
				auto a = A[i][k];
				auto b = B[k][j];
				C[i][j] =  a*b ;
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
	row_iter(C, i) {
		#pragma omp parallel for
		col_iter(C, j) {
			for (int k = 0; k < C.rowSize(); k++) {
				C[i][j] = A[i][k] * B[k][j];
			}
		}
	}
}

void tiledMatmul(const Matrix& A, const Matrix& B, Matrix& C, int tileSize) {
	
}

void tempDACMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff) {
	if ( sizeCutoff == 1 ) {
		//C[0] = A[0]*B[0];
	} else {
		auto T = generateMatrix(sizeCutoff, false);

		//
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