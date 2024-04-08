#include "algorithms.h"

void vanillaMatmul(const Matrix& A, const Matrix& B, Matrix& C, int placeholder) {
	/*
	* Performs a standard matrix multiplication. 
	* The placeholder argument is to have the same type signature as other algorithms.
	*/
	for (int i = 0; i < C.size(); i++) {
		for (int j = 0; j < C.size(); j++) {
			for (int k = 0; k < C.size(); k++) {
				C[j][i] = A[k][i] * B[j][k];
			}
		}
	}
}

void parforMatmul(const Matrix& A, const Matrix& B, Matrix& C, int placeholder) {
	/*
	* Performs a standard matrix multiplication with the outer two loops parallelized.
	* The placeholder argument is to have the same type signature as other algorithms.
	*/
	#pragma omp parallel for
	for (int i = 0; i < C.size(); i++) {
		#pragma omp parallel for
		for (int j = 0; j < C.size(); j++) {
			for (int k = 0; k < C.size(); k++) {
				C[j][i] = A[k][i] * B[j][k];
			}
		}
	}
}

void tiledMatmul(const Matrix& A, const Matrix& B, Matrix& C, int tileSize) {
	// TODO
}

void tempDACMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff) {
	// TODO
}

void noTempDACMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff) {
	// TODO
}

void strassensMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff) {
	//TODO
}