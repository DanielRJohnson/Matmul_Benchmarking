#include <vector>
#include <random>
#include "matrix.h"

Matrix generateMatrix(int N, bool fillRandomly) {
	/*
	* Generates and NxN matrix and optionally fills with random values
	*/
	Matrix matrix(N, std::vector<float>(N));

	if (fillRandomly) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dist(0.0f, 10000.0f);
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				matrix[j][i] = dist(gen);
			}
		}
	}

	return matrix;
}

bool matrixEquality(const Matrix& A, const Matrix& B) {
	/*
	* Checks the equality of two matrices, which is defined as follows:
	*	A must have the same size as B AND all A values must differ from B values by less than 1e-6
	*/
	if (A.size() != B.size() || A[0].size() != B[0].size()) {
		return false;
	}
	for (size_t i = 0; i < A.size(); i++) {
		for (size_t j = 0; j < A[i].size(); j++) {
			if (std::abs(A[j][i] - B[j][i]) > 1e-6) {
				return false;
			}
		}
	}
	return true;
}
