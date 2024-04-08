#pragma once

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

using Matrix = std::vector<std::vector<float>>;

Matrix generateMatrix(int N, bool fillRandomly);
bool matrixEquality(const Matrix& A, const Matrix& B);

#endif