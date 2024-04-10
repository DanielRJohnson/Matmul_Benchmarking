#pragma once

#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include "matrix.h"

void vanillaMatmul(const Matrix& A, const Matrix& B, Matrix& C, int placeholder);
void parforMatmul(const Matrix& A, const Matrix& B, Matrix& C, int placeholder);
void tiledMatmul(const Matrix& A, const Matrix& B, Matrix& C, int tileSize);
void tempDACMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff);
void _tempDACMatmulInner(const Matrix& A, const Matrix& B, Matrix& C, int size, int sizeCutoff);
void noTempDACMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff);
void strassensMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff);

#endif