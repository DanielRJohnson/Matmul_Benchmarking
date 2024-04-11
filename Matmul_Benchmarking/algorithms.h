#pragma once

#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include "matrix.h"

void vanillaMatmul(const Matrix& A, const Matrix& B, Matrix& C, int placeholder);
void _vanillaAdjusted(const Matrix& A, const Matrix& B, Matrix& C);
void parforMatmul(const Matrix& A, const Matrix& B, Matrix& C, int placeholder);
void tiledMatmul(const Matrix& A, const Matrix& B, Matrix& C, int tileSize);
void tempDACMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff);
void noTempDACMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff);
void matAdd(const Matrix& A, const Matrix& B, Matrix& C, bool neg);
void strassensMatmul(const Matrix& A, const Matrix& B, Matrix& C, int sizeCutoff);

#endif