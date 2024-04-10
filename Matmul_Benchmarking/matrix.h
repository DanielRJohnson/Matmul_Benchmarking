#pragma once

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <ostream>

class Matrix final {
public:
    /**
     * Create a Matrix wrapper of size `innerSize*innerSize`
     */
    explicit Matrix(float* m, std::size_t innerSize);
    ~Matrix();
    
    /**
     * Creates a Matrix view of the internal array with different bounds
     * There is no error checking (because performance), so make sure the bounds
     * are actually within the range of the internal array
     */
    Matrix subMatrix(std::size_t rowLB, std::size_t rowUB, std::size_t colLB, std::size_t colUB);

    /**
     * Creates a Matrix view of the internal array with different bounds
     * There is no error checking (because performance), so make sure the bounds
     * are actually within the range of the internal array
     */
    const Matrix subMatrix(std::size_t rowLB, std::size_t rowUB, std::size_t colLB, std::size_t colUB) const;

    /** The dimension of the internal matrix */
    std::size_t innerSize() const noexcept;
    /** Size of a row */
    std::size_t rowSize() const noexcept;
    /** Size of a column */
    std::size_t colSize() const noexcept;
    /** Lower bound of a row (inclusive) */
    std::size_t rowLowerBound() const noexcept;
    /** Upper bound of a row (exclusive) */
    std::size_t rowUpperBound() const noexcept;
    /** Lower bound of a column (inclusive) */
    std::size_t colLowerBound() const noexcept;
    /** Upper bound of a column (exclusive) */
    std::size_t colUpperBound() const noexcept;
    /** Pointer to the beginning of row `index` (indexed based on the internal matrix, ignores bounds) */
    float* operator[](std::size_t index);
    /** Pointer to the beginning of row `index` (indexed based on the internal matrix, ignores bounds) */
    float* operator[](std::size_t index) const;

    /** output matrix */
    void print(std::ostream&) const;
protected:
    Matrix(float* m, std::size_t innerSize, 
        std::size_t rowLB, std::size_t rowUB, std::size_t colLB, std::size_t colUB);

    std::size_t _rowLB, _rowUB, _colLB, _colUB;
    /** The internal array */
    float* _matrix;
    std::size_t _innerSize;
    /**
     * I was too lazy to write a reference counter for the internal array.
     * `_matrix` is deleted if public constructor is used, but not if `subMatrix` is called
     */
    bool _del = false;
};

#define row_iter(window, iter) \
    for ( iter = window.rowLowerBound() ; iter < window.rowUpperBound() ; iter++ )
#define col_iter(window, iter) \
    for ( iter = window.colLowerBound() ; iter < window.colUpperBound() ; iter++ )

Matrix generateMatrix(int N, bool fillRandomly);
bool matrixEquality(const Matrix& A, const Matrix& B);

#endif