# Type definitions for word2vec_fast
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
#
# Copyright (C) 2015 Jean Maillard <jean@maillard.it>
#
# Licensed under the GNU GPL v3.0 - <http://www.gnu.org/licenses/gpl-3.0.html>

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

cimport numpy as np
ctypedef np.float32_t floatX_t

# BLAS routine signatures
ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil

cdef scopy_ptr scopy
cdef saxpy_ptr saxpy
cdef sdot_ptr sdot
cdef dsdot_ptr dsdot
cdef snrm2_ptr snrm2
cdef sscal_ptr sscal

# precalculated sigmoid table
DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6
cdef floatX_t[EXP_TABLE_SIZE] EXP_TABLE

# function implementations swapped based on BLAS detected in word2vec_inner.pyx init()
ctypedef floatX_t (*our_dot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef void (*our_saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil

cdef our_dot_ptr our_dot
cdef our_saxpy_ptr our_saxpy

# for when fblas.sdot returns a double
cdef floatX_t our_dot_double(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil

# for when fblas.sdot returns a float
cdef floatX_t our_dot_float(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil

# for when no blas available
cdef floatX_t our_dot_noblas(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
cdef void our_saxpy_noblas(const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil

# to support random draws from negative-sampling cum_table
cdef unsigned long long bisect_left(np.uint32_t *a, unsigned long long x, unsigned long long lo, unsigned long long hi) nogil

cdef unsigned long long random_int32(unsigned long long *next_random) nogil
