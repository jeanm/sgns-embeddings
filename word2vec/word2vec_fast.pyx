#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
#
# Copyright (C) 2015 Jean Maillard <jean@maillard.it>
#
# Licensed under the GNU GPL v3.0 - <http://www.gnu.org/licenses/gpl-3.0.html>

import cython
import numpy as np
cimport numpy as np

floatX = np.float32

from libc.math cimport exp,sqrt,log
from libc.string cimport memset
from libc.stdlib cimport calloc,malloc,free

# scipy <= 0.15
try:
    from scipy.linalg.blas import fblas
except ImportError:
    # in scipy > 0.15, fblas function has been removed
    import scipy.linalg.blas as fblas

DEF MAX_SENTENCE_LEN = 10000
DEF MAX_2WINDOW_LEN = 20

cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef floatX_t[EXP_TABLE_SIZE] EXP_TABLE
cdef floatX_t[EXP_TABLE_SIZE] LOG_TABLE

cdef int ONE = 1
cdef floatX_t ONEF = <floatX_t>1.0

# for when fblas.sdot returns a double
cdef floatX_t our_dot_double(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <floatX_t>dsdot(N, X, incX, Y, incY)

# for when fblas.sdot returns a float
cdef floatX_t our_dot_float(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <floatX_t>sdot(N, X, incX, Y, incY)

# for when no blas available
cdef floatX_t our_dot_noblas(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    # not a true full dot()-implementation: just enough for our cases
    cdef int i
    cdef floatX_t a
    a = <floatX_t>0.0
    for i from 0 <= i < N[0] by 1:
        a += X[i] * Y[i]
    return a

# for when no blas available
cdef void our_saxpy_noblas(const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil:
    cdef int i
    for i from 0 <= i < N[0] by 1:
        Y[i * (incY[0])] = (alpha[0]) * X[i * (incX[0])] + Y[i * (incY[0])]


# to support random draws from negative-sampling ns_table
cdef inline unsigned long long bisect_left(np.uint32_t *a, unsigned long long x, unsigned long long lo, unsigned long long hi) nogil:
    cdef unsigned long long mid
    while hi > lo:
        mid = (lo + hi) >> 1
        if a[mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return lo

# this quick & dirty RNG apparently matches Java's (non-Secure)Random
# note this function side-effects next_random to set up the next number
cdef inline unsigned long long random_int32(unsigned long long *next_random) nogil:
    cdef unsigned long long this_random = next_random[0] >> 16
    next_random[0] = (next_random[0] * <unsigned long long>25214903917ULL + 11) & 281474976710655ULL
    return this_random

cdef unsigned long long fast_sentence_sg_neg(
    const int negative, np.uint32_t *ns_table, unsigned long long ns_table_len,
    floatX_t *syn0, floatX_t *syn1, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const floatX_t alpha, floatX_t *work,
    unsigned long long next_random) nogil:

    cdef long long a
    cdef long long row1 = <long long>word2_index * size
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef floatX_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(floatX_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(ns_table, (next_random >> 16) % ns_table[ns_table_len-1], 0, ns_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <floatX_t>0.0

        row2 = <long long>target_index * size
        f = our_dot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1[row2], &ONE)

    our_saxpy(&size, &ONEF, work, &ONE, &syn0[row1], &ONE)

    return next_random

# the only difference between this function and `fast_sentence_sg_neg` is that
# this function does not update the weights `syn1` (aka the contexts)
cdef unsigned long long fast_sentence_sg_neg_const_syn1(
    const int negative, np.uint32_t *ns_table, unsigned long long ns_table_len,
    floatX_t *syn0, floatX_t *syn1, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const floatX_t alpha, floatX_t *work,
    unsigned long long next_random) nogil:

    cdef long long a
    cdef long long row1 = <long long>word2_index * size
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef floatX_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(floatX_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(ns_table, (next_random >> 16) % ns_table[ns_table_len-1], 0, ns_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <floatX_t>0.0

        row2 = <long long>target_index * size
        f = our_dot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        #our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1[row2], &ONE)

    our_saxpy(&size, &ONEF, work, &ONE, &syn0[row1], &ONE)

    return next_random

def train_sentence(model, sentence, alpha, _syn0, _work):
    cdef int negative = model.negative
    cdef int sample = (model.sample != 0)

    cdef floatX_t *syn0 = <floatX_t *>(np.PyArray_DATA(_syn0))
    cdef floatX_t *syn1 = <floatX_t *>(np.PyArray_DATA(model.contexts))
    cdef floatX_t *work
    cdef floatX_t _alpha = alpha
    cdef int size = model.dim

    cdef np.uint32_t indices[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    # for negative sampling
    cdef np.uint32_t *ns_table
    cdef unsigned long long ns_table_len
    ns_table = <np.uint32_t *>(np.PyArray_DATA(model.ns_table))
    ns_table_len = len(model.ns_table)

    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random
    next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <floatX_t *>np.PyArray_DATA(_work)

    sample_int = model.index2sample
    i = 0
    for word in sentence:
        if sample and sample_int[word] < random_int32(&next_random): # subsampling
            continue
        indices[i] = word
        i += 1
        if i == MAX_SENTENCE_LEN:
            break
    sentence_len = i

    # single randint() call avoids a big thread-sync slowdown
    for i, item in enumerate(np.random.randint(0, window, sentence_len)):
        reduced_windows[i] = item

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            j = i - window + reduced_windows[i]
            if j < 0:
                j = 0
            k = i + window + 1 - reduced_windows[i]
            if k > sentence_len:
                k = sentence_len
            for j in range(j, k):
                if j == i:
                    continue
                next_random = fast_sentence_sg_neg(negative, ns_table, ns_table_len, syn0, syn1, size, indices[i], indices[j], _alpha, work, next_random)

def train_single(model, example, alpha, _syn0, _work):
    cdef int negative = model.negative
    cdef int sample = (model.sample != 0)

    cdef floatX_t *syn0 = <floatX_t *>(np.PyArray_DATA(_syn0))
    cdef floatX_t *syn1 = <floatX_t *>(np.PyArray_DATA(model.contexts))
    cdef floatX_t *work
    cdef floatX_t _alpha = alpha
    cdef int size = model.dim

    cdef np.uint32_t compound = example[0]
    cdef np.uint32_t indices[MAX_2WINDOW_LEN]
    cdef np.uint32_t reduced_window
    cdef int i

    # for negative sampling
    cdef np.uint32_t *ns_table
    cdef unsigned long long ns_table_len
    ns_table = <np.uint32_t *>(np.PyArray_DATA(model.ns_table))
    ns_table_len = len(model.ns_table)

    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random
    next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <floatX_t *>np.PyArray_DATA(_work)

    # build reduced window and downsample
    sample_int = model.index2sample
    i = 0
    for word in example[1:]:
        if sample and sample_int[word] < random_int32(&next_random): # subsampling
            continue
        indices[i] = word
        i += 1
        if i == MAX_2WINDOW_LEN:
            break
    if i < 1:
        return

    reduced_window = np.random.randint(0, i)
    # release GIL & train on the sentence
    with nogil:
        for i in range(reduced_window):
            #next_random = fast_sentence_sg_neg(negative, ns_table, ns_table_len, syn0, syn1, size, indices[i], compound, _alpha, work, next_random)
            next_random = fast_sentence_sg_neg_const_syn1(negative, ns_table, ns_table_len, syn0, syn1, size, indices[i], compound, _alpha, work, next_random)

def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.  Also calculate log(sigmoid(x)) into LOG_TABLE.

    """
    global our_dot
    global our_saxpy

    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <floatX_t>exp((i / <floatX_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <floatX_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
        LOG_TABLE[i] = <floatX_t>log( EXP_TABLE[i] )

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if (abs(d_res - expected) < 0.0001):
        our_dot = our_dot_double
        our_saxpy = saxpy
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        our_dot = our_dot_float
        our_saxpy = saxpy
        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        our_dot = our_dot_noblas
        our_saxpy = our_saxpy_noblas
        return 2

FAST_VERSION = init()  # initialize the module
