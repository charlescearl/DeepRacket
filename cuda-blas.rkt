;;; Provides some low level interfaces to the
;;; NVIDIA CUDA BLAS api
#lang racket/base
(require
 racket/path
 ffi/unsafe
 ffi/unsafe/define)

(define cublas-path (build-path "/" "usr" "local" "cuda-8.0" "targets" "x86_64-linux" "lib" "libcublas"))

(define-cublas-definer define-cublas (ffi-lib cublas-path))

;;Copy a matrix to gpu memory


