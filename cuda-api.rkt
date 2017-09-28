;; Provides interfaces to the nvidia cuda api
;;Typed tensors
#lang typed/racket

(require/typed
 "lib-cudann.rkt"
 [_cudnn-status_t CType]
 [_cudnnHandle_t CType]
 [_cudaEvent_t CType]
 [_cuda-memcpy-kind_t CType]
 [cudaDeviceSynchronize ( -> CType)]
 [cudaEventCreate (CType -> CType)]
 [cudaEventRecord (CType -> CType)]
 [cudaEventSynchronize (CType -> CType)]
 [cudaMemcpy (CType CType Exact-Nonnegative-Integer CType -> CType)]
 [cudaFree (CType -> CType)]
 [cudaMalloc (CPointer Exact-Nonnegative-Integer -> CType)]
 [cudnnDestroy (CType -> CType)]
 )

(require "ffi-functional.rkt")

;; Pointer size
(: POINTER-SIZE Exact-Nonnegative-Integer)
(define POINTER-SIZE 8)

;;Get a pointer
(: get-pointer (-> CPointer))
(define (get-pointer )
  (malloc 'atomic-interior POINTER-SIZE))

(provide
 cudaDeviceSynchronize
 cudaEventCreate
 cudaEventRecord
 cudaEventSynchronize
 cudaMemcpy
 cudaMalloc
 cudaFree
 cudnnDestroy
 get-pointer
 )

