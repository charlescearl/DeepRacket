;; Provides interfaces to the nvidia cuda api
;;Typed tensors
#lang typed/racket

(require/typed
 "lib-cudann.rkt"
 [_cudnn-status_t CType]
 [_cudnnHandle_t CType]
 [_cudaEvent_t CType]
 [_cuda-memcpy-kind_t CType]
 [cuda-create-handle ( -> CPointer)]
 [initGPUData (CPointer Exact-Nonnegative-Integer Flonum -> Void)]
 [dref-handle (CPointer -> CPointer)]
 [dref-ptr (CPointer -> CPointer)]
 [cudaDeviceSynchronize ( -> Symbol)]
 [cudaEventCreate (CType -> CType)]
 [cudaEventRecord (CType -> CType)]
 [cudaEventSynchronize (CType -> CType)]
 [cudaMemcpy (CPointer CPointer Exact-Nonnegative-Integer CType -> Symbol)]
 [cudaFree (CType -> CType)]
 [cudaMalloc (CPointer Exact-Nonnegative-Integer -> Symbol)]
 [cudnnDestroy (CType -> CType)]
 [cuda-host-to-device-copy (CPointer CPointer Exact-Nonnegative-Integer -> Symbol)]
 [cuda-device-to-host-copy (CPointer CPointer Exact-Nonnegative-Integer -> Symbol)]
 [cuda-host-to-host-copy (CPointer CPointer Exact-Nonnegative-Integer -> Symbol)]
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
 cuda-create-handle
 cuda-host-to-device-copy
 cuda-device-to-host-copy
 cuda-host-to-host-copy
 dref-handle
 dref-ptr
 initGPUData
 )

