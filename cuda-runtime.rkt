;;Typed tensors
#lang typed/racket
(require/typed
  ffi/unsafe
  [#:opaque CPointer cpointer?]
  [#:opaque CType ctype?]
  [_double CType]
  [_float CType]
  [_uintptr CType]
  [_pointer CType]
  [_int CType]
  [_size CType]
  [flvector->cpointer (FlVector -> CPointer)]
  [ptr-ref (CPointer CType Exact-Nonnegative-Integer -> Any)]
  [ptr-set! (CPointer CType Exact-Nonnegative-Integer Any -> Void)]
  [ctype-sizeof (CType -> Exact-Nonnegative-Integer)]
  
  )
(require/typed
 ffi/unsafe/cvector
 [#:opaque CVector cvector?]
 [cvector (CType Any -> CVector)])
(require math/base)
(require math/array)

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
 [cudnnDestroy (CType -> CType)]
 )

(provide
 cudaDeviceSynchronize
 cudaEventCreate
 cudaEventRecord
 cudaEventSynchronize
 cudaMemcpy
 cudaFree
 cudnnDestroy
 )
