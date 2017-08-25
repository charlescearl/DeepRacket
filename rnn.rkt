;;Typed tensors
#lang typed/racket
(require/typed
  ffi/unsafe
  [#:opaque CPointer cpointer?]
  [#:opaque CType ctype?]
  [_double CType]
  [_uintptr CType]
  [flvector->cpointer (FlVector -> CPointer)]
  [ptr-ref (CPointer CType Exact-Nonnegative-Integer -> Any)]
  )
(require math/base)
(require math/array)
(require/typed
 "lib-cudann.rkt"
 [_cudnnHandle_t CType]
 [_cudnnTensorDescriptor_t CType]
 )

;; Return pointer for start of an array
;;(array->cptr (array #[#[1.0 2.24554] #[3.2323 4.001] #[5.454 5.86868]]))
(: array->cptr (-> (Array Float) CPointer))
(define (array->cptr arr)
  (flvector->cpointer (flarray-data (array->flarray arr))))


;; Utility for getting a referenced value from an array
(: ptr-array-ref (-> CPointer Exact-Nonnegative-Integer Flonum))
(define (ptr-array-ref arr-ptr idx)
  (cast (ptr-ref arr-ptr _double idx) Flonum))


;; define Tensor struct
(struct tensor ([in-vect : (Array Float)] [gpu-vect : CPointer]
                [desc : CType]))

;; Define the RNN
(struct rnn
  ([x : tensor]
   [y : tensor]
   [dx : tensor]
   [dy : tensor]
   [hx : tensor]
   [hy : tensor]
   [cy : tensor]
   [cx : tensor]
   [dhx : tensor]
   [dcx : tensor]
   [dhy : tensor]
   [dcy : tensor]
   ))

(provide tensor rnn ptr-array-ref array->cptr)
