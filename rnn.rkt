;;Typed tensors
#lang typed/racket
(require/typed
  ffi/unsafe
  [#:opaque CPointer cpointer?]
  [#:opaque CType ctype?]
  [_double CType]
  [_float CType]
  [_uintptr CType]
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
 [_cudnnHandle_t CType]
 [_cudnnTensorDescriptor_t CType]
 [cudaMalloc (CPointer Exact-Nonnegative-Integer -> CType)]
 [cudnnSetTensorNdDescriptor (CPointer Exact-Nonnegative-Integer
				       Exact-Nonnegative-Integer
				       CPointer
				       CPointer
				       -> CType)]
 [cudnnCreateTensorDescriptor (CPointer -> CType)]
 [cudnnCreateDropoutDescriptor (CPointer -> CType)]
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

;; Utility for getting a referenced value from an array
(: ptr-array-set! (-> CPointer Exact-Nonnegative-Integer Flonum Void))
(define (ptr-array-set! arr-ptr idx val)
  (ptr-set! arr-ptr _double idx val))

;; define Tensor struct
;; Provide an array, then store the FlVector of it and pointer
;; Create a gpu vector of size for the representation
;; We then memcpy between this vector and gpu vector to return
;; values.
;; desc stores the tensordescriptor

(struct tensor ([in-vect : FlVector]
                [src-ptr : CPointer]
                [gpu-vect : CPointer]
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

(: FLOAT-SIZE Exact-Nonnegative-Integer)
(define FLOAT-SIZE (ctype-sizeof _float))

;; Utility for getting a referenced value from an array
(: allocate-layer-mem (-> CPointer Exact-Nonnegative-Integer CType))
(define (allocate-layer-mem ptr size)
  (cudaMalloc ptr size))

;; Utility for getting a referenced value from an array
(: allocate-layer-array (-> CPointer Exact-Nonnegative-Integer Exact-Nonnegative-Integer Exact-Nonnegative-Integer CType))
(define (allocate-layer-array ptr seq-length input-size mini-batch)
  (allocate-layer-mem ptr (* seq-length input-size mini-batch FLOAT-SIZE)))

;; Allocate array of tensor descriptors
(: allocate-tensor-descriptors (-> Exact-Nonnegative-Integer CVector))
(define (allocate-tensor-descriptors seq-len)
  (cvector _cudnnTensorDescriptor_t seq-len))


(provide tensor rnn ptr-array-ref array->cptr)
