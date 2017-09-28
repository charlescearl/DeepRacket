;;Typed tensors
#lang typed/racket
(require/typed
 "lib-cudann.rkt"
 ffi/unsafe
 math/base
 math/matrix)

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


;; Tensor
;; [ [ [1 2 3] [3 4 5] [3 4 5] [3 4 5] ] [ [5 6 7] [5 6 7] [5 6 7] [7 8 9]]]
;; batch size is 2, sequence length is 4
;; input size is 3


;; For input layers
(: make-input-tensor (-> Integer tensor))
(define (make-input-tensor array-size)
  (let ([ptr (init-ptr)])
    (cudaMalloc ptr size)
    (tensor ptr)))

;; For hidden layers of the network
(: make-layer-tensor (-> Integer tensor))
(define (make-layer-tensor array-size)
  (let ([ptr (init-ptr)])
    (cudaMalloc ptr size)
    (tensor ptr)))

