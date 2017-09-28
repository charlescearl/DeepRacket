;; Tensors seem to be very domain specific
;; This package provides utilities for managing
;; cudnn-tensors
;;Typed tensors
#lang typed/racket
(require
 "ffi-functional.rkt"
 "cudnn-api.rkt"
 "cuda-api.rkt"
 "mem-utils.rkt"
 math/base
 math/array
 math/matrix)

;; define Tensor struct
;; Provide an array, then store the FlVector of it and pointer
;; Create a gpu vector of size for the representation
;; We then memcpy between this vector and gpu vector to return
;; values.
;; desc stores the tensordescriptor

(struct cudnn-tensor ([in-array : (Array Flonum)]
		      [in-vect : FlVector]
		      [src-ptr : CPointer]
		      [desc : CPointer]))

;; Tensor
;; [ [ [1 2 3] [3 4 5] [3 4 5] [3 4 5] ] [ [5 6 7] [5 6 7] [5 6 7] [7 8 9]]]
;; batch size is 2, sequence length is 4
;; input size is 3


;; For input layers
(: make-cudnn-input-tensor (-> (Array Flonum) cudnn-tensor))
(define (make-cudnn-input-tensor arr)
  (match-let ([ptr (get-pointer)]
	      [(vector batch-size input-size sequence-size) (array-shape arr)])
    (cudaMalloc ptr (* batch-size input-size sequence-size))
    (cudnn-tensor arr
	    (flarray-data (array->flarray arr))
	    ptr
	    (init-input-tensor-descriptor batch-size input-size sequence-size)
	    )))


;; Creates the tensor descriptor
(: init-input-tensor-descriptor (-> Nonnegative-Integer
				    Nonnegative-Integer
				    Nonnegative-Integer
				    CPointer))
(define (init-input-tensor-descriptor batch-size input-size sequence-size)
  (let
      (
       ;; allocate a block of tensor descriptors
       [tensor-descriptor-array (get-tensor-desc-array sequence-size)]
       [dimA (get-int-array-block (list 3))]
       [strideA (get-int-array-block (list 3))]
       )
    (copy-int-to-block dimA (list batch-size input-size 1))
    (copy-int-to-block strideA (list input-size 1 1))
    (for ([desc-idx : Nonnegative-Integer (in-range sequence-size)])
      (let
	  ([desc-ptr (get-tensor-desc-ptr tensor-descriptor-array desc-idx)])
	(cudnnCreateTensorDescriptor desc-ptr) ;;address of the tensor descriptor
	(cudnnSetTensorNdDescriptor
	 (dref-tensor-desc-ptr desc-ptr) ;; the tensor descriptor
	 0
	 3
	 dimA
	 strideA)))
    tensor-descriptor-array))




(: init-layer-tensor-descriptor (-> Nonnegative-Integer
				    Nonnegative-Integer
				    Nonnegative-Integer CPointer))
(define (init-layer-tensor-descriptor layers batch-size hidden-size)
  (let
      (
       [tensor-descriptor-array (get-tensor-desc-array 1)]
       [dimA (get-int-array-block (list 3))]
       [strideA (get-int-array-block (list 3))]
       )
    (copy-int-to-block dimA (list layers batch-size hidden-size))
    (copy-int-to-block strideA (list (* batch-size hidden-size) hidden-size 1))
    (cudnnCreateTensorDescriptor tensor-descriptor-array)
    (cudnnSetTensorNdDescriptor (dref-tensor-desc-ptr tensor-descriptor-array) 0 3
				dimA strideA)
    tensor-descriptor-array))
