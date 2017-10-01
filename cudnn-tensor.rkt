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
 (require racket/flonum)
;; define Tensor struct
;; Provide an array, then store the FlVector of it and pointer
;; Create a gpu vector of size for the representation
;; We then memcpy between this vector and gpu vector to return
;; values.
;; desc stores the tensordescriptor

(struct cudnn-tensor ([in-array : (Array Flonum)]
		      [in-vect : FlVector]
		      [src-ptr : CPointer]
		      [gpu-ptr : CPointer]
		      [size : Exact-Nonnegative-Integer]
		      [desc : CPointer]))

;; Tensor
;; [ [ [1 2 3] [3 4 5] [3 4 5] [3 4 5] ] [ [5 6 7] [5 6 7] [5 6 7] [7 8 9]]]
;; batch size is 2, sequence length is 4
;; input size is 3


;; For input layers
(: make-cudnn-input-tensor (-> (Array Flonum) cudnn-tensor))
(define (make-cudnn-input-tensor arr)
  (match-let* ([ptr (get-pointer)]
	       [(vector batch-size input-size sequence-size) (array-shape arr)]
	       [size (* batch-size input-size sequence-size DOUBLE-SIZE)]
	       ;;cudaMalloc expects the address of the pointer
	       [resAlloc  (cudaMalloc ptr size)]
	       )
    (display (format "Result of gpu allocation request is ~a" resAlloc))
    ;(initGPUData ptr (* batch-size input-size sequence-size ) 1.0)
    (let ([tens
	   (cudnn-tensor arr
			 (flarray-data (array->flarray arr))
			 (array->cptr arr)
			 (dref-ptr ptr)
			 size
			 (init-simple-tensor-descriptor batch-size input-size sequence-size))])
      (print "Initialized a tensor")
      tens)))


;; Creates the tensor descriptor
(: init-input-tensor-descriptor (-> Nonnegative-Integer
				    Nonnegative-Integer
				    Nonnegative-Integer
				    CPointer))
(define (init-input-tensor-descriptor batch-size input-size sequence-size)
  (let
      (
       ;; allocate a block of tensor descriptors
       [tensor-descriptor-array (cuda-create-tensor-descriptr-ptr sequence-size)]
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


;; Allocate a tensor descriptor
(: init-simple-tensor-descriptor (-> Nonnegative-Integer
				    Nonnegative-Integer
				    Nonnegative-Integer CPointer))
(define (init-simple-tensor-descriptor layers batch-size hidden-size)
  (let
      (
       [tensor-descriptor-array (cuda-create-tensor-descriptr-ptr 1)]
       [dimA (get-int-array-block (list 3))]
       [strideA (get-int-array-block (list 3))]
       )
    (print "Attempting copy")
    (copy-int-to-block dimA (list layers batch-size hidden-size))
    (copy-int-to-block strideA (list (* batch-size hidden-size) hidden-size 1))
    (print "Finished copy, attempting descriptor creation")
    (cudnnCreateTensorDescriptor tensor-descriptor-array)
    (print "Setting the descriptor")
    (let ([tensor-desc (dref-tensor-desc-ptr tensor-descriptor-array)])
      (cudnnSetTensorNdDescriptor tensor-desc 0 3
				  dimA strideA)
      (print "Completed setting the descriptor")
      tensor-desc)))


;; Allocate an array of tensor descriptors for each training iteration
(: init-layer-tensor-descriptor (-> Nonnegative-Integer
				    Nonnegative-Integer
				    Nonnegative-Integer CPointer))
(define (init-layer-tensor-descriptor layers batch-size hidden-size)
  (let
      (
       [tensor-descriptor-array (cuda-create-tensor-descriptr-ptr 1)]
       [dimA (get-int-array-block (list 3))]
       [strideA (get-int-array-block (list 3))]
       )
    (copy-int-to-block dimA (list layers batch-size hidden-size))
    (copy-int-to-block strideA (list (* batch-size hidden-size) hidden-size 1))
    (cudnnCreateTensorDescriptor tensor-descriptor-array)
    (cudnnSetTensorNdDescriptor (dref-tensor-desc-ptr tensor-descriptor-array) 0 3
				dimA strideA)
    (dref-tensor-desc-ptr tensor-descriptor-array)))

;; allocate the gpu memory
(: tensor-allocate-gpu (cudnn-tensor -> cudnn-tensor))
(define (tensor-allocate-gpu tensor)
  tensor)

;; copy cpu memory to gpu
(: tensor-cpu->gpu (cudnn-tensor -> Symbol))
(define (tensor-cpu->gpu tensor)
  (display (format "Starting copy to GPU with:\n" ))
  (print-double-block (cudnn-tensor-src-ptr tensor) (flvector-length (cudnn-tensor-in-vect tensor)))
  (cudaDeviceSynchronize)
  (cuda-host-to-device-copy (cudnn-tensor-gpu-ptr tensor) (cudnn-tensor-src-ptr tensor)  16)
  ;;(cudnn-tensor-size tensor)

  )

;; copy gpu memory to cpu
(: tensor-gpu->cpu (cudnn-tensor -> Symbol))
(define (tensor-gpu->cpu tensor)
  (display (format "Starting copy from GPU\n"))
  (cuda-device-to-host-copy 
   (cudnn-tensor-src-ptr tensor)
   (cudnn-tensor-gpu-ptr tensor)
   (cudnn-tensor-size tensor)))

;; run the cudnnAddTensor function
(: tensor-add-tensors (CPointer cudnn-tensor cudnn-tensor -> Symbol))
(define (tensor-add-tensors handle tensor-a tensor-b)
  (let*
      ([alpha (get-double-array-block (list 1))]
       [beta (get-double-array-block (list 1))])
    (copy-double-to-block alpha (list 1.0))
    (display (format "Alpha is ~a\n" (print-double-block alpha 1)))
    (copy-double-to-block beta (list 1.0))
    (display (format "Beta is ~a\n" (print-double-block beta 1)))
    (cudnnAddTensor handle
		    alpha
		    (cudnn-tensor-desc tensor-a)
		    (cudnn-tensor-gpu-ptr tensor-a)
		    beta
		    (cudnn-tensor-desc tensor-b)
		    (cudnn-tensor-gpu-ptr tensor-b))))

;; print the values in a tensor
(define (tensor-print-values [tensor : cudnn-tensor])
  (print-double-block (cudnn-tensor-src-ptr tensor) (flvector-length (cudnn-tensor-in-vect tensor)))
  tensor)

(provide
 tensor-print-values
 tensor-add-tensors
 tensor-gpu->cpu
 tensor-cpu->gpu
 tensor-allocate-gpu
 init-layer-tensor-descriptor
 init-input-tensor-descriptor
 cudnn-tensor
 make-cudnn-input-tensor
 init-simple-tensor-descriptor
 cudnn-tensor-src-ptr
 cudnn-tensor-size
 )
