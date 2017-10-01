#lang typed/racket
(require typed/rackunit
	 typed/rackunit/text-ui)
(require "cudnn-api.rkt")
(require "cuda-api.rkt")
(require "mem-utils.rkt")
(require "cudnn-tensor.rkt")
(require math/array)

(define add-tests
  (test-suite
   "Test addition of tensors"
   (test-case
    "Check addition of two tensors"
    (let*
	([cuda-handle (cuda-create-handle)]
	 [tensor-a (make-cudnn-input-tensor (array #[#[#[1.0 1.9] #[3.2 3.3]] #[#[1.8 3.3] #[2.1 2.2]]]))]
	 [tensor-b (make-cudnn-input-tensor (array #[#[#[1.0 .1]  #[-1.2 -1.3]] #[#[0.2 -1.3] #[-0.1 -0.2]]]))]
	)
      ;;Initialization of GPU memory handled in make-cudnn-input-tensor call
      ;;Copy Host to GPU
      (tensor-cpu->gpu tensor-a)
      (tensor-cpu->gpu tensor-b)
      ;;Do we need workspace?
      ;;Do we need reserve space?
      (cudaDeviceSynchronize)
      (tensor-add-tensors cuda-handle tensor-a tensor-b)
      (cudaDeviceSynchronize)
      ;;Check
      ;;everything tensor-b is now 2.0
      (tensor-gpu->cpu tensor-b)
      (tensor-print-values tensor-b)))))

(define (simple-eval)
  (let*
	([cuda-handle (dref-handle (cuda-create-handle))]
	 [tensor-a (make-cudnn-input-tensor (array #[#[#[1.0 1.9] #[3.2 3.3]] #[#[1.8 3.3] #[2.1 2.2]]]))]
	 [tensor-b (make-cudnn-input-tensor (array #[#[#[1.0 .1]  #[-1.2 -1.3]] #[#[0.2 -1.3] #[-0.1 -0.2]]]))]
	)
      ;;Initialization of GPU memory handled in make-cudnn-input-tensor call
      ;;Copy Host to GPU
    (let ([resA (tensor-cpu->gpu tensor-a)]
	  [resB (tensor-cpu->gpu tensor-b)]
	  [resC (cuda-host-to-host-copy (cudnn-tensor-src-ptr tensor-b)
					(cudnn-tensor-src-ptr tensor-a)
					(cudnn-tensor-size tensor-a)
					)])
      (display (format "GPU copy return ~a and ~a.\n" resA resB))
      (display (format "Tensor A -> B copy return ~a.\n" resC))
      ;;Do we need workspace?
      ;;Do we need reserve space?
      ;(cudaDeviceSynchronize)
      (let
	  ([tens-add-res (tensor-add-tensors cuda-handle tensor-a tensor-b)])
	(display (format "tensor-add call returned ~a.\n" tens-add-res))
	;(cudaDeviceSynchronize)
	;;Check
	;;everything tensor-b is now 2.0
	(let ([gpu-cp (tensor-gpu->cpu tensor-b)])
	  (display (format "tensor copy call returned ~a.\n" gpu-cp))
	  ;(cudaDeviceSynchronize)
	  (tensor-print-values tensor-b))))))
  
