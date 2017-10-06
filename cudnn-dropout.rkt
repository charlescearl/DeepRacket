#lang typed/racket
(require "cudnn-tensor.rkt")
(require "cuda-api.rkt") ; for get-pointer
(require "cudnn-api.rkt")
(require "mem-utils.rkt")
(require "ffi-functional.rkt")

;;;The dropout object

(struct cudnn-dropout
  ([desc : CPointer]
   [state-size : Integer]
   [states : CPointer]
   [dropout-factor : Flonum]
   ))

(define (make-dropout [handle : CPointer] [dropout : Flonum]) : cudnn-dropout
  (let ([desc-ptr (cuda-create-dropout-descriptr-ptr 1)]
	[states (get-pointer)]
	[stateSize (get-int-pointer)]
	[seed : Positive-Integer 1337])
    (print (format "Vars ~a ~a ~a ~a" desc-ptr states stateSize seed))
    (let ([create-res (cudnnCreateDropoutDescriptor desc-ptr)])
      (print (format "Dropout creation result ~a" create-res))
      (cudnnDropoutGetStatesSize handle stateSize)
      (cudaMalloc states (dref-int-ptr stateSize))
      (let
	  ([drp-res
	    (cudnnSetDropoutDescriptor (dref-dropout-desc-ptr desc-ptr)
				       handle
				       dropout
				       states
				       (dref-int-ptr stateSize)
				       seed)])
	(print (format "Dropout allocation returned ~a" drp-res))
	(cudnn-dropout (dref-dropout-desc-ptr desc-ptr)
		       (dref-int-ptr stateSize)
		       states
		       dropout)))))

(provide
 make-dropout
 cudnn-dropout
 cudnn-dropout?
 cudnn-dropout-desc
 )
