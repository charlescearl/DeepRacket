#lang typed/racket
(require typed/rackunit
	 typed/rackunit/text-ui)
(require "cudnn-api.rkt")
(require "cuda-api.rkt")
(require "mem-utils.rkt")
(require "cudnn-tensor.rkt")
(require "cudnn-dropout.rkt")
(require math/array)

(define dropout-tests
  (test-suite
   "Test creation and release of dropout descriptor"
   (test-case
    "Check that we can create and release a dropout descriptor"
    (let*
	([cuda-handle (dref-handle (cuda-create-handle))]
	 [droupout-struct (make-dropout cuda-handle 0.5)])
      (check-equal? (cudnn-dropout? droupout-struct) #t "Dropout struct created")
      (let ([des-res (cudnnDestroyDropoutDescriptor (cudnn-dropout-desc droupout-struct))])
	(check-equal? des-res 'success "Dropout structure released"))))))

(define (dropout-test)
  (let*
	([cuda-handle (dref-handle (cuda-create-handle))]
	 [droupout-struct (make-dropout cuda-handle 0.5)]
	 [val : Symbol (cudnnDestroyDropoutDescriptor (cudnn-dropout-desc droupout-struct))]
	 [tval : Symbol 'success])
    (check-equal? (cudnn-dropout? droupout-struct) #t "Dropout struct created")
    (assert (equal? val tval))
    ))

    
