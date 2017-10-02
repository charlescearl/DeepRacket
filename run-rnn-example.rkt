#lang typed/racket
(require "cudnn-rnn.rkt")
(require "cuda-api.rkt")
(require "cudnn-api.rkt")

(define (run-example params)
  (cudaDeviceSynchronize)
  (let* (
	 [rnn (make-cudnn-rnn params)]
	 [rnn-workspace (make-cudnn-rnn-workspace params)]
	 [timings (run-workspace-forward-train rnn-workspace)]
	)
    (print-timings timings)))
    
