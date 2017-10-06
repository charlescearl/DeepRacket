#lang typed/racket
(require "ffi-functional.rkt")
(require/typed
 "lib-cudann.rkt"
 [runRNN (Positive-Integer Positive-Integer
			   Positive-Integer Positive-Integer
			   Positive-Integer Flonum
			   Nonnegative-Integer Nonnegative-Integer -> Flonum)]
 )

;;(runRNN 16 8 16 16 4 0.5 0 0)
