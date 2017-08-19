#lang racket/base
(require
 ffi/unsafe)

(define (malloc-interior size)
  (malloc 'atomic-interior size))

(provide malloc-interior cpointer?)
