#lang racket/base
(require
 rackunit
 rackunit/text-ui
 ffi/unsafe
 "lib-cudann.rkt")

;; define a test case
(define handle-tests
  (test-suite
   "Tests for cudann"
   (test-case
       "Check handle creation"
     (let* ([block (malloc 'atomic-interior 8)]
            [hndlptr (cast block _pointer _cudnnHandle_t-pointer)]
            [res (cudnnCreate hndlptr)])
       (check-equal? res 'success "Handle Creation")
       (let ([hndl (ptr-ref hndlptr _cudnnHandle_t)])
         (check-equal? (cudnnDestroy hndl) 'success "Handle Release")
       )))))



