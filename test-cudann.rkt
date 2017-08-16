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
         )))
   (test-case
       "Check creation of a float array"
     (let* ([array-ptr (malloc 'atomic-interior 8)]
            [res (cudaMalloc array-ptr 4096)])
       (check-equal? res 'success "Array Creation")
       (check-equal? (cudaFree (ptr-ref array-ptr _pointer)) 'success "Array Release")
       ))
   (test-case
       "Check creation of tensor descriptor"
     (let* ([desc-ptr (malloc 'atomic-interior (ctype-sizeof _cudnnTensorDescriptor_t))]
            [res (cudnnCreateTensorDescriptor (cast desc-ptr _pointer _cudnnTensorDescriptor_t))]
                 )
       (check-equal? res 'success "Tensor Descriptor Creation")
       ))))



