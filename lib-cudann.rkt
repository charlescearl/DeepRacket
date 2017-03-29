#lang racket/base
(require
 racket/path
 ffi/unsafe
 ffi/unsafe/define)

(define cudann-path (build-path "/" "media" "Hadoop2" "nvidia" "cuda" "lib64" "libcudnn"))

(define-ffi-definer define-cudann (ffi-lib cudann-path))

;; Using cudnn.torch as a guide https://github.com/soumith/cudnn.torch
(define _cudnnStatus_t
  (_enum '(success = 0
		   not_initialized
		   alloc_failed
		   bad_param
		   internal_error
		   invalid_value
		   arch_mismatch
		   mapping_error
		   execution_failed
		   not_supported
		   license_error)))

(define (create-lstm-layer nodes-in nodes-out)
  (print (string-append "Building a LSTM with " (make-string nodes-in) " and " (make-string nodes-out) " nodes out ")))


