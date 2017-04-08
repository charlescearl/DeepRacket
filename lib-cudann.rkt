#lang racket/base
(require
 racket/path
 ffi/unsafe
 ffi/unsafe/define)

(define cudann-path (build-path "/" "media" "Hadoop2" "nvidia" "cuda" "lib64" "libcudnn"))

(define-ffi-definer define-cudann (ffi-lib cudann-path))


;;See https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__TYPES_ge15d9c8b7a240312b533d6122558085a.html#ge15d9c8b7a240312b533d6122558085a
;;typedef struct CUstream_st* cudaStream_t
(define _cudaStream_t (_cpointer '_CUstream_st))
(define _cudaStream_t-pointer (_cpointer '_cudaStream_t))
(define _cudnnHandle_t (_cpointer '_cudnnContext))
(define _cudnnHandle_t-pointer (_cpointer '_cudnnHandle_t))
;; Using cudnn.torch as a guide https://github.com/soumith/cudnn.torch
;; cudnn return codes
(define _cudnn-status_t
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

;; Error messages
;; const char *              cudnnGetErrorString(cudnnStatus_t status);
(define-cudann cudnnGetErrorString (_fun _cudnn-status_t -> _string))

;;cudnnStatus_t             cudnnCreate        (cudnnHandle_t *handle);
(define-cudann cudnnCreate (_fun _cudnnHandle_t-pointer -> _cudnn-status_t))

;;cudnnStatus_t             cudnnDestroy       (cudnnHandle_t handle);
(define-cudann cudnnDestroy (_fun _cudnnHandle_t -> _cudnn-status_t))

;;cudnnStatus_t             cudnnSetStream     (cudnnHandle_t handle, cudaStream_t streamId);
(define-cudann cudnnSetStream (_fun _cudnnHandle_t _cudaStream_t -> _cudnn-status_t))

;;cudnnStatus_t             cudnnGetStream     (cudnnHandle_t handle, cudaStream_t *streamId);
(define-cudann cudnnGetStream (_fun _cudnnHandle_t _cudaStream_t-pointer -> _cudnn-status_t))

(define (create-lstm-layer nodes-in nodes-out)
  (print (string-append "Building a LSTM with " (make-string nodes-in) " and " (make-string nodes-out) " nodes out ")))


