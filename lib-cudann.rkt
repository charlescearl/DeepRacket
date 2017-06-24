;;Based on github.com/soumith/cudnn.torch/blob/master/ffi.lua
;;> (enter! "lib-cudann.rkt") to load
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


;;typedef struct cudnnTensorStruct*          cudnnTensorDescriptor_t;
(define _cudnnTensorDescriptor_t (_cpointer '_cudnnTensorStruct))
;;typedef struct cudnnConvolutionStruct*     cudnnConvolutionDescriptor_t;
(define _cudnnConvolutionDescriptor_t (_cpointer '_cudnnConvolutionStruct))
;; typedef struct cudnnPoolingStruct*         cudnnPoolingDescriptor_t;
(define _cudnnPoolingDescriptor_t (_cpointer '_cudnnPoolingStruct))
;; ;; typedef struct cudnnFilterStruct*          cudnnFilterDescriptor_t
(define _cudnnFilterDescriptor_t (_cpointer '_cudnnFilterStruct))

;; typedef struct cudnnLRNStruct*             cudnnLRNDescriptor_t;
(define _cudnnLRNDescriptor_t (_cpointer '_cudnnLRNStruct))
;; typedef struct cudnnActivationStruct*      cudnnActivationDescriptor_t;
(define _cudnnActivationDescriptor_t (_cpointer '_cudnnActivationStruct))
;; typedef struct cudnnSpatialTransformerStruct* cudnnSpatialTransformerDescriptor_t;
(define _cudnnSpatialTransformerDescriptor_t (_cpointer '_cudnnSpatialTransformerStruct))
;; typedef struct cudnnOpTensorStruct*        cudnnOpTensorDescriptor_t;
(define _cudnnOpTensorDescriptor_t (_cpointer '_cudnnOpTensorStruct))

;; CUDNN data type
(define _cudnn-data-type_t
  (_enum '(float = 0
		   double
		   half)))

;; CUDNN propaget NAN
(define _cudnn-nan-propagation_t
  (_enum '(not_propagate_nan = 0
                             propagate_nan)))

;;Maximum supported number of tensor dimensions
(define _cudnn-dim-max-fake-enum_t
  (_enum '(dim_max = 8)))

;;Create an instance of a generic Tensor descriptor
(define-cudann cudnnCreateTensorDescriptor (_fun _cudnnTensorDescriptor_t  -> _cudnn-status_t))


(define _cudnn-tensor-format_t
  (_enum '(tensor_nchw = 0
                       tensor_nhwc)))

;;Dropout layer descriptor
;; The Dropout structure
(define _cudnnDropoutDescriptor_t (_cpointer '_cudnnDropoutStruct))

;;Create Dropout descriptor
(define-cudann cudnnCreateDropoutDescriptor (_fun _cudnnDropoutDescriptor_t -> _cudnn-status_t))

;;Destroy Dropout descriptor
(define-cudann cudnnDestroyDropoutDescriptor (_fun _cudnnDropoutDescriptor_t -> _cudnn-status_t))

;;helper function to determine size of the states to be passed to cudnnSetDropoutDescriptor
(define-cudann cudnnDropoutGetReserveSpaceSize (_fun _cudnnHandle_t _size -> _cudnn-status_t))

;;Set parameters of the Dropout Descriptor
(define-cudann
  cudnnSetDropoutDescriptor
  (_fun
   _cudnnDropoutDescriptor_t
   _cudnnHandle_t
   _float ;;percent of dropout?
   _pointer ;;*states
   _size ;; state size in bytes
   _ullong ;;seed
   -> _cudnn-status_t
   ))
                                          

;;RNN API
(define _cudnn-rnn-mode_t
  (_enum '(rnn_relu = 0
                    rnn_tanh
                    lstm
                    gru)))

(define _cudnn-direction-mode_t
  (_enum '(unidirectional = 0
                          bidirectional)))


(define _cudnn-rnn-input-mode_t
  (_enum '(linear_input = 0
                    skip_input)))



;; The RNN structure
(define _cudnnRNNDescriptor_t (_cpointer '_cudnnRNNStruct))

;;Create RNN descriptor
(define-cudann cudnnCreateRNNDescriptor (_fun _cudnnRNNDescriptor_t -> _cudnn-status_t))

;;Destroy RNN descriptor
(define-cudann cudnnDestroyRNNDescriptor (_fun _cudnnRNNDescriptor_t -> _cudnn-status_t))

;;Destroy RNN descriptor
;;(define-cudann cudnnSetRNNDescriptor
;;  (_fun _cudnnRNNDescriptor_t
;;        int ;; hiddenSize
;;        int ;; layers
;;        -> _cudnn-status_t))

(define (create-lstm-layer nodes-in nodes-out)
  (print (string-append "Building a LSTM with "
                        (make-string nodes-in) " and "
                        (make-string nodes-out) " nodes out ")))


