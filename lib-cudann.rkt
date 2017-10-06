;;Based on github.com/soumith/cudnn.torch/blob/master/ffi.lua
;;> (enter! "lib-cudann.rkt") to load
#lang racket/base
(require
 racket/path
 ffi/unsafe
 ffi/unsafe/define)
(require yaml)

(define gpu-mem-path (build-path "libgpumem"))

(define rnn-api-path (build-path "librnn"))

;; local file ./config.yaml contains path of
;; cuda-lib and cudnn-lib like:
;; cuda-lib-path: /usr/local/cuda-8.0/targets/x86_64-linux/lib/libcudart
;; cudnn-lib-path: /usr/local/nvidia/cuda/lib64/libcudnn
(define config-hash (with-input-from-file
			"config.yaml"
		      (lambda () (read-yaml))))

(define-ffi-definer define-gpu-mem (ffi-lib gpu-mem-path))

(define-ffi-definer define-rnn-api (ffi-lib rnn-api-path))

(define-gpu-mem initGPUData (_fun _pointer _int _float -> _void))

(define-rnn-api runRNN (_fun _int _int _int _int _int _float _int _int -> _float))

(define cudann-path (hash-ref config-hash "cudnn-lib-path" ))

(define-ffi-definer define-cudann (ffi-lib cudann-path))

(define cuda-path (hash-ref config-hash "cuda-lib-path" ))

(define-ffi-definer define-cuda (ffi-lib cuda-path))
;;See https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__TYPES_ge15d9c8b7a240312b533d6122558085a.html#ge15d9c8b7a240312b533d6122558085a
;;typedef struct CUstream_st* cudaStream_t
(define-cpointer-type _cudaStream_t (_cpointer '_CUstream_st))
					;(define _cudaStream_t (_cpointer '_CUstream_st))
(define-cpointer-type _cudaStream_t-pointer (_cpointer '_cudaStream_t))
(define-cpointer-type _cudaEvent_t (_cpointer '_CUevent_st))
(define-cpointer-type _cudaEvent_t-pointer (_cpointer '_cudaEvent_t))
(define-cpointer-type _cudnnHandle_t (_cpointer '_cudnnContext))
(define-cpointer-type _cudnnHandle_t-pointer (_cpointer '_cudnnHandle_t))
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

(define  _cuda-error_t
  (_enum '(success = 0
		   ErrorMissingConfiguration
		   ErrorMemoryAllocation
		   ErrorInitializationError
		   ErrorLaunchFailure
		   ErrorPriorLaunchFailure
		   ErrorLaunchTimeout
		   ErrorLaunchOutOfResources
		   ErrorInvalidDeviceFunction
		   ErrorInvalidConfiguration
		   ErrorInvalidDevice
		   ErrorInvalidValue
		   ErrorInvalidPitchValue
		   ErrorInvalidSymbol
		   ErrorMapBufferObjectFailed
		   ErrorUnmapBufferObjectFailed
		   ErrorInvalidHostPointer
		   ErrorInvalidDevicePointer
		   ErrorInvalidTexture
		   ErrorInvalidTextureBinding
		   ErrorInvalidChannelDescriptor
		   ErrorInvalidMemcpyDirection
		   ErrorAddressOfConstant
		   ErrorTextureFetchFailed
		   ErrorTextureNotBound
		   ErrorSynchronizationError
		   )))

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


;;Define the caller for cudaMalloc
(define-cuda cudaMalloc (_fun _pointer _size ->  _cuda-error_t))

;;Define the caller for cudaFree
(define-cuda cudaFree (_fun _pointer -> _cudnn-status_t))

;;Synchronize device
(define-cuda cudaDeviceSynchronize (_fun -> _cudnn-status_t))

;;Create an event
(define-cuda cudaEventCreate (_fun _cudaEvent_t -> _cudnn-status_t))

;;Record event
(define-cuda cudaEventRecord (_fun _cudaEvent_t-pointer -> _cudnn-status_t))

;;Synch event
(define-cuda cudaEventSynchronize (_fun _cudaEvent_t-pointer -> _cudnn-status_t))

;;Define enumeration for specifying the type of copy
(define _cuda-memcpy-kind_t
  (_enum '(host-to-host = 0
		   host-to-device
		   device-to-host
                   device-to-device)))

;;Define caller for cudaMemcpy
(define-cuda cudaMemcpy (_fun _pointer _pointer _size _int -> _cuda-error_t))

;;Wrapper for host to device copy
(define (cuda-host-to-device-copy dst-prt src-ptr size)
  (cudaMemcpy dst-prt src-ptr size 1))

;;Wrapper to device to host copy
(define (cuda-device-to-host-copy dst-prt src-ptr size)
  (cudaMemcpy dst-prt src-ptr size 2))

;;Wrapper to device to host copy
(define (cuda-host-to-host-copy dst-prt src-ptr size)
  (cudaMemcpy dst-prt src-ptr size 0))




;;typedef struct cudnnTensorStruct*          cudnnTensorDescriptor_t;
(define-cpointer-type _cudnnTensorDescriptor_t (_cpointer '_cudnnTensorStruct))
(define-cpointer-type _cudnnTensorDescriptor_t-ptr (_cpointer '_cudnnTensorDescriptor_t))

;;typedef struct cudnnConvolutionStruct*     cudnnConvolutionDescriptor_t;
(define _cudnnConvolutionDescriptor_t (_cpointer '_cudnnConvolutionStruct))
;; typedef struct cudnnPoolingStruct*         cudnnPoolingDescriptor_t;
(define _cudnnPoolingDescriptor_t (_cpointer '_cudnnPoolingStruct))
;; ;; typedef struct cudnnFilterStruct*          cudnnFilterDescriptor_t
(define-cpointer-type _cudnnFilterDescriptor_t (_cpointer '_cudnnFilterStruct))
(define-cpointer-type _cudnnFilterDescriptor_t-ptr (_cpointer '_cudnnFilterDescriptor_t))

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

(define _cudnn-tensor-format_t
  (_enum '(tensor_nchw = 0
                       tensor_nhwc)))
;;Filters
(define-cudann cudnnCreateFilterDescriptor (_fun _cudnnFilterDescriptor_t  -> _cudnn-status_t))
(define-cudann cudnnDestroyFilterDescriptor (_fun _cudnnFilterDescriptor_t -> _cudnn-status_t))
(define-cudann cudnnSetFilterNdDescriptor (_fun _cudnnFilterDescriptor_t
						_cudnn-data-type_t
						_cudnn-tensor-format_t
						_int
						_pointer
						-> _cudnn-status_t))

;;Create an instance of a generic Tensor descriptor
(define-cudann cudnnCreateTensorDescriptor (_fun _cudnnTensorDescriptor_t-ptr  -> _cudnn-status_t))

(define-cudann cudnnSetTensorNdDescriptor (_fun _cudnnTensorDescriptor_t
                                                _int;_cudnn-data-type_t
                                                _int
                                                _pointer;_uintptr
                                                _pointer;_uintptr
                                                -> _cudnn-status_t))





;;Softmax function
(define _cudnn-sotmax-algorithm_t
  (_enum '(softmax_fast = 0
                        softmax_accurate
                        softmax_log)))

(define _cudnn-softmax-mode_t
  (_enum '(softmax_mode_instance = 0
                                 softmax_mode_channel)))

;;Softmax forward
(define-cudann cudnnSoftmaxForward
  (_fun
   _cudnnHandle_t
   _cudnn-sotmax-algorithm_t ;; the algo
   _cudnn-softmax-mode_t ;; mode
   _pointer ;; *alpha
   _cudnnTensorDescriptor_t ;; xDesc
   _pointer ;; *x
   _pointer ;; *beta
   _cudnnTensorDescriptor_t ;; yDesc
   _pointer ;; *y
   -> _cudnn-status_t))

;;Softmax backward
(define-cudann cudnnSoftmaxBackward
  (_fun
   _cudnnHandle_t
   _cudnn-sotmax-algorithm_t ;; the algo
   _cudnn-softmax-mode_t ;; mode
   _pointer ;; *alpha
   _cudnnTensorDescriptor_t ;; yDesc
   _pointer ;; *y
   _cudnnTensorDescriptor_t ;; dyDesc
   _pointer ;; *dy
   _pointer ;; *beta
   _cudnnTensorDescriptor_t ;; dxDesc
   _pointer ;; *dx
   -> _cudnn-status_t))

;;Tensor addition for testing
;; from cudnnStatus_t cudnnAddTensor( cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t aDesc, const void *A, const void *beta, const cudnnTensorDescriptor_t cDesc, void *C)
(define-cudann cudnnAddTensor
  (_fun
   _cudnnHandle_t
   _pointer ;; *alpha
   _cudnnTensorDescriptor_t ;; aDesc
   _pointer ;; *A
   _pointer ;; *beta
   _cudnnTensorDescriptor_t ;; cDesc
   _pointer ;; *C
   -> _cudnn-status_t))

;;;Activation functions
(define _cudnn-activation-mode_t
  (_enum '(activation_sigmoid = 0
                              activation_relu
                              activation_tanh
                              activation_clipped_relu)))

;;Dropout layer descriptor
;; The Dropout structure
;(define _cudnnDropoutDescriptor_t (_cpointer '_cudnnDropoutStruct))
(define-cpointer-type _cudnnDropoutDescriptor_t (_cpointer '_cudnnDropoutStruct))
(define-cpointer-type _cudnnDropoutDescriptor_t-ptr (_cpointer '_cudnnDropoutDescriptor_t))
;;Create Dropout descriptor
(define-cudann cudnnCreateDropoutDescriptor (_fun _cudnnDropoutDescriptor_t -> _cudnn-status_t))

;;Destroy Dropout descriptor
(define-cudann cudnnDestroyDropoutDescriptor (_fun _cudnnDropoutDescriptor_t -> _cudnn-status_t))



;;helper function to determine size of the states to be passed to cudnnSetDropoutDescriptor
(define-cudann cudnnDropoutGetReserveSpaceSize (_fun _cudnnHandle_t _size -> _cudnn-status_t))

;;helper function to determine size of the states to be passed to cudnnSetDropoutDescriptor
(define-cudann cudnnDropoutGetStatesSize (_fun _cudnnHandle_t _pointer -> _cudnn-status_t))

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

;;Define forward dropout
(define-cudann
  cudnnDropoutForward
  (_fun
   _cudnnHandle_t
   _cudnnDropoutDescriptor_t
   _cudnnTensorDescriptor_t ;; x descriptor
   _pointer ;; x
   _cudnnTensorDescriptor_t ;; y descriptor
   _pointer ;; y
   _pointer ;; reserveSpace
   _size ;; reserveSpaceSize in bytes
   -> _cudnn-status_t
   ))

;;Define backward dropout
(define-cudann
  cudnnDropoutBackward
  (_fun
   _cudnnHandle_t
   _cudnnDropoutDescriptor_t
   _cudnnTensorDescriptor_t ;; dx descriptor
   _pointer ;; dx
   _cudnnTensorDescriptor_t ;; dy descriptor
   _pointer ;; dy
   _pointer ;; reserveSpace
   _size ;; reserveSpaceSize in bytes
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
(define-cpointer-type _cudnnRNNDescriptor_t (_cpointer '_cudnnRNNStruct))
(define-cpointer-type _cudnnRNNDescriptor_t-ptr (_cpointer '_cudnnRNNDescriptor_t))


;;Create RNN descriptor
(define-cudann cudnnCreateRNNDescriptor (_fun _cudnnRNNDescriptor_t -> _cudnn-status_t))

;;Destroy RNN descriptor
(define-cudann cudnnDestroyRNNDescriptor (_fun _cudnnRNNDescriptor_t -> _cudnn-status_t))

;;Set RNN descriptor
(define-cudann cudnnSetRNNDescriptor
  (_fun
   _cudnnRNNDescriptor_t
   _int ;; hiddenSize
   _int ;; layers
   _cudnnDropoutDescriptor_t
   _cudnn-rnn-input-mode_t
   _cudnn-rnn-mode_t
   _cudnn-data-type_t ;; used to describe math precision
   -> _cudnn-status_t))

;;Get workspace size
(define-cudann cudnnGetRNNWorkspaceSize
  (_fun
   _cudnnHandle_t
   _cudnnRNNDescriptor_t
   _int ;; sequence length
   _cudnnTensorDescriptor_t ;; x descriptor
   _pointer ;; pointer to size
   -> _cudnn-status_t))

;;Get training reserve size
(define-cudann cudnnGetRNNTrainingReserveSize
  (_fun
   _cudnnHandle_t
   _cudnnRNNDescriptor_t
   _int ;; sequence length
   _cudnnTensorDescriptor_t ;; x descriptor
   _pointer ;; pointer to size
   -> _cudnn-status_t))

;;Get params size
(define-cudann cudnnGetRNNParamsSize
  (_fun
   _cudnnHandle_t
   _cudnnRNNDescriptor_t
   _cudnnTensorDescriptor_t ;; x descriptor
   _pointer ;; pointer to size
   _cudnn-data-type_t
   -> _cudnn-status_t))

;;Get Matrix Parameters of input layer
(define-cudann cudnnGetRNNLinLayerMatrixParams
  (_fun
   _cudnnHandle_t
   _cudnnRNNDescriptor_t
   _int ;; layer
   _cudnnTensorDescriptor_t ;; x descriptor
   _cudnnFilterDescriptor_t ;; filter descriptor
   _pointer ;; pointer to w
   _int ;; lin layer ID
   _cudnnFilterDescriptor_t ;; filter descriptor
   _pointer ;; matrix pointer?
   -> _cudnn-status_t))

;;Get Bias Params
(define-cudann cudnnGetRNNLinLayerBiasParams
  (_fun
   _cudnnHandle_t
   _cudnnRNNDescriptor_t
   _int ;; layer
   _cudnnTensorDescriptor_t ;; x descriptor
   _cudnnFilterDescriptor_t ;; w descriptor
   _pointer ;; pointer to w
   _int ;; lin layer ID
   _cudnnFilterDescriptor_t ;; filter descriptor
   _pointer ;; lin layer bias
   -> _cudnn-status_t))


;;;Forward inference
(define-cudann cudnnRNNForwardInference
  (_fun
   _cudnnHandle_t
   _cudnnRNNDescriptor_t
   _int ;; sequence length
   _cudnnTensorDescriptor_t ;; x descriptor
   _pointer ;; *x
   _cudnnTensorDescriptor_t ;; h descriptor
   _pointer ;; *hx
   _cudnnTensorDescriptor_t ;; c descriptor
   _pointer ;; *cx
   _cudnnFilterDescriptor_t ;; w descriptor
   _pointer ;; pointer to w
   _cudnnTensorDescriptor_t ;; y descriptor
   _pointer ;; pointer to y
   _cudnnTensorDescriptor_t ;; hy descriptor
   _pointer ;; pointer to hy
   _cudnnTensorDescriptor_t ;; cy descriptor
   _pointer ;; pointer to cy
   _pointer ;; pointer to workspace
   _size    ;; workspace size in bytes
   -> _cudnn-status_t))


;;;Forward inference
(define-cudann cudnnRNNForwardTraining
  (_fun
   _cudnnHandle_t
   _cudnnRNNDescriptor_t
   _int ;; sequence length
   _cudnnTensorDescriptor_t ;; x descriptor
   _pointer ;; *x
   _cudnnTensorDescriptor_t ;; h descriptor
   _pointer ;; *hx
   _cudnnTensorDescriptor_t ;; c descriptor
   _pointer ;; *cx
   _cudnnFilterDescriptor_t ;; w descriptor
   _pointer ;; pointer to w
   _cudnnTensorDescriptor_t ;; y descriptor
   _pointer ;; pointer to y
   _cudnnTensorDescriptor_t ;; hy descriptor
   _pointer ;; pointer to hy
   _cudnnTensorDescriptor_t ;; cy descriptor
   _pointer ;; pointer to cy
   _pointer ;; pointer to workspace
   _size    ;; workspace size in bytes
   _pointer ;; pointer to reserve space
   _size    ;; reserve size in bytes
   -> _cudnn-status_t))


;;;Forward inference
(define-cudann cudnnRNNBackwardData
  (_fun
   _cudnnHandle_t
   _cudnnRNNDescriptor_t
   _int ;; sequence length
   _cudnnTensorDescriptor_t ;; y descriptor
   _pointer ;; *y
   _cudnnTensorDescriptor_t ;; dy descriptor
   _pointer ;; *dy
   _cudnnTensorDescriptor_t ;; dhy descriptor
   _pointer ;; *dhy
   _cudnnTensorDescriptor_t ;; dcy descriptor
   _pointer ;; *dcy
   _cudnnFilterDescriptor_t ;; w descriptor
   _pointer ;; pointer to w
   _cudnnTensorDescriptor_t ;; hx descriptor
   _pointer ;; pointer to hx
   _cudnnTensorDescriptor_t ;; cx descriptor
   _pointer ;; pointer to cx
   _cudnnTensorDescriptor_t ;; dx descriptor
   _pointer ;; pointer to dx
   _cudnnTensorDescriptor_t ;; dhx descriptor
   _pointer ;; pointer to dhx
   _cudnnTensorDescriptor_t ;; dcx descriptor
   _pointer ;; pointer to dcx
   _pointer ;; pointer to workspace
   _size    ;; workspace size in bytes
   _pointer ;; pointer to reserve space
   _size    ;; reserve size in bytes
   -> _cudnn-status_t))

;;;Backward weights
(define-cudann cudnnRNNBackwardWeights
  (_fun
   _cudnnHandle_t
   _cudnnRNNDescriptor_t
   _int ;; sequence length
   _cudnnTensorDescriptor_t ;; x descriptor
   _pointer ;; *x
   _cudnnTensorDescriptor_t ;; hx descriptor
   _pointer ;; *hx
   _cudnnTensorDescriptor_t ;; y descriptor
   _pointer ;; *y
   _pointer ;; pointer to workspace
   _size    ;; workspace size in bytes
   _cudnnFilterDescriptor_t ;; dw descriptor
   _pointer ;; pointer to dw
   _pointer ;; pointer to reserve space
   _size    ;; reserve size in bytes
   -> _cudnn-status_t))

(define (cuda-create-handle)
  (let*
      ([block (malloc 'atomic-interior 8)]
       [hndlptr (cast block _pointer _cudnnHandle_t-pointer)]
       [res (cudnnCreate hndlptr)])
    hndlptr))

(define (dref-handle ptr)
  (let
      ([pref (ptr-ref ptr _cudnnHandle_t)])
    pref
  ))


(define (dref-ptr ptr)
  (let
      ([pref (ptr-ref ptr _pointer)])
    pref
  ))

;; Allocate a tensor descriptor of the right type
;; Basically need an allocator for each of the
;; descriptor pointer types -- create and dref
(define (cuda-create-tensor-descriptr-ptr size)
  (let ([desc-ptr (malloc 'atomic-interior (ctype-sizeof _cudnnTensorDescriptor_t))])
    (cast desc-ptr _pointer _cudnnTensorDescriptor_t-ptr)))

(define (dref-tensor-desc-ptr ptr)
  (let
      ([pref (ptr-ref ptr _cudnnTensorDescriptor_t)])
    pref
    ))

;;create & dref for dropout
(define (cuda-create-dropout-descriptr-ptr size)
  (print (format "Size of drop out is ~a" (ctype-sizeof _cudnnDropoutDescriptor_t)))
  (let ([desc-ptr (malloc 'atomic-interior (ctype-sizeof _cudnnDropoutDescriptor_t))])
    (cast desc-ptr _pointer _cudnnDropoutDescriptor_t)))

(define (dref-dropout-desc-ptr ptr)
  (let
      ([pref (ptr-ref ptr _cudnnDropoutDescriptor_t)])
    pref
    ))


;;create & dref for rnn
(define (cuda-create-rnn-descriptr-ptr size)
  (let ([desc-ptr (malloc 'atomic-interior (ctype-sizeof  _cudnnRNNDescriptor_t))])
    (cast desc-ptr _pointer  _cudnnRNNDescriptor_t-ptr)))

(define (dref-rnn-desc-ptr ptr)
  (let
      ([pref (ptr-ref ptr  _cudnnRNNDescriptor_t)])
    pref
    ))


;;create & dref for filter
(define (cuda-create-filter-descriptr-ptr size)
  (let ([desc-ptr (malloc 'atomic-interior (ctype-sizeof  _cudnnFilterDescriptor_t))])
    (cast desc-ptr _pointer  _cudnnFilterDescriptor_t-ptr)))

(define (dref-filter-desc-ptr ptr)
  (let
      ([pref (ptr-ref ptr  _cudnnFilterDescriptor_t)])
    pref
    ))

(define (dref-int-ptr ptr)
  (let
      ([pref (ptr-ref ptr _int)])
    pref
    ))
  


;;create & dref lin layer

;;create & dref lin layer bias


(define (create-lstm-layer nodes-in nodes-out)
  (print (string-append "Building a LSTM with "
                        (make-string nodes-in) " and "
                        (make-string nodes-out) " nodes out ")))

(provide cudnnCreate cudnnDestroy _cudnnHandle_t-pointer
	 cuda-create-handle
         cudaMalloc cudaFree cudaMemcpy cudaDeviceSynchronize
         cudnnCreateTensorDescriptor
	 cudnnSetTensorNdDescriptor
	 cudnnCreateDropoutDescriptor
	 cudnnSetDropoutDescriptor
         cudnnDropoutGetStatesSize
	 cudnnCreateRNNDescriptor
	 cudnnSetRNNDescriptor
         cudnnGetRNNParamsSize
         cudnnGetRNNTrainingReserveSize
         cudnnGetRNNWorkspaceSize
         cudnnGetRNNLinLayerMatrixParams
         cudnnGetRNNLinLayerBiasParams
         cudnnCreateFilterDescriptor
	 cudnnDestroyFilterDescriptor
	 cudnnDestroyDropoutDescriptor
	 cudnnSetFilterNdDescriptor
	 cudnnRNNForwardTraining
	 cudnnRNNBackwardWeights
	 cudnnRNNBackwardData
	 cudaEventCreate
	 cudaEventRecord
	 cudaEventSynchronize
	 cudnnAddTensor
	 cuda-host-to-device-copy
	 cuda-device-to-host-copy
         _cudnnHandle_t _cuda-memcpy-kind_t
         _cudnnTensorDescriptor_t
	 _cudnnTensorDescriptor_t-ptr
         _cudnnFilterDescriptor_t
	 _cudnnFilterDescriptor_t-ptr
	 _cudnnDropoutDescriptor_t
	 _cudnnDropoutDescriptor_t-ptr
	 _cudnnRNNDescriptor_t
	 _cudnnRNNDescriptor_t-ptr
         _cudaEvent_t
	 _cudaEvent_t-pointer
	 _cudnn-data-type_t
	 _cudnn-rnn-input-mode_t
	 _cudnn-rnn-mode_t
	 _cudnn-tensor-format_t
	 cuda-create-tensor-descriptr-ptr
	 cuda-create-filter-descriptr-ptr
	 cuda-create-dropout-descriptr-ptr
	 cuda-create-rnn-descriptr-ptr
	 cuda-host-to-host-copy
	 dref-tensor-desc-ptr
	 dref-dropout-desc-ptr
	 dref-filter-desc-ptr
	 dref-rnn-desc-ptr
	 dref-handle
	 dref-ptr
	 dref-int-ptr
	 initGPUData
	 runRNN
         _cudnn-status_t)
