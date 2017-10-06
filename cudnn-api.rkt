;; Provides typed interfaces to nvida cudnn api
;;Typed tensors
#lang typed/racket
(require "ffi-functional.rkt")
(require/typed
 "lib-cudann.rkt"
 [_cudnnHandle_t CType]
 [_cudnn-status_t CType]
 [_cudnnTensorDescriptor_t CType]
 [_cudnn-data-type_t CType]
 [_cudnnDropoutDescriptor_t CType]
 [_cudnnRNNDescriptor_t CType]
 [_cudnnFilterDescriptor_t CType]
 [_cudnn-rnn-input-mode_t CType]
 [_cudnn-rnn-mode_t CType]
 [_cudnn-tensor-format_t CType]
 [cuda-create-tensor-descriptr-ptr (Exact-Nonnegative-Integer -> CPointer)]
 [dref-tensor-desc-ptr (CPointer -> CPointer)]
 [cuda-create-filter-descriptr-ptr (Exact-Nonnegative-Integer -> CPointer)]
 [dref-filter-desc-ptr (CPointer -> CPointer)]
 [cuda-create-rnn-descriptr-ptr (Exact-Nonnegative-Integer -> CPointer)]
 [dref-rnn-desc-ptr (CPointer -> CPointer)]
 [cuda-create-dropout-descriptr-ptr (Exact-Nonnegative-Integer -> CPointer)]
 [dref-dropout-desc-ptr (CPointer -> CPointer)]
 [cudnnSetDropoutDescriptor (CPointer CPointer Flonum
				      CPointer Positive-Integer
				      Positive-Integer -> Symbol)]
 [cudnnSetTensorNdDescriptor (CPointer Exact-Nonnegative-Integer
				    Exact-Nonnegative-Integer
				    CPointer CPointer -> Symbol)]
 [cudnnCreateTensorDescriptor (CPointer -> Symbol)]
 [cudnnCreateDropoutDescriptor (CPointer -> Symbol)]
 [cudnnSetRNNDescriptor (CPointer
  			 Exact-Nonnegative-Integer
  			 Exact-Nonnegative-Integer
  			 CPointer
  			 Exact-Nonnegative-Integer
  			 Exact-Nonnegative-Integer
  			 Exact-Nonnegative-Integer
  			 -> Symbol)]
 [cudnnCreateRNNDescriptor (CPointer -> Symbol)]
 [cudnnCreateFilterDescriptor (CPointer -> Symbol)]
 [cudnnDestroyFilterDescriptor (CPointer -> Symbol)]
 [cudnnDestroyDropoutDescriptor (CPointer -> Symbol)]
 [cudnnSetFilterNdDescriptor (CPointer
  			      Exact-Nonnegative-Integer
  			      Exact-Nonnegative-Integer
  			      Exact-Nonnegative-Integer
  			      CPointer
  			      -> Symbol)]
 [cudnnGetRNNParamsSize (CPointer CPointer CPointer
  				  CPointer Exact-Nonnegative-Integer
  				  -> Symbol)]
 [cudnnDropoutGetStatesSize (CPointer CPointer -> Symbol)]
 [cudnnGetRNNWorkspaceSize (CPointer CPointer Exact-Nonnegative-Integer
				  CPointer CPointer -> Symbol)]
 [cudnnGetRNNTrainingReserveSize (CPointer CPointer Exact-Nonnegative-Integer
					CPointer CPointer -> Symbol)]
 [cudnnGetRNNLinLayerMatrixParams (CPointer
				   CPointer
				   Exact-Nonnegative-Integer
				   CPointer
				   CPointer
				   CPointer
				   Exact-Nonnegative-Integer
				   CPointer
				   CPointer
				   -> Symbol)]
 [cudnnGetRNNLinLayerBiasParams (CPointer
				   CPointer
				   Exact-Nonnegative-Integer
				   CPointer
				   CPointer
				   CPointer
				   Exact-Nonnegative-Integer
				   CPointer
				   CPointer
				   -> Symbol)]
 [cudnnRNNForwardTraining ( CPointer ;; the handle
   CPointer ;;_cudnnRNNDescriptor_t
   Exact-Nonnegative-Integer ;; _int ;; sequence length
   CPointer ;;_cudnnTensorDescriptor_t ;; x descriptor
   CPointer ;;_pointer ;; *x
   CPointer ;;_cudnnTensorDescriptor_t ;; h descriptor
   CPointer ;;_pointer ;; *hx
   CPointer ;;_cudnnTensorDescriptor_t ;; c descriptor
   CPointer ;;_pointer ;; *cx
   CPointer ;;_cudnnFilterDescriptor_t ;; w descriptor
   CPointer ;;_pointer ;; pointer to w
   CPointer ;;_cudnnTensorDescriptor_t ;; y descriptor
   CPointer ;;_pointer ;; pointer to y
   CPointer ;;_cudnnTensorDescriptor_t ;; hy descriptor
   CPointer ;;_pointer ;; pointer to hy
   CPointer ;;_cudnnTensorDescriptor_t ;; cy descriptor
   CPointer ;; _pointer ;; pointer to cy
   CPointer ;;_pointer ;; pointer to workspace
   Exact-Nonnegative-Integer  ;;_size    ;; workspace size in bytes
   CPointer ;; _pointer ;; pointer to reserve space
   Exact-Nonnegative-Integer ;;_size    ;; reserve size in bytes
   -> CType
   )]

 [cudnnAddTensor (CPointer ;; Handle
		  CPointer ;; Pointer to the data for alpha
		  CPointer ;; Tensor descriptor
		  CPointer ;; Pointer to A
		  CPointer ;; Pointer to beta
		  CPointer ;; Descriptor for C
		  CPointer ;; Pointer to C
		  -> Symbol)]
		  
 [cudnnRNNBackwardWeights
  (
   CPointer ;;_cudnnHandle_t
   CPointer ;;_cudnnRNNDescriptor_t
   Exact-Nonnegative-Integer  ;;_int ;; sequence length
   CPointer ;;_cudnnTensorDescriptor_t ;; x descriptor
   CPointer ;;_pointer ;; *x
   CPointer ;;_cudnnTensorDescriptor_t ;; hx descriptor
   CPointer ;;_pointer ;; *hx
   CPointer ;;_cudnnTensorDescriptor_t ;; y descriptor
   CPointer ;;_pointer ;; *y
   CPointer ;;_pointer ;; pointer to workspace
   Exact-Nonnegative-Integer  ;;_size    ;; workspace size in bytes
   CPointer ;;_cudnnFilterDescriptor_t ;; dw descriptor
   CPointer ;;_pointer ;; pointer to dw
   CPointer ;;_pointer ;; pointer to reserve space
   Exact-Nonnegative-Integer  ;;_size    ;; reserve size in bytes
   -> Symbol;; _cudnn-status_t
   )]
 [cudnnRNNBackwardData
  (
   CPointer ;;_cudnnHandle_t
   CPointer ;;_cudnnRNNDescriptor_t
   Exact-Nonnegative-Integer  ;;_int ;; sequence length
   CPointer ;;_cudnnTensorDescriptor_t ;; y descriptor
   CPointer ;;_pointer ;; *y
   CPointer ;;_cudnnTensorDescriptor_t ;; dy descriptor
   CPointer ;;_pointer ;; *dy
   CPointer ;;_cudnnTensorDescriptor_t ;; dhy descriptor
   CPointer ;;_pointer ;; *dhy
   CPointer ;;_cudnnTensorDescriptor_t ;; dcy descriptor
   CPointer ;;_pointer ;; *dcy
   CPointer ;;_cudnnFilterDescriptor_t ;; w descriptor
   CPointer ;;_pointer ;; pointer to w
   CPointer ;;_cudnnTensorDescriptor_t ;; hx descriptor
   CPointer ;;_pointer ;; pointer to hx
   CPointer ;;_cudnnTensorDescriptor_t ;; cx descriptor
   CPointer ;;_pointer ;; pointer to cx
   CPointer ;;_cudnnTensorDescriptor_t ;; dx descriptor
   CPointer ;;_pointer ;; pointer to dx
   CPointer ;;_cudnnTensorDescriptor_t ;; dhx descriptor
   CPointer ;;_pointer ;; pointer to dhx
   CPointer ;;_cudnnTensorDescriptor_t ;; dcx descriptor
   CPointer ;;_pointer ;; pointer to dcx
   CPointer ;;_pointer ;; pointer to workspace
   Exact-Nonnegative-Integer  ;;_size    ;; workspace size in bytes
   CPointer ;;_pointer ;; pointer to reserve space
   Exact-Nonnegative-Integer  ;;_size    ;; reserve size in bytes
   -> Symbol ;; _cudnn-status_t)
   )]
)

(: TENSOR-DESC-SIZE  Exact-Nonnegative-Integer)
(define TENSOR-DESC-SIZE (ctype-sizeof _cudnnTensorDescriptor_t))

(: get-tensor-desc-array (-> Exact-Nonnegative-Integer CPointer))
(define (get-tensor-desc-array size)
  (print "Size of descriptor is " )
  (print size )
  (let ([ptr (malloc 'atomic-interior (* TENSOR-DESC-SIZE size))])
    (print "Allocated ")
    (print ptr)
    ptr))

(: get-tensor-desc-ptr (-> CPointer Exact-Nonnegative-Integer CPointer))
(define (get-tensor-desc-ptr block offset)
  (ptr-add block offset _cudnnTensorDescriptor_t)
  )


;; (provide tensor rnn ptr-array-ref array->cptr)
(provide
 cudnnSetTensorNdDescriptor 
 cudnnCreateTensorDescriptor 
 cudnnCreateDropoutDescriptor 
 cudnnSetRNNDescriptor 
 cudnnCreateRNNDescriptor 
 cudnnCreateFilterDescriptor 
 cudnnDestroyFilterDescriptor 
 cudnnSetFilterNdDescriptor 
 cudnnGetRNNParamsSize
 cudnnSetDropoutDescriptor
 cudnnDropoutGetStatesSize
 cudnnDestroyDropoutDescriptor
 cudnnGetRNNWorkspaceSize 
 cudnnGetRNNTrainingReserveSize 
 cudnnGetRNNLinLayerMatrixParams 
 cudnnGetRNNLinLayerBiasParams 
 cudnnRNNForwardTraining 
 cudnnRNNBackwardWeights
 cudnnRNNBackwardData
 cudnnAddTensor
 get-tensor-desc-array
 get-tensor-desc-ptr
 cuda-create-tensor-descriptr-ptr
 dref-tensor-desc-ptr
 cuda-create-dropout-descriptr-ptr
 dref-dropout-desc-ptr
 cuda-create-rnn-descriptr-ptr
 dref-rnn-desc-ptr
 cuda-create-filter-descriptr-ptr
 dref-filter-desc-ptr
 )


