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
 [dref-tensor-desc-ptr (CPointer -> CPointer)]
 [cuda-create-tensor-descriptr-ptr (Exact-Nonnegative-Integer -> CPointer)]
 [cudnnSetTensorNdDescriptor (CPointer Exact-Nonnegative-Integer
				    Exact-Nonnegative-Integer
				    CPointer CPointer -> Symbol)]
 [cudnnCreateTensorDescriptor (CPointer -> Symbol)]
 [cudnnCreateDropoutDescriptor (CType -> CType)]
 [cudnnSetRNNDescriptor (CType
  			 Exact-Nonnegative-Integer
  			 Exact-Nonnegative-Integer
  			 CType
  			 CType
  			 CType
  			 CType
  			 -> CType)]
 [cudnnCreateRNNDescriptor (CType -> CType)]
 [cudnnCreateFilterDescriptor (CType -> CType)]
 [cudnnDestroyFilterDescriptor (CType -> CType)]
 [cudnnSetFilterNdDescriptor (CType
  			      CType
  			      CType
  			      Exact-Nonnegative-Integer
  			      CType
  			      -> CType)]
 [cudnnGetRNNParamsSize (CType CType CType
  				  CType CType
  				  -> CType)]
 [cudnnDropoutGetStatesSize (CType CType -> CType)]
 [cudnnGetRNNWorkspaceSize (CType CType Exact-Nonnegative-Integer
				  CType CType -> CType)]
 [cudnnGetRNNTrainingReserveSize (CType CType Exact-Nonnegative-Integer
					CType CType -> CType)]
 [cudnnGetRNNLinLayerMatrixParams (CType
				   CType
				   Exact-Nonnegative-Integer
				   CType
				   CType
				   CType
				   Exact-Nonnegative-Integer
				   CType
				   CType
				   -> CType)]
 [cudnnGetRNNLinLayerBiasParams (CType
				   CType
				   Exact-Nonnegative-Integer
				   CType
				   CType
				   CType
				   Exact-Nonnegative-Integer
				   CType
				   CType
				   -> CType)]
 [cudnnRNNForwardTraining ( CType ;; the handle
   CType ;;_cudnnRNNDescriptor_t
   Exact-Nonnegative-Integer ;; _int ;; sequence length
   CType ;;_cudnnTensorDescriptor_t ;; x descriptor
   CType ;;_pointer ;; *x
   CType ;;_cudnnTensorDescriptor_t ;; h descriptor
   CType ;;_pointer ;; *hx
   CType ;;_cudnnTensorDescriptor_t ;; c descriptor
   CType ;;_pointer ;; *cx
   CType ;;_cudnnFilterDescriptor_t ;; w descriptor
   CType ;;_pointer ;; pointer to w
   CType ;;_cudnnTensorDescriptor_t ;; y descriptor
   CType ;;_pointer ;; pointer to y
   CType ;;_cudnnTensorDescriptor_t ;; hy descriptor
   CType ;;_pointer ;; pointer to hy
   CType ;;_cudnnTensorDescriptor_t ;; cy descriptor
   CType ;; _pointer ;; pointer to cy
   CType ;;_pointer ;; pointer to workspace
   Exact-Nonnegative-Integer  ;;_size    ;; workspace size in bytes
   CType ;; _pointer ;; pointer to reserve space
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
   CType ;;_cudnnHandle_t
   CType ;;_cudnnRNNDescriptor_t
   Exact-Nonnegative-Integer  ;;_int ;; sequence length
   CType ;;_cudnnTensorDescriptor_t ;; x descriptor
   CType ;;_pointer ;; *x
   CType ;;_cudnnTensorDescriptor_t ;; hx descriptor
   CType ;;_pointer ;; *hx
   CType ;;_cudnnTensorDescriptor_t ;; y descriptor
   CType ;;_pointer ;; *y
   CType ;;_pointer ;; pointer to workspace
   Exact-Nonnegative-Integer  ;;_size    ;; workspace size in bytes
   CType ;;_cudnnFilterDescriptor_t ;; dw descriptor
   CType ;;_pointer ;; pointer to dw
   CType ;;_pointer ;; pointer to reserve space
   Exact-Nonnegative-Integer  ;;_size    ;; reserve size in bytes
   -> CType ;; _cudnn-status_t
   )]
 [cudnnRNNBackwardData
  (
   CType ;;_cudnnHandle_t
   CType ;;_cudnnRNNDescriptor_t
   Exact-Nonnegative-Integer  ;;_int ;; sequence length
   CType ;;_cudnnTensorDescriptor_t ;; y descriptor
   CType ;;_pointer ;; *y
   CType ;;_cudnnTensorDescriptor_t ;; dy descriptor
   CType ;;_pointer ;; *dy
   CType ;;_cudnnTensorDescriptor_t ;; dhy descriptor
   CType ;;_pointer ;; *dhy
   CType ;;_cudnnTensorDescriptor_t ;; dcy descriptor
   CType ;;_pointer ;; *dcy
   CType ;;_cudnnFilterDescriptor_t ;; w descriptor
   CType ;;_pointer ;; pointer to w
   CType ;;_cudnnTensorDescriptor_t ;; hx descriptor
   CType ;;_pointer ;; pointer to hx
   CType ;;_cudnnTensorDescriptor_t ;; cx descriptor
   CType ;;_pointer ;; pointer to cx
   CType ;;_cudnnTensorDescriptor_t ;; dx descriptor
   CType ;;_pointer ;; pointer to dx
   CType ;;_cudnnTensorDescriptor_t ;; dhx descriptor
   CType ;;_pointer ;; pointer to dhx
   CType ;;_cudnnTensorDescriptor_t ;; dcx descriptor
   CType ;;_pointer ;; pointer to dcx
   CType ;;_pointer ;; pointer to workspace
   Exact-Nonnegative-Integer  ;;_size    ;; workspace size in bytes
   CType ;;_pointer ;; pointer to reserve space
   Exact-Nonnegative-Integer  ;;_size    ;; reserve size in bytes
   -> CType ;; _cudnn-status_t)
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

;(: dref-tensor-desc-ptr (-> CPointer CPointer))
;(define (dref-tensor-desc-ptr block )
;  (let
;      ([pref (ptr-ref block _cudnnTensorDescriptor_t)])
;    (print pref)
;    (cast pref CPointer)
;  ))



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
 cudnnDropoutGetStatesSize 
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
 dref-tensor-desc-ptr
 cuda-create-tensor-descriptr-ptr
 )


