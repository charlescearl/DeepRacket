;;Typed tensors
#lang typed/racket
(require/typed
  ffi/unsafe
  [#:opaque CPointer cpointer?]
  [#:opaque CType ctype?]
  [_double CType]
  [_float CType]
  [_uintptr CType]
  [_pointer CType]
  [_int CType]
  [_size CType]
  [flvector->cpointer (FlVector -> CPointer)]
  [ptr-ref (CPointer CType Exact-Nonnegative-Integer -> Any)]
  [ptr-set! (CPointer CType Exact-Nonnegative-Integer Any -> Void)]
  [ctype-sizeof (CType -> Exact-Nonnegative-Integer)]
  
  )
(require/typed
 ffi/unsafe/cvector
 [#:opaque CVector cvector?]
 [cvector (CType Any -> CVector)])
(require math/base)
(require math/array)

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
 [cudaMalloc (CType Exact-Nonnegative-Integer -> CType)]
 [cudnnSetTensorNdDescriptor (CType CType  CType  CType CType -> CType)]
 [cudnnCreateTensorDescriptor (CType -> CType)]
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

;; Return pointer for start of an array
;;(array->cptr (array #[#[1.0 2.24554] #[3.2323 4.001] #[5.454 5.86868]]))
(: array->cptr (-> (Array Float) CPointer))
(define (array->cptr arr)
  (flvector->cpointer (flarray-data (array->flarray arr))))

;; Utility for getting a referenced value from an array
(: ptr-array-ref (-> CPointer Exact-Nonnegative-Integer Flonum))
(define (ptr-array-ref arr-ptr idx)
  (cast (ptr-ref arr-ptr _double idx) Flonum))

;; Utility for getting a referenced value from an array
(: ptr-array-set! (-> CPointer Exact-Nonnegative-Integer Flonum Void))
(define (ptr-array-set! arr-ptr idx val)
  (ptr-set! arr-ptr _double idx val))

;; define Tensor struct
;; Provide an array, then store the FlVector of it and pointer
;; Create a gpu vector of size for the representation
;; We then memcpy between this vector and gpu vector to return
;; values.
;; desc stores the tensordescriptor

(struct tensor ([in-vect : FlVector]
                [src-ptr : CPointer]
                [gpu-vect : CPointer]
                [desc : CType]))


;; Define the RNN
(struct rnn
  ([x : tensor]
   [y : tensor]
   [dx : tensor]
   [dy : tensor]
   [hx : tensor]
   [hy : tensor]
   [cy : tensor]
   [cx : tensor]
   [dhx : tensor]
   [dcx : tensor]
   [dhy : tensor]
   [dcy : tensor]
   ))

(: FLOAT-SIZE Exact-Nonnegative-Integer)
(define FLOAT-SIZE (ctype-sizeof _float))

;; Utility for getting a referenced value from an array
;; (: allocate-layer-mem (-> CType Exact-Nonnegative-Integer CType))
;; (define (allocate-layer-mem ptr size)
;;   (cudaMalloc ptr size))

;; ;; Utility for getting a referenced value from an array
;; (: allocate-layer-array (-> CType Exact-Nonnegative-Integer Exact-Nonnegative-Integer Exact-Nonnegative-Integer CType))
;; (define (allocate-layer-array ptr seq-length input-size mini-batch)
;;   (allocate-layer-mem ptr (* seq-length input-size mini-batch FLOAT-SIZE)))

;; ;; Allocate array of tensor descriptors
;; ;(: allocate-tensor-descriptors (-> Exact-Nonnegative-Integer CVector))
;; ;(define (allocate-tensor-descriptors seq-len)
;; ;  (cvector _cudnnTensorDescriptor_t seq-len))


;; (provide tensor rnn ptr-array-ref array->cptr)
(provide
 tensor
 rnn
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
  )
