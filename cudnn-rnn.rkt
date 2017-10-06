#lang typed/racket
(require "cudnn-tensor.rkt")
(require "cuda-api.rkt") ; for get-pointer
(require "cudnn-api.rkt")
(require "mem-utils.rkt")
(require "ffi-functional.rkt")
(require "cudnn-dropout.rkt")

;; Define the RNN
(struct cudnn-rnn
  ([x : cudnn-tensor]
   [y : cudnn-tensor]
   [dx : cudnn-tensor]
   [dy : cudnn-tensor]
   [hx : cudnn-tensor]
   [hy : cudnn-tensor]
   [cy : cudnn-tensor]
   [cx : cudnn-tensor]
   [dhx : cudnn-tensor]
   [dcx : cudnn-tensor]
   [dhy : cudnn-tensor]
   [dcy : cudnn-tensor]
   [dropout-str : cudnn-dropout]
   [rnn-desc : CPointer]
   ))

;; Initialize the workspace
(define (cudnn-rnn-init-workspace [workspace : CPointer]
				  [reserve-space : CPointer]
				  [worksize : CPointer]
				  [reserve-size : CPointer]
				  [handle : CPointer]
				  [rnn-desc : CPointer]
				  [seq-length : Nonnegative-Integer]
				  [x : cudnn-tensor])
  (cudnnGetRNNWorkspaceSize
   handle
   rnn-desc
   seq-length
   (cudnn-tensor-desc x)
   worksize)
  (cudnnGetRNNTrainingReserveSize
    handle
   rnn-desc
   seq-length
   (cudnn-tensor-desc x)
   reserve-size)
  (cudaMalloc workspace (dref-int-ptr worksize))
  (cudaMalloc reserve-space (dref-int-ptr reserve-size)))


;;; Initialize the weights
(define (cudnn-rnn-init-weights [hx : cudnn-tensor] [cx : cudnn-tensor]
				[dhy : cudnn-tensor] [dcy : cudnn-tensor])
  (when (cudnn-tensor? hx)
      (initGPUData (cudnn-tensor-gpu-ptr hx)
		   (cudnn-tensor-size hx)
		   1.0))
  (when (cudnn-tensor? cx)
      (initGPUData (cudnn-tensor-gpu-ptr cx)
		   (cudnn-tensor-size cx)
		   1.0))
  (when (cudnn-tensor? dhy)
      (initGPUData (cudnn-tensor-gpu-ptr dhy)
	       (cudnn-tensor-size dhy)
	       1.0))
  (when (cudnn-tensor? dcy)
      (initGPUData (cudnn-tensor-gpu-ptr dcy)
	       (cudnn-tensor-size dcy)
	       1.0)))

;; Run loop that initializes the layers
(define (cudnn-rnn-initialize-layers [num-layers : Nonnegative-Integer]
				     [num-lin-layers : Nonnegative-Integer]
				     [handle : CPointer]
				     [rnn-desc : CPointer ]
				     [xdesc : CPointer]
				     [wdesc : CPointer]
				     [w : CPointer ] )
  (for ([layer : Nonnegative-Integer (range num-layers)])
    (for ([lin-layer-id : Nonnegative-Integer (range num-lin-layers)])
      (cudnn-rnn-create-layer layer lin-layer-id handle
		    rnn-desc xdesc wdesc w))))

;; Inner call for creating the layer
(define (cudnn-rnn-create-layer [layer : Nonnegative-Integer]
				[lin-layer-id : Nonnegative-Integer]
				[handle : CPointer]
			 
				[rnn-desc : CPointer]
				[xdesc : CPointer]
				[wdesc : CPointer]
				[w : CPointer ] ) : Symbol
  'true)
  ;; (let
  ;;     (
  ;;      [lin-layer-mat-desc (get-lin-layer-desc)]
  ;;      [lin-layer-mat (get-pointer)]
  ;;      [dtype (get-pointer)]
  ;;      [format (get-pointer)]
  ;;      [nb-dims (get-int-pointer)]
  ;;      [lin-layer-bias-desc (get-pointer)]
  ;;      [filter-dim-a (get-int-pointer 3)]
  ;;      [linLayerBiasDesc (get-bias-desc)]
  ;;      [lin-layer-bias (get-float-ptr)]
  ;;      )
  ;;   (cudnnGetRNNLinLayerMatrixParams
  ;;    handle rnn-desc layer xdesc wdesc w lin-layer-id
  ;;    lin-layer-mat-desc lin-layer-mat)
  ;;   (cudnnGetFilterNdDescriptor
  ;;    lin-layer-mat-desc 3 dtype format nb-dims filter-dim-a)
  ;;   (initGPUData lin-layer-mat (foldl * filter-dim-a) (/ 1.0 (foldl * filter-dim-a)))
  ;;   (cudnnDestroyFilterDescriptor lin-layer-mat-desc)
  ;;   (cudnnCreateFilterDescriptor linLayerBiasDesc)
  ;;   (cudnnGetRNNLinLayerBiasParams handle rnn-desc layer
  ;; 				   xdesc wdesc w lin-layer-id lin-lyaer-bias-desc
  ;; 				   lin-layer-bias)
  ;;   (cudnnGetFilterNdDescriptor lin-lyaer-bias-desc
  ;; 				3 dtype
  ;; 				format nb-dims filter-dim-a)
  ;;   (initGPUData lin-layer-bias (foldl * filter-dim-a) 1.0)
  ;;   (cudnnDestroyFilterDescriptor)))
    
       

