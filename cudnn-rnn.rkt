#lang typed/racket
(require "cudnn-tensor.rkt")

;;;The dropout object
;;;TODO: notion of a workspace and timing functions

(struct cudnn-dropout
  ([desc : CPointer]
   [state-size : Integer]
   [states : Pointer]
   ))

(define (make-dropout handle dropout) : cudnn-dropout
  (let ([desc-ptr (get-dropout-ptr)]
	[states (get-pointer)]
	[stateSize (get-pointer)]
	[seed : Nonnegative-Integer 1337])
    (cudnnCreateDropoutDescriptor desc-ptr)
    (cudnnDropoutGetStatesSize cudnnHandle stateSize)
    (cudaMalloc states statesize)
    (cudnnSetDropoutDescriptor (dref-dropout desc-ptr)
			       handle
			       dropout
			       states
			       stateSize
			       seed)
    (cudnn-dropout (dref-dropout desc-ptr) states stateSize seed)))


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

;; Structure creation
(define (make-cudnn-rnn handle input-data input-size mini-batch hidden-size
			num-layers seq-length training-data)
  (let
      (
       [x (make-cudnn-input-tensor input-data)]
       [hx (make-cudnn-layer-tensor num-layers hidden-size mini-batch)]
       [cx (make-cudnn-layer-tensor num-layers hidden-size mini-batch)]
       [dx (make-cudnn-input-tensor seq-length input-size  mini-batch)]
       [dhx (make-cudnn-layer-tensor num-layers hidden-size mini-batch)]
       [dcx (make-cudnn-layer-tensor num-layers hidden-size mini-batch)]
       [y (make-cudnn-input-tensor training-data)]
       [hy (make-cudnn-layer-tensor num-layers hidden-size mini-batch)]
       [cy (make-cudnn-layer-tensor num-layers hidden-size mini-batch)]
       [dy (make-cudnn-input-tensor seq-length input-size  mini-batch)]
       [dhy (make-cudnn-layer-tensor num-layers hidden-size mini-batch)]
       [dcy (make-cudnn-layer-tensor num-layers hidden-size mini-batch)]
       [dropout-str (make-dropout handle dropout)]
       [rnn-desc (get-rnn-desc-ptr)]
       [w (get-ponter)]
       [dw (get-pointer)]
       [wDesc (get-filter-desc-pointer)]
       [dwDesc (get-filter-desc-pointer)]
       [dimW (get-int-array (list 3))]
       [weight-size (get-int-pointer)]
       [workspace (get-pointer)]
       [worksize (get-pointer)]
       [reservespace (get-pointer)]
       )
    (cudnnCreateRNNDescriptor rnn-desc)
    (cudnnSetRNNDescriptor (dref-rnn-desc rnn-desc)
			   hidden-size num-layers (cudnn-dropout-desc dropout-st)
			   CUDNN_LINEAR_INPUT CUDNN_UNIDIRECTIONAL
			   mode CUDNN_DATA_FLOAT)
    (cudnn-rnn-init-parameters handle rnn-desc w dw wDesc dwDesc   dimW weight-size)
    (cudnn-rnn-init-workspace workspace reserve-space worksize reserve-size handle
			      rnn-desc seq-length x)
    (cudnn-rnn-init-weights rnn)
    (cudnn-rnn-initialize-layers num-layers num-lin-layers handle
			   rnn-desc xdesc wdesc w)
    ))


;; Initialize the parameters
(define (cudnn-rnn-init-parameters handle rnn-desc w dw wDesc dwDesc  dimW weight-size)
  (cudnnCreateFilterDescriptor wDesc)
  (cudnnCreateFilterDescriptor dwDesc)
  (cudnnGetRNNParamsSize handle rnn-desc xDesc weight-size CUDNN_DATA_FLOAT)
  (set-array-data dimW 0 (floor (/ (dref-int weight-size) FLOAT-SIZE)))
  (set-array-data dimW 1 1)
  (set-array-data dimW 2 1)
  (cudnnSetFilterNdDescriptor wDesc CUDNN_DATA_FLOAT CUDNN_TENSOR_NCHW 3 dimW)
  (cudnnSetFilterNdDescriptor dwDesc CUDNN_DATA_FLOAT CUDNN_TENSOR_NCHW 3 dimW)
  (cudaMalloc w (dref-int weight-size))
  (cudaMalloc dw (dref-int weight-size)))
  
  
;; Initialize the workspace
(define (cudnn-rnn-init-workspace workspace reserve-space worksize reserve-size handle
			rnn-desc seq-length x)
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
  (cudaMalloc workspace worksize)
  (cudaMalloc reserve-space reserve-size))


;;; Initialize the weights
(define (cudnn-rnn-init-weights [rnn : cudnn-rnn])
  (if (cudnn-tensor? (rnn-hx rnn))
      (initGPUData (cudnn-tensor-gpu-ptr (cudnn-rnn-hx rnn))
		   (cudnn-tensor-size (cudnn-rnn-hx rnn))
		   1.0))
  (if (cudnn-tensor? (rnn-cx rnn))
      (initGPUData (cudnn-tensor-gpu-ptr (cudnn-rnn-cx rnn))
		   (cudnn-tensor-size (cudnn-rnn-cx rnn))
		   1.0))
  (if (cudnn-tensor? (rnn-dhy rnn))
      (initGPU (cudnn-tensor-gpu-ptr (cudnn-rnn-dhy rnn))
	       (cudnn-tensor-size (cudnn-rnn-dhy rnn))
	       1.0))
  (if (cudnn-tensor? (rnn-dcy rnn))
      (initGPU (cudnn-tensor-gpu-ptr (cudnn-rnn-dcy rnn))
	       (cudnn-tensor-size (cudnn-rnn-dcy rnn))
	       1.0)))

;; Run loop that initializes the layers
(define (cudnn-rnn-initialize-layers num-layers num-lin-layers handle
			   rnn-desc xdesc wdesc w)
  (for ([layer : Nonnegative-Integer (range num-layers)])
    (for ([lin-layer-id : Nonnegative-Integer (range num-lin-layers)])
      (cudnn-rnn-create-layer layer lin-layer-id handle
		    rnn-desc xdesc wdesc w))))
   

;; Inner call for creating the layer
(define (cudnn-rnn-create-layer layer lin-layer-id handle
		      rnn-desc xdesc wdesc w )
  (let
      (
       [lin-layer-mat-desc (get-lin-layer-desc)]
       [lin-layer-mat (get-pointer)]
       [dtype (get-pointer)]
       [format (get-pointer)]
       [nb-dims (get-pointer)]
       [lin-layer-bias-desc (get-pointer)]
       [filter-dim-a (get-int-pointer 3)]
       [linLayerBiasDesc (get-bias-desc)]
       [lin-layer-bias (get-float-ptr)]
       )
    (cudnnGetRNNLinLayerMatrixParams
     handle rnn-desc layer xdesc wdesc w lin-layer-id
     lin-layer-mat-desc lin-layer-mat)
    (cudnnGetFilterNdDescriptor
     lin-layer-mat-desc 3 dtype format nb-dims filter-dim-a)
    (initGPUData lin-layer-mat (foldl * filter-dim-a) (/ 1.0 (foldl * filter-dim-a)))
    (cudnnDestroyFilterDescriptor lin-layer-mat-desc)
    (cudnnCreateFilterDescriptor linLayerBiasDesc)
    (cudnnGetRNNLinLayerBiasParams handle rnn-desc layer
				   xdesc wdesc w lin-layer-id lin-lyaer-bias-desc
				   lin-layer-bias)
    (cudnnGetFilterNdDescriptor lin-lyaer-bias-desc
				3 dtype
				format nb-dims filter-dim-a)
    (initGPUData lin-layer-bias (foldl * filter-dim-a) 1.0)
    (cudnnDestroyFilterDescriptor)))
    
       

