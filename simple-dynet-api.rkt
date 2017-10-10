#lang racket

(require
 racket/path
 ffi/unsafe
 ffi/unsafe/define
 )

;; Load POC library code
(define-ffi-definer define-dynet (ffi-lib "./libdynetsimple"))

;; ;; Initialize memory
(define-dynet init_dynet (_fun -> _void))


(define _computation_graph_pointer (_cpointer 'ComputationGraph))
(define _dynet_vector_pointer (_cpointer 'vector))
(define _parameter_collection_pointer (_cpointer 'ParameterCollection))
(define _expression_pointer (_cpointer 'Expression))
(define _sgd_trainer_pointer (_cpointer 'SimpleSGDTrainer))
(define _variable_index _uint)
(define _real _float )
(define _real_ptr (_cpointer '_real))


(define-cstruct _Expression ([pg _computation_graph_pointer]
                         [_i _variable_index]
                         [_graph_id _uint]
                         ))

;(define-dynet print_call_info (_fun _cpointer -> void))

;; ;; Get a parameter container
(define-dynet  get_parameter_collection (_fun ->  _parameter_collection_pointer))

;; ;; Get the computation graph
(define-dynet  get_computation_graph (_fun  -> _computation_graph_pointer ))

(define-dynet  get_simple_sgd (_fun _parameter_collection_pointer ->  _sgd_trainer_pointer ))

;; ;; Add a matrix parameter
(define-dynet add_parameters_shape_two (_fun _computation_graph_pointer _int _int _parameter_collection_pointer -> _Expression))

;; ;; Add a vector parameter
(define-dynet add_parameters_shape_one (_fun _computation_graph_pointer  _int _parameter_collection_pointer -> _Expression))

;; ;;Add an input parameter
(define-dynet create_n_inputs_vtr (_fun _computation_graph_pointer  _dynet_vector_pointer _int -> _Expression))

(define-dynet create_n_inputs (_fun _computation_graph_pointer  _int -> _Expression))

;; ;;Add a scalar output
(define-dynet create_outputs (_fun _computation_graph_pointer _dynet_vector_pointer -> _Expression))

;; ;;Add a graph computation with tanh nonlinearity
(define-dynet create_tanh (_fun _Expression _Expression _Expression -> _Expression))

;; ;;Add a graph computation
(define-dynet create_pred (_fun _Expression _Expression _Expression -> _Expression))

;; ;;Add a loss expression
(define-dynet create_loss (_fun _Expression _Expression -> _Expression))

;; ;;Get a scalar loss
(define-dynet get_scalar_loss (_fun _computation_graph_pointer _Expression -> _double))

;; ;;Run loss backward to get gradients
(define-dynet do_backward_loss (_fun _computation_graph_pointer _Expression -> _int))

(define-dynet get_dynet_vector ( _fun _int -> _dynet_vector_pointer))

(define-dynet update_params (_fun _sgd_trainer_pointer _float -> _int))

(define-dynet get_vect_pointer (_fun -> _real_ptr))

(define-dynet get_dynet_vect_val (_fun _dynet_vector_pointer _uint -> _real))

(define-dynet get_dynet_vect_ptr (_fun _dynet_vector_pointer -> _real_ptr))

(define (set_dynet_vptr ptr idx val)
  (ptr-set! ptr _real idx val))
(module+ test
  (require rackunit)
  (let* ([yval (get_dynet_vector 1)]
         [yval-ptr (get_dynet_vect_ptr yval)])
    (display "Testing setting of vector pointers.\n")
    (set_dynet_vptr yval-ptr 0 1.0)
    (check-equal? (get_dynet_vect_val yval 0) 1.0)
    (set_dynet_vptr yval-ptr 0 11.0)
    (check-equal? (get_dynet_vect_val yval 0) 11.0)
    (display "Setting vector values works ok.\n")))

 (provide
  init_dynet
  get_computation_graph
  get_parameter_collection
  add_parameters_shape_two
  add_parameters_shape_one
  create_n_inputs
  create_outputs
  create_tanh
  create_pred
  create_loss
  get_scalar_loss
  do_backward_loss
  update_params
  get_dynet_vector
  get_simple_sgd
  create_n_inputs_vtr
  get_dynet_vector
  get_vect_pointer
  get_dynet_vect_ptr
  set_dynet_vptr
  get_dynet_vect_val
  )
;;  )
