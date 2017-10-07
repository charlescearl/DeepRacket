#lang racket

(require
 "simple-dynet-api.rkt")

(begin
  (init_dynet)
  (let*
      (
       [hidden-size 8]
       [cg (get_computation_graph)]
       [pc (get_parameter_collection)]
       [sgd (get_simple_sgd pc)]
       [w (add_parameters_shape_two cg 8 2 pc)]
       [q (add_parameters_shape_two cg 8 2 pc)]
       [yval (get_dynet_vector 1)]
       [xval (get_dynet_vector 2)]
       [xval_ptr (get_dynet_vect_ptr xval)]
       [yval_ptr (get_dynet_vect_ptr yval)]
       [y (create_outputs cg yval)]
       [x (create_n_inputs_vtr cg xval 2)]
       [v (add_parameters_shape_two cg 1 8 pc)]
       [b (add_parameters_shape_one cg 8 pc)]
       [a (add_parameters_shape_one cg 1 pc)]
       [h (create_tanh w x b)]
       [pred (create_pred v h a)]
       [loss_expr (create_loss pred y)]
       [loss 0.0]
       )
    (for ([epoch (in-range 30)])
      (for* ([mi (in-range 4)]
     	     [x1 (modulo mi 2)]
     	     [x2 (modulo (quotient mi 2) 2)])
	 (if (eq? x1 1)
	     (set_dynet_vptr xval_ptr 0 1.0)
	     (set_dynet_vptr xval_ptr 1 -1.0))
	 (if (not (eq? x2 x1))
	      (set_dynet_vptr yval_ptr 0 1.0)
	      (set_dynet_vptr yval_ptr 0 -1.0)
	      )
	 (set! loss (+ loss (get_scalar_loss cg loss_expr)))
	 
	 (do_backward_loss cg loss_expr)
	 (update_params sgd 1.0))
      (set! loss (/ loss 4.0))
      (display (format "Current loss is: ~a\n" loss)))))


