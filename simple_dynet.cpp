#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"

#include <iostream>

using namespace std;
using namespace dynet;

extern "C" void init_dynet(){
  int argc = 0;
  char **argv;
  dynet::initialize(argc, argv);
}

extern "C" ParameterCollection * get_parameter_collection(){
  return new ParameterCollection();
}

extern "C" ComputationGraph * get_computation_graph(){
  return new ComputationGraph();
}

extern "C" SimpleSGDTrainer * get_simple_sgd(ParameterCollection* m){
  return new SimpleSGDTrainer(*m);
}

extern "C" struct SimpleExpression {
  ComputationGraph *pg;
  VariableIndex i;
};

extern "C" void print_call_info(ComputationGraph * cg){
  printf("Taking a computation graph");
}

extern "C" Expression get_expression(ComputationGraph * cg, const Dim&d,
				       ParameterCollection pc){
  printf("Adding a new expression\n");
  return parameter(*cg, pc.add_parameters(d));
  //printf("Cmopleted the add\n");
  //return new Expression(cg, expr.i);
}

extern "C" Expression add_parameters_shape_two(ComputationGraph * cg, int dimA,
						 int dimB,
						 ParameterCollection *pc){
  printf("Adding a new expression of shape %d %d\n", dimA, dimB);
  Parameter p_W = pc->add_parameters({(unsigned int)dimA, (unsigned int)dimB});
  return parameter(*cg, p_W);

  //return get_expression(cg, {(unsigned int)dimA, (unsigned int)dimB}, *pc);
}

extern "C" Expression add_parameters_shape_one(ComputationGraph * cg, int dimA,
						 ParameterCollection *pc){
  printf("Adding a new expression of shape %d\n", dimA);
  Parameter p_b = pc->add_parameters({(unsigned int)dimA});
  return parameter(*cg, p_b);
  //return get_expression(cg, {(unsigned int)dimA}, *pc);
}

extern "C" Expression create_n_inputs(ComputationGraph * cg, int len){
  // // Set x_values to change the inputs to the network.
  //vector<dynet::real> x_values(len);
  vector<dynet::real>* x_values;
  x_values = new vector<dynet::real>(len);
  return input(*cg, {(unsigned int)len}, x_values);
  //return new Expression(cg, x.i);
}

extern "C" Expression create_n_inputs_vtr(ComputationGraph * cg, vector<dynet::real>* x_values, int len){
  // // Set x_values to change the inputs to the network.
  //vector<dynet::real> x_values(len);
  return input(*cg, {(unsigned int)len}, x_values);
  //return new Expression(cg, x.i);
}

extern "C" dynet::real * get_dynet_vect_ptr(vector<dynet::real>* x_values ){
  // // Set x_values to change the inputs to the network.
  //vector<dynet::real> x_values(len);
  return x_values->data();
  //return new Expression(cg, x.i);
}

extern "C" dynet::real get_dynet_vect_val(vector<dynet::real>* x_values, int idx ){
  return x_values->at(idx);
}

extern "C" vector<dynet::real>* get_dynet_vector(int size){
  return new vector<dynet::real>(size);
}

extern "C" void set_dynet_real(dynet::real * v, double val){
  *v = val;
}

extern "C" void set_dynet_vect(vector<dynet::real> * vtr, int idx, double val){
  (*vtr)[idx] = val;
}

extern "C" Expression create_outputs(ComputationGraph * cg, vector<dynet::real> * vtr){
  // // Set x_values to change the inputs to the network.
  // dynet::real y_value;  // Set y_value to change the target output.
  printf("Adding a new expression of for y \n");
  return input(*cg, &((*vtr)[0]));
  //printf("Finished input expression \n");
  //return new Expression(cg, y.i);
}

extern "C" Expression create_outputs_val(ComputationGraph * cg, dynet::real* y_value){
  // // Set x_values to change the inputs to the network.
  return input(*cg, y_value);
}

extern "C" Expression create_tanh(Expression W, Expression x, Expression b){
  return tanh(W*x + b);
  //return new Expression(h.pg, h.i);

}

extern "C" Expression create_pred(Expression V, Expression h, Expression a){
  return V*h + a;
  //return new Expression(y_pred.pg, y_pred.i);
}

extern "C" Expression create_loss(Expression y_pred, Expression y){
  return squared_distance(y_pred, y);
  //return new Expression(loss_expr.pg, loss_expr.i);
}

extern "C" double get_scalar_loss(ComputationGraph *cg, Expression loss_expr){
  return as_scalar(cg->forward(loss_expr));
}

extern "C" int do_backward_loss(ComputationGraph * cg, Expression loss_expr){
  cg->backward(loss_expr);
  return 0;
}

extern "C" dynet::real * get_vect_pointer(){
  vector<dynet::real> x_values(2);
  return x_values.data();
}

extern "C" int update_params(SimpleSGDTrainer * sgd, float rate){
  sgd->update(rate);
  return 0;
}
