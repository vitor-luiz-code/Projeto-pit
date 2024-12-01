#ifndef SCALAR_HPP
#define SCALAR_HPP
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <set>
#include <vector>
class scalar;
class value;

value operator ^(value *a, value b);
value operator *(value *a, value b);
value operator /(value *a, value b);
value operator +(value *a, value b);
value operator -(value *a, value b);
value operator *(value *a, scalar b);
value operator ^(value *a, scalar b);
value operator -(value *a, scalar b);
value operator +(value *a, scalar b);
value operator /(value *a, scalar b);

struct static_graph
{
  public:
    std::vector<value *>graph;
    std::set<value *> aux_visited;
    void  forward();
    void  backward();
    float get_data();
};

enum type_node {VAR, TEMP};

class value
{
  public :
  value *self         = this;
  float grad          = 0.0;
  float data          = 0.0;
  int verified        = 0;
  enum type_node type = TEMP;
  const char * label  = NULL;
  value * child[2]    = {NULL, NULL};
  void (*_backward)(value *) = NULL;
  void (*_forward)(value *)  = NULL;
  std::set<value *>depends   = {this->self};
  value();
  value(int);
  value(float);
  value(double);
  void build_topological(std::vector<value*> *top, std::set<value*> *visited, value * v);
  void clean_up(void);
  void backward();
  value  relu();
  value sin();
  value cos();
  value tanh();
  value exp();
  value log();
  void generate_visualization( const char *image_name);  
  struct static_graph * freeze_graph();


  value operator-();
  value operator^(scalar b);
  value operator^(value *b);
  value operator*(value *b);
  value operator*(scalar b);
  value operator-(scalar b);
  value operator-(value *b);
  value operator+(value *); 
  value operator+(scalar b);
  value operator/(scalar b);
  value operator/(value *b);
  value operator/(value b);
  value operator*(value b);
  value operator+(value b);
  value operator-(value b);
  value operator^(value b); 
};

class scalar
{
  public:
    float _value;
    scalar(int v);
    scalar(float v);
    scalar(double v);
    value operator-(); 
    value operator+(value *x);
    value operator*(value *x);
    value operator-(value *x);
    value operator/(value *x);
    value operator^(value *x);
    value operator+(value x);
    value operator*(value x);
    value operator-(value x);
    value operator/(value x);
    value operator^(value x);
    scalar operator+(scalar x);
    scalar operator*(scalar x);
    scalar operator-(scalar x);
    scalar operator/(scalar x);
    scalar operator^(scalar x);
};

std::vector<value  >vec_mul_vec(std::vector<value >, std::vector<value >);
std::vector <value > vec_mul_matrix(std::vector<value > in_1, std::vector<std::vector<value >>);
std::vector<std::vector<value >> mat_mul(std::vector<std::vector<value>> , std::vector<std::vector<value>>);
std::vector<std::vector<value >> mat_sum(std::vector<std::vector<value>> , std::vector<std::vector<value>> );
std::vector<std::vector<value >> mat_sub(std::vector<std::vector<value>> , std::vector<std::vector<value>>);
std::vector<std::vector<value >> mat_const_sub(std::vector<std::vector<value>>, float);
std::vector<std::vector<value >> mat_const_mul(std::vector<std::vector<value>> , float);
std::vector<std::vector<value >> mat_const_div(std::vector<std::vector<value>> , float);
std::vector<std::vector<value >> mat_const_sum(std::vector<std::vector<value>> , float);

void zero_grad_node_temp(value *v);
void zero_grad(value *v);
void print_graph(value *v);
void pow_backward3(value* v);
void pow_backward_xx(value* v);
void pow_backward(value*);
void pow_backward2(value*);
void sum_backward(value*);
void mul_backward(value*);
void relu_backward(value*);
void mul_forward(value*);
void minus_forward(value*);
void plus_forward(value*);
void div_forward(value*v);
void pot_forward(value*v);
void relu_forward(value *v);
void acumula_grad(value *v);
void clean_verified(value *v);
void sin_backward(value *v);
void cos_backward(value *v);
void tanh_backward(value *v);
void exp_backward(value *v);
void log_backward(value *v);
void sin_forward(value *v);
void cos_forward(value *v);
void tanh_forward(value *v);
void exp_forward(value *v);
void log_forward(value *v);

#endif
