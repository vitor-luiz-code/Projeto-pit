
#include "scalar.hpp"

value :: value (){}
value :: value(int v)      :data((float)v), type(VAR){};
value :: value(float v)    :data((float)v), type(VAR){};
value :: value(double v)   :data((float)v), type(VAR){};
scalar::scalar(int v)    {_value = (float)v;}
scalar::scalar(float v)  {_value = (float)v;}
scalar::scalar(double v) {_value  =(float)v;}


#define pointer_or_value(init,op,end) initopend

#define init_define_func_operators(child_exp1,child_exp0, cond_alloc,out, aux, var_a, var_b, \
data_exp, forward_func, backward_func,\
var_a_arrow_or_point,var_b_arrow_or_point, _label,cond_backward, value_, _self)\
do {\
    out           = new value();\
    if (cond_alloc) \
    { aux = new value();\
      aux->self = aux;\
      out->child_exp0 = aux;\
      aux->data = pointer_or_value(var_a, var_a_arrow_or_point,value_);\
      out->self     = out;\
      out->child_exp1 = pointer_or_value(var_b,var_b_arrow_or_point,_self);\
    }else \
    { out->child[0] = pointer_or_value(var_a,var_a_arrow_or_point, _self);\
      out->child[1] = pointer_or_value(var_b,var_b_arrow_or_point, _self);\
    } \
      out->label = _label;\
      if (cond_backward)\
        out->_backward = backward_func;\
      out->_forward = forward_func;\
      out->depends.insert(out->child[1]);\
      out->depends.insert(out->child[0]);\
      out->data      = data_exp;\
    } while (0);\

void static_graph :: backward()
{
  int len = this->graph.size()-1;
  this->graph[len]->grad = 1.0;
  while (len >= 0)
  { value * aux = this->graph[len--];
    if (aux->_backward)
      aux->_backward(aux);
  }

  zero_grad_node_temp(this->graph[this->graph.size()-1]);

}
void static_graph :: forward()
{
  int len = this->graph.size();
  int i = 0;
  while (i < len)
  {
    value * aux = this->graph[i++];
    if (aux->_forward)
      aux->_forward(aux);
  }
}

float static_graph :: get_data()
{
  int len = this->graph.size()-1;
  return this->graph[len]->data;
}
struct static_graph * value :: freeze_graph()
{ 
  struct static_graph *graph = new static_graph();
  build_topological(&graph->graph, &graph->aux_visited, this->self);
  return graph;
}
void value ::generate_visualization( const char *image_name)
{
  std::vector<value *>  node;
  std::set<value *>   verified;
  build_topological(&node, &verified, this);

  FILE * fd = fopen(image_name, "w");
  
  fprintf(fd, "digraph G{\n");
  char alpha = 97;
  char alpha2 = 0; 
  for (int i = 0; i < node.size(); i++)
  {  for (int j = 0; j < node.size(); j++)
     { if (node[i]->child[0] == node[j])
      { //printf("[%s]\n", node[i]->label);
        fprintf(fd,"%s_%i->",node[i]->label,alpha+i);
        char t[] = {(char)(alpha+j),'\0'};
        fprintf(fd,"%s_%i;", node[j]->label  ? node[j]->label :t, alpha+j);
      }
      if (node[i]->child[1] == node[j])
      { //printf("[%s]\n", node[i]->label);
        fprintf(fd,"%s_%i->",node[i]->label,alpha+i);
        char t[] = {(char)(alpha+j),'\0'};
        fprintf(fd,"%s_%i;", node[j]->label  ? node[j]->label :t, alpha+j);
      }
    }    
    //fprintf(fd, ";\n");    
  }
  fprintf(fd,"}");
  fclose(fd);
}

void get_depends(std::set<value *> *depends, value *v)
{
  if (v)
  { if (!depends->count(v))
      depends->insert(v);    
    if (v->child[0])  
      get_depends(depends,v->child[0]);
    if (v->child[1])  
      get_depends(depends, v->child[1]); 
  }
}

void value :: clean_up(void)
{
  std::set<value*> verified;
  std::vector<value*> topological;
  build_topological(&topological, &verified, this->child[0]);
  build_topological(&topological, &verified, this->child[1]);
  
  int len = topological.size();
  int i = 0;
  while (i <  len)
  { value *v = topological[i];
    if (v)
      delete v;
    i++;  
  }    
}

void value :: build_topological(std::vector<value*> *top, std::set<value*> *verified, value*v)
  { 
    if (!v)
    { printf("value NULL em build_topological");
      return ;
    }
    if(!verified->count(v))
    { verified->insert(v);
      if (v->child[0])
        build_topological(top, verified, v->child[0]);
      if (v->child[1])
        build_topological(top, verified, v->child[1]);
      top->push_back(v);
    }
  }
  void value :: backward()
  {
    std::set<value*> verified;
    std::vector<value*> topological;
    build_topological(&topological, &verified, this->self);
    int len = topological.size()-1;

    this->self->grad = 1.0;

    while (len >= 0)
    { value *v = topological[len--];
      if (v->_backward)
        v->_backward(v);
    }

    zero_grad_node_temp(this->self);
  }
  


  value value :: sin()
  {
    value *out    = new value();
    out->self     = out;
    out->child[0] = this->self;
    out->child[1] = NULL;
    out->label    = "sin";
    out->data = sinf(this->data);
    out->_backward = sin_backward;
    out->_forward  = sin_forward;
    out->depends.insert(out->child[0]);
    out->depends.insert(out->child[1]);
    return *out;
  }

  value value :: cos()
  {
    value *out    = new value();
    out->self     = out;
    out->child[0] = this->self;
    out->child[1] = NULL;
    out->label    = "cos";
    out->data = cosf(this->data);
    out->_backward = cos_backward;
    out->_forward  = cos_forward;
    out->depends.insert(out->child[0]);
    out->depends.insert(out->child[1]);
    return *out;

  }

  value value :: tanh()
  {
    value *out    = new value();
    out->self     = out;
    out->child[0] = this->self;
    out->child[1] = NULL;
    out->label    = "tanh";
    out->data = tanhf(this->data);
    out->_backward = tanh_backward;
    out->_forward  = tanh_forward;
    out->depends.insert(out->child[0]);
    out->depends.insert(out->child[1]);
    return *out;
  }

  value value :: exp()
  {
    value *out     = new value();
    out->self      = out;
    out->child[0]  = this->self;
    out->child[1]  = NULL;
    out->label     = "exp";
    out->data      = expf(this->data);
    out->_backward = exp_backward;
    out->_forward  = exp_forward;
    out->depends.insert(out->child[0]);
    out->depends.insert(out->child[1]);
    return *out;

  }

  value value ::log()
  {
    value *out     = new value();
    out->self      = out;
    out->child[0]  = this->self;
    out->child[1]  = NULL;
    out->label     = "log";
    out->data      = logf(this->data);
    out->_backward = log_backward;
    out->_forward  = log_forward;
    out->depends.insert(out->child[0]);
    out->depends.insert(out->child[1]);
    return *out;
  }
  value value :: operator-(value *b)
  {   scalar _const =scalar(-1);  
      return *this + (b*_const);
  } 
  value operator ^(value *a, value b)
  {

    value *out    = new value();
    out->self     = out;
    out->child[0] = a->self;
    out->child[1] = b.self;
    out->label = "pot";
    out->_forward = pot_forward;
    if (b.self == a->self)
      out->_backward = pow_backward_xx;
    else
      out->_backward = pow_backward;
    out->depends.insert(out->child[0]);
    out->depends.insert(out->child[1]);
    out->data      = pow(a->data, b.data);
    return *out;
  }
  
  value  operator * (value *a, value b)
  { /*
    value *out    = new value();
    out->self     = out;
    out->_backward = mul_backward;
    out->child[0] = a->self;
    out->child[1] = b.self;
  
    out->data     = a->data * b.data;
    
    get_depends(&out->depends,out->child[0]);
    get_depends(&out->depends,out->child[1]);
    */
    return b * a; 
  }
  
  value value ::operator +(value *b)
  {
    value *out    = new value();
    out->self     = out;
    out->child[0] = this->self;
    out->child[1] = b->self;
    out->_forward = plus_forward;
    out->label = "plus";
    out->data     = this->data + b->data;
    out->_backward = sum_backward;
       out->depends.insert(out->child[0]);
    out->depends.insert(out->child[1]);
    return *out;
  }
  value operator /(value *a, value b)
  {
    scalar _const = scalar(-1);
    return a *(b ^  _const) ;
  }
  value operator +(value *a, value b)
  {
    return b + a;
  }
  value operator -(value *a, value b)
  {
    scalar _const = scalar(-1); 
    return (b * _const) + a ;
  }
  value value :: operator +(scalar b)
  {
    value *out = new value();
    out->self = out;
    value *bb  = new value();
    bb->self = bb;
    out->_forward = plus_forward;
    bb->data = b._value;
    out->label = "plus";
    out->data = data + b._value;
    out->child[0] = this->self;
    out->child[1] = bb;
    out->_backward = sum_backward;
     out->depends.insert(out->child[0]);
    out->depends.insert(out->child[1]);
    return *out;
    
  }
  
 value value :: relu()
  { 
    value *out = NULL;
    out = new value();
    out->data = data > 0 ? data : 0;
    out->child[0] = this->self;
    out->child[1] = NULL;
    out->label = "reLu";
    out->_forward = relu_forward;
    out->_backward = relu_backward;
    out->depends.insert(out->child[0]);
    return *out;
  }
  
  value value :: operator -()
  { this->data  = -data;
    return *this;
  }
  
 value value:: operator ^(scalar b)
  {
    value *out = NULL,*a = NULL, *bb = NULL;
    out= new value();
    bb =  new value();
    bb->data = b._value;
    out->label = "pot";
    out->_forward = pot_forward;
    out->child[0] = this->self;
    out->child[1]= bb;
    out->data  = pow(data, b._value);
    out->_backward = pow_backward2;
    out->depends.insert(out->child[0]);
    out->depends.insert(out->child[1]);
    return *out;
  }
  
 value value :: operator ^(value *b)
  { 
    value *out;
    out = new value();
    out->child[0] = this->self;
    out->child[1] = b->self;
    out->label = "pot";
    out->_forward = pot_forward;
    if (this->self == b->self)
      out->_backward = pow_backward_xx;
    else
      out->_backward = pow_backward;
    out->data  = powf(data ,b->data );
    out->depends.insert(out->child[0]);
    out->depends.insert(out->child[1]);
    return *out;
  }
  

 value value :: operator *(value *b)
  { 
    value *out = NULL;
    out = new value();
    out->self = out;
    out->child[0] = this->self;
    out->child[1]= b->self;
    out->label = "mul";
    out->_forward = mul_forward;
    out->data  = data * (b->data);
    out->_backward = mul_backward;
    out->depends.insert(out->child[0]);
    out->depends.insert(out->child[1]);
    return *out;
  }
  
 value value :: operator *(scalar b)
  {
    return b * this;
  }
  
 value value :: operator -(scalar b)
  {
    return -b + this;
  }
  

  value operator * (value *a, scalar b)
  {
    value *out = new value();
    out->self = out;
    value *bb  = new value();
    bb->self = bb;
    bb->data = b._value;
    out->_forward = mul_forward;
    out->label = "mul";
     out->child[0] = a->self;
    out->child[1] = bb;
    out->data = a->data * bb->data;
    out->depends.insert(out->child[0]);
    out->depends.insert(out->child[1]);
    out->_backward = mul_backward;
    return *out;
  }
  
value operator ^(value *a, scalar b)
{
  value *out = new value();
  out->self  = out;
  value *bb = new value();
  out->child[0] = a->self;
  out->child[1] = bb;
  bb->self = bb;
  out->label = "pot";
  out->_forward = pot_forward;
  bb->data = b._value;
  out->data = pow(a->data, b._value);
  out->_backward = pow_backward2;
   out->depends.insert(out->child[0]);
    out->depends.insert(out->child[1]);
  return *out;
}

value operator -(value *a, scalar b)
{ 
  scalar _const = scalar(-1);  
  return -b + a;
}
value operator +(value *a, scalar b)
  {
    return b + a;
  }

value operator /(value *a, scalar b)
  {
    scalar _const = scalar(-1);
    return a * (b ^  _const);
  }

  value value :: operator /(scalar b)
  { 
    value *_const = new value();
    _const->data = -1;
    return this * (b ^ _const);
  }
  
 value value :: operator /(value *b)
  { 
    scalar _const  = scalar(-1); 
    return this * (b ^ _const);
  }
  
  value value ::operator/(value b)
  { 
    scalar _const = scalar(-1);
    return this * (b ^ _const);  
  }
  value value :: operator*(value b)
  {
    return b * this;
  }
  value value  ::operator+(value b)
  {
    return b + this;
  }
  value value :: operator-(value b)
  {
    return -b + this; 
  }
  value value ::operator^(value b)
  {
    return this ^ b; 
  }


value scalar :: operator+(value *x)
{
  value *a   = new value();
  a->self = a;
  value *out = new value();
  out->self = out;
  a->data = _value;
  out->label = "plus";
  out->_forward = plus_forward;
  out->child[0] = a;
  out->child[1] = x->self;
  out->data = _value + x->data ;
  out->_backward = sum_backward;
  out->depends.insert(out->child[0]);
    out->depends.insert(out->child[1]);
  return *out;
}
 value scalar ::operator*(value *x)
{
  value *a   = new value();
  a->self = a;
  value *out = new value();
  out->self = out;
  a->data = _value;
  out->label = "mul";
  out->_forward = mul_forward;
  out->child[0] = a;
  out->child[1] = x->self;
  out->data = x->data * _value;
  out->_backward = mul_backward;
 out->depends.insert(out->child[0]);
    out->depends.insert(out->child[1]);
  return *out;
}

value scalar :: operator -()
{
  value *out =  new value();
  out->data = -this->_value;
  return *out;
}
value scalar ::operator-(value *x)
{ 
  scalar _const = scalar(-1);
  return *this - (*x);
}


value scalar :: operator/(value *x)
{
  scalar _const = scalar(-1);
  return *this * (x ^ _const);
}


value scalar :: operator ^ (value *x)
{
  value * out = new value();
  out->self = out;
  value * a   = new value();
  a->self = a;
  a->data     = _value;
  out->_forward = pot_forward;
  out->data   = pow(_value, x->data);
  out->child[0] = a;
  out->child[1] = x->self;
  out->label = "pot";
    out->_backward = pow_backward3;
   out->depends.insert(out->child[0]);
    out->depends.insert(out->child[1]);
  return *out;
}
value scalar :: operator+(value x)
{
  value * out = new value();
  out->self = out;
  value * a   = new value();
  a->self = a;
  a->data     = _value;
  out->_forward = plus_forward;
  out->data   = _value + x.data;
  out->label = "plus";
  out->child[0] = a;
  out->child[1] = x.self;
  out->_backward = sum_backward;
 out->depends.insert(out->child[0]);
    out->depends.insert(out->child[1]);
  return *out;
}
value scalar :: operator*(value x)
{

  value * out = new value();
  out->self = out;
  value * a   = new value();
  a->self = a;
  a->data     = _value;
  out->_forward = mul_forward;
  out->data   = _value * x.data;
  out->label = "mul";
  out->child[0] = a->self;
  out->child[1] = x.self;
  out->_backward = mul_backward;
 out->depends.insert(out->child[0]);
    out->depends.insert(out->child[1]);

 return *out;
}
value scalar :: operator-(value x)
{ /*
  value * out = new value();
  out->self = out;
  value * a   = new value();
  a->self = a;
  a->data     = _value;
  out->data   = _value - x.data;
  out->child[0] = a->self;
  out->child[1] = x.self;
  out->_backward = sum_backward;
  */
  scalar _const = scalar(-1);
  return *this + (x *_const);
}
value scalar :: operator/(value x)
{ 
  /*
  value * out = new value();
  out->self = out;
  value * a   = new value();
  a->self = a;
  a->data     = _value;
  out->data   = pow(_value, x->data);
  out->child[0] = a;
  out->child[1] = x;
  out->_backward = pow_backward;
  */
  scalar _const = scalar(-1);
  return (x ^ _const) * *this;
}

value scalar :: operator^ (value x)
{
  value * out = new value();
  out->self = out;
  value * a   = new value();
  a->self = a;
  out->label = "pot";
  out->_forward = pot_forward;
  a->data     = _value;
  out->data   = pow(_value, x.data);
  out->child[0] = a;
  out->child[1] = x.self;
  out->_backward = pow_backward3;
 out->depends.insert(out->child[0]);
  out->depends.insert(out->child[1]);
  return *out;
}

scalar scalar::operator+(scalar x)
{
  return scalar(x._value + _value);
}

scalar scalar::operator*(scalar x)
{
  return scalar(x._value * _value);
}

scalar scalar::operator-(scalar x)
{
  return scalar(_value - x._value);
}

scalar scalar::operator/(scalar x)
{
  return scalar(_value / x._value);
}

scalar scalar::operator^(scalar b)
{
  return scalar(pow(_value,b._value));
}

void pow_backward_xx(value* v)
{ 
  if (v)
    if (v->child[0] && v->child[1])
    {
        v->child[1]->grad += (1 + logf(v->child[0]->data)) * (powf(v->child[0]->data, v->child[1]->data) * v->grad);
    }
}

void pow_backward(value* v)
{  
  if (v)
  { 
    if (v->child[0] && v->child[1])
    {
      v->child[0]->grad += v->child[1]->data * pow(v->child[0]->data, v->child[1]->data-1) * v->grad;
      v->child[1]->grad += (logf(v->child[0]->data) * powf(v->child[0]->data, v->child[1]->data)) * v->grad;
    }
  }
}
void pow_backward3(value* v)
{  if (v)
    if (v->child[0] && v->child[1])
    {
        v->child[1]->grad += logf(v->child[0]->data) * (powf(v->child[0]->data, v->child[1]->data) * v->grad);
    }
}
void pow_backward2(value* v)
{ 
  if (v)
    if(v->child[0] && v->child[1])
      v->child[0]->grad += v->child[1]->data * powf(v->child[0]->data, v->child[1]->data-1) * v->grad;
      
}

void sum_backward(value* v)
{   
  if (v)
  { if (v->child[0])
      (v->child[0]->grad) += v->grad;
    if (v->child[1])
      (v->child[1]->grad) += v->grad;
  }      
}

void mul_backward(value*v)
{   
  if (v)  
  { 

    if (v->child[0] && v->child[1])
      {
            (v->child[0]->grad) += v->child[1]->data * v->grad;
            (v->child[1]->grad) += v->child[0]->data * v->grad;
      }
    }

}


void relu_backward(value*v)
{
  if (v)
    if (v->child[0])
      (v->child[0]->grad) += (v->data > 0) * v->grad;
}

void print_graph(value *v)
{
  if (v)
  {
    print_graph(v->child[0]);
    printf("[end %p %s data = %f grad %f]\n",v, v->label, v->data, v->grad);
    print_graph(v->child[1]);
  }
}



void zero_grad_node_temp(value *v)
{ if (v)
  { zero_grad_node_temp(v->child[0]);
    if (v->type == TEMP)
      v->grad = 0;
    zero_grad_node_temp(v->child[1]);
  }
}

void zero_grad(value *v)
{ if (v)
  {
    zero_grad(v->child[0]);
    v->grad = 0;
    zero_grad(v->child[1]);
  }  
}

void acumula_grad(value *v)
{
  if (v)
  {
    acumula_grad(v->child[0]);
    printf("[acumulando para %s grad = %f]\n", v->label, v->grad);
    v->grad += v->grad ;
    printf("[acumulando para %s grad = %f]\n", v->label, v->grad);
    acumula_grad(v->child[1]);
  }  
}



void sin_backward(value *v)
{
  if (v && v->child[0])
    v->child[0]->grad += cosf(v->child[0]->data) * v->grad;
}
void cos_backward(value *v)
{
   if (v && v->child[0])
     v->child[0]->grad += -sinf(v->child[0]->data) * v->grad;
}

void tanh_backward(value *v)
{
   if (v && v->child[0])
     v->child[0]->grad += (1 - (v->data * v->data)) * v->grad;
}

void exp_backward(value *v)
{
  if (v && v->child[0])
     v->child[0]->grad += (v->data) * v->grad;
}

void log_backward(value *v)
{
  if (v && v->child[0])
    v->child[0]->grad += (1 / v->child[0]->data)  * v->grad;
}

void sin_forward(value *v)
{
  v->data = sinf(v->child[0]->data);
}

void cos_forward(value *v)
{
  v->data = cosf(v->child[0]->data);
}

void tanh_forward(value *v)
{
  v->data = tanhf(v->child[0]->data);
}
void exp_forward(value *v)
{
  v->data = expf(v->child[0]->data);
}
void log_forward(value *v)
{
 v->data = logf(v->child[0]->data);
}

void relu_forward(value *v)
{
  v->data = v->child[0]->data > 0 ? v->child[0]->data : 0;
}

void mul_forward(value* v)
{
  v->data = v->child[0]->data * v->child[1]->data;
}
void plus_forward(value*v)
{
  v->data = v->child[0]->data + v->child[1]->data;
}
void minus_forward(value*v)
{
  v->data = v->child[0]->data - v->child[1]->data;
}
void div_forward(value*v)
{
  v->data = v->child[0]->data / v->child[1]->data;
}
void pot_forward(value*v)
{
  v->data = pow(v->child[0]->data, v->child[1]->data);
}



std::vector<value  > vec_mul_vec(std::vector<value > in_1, std::vector<value > in_2)
{
  std::vector<value > ret(in_1.size());

  for (int i = 0; i < in_1.size(); i++)
  {
    ret[i] = in_1[i] * in_2[i];
  }

  return ret;
}


std::vector <value > vec_mul_matrix(std::vector<value > in_1, std::vector<std::vector<value >> in_2)
{
  std::vector<value >ret(in_1.size());
  if (in_1.size() != in_2[0].size())
  { puts("tamanhos invalidos");
    return ret;
  }

  for (int i = 0; i < in_1.size(); ++i)
  { for (int j = 0; j < in_2[0].size(); ++j)
    {
      for (int k = 0; k < in_1.size(); ++k)
        ret[i]  =  in_1[i] * in_2[k][j];
    }
  }
  return ret;
}

std::vector<std::vector<value >> mat_mul(std::vector<std::vector<value >>  in_1,
                                          std::vector<std::vector<value >>  in_2)
{
  std::vector<std::vector<value >>ret(in_1.size(), std::vector<value>(in_2[0].size()));

  if (in_1.size() != in_2[0].size())
  { puts("tamanhos invalidos");
    return ret;
  }

  for (int i = 0; i <   in_1.size(); ++i)
  { for (int j = 0; j < in_2[0].size(); ++j)
    {
      for (int k = 0; k < in_1[0].size(); ++k)
        ret[i][j]  =  in_1[i][k] * in_2[k][j];
    }
  }
  return ret;
}
std::vector<std::vector<value >> mat_sum(std::vector<std::vector<value >>  in_1,
                                          std::vector<std::vector<value >>  in_2)
{
  std::vector<std::vector<value >>ret(in_1.size(), std::vector<value>(in_1[0].size()));

  if (in_1.size() != in_2.size() || in_1[0].size() != in_2[0].size())
  { puts("tamanhos invalidos");
    return ret;
  }

  for (int i = 0; i < in_1.size(); i++)
  { for (int j = 0; j < in_1[0].size(); j++)
    {
      ret[i][j] = in_1[i][j] + in_2[i][j];
    }
  }
  return ret;
}
std::vector<std::vector<value >> mat_sub(std::vector<std::vector<value >>  in_1,
                                          std::vector<std::vector<value >>  in_2)
{
  std::vector<std::vector<value >>ret(in_1.size(), std::vector<value>(in_1[0].size()));

  if (in_1.size() != in_2.size() || in_1[0].size() != in_2[0].size())
  { puts("tamanhos invalidos");
    return ret;
  }

  for (int i = 0; i < in_1.size(); i++)
  { for (int j = 0; j < in_1[0].size(); j++)
    {
      ret[i][j] =  in_1[i][j] - in_2[i][j];
    }
  }
  return ret;
}
std::vector<std::vector<value >> mat_const_sub(std::vector<std::vector<value >>  in_1,
                                          float _const)
{
  std::vector<std::vector<value >>ret(in_1.size(), std::vector<value>(in_1[0].size()));

  scalar c = scalar(_const);

  for (int i = 0; i < in_1.size(); i++)
  { for (int j = 0; j < in_1[0].size(); j++)
    {
      ret[i][j] =  in_1[i][j] - c;
    }
  }
  return ret;
}
std::vector<std::vector<value >> mat_const_mul(std::vector<std::vector<value >>  in_1,
                                          float _const)
{
  std::vector<std::vector<value >>ret(in_1.size(), std::vector<value>(in_1[0].size()));

  scalar c = scalar(_const);

  for (int i = 0; i < in_1.size(); i++)
  { for (int j = 0; j < in_1[0].size(); j++)
    {
      ret[i][j] =  in_1[i][j] * c;
    }
  }
  return ret;
}

std::vector<std::vector<value >> mat_const_div(std::vector<std::vector<value >>  in_1,
                                          float _const)
{
  std::vector<std::vector<value >>ret(in_1.size(), std::vector<value>(in_1[0].size()));

  scalar c = scalar(_const);

  for (int i = 0; i < in_1.size(); i++)
  { for (int j = 0; j < in_1[0].size(); j++)
    {
      ret[i][j] = in_1[i][j] / c;
    }
  }
  return ret;
}

std::vector<std::vector<value >> mat_const_sum(std::vector<std::vector<value >>  in_1,
                                          float _const)
{
  std::vector<std::vector<value >>ret(in_1.size(), std::vector<value>(in_1[0].size()));

  scalar c = scalar(_const);
  
  for (int i = 0; i < in_1.size(); i++)
  { for (int j = 0; j < in_1[0].size(); j++)
    {
      ret[i][j] = in_1[i][j] + c;
    }
  }
  return ret;
}

