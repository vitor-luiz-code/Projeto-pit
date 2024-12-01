#include "scalar.hpp"

int main(void)
{
  scalar constante = scalar(2);

  value *x1 = new value(3.0);
  value *x2 = new value(2.0);
  value *x3 = new value(3.0);

  value ret = (x1 * *x2) ^ (*x1 * x2);

  ret.backward();
  printf("x2 grad -> %f\n" , x1->grad);
  value ret2 = (*x3 * constante) ^ (*x3 * constante);
  ret2.backward();
  printf("x3 grad -> %f\n" , x3->grad);
}


