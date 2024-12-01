#include "scalar.hpp"
#include <cstring>

int main()
{
  // status : feito 
      // primeira melhoria inclusao de uma funcao para criacao em imagem do grafo computacional
      // a iamgem do grafo foi criada utilizando a linguagem dot do software de visualizacao de grafos 
      // graphviz
      
   value *x1 = new value(3.0);
   value *x2 = new value(2.0);
   x2->label = "x2"; // setando rotulo da variavel
   x1->label = "x1"; 
   value ret = (*x1 * *x2) ^ (*x1 * *x2);
   printf("resultado = %f", ret.data);
   const char * name = "graph.dot";
   ret.generate_visualization(name);
       
  // status : nao feito
      // inclusao de funcoes de erro como cross entropy, erro quadratico medio e outras
      // porque nao foi incluso?
      // o motivo da nao inclusao e que acho que foge da proposta. A funcao de erro seria
      // algo utilizado ja numa biblioteca de redes neurais, entao quem escrevesse a biblioteca de 
      // redes neurais escreveria a funcao de erro que gostaria de usar. a biblioteca de auto_diff
      // estaria num nivel abaixo.
      
  // status : feito
      // inclusao de funcoes como multiplicacao de matrizes do tipo Value tanto por vetores de tipo Value 
      // quanto outras matrizes tipo Value, as funcoes escritas foram matriz X matriz , Matriz X vetor, 
      // Matriz X constante ,vetor X vetor, matriz - constante,matriz + constante, matriz / constante
      // matriz * constante, matriz - matriz e matriz + matriz.
      
      // incializada com zeros
      std::vector<std::vector<value>>  matriz_a(2, std::vector<value>(2));  
      
      std::vector<std::vector<value>> matriz_ret = mat_const_sum(matriz_a, 10);
      
      for (int i = 0; i < 2; i++, puts(""))
      { for (int j = 0; j < 2; j++)
        {  printf("%f ",matriz_ret[i][j].data);
        }
      }
      
  // status : feito
    // inclusao da funcao para criacao de um grafo estatico, ou seja um grafo que fica salvo e voce pode executar 
    // backward e forward varia vezes apenas alterando os valores das variaveis
      
    struct static_graph *graph = ret.freeze_graph();
    graph->forward();
    graph->backward();
    printf("resultado forward  = %f grad x1 %f\n", graph->get_data(), x1->grad);
    x1->data = 1;
    x2->data = 1;
    graph->forward();
    printf("resultado forward  = %f\n", graph->get_data());
  // status : nao feito 
     // foi sugerido a implementacao de derivadas de matrizes, estou estudando como adicionar na bibloteca
     // num futuro proximo sera incluido.
     
  return 0;
}


