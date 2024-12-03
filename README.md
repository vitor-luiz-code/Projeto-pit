# Projeto-pit

# Introdução

O Micrograd é uma implementação de um auto diferenciador em Python escrito por Andrej karpathy, projetado para calcular derivadas parciais exatas de expressões matemáticas. Ele é especialmente útil em aplicações de aprendizado de máquina, onde o cálculo automático de gradientes é essencial para treinar modelos.

Esta reimplementação do Micrograd em C++ tem dois objetivos principais:

1 -> Aprender C++: Explorar os recursos da linguagem, como sobrecarga de operadores e gerenciamento de memória.

2 -> Construção Natural do Grafo Computacional: Utilizar a sobrecarga de operadores para representar operações matemáticas de forma intuitiva e construir automaticamente o grafo computacional durante a avaliação de expressões.

# Como Funciona -> Grafo Computacional

O núcleo do Micrograd é o grafo computacional, que organiza as operações matemáticas em um DAG (Directed Acyclic Graph ou grafo acíclico direcionado). Cada nó no grafo representa uma operação ou valor, e as arestas definem as dependências entre essas operações.

Ordenação Topológica
Para calcular as derivadas parciais, o grafo computacional é linearizado por meio de um processo chamado ordenação topológica. Esse processo gera uma sequência linear dos nós, respeitando as dependências. Essa ordem é usada em duas passagens principais:

1 -> Forward Pass: Avalia a expressão armazenada no grafo, propagando valores dos nós de entrada até os nós de saída.

2 -> Backward Pass: Calcula as derivadas parciais de cada nó em relação às suas dependências, seguindo a ordem inversa do forward.

Regras de Derivadas

Cada nó do grafo está associado a uma função que define como calcular sua derivada parcial em relação às entradas. Isso permite que a construção do grafo e o cálculo do gradiente sejam automáticos e eficientes.

# Estrutura da reimplementação

A biblioteca possui duas classes principais a classe Value e a classe Scalar

# classe Value 

Ela é o núcleo da estrutura do grafo computacional. Ela desempenha vários papéis fundamentais:

1 -> Representação das Variáveis: Cada instância de Value pode representar uma variável no grafo.
2 -> Operações Matemáticas: Serve como tipo base para a execução de operações matemáticas.
3 -> Dependências: Armazena as entradas de que depende (por exemplo, em uma expressão como a=x+y, a é um nó temporário que depende de x e y).
4 -> Resultados e Gradientes: Mantém o valor da expressão (resultado de uma operação anterior) e o gradiente associado para o cálculo do gradiente reverso (backward pass).
5 -> Funções de Passagem: Inclui ponteiros para as funções de forward e backward pass, garantindo que o grafo possa ser percorrido de forma eficiente.

Com essa estrutura, a classe Value é essencial para construir, armazenar e manipular o grafo de dependências, gerenciando tanto os valores das expressões quanto suas derivadas.

# Classe Scalar

A classe Scalar desempenha dois papéis principais:

1 -> Redução de Código: Simplifica a implementação ao permitir que os tipos primitivos da linguagem (como int, float, double, etc.) sejam representados como objetos. Isso evita a necessidade de sobrecarregar manualmente as operações matemáticas básicas (como +, -, *, /, ^) para cada tipo primitivo. Em vez disso, essas operações são implementadas diretamente para a classe Scalar.

Exemplo: Operações entre inteiros e objetos Value podem ser tratadas automaticamente por meio de conversões para Scalar, reduzindo a complexidade do código.

2 -> Representação de Constantes: Também é utilizada para representar constantes dentro do grafo computacional, facilitando a manipulação de valores fixos durante os cálculos.

# Gerência de memória

A gerência de memória, atualmente, é feita de forma manual e ainda apresenta certa complexidade. Isso ocorre porque parti do seguinte princípio: todas as variáveis devem ser alocadas dinamicamente para que "vivam" até o término do programa e possam ser acessadas entre chamadas de funções.

No entanto, esse requisito inicialmente gerou problemas na implementação, especialmente na sobrecarga de operadores. A linguagem permite utilizar ponteiros como argumentos, mas não dois ponteiros simultaneamente (por exemplo, em expressões do tipo ponteiro op ponteiro). Para contornar essa limitação, adotei uma solução alternativa:
em vez de retornar ponteiros retorno valores(quero dizer o valor do endereço de memória apontado pelo ponteiro).

Essa abordagem funciona da seguinte maneira: o valor retornado (uma instância de Value) inclui tanto o resultado da expressão quanto um endereço de memória (da classe Value) que foi alocado dinamicamente. Dentro da classe Value, utilizei um ponteiro chamado self, que aponta para a própria instância, permitindo acesso à memória alocada sem a necessidade de retornar ponteiros explícitos. Essa estratégia evita o problema com expressões do tipo ponteiro op ponteiro e possibilita a construção do grafo computacional.
Liberação de Memória

Apesar de funcional, a gerência de memória ainda pode ser considerada confusa, já que requer atenção especial para garantir que não haja vazamentos. A liberação da memória é realizada por um objeto local, que mantém uma referência ao grafo construído. Esta variável (tipo Value) utiliza o método clean_up, que realiza uma nova ordenação topológica para liberar, em ordem correta, toda a memória alocada dinâmicamente, incluindo as variáveis associadas.

Portanto, recomendo que todas as variáveis sejam alocadas dinâmicamente, pois o comportamento do programa será indefinido caso essa regra não seja seguida.

exemplo : 
    value *a = new value(10);
    value *b = new value(3);
    scalar constante = scalar(2);
    value ret = a * (*b) + c;
    // libreando a memória 
    ret.clean_up()
    // uso posterior de a e b indefinidos
    
# Operações suportadas

Até o momento são suportadas +,-,*, / e ^ entre as combinações possíveis entre Value e Value , Scalar e Scalar, ponteiro para Value e Value e  ponteiro para Value e Scalar (trocando as ordens também vale). Scalar e Scalar requer uma atenção pois eles não retornam Value e sim Scalar e a idéia é e as funções sobrecarregadas impõem isso é que Scalar seja sempre por valor.

As derivadas atualmente suportadas são : cx, x^n, x^x , a^x, u/v, exp, log, sin, cos e tanh.
e também suporte a criação de um grafo estático apartir de uma expressao permitindo forward apenas mudando os valores das variaveis e backward, vale 
dizer que o grafo compartilha a mesma memória que foi utilizada na criacao da expressão que foi salva na variavel local que guarda a referência ao grafo
então se chamar clean_up e após fazer um forward ou backward com o grafo estático (variavel do tipo static_graph que é retornada por uma função membro da classe Value chamada freeze_graph) terá comportamento indefinido.

.No arquivo: teste_melhorias tem exemplos simples.

Uma limitação importante, pelo fato de ter sobrecarregado o operador xor para ser utilizado como pontenciação sua associatividade nao condiz com a operação 
então é recomendado que use parenteses por exemplo a * b ^ c a não ser que vocẽ queira esta expressão [a * b ] ^ c e sim a * [b ^ c] use parênteses para que isso funcione corretamente.

# Ideias para implementação futura e melhorias
Uma gerência de memória adequada e cálculo de derivadas de Matrizes.  
