# softmax\_layer

### softmax란?

참조 : [https://ratsgo.github.io/deep%20learning/2017/10/02/softmax/](https://ratsgo.github.io/deep%20learning/2017/10/02/softmax/)

* 입력의 모든 합을 1로 만드는 함수 입니다.
* $$p_i = \frac{exp(x_i)}{\sum^{C}_{c=1} exp{x_c}}$$

softmax는 역전파의 시작점 입니다.

우리는 error를 통해서 softmax input의 gradient를 구해야합니다. (그래야 뒤로 쭉쭉 갈수 있겠죠?)

먼저 연산을 위해 미리 softmax를 미분한다면

$$i = j$$ 일때

* $$\frac{\partial p_i}{\partial x_i} = \frac{\partial \frac{exp(x_i)}{\sum^{C}_{c=1} exp(x_c)}}{\partial x_i}$$
* $$\frac{\partial p_i}{\partial x_i} = \frac{exp(x_i) \sum^{C}_{c=1} exp(x_c) - exp(x_i) exp(x_i)}{(\sum^{C}_{c=1} exp(x_c))^2}$$
* $$= \frac{exp(x_i) [ \sum^{C}_{c=1} \left \{ \exp(x_c) \right \} - exp(x_i)]}{(\sum^{C}_{c=1} exp(x_c))^2}$$
* $$= \frac{exp(x_i)}{\sum^{C}_{c=1} exp(x_c)} \frac{\sum^{C}_{c=1} \left \{ exp(x_c) \right \} - exp(x_i) }{(\sum^{C}_{c=1} exp(x_c))}$$
* $$= \frac{exp(x_i)}{\sum^{C}_{c=1} exp(x_c)} \left ( 1 - \frac{exp(x_i)}{\sum^{C}_{c=1} exp(x_c)} \right )$$
* $$= p_i (1 - p_i)$$

$$i \neq j$$ 일때

* $$\frac{\partial p_i}{\partial x_j} = \frac{0 - exp(x_i) exp(x_j)}{(\sum^{C}_{c=1} exp(x_c))^2}$$
* $$= - \frac{exp(x_i)}{\sum^{C}_{c = 1} exp(x_c)} \frac{exp(x_j)}{\sum^{C}_{c=1} exp(x_c)}$$
* $$= - p_i p_j$$

역전파

* $$\frac{\partial L}{\partial x_i} = \frac{\partial (- \sum_{j} y_j \log p_j )}{ \partial x_i }$$
* $$= - \sum_j y_j \frac{\partial \log p_j}{\partial x_i}$$
* $$= - \sum_j y_j \frac{1}{p_j} \frac{\partial p_j}{\partial x_i}$$
* $$= - \frac{y_i}{p_i} p_i (1 - p_j) - \sum_{i \neq j} \frac{y_j}{p_j} (- p_i p_j)$$
* $$= - y_i + y_i p_i + \sum_{i \neq j} y_j p_i$$
* $$= - y_i + \sum_j y_j p_i$$
* $$= - y_i + p_i \sum_j y_j$$
* $$p_i - y_i$$

***

## softmax\_layer.c

### forward\_softmax\_layer

```c
void forward_softmax_layer(const softmax_layer l, network net)
{
    if(l.softmax_tree){
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output + count);
            count += group_size;
        }
    } else {
        softmax_cpu(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output);
    }

    if(net.truth && !l.noloss){
        softmax_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}
```

함수 이름: forward\_softmax\_layer

입력:

* softmax\_layer l: softmax 레이어의 정보를 담은 구조체
* network net: 뉴럴 네트워크의 정보를 담은 구조체

동작:

* softmax 레이어의 forward propagation을 수행한다.
* softmax\_tree가 존재하면, softmax\_tree를 이용하여 그룹화된 입력값들에 대해 softmax 연산을 수행한다.
* softmax\_tree가 존재하지 않으면, 입력값에 대해 softmax 연산을 수행한다.
* 만약 net.truth가 존재하고, l.noloss가 false이면, softmax\_cross\_entropy\_loss 를 이용하여 손실을 계산한다.

설명:

* softmax 레이어는 출력값을 확률값으로 변환해준다.
* softmax\_tree는 계층 구조를 이용하여 그룹화된 노드를 softmax 연산하기 위해 사용된다.
* softmax\_cpu 함수는 입력값에 대해 softmax 연산을 수행하고, 결과를 출력값으로 저장한다.
* softmax\_x\_ent\_cpu 함수는 softmax\_cross\_entropy\_loss 를 계산하고, 손실값을 loss 배열에 저장한다.
* 손실값은 l.cost\[0]에 저장된다.



### backward\_softmax\_layer

```c
void backward_softmax_layer(const softmax_layer l, network net)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1); // network delta = layer delta
}
```

함수 이름: backward\_softmax\_layer

입력:

* softmax\_layer l: softmax 레이어의 구조를 저장하는 구조체
* network net: 신경망 전체의 구조와 데이터를 저장하는 구조체

동작:

* 입력값에 대한 softmax 함수의 역전파 수행
* axpy\_cpu 함수를 사용하여, 현재 레이어의 델타 값을 이전 레이어의 델타 값에 더해줌으로써, 역전파를 계속 진행할 수 있도록 함

설명:

* softmax 레이어의 역전파는, softmax 함수의 출력값과 레이어의 델타 값을 이용해 수행됨
* 이 함수에서는, 현재 레이어의 델타 값을 이전 레이어의 델타 값에 더해주는 과정을 수행함
* 이렇게 함으로써, 이전 레이어의 델타 값은 현재 레이어의 델타 값에 영향을 받도록 되어, 역전파를 계속 진행할 수 있음
* 이 함수는 델타 값을 직접 수정하므로, 반환값이 없음

### make\_softmax\_layer

```c
softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "softmax                                        %4d\n",  inputs);
    softmax_layer l = {0};
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;


    return l;
}
```

함수 이름: make\_softmax\_layer

입력:

* int batch: 배치 크기
* int inputs: 입력 뉴런 수
* int groups: 그룹 수

동작:

* 입력 받은 batch, inputs, groups 값으로 softmax 레이어를 생성한다.
* 출력 뉴런 수는 입력 뉴런 수와 같다.
* loss, output, delta, cost 배열을 초기화한다.
* forward, backward 함수를 할당한다.

설명:

* Softmax 레이어는 출력값을 확률로 변환하는 레이어로, 입력값의 지수 함수를 취한 후, 해당 값을 소프트맥스 함수의 분모로 사용해 출력값을 구한다.
* 이 함수에서는 입력으로 받은 batch, inputs, groups 값으로 Softmax 레이어를 생성한다.
* assert 문을 사용하여, 입력 뉴런 수(inputs)가 그룹 수(groups)로 나누어 떨어지지 않을 경우 에러 메시지를 출력한다.
* l.loss, l.output, l.delta, l.cost 배열을 초기화한다.
* forward, backward 함수를 할당한다.
* 생성된 Softmax 레이어(l)를 반환한다.
