# route\_layer

## forward\_route\_layer

```c
void forward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *input = net.layers[index].output;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            copy_cpu(input_size, input + j*input_size, 1, l.output + offset + j*l.outputs, 1);
        }
        offset += input_size;
    }
}
```

함수 이름: forward\_route\_layer

입력:

* const route\_layer l: route layer 구조체
* network net: neural network 구조체

동작:

* 입력으로 받은 neural network의 route layer를 순전파(forward propagation) 진행
* route layer에 연결된 모든 input layer의 출력을 하나로 이어붙여(l.output) 반환

설명:

* l.input\_layers: route layer와 연결된 input layer의 인덱스를 저장하는 int 배열
* l.input\_sizes: route layer와 연결된 input layer의 출력 크기를 저장하는 int 배열
* l.batch: mini-batch 크기
* l.outputs: route layer 출력의 크기
* 각 input layer의 출력을 mini-batch 단위로 이어붙여서 route layer의 출력을 만듦
* copy\_cpu 함수: OpenBLAS 라이브러리 함수로, 배열의 복사를 수행함



## backward\_route\_layer

```c
void backward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *delta = net.layers[index].delta;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            axpy_cpu(input_size, 1, l.delta + offset + j*l.outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
}
```

함수 이름: backward\_route\_layer

입력:

* route\_layer l: route\_layer 구조체
* network net: 네트워크 구조체

동작:

* route\_layer의 역전파를 수행함
* route\_layer에 입력된 레이어들의 delta 값을 계산하여 더함

설명:

* route\_layer는 여러 입력 레이어들의 출력을 합침(concatenate)으로써 이전 레이어의 출력을 다음 레이어의 입력으로 사용할 수 있도록 함
* 따라서, route\_layer의 입력으로 사용된 모든 레이어들의 delta 값을 계산해야 함
* 이를 위해, route\_layer에 입력된 레이어들의 delta 값을 더해줌
* offset 변수는 입력 레이어들의 출력이 route\_layer의 출력에 어디서부터 복사되는지를 나타내는 인덱스 역할을 함



## resize\_route\_layer

```c
void resize_route_layer(route_layer *l, network *net)
{
    int i;
    layer first = net->layers[l->input_layers[0]];
    l->out_w = first.out_w;
    l->out_h = first.out_h;
    l->out_c = first.out_c;
    l->outputs = first.outputs;
    l->input_sizes[0] = first.outputs;
    for(i = 1; i < l->n; ++i){
        int index = l->input_layers[i];
        layer next = net->layers[index];
        l->outputs += next.outputs;
        l->input_sizes[i] = next.outputs;
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            l->out_c += next.out_c;
        }else{
            printf("%d %d, %d %d\n", next.out_w, next.out_h, first.out_w, first.out_h);
            l->out_h = l->out_w = l->out_c = 0;
        }
    }
    l->inputs = l->outputs;
    l->delta =  realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = realloc(l->output, l->outputs*l->batch*sizeof(float));

}
```

함수 이름: resize\_route\_layer

입력:&#x20;

* route\_layer \*l (route\_layer 구조체 포인터)
* network \*net (network 구조체 포인터)

동작:&#x20;

* route\_layer 구조체의 출력 크기와 입력 크기를 업데이트하고 메모리를 재할당한다.&#x20;
* 입력 레이어 중 첫 번째 레이어의 출력 크기를 사용하여 route\_layer의 출력 크기 및 출력 채널 수를 초기화한다.&#x20;
* 그런 다음 나머지 입력 레이어를 확인하고 출력 크기를 누적한다.&#x20;
* 모든 입력 레이어의 출력 크기가 같은 경우 출력 채널 수를 증가시킨다.&#x20;
* 그렇지 않은 경우 출력 크기와 출력 채널 수를 0으로 설정한다. 마지막으로 메모리를 재할당한다.

설명:&#x20;

* route\_layer는 입력 레이어에서 여러 출력을 결합하는 데 사용되는 레이어이다.&#x20;
* 이 함수는 route\_layer의 출력 크기와 입력 크기를 업데이트하고 메모리를 재할당하는 데 사용된다.&#x20;
* 또한 입력 레이어의 출력 크기가 다른 경우 경고 메시지를 출력한다.&#x20;
* 이 함수는 네트워크에서 route\_layer를 다시 크기 조정해야 할 때 호출된다.



## make\_route\_layer

```c
route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes)
{
    fprintf(stderr,"route ");
    route_layer l = {0};
    l.type = ROUTE;
    l.batch = batch;
    l.n = n;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    int i;
    int outputs = 0;
    for(i = 0; i < n; ++i){
        fprintf(stderr," %d", input_layers[i]);
        outputs += input_sizes[i];
    }
    fprintf(stderr, "\n");
    l.outputs = outputs;
    l.inputs = outputs;
    l.delta =  calloc(outputs*batch, sizeof(float));
    l.output = calloc(outputs*batch, sizeof(float));;

    l.forward = forward_route_layer;
    l.backward = backward_route_layer;

    return l;
}
```

함수 이름: make\_route\_layer

입력:

* batch: int형, 배치 크기
* n: int형, 이전 레이어의 개수
* input\_layers: int형 배열, 이전 레이어의 인덱스를 저장한 배열
* input\_sizes: int형 배열, 이전 레이어의 출력 크기를 저장한 배열

동작:&#x20;

* 입력으로 받은 정보를 바탕으로 route 레이어를 생성하고 초기화한다.&#x20;
* 출력값과 입력값의 크기를 계산하고, delta와 output 메모리를 동적 할당한다. forward와 backward 함수를 할당하고 생성된 레이어를 반환한다.

설명:&#x20;

* make\_route\_layer 함수는 입력으로 받은 정보를 바탕으로 route 레이어를 생성하고 초기화하는 함수이다. 이 함수는 생성된 route\_layer 구조체를 반환한다.
* 배치 크기(batch), 이전 레이어의 개수(n), 이전 레이어의 인덱스(input\_layers), 이전 레이어의 출력 크기(input\_sizes)를 인자로 받는다.
* 출력값과 입력값의 크기를 계산하고, delta와 output 메모리를 동적 할당한다. forward와 backward 함수를 할당하고 생성된 레이어를 반환한다.
* route\_layer 구조체를 초기화하기 위해 다음 필드를 설정한다.
  * type: ROUTE
  * batch: 입력으로 받은 배치 크기(batch)
  * n: 입력으로 받은 이전 레이어의 개수(n)
  * input\_layers: 입력으로 받은 이전 레이어의 인덱스(input\_layers)
  * input\_sizes: 입력으로 받은 이전 레이어의 출력 크기(input\_sizes)
  * outputs: 이전 레이어의 출력 크기를 모두 합한 값
  * inputs: 이전 레이어의 출력 크기를 모두 합한 값
  * delta: 크기가 outputs \* batch인 0으로 초기화된 float형 배열
  * output: 크기가 outputs \* batch인 0으로 초기화된 float형 배열
  * forward: forward\_route\_layer 함수의 포인터
  * backward: backward\_route\_layer 함수의 포인터
* 마지막으로 생성된 route\_layer 구조체를 반환한다.

