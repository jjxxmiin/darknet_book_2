# logistic\_layer

## forward\_logistic\_layer

```c
void forward_logistic_layer(const layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, LOGISTIC);
    if(net.truth){
        logistic_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}
```

함수 이름: forward\_logistic\_layer

입력:

* const layer l: 레이어 구조체 포인터
* network net: 네트워크 구조체

동작:&#x20;

* 로지스틱 함수를 사용하여 레이어의 출력값을 계산하고, 만약 net.truth가 참(True)이면 로지스틱 손실 함수를 계산하여 l.loss와 l.delta를 업데이트하고, l.cost에 l.loss의 합을 저장한다.

설명:

* copy\_cpu(l.outputs_l.batch, net.input, 1, l.output, 1): net.input에서 l.output으로 l.outputs_l.batch 개의 실수값을 복사한다.
* activate\_array(l.output, l.outputs\*l.batch, LOGISTIC): l.output 배열에 있는 모든 값에 대해 로지스틱 활성화 함수를 적용한다.
* if(net.truth): 만약 net.truth가 참(True)이면 로지스틱 손실 함수를 계산하여 l.loss와 l.delta를 업데이트하고, l.cost에 l.loss의 합을 저장한다.
  * logistic\_x\_ent\_cpu(l.batch\*l.inputs, l.output, net.truth, l.delta, l.loss): 로지스틱 손실 함수와 그 도함수를 계산하고, l.delta와 l.loss를 업데이트한다.
  * l.cost\[0] = sum\_array(l.loss, l.batch\*l.inputs): l.loss 배열의 모든 값을 더하여 l.cost\[0]에 저장한다.



## backward\_logistic\_layer

```c
void backward_logistic_layer(const layer l, network net)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}
```

함수 이름: backward\_logistic\_layer

입력:&#x20;

* const layer l (레이어 구조체)
* network net (네트워크 구조체)

동작:&#x20;

* 로지스틱 회귀 레이어의 역전파(backpropagation)를 수행합니다.&#x20;
* 입력 데이터에 대한 오류 그래디언트를 계산하고, 네트워크의 이전 레이어로 그래디언트를 전파합니다.

설명:&#x20;

* 로지스틱 회귀 레이어의 역전파 함수입니다.&#x20;
* 입력으로는 로지스틱 회귀 레이어를 나타내는 layer 구조체와, 해당 레이어를 소유하는 네트워크를 나타내는 network 구조체가 입력됩니다.&#x20;
* 함수는 l.delta와 net.delta를 이용하여 그래디언트를 계산하고, net.delta에 결과를 저장합니다.



## make\_logistic\_layer

```c
layer make_logistic_layer(int batch, int inputs)
{
    fprintf(stderr, "logistic x entropy                             %4d\n",  inputs);
    layer l = {0};
    l.type = LOGXENT;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.forward = forward_logistic_layer;
    l.backward = backward_logistic_layer;
    #ifdef GPU
    l.forward_gpu = forward_logistic_layer_gpu;
    l.backward_gpu = backward_logistic_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.loss_gpu = cuda_make_array(l.loss, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
    #endif
    return l;
}
```

함수 이름: make\_logistic\_layer

입력:

* batch: 배치 크기 (int)
* inputs: 입력 데이터의 크기 (int)

동작:

* 로지스틱 회귀와 교차 엔트로피 손실 함수를 사용하는 레이어를 생성합니다.
* 입력 데이터의 크기와 배치 크기를 설정하고, 출력, 로스, 델타, 코스트 등을 초기화합니다.
* 포워드(forward)와 백워드(backward) 함수를 설정합니다.

설명:

* 로지스틱 회귀는 분류 문제에서 사용되는 대표적인 알고리즘 중 하나로, 입력 데이터를 이진 분류(binary classification)하는 데 사용됩니다.
* 교차 엔트로피 손실 함수는 로지스틱 회귀에서 사용되는 손실 함수 중 하나로, 예측 값과 실제 값의 차이를 계산하여 모델의 손실을 계산합니다.
* 입력 데이터의 크기는 모델의 입력 크기를 의미하며, 배치 크기는 한 번에 처리할 데이터의 개수를 의미합니다.

