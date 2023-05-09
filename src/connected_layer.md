# connected\_layer

## Fully Connected Layer 란?

이전 Layer의 모든 노드가 다음 Layer의 모든 노드에 각각 하나씩 연결되어있는 Layer 입니다.

* 가장 기본적인 Layer
* 1차원 배열로만 이루어져 있습니다.

이해를 돕기위해 그림으로 살펴보겠습니다.

크게 복잡하지 않고 단순한 연산으로만 이루어져 있습니다.

Fully Connected Layer 역전파는 쉽게 표현하는 경우 아래 그림과 같습니다.

output을 계산하기 위해서 각자의 id를 가지고 있는 weight가 사용된 곳을 보시면 이해하기 쉽습니다. 예를 들어서 $$w_{11}$$은 $$h_{11}$$를 연산하는데만 사용되었기 때문에 해당 값만 사용합니다.

***

### forward\_connected\_layer

```c
void forward_connected_layer(layer l, network net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = net.input;
    float *b = l.weights;
    float *c = l.output;
    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.outputs, 1);
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
}
```

함수 이름: forward\_connected\_layer

입력:

* layer l: 연결 층(layer) 구조체
* network net: 네트워크(network) 구조체

동작:

* l.output 배열을 0으로 채움
* 행렬 곱 연산(GEMM)을 수행하여 l.output 배열을 새로운 값으로 업데이트 함
* 배치 정규화(batch normalization)가 활성화되어 있으면, forward\_batchnorm\_layer 함수를 호출하여 l.output 배열을 업데이트 함
* 배치 정규화가 비활성화되어 있으면, l.output 배열에 l.biases 값을 더함
* l.activation 함수를 사용하여 l.output 배열의 모든 원소에 활성화 함수를 적용함

설명:

* forward\_connected\_layer 함수는 완전 연결(fully connected) 층의 순전파(forward propagation) 연산을 수행하는 함수입니다.
* fill\_cpu 함수를 사용하여 l.output 배열을 0으로 초기화합니다.
* GEMM 함수를 사용하여 입력(input) 데이터와 가중치(weights)를 곱하여 l.output 배열을 새로운 값으로 업데이트합니다.
* 배치 정규화가 활성화되어 있으면, forward\_batchnorm\_layer 함수를 호출하여 l.output 배열을 업데이트합니다. 배치 정규화가 비활성화되어 있으면, l.output 배열에 l.biases 값을 더합니다.
* 마지막으로, activate\_array 함수를 사용하여 l.activation 함수를 적용하여 l.output 배열의 모든 원소에 활성화 함수를 적용합니다.



### backward\_connected\_layer

```c
void backward_connected_layer(layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.outputs, 1);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float *a = l.delta;
    float *b = net.input;
    float *c = l.weight_updates;
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta;
    b = l.weights;
    c = net.delta;

    if(c) gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
}
```

함수 이름: backward\_connected\_layer

입력:

* layer l: backpropagation이 수행될 fully connected layer
* network net: 연결된 neural network

동작:

* l에서 출력(l.output)과 활성화 함수(l.activation)를 사용하여 delta(l.delta)를 계산
* l이 batch normalization을 사용하는 경우, backward\_batchnorm\_layer 함수를 사용하여 backpropagation을 수행하고 그렇지 않으면 backward\_bias 함수를 사용하여 편향(l.bias\_updates)의 delta를 계산
* l.delta와 입력(net.input)을 사용하여 가중치(l.weights)의 업데이트(l.weight\_updates)를 계산하기 위해 GEMM 함수를 호출
* l.delta와 l.weights를 사용하여 입력(net.delta)의 delta를 계산하기 위해 GEMM 함수를 호출



### update\_connected\_layer

```c
void update_connected_layer(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    if(l.batch_normalize){
        axpy_cpu(l.outputs, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.outputs, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}
```

함수 이름: update\_connected\_layer

입력:

* layer l: 연결 계층(layer)을 나타내는 구조체
* update\_args a: 모델 업데이트를 위한 매개변수를 담은 구조체

동작:

* 연결 계층의 가중치(weights)와 편향(biases)을 업데이트하는 함수
* 매개변수로 주어진 a에 따라 learning\_rate, momentum, decay, batch 크기를 설정하고, 이를 사용하여 가중치와 편향을 업데이트한다.

설명:

* l.outputs: 현재 계층의 출력 개수
* l.bias\_updates: 편향의 업데이트에 사용될 값들이 저장된 배열
* l.biases: 현재 계층의 편향값이 저장된 배열
* l.scale\_updates: 배치 정규화(batch normalization)가 사용되는 경우, 스케일의 업데이트에 사용될 값들이 저장된 배열
* l.scales: 배치 정규화가 사용되는 경우, 스케일값이 저장된 배열
* l.inputs: 이전 계층의 출력 개수 혹은 입력 데이터의 차원 수
* l.weights: 가중치값이 저장된 배열
* l.weight\_updates: 가중치의 업데이트에 사용될 값들이 저장된 배열
* learning\_rate: 학습률(learning rate) 값
* momentum: 모멘텀(momentum) 값
* decay: 가중치 감소(weight decay) 값
* batch: 현재 배치(batch)의 크기
* axpy\_cpu(): BLAS 라이브러리 함수 중 하나로, 벡터 간의 연산을 수행하는 함수
* scal\_cpu(): 벡터에 스칼라 값을 곱하는 함수
* 먼저, 편향 업데이트를 수행한다. 이 때, axpy\_cpu() 함수를 사용하여 편향의 업데이트값을 편향값에 더하고, scal\_cpu() 함수를 사용하여 모멘텀 값으로 곱해준다.
* 만약 배치 정규화가 사용되는 경우, 스케일 업데이트도 수행한다. 이 때, axpy\_cpu() 함수를 사용하여 스케일의 업데이트값을 스케일값에 더하고, scal\_cpu() 함수를 사용하여 모멘텀 값으로 곱해준다.
* 가중치 업데이트를 수행한다. 이 때, axpy\_cpu() 함수를 사용하여 가중치의 업데이트값에 대한 값을 먼저 weight\_updates에 더한 다음, 가중치에 이 값을 더한다. 이후, scal\_cpu() 함수를 사용하여 모멘텀 값으로 곱해준다.



### make\_connected\_layer

```c
layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer l = {0};
    l.learning_rate_scale = 1;
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch=batch;
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;

    l.output = calloc(batch*outputs, sizeof(float));
    l.delta = calloc(batch*outputs, sizeof(float));

    l.weight_updates = calloc(inputs*outputs, sizeof(float));
    l.bias_updates = calloc(outputs, sizeof(float));

    l.weights = calloc(outputs*inputs, sizeof(float));
    l.biases = calloc(outputs, sizeof(float));

    l.forward = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update = update_connected_layer;

    //float scale = 1./sqrt(inputs);
    float scale = sqrt(2./inputs);
    for(i = 0; i < outputs*inputs; ++i){
        l.weights[i] = scale*rand_uniform(-1, 1);
    }

    for(i = 0; i < outputs; ++i){
        l.biases[i] = 0;
    }

    if(adam){
        l.m = calloc(l.inputs*l.outputs, sizeof(float));
        l.v = calloc(l.inputs*l.outputs, sizeof(float));
        l.bias_m = calloc(l.outputs, sizeof(float));
        l.scale_m = calloc(l.outputs, sizeof(float));
        l.bias_v = calloc(l.outputs, sizeof(float));
        l.scale_v = calloc(l.outputs, sizeof(float));
    }
    if(batch_normalize){
        l.scales = calloc(outputs, sizeof(float));
        l.scale_updates = calloc(outputs, sizeof(float));
        for(i = 0; i < outputs; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(outputs, sizeof(float));
        l.mean_delta = calloc(outputs, sizeof(float));
        l.variance = calloc(outputs, sizeof(float));
        l.variance_delta = calloc(outputs, sizeof(float));

        l.rolling_mean = calloc(outputs, sizeof(float));
        l.rolling_variance = calloc(outputs, sizeof(float));

        l.x = calloc(batch*outputs, sizeof(float));
        l.x_norm = calloc(batch*outputs, sizeof(float));
    }

    l.activation = activation;
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}
```

함수 이름: make\_connected\_layer

입력:

* batch: int형, 배치 크기(batch size)
* inputs: int형, 입력 크기(input size)
* outputs: int형, 출력 크기(output size)
* activation: ACTIVATION 열거형, 활성화 함수(activation function)
* batch\_normalize: int형, 배치 정규화 여부(batch normalization flag)
* adam: int형, Adam 최적화 알고리즘 사용 여부(Adam optimization flag)

동작:

* 입력값과 출력값 사이의 fully connected layer를 생성한다.
* 배치 정규화를 사용하는 경우, 배치 정규화 계층(batch normalization layer)을 생성한다.
* Adam 최적화 알고리즘을 사용하는 경우, Adam에 필요한 변수들을 초기화한다.
* 가중치(weight), 편향(bias) 등의 변수들을 초기화한다.

설명:

* 입력값과 출력값 사이의 fully connected layer를 생성하는 함수이다.
* layer 구조체를 선언하고, 필요한 변수들을 초기화한 후 반환한다.
* layer 구조체의 fields:
  * type: 레이어의 타입을 나타내는 열거형(enum) 변수
  * inputs: 입력 크기
  * outputs: 출력 크기
  * batch: 배치 크기
  * batch\_normalize: 배치 정규화 사용 여부
  * h, w, c: 레이어의 높이, 너비, 채널 수
  * out\_h, out\_w, out\_c: 출력 레이어의 높이, 너비, 채널 수
  * output: 레이어의 출력값
  * delta: 레이어의 역전파 시 그레이디언트 값
  * weights: 가중치
  * biases: 편향
  * weight\_updates: 가중치 갱신 값
  * bias\_updates: 편향 갱신 값
  * forward: 레이어의 순전파 함수 포인터
  * backward: 레이어의 역전파 함수 포인터
  * update: 레이어의 가중치와 편향을 갱신하는 함수 포인터
  * scales: 배치 정규화 계층의 스케일(scale) 값
  * scale\_updates: 배치 정규화 계층의 스케일 갱신 값
  * mean: 배치 정규화 계층의 평균(mean) 값
  * mean\_delta: 배치 정규화 계층의 평균 갱신 값
  * variance: 배치 정규화 계층의 분산(variance) 값
  * variance\_delta: 배치 정규화 계층의 분산 갱신 값
  * rolling\_mean: 배치 정규화 계층의 이동 평균 값
  * rolling\_variance: 배치 정규화 계층의 이동 분

