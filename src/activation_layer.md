# activation\_layer

### make\_activation\_layer

```c
layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer l = {0};
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    l.output = calloc(batch*inputs, sizeof(float*));
    l.delta = calloc(batch*inputs, sizeof(float*));

    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;

    l.activation = activation;
    fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    return l;
}
```

함수 이름: make\_activation\_layer

입력:&#x20;

* batch: 배치 크기
* inputs: 입력의 크기
* activation: 활성화 함수

동작: 입

* 입력값을 받아 활성화 함수를 적용한 출력값을 반환하는 레이어를 생성한다.

설명:

* layer 구조체를 초기화하고, 필드에 입력값을 설정한다.
* 출력값과 delta를 저장할 메모리 공간을 동적으로 할당한다.
* forward\_activation\_layer 및 backward\_activation\_layer 함수를 설정한다.
* activation 함수를 설정하고, 초기화된 layer 구조체를 반환한다.
* 활성화 레이어가 생성될 때, 입력값의 크기(inputs)와 활성화 함수의 종류를 출력한다.



### forward\_activation\_layer

```c
void forward_activation_layer(layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, l.activation);
}
```

함수 이름: forward\_activation\_layer

입력:&#x20;

* l: layer 구조체&#x20;
* net: network 구조체

동작:&#x20;

* 현재 층(layer)의 출력 값을 네트워크(network)의 입력 값으로 복사한 뒤, 활성화 함수(activation function)를 이용해 출력 값을 변경

설명:&#x20;

* 딥러닝 신경망의 순전파(forward propagation) 과정 중, 현재 층의 출력 값을 다음 층의 입력 값으로 전달하기 전에 활성화 함수를 적용해 출력 값을 변경하는 함수입니다.&#x20;
* 먼저, l.outputs\*l.batch 크기의 메모리를 net.input에서 l.output으로 복사합니다.&#x20;
* 이후, l.activation으로 지정된 활성화 함수를 l.output에 적용해 출력 값을 변경합니다.



### backward\_activation\_layer

```c
void backward_activation_layer(layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}
```

함수 이름: backward\_activation\_layer

입력:

* layer l: 역전파를 수행할 activation layer
* network net: 역전파의 결과를 저장할 네트워크

동작:

* l.output의 gradient를 l.delta로 계산
* l.delta를 net.delta로 복사

설명:

* Activation layer에서는 입력값을 활성화 함수를 통해 변환한 뒤 출력값을 계산하게 됩니다.
* 이때, 역전파를 수행하기 위해서는 출력값에 대한 gradient를 계산해야 합니다.
* backward\_activation\_layer 함수는 l.output에 대한 gradient를 l.activation 함수를 이용하여 계산하고, 그 결과를 l.delta에 저장합니다.
* 마지막으로, l.delta를 네트워크의 전체 delta값인 net.delta로 복사합니다.

