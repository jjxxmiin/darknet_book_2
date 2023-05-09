# l2norm\_layer

## forward\_l2norm\_layer

```c
void forward_l2norm_layer(const layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    l2normalize_cpu(l.output, l.scales, l.batch, l.out_c, l.out_w*l.out_h);
}
```

함수 이름: forward\_l2norm\_layer

입력:&#x20;

* layer l
* network net

동작:&#x20;

* 입력으로 들어온 네트워크에서 l2norm 레이어를 forward propagation 한다.&#x20;
* 입력값을 복사하고 l2normalize 함수를 호출한다.

설명:

* l2norm 레이어: 입력값을 L2 norm으로 정규화하는 레이어
* copy\_cpu(a, x, incx, y, incy): x에서 y로 a개의 원소를 복사한다.
* l2normalize\_cpu(x, norm, batch, filters, spatial): x의 각 batch에서 filters\* spatial의 범위에서 L2 norm으로 정규화한다. 이 때, 각 filters마다 각각 스케일값이 있으며 norm 배열에 저장되어 있다.



## backward\_l2norm\_layer

```c
void backward_l2norm_layer(const layer l, network net)
{
    //axpy_cpu(l.inputs*l.batch, 1, l.scales, 1, l.delta, 1);
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}
```

함수 이름: backward\_l2norm\_layer

입력:

* const layer l: 레이어 정보를 담고 있는 구조체 포인터
* network net: 신경망 정보를 담고 있는 구조체

동작:

* 입력값으로 받은 레이어와 신경망 정보를 이용하여 L2 normalization을 수행한 결과 값을 이용하여 역전파를 진행함
* l.delta 값을 이용하여 net.delta 값을 계산하고 업데이트함

설명:

* 입력값으로 받은 레이어 정보에서 l.delta 값은 해당 레이어에서의 역전파에 대한 오차값을 나타냄
* 이전 레이어에서 역전파할 때 이용할 net.delta 값을 axpy\_cpu 함수를 이용하여 업데이트함



## make\_l2norm\_layer

```c
layer make_l2norm_layer(int batch, int inputs)
{
    fprintf(stderr, "l2norm                                         %4d\n",  inputs);
    layer l = {0};
    l.type = L2NORM;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.output = calloc(inputs*batch, sizeof(float));
    l.scales = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));

    l.forward = forward_l2norm_layer;
    l.backward = backward_l2norm_layer;

    return l;
}
```

함수 이름: make\_l2norm\_layer&#x20;

입력:

* batch: int 타입. 미니배치 크기
* inputs: int 타입. 입력 데이터의 차원 수

동작:&#x20;

* L2 normalization을 수행하는 layer를 생성한다.

설명:

* 입력 데이터의 크기는 batch \* inputs이다.
* 출력 데이터의 크기도 batch \* inputs이다.
* l.scales, l.delta, l.output은 모두 크기가 batch \* inputs인 float형 배열이다.
* 이 layer의 forward pass는 입력 데이터를 l2normalize\_cpu() 함수를 사용하여 L2 normalization을 수행한다.
* 이 layer의 backward pass는 입력 데이터의 미분을 계산하고, 이를 이전 레이어로 전파한다.
* 생성된 layer 구조체를 반환한다.

