# gru\_layer

### GRU layer 란?

GRU (Gated Recurrent Unit) 레이어는 반복 신경망 (Recurrent Neural Network, RNN)의 한 종류로, 긴 시퀀스를 처리하는 데에 사용됩니다.

GRU는 기본적으로 LSTM (Long Short-Term Memory)과 유사한 아이디어를 기반으로 하고 있습니다. LSTM과 마찬가지로, GRU도 RNN 계열의 레이어로서 시퀀스 데이터를 처리할 수 있습니다. 하지만 LSTM과는 달리, GRU는 게이트 메커니즘을 사용하여 기억을 보호하고, 이전 상태에서 정보를 가져오는 방법을 간단화하여 더 적은 계산으로 장기적인 상태를 유지할 수 있도록 합니다.

GRU는 LSTM보다 더 간단한 구조를 가지고 있으며, 더 적은 파라미터를 필요로 합니다. GRU는 LSTM보다 학습 속도가 더 빠르고, 작은 데이터셋에서 더 일반적인 모델을 만들어내는 경향이 있습니다.

GRU 레이어는 2개의 게이트를 사용하여 기억을 조절합니다. 첫 번째 게이트는 "업데이트 게이트"라고 불리며, 현재 입력과 이전 상태를 결합하여 새로운 상태를 생성합니다. 두 번째 게이트는 "재설정 게이트"라고 불리며, 이전 상태의 일부를 버리고 새로운 상태를 만듭니다. GRU 레이어는 이러한 게이트들을 사용하여 입력 시퀀스와 이전 상태를 기반으로 한 다음, 새로운 상태를 출력합니다.

GRU 레이어는 주로 시퀀스 데이터를 다루는 자연어 처리(NLP) 분야에서 사용됩니다. GRU 레이어를 적용한 모델은 텍스트 생성, 번역, 감성 분석 등 다양한 태스크에서 좋은 성능을 보입니다.

***

### increment\_layer

```c
static void increment_layer(layer *l, int steps)
{
    int num = l->outputs*l->batch*steps;
    l->output += num;
    l->delta += num;
    l->x += num;
    l->x_norm += num;

#ifdef GPU
    l->output_gpu += num;
    l->delta_gpu += num;
    l->x_gpu += num;
    l->x_norm_gpu += num;
#endif
}
```

함수 이름: increment\_layer

입력:

* layer \*l: 업데이트할 레이어
* int steps: 이동할 스텝 수

동작:

* layer 구조체 포인터인 l의 output, delta, x, x\_norm에 steps만큼 이동한 포인터를 할당한다.
* GPU 환경에서는 l의 output\_gpu, delta\_gpu, x\_gpu, x\_norm\_gpu에 steps만큼 이동한 포인터를 할당한다.

설명:&#x20;

* 해당 함수는 레이어의 포인터를 steps만큼 이동시켜 업데이트하는 함수이다.&#x20;
* 포인터를 이동시켜서 이전의 값을 참조하지 않고 새로운 값을 참조할 수 있도록 한다.&#x20;
* GPU 환경에서는 GPU 메모리 상의 포인터를 이동시킨다.



### forward\_gru\_layer

```c
void forward_gru_layer(layer l, network net)
{
    network s = net;
    s.train = net.train;
    int i;
    layer uz = *(l.uz);
    layer ur = *(l.ur);
    layer uh = *(l.uh);

    layer wz = *(l.wz);
    layer wr = *(l.wr);
    layer wh = *(l.wh);

    fill_cpu(l.outputs * l.batch * l.steps, 0, uz.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, ur.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, uh.delta, 1);

    fill_cpu(l.outputs * l.batch * l.steps, 0, wz.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wr.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wh.delta, 1);
    if(net.train) {
        fill_cpu(l.outputs * l.batch * l.steps, 0, l.delta, 1);
        copy_cpu(l.outputs*l.batch, l.state, 1, l.prev_state, 1);
    }

    for (i = 0; i < l.steps; ++i) {
        s.input = l.state;
        forward_connected_layer(wz, s);
        forward_connected_layer(wr, s);

        s.input = net.input;
        forward_connected_layer(uz, s);
        forward_connected_layer(ur, s);
        forward_connected_layer(uh, s);


        copy_cpu(l.outputs*l.batch, uz.output, 1, l.z_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, wz.output, 1, l.z_cpu, 1);

        copy_cpu(l.outputs*l.batch, ur.output, 1, l.r_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, wr.output, 1, l.r_cpu, 1);

        activate_array(l.z_cpu, l.outputs*l.batch, LOGISTIC);
        activate_array(l.r_cpu, l.outputs*l.batch, LOGISTIC);

        copy_cpu(l.outputs*l.batch, l.state, 1, l.forgot_state, 1);
        mul_cpu(l.outputs*l.batch, l.r_cpu, 1, l.forgot_state, 1);

        s.input = l.forgot_state;
        forward_connected_layer(wh, s);

        copy_cpu(l.outputs*l.batch, uh.output, 1, l.h_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, wh.output, 1, l.h_cpu, 1);

        if(l.tanh){
            activate_array(l.h_cpu, l.outputs*l.batch, TANH);
        } else {
            activate_array(l.h_cpu, l.outputs*l.batch, LOGISTIC);
        }

        weighted_sum_cpu(l.state, l.h_cpu, l.z_cpu, l.outputs*l.batch, l.output);

        copy_cpu(l.outputs*l.batch, l.output, 1, l.state, 1);

        net.input += l.inputs*l.batch;
        l.output += l.outputs*l.batch;
        increment_layer(&uz, 1);
        increment_layer(&ur, 1);
        increment_layer(&uh, 1);

        increment_layer(&wz, 1);
        increment_layer(&wr, 1);
        increment_layer(&wh, 1);
    }
}
```

함수 이름: forward\_gru\_layer

입력:

* layer l: GRU 레이어의 정보와 매개변수를 담고 있는 layer 구조체
* network net: 네트워크의 정보와 매개변수를 담고 있는 network 구조체

동작:&#x20;

* 입력 데이터의 GRU 레이어를 통해 순방향 전파(forward propagation)를 수행하는 함수로, 입력 데이터를 GRU 레이어를 통해 처리하여 출력 값을 계산하고, 그 값을 다음 레이어의 입력으로 넘겨줌.&#x20;
* 이때, backward propagation을 위해 필요한 중간값들을 저장해 놓음.

설명:

* GRU 레이어의 매개변수들 중에서 uz, ur, uh는 이전 상태(previous state)로부터의 입력(input)을 처리하는 가중치(weight) 매개변수이고, wz, wr, wh는 현재 입력(input)을 처리하는 가중치 매개변수임.
* GRU 레이어는 시계열(sequence) 데이터를 처리하기 위한 RNN의 한 종류로, 이전 시점의 상태(previous state)를 재사용하는 레이어임.
* forward\_connected\_layer 함수를 통해 가중치와 입력을 곱한 값과 bias를 더한 값을 계산하여 활성화 함수(Logistic 또는 Tanh)를 적용함.
* uz, ur, uh 레이어에서 나온 출력값과 wz, wr, wh 레이어에서 나온 출력값을 이용하여 z와 r 값을 계산함.
* z값은 이전 상태와 현재 입력을 조합한 후 로지스틱 함수를 적용하여 계산함.
* r값은 z값과 마찬가지로 이전 상태와 현재 입력을 조합한 후 로지스틱 함수를 적용하여 계산함.
* h값은 z값과 이전 상태를 이용하여 새로운 상태를 계산하기 위한 게이트(gate)를 계산함.
* 계산된 h값에 Tanh 또는 Logistic 함수를 적용하여 출력값(output)을 계산함.
* GRU 레이어는 여러 시점(time step)으로 구성되어 있으므로, steps 만큼 반복적으로 forward\_connected\_layer 함수를 호출하여 중간값들을 계산함.



### backward\_gru\_layer

```c
void backward_gru_layer(layer l, network net)
{
}
```

함수 이름: backward\_gru\_layer&#x20;

입력:&#x20;

* layer l
* network net (둘 다 구조체)&#x20;

동작:&#x20;

* GRU (게이트 순환 유닛) 레이어의 역전파(backpropagation)를 계산하고 이전 레이어에게 오차 신호(error signal)를 전달합니다.&#x20;
* 이를 위해 입력 신호와 가중치(weight)에 대한 미분(gradient)을 계산합니다.&#x20;

설명:

* l: GRU 레이어의 구조체로, 입력 신호와 가중치, 출력과 같은 다양한 정보를 담고 있습니다.
* net: 신경망 구조체로, 역전파 시에 이전 레이어로 오차 신호를 전달하기 위해 사용됩니다.

이 함수는 빈 상태로 남겨둔 것이 아니라, 구현 내용이 없는 것입니다. 함수를 호출할 때 실제로 계산이 이루어집니다.



### update\_gru\_layer

```c
void update_gru_layer(layer l, update_args a)
{
    update_connected_layer(*(l.ur), a);
    update_connected_layer(*(l.uz), a);
    update_connected_layer(*(l.uh), a);
    update_connected_layer(*(l.wr), a);
    update_connected_layer(*(l.wz), a);
    update_connected_layer(*(l.wh), a);
}
```

함수 이름: update\_gru\_layer

입력:&#x20;

* layer l: GRU 레이어 구조체
* update\_args a: 업데이트 인자 구조체

동작:&#x20;

* GRU 레이어의 각각의 연결된 레이어(ur, uz, uh, wr, wz, wh)들의 가중치(weight)와 bias를 업데이트하는 함수

설명:&#x20;

* 입력으로 주어진 GRU 레이어 구조체 l의 연결된 레이어(ur, uz, uh, wr, wz, wh)들의 가중치와 bias를 업데이트하는 함수이다.&#x20;
* 이를 위해 update\_connected\_layer() 함수를 각 레이어에 대해 호출하여 가중치를 업데이트한다.



### make\_gru\_layer

```c
layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam)
{
    fprintf(stderr, "GRU Layer: %d inputs, %d outputs\n", inputs, outputs);
    batch = batch / steps;
    layer l = {0};
    l.batch = batch;
    l.type = GRU;
    l.steps = steps;
    l.inputs = inputs;

    l.uz = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.uz) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.uz->batch = batch;

    l.wz = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wz) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wz->batch = batch;

    l.ur = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.ur) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.ur->batch = batch;

    l.wr = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wr) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wr->batch = batch;

    l.uh = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.uh) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.uh->batch = batch;

    l.wh = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wh) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wh->batch = batch;

    l.batch_normalize = batch_normalize;


    l.outputs = outputs;
    l.output = calloc(outputs*batch*steps, sizeof(float));
    l.delta = calloc(outputs*batch*steps, sizeof(float));
    l.state = calloc(outputs*batch, sizeof(float));
    l.prev_state = calloc(outputs*batch, sizeof(float));
    l.forgot_state = calloc(outputs*batch, sizeof(float));
    l.forgot_delta = calloc(outputs*batch, sizeof(float));

    l.r_cpu = calloc(outputs*batch, sizeof(float));
    l.z_cpu = calloc(outputs*batch, sizeof(float));
    l.h_cpu = calloc(outputs*batch, sizeof(float));

    l.forward = forward_gru_layer;
    l.backward = backward_gru_layer;
    l.update = update_gru_layer;

    return l;
}
```

함수 이름: make\_gru\_layer

입력:

* int batch: 배치 크기
* int inputs: 입력의 크기
* int outputs: 출력의 크기
* int steps: 시간 스텝의 수
* int batch\_normalize: 배치 정규화 사용 여부
* int adam: Adam 옵티마이저 사용 여부

동작:&#x20;

* GRU 레이어를 생성하고 초기화하는 함수이다. GRU 레이어는 uz, wr, uh, wh 등의 연결 레이어로 구성되어 있다.

설명:

* 입력값으로 받은 batch 값은 steps로 나누어져서 사용된다.
* 레이어의 타입은 GRU로 설정된다.
* uz, wz, ur, wr, uh, wh 등의 연결 레이어가 생성되고 초기화된다.
* 출력값, delta, state, prev\_state, forgot\_state, forgot\_delta, r\_cpu, z\_cpu, h\_cpu 등의 값들이 초기화된다.
* forward, backward, update 함수가 설정된다.
* 초기화된 GRU 레이어가 반환된다.
