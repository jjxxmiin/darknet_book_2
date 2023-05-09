# rnn\_layer

## RNN Layer란?

RNN (Recurrent Neural Network) Layer는 순환 신경망 모델 중 하나로, 입력 데이터의 순서와 상태 정보를 모델링하는 데에 적합한 모델입니다. RNN은 이전에 계산된 값을 다시 현재 계산에 활용하기 때문에 이전의 입력에 대한 정보를 기억하고 이를 다음 계산에 활용합니다. 이전 입력에 대한 정보를 현재 계산에 활용하기 때문에 시계열 데이터와 같이 입력 간의 순서가 중요한 데이터를 다룰 때 매우 유용합니다.

RNN Layer는 시간 축으로 펼쳐진 형태의 네트워크 구조를 가지며, 시간 t에서의 입력 데이터를 받아 출력 데이터를 계산하는 동작을 반복적으로 수행합니다. RNN Layer의 각 뉴런은 현재 입력 데이터와 이전 시점의 출력값을 입력으로 받아 현재 시점의 출력값을 계산합니다. 이를 수식으로 나타내면 다음과 같습니다.

$$h_t = f_h(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

여기서 $$x_t$$는 시간 t에서의 입력 벡터, $$h_t$$는 시간 t에서의 출력 벡터, $$W_{xh}$$는 입력 가중치 행렬, $$W_{hh}$$는 이전 시점의 출력값과 현재 입력값을 연결한 가중치 행렬, $$b_h$$는 편향 벡터, $$f_h$$는 활성화 함수입니다.

RNN Layer는 다양한 종류가 있으며, 대표적으로 Simple RNN, LSTM(Long Short-Term Memory), GRU(Gated Recurrent Unit) 등이 있습니다. 이들은 각각 입력과 출력 사이의 정보 흐름을 다르게 조절하여, 다양한 시계열 데이터에 대한 모델링을 가능하게 합니다.

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
}
```

함수 이름: increment\_layer

입력:&#x20;

* layer \*l
* int steps

동작:&#x20;

* RNN 레이어에서 timestep 단위로 입력 데이터를 처리할 때, 이전 timestep에서 출력한 결과값을 현재 timestep에서 입력으로 사용하기 위해, 현재 timestep에 해당하는 레이어 포인터(l)가 가리키는 데이터 포인터(output, delta, x, x\_norm)를 steps(현재 timestep과 이전 timestep 간의 차이)만큼 증가시켜주는 함수이다.

설명:&#x20;

* RNN 레이어는 시퀀스 형태의 데이터를 처리할 때, 이전 timestep에서 출력한 결과값을 현재 timestep에서 다시 입력으로 사용한다.&#x20;
* 이 때, 현재 timestep의 레이어 포인터가 이전 timestep에서의 레이어 포인터와 가리키는 데이터의 위치가 달라지기 때문에, 현재 timestep에서의 데이터 포인터를 이전 timestep에서의 데이터 포인터에서 적절히 이동시켜주어야 한다.&#x20;
* 이 함수는 이를 수행하는 역할을 한다. 입력으로 현재 timestep의 레이어 포인터(l)와 이전 timestep과의 차이(steps)를 받아서, l이 가리키는 데이터 포인터(output, delta, x, x\_norm)를 steps만큼 증가시켜준다.&#x20;
* 이 때, 데이터 포인터의 증가량은 timestep 간의 차이에 데이터의 크기(output, delta, x, x\_norm)와 배치 크기(batch)를 곱한 값이다.



### forward\_rnn\_layer

```c
void forward_rnn_layer(layer l, network net)
{
    network s = net;
    s.train = net.train;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);

    fill_cpu(l.outputs * l.batch * l.steps, 0, output_layer.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, self_layer.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, input_layer.delta, 1);
    if(net.train) fill_cpu(l.outputs * l.batch, 0, l.state, 1);

    for (i = 0; i < l.steps; ++i) {
        s.input = net.input;
        forward_connected_layer(input_layer, s);

        s.input = l.state;
        forward_connected_layer(self_layer, s);

        float *old_state = l.state;
        if(net.train) l.state += l.outputs*l.batch;
        if(l.shortcut){
            copy_cpu(l.outputs * l.batch, old_state, 1, l.state, 1);
        }else{
            fill_cpu(l.outputs * l.batch, 0, l.state, 1);
        }
        axpy_cpu(l.outputs * l.batch, 1, input_layer.output, 1, l.state, 1);
        axpy_cpu(l.outputs * l.batch, 1, self_layer.output, 1, l.state, 1);

        s.input = l.state;
        forward_connected_layer(output_layer, s);

        net.input += l.inputs*l.batch;
        increment_layer(&input_layer, 1);
        increment_layer(&self_layer, 1);
        increment_layer(&output_layer, 1);
    }
}
```

함수 이름: forward\_rnn\_layer

입력:&#x20;

* layer l (현재 RNN 레이어)
* network net (네트워크)

동작:&#x20;

* 현재 RNN 레이어를 포워드 패스하는 함수로, RNN 레이어의 입력, 은닉 상태, 출력을 계산합니다.&#x20;
* 입력, 은닉 상태, 출력 계산을 위해 connected 레이어가 사용됩니다.&#x20;
* RNN 레이어의 입력은 현재 레이어 이전의 출력과 현재 시점의 입력을 더한 값입니다.&#x20;
* 이전의 출력을 더해주는 이유는 RNN이 이전의 정보를 기억하기 위해서입니다.&#x20;
* 또한, 현재 시점의 입력과 은닉 상태를 더해주는 이유는 현재 입력과 이전 상태가 다음 상태의 출력에 영향을 미치기 때문입니다.&#x20;
* 레이어의 입력, 출력, 은닉 상태, 델타 값이 업데이트됩니다.

설명:&#x20;

* 이 함수는 RNN 레이어를 포워드 패스하는 함수로, 이전 레이어에서 출력된 값을 현재 레이어의 입력으로 사용합니다.&#x20;
* 이전 레이어에서 출력된 값과 현재 입력 값을 더한 값을 RNN 레이어의 입력으로 사용하며, 은닉 상태를 업데이트하고 출력을 계산합니다.&#x20;
* 이전 상태와 현재 입력이 다음 상태에 영향을 미치므로, 이전 상태와 현재 입력을 더해주는 것입니다.&#x20;
* 이 함수는 네트워크를 학습 중인지 아닌지에 따라 네트워크 상태를 변경합니다.&#x20;
* 또한, RNN 레이어는 연속된 스텝을 계산해야 하므로, 입력, 은닉 상태, 출력 레이어를 스텝 수만큼 반복적으로 계산합니다.



### backward\_rnn\_layer

```c
void backward_rnn_layer(layer l, network net)
{
    network s = net;
    s.train = net.train;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);

    increment_layer(&input_layer, l.steps-1);
    increment_layer(&self_layer, l.steps-1);
    increment_layer(&output_layer, l.steps-1);

    l.state += l.outputs*l.batch*l.steps;
    for (i = l.steps-1; i >= 0; --i) {
        copy_cpu(l.outputs * l.batch, input_layer.output, 1, l.state, 1);
        axpy_cpu(l.outputs * l.batch, 1, self_layer.output, 1, l.state, 1);

        s.input = l.state;
        s.delta = self_layer.delta;
        backward_connected_layer(output_layer, s);

        l.state -= l.outputs*l.batch;
        /*
           if(i > 0){
           copy_cpu(l.outputs * l.batch, input_layer.output - l.outputs*l.batch, 1, l.state, 1);
           axpy_cpu(l.outputs * l.batch, 1, self_layer.output - l.outputs*l.batch, 1, l.state, 1);
           }else{
           fill_cpu(l.outputs * l.batch, 0, l.state, 1);
           }
         */

        s.input = l.state;
        s.delta = self_layer.delta - l.outputs*l.batch;
        if (i == 0) s.delta = 0;
        backward_connected_layer(self_layer, s);

        copy_cpu(l.outputs*l.batch, self_layer.delta, 1, input_layer.delta, 1);
        if (i > 0 && l.shortcut) axpy_cpu(l.outputs*l.batch, 1, self_layer.delta, 1, self_layer.delta - l.outputs*l.batch, 1);
        s.input = net.input + i*l.inputs*l.batch;
        if(net.delta) s.delta = net.delta + i*l.inputs*l.batch;
        else s.delta = 0;
        backward_connected_layer(input_layer, s);

        increment_layer(&input_layer, -1);
        increment_layer(&self_layer, -1);
        increment_layer(&output_layer, -1);
    }
}
```

함수 이름: backward\_rnn\_layer

입력:

* layer l: 역전파를 수행할 RNN 레이어
* network net: RNN 레이어를 포함하는 신경망

동작:&#x20;

* RNN 레이어의 역전파(backpropagation)를 수행한다. RNN 레이어는 시간 스텝(time step)이 있기 때문에, 역전파는 시간의 반대 방향으로(step-by-step) 수행된다.

설명:

* 먼저, 입력 레이어(input\_layer), 자기 반복 레이어(self\_layer), 출력 레이어(output\_layer)를 가져온다.
* 각 레이어의 출력(delta)을 0으로 초기화한다.
* RNN 레이어가 학습 모드(train mode)일 경우, 상태(state)를 0으로 초기화한다.
* 모든 시간 스텝에 대해 반복하며, 다음을 수행한다:
  * input\_layer에 현재 입력(net.input)을 넣고, forward\_connected\_layer()를 호출하여 input\_layer의 출력(output)을 계산한다.
  * self\_layer에 이전 시간 스텝의 상태(l.state)를 넣고, forward\_connected\_layer()를 호출하여 self\_layer의 출력(output)을 계산한다.
  * RNN 레이어의 현재 상태를 계산한다. 이전 상태(old\_state)를 유지한 후, input\_layer와 self\_layer의 출력을 더한 값을 현재 상태(l.state)로 갱신한다.
  * output\_layer에 현재 상태(l.state)를 넣고, forward\_connected\_layer()를 호출하여 출력(output)을 계산한다.
  * input\_layer, self\_layer, output\_layer의 출력(delta)를 계산한다.
  * net.input을 다음 시간 스텝의 입력으로 이동한다.
  * input\_layer, self\_layer, output\_layer를 한 시간 스텝 앞으로 이동시킨다.
* 모든 시간 스텝에 대해 역전파를 수행하며, 다음을 수행한다:
  * output\_layer의 역전파(delta)를 계산한다.
  * self\_layer의 역전파(delta)를 계산한다.
  * input\_layer의 역전파(delta)를 계산한다.
  * input\_layer, self\_layer, output\_layer를 한 시간 스텝 뒤로 이동시킨다.



### update\_rnn\_layer

```c
void update_rnn_layer(layer l, update_args a)
{
    update_connected_layer(*(l.input_layer),  a);
    update_connected_layer(*(l.self_layer),   a);
    update_connected_layer(*(l.output_layer), a);
}
```

함수 이름: update\_rnn\_layer

입력:&#x20;

* layer l (RNN 레이어)
* update\_args a (가중치 업데이트에 필요한 인자들)

동작:&#x20;

* RNN 레이어의 입력 레이어, 자기 상태 레이어, 출력 레이어 각각에 대해 update\_connected\_layer 함수를 호출하여 가중치를 업데이트함.

설명:&#x20;

* 이 함수는 RNN 레이어의 가중치를 업데이트하기 위해 호출됩니다.&#x20;
* RNN은 입력 시퀀스를 처리할 때, 시퀀스 내 이전 시점에서의 자기 상태를 사용하므로, 입력 레이어, 자기 상태 레이어, 출력 레이어 각각에 대해 가중치를 업데이트해야 합니다.&#x20;
* 이를 위해 입력 레이어, 자기 상태 레이어, 출력 레이어에 대해 update\_connected\_layer 함수를 호출합니다.



### make\_rnn\_layer

```c
layer make_rnn_layer(int batch, int inputs, int outputs, int steps, ACTIVATION activation, int batch_normalize, int adam)
{
    fprintf(stderr, "RNN Layer: %d inputs, %d outputs\n", inputs, outputs);
    batch = batch / steps;
    layer l = {0};
    l.batch = batch;
    l.type = RNN;
    l.steps = steps;
    l.inputs = inputs;

    l.state = calloc(batch*outputs, sizeof(float));
    l.prev_state = calloc(batch*outputs, sizeof(float));

    l.input_layer = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.input_layer) = make_connected_layer(batch*steps, inputs, outputs, activation, batch_normalize, adam);
    l.input_layer->batch = batch;

    l.self_layer = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.self_layer) = make_connected_layer(batch*steps, outputs, outputs, activation, batch_normalize, adam);
    l.self_layer->batch = batch;

    l.output_layer = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.output_layer) = make_connected_layer(batch*steps, outputs, outputs, activation, batch_normalize, adam);
    l.output_layer->batch = batch;

    l.outputs = outputs;
    l.output = l.output_layer->output;
    l.delta = l.output_layer->delta;

    l.forward = forward_rnn_layer;
    l.backward = backward_rnn_layer;
    l.update = update_rnn_layer;

    return l;
}
```

함수 이름: make\_rnn\_layer

입력:

* batch: int형, 배치 크기
* inputs: int형, 입력 데이터의 차원 수
* outputs: int형, 출력 데이터의 차원 수
* steps: int형, 순환하는 단계 수
* activation: ACTIVATION 열거형, 활성화 함수
* batch\_normalize: int형, 배치 정규화 사용 여부 (1: 사용, 0: 미사용)
* adam: int형, Adam 알고리즘 사용 여부 (1: 사용, 0: 미사용)

동작:

* 입력값을 바탕으로 RNN 레이어를 생성하고 초기화한다.
* 입력값으로부터 연결된 레이어들을 생성하고 초기화한다.
* 생성된 레이어들 간에 상호 연결을 설정한다.
* 생성된 RNN 레이어를 반환한다.

설명:

* 입력값으로부터 RNN 레이어를 생성하고 초기화하는 함수이다.
* RNN 레이어는 입력 데이터를 받아서 순환 신경망 연산을 수행한다.
* 이 함수는 입력값을 바탕으로 연결된 입력 레이어, 자기 연결 레이어, 출력 레이어를 생성하고 초기화한다.
* 이 함수는 생성된 레이어들 간에 상호 연결을 설정한다.
* 이 함수는 생성된 RNN 레이어를 반환한다.

