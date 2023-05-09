# crnn\_layer

cnn과 rnn을 결합한 layer 입니다.

rnn에서 fully connected 연산을 convolutional 연산으로 바뀌어진 것 외에 딱히 변화가 없습니다.

## increment\_layer

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

* layer 포인터 l
* int steps

동작:&#x20;

* l의 output, delta, x, x\_norm 포인터를 steps \* l->outputs \* l->batch 만큼 증가시킴

설명:&#x20;

* 이 함수는 미니배치 처리를 위해 필요한 함수 중 하나로, 각 레이어의 포인터를 미니배치에 따라 적절히 이동시켜주는 역할을 합니다.&#x20;
* 이동시켜야 하는 양은 steps \* l->outputs \* l->batch 로 계산됩니다.&#x20;
* 이 함수를 사용하면 한 번에 처리해야 하는 미니배치의 크기를 조절할 수 있습니다.



## forward\_crnn\_layer

```c
void forward_crnn_layer(layer l, network net)
{
    network s = net;
    s.train = net.train;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);

    fill_cpu(l.outputs * l.batch * l.steps, 0, output_layer.delta, 1);
    fill_cpu(l.hidden * l.batch * l.steps, 0, self_layer.delta, 1);
    fill_cpu(l.hidden * l.batch * l.steps, 0, input_layer.delta, 1);
    if(net.train) fill_cpu(l.hidden * l.batch, 0, l.state, 1);

    for (i = 0; i < l.steps; ++i) {
        s.input = net.input;
        forward_convolutional_layer(input_layer, s);

        s.input = l.state;
        forward_convolutional_layer(self_layer, s);

        float *old_state = l.state;
        if(net.train) l.state += l.hidden*l.batch;
        if(l.shortcut){
            copy_cpu(l.hidden * l.batch, old_state, 1, l.state, 1);
        }else{
            fill_cpu(l.hidden * l.batch, 0, l.state, 1);
        }
        axpy_cpu(l.hidden * l.batch, 1, input_layer.output, 1, l.state, 1);
        axpy_cpu(l.hidden * l.batch, 1, self_layer.output, 1, l.state, 1);

        s.input = l.state;
        forward_convolutional_layer(output_layer, s);

        net.input += l.inputs*l.batch;
        increment_layer(&input_layer, 1);
        increment_layer(&self_layer, 1);
        increment_layer(&output_layer, 1);
    }
}
```

함수 이름: forward\_crnn\_layer

입력:

* layer l: CRNN 레이어
* network net: 레이어가 속한 네트워크

동작:

* CRNN 레이어의 forward 연산을 수행한다.
* 입력 데이터를 한 스텝씩 처리하며, 입력 레이어, self 레이어, 출력 레이어를 차례대로 거친다.
* 각 스텝에서 입력, self 레이어의 출력을 더하여 state를 구하고, 출력 레이어를 거쳐 출력을 계산한다.
* 각 스텝에서 사용된 레이어의 인덱스를 1씩 증가시킨다.

설명:

* CRNN(Convolutional Recurrent Neural Network)은 컨볼루션 레이어와 순환 레이어가 결합된 구조를 가지는 딥러닝 모델이다.
* 이 함수는 CRNN 레이어의 forward 연산을 수행하는 함수이다.
* 입력으로는 CRNN 레이어와 레이어가 속한 네트워크가 들어온다.
* 함수 내부에서는 입력 데이터를 한 스텝씩 처리하며, 입력 레이어, self 레이어, 출력 레이어를 차례대로 거친다.
* 각 스텝에서 입력, self 레이어의 출력을 더하여 state를 구하고, 출력 레이어를 거쳐 출력을 계산한다.
* 함수 내부에서는 각 스텝에서 사용된 레이어의 인덱스를 1씩 증가시킨다.



## backward\_crnn\_layer

```c
void backward_crnn_layer(layer l, network net)
{
    network s = net;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);

    increment_layer(&input_layer, l.steps-1);
    increment_layer(&self_layer, l.steps-1);
    increment_layer(&output_layer, l.steps-1);

    l.state += l.hidden*l.batch*l.steps;
    for (i = l.steps-1; i >= 0; --i) {
        copy_cpu(l.hidden * l.batch, input_layer.output, 1, l.state, 1);
        axpy_cpu(l.hidden * l.batch, 1, self_layer.output, 1, l.state, 1);

        s.input = l.state;
        s.delta = self_layer.delta;
        backward_convolutional_layer(output_layer, s);

        l.state -= l.hidden*l.batch;
        /*
           if(i > 0){
           copy_cpu(l.hidden * l.batch, input_layer.output - l.hidden*l.batch, 1, l.state, 1);
           axpy_cpu(l.hidden * l.batch, 1, self_layer.output - l.hidden*l.batch, 1, l.state, 1);
           }else{
           fill_cpu(l.hidden * l.batch, 0, l.state, 1);
           }
         */

        s.input = l.state;
        s.delta = self_layer.delta - l.hidden*l.batch;
        if (i == 0) s.delta = 0;
        backward_convolutional_layer(self_layer, s);

        copy_cpu(l.hidden*l.batch, self_layer.delta, 1, input_layer.delta, 1);
        if (i > 0 && l.shortcut) axpy_cpu(l.hidden*l.batch, 1, self_layer.delta, 1, self_layer.delta - l.hidden*l.batch, 1);
        s.input = net.input + i*l.inputs*l.batch;
        if(net.delta) s.delta = net.delta + i*l.inputs*l.batch;
        else s.delta = 0;
        backward_convolutional_layer(input_layer, s);

        increment_layer(&input_layer, -1);
        increment_layer(&self_layer, -1);
        increment_layer(&output_layer, -1);
    }
}
```

함수 이름: backward\_crnn\_layer

입력:&#x20;

* layer l: 역전파를 수행할 CRNN 레이어
* network net: 레이어를 포함하는 네트워크

동작:&#x20;

* CRNN 레이어의 역전파를 수행합니다.&#x20;
* 먼저, 입력 레이어, self 레이어, output 레이어에 대한 포인터를 초기화합니다.&#x20;
* 그런 다음, l.steps 번 반복하면서 각 스텝에서 다음을 수행합니다.&#x20;
* 입력 레이어와 self 레이어의 출력 값을 합쳐서 l.state에 저장한 후, 출력 레이어의 역전파를 수행합니다.&#x20;
* 그 후, self 레이어의 역전파를 수행하고, 이전 스텝의 self 레이어 업데이트 델타를 현재 스텝의 입력 레이어 업데이트 델타로 복사합니다.&#x20;
* 마지막으로, 현재 스텝의 입력 데이터에 대한 역전파를 수행합니다.

설명:&#x20;

* 이 함수는 CRNN 레이어의 역전파를 수행하는 함수로, 네트워크가 학습 중인 경우에 사용됩니다.&#x20;
* l은 역전파를 수행할 레이어를 나타내는 layer 구조체이며, net은 레이어를 포함하는 네트워크를 나타내는 network 구조체입니다.&#x20;
* 이 함수는 각 레이어의 출력 값을 계산하고 델타 값을 업데이트합니다.



## update\_crnn\_layer

```c
void update_crnn_layer(layer l, update_args a)
{
    update_convolutional_layer(*(l.input_layer),  a);
    update_convolutional_layer(*(l.self_layer),   a);
    update_convolutional_layer(*(l.output_layer), a);
}
```

함수 이름: update\_crnn\_layer

입력:

* layer l: 업데이트할 CRNN 레이어
* update\_args a: 업데이트에 사용할 인자들 (learning rate, momentum 등)

동작:&#x20;

* 주어진 업데이트 인자들을 사용하여 입력으로 주어진 CRNN 레이어의 input\_layer, self\_layer, output\_layer를 각각 업데이트하는 함수입니다.
* update\_convolutional\_layer 함수를 호출하여 각 레이어를 업데이트합니다.

설명:&#x20;

* CRNN 레이어는 입력 시퀀스를 처리하기 위한 컨볼루션 레이어와 RNN 레이어의 결합입니다.&#x20;
* 이 함수는 그 중 컨볼루션 레이어를 업데이트하는 함수입니다.&#x20;
* 이 함수는 입력으로 받은 update\_args를 사용하여 각 레이어의 파라미터를 업데이트합니다.&#x20;
* 먼저, input\_layer, self\_layer, output\_layer 각각에 대해 update\_convolutional\_layer 함수를 호출하여 그 레이어의 파라미터를 업데이트합니다.&#x20;
* 이 함수는 컨볼루션 레이어의 파라미터를 업데이트하기 위해 사용되는 함수입니다.



## make\_crnn\_layer

```c
layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int steps, ACTIVATION activation, int batch_normalize)
{
    fprintf(stderr, "CRNN Layer: %d x %d x %d image, %d filters\n", h,w,c,output_filters);
    batch = batch / steps;
    layer l = {0};
    l.batch = batch;
    l.type = CRNN;
    l.steps = steps;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_h = h;
    l.out_w = w;
    l.out_c = output_filters;
    l.inputs = h*w*c;
    l.hidden = h * w * hidden_filters;
    l.outputs = l.out_h * l.out_w * l.out_c;

    l.state = calloc(l.hidden*batch*(steps+1), sizeof(float));

    l.input_layer = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.input_layer) = make_convolutional_layer(batch*steps, h, w, c, hidden_filters, 1, 3, 1, 1,  activation, batch_normalize, 0, 0, 0);
    l.input_layer->batch = batch;

    l.self_layer = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.self_layer) = make_convolutional_layer(batch*steps, h, w, hidden_filters, hidden_filters, 1, 3, 1, 1,  activation, batch_normalize, 0, 0, 0);
    l.self_layer->batch = batch;

    l.output_layer = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.output_layer) = make_convolutional_layer(batch*steps, h, w, hidden_filters, output_filters, 1, 3, 1, 1,  activation, batch_normalize, 0, 0, 0);
    l.output_layer->batch = batch;

    l.output = l.output_layer->output;
    l.delta = l.output_layer->delta;

    l.forward = forward_crnn_layer;
    l.backward = backward_crnn_layer;
    l.update = update_crnn_layer;

    return l;
}
```

함수 이름: make\_crnn\_layer

입력:

* int batch: 배치 크기
* int h: 입력 이미지 높이
* int w: 입력 이미지 너비
* int c: 입력 이미지 채널 수
* int hidden\_filters: 숨겨진 레이어에서 사용되는 필터 수
* int output\_filters: 출력 레이어에서 사용되는 필터 수
* int steps: 시퀀스 길이 (스텝 수)
* ACTIVATION activation: 활성화 함수 유형
* int batch\_normalize: 배치 정규화 여부

동작:&#x20;

* CRNN 레이어를 만들고 초기화합니다.

설명:&#x20;

* 이 함수는 입력 이미지의 높이, 너비, 채널 수 및 시퀀스 길이와 같은 인수를 사용하여 CRNN(Convolutional Recurrent Neural Network) 레이어를 만듭니다.&#x20;
* 이 레이어는 숨겨진 레이어와 출력 레이어 각각에 대해 3x3 커널과 같은 하이퍼파라미터를 사용한 1D 컨볼루션 레이어를 포함합니다.&#x20;
* 이 함수는 이러한 레이어를 만들고 초기화한 후 CRNN 레이어를 반환합니다.

