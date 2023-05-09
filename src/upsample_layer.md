# upsample\_layer

## Upsample Layer란?

Upsample Layer는 Feature Maps의 크기를 키우는 Layer입니다.

***

## upsample\_layer.c

### forward\_upsample\_layer

```c
void forward_upsample_layer(const layer l, network net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    if(l.reverse){
        upsample_cpu(l.output, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input);
    }else{
        upsample_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output);
    }
}
```

함수 이름: forward\_upsample\_layer

입력:&#x20;

* l: layer 구조체
* net: network 구조체

동작:&#x20;

* 업샘플링 레이어를 전방향 패스(forward pass)로 처리하는 함수입니다.&#x20;
* 입력 이미지를 업샘플링하여 출력값을 계산합니다. 이때, 출력값은 l.output에 저장됩니다.

설명:

* fill\_cpu() 함수는 l.output을 0으로 초기화합니다.
* l.reverse가 참(True)일 경우, upsample\_cpu() 함수를 사용하여 l.output에 업샘플링된 이미지를 저장합니다. 이때, 입력 이미지는 net.input입니다.
* l.reverse가 거짓(False)일 경우, upsample\_cpu() 함수를 사용하여 net.input 이미지를 업샘플링하여 l.output에 저장합니다. 이때, 출력 크기는 l.w, l.h로 설정되어 있습니다.
* 최종적으로, 출력값은 l.output에 저장됩니다.



### backward\_upsample\_layer

```c
void backward_upsample_layer(const layer l, network net)
{
    if(l.reverse){
        upsample_cpu(l.delta, l.out_w, l.out_h, l.c, l.batch, l.stride, 1, l.scale, net.delta);
    }else{
        upsample_cpu(net.delta, l.w, l.h, l.c, l.batch, l.stride, 0, l.scale, l.delta);
    }
}
```

함수 이름: backward\_upsample\_layer

입력:&#x20;

* l: layer 구조체
* net: network 구조체

동작:&#x20;

* 업샘플링 레이어의 역전파(backward pass) 연산을 수행합니다.&#x20;
* 입력값으로는 l과 net을 받으며, l은 현재 레이어의 정보와 이전 레이어의 출력값에 대한 정보를 담고 있으며, net은 전체 네트워크에 대한 정보를 담고 있습니다.&#x20;
* 업샘플링 레이어는 입력값의 크기를 늘리는 작업을 수행하는데, 이를 역전파 할 때는 이전 레이어의 delta 값을 현재 레이어의 크기에 맞게 축소(upsample)하여 전달합니다. 이 과정에서 scale 값을 사용하여 크기를 조절합니다.

설명:&#x20;

* 이 함수는 Darknet 딥러닝 프레임워크의 업샘플링 레이어에 대한 역전파 연산을 담당합니다.&#x20;
* Darknet에서는 입력값의 크기를 늘리는 작업을 수행하는 데에 업샘플링 레이어를 사용하며, 이 레이어의 출력값은 다음 레이어의 입력값으로 사용됩니다. 따라서 이전 레이어의 delta 값을 현재 레이어의 크기에 맞게 축소하여 전달해야 합니다.&#x20;
* 이를 위해 upsample\_cpu() 함수를 사용하며, 이 함수에서는 입력값의 크기를 늘리는 작업을 수행합니다.&#x20;
* scale 값을 사용하여 크기를 조절하며, reverse 값이 1인 경우에는 이전 레이어의 delta 값을 축소하여 전달하고, 0인 경우에는 net에서 이전 레이어의 delta 값을 가져와서 크기를 늘린 후 현재 레이어의 delta 값으로 사용합니다.



### resize\_upsample\_layer

```c
void resize_upsample_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->out_w = w*l->stride;
    l->out_h = h*l->stride;
    if(l->reverse){
        l->out_w = w/l->stride;
        l->out_h = h/l->stride;
    }
    l->outputs = l->out_w*l->out_h*l->out_c;
    l->inputs = l->h*l->w*l->c;
    l->delta =  realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = realloc(l->output, l->outputs*l->batch*sizeof(float));  
}
```

함수 이름: resize\_upsample\_layer

입력:

* layer \*l: upsample 레이어의 포인터
* int w: upsample 레이어의 출력 이미지 가로 길이
* int h: upsample 레이어의 출력 이미지 세로 길이

동작:&#x20;

* upsample 레이어의 출력 이미지 가로, 세로 길이를 조절하는 함수이다.&#x20;
* 입력으로 받은 가로, 세로 길이로 레이어의 가로, 세로 길이를 업데이트하고, 이를 기반으로 출력 이미지 가로, 세로 길이를 다시 계산한다.&#x20;
* 만약 레이어가 reverse 모드이면 출력 이미지 가로, 세로 길이는 입력 이미지 가로, 세로 길이에 stride를 나눈 값이 되고, outputs 값도 이에 맞게 계산된다.&#x20;
* 그리고 레이어의 입력 채널 수에 맞게 outputs 값을 계산한다. 마지막으로 delta와 output을 업데이트한다.

설명:&#x20;

* upsample 레이어는 입력 이미지를 확대하는 역할을 한다. 이 함수는 upsample 레이어의 출력 이미지 가로, 세로 길이를 조정하는 함수이다.&#x20;
* 이 함수는 l->w와 l->h를 입력으로 받은 가로, 세로 길이로 변경하고, 이를 기반으로 l->out\_w와 l->out\_h를 다시 계산한다.&#x20;
* 이때, l->out\_w와 l->out\_h는 upsample 레이어의 출력 이미지의 가로, 세로 길이이다.&#x20;
* 만약 upsample 레이어가 reverse 모드이면, l->out\_w와 l->out\_h는 입력 이미지의 가로, 세로 길이에 stride를 나눈 값이 된다.&#x20;
* 그리고 l->outputs 값도 이에 맞게 계산된다. l->outputs는 upsample 레이어의 출력 채널 수에 맞게 출력 이미지의 픽셀 수를 나타낸다.&#x20;
* 마지막으로 delta와 output을 업데이트하여 메모리를 할당한다.

### make\_upsample\_layer

```c
layer make_upsample_layer(int batch, int w, int h, int c, int stride)
{
    layer l = {0};
    l.type = UPSAMPLE;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w*stride;
    l.out_h = h*stride;
    l.out_c = c;
    if(stride < 0){
        stride = -stride;
        l.reverse=1;
        l.out_w = w/stride;
        l.out_h = h/stride;
    }
    l.stride = stride;
    l.outputs = l.out_w*l.out_h*l.out_c;
    l.inputs = l.w*l.h*l.c;
    l.delta =  calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));;

    l.forward = forward_upsample_layer;
    l.backward = backward_upsample_layer;

    if(l.reverse) fprintf(stderr, "downsample         %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    else fprintf(stderr, "upsample           %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}
```

함수 이름: make\_upsample\_layer

입력:

* batch: int 형식의 배치 크기
* w: int 형식의 입력 너비
* h: int 형식의 입력 높이
* c: int 형식의 입력 채널 수
* stride: int 형식의 업샘플링 배율 또는 다운샘플링 배율(음수)

동작:

* 입력으로 받은 batch, w, h, c, stride 값을 사용하여 upsample 레이어를 생성한다.
* stride 값이 음수인 경우 다운샘플링으로 동작한다.
* 레이어의 출력 크기(out\_w, out\_h, out\_c)와 출력 요소 수(outputs)를 계산한다.
* 레이어의 입력 요소 수(inputs), 델타(delta), 출력(output)을 메모리에 할당하고, 0으로 초기화한다.
* 업샘플링 레이어의 forward와 backward 함수를 설정한다.
* 업샘플링 레이어의 정보를 출력한다.

설명:

* 업샘플링 레이어를 만드는 함수이다.
* 이 함수에서 생성된 레이어는 입력 이미지를 upsample 또는 downsample하여 출력하는 역할을 한다.
* 입력 이미지의 크기를 키우는 upsample과 크기를 줄이는 downsample이 있다.
* 이 함수에서는 stride 값을 이용하여 upsample과 downsample을 구분하고, 역전파 함수에서 해당 연산을 처리한다.
* 레이어의 출력 크기는 입력 크기와 stride 값에 따라 결정된다.
* 이 함수에서는 메모리 할당과 초기화를 수행하고, forward와 backward 함수를 설정하는 작업을 한다.
