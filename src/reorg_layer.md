# reorg\_layer

## forward\_reorg\_layer

```c
void forward_reorg_layer(const layer l, network net)
{
    int i;
    if(l.flatten){
        memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
        if(l.reverse){
            flatten(l.output, l.w*l.h, l.c, l.batch, 0);
        }else{
            flatten(l.output, l.w*l.h, l.c, l.batch, 1);
        }
    } else if (l.extra) {
        for(i = 0; i < l.batch; ++i){
            copy_cpu(l.inputs, net.input + i*l.inputs, 1, l.output + i*l.outputs, 1);
        }
    } else if (l.reverse){
        reorg_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.output);
    } else {
        reorg_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 0, l.output);
    }
}
```

함수 이름: forward\_reorg\_layer

입력:&#x20;

* layer l (reorg layer의 정보를 담고 있는 구조체)
* network net (신경망 정보를 담고 있는 구조체)

동작:&#x20;

* reorg 레이어의 forward propagation을 수행한다. 입력 데이터를 reorg 연산을 통해 출력으로 변환한다.

설명:&#x20;

* 입력 데이터는 l.input에 저장되어 있으며, l.output에 출력이 저장된다. 만약 l.flatten이 참이면 입력 데이터를 flatten하고, l.reverse가 참이면 역순으로 정렬한다.&#x20;
* l.extra가 참이면 입력 데이터를 그대로 복사하고, 그렇지 않으면 입력 데이터를 reorg 함수로 처리하여 출력으로 저장한다.



## backward\_reorg\_layer

```c
void backward_reorg_layer(const layer l, network net)
{
    int i;
    if(l.flatten){
        memcpy(net.delta, l.delta, l.outputs*l.batch*sizeof(float));
        if(l.reverse){
            flatten(net.delta, l.w*l.h, l.c, l.batch, 1);
        }else{
            flatten(net.delta, l.w*l.h, l.c, l.batch, 0);
        }
    } else if(l.reverse){
        reorg_cpu(l.delta, l.w, l.h, l.c, l.batch, l.stride, 0, net.delta);
    } else if (l.extra) {
        for(i = 0; i < l.batch; ++i){
            copy_cpu(l.inputs, l.delta + i*l.outputs, 1, net.delta + i*l.inputs, 1);
        }
    }else{
        reorg_cpu(l.delta, l.w, l.h, l.c, l.batch, l.stride, 1, net.delta);
    }
}
```

함수 이름: backward\_reorg\_layer&#x20;

입력:&#x20;

* layer l (reorg layer의 정보를 담은 구조체)
* network net (현재의 신경망 구조)&#x20;

동작:&#x20;

* reorg 레이어의 역전파를 계산하고, 결과를 입력층으로 전달합니다.&#x20;

설명:

* l.flatten이 참일 경우, l.delta에서 net.delta로 데이터를 복사하고, l.reverse가 참이면 flatten(l.delta, l.w_l.h, l.c, l.batch, 1)를 호출합니다. 그렇지 않으면 flatten(l.delta, l.w_l.h, l.c, l.batch, 0)를 호출합니다.
* l.reverse가 참이면, reorg\_cpu(l.delta, l.w, l.h, l.c, l.batch, l.stride, 0, net.delta)를 호출합니다.
* l.extra가 참일 경우, l.delta에서 net.delta로 데이터를 복사합니다.
* 그렇지 않으면, reorg\_cpu(l.delta, l.w, l.h, l.c, l.batch, l.stride, 1, net.delta)를 호출합니다.

## resize\_reorg\_layer

```c
void resize_reorg_layer(layer *l, int w, int h)
{
    int stride = l->stride;
    int c = l->c;

    l->h = h;
    l->w = w;

    if(l->reverse){
        l->out_w = w*stride;
        l->out_h = h*stride;
        l->out_c = c/(stride*stride);
    }else{
        l->out_w = w/stride;
        l->out_h = h/stride;
        l->out_c = c*(stride*stride);
    }

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->outputs;
    int output_size = l->outputs * l->batch;

    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));
}
```

함수 이름: resize\_reorg\_layer

입력:

* layer \*l: reorg layer의 포인터
* int w: 새로운 가로 크기
* int h: 새로운 세로 크기

동작:&#x20;

* reorg layer의 가로, 세로 크기를 새로운 값으로 업데이트하고, 입력 이미지를 reorg한 결과의 크기를 다시 계산합니다.&#x20;
* 이후, output과 delta 배열의 크기를 업데이트합니다.

설명:&#x20;

* 이 함수는 입력 이미지를 reorg하는 layer의 가로, 세로 크기를 업데이트하고, output과 delta 배열의 크기를 다시 할당합니다.&#x20;
* 이 때, reverse 플래그에 따라 가로, 세로 크기를 조절합니다. 함수 내에서는 입력 이미지를 reorg한 결과의 가로, 세로, 채널 수를 계산합니다.&#x20;
* 이후, output과 delta 배열의 크기를 업데이트하며, realloc 함수를 사용하여 메모리를 다시 할당합니다.



## make\_reorg\_layer

```c
layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra)
{
    layer l = {0};
    l.type = REORG;
    l.batch = batch;
    l.stride = stride;
    l.extra = extra;
    l.h = h;
    l.w = w;
    l.c = c;
    l.flatten = flatten;
    if(reverse){
        l.out_w = w*stride;
        l.out_h = h*stride;
        l.out_c = c/(stride*stride);
    }else{
        l.out_w = w/stride;
        l.out_h = h/stride;
        l.out_c = c*(stride*stride);
    }
    l.reverse = reverse;

    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    if(l.extra){
        l.out_w = l.out_h = l.out_c = 0;
        l.outputs = l.inputs + l.extra;
    }

    if(extra){
        fprintf(stderr, "reorg              %4d   ->  %4d\n",  l.inputs, l.outputs);
    } else {
        fprintf(stderr, "reorg              /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n",  stride, w, h, c, l.out_w, l.out_h, l.out_c);
    }
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));

    l.forward = forward_reorg_layer;
    l.backward = backward_reorg_layer;

    return l;
}
```

함수 이름: make\_reorg\_layer&#x20;

입력:

* batch (int): 배치 크기
* w (int): 입력 너비
* h (int): 입력 높이
* c (int): 입력 채널 수
* stride (int): 스트라이드 값
* reverse (int): 역방향 여부
* flatten (int): 플래튼 여부
* extra (int): 추가 값

동작:&#x20;

* Reorg 레이어를 생성하고 초기화합니다.

설명:

* 입력 크기와 스트라이드 값을 사용하여 출력 크기를 계산합니다.
* reverse 값이 true이면 출력 크기는 w와 h를 stride배 확대하고 c를 stride^2로 축소합니다. false이면 출력 크기는 w와 h를 stride배 축소하고 c를 stride^2배 확대합니다.
* flatten 값이 true이면 입력값이 평탄화(flatten)된 것으로 간주하고, delta 값을 복사합니다.
* extra 값이 있는 경우, 출력 크기 대신에 extra값을 outputs으로 사용합니다.
* output과 delta 배열을 동적 할당하고, forward와 backward 함수를 설정합니다.
* 생성된 Reorg 레이어를 반환합니다.

