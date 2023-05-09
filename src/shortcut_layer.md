# shortcut\_layer

## shortcut layer 란?

ResNet에서 제안된 skip connection과 유사합니다.

잠시 출력을 저장하고 그 후에 layer의 출력과 합치는 작업에서 사용 됩니다.

## shortcut.c

### forward\_shortcut\_layer

```c
void forward_shortcut_layer(const layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);                                                                  // network input -> layer output
    shortcut_cpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output);  // layer output += i-th layer output
    activate_array(l.output, l.outputs*l.batch, l.activation);
}
```

함수 이름: forward\_shortcut\_layer

입력:

* const layer l: 현재 layer 정보
* network net: 현재 network 정보

동작:

* 현재 layer의 출력값으로 네트워크 입력값을 복사
* 현재 layer의 출력값에 shortcut 연결된 이전 layer의 출력값을 더해줌
* 현재 layer의 출력값에 활성화 함수를 적용

설명:

* shortcut 연결을 통해 다른 layer의 출력값을 현재 layer의 출력값에 더해줌으로써, 네트워크의 학습 효율성을 높이기 위한 레이어
* forward\_shortcut\_layer 함수는 해당 layer의 forward propagation을 수행하며, 입력값을 현재 layer의 출력값으로 복사하고 shortcut 연결된 이전 layer의 출력값을 더해주며 활성화 함수를 적용하는 역할을 수행함



### backward\_shortcut\_layer

```c
void backward_shortcut_layer(const layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);                                                       // layer delta -> activation grad
    axpy_cpu(l.outputs*l.batch, l.alpha, l.delta, 1, net.delta, 1);                                                           // network delta += alpha * layer delta
    shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta);           // i-th layer delta += layer delta
}
```

함수 이름: backward\_shortcut\_layer

입력:

* const layer l: shortcut layer의 정보를 담고 있는 구조체
* network net: 신경망을 구성하는 layer들의 정보를 담고 있는 구조체

동작:

* layer output의 activation gradient를 계산하여 layer delta에 저장한다.
* network delta에 alpha값과 layer delta값을 곱하여 더해준다.
* i-th layer delta에는 beta값과 layer delta값을 곱하여 더해준다.

설명:&#x20;

* Shortcut layer는 입력값과 이전 layer의 출력값을 더하여 출력값을 만들어낸다.&#x20;
* 따라서 forward pass에서는 이전 layer의 출력값을 현재 layer의 입력값과 더하여 출력값을 계산하게 된다.&#x20;
* Backward pass에서는 현재 layer의 출력값에 대한 activation gradient를 계산하고, 이전 layer의 delta값에도 현재 layer의 delta값을 더하여 전파하게 된다.



### resize\_shortcut\_layer

```c
void resize_shortcut_layer(layer *l, int w, int h)
{
    assert(l->w == l->out_w);
    assert(l->h == l->out_h);
    l->w = l->out_w = w;
    l->h = l->out_h = h;
    l->outputs = w*h*l->out_c;
    l->inputs = l->outputs;
    l->delta =  realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = realloc(l->output, l->outputs*l->batch*sizeof(float));
}
```

함수 이름: resize\_shortcut\_layer

입력:

* layer \*l: 크기를 조정할 shortcut layer의 포인터
* int w: 새로운 너비
* int h: 새로운 높이

동작:

* l의 w와 out\_w가 같아야 함을 확인(assert)
* l의 h와 out\_h가 같아야 함을 확인(assert)
* l의 w와 out\_w를 w로 업데이트
* l의 h와 out\_h를 h로 업데이트
* l의 outputs를 w, h, out\_c의 곱으로 업데이트
* l의 inputs를 outputs와 같게 업데이트
* l의 delta 메모리를 outputs \* batch 크기만큼 재할당
* l의 output 메모리를 outputs \* batch 크기만큼 재할당

설명:&#x20;

* 이 함수는 shortcut layer의 크기를 조정하는 역할을 한다.&#x20;
* shortcut layer는 input과 output의 크기가 같아야 하기 때문에 l의 w와 out\_w, h와 out\_h가 같은지 확인하고 같지 않으면 에러를 발생시킨다.&#x20;
* 그 후 w와 h로 각각 크기를 조정해주고, outputs와 inputs를 업데이트한다.&#x20;
* 마지막으로, delta와 output 메모리를 새로운 outputs \* batch 크기로 재할당한다.



### make\_shortcut\_layer

```c
layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
    fprintf(stderr, "res  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d\n",index, w2,h2,c2, w,h,c);
    layer l = {0};
    l.type = SHORTCUT;
    l.batch = batch;
    l.w = w2;
    l.h = h2;
    l.c = c2;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w*h*c;
    l.inputs = l.outputs;

    l.index = index;

    l.delta =  calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));;

    l.forward = forward_shortcut_layer;
    l.backward = backward_shortcut_layer;

    return l;
}
```

함수 이름: make\_shortcut\_layer

입력:

* batch: 배치 크기
* index: 레이어 인덱스
* w: 입력 이미지 가로 크기
* h: 입력 이미지 세로 크기
* c: 입력 이미지 채널 수
* w2: shortcut 연결되는 레이어의 가로 크기
* h2: shortcut 연결되는 레이어의 세로 크기
* c2: shortcut 연결되는 레이어의 채널 수

동작:

* shortcut 레이어를 생성하고, 필드 값들을 초기화한다.

설명:

* shortcut 레이어는 skip connection을 구현하는 데 사용되는 레이어이다.
* 입력 이미지의 크기와 shortcut으로 연결되는 레이어의 출력 크기가 같은 경우에 사용된다.
* 출력 크기는 입력 이미지의 크기와 같고, 입력 이미지와 shortcut으로 연결되는 레이어의 출력을 더한 결과가 출력값이 된다.
* l.delta와 l.output은 모두 출력값을 저장하는 배열이다.
* l.forward와 l.backward는 해당 레이어에서의 순전파와 역전파 연산을 수행하는 함수 포인터이다.
* fprintf 함수를 사용하여 현재 shortcut 레이어의 정보를 출력한다.

