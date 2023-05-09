# maxpool

### Max Pooling Layer 란?

Max Pooling Layer는 Convolutional Neural Network(CNN)의 구성 요소 중 하나입니다. CNN은 이미지, 음성 또는 비디오와 같은 입력 데이터를 처리할 때 사용됩니다. 이전의 Convolutional Layer에서 생성된 feature map을 다운 샘플링하여 공간 해상도를 줄이는 역할을 합니다. 이를 통해 네트워크가 과적합되는 것을 방지하고 연산 속도를 높일 수 있습니다.

Max Pooling Layer는 각 feature map에서 가장 큰 값을 선택하여 출력합니다. 이를 통해 feature map의 크기가 줄어들고, 이미지의 위치 이동에 대한 불변성(invariance)이 증가합니다. 예를 들어, 어떤 이미지 내에서 개의 얼굴이 있을 때, 개의 위치가 다르더라도 Max Pooling Layer는 개의 특징을 인식하도록 도와줍니다.

일반적으로, Max Pooling Layer는 2x2의 윈도우와 2의 스트라이드(stride)를 사용하여 작동합니다. 이는 입력 feature map을 2배로 다운샘플링하여 크기를 줄입니다. 그러나 윈도우 크기와 스트라이드 크기는 문제에 따라 다를 수 있습니다. Max Pooling Layer는 학습 가능한 매개 변수가 없기 때문에, 네트워크 파라미터 수를 줄이고 과적합을 방지하는 데 도움이 됩니다.



### make\_maxpool\_layer

```c
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l = {0};
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = (h + padding - size)/stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;

    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}
```

함수 이름: make\_maxpool\_layer&#x20;

입력:

* batch: int 형태의 배치 크기
* h: int 형태의 입력 이미지 높이
* w: int 형태의 입력 이미지 너비
* c: int 형태의 입력 이미지 채널 수
* size: int 형태의 맥스풀링 필터 크기
* stride: int 형태의 맥스풀링 스트라이드 크기
* padding: int 형태의 패딩 크기

동작:&#x20;

* 맥스풀링 레이어를 생성하고 초기화한 뒤 반환한다.&#x20;
* 출력 크기를 계산하고, 메모리를 동적으로 할당하여 초기화하며, 순전파와 역전파 함수를 설정한다.

설명:

* l: maxpool\_layer 구조체 변수
* l.type: 레이어 타입으로 MAXPOOL로 설정
* l.out\_c: 출력 이미지 채널 수로 입력 이미지 채널 수와 같게 설정
* l.outputs: 출력 이미지의 크기(높이x너비x채널 수)로 출력 크기 계산
* l.inputs: 입력 이미지의 크기(높이x너비x채널 수)로 입력 크기 계산
* l.indexes: 맥스풀링 연산에서 최댓값 위치를 기록하기 위한 배열 동적 할당
* l.output: 맥스풀링 연산 결과를 저장하기 위한 배열 동적 할당
* l.delta: 역전파 과정에서 계산된 그래디언트를 저장하기 위한 배열 동적 할당
* l.forward: 맥스풀링 레이어의 순전파 연산 함수 포인터 설정
* l.backward: 맥스풀링 레이어의 역전파 연산 함수 포인터 설정
* fprintf: 디버깅용으로 맥스풀링 레이어의 크기를 출력



### forward\_maxpool\_layer

```c
void forward_maxpool_layer(const maxpool_layer l, network net) { 
    int b,i,j,k,m,n; 
    int w_offset = -l.pad/2; 
    int h_offset = -l.pad/2;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? net.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l.output[out_index] = max;
                    l.indexes[out_index] = max_i;
                }
            }
        }
    }
}
```

함수 이름: forward\_maxpool\_layer

입력:

* const maxpool\_layer l: Max pooling 레이어를 나타내는 구조체
* network net: 신경망 구조를 나타내는 구조체

동작:&#x20;

* 주어진 Max pooling 레이어 구조체 l과 신경망 구조체 net를 사용하여 Max pooling 레이어를 순전파하는 함수이다. 이 함수는 입력값에서 Max pooling 연산을 수행하여 출력값을 계산한다. Max pooling 연산은 각 윈도우 내의 최댓값을 찾아 출력값으로 사용한다.

설명:&#x20;

* 이 함수는 Max pooling 레이어에서 사용되는 forward propagation 함수이다. 입력값으로는 Max pooling 레이어를 나타내는 구조체 l과 신경망 구조체 net이 주어진다. Max pooling 레이어의 출력값은 구조체 l의 output 배열에 저장된다. Max pooling 레이어의 출력값을 계산하기 위해, 입력값에서 Max pooling 연산을 수행하는데, 이를 위해 입력값에서 윈도우를 이동하면서 각 윈도우 내의 최댓값을 찾아 출력값으로 사용한다.
* 이 함수는 입력값의 배치 크기 l.batch, 출력값의 높이 l.out\_h, 출력값의 너비 l.out\_w, 채널 수 l.c, 패딩 크기 l.pad, 스트라이드 크기 l.stride, 윈도우 크기 l.size 등의 정보를 사용한다. 이 함수에서는 입력값에서 윈도우를 이동하면서 각 윈도우 내의 최댓값을 찾고, 구조체 l의 output 배열에 저장한다. 최댓값의 위치 정보를 저장하기 위해 구조체 l의 indexes 배열도 함께 업데이트한다.
* 이 함수는 for문을 사용하여 입력값의 모든 위치에 대해 Max pooling 연산을 수행한다. 구체적으로는 입력값의 배치별로, 채널별로, 출력값의 위치별로 입력값에서 윈도우를 이동하면서 최댓값을 찾는다. 입력값에서 윈도우를 이동하기 위해 h\_offset과 w\_offset을 사용하며, 이 값은 패딩 정보와 스트라이드 정보를 이용하여 계산된다. 최댓값을 찾을 때는 윈도우 내의 모든 값과 비교하면서 최댓값과 최댓값의 위치를 찾는다. 최댓값은 출력값으로 사용되며, 최댓값의 위치 정보는 indexes 배열에 저장된다.
* 마지막으로, 이 함수는 입력값과 출력값의 메모리 할당 및 해제 등을 수행한다.



### backward\_maxpool\_layer

```c
void backward_maxpool_layer(const maxpool_layer l, network net)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        net.delta[index] += l.delta[i];
    }
}
```

함수 이름: backward\_maxpool\_layer

입력:&#x20;

* maxpool\_layer l (maxpool 레이어 구조체)
* network net (뉴럴 네트워크 구조체)

동작:&#x20;

* maxpool 레이어의 역전파를 수행하여, 뉴럴 네트워크의 delta 값을 업데이트한다.&#x20;
* 역전파 수행을 위해, maxpool 레이어에서 사용된 max 값을 가지는 원소들의 인덱스를 l.indexes 배열에 저장해 놓았으며, 이를 이용하여 delta 값을 해당 인덱스의 원소에 더해준다.

설명:&#x20;

* 입력으로 들어온 maxpool 레이어 구조체 l과 뉴럴 네트워크 구조체 net을 이용하여, maxpool 레이어의 역전파를 수행한다.&#x20;
* l.indexes 배열에는 max 값을 가지는 원소들의 인덱스가 저장되어 있으므로, 이를 이용하여 net.delta 배열의 해당 인덱스의 원소에 l.delta 배열의 값을 더해준다.&#x20;
* 이렇게 하면, maxpool 레이어로 전달된 델타 값이 이전 레이어로 역전파되며, 네트워크의 학습이 이루어진다.



### get\_maxpool\_image

```c
image get_maxpool_image(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.output);
}
```

함수 이름: get\_maxpool\_image&#x20;

입력:&#x20;

* maxpool\_layer 구조체 l&#x20;

동작:&#x20;

* max pooling 레이어의 출력값을 이미지 형태로 변환하여 반환합니다.&#x20;

설명:

* max pooling 레이어의 출력값을 특정 크기의 이미지로 변환한 후, float\_to\_image 함수를 이용하여 image 형태로 변환하여 반환합니다.&#x20;
* 출력값을 이미지 형태로 변환하는 이유는 이미지를 시각화하여 max pooling 레이어가 어떤 작업을 수행하는지 쉽게 이해할 수 있기 때문입니다.



### get\_maxpool\_delta

```c
image get_maxpool_delta(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}
```

함수 이름: get\_maxpool\_delta

입력:&#x20;

* maxpool\_layer l (maxpool 레이어 구조체)

동작:&#x20;

* 입력으로 받은 maxpool 레이어의 출력값에 대한 delta 값을 가지고 새로운 이미지를 생성하여 반환합니다.&#x20;
* 생성된 이미지는 float\_to\_image 함수를 통해 float 형식으로 변환됩니다.

설명:&#x20;

* maxpool 레이어는 입력 이미지를 최대값 풀링 연산을 통해 축소한 후 출력값을 생성합니다.&#x20;
* 이때, 손실 함수의 역전파를 위해 이전 레이어에서 전달된 delta 값을 maxpool 레이어에서도 그대로 전달해야 합니다.&#x20;
* 이 함수는 이러한 delta 값을 이용하여 새로운 이미지를 생성하고 반환합니다. 이때, 생성된 이미지의 너비, 높이, 채널 수는 maxpool 레이어의 출력값과 동일합니다.



### resize\_maxpool\_layer

```c
void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + l->pad - l->size)/l->stride + 1;
    l->out_h = (h + l->pad - l->size)/l->stride + 1;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = realloc(l->indexes, output_size * sizeof(int));
    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));
}
```

함수 이름: resize\_maxpool\_layer 입력:

* maxpool\_layer \*l : maxpool 레이어 구조체 포인터
* int w : 변경할 너비
* int h : 변경할 높이

동작:&#x20;

* 입력받은 maxpool 레이어 구조체 포인터(\*l)의 w와 h 필드를 입력받은 값으로 변경하고, 그에 따라 다른 필드들도 새로 계산하여 업데이트한다. 새롭게 계산된 필드들은 다음과 같다:
* inputs : 입력 이미지 데이터의 총 개수(h \* w \* c)
* out\_w : 출력 이미지의 너비
* out\_h : 출력 이미지의 높이
* outputs : 출력 이미지 데이터의 총 개수(out\_w \* out\_h \* c)
* indexes : 출력값 중 최댓값의 위치를 저장하는 배열을 output\_size 크기로 재할당한다.
* output : 출력값을 저장하는 배열을 output\_size 크기로 재할당한다.
* delta : 출력값에 대한 미분값(gradient)을 저장하는 배열을 output\_size 크기로 재할당한다.

설명:&#x20;

* 이 함수는 maxpool 레이어의 크기를 조정하기 위한 함수이다.&#x20;
* 입력받은 maxpool 레이어 구조체 포인터(\*l)의 w와 h 필드를 변경한 뒤, 이 값에 따라 다른 필드들도 업데이트한다. 이 때, maxpool 레이어는 입력 이미지에서 stride와 size를 기반으로 최댓값을 추출하여 출력 이미지를 생성하는 레이어이다.&#x20;
* 이 함수에서는 입력 이미지 데이터의 총 개수(inputs), 출력 이미지의 너비(out\_w)와 높이(out\_h), 출력 이미지 데이터의 총 개수(outputs)를 새롭게 계산한다.&#x20;
* 또한, 출력값 중 최댓값의 위치를 저장하는 배열(indexes), 출력값을 저장하는 배열(output), 그리고 출력값에 대한 미분값(gradient)을 저장하는 배열(delta)을 재할당한다.&#x20;
* 이렇게 필요한 필드들을 새롭게 계산하고 재할당함으로써, maxpool 레이어의 크기를 조정할 수 있다.

