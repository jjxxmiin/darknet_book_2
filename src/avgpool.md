# avgpool

## Average Pooling Layer 란?

Feature Map의 평균 값을 계산해 전파시키는 Layer 입니다.

```c
//avgpool_layer.h

typedef layer avgpool_layer;
```

### make\_avgpool\_layer

```c
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    avgpool_layer l = {0};
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_avgpool_layer;
    l.backward = backward_avgpool_layer;

    return l;
}
```

함수 이름: make\_avgpool\_layer

입력:&#x20;

* batch: 배치 크기
* w: 너비
* h: 높이
* c: 채널 수

동작:&#x20;

* Average pooling 레이어를 생성합니다.

설명:&#x20;

* 이 함수는 Average pooling 레이어를 생성합니다.&#x20;
* 입력으로는 배치 크기(batch), 너비(w), 높이(h), 채널 수(c)를 받습니다.
* 함수는 먼저 생성된 레이어를 초기화하고, 필드에 각 값을 할당합니다. 그 다음, 출력 크기와 입력 크기를 계산하고, 메모리를 동적으로 할당합니다. 이 함수에서는 l.output과 l.delta를 메모리 할당합니다.
* 마지막으로, 레이어의 forward와 backward 함수를 각각 forward\_avgpool\_layer와 backward\_avgpool\_layer 함수로 설정하고, 레이어를 반환합니다.
* Average pooling 레이어는 입력 데이터를 정해진 영역으로 나누어 각 영역의 평균값을 계산합니다. 이를 통해 입력 데이터의 공간적인 정보를 유지하면서, 데이터의 크기를 줄일 수 있습니다. 이는 Convolutional Neural Network에서 특징 맵의 크기를 줄이는데에 주로 사용됩니다.



### forward\_avgpool\_layer

<pre class="language-c"><code class="lang-c"><strong>void forward_avgpool_layer(const avgpool_layer l, network net)
</strong>{
    int b,i,k;

    for(b = 0; b &#x3C; l.batch; ++b){
        for(k = 0; k &#x3C; l.c; ++k){
            int out_index = k + b*l.c;
            l.output[out_index] = 0;
            for(i = 0; i &#x3C; l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                l.output[out_index] += net.input[in_index];
            }
            l.output[out_index] /= l.h*l.w;
        }
    }
}
</code></pre>

함수 이름: forward\_avgpool\_layer

입력:&#x20;

* l: avgpool\_layer 구조체
* net: network 구조체

동작:&#x20;

* Average pooling 레이어의 forward 연산을 수행합니다.

설명:&#x20;

* 이 함수는 Average pooling 레이어의 forward 연산을 수행합니다. 입력으로는 avgpool\_layer 구조체와 network 구조체를 받습니다.
* 함수는 먼저 입력 데이터를 순회하면서, 입력 데이터를 필터 크기(h, w)로 나누어 평균값을 계산합니다. 이를 통해 출력 데이터의 크기를 줄입니다. 이후 평균값을 출력 데이터에 저장합니다.
* Average pooling 레이어는 입력 데이터를 정해진 영역으로 나누어 각 영역의 평균값을 계산합니다. 이를 통해 입력 데이터의 공간적인 정보를 유지하면서, 데이터의 크기를 줄일 수 있습니다. 이는 Convolutional Neural Network에서 특징 맵의 크기를 줄이는데에 주로 사용됩니다.



### backward\_avgpool\_layer

```c
void backward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                net.delta[in_index] += l.delta[out_index] / (l.h*l.w);
            }
        }
    }
}
```

함수 이름: backward\_avgpool\_layer

입력:

* l: avgpool 레이어 구조체&#x20;
* net: 네트워크 구조체

동작:

* 이 함수는 avgpool 레이어의 역전파(backpropagation)를 수행한다.&#x20;
* 입력값으로 avgpool 레이어 구조체 l과 네트워크 구조체 net을 받아들인다.&#x20;
* 각 배치(b)와 필터(k)에 대해, 델타값(delta)의 평균을 계산하고, 이를 각각의 입력값에 더해주어 역전파를 수행한다.&#x20;
* 이를 통해 avgpool 레이어의 입력값에 대한 미분값(gradient)을 계산할 수 있다.

설명:

* 이 함수는 avgpool 레이어의 역전파를 구현한 것이다.&#x20;
* avgpool 레이어는 입력값을 작은 사각 영역으로 나누어 평균값을 구한 후 출력값으로 내보내는 레이어이다. 따라서 이 함수에서는 각각의 입력값에 대한 미분값을 구하는 것이 핵심이다.&#x20;
* 델타값(delta)는 출력값과 동일한 차원을 가지고 있으며, 이 값은 이전 레이어의 미분값을 받아들이는 역할을 한다.&#x20;
* 역전파 과정에서는, 이전 레이어의 미분값과 현재 레이어의 출력값을 이용하여 현재 레이어의 입력값에 대한 미분값을 계산한다.&#x20;
* avgpool 레이어의 경우 입력값을 평균화하는 과정이 필요하므로, 델타값의 평균을 구하여 각각의 입력값에 더해주어야 한다.&#x20;
* 이를 위해 출력값의 인덱스(out\_index)와 입력값의 인덱스(in\_index)를 계산하여 값을 업데이트한다.



### resize\_avgpool\_layer

```c
void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
}
```

함수 이름: resize\_avgpool\_layer

입력:&#x20;

* l: avgpool\_layer 구조체 포인터
* w: 너비
* h: 높이

동작:&#x20;

* Average pooling 레이어의 입력 데이터 크기를 조정한다.

설명:&#x20;

* 이 함수는 Average pooling 레이어의 입력 데이터 크기를 조정하는데 사다.&#x20;
* 입력으로는 avgpool\_layer 구조체 포인터와 새로운 입력 이미지의 폭(w)과 높이(h)를 받는다.
* 이 함수는 Average pooling 레이어의 w, h, inputs 변수를 입력받은 값으로 갱신한다.&#x20;
* 이때, c는 그대로 유지된다.&#x20;
* Average pooling 레이어는 입력 데이터를 필터 크기(h, w)로 나누어 평균값을 계산하기 때문에, 입력 이미지 크기가 바뀌면 입력 데이터 크기(inputs)도 바뀌어야 한다.

된
