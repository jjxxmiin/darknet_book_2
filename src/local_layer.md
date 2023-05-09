# local\_layer

## local\_out\_height

```c
int local_out_height(local_layer l)
{
    int h = l.h;
    if (!l.pad) h -= l.size;
    else h -= 1;
    return h/l.stride + 1;
}
```

함수 이름: local\_out\_height&#x20;

입력:&#x20;

* local\_layer l: 로컬 레이어 구조체

동작:&#x20;

* 입력으로 받은 로컬 레이어의 높이(height)에 대한 출력 높이(output height)를 계산한다.&#x20;
* 패딩(padding)이 적용되어 있지 않은 경우 필터(filter) 크기(size)만큼 높이를 줄이고, 패딩이 적용된 경우 높이에서 1만큼 빼준다.&#x20;
* 그리고 나서 출력 높이를 계산하기 위해 stride로 나누고 1을 더해준다.&#x20;

설명:&#x20;

* 이 함수는 로컬 레이어의 출력 높이를 계산하는 함수로, 필터와 입력 데이터의 크기, 스트라이드 등의 정보를 이용해 계산한다.&#x20;
* 이 계산은 로컬 레이어의 순전파(forward propagation) 단계에서 필요하며, 출력 높이를 계산하는 것은 출력 데이터의 크기를 결정하는 중요한 요소 중 하나이다.



## local\_out\_width

```c
int local_out_width(local_layer l)
{
    int w = l.w;
    if (!l.pad) w -= l.size;
    else w -= 1;
    return w/l.stride + 1;
}
```

함수 이름: local\_out\_width&#x20;

입력:&#x20;

* local\_layer l (로컬 레이어 구조체)&#x20;

동작:&#x20;

* 로컬 레이어의 출력 너비를 계산하여 반환합니다.&#x20;

설명:&#x20;

* 입력 이미지에 대해 로컬 필터링을 수행한 후 출력 이미지의 너비를 계산합니다.&#x20;
* 너비는 패딩이 적용된 경우 입력 너비에서 필터 크기를 뺀 값에 1을 더한 후, 스트라이드로 나누어 계산됩니다.



## forward\_local\_layer

```c
void forward_local_layer(const local_layer l, network net)
{
    int out_h = local_out_height(l);
    int out_w = local_out_width(l);
    int i, j;
    int locations = out_h * out_w;

    for(i = 0; i < l.batch; ++i){
        copy_cpu(l.outputs, l.biases, 1, l.output + i*l.outputs, 1);
    }

    for(i = 0; i < l.batch; ++i){
        float *input = net.input + i*l.w*l.h*l.c;
        im2col_cpu(input, l.c, l.h, l.w,
                l.size, l.stride, l.pad, net.workspace);
        float *output = l.output + i*l.outputs;
        for(j = 0; j < locations; ++j){
            float *a = l.weights + j*l.size*l.size*l.c*l.n;
            float *b = net.workspace + j;
            float *c = output + j;

            int m = l.n;
            int n = 1;
            int k = l.size*l.size*l.c;

            gemm(0,0,m,n,k,1,a,k,b,locations,1,c,locations);
        }
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
}
```

함수 이름: forward\_local\_layer&#x20;

입력:&#x20;

* const local\_layer l&#x20;
* network net&#x20;

동작:&#x20;

* 로컬 레이어의 순전파 연산을 수행합니다.&#x20;
* 입력 데이터를 im2col 방식으로 전처리하고, 커널과의 행렬곱을 계산하여 출력값을 얻습니다.&#x20;
* 마지막으로 활성화 함수를 적용합니다.&#x20;

설명:

* l: 로컬 레이어의 정보를 담고 있는 구조체
* net: 네트워크 정보를 담고 있는 구조체
* out\_h: 출력값의 높이
* out\_w: 출력값의 너비
* locations: 출력값의 전체 크기
* biases: 로컬 레이어의 편향값
* input: 네트워크의 입력 데이터
* output: 로컬 레이어의 출력값
* weights: 로컬 레이어의 가중치값
* a: 커널과 입력값을 행렬곱하기 위한 배열
* b: im2col 방식으로 전처리된 입력값
* c: 출력값을 저장하기 위한 배열
* m, n, k: 행렬곱을 위한 매개변수
* activate\_array: 활성화 함수를 적용하는 함수



## backward\_local\_layer

```c
void backward_local_layer(local_layer l, network net)
{
    int i, j;
    int locations = l.out_w*l.out_h;

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    for(i = 0; i < l.batch; ++i){
        axpy_cpu(l.outputs, 1, l.delta + i*l.outputs, 1, l.bias_updates, 1);
    }

    for(i = 0; i < l.batch; ++i){
        float *input = net.input + i*l.w*l.h*l.c;
        im2col_cpu(input, l.c, l.h, l.w,
                l.size, l.stride, l.pad, net.workspace);

        for(j = 0; j < locations; ++j){
            float *a = l.delta + i*l.outputs + j;
            float *b = net.workspace + j;
            float *c = l.weight_updates + j*l.size*l.size*l.c*l.n;
            int m = l.n;
            int n = l.size*l.size*l.c;
            int k = 1;

            gemm(0,1,m,n,k,1,a,locations,b,locations,1,c,n);
        }

        if(net.delta){
            for(j = 0; j < locations; ++j){
                float *a = l.weights + j*l.size*l.size*l.c*l.n;
                float *b = l.delta + i*l.outputs + j;
                float *c = net.workspace + j;

                int m = l.size*l.size*l.c;
                int n = 1;
                int k = l.n;

                gemm(1,0,m,n,k,1,a,m,b,locations,0,c,locations);
            }

            col2im_cpu(net.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, net.delta+i*l.c*l.h*l.w);
        }
    }
}
```

함수 이름: backward\_local\_layer

입력:&#x20;

* local\_layer 구조체 l
* network 구조체 net

동작:&#x20;

* local\_layer를 역전파하는 함수입니다.&#x20;
* 출력값에 대한 델타를 계산하고, 바이어스 업데이트 및 가중치 업데이트를 수행합니다.&#x20;
* 이후 입력값에 대한 델타를 계산합니다.

설명:

* l.delta: 출력값의 델타를 저장하는 배열
* l.bias\_updates: 바이어스 업데이트를 저장하는 배열
* l.weight\_updates: 가중치 업데이트를 저장하는 배열
* net.workspace: im2col 연산의 결과를 저장하는 배열
* net.delta: 이전 레이어의 델타를 저장하는 배열

1. 출력값에 대한 델타를 계산합니다.
2. 모든 배치에 대해 바이어스 업데이트를 수행합니다.
3. 모든 배치에 대해 im2col 연산을 수행합니다.
4. 모든 배치 및 위치에 대해 가중치 업데이트를 수행합니다.
5. 이전 레이어의 델타를 계산하고 net.delta 배열에 저장합니다.



## update\_local\_layer

```c
void update_local_layer(local_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int locations = l.out_w*l.out_h;
    int size = l.size*l.size*l.c*l.n*locations;
    axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}
```

함수 이름: update\_local\_layer

입력:

* local\_layer l: 로컬 레이어 객체
* update\_args a: 업데이트 인자 객체

동작:&#x20;

* 로컬 레이어의 가중치와 편향을 업데이트하는 함수입니다.&#x20;
* 업데이트는 경사 하강법을 사용하여 수행됩니다.&#x20;
* 편향은 배치 크기로 나눈 학습률과 모멘텀을 사용하여 업데이트하고, 가중치는 학습률과 가중치 감쇠, 모멘텀을 사용하여 업데이트합니다.

설명:

* local\_layer: 로컬 레이어 객체로, 로컬 레이어의 출력, 가중치, 편향 등의 정보를 저장합니다.
* update\_args: 업데이트 인자 객체로, 학습률, 모멘텀, 가중치 감쇠, 배치 크기 등의 업데이트에 필요한 정보를 저장합니다.
* axpy\_cpu(): 벡터 덧셈과 스칼라 곱을 수행하는 함수입니다.
* scal\_cpu(): 벡터를 스칼라로 곱하는 함수입니다.



## make\_local\_layer

```c
local_layer make_local_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation)
{
    int i;
    local_layer l = {0};
    l.type = LOCAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = pad;

    int out_h = local_out_height(l);
    int out_w = local_out_width(l);
    int locations = out_h*out_w;
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.weights = calloc(c*n*size*size*locations, sizeof(float));
    l.weight_updates = calloc(c*n*size*size*locations, sizeof(float));

    l.biases = calloc(l.outputs, sizeof(float));
    l.bias_updates = calloc(l.outputs, sizeof(float));

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c));
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1,1);

    l.output = calloc(l.batch*out_h * out_w * n, sizeof(float));
    l.delta  = calloc(l.batch*out_h * out_w * n, sizeof(float));

    l.workspace_size = out_h*out_w*size*size*c;

    l.forward = forward_local_layer;
    l.backward = backward_local_layer;
    l.update = update_local_layer;

    l.activation = activation;

    fprintf(stderr, "Local Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);

    return l;
}
```

함수 이름: make\_local\_layer&#x20;

입력:

* int batch: 배치 크기
* int h: 입력 이미지 높이
* int w: 입력 이미지 너비
* int c: 입력 이미지 채널 수
* int n: 필터 수
* int size: 필터 크기
* int stride: 스트라이드
* int pad: 패딩
* ACTIVATION activation: 활성화 함수

동작:&#x20;

* 로컬 레이어를 생성하고 초기화한 후 반환한다.

설명:

* 로컬 레이어를 초기화하기 위해 필요한 파라미터를 입력으로 받는다.
* 로컬 레이어의 출력 크기와 필요한 메모리를 계산한다.
* 로컬 레이어의 가중치, 편향, 출력, 델타, 가중치 업데이트, 편향 업데이트 등을 저장할 메모리를 할당한다.
* 가중치는 sqrt(2./(size_size_c))로 스케일링된 값으로 초기화하며, 편향은 0으로 초기화한다.
* 로컬 레이어의 forward, backward, update 함수를 설정한다.
* 초기화된 로컬 레이어를 반환한다.

