# deconvolutional\_layer

## Deconvolutional Layer 란?

deconvolution은 convolution을 반대로 연산합니다.

기존의 convolution은 입력 특징맵과 필터를 컨볼루션 연산하여 출력 특징맵을 생성합니다. 이로인해 특징맵의 크기는 줄어듭니다. deconvolution은 이와 반대로 특징맵의 크기를 증가시킵니다.

간혹 deconvolution과 transpose convolution이 헷갈리는데

* 만약 5 x 5 이미지가 stride가 2, kernel size가 3인 컨볼루션 연산을 한다면 2 x 2 출력이 생성됩니다.
* 이 과정을 반대로 하면 하나의 픽셀에서 9개의 값이 생성되도록 수학적 연산을 거꾸로 하면 deconvolution 입니다.
* padding을 추가합니다.
* 거꾸로 연산을 해서 되돌리지는 않습니다.

추가적으로 dilated convolution은 연산량을 늘리지 않고 receptive field를 크게 만드는 효과적인 방법입니다.

## get\_workspace\_size

```c
static size_t get_workspace_size(layer l){
    return (size_t)l.h*l.w*l.size*l.size*l.n*sizeof(float);
}
```

"get\_workspace\_size" 함수는 다음과 같은 입력을 받습니다:

* layer l: 계산할 레이어의 정보를 담고 있는 layer 구조체

이 함수는 입력으로 받은 레이어에 필요한 workspace의 크기를 계산하여 반환합니다. 계산 방식은 다음과 같습니다:

* 레이어의 높이(l.h), 너비(l.w), 필터 크기(l.size), 필터 개수(l.n)를 이용하여 workspace 크기를 계산합니다.
* 계산된 크기는 float 자료형의 크기와 곱하여 반환합니다.



## bilinear\_init

```c
void bilinear_init(layer l)
{
    int i,j,f;
    float center = (l.size-1) / 2.;
    for(f = 0; f < l.n; ++f){
        for(j = 0; j < l.size; ++j){
            for(i = 0; i < l.size; ++i){
                float val = (1 - fabs(i - center)) * (1 - fabs(j - center));
                int c = f%l.c;
                int ind = f*l.size*l.size*l.c + c*l.size*l.size + j*l.size + i;
                l.weights[ind] = val;
            }
        }
    }
}
```

함수 이름: bilinear\_init

입력:&#x20;

* l: layer 타입

동작:&#x20;

* Bilinear interpolation을 위한 layer의 weights를 초기화한다.&#x20;
* 각각의 filter에 대해 2D bilinear interpolation kernel을 만들어 weights 값을 할당한다.

설명:&#x20;

* 입력으로 주어진 layer의 weights를 초기화한다.&#x20;
* 이 때, 각 filter에 대해 2D bilinear interpolation kernel을 만들어서 weights 값을 할당한다.&#x20;
* 이 kernel은 center를 기준으로 x, y 축 방향으로의 거리를 계산하여 weight 값을 할당하게 된다.&#x20;
* 따라서, 해당 함수에서는 filter의 개수, 크기, 그리고 center를 이용하여 2D bilinear interpolation kernel을 생성하고, 이를 이용하여 weights 값을 할당한다.



## forward\_deconvolutional\_layer

```c
void forward_deconvolutional_layer(const layer l, network net)
{
    int i;

    int m = l.size*l.size*l.n;
    int n = l.h*l.w;
    int k = l.c;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    for(i = 0; i < l.batch; ++i){
        float *a = l.weights;
        float *b = net.input + i*l.c*l.h*l.w;
        float *c = net.workspace;

        gemm_cpu(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_cpu(net.workspace, l.out_c, l.out_h, l.out_w, l.size, l.stride, l.pad, l.output+i*l.outputs);
    }
    if (l.batch_normalize) {
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_w*l.out_h);
    }
    activate_array(l.output, l.batch*l.n*l.out_w*l.out_h, l.activation);
}
```

함수 이름: forward\_deconvolutional\_layer&#x20;

입력:

* const layer l: 디컨볼루션 레이어 정보를 담은 구조체
* network net: 네트워크 정보를 담은 구조체

동작:

* 디컨볼루션 레이어를 통해 입력 데이터를 역전파하여 출력값을 계산함
* 먼저, 입력값과 가중치를 행렬 곱셈하여 출력값을 계산함
* 그 후, col2im 함수를 사용하여 출력값을 4D 텐서로 reshape 함
* batch normalization이 설정되어 있으면 forward\_batchnorm\_layer 함수를 호출하여 출력값을 정규화 함
* 그렇지 않으면, 출력값에 bias를 더하여 출력값을 계산함
* 마지막으로, 활성화 함수를 적용하여 최종 출력값을 계산함

설명:

* 디컨볼루션 레이어는 합성곱 연산과 반대로 입력 데이터를 역전파하여 출력값을 계산하는 레이어이다.
* 입력값과 가중치를 행렬 곱셈하여 출력값을 계산하는 과정에서는, col2im 함수를 사용하여 출력값을 4D 텐서로 reshape 해준다.
* batch normalization이 설정되어 있으면, forward\_batchnorm\_layer 함수를 호출하여 출력값을 정규화한다.
* 그렇지 않으면, bias를 더하여 출력값을 계산하고, 마지막으로 활성화 함수를 적용하여 최종 출력값을 계산한다.



## backward\_deconvolutional\_layer

```c
void backward_deconvolutional_layer(layer l, network net)
{
    int i;

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, l.out_w*l.out_h);
    }

    //if(net.delta) memset(net.delta, 0, l.batch*l.h*l.w*l.c*sizeof(float));

    for(i = 0; i < l.batch; ++i){
        int m = l.c;
        int n = l.size*l.size*l.n;
        int k = l.h*l.w;

        float *a = net.input + i*m*k;
        float *b = net.workspace;
        float *c = l.weight_updates;

        im2col_cpu(l.delta + i*l.outputs, l.out_c, l.out_h, l.out_w,
                l.size, l.stride, l.pad, b);
        gemm_cpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if(net.delta){
            int m = l.c;
            int n = l.h*l.w;
            int k = l.size*l.size*l.n;

            float *a = l.weights;
            float *b = net.workspace;
            float *c = net.delta + i*n*m;

            gemm_cpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
}
```

함수 이름: backward\_deconvolutional\_layer

입력:

* layer l: 디컨볼루션 레이어 구조체
* network net: 네트워크 구조체

동작:

* 디컨볼루션 레이어의 역전파(backpropagation) 수행
* gradient\_array 함수를 사용하여 델타(delta) 값을 구하고, 배치 정규화(batch normalization)를 사용하는 경우 backward\_batchnorm\_layer 함수를 호출하고, 그렇지 않은 경우 backward\_bias 함수를 사용하여 편향(bias) 값을 업데이트함
* 네트워크의 델타(delta) 값을 0으로 초기화하고, im2col\_cpu 함수를 사용하여 델타(delta) 값을 4차원 텐서에서 2차원 행렬로 변환함
* gemm\_cpu 함수를 사용하여 입력값과 변환된 델타(delta) 값을 행렬 곱셈한 결과를 가중치(weight) 업데이트에 사용될 행렬로 변환함
* 네트워크의 델타(delta) 값을 gemm\_cpu 함수를 사용하여 업데이트함

설명:

* 디컨볼루션 레이어의 역전파는 레이어의 출력값과 델타 값을 사용하여 가중치(weight)와 편향(bias)을 업데이트하는 과정을 말합니다.
* 이 함수에서는 레이어의 출력값에서 gradient\_array 함수를 사용하여 델타 값을 구하고, 배치 정규화를 사용하는 경우 backward\_batchnorm\_layer 함수를 호출하여 배치 정규화 계층의 역전파를 수행합니다. 배치 정규화를 사용하지 않는 경우에는 backward\_bias 함수를 사용하여 편향 값을 업데이트합니다.
* 이후, 델타 값을 4차원 텐서에서 2차원 행렬로 변환한 뒤, gemm\_cpu 함수를 사용하여 입력값과 곱셈을 수행한 결과를 가중치(weight) 업데이트에 사용될 행렬로 변환합니다.
* 마지막으로, 네트워크의 델타(delta) 값을 gemm\_cpu 함수를 사용하여 업데이트합니다.



## update\_deconvolutional\_layer

```c
void update_deconvolutional_layer(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int size = l.size*l.size*l.c*l.n;
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}
```

함수 이름: update\_deconvolutional\_layer

입력:

* layer l: 업데이트할 디컨벌루션 레이어
* update\_args a: 업데이트를 위한 인자들(learning\_rate, momentum, decay, batch 등)

동작:

* l.bias\_updates와 l.scale\_updates를 이용하여 l.biases와 l.scales를 업데이트함
* l.weight\_updates를 이용하여 l.weights를 업데이트함
* 업데이트에 사용되는 인자들(learning\_rate, momentum, decay, batch)을 이용하여 업데이트 과정을 조절함

설명:

* 디컨벌루션 레이어의 가중치, 편향, 스케일 등을 업데이트하는 함수임
* 업데이트에 필요한 인자들(learning\_rate, momentum, decay, batch)을 입력받아 사용함
* 편향 업데이트: l.bias\_updates와 l.biases를 이용하여 업데이트함. learning\_rate와 batch를 이용하여 조절하고, momentum을 이용하여 이전 업데이트 값과의 비율을 결정함
* 스케일 업데이트: l.scales가 존재하면 l.scale\_updates와 l.scales를 이용하여 업데이트함. 편향 업데이트와 동일한 방식으로 진행함
* 가중치 업데이트: l.weight\_updates와 l.weights를 이용하여 업데이트함. decay와 batch를 이용하여 가중치 감소와 스케일 조정을 함. learning\_rate를 이용하여 업데이트 비율을 결정하고, momentum을 이용하여 이전 업데이트 값과의 비율을 결정함



## resize\_deconvolutional\_layer

```c
void resize_deconvolutional_layer(layer *l, int h, int w)
{
    l->h = h;
    l->w = w;
    l->out_h = (l->h - 1) * l->stride + l->size - 2*l->pad;
    l->out_w = (l->w - 1) * l->stride + l->size - 2*l->pad;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

    l->workspace_size = get_workspace_size(*l);
}
```

함수 이름: resize\_deconvolutional\_layer

입력:

* layer \*l: 크기가 조정될 deconvolutional layer의 포인터
* int h: 조정된 높이
* int w: 조정된 너비

동작:&#x20;

* deconvolutional layer의 높이와 너비를 조정하고, 그에 따라 출력, 델타, x, x\_norm 및 workspace의 크기도 조정합니다.

설명:&#x20;

* 이 함수는 입력으로 받은 deconvolutional layer의 높이와 너비를 조정하고, 출력, 델타, x, x\_norm 및 workspace의 크기도 조정합니다.&#x20;
* 높이와 너비가 조정되면, 출력의 높이와 너비, 입력 및 출력의 전체 크기가 바뀝니다.&#x20;
* 이에 따라 메모리를 다시 할당해야 합니다. 함수는 realloc()을 사용하여 각각의 메모리 영역에 대해 새로운 크기로 메모리를 다시 할당합니다.&#x20;
* 마지막으로, 새로운 workspace의 크기를 계산합니다.



## make\_deconvolutional\_layer

```c
layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer l = {0};
    l.type = DECONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.batch = batch;
    l.stride = stride;
    l.size = size;

    l.nweights = c*n*size*size;
    l.nbiases = n;

    l.weights = calloc(c*n*size*size, sizeof(float));
    l.weight_updates = calloc(c*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));
    //float scale = n/(size*size*c);
    //printf("scale: %f\n", scale);
    float scale = .02;
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_normal();
    //bilinear_init(l);
    for(i = 0; i < n; ++i){
        l.biases[i] = 0;
    }
    l.pad = padding;

    l.out_h = (l.h - 1) * l.stride + l.size - 2*l.pad;
    l.out_w = (l.w - 1) * l.stride + l.size - 2*l.pad;
    l.out_c = n;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.w * l.h * l.c;

    scal_cpu(l.nweights, (float)l.out_w*l.out_h/(l.w*l.h), l.weights, 1);

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_deconvolutional_layer;
    l.backward = backward_deconvolutional_layer;
    l.update = update_deconvolutional_layer;

    l.batch_normalize = batch_normalize;

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.m = calloc(c*n*size*size, sizeof(float));
        l.v = calloc(c*n*size*size, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }

    l.activation = activation;
    l.workspace_size = get_workspace_size(l);

    fprintf(stderr, "deconv%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

    return l;
}
```

함수 이름: make\_deconvolutional\_layer&#x20;

입력:

* batch: 레이어를 적용할 이미지의 개수
* h: 입력 이미지의 높이
* w: 입력 이미지의 너비
* c: 입력 이미지의 채널 수
* n: 필터 개수
* size: 필터의 높이와 너비
* stride: 필터를 적용하는 간격
* padding: 입력 이미지 주위에 추가되는 패딩의 크기
* activation: 활성화 함수의 종류
* batch\_normalize: 배치 정규화를 사용하는지 여부
* adam: Adam 알고리즘을 사용하는지 여부

동작:&#x20;

* 입력으로 받은 정보를 사용하여 deconvolutional 레이어를 생성하고 초기화한 후, 해당 레이어를 반환한다.

설명:&#x20;

* make\_deconvolutional\_layer 함수는 입력 이미지에 대해 deconvolutional 연산을 수행하는 레이어를 생성한다.&#x20;
* 입력으로 받은 batch, h, w, c, n, size, stride, padding, activation, batch\_normalize, adam 등의 정보를 사용하여 레이어를 초기화하고 필요한 메모리 공간을 할당한다.&#x20;
* 이후 레이어를 반환하며, 반환된 레이어는 convolutional 레이어와 마찬가지로 forward, backward, update 함수를 가지고 있다.&#x20;
* 출력 이미지의 크기는 입력으로 받은 정보와 forward 함수에서 계산되어 결정된다.&#x20;
* 또한, batch\_normalize와 adam 알고리즘을 사용하는 경우에 필요한 변수들도 초기화한다.&#x20;
* 함수가 실행되면, 생성된 레이어의 정보를 출력한다.



## denormalize\_deconvolutional\_layer

```c
void denormalize_deconvolutional_layer(layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c*l.size*l.size; ++j){
            l.weights[i*l.c*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}
```

함수 이름: denormalize\_deconvolutional\_layer

입력:&#x20;

* layer l (deconvolutional layer)

동작:&#x20;

* 배치 정규화를 수행한 후에 네트워크의 출력 값을 다시 원래의 분포로 돌리기 위해 사용되는 함수입니다.&#x20;
* 이 함수는 테스트 시에 사용됩니다. 배치 정규화를 통해 스케일링 및 이동한 가중치 및 편향 값을 원래의 값으로 되돌립니다.&#x20;
* 이 함수는 반드시 forward 함수 호출 이후에 호출되어야 합니다.

설명:

* i, j: 반복문을 위한 인덱스 변수
* scale: 배치 정규화에서 사용한 스케일링 및 이동 값을 되돌리기 위한 스케일 값
* l.scales\[i]/sqrt(l.rolling\_variance\[i] + .00001): 스케일 값 계산
* l.weights\[i_l.c_l.size\*l.size + j] \*= scale: 가중치 값에 스케일 값을 곱하여 되돌림
* l.biases\[i] -= l.rolling\_mean\[i] \* scale: 편향 값에 스케일 값을 곱한 롤링 평균 값을 빼서 되돌림
* l.scales\[i] = 1: 스케일 값 초기화
* l.rolling\_mean\[i] = 0: 롤링 평균 값 초기화
* l.rolling\_variance\[i] = 1: 롤링 분산 값 초기화

