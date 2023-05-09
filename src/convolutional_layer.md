# convolutional\_layer

## Convolutional Layer 란?

Convolution은 합성곱으로 2가지 연산을 사용합니다.

* 각 원소끼리 곱합니다. (element wise multiplication)
* 각 원소를 더합니다.

이해를 돕기위해 그림으로 살펴보겠습니다. 아래 그림과 같이 각 원소를 곱하고 합한 값을 활성화 함수를 통과하여 최종적으로 값을 만듭니다.(아래 그림은 활성화 함수를 생략한 그림입니다.)

Convolutional Layer는 Feature Maps과 Filters의 Convolution 연산을 통해 그 다음 Feature Maps을 만들어 내는 작업을 반복합니다. 여기서 filters가 학습 파라미터 입니다.

`입력 이미지` -> `Filters(kernel)` -> `Feature Maps(Channels)` -> `Filters(kernel)` -> `Feature Maps(Channels)` -> `...`

Convolutional Layer는 설정 가능한 파라미터가 있습니다.

* `stride` : filter가 움직이는 간격입니다.
* `padding` : Feature Map의 테두리 부분의 정보 손실을 줄이기 위해서 테두리를 특정한 값(보통 0)으로 채워 넣는 방법입니다. padding은 몇개의 테두리를 채울지에 대한 값입니다.
* filter의 수(가중치의 수) : $$k \times k \times C_1 \times C_2$$
* $$W_2 = \frac{W_1 - k + 2 \times padding}{stride_w} + 1$$
* $$H_2 = \frac{H_1 - k + 2 \times padding}{stride_h} + 1$$

Convolutional Layer 역전파는 쉽게 표현하는 경우 아래 그림과 같습니다.

output을 계산하기 위해서 각자의 id를 가지고 있는 weight가 사용된 곳을 보시면 이해하기 쉽습니다. 예를 들어서 $$w_11$$은 $$h_11, h_12, h_21, h_22$$를 연산하는데 각각 사용되었기 때문에 이들의 미분 값의 합으로 최종적으로 업데이트 할 기울기를 만듭니다. 역전파는 Layer 따로 따로 간단하게 어떻게 동작하는지를 전부 살펴보고 마지막에 보면 더 쉬운거 같습니다.

***

### forward\_convolutional\_layer

```c
void forward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);                                    /// output을 0으로 초기화

    if(l.xnor){                                                                     
        binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        net.input = l.binary_input;
    }

    int m = l.n/l.groups;                                                           /// filter 개수
    int k = l.size*l.size*l.c/l.groups;                                             /// filter 크기
    int n = l.out_w*l.out_h;                                                        /// output feature map 크기
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights + j*l.nweights/l.groups;                           /// 학습 시작 포인터
            float *b = net.workspace;                                               
            float *c = l.output + (i*l.groups + j)*n*m;                             /// output 시작 포인터
            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;         /// input 시작 포인터

            if (l.size == 1) {                                                      
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b); /// 이미지를 columns로 변환
            }
            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);                                        /// 컨볼루션 연산
        }
    }

    if(l.batch_normalize){                                                          
        forward_batchnorm_layer(l, net);                                            
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);                
    }

    activate_array(l.output, l.outputs*l.batch, l.activation);                      
    if(l.binary || l.xnor) swap_binary(&l);                                         
}
```

함수 이름: forward\_convolutional\_layer

입력:

* convolutional\_layer l: 컨볼루션 레이어 구조체
* network net: 네트워크 구조체

동작:

* 컨볼루션 연산을 수행하여 l.output에 결과값을 저장
* 배치 정규화를 사용하는 경우, forward\_batchnorm\_layer 함수를 호출하여 배치 정규화를 수행
* 활성화 함수를 수행하여 l.output을 업데이트
* 이진 컨볼루션 또는 XNOR-Networks를 사용하는 경우, swap\_binary 함수를 호출하여 가중치와 입력값을 이진화

설명:

* 컨볼루션 레이어에 대한 forward 연산을 수행하는 함수이다.
* l.weights, l.biases, l.activation 등 컨볼루션 레이어의 필수 구성 요소들을 사용하여 입력값을 컨볼루션 연산하여 출력값을 계산한다.
* 컨볼루션 연산을 수행하기 위해 입력값을 이미지를 columns로 변환한다.
* l.batch\_normalize가 true인 경우, forward\_batchnorm\_layer 함수를 호출하여 배치 정규화를 수행한다.
* activate\_array 함수를 사용하여 활성화 함수를 수행하여 l.output을 업데이트한다.
* l.binary 또는 l.xnor가 true인 경우, swap\_binary 함수를 호출하여 가중치와 입력값을 이진화한다.



### backward\_convolutional\_layer

```c
void backward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;
    int m = l.n/l.groups;                                                           /// filter 개수
    int k = l.size*l.size*l.c/l.groups;                                             /// filter 크기
    int n = l.out_w*l.out_h;                                                        /// output feature map 크기

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);             /// activation function 역전파

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);                                           /// batch normalize 역전파
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);                    /// bias 역전파
    }

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta + (i*l.groups + j)*m*k;                              /// gradient 포인터 이동
            float *b = net.workspace;                                               
            float *c = l.weight_updates + j*l.nweights/l.groups;                    /// update 포인터 이동

            float *im  = net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;         /// 이미지 포인터
            float *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;         

            if(l.size == 1){                                                       
                b = im;                                                             
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w,
                        l.size, l.stride, l.pad, b);                                /// 이미지를 columns로 변환
            }

            gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);                                        /// b(image)를 전치행렬로 컨볼루션 연산

            if (net.delta) {
                a = l.weights + j*l.nweights/l.groups;                              /// weight 포인터 이동          
                b = l.delta + (i*l.groups + j)*m*k;                                 /// gradient 포인터 이동
                c = net.workspace;                                                  
                if (l.size == 1) {
                    c = imd;
                }

                gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);                                    /// a(weight)를 전치행렬로 컨볼루션 연산

                if (l.size != 1) {
                    col2im_cpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);    /// columns을 이미지로 변환
                }
            }
        }
    }
}
```

함수 이름: backward\_convolutional\_layer

입력:&#x20;

* convolutional\_layer l: 컨볼루션 레이어 구조체
* network net: 네트워크 구조체

동작:

1. activation function의 gradient를 계산한다.
2. batch normalization이 적용되었다면 batch normalization의 gradient를 계산하고, 그렇지 않으면 bias의 gradient를 계산한다.
3. 각 배치에 대해, 각 그룹에서 gradient와 weight를 곱해 weight update를 계산한다.
4. 이미지를 columns로 변환하여, 각 그룹에서 weight와 gradient를 곱해 input delta를 계산한다.
5. columns을 이미지로 변환하여 input delta를 저장한다.

설명:&#x20;

* convolutional layer에서는 이미지와 weight를 컨볼루션 연산하여 output feature map을 계산하고, 이후 activation function을 적용한다.&#x20;
* 역전파에서는 output feature map의 gradient를 계산하고, 이를 이용하여 input delta와 weight update를 계산한다.&#x20;
* 이때, gradient와 weight를 곱해 weight update를 계산할 때는 해당 그룹의 weight를 모두 사용하며, 각 그룹마다 input delta를 계산하여 누적하여 저장한다.&#x20;
* 이후, columns를 이미지로 변환하여 input delta를 계산한다.&#x20;
* 만약 batch normalization이 적용된 convolutional layer라면, 이전 layer에서 전달받은 error를 이용하여 batch normalization의 gradient를 계산하고, 이를 이용하여 gamma와 beta를 업데이트한다.



### update\_convolutional\_layer

```c
void update_convolutional_layer(convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}
```

함수 이름: update\_convolutional\_layer

입력:

* convolutional\_layer l: 업데이트할 convolutional layer 구조체
* update\_args a: 업데이트에 필요한 인자들을 담은 구조체. learning\_rate, momentum, decay, batch 값을 가짐

동작:

* Convolutional layer의 bias와 weight를 업데이트함
* learning\_rate, momentum, decay, batch 값을 사용하여 업데이트에 필요한 계산 수행

설명:

* axpy\_cpu 함수를 사용하여 bias와 scale을 업데이트 함. axpy\_cpu 함수는 y = a\*x + y 연산을 수행함
* scal\_cpu 함수를 사용하여 momentum을 적용함. scal\_cpu 함수는 벡터의 모든 원소에 스칼라 값을 곱해줌
* weight의 경우 decay를 적용하고, axpy\_cpu 함수를 사용하여 weight를 업데이트 함. 이때 batch 값을 사용하여 mini-batch gradient descent를 수행함



### resize\_convolutional\_layer

```c
void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

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

함수 이름: resize\_convolutional\_layer

입력:

* convolutional\_layer \*l: 크기를 조절할 컨볼루션 레이어
* int w: 조절할 가로 크기
* int h: 조절할 세로 크기

동작:&#x20;

* 주어진 가로 크기와 세로 크기에 따라 컨볼루션 레이어의 크기를 조정합니다.&#x20;
* 이때, 출력 크기(out\_w, out\_h)도 계산하고, 컨볼루션 레이어의 출력, 델타, x, x\_norm, workspace의 크기를 새로운 크기에 맞게 재할당합니다.

설명:&#x20;

* 컨볼루션 레이어의 크기를 조절하는 함수입니다.&#x20;
* 입력으로 주어진 컨볼루션 레이어의 가로와 세로 크기를 주어진 w와 h 값으로 각각 바꾸어주며, 출력 크기(out\_w, out\_h)도 이에 맞게 다시 계산합니다.&#x20;
* 그리고 출력, 델타, x, x\_norm, workspace의 크기를 새로운 크기에 맞게 realloc 함수를 사용하여 재할당합니다.&#x20;
* 이때, batch\_normalize가 사용되는 경우에는 x와 x\_norm도 재할당합니다.&#x20;
* 마지막으로 workspace\_size를 get\_workspace\_size 함수를 통해 다시 계산하여 저장합니다.



### make\_convolutional\_layer

```c
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    int i;
    convolutional_layer l = {0};
    l.type = CONVOLUTIONAL;

    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = calloc(c/groups*n*size*size, sizeof(float));
    l.weight_updates = calloc(c/groups*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c/l.groups));
    //printf("convscale %f\n", scale);
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();
    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if(binary){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.cweights = calloc(l.nweights, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }

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
        l.m = calloc(l.nweights, sizeof(float));
        l.v = calloc(l.nweights, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }

    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);

    return l;
}
```

함수 이름: make\_convolutional\_layer

입력:

* batch: 배치 크기
* h: 입력 이미지의 높이
* w: 입력 이미지의 너비
* c: 입력 이미지의 채널 수
* n: 필터 개수
* groups: 그룹 수
* size: 필터 크기
* stride: 스트라이드
* padding: 패딩 크기
* activation: 활성화 함수
* batch\_normalize: 배치 정규화 여부
* binary: 이진화 여부
* xnor: XNOR 여부
* adam: Adam 옵티마이저 사용 여부

동작:

* 입력 이미지와 필터를 합성곱 연산하여 출력을 계산하는 합성곱 레이어를 생성합니다.
* 필요한 메모리를 동적 할당합니다.
* 가중치(weight)와 편향(bias)을 초기화합니다.
* 활성화 함수, 배치 정규화, 이진화, XNOR, Adam 옵티마이저를 사용하는 경우 필요한 메모리와 변수를 할당하고 초기화합니다.
* 생성된 레이어의 출력 크기와 필요한 메모리 크기를 계산합니다.
* 생성된 레이어와 연관된 함수 포인터를 설정합니다.
* 생성된 레이어를 반환합니다.

설명:

* 이 함수는 입력 이미지와 필터의 합성곱 연산을 수행하는 합성곱 레이어를 생성하는 함수입니다.
* 입력으로 받은 파라미터를 사용하여 필요한 메모리를 동적 할당하고 초기화합니다.
* 필터(weight)는 랜덤한 값으로 초기화하며, Xavier 초기화 방법을 사용합니다.
* 활성화 함수는 ReLU, LeakyReLU, linear 함수를 사용할 수 있습니다.
* 배치 정규화는 입력 데이터의 배치 단위로 정규화를 수행하여 학습을 안정화시키는 방법입니다.
* 이진화는 모델의 가중치를 이진 형태로 변환하여 모델의 크기를 줄이고 연산 속도를 높이는 방법입니다.
* XNOR는 이진화된 가중치와 이진화된 입력을 사용하여 합성곱 연산을 수행하는 방법으로, 이진화보다 더 큰 모델 압축과 빠른 연산 속도를 제공합니다.
* Adam 옵티마이저는 경사 하강법을 사용하는 옵티마이저 중 하나로, 모멘텀과 RMSProp을 결합한 방법입니다. 학습 속도를 자동으로 조절하여 더 빠르게 수렴하는 특징이 있습니다.



### denormalize\_convolutional\_layer

```c
void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c/l.groups*l.size*l.size; ++j){
            l.weights[i*l.c/l.groups*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}
```

함수 이름: denormalize\_convolutional\_layer

입력:

* convolutional\_layer l: denormalization이 필요한 convolutional layer

동작:

* batch normalization을 적용한 convolutional layer를 denormalize 함
* convolutional layer의 weights, biases, scales, rolling\_mean, rolling\_variance 값을 수정함

설명:

* batch normalization은 데이터의 분포를 조절해 학습을 안정화시키는 기술이다.
* 학습 과정에서 이전 미니배치의 평균과 분산을 이용해 현재 미니배치의 데이터를 normalize한다.
* 하지만, 학습이 끝난 모델을 사용할 때는 이전 미니배치의 평균과 분산 대신 전체 데이터셋의 평균과 분산을 이용해 denormalize해야 한다.
* 이 함수는 그 역할을 수행하는 함수이다.
* 각 채널마다 denormalize에 필요한 값을 계산하고, weights와 biases 값을 수정한다.
* scales 값은 1로, rolling\_mean과 rolling\_variance 값은 0과 1로 초기화한다.



### add\_bias

```c
void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}
```

함수 이름: add\_bias

입력:

* output: float 형식의 출력 값 포인터
* biases: float 형식의 bias 값 포인터
* batch: int 형식의 batch 크기
* n: int 형식의 필터 개수
* size: int 형식의 출력 값 크기

동작:

* 출력 값에 bias 값을 더함
* 출력 값의 크기는 batch x n x size

설명:

* 각 필터마다 bias 값을 더해주는 함수
* 출력 값의 각 요소에 biases\[i] 값을 더해줌으로써 bias 값을 적용함
* 출력 값의 크기는 batch x n x size 이므로, 반복문을 통해 각 요소에 biases\[i] 값을 더해줌



### scale\_bias

```c
void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}
```

함수 이름: scale\_bias

입력:

* float \*output: 출력값 포인터
* float \*scales: 스케일값 포인터
* int batch: 배치 크기
* int n: 출력 채널 수
* int size: 출력값 크기 (가로, 세로)

동작:&#x20;

* 각 배치별로 출력값(output)의 각 채널에 대해, scales 배열의 해당 채널 값으로 출력값을 스케일링(scale)합니다.

설명:&#x20;

* 이 함수는 출력값(output)에 대해 각 채널에 대한 스케일링 작업을 수행하는 함수입니다.&#x20;
* 배치(batch) 크기만큼의 데이터를 처리하며, 각 배치에 대해 출력값(output)의 n개 채널에 대해 스케일링 작업을 수행합니다.&#x20;
* 출력값(output)은 배치, 채널, 가로, 세로의 4차원 배열 구조로 이루어져 있으며, 스케일(scales)도 출력 채널 수(n)만큼의 1차원 배열로 주어집니다.&#x20;
* 이 함수는 각 배치(b), 채널(i), 가로(j), 세로(k)에 대한 반복문을 수행하며, 출력값(output)의 (b\*n+i)\*size+j 위치에 해당하는 값을 스케일(scales) 배열의 i번째 값으로 곱해주는 작업을 수행합니다.&#x20;
* 이렇게 스케일링된 값을 출력값(output)에 저장합니다.



### backward\_bias

```c
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}
```

함수 이름: backward\_bias

입력:

* bias\_updates: 각 레이어의 편향(bias) 업데이트 값을 저장할 배열
* delta: 각 뉴런의 오차 값(error) 배열
* batch: 미니배치(batch) 크기
* n: 레이어 내 뉴런 개수
* size: 뉴런이 가지고 있는 입력(input) 데이터 크기

동작:&#x20;

* 이 함수는 convolutional neural network에서 편향(bias) 업데이트를 계산합니다.&#x20;
* delta는 현재 레이어의 뉴런에서의 오차 값입니다.&#x20;
* bias\_updates 배열은 각 레이어의 편향 값을 업데이트할 때 사용됩니다.

설명:&#x20;

* 이 함수는 미니배치(batch) 내의 모든 뉴런에 대해 bias\_updates 배열에 대한 업데이트 값을 계산합니다.&#x20;
* 먼저 for문을 이용하여 batch 내 각 뉴런에 대한 bias\_updates 값을 계산합니다.&#x20;
* 그리고 sum\_array 함수를 사용하여 delta 배열의 해당 뉴런의 오차 값을 계산하고, bias\_updates 배열에 더합니다.&#x20;
* 따라서 이 함수는 backward propagation 과정에서 편향 업데이트를 수행합니다.



### swap\_binary

```c
void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;
}
```

함수 이름: swap\_binary

입력:&#x20;

* convolutional\_layer \*l: convolutional\_layer 구조체 포인터

동작:&#x20;

* 이 함수는 convolutional layer의 가중치(weights)와 binary weights를 교환(swap)합니다.

설명:&#x20;

* 이 함수는 convolutional layer의 가중치를 binary weights로 교환합니다.&#x20;
* 이는 forward pass에서 기존의 가중치를 사용하여 연산을 수행하는 대신, 이진 형태의 가중치(binary weights)를 사용하여 더욱 빠르게 연산을 수행할 수 있도록 하기 위한 것입니다.&#x20;
* 이 함수에서는 포인터를 사용하여 가중치와 binary weights를 교환합니다.



### binarize\_weights

```c
void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}
```

함수 이름: binarize\_weights

입력:

* float \*weights: 이진화할 가중치 배열
* int n: 가중치의 채널 수
* int size: 가중치의 크기
* float \*binary: 이진화된 가중치를 저장할 배열

동작:&#x20;

* 주어진 가중치 배열(weights)을 이진화하여 이진화된 가중치(binary)를 계산합니다.&#x20;
* 이진화된 가중치는 각 가중치의 절댓값 평균(mean)으로 계산됩니다.&#x20;
* 가중치 값이 평균보다 크면 이진화된 가중치 값은 평균이고, 작으면 -평균입니다.

설명:&#x20;

* Convolutional neural network에서 이진화된 가중치는 더 적은 메모리 공간을 차지하고 빠른 연산이 가능하여 모델의 실행 속도를 향상시키는 데 도움을 줍니다.&#x20;
* 이진화된 가중치를 사용하면 계산 복잡도를 줄이는 동시에 정확도를 유지할 수 있습니다.



### binarize\_cpu

```c
void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}
```

함수 이름: binarize\_cpu

입력:

* input: float 형태의 1차원 배열
* n: input 배열의 길이
* binary: float 형태의 1차원 배열

동작:

* input 배열의 요소를 0을 기준으로 1 또는 -1로 이진화하여 binary 배열에 저장함.

설명:

* 이진화란, 입력값을 0과 1 또는 -1과 1과 같은 이진수 형태로 바꾸는 작업을 의미함.
* 이 함수에서는 입력된 input 배열의 요소를 0을 기준으로 1 또는 -1로 이진화하여 binary 배열에 저장함.
* 이진화된 값은 다음과 같은 조건문으로 결정됨:
  * input\[i] > 0 이면, binary\[i] = 1
  * input\[i] <= 0 이면, binary\[i] = -1



### binarize\_input

```c
void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}
```

함수 이름: binarize\_input

입력:

* float \*input: 이진화할 입력 배열
* int n: 입력 배열의 채널 수
* int size: 입력 배열의 크기 (가로 또는 세로 한 변의 길이)
* float \*binary: 이진화된 값을 저장할 출력 배열

동작:&#x20;

* 입력 배열을 이진화하여 출력 배열에 저장합니다.

설명:&#x20;

* 입력 배열의 각 채널과 위치에 대해 평균값을 계산하고, 입력 값이 평균보다 크면 1, 작으면 -1을 출력 배열에 저장합니다.&#x20;
* 평균값은 각 채널과 위치마다 다르게 계산됩니다.



### convolutional\_out\_height

```c
int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}
```

함수 이름: convolutional\_out\_height

입력:&#x20;

* convolutional\_layer l: 합성곱 레이어 구조체

동작:&#x20;

* 합성곱 레이어의 입력 이미지 높이(l.h), 패딩 크기(l.pad), 필터 크기(l.size), 및 스트라이드 크기(l.stride)를 고려하여 출력 이미지 높이를 계산한다.

설명:&#x20;

* 합성곱 레이어에서 필터와 입력 이미지의 합성곱 연산을 수행하면 출력 이미지가 생성된다.&#x20;
* 이때, 출력 이미지의 높이를 계산하는 함수이다.&#x20;
* 합성곱 레이어 구조체에서 입력 이미지의 높이(l.h), 패딩 크기(l.pad), 필터 크기(l.size), 및 스트라이드 크기(l.stride)를 이용하여 출력 이미지의 높이를 계산하고 반환한다.



### convolutional\_out\_width

```c
int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}
```

함수 이름: convolutional\_out\_width

입력:

* convolutional\_layer l: 컨볼루션 레이어 구조체

동작:

* 입력 이미지의 너비, 패딩, 스트라이드, 필터 크기를 고려하여 컨볼루션 레이어의 출력 너비를 계산한다.

설명:

* 컨볼루션 레이어의 출력 너비를 반환하는 함수이다.
* 입력 이미지의 너비, 패딩, 스트라이드, 필터 크기를 고려하여 컨볼루션 레이어의 출력 너비를 계산하고 반환한다.
* 출력 너비는 아래의 공식을 따른다: (입력 너비 + 2 x 패딩 - 필터 크기) / 스트라이드 + 1



### get\_convolutional\_image

```c
image get_convolutional_image(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.output);
}
```

함수 이름: get\_convolutional\_image

입력:&#x20;

* convolutional\_layer 구조체

동작:&#x20;

* convolutional\_layer의 출력을 float\_to\_image 함수를 사용하여 image 구조체로 변환

설명:&#x20;

* convolutional\_layer의 출력을 이미지 형식으로 변환하여 반환하는 함수이다.&#x20;
* 반환된 이미지는 float\_to\_image 함수를 사용하여 변환된다.



### get\_convolutional\_delta

```c
image get_convolutional_delta(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}
```

함수 이름: get\_convolutional\_delta

입력:&#x20;

* convolutional\_layer 구조체

동작:&#x20;

* convolutional\_layer 구조체 내 delta 배열을 out\_w, out\_h, out\_c 크기의 이미지 구조체 형태로 변환

설명:&#x20;

* convolutional\_layer 구조체 내 delta 배열은 해당 층에서 역전파(backpropagation)를 통해 계산된 출력 값의 오차 값을 저장하고 있다.&#x20;
* 이 함수는 해당 delta 배열을 이미지 형태로 변환하여 반환한다.&#x20;
* 이 때, 변환된 이미지의 크기는 해당 층의 출력 값 크기(out\_w, out\_h, out\_c)와 동일하다.



### get\_convolutional\_weight

```c
image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c/l.groups;
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}
```

함수 이름: get\_convolutional\_weight

입력:

* convolutional\_layer l: 컨볼루션 레이어 구조체
* int i: 가져올 가중치의 인덱스

동작:

* 컨볼루션 레이어에서 i번째 가중치의 크기를 이용하여 float\_to\_image 함수를 호출해 가중치 이미지를 생성한다.

설명:

* 컨볼루션 레이어에서 i번째 가중치의 값을 이용하여 해당 가중치를 시각화할 수 있는 이미지를 생성한다.
* 생성된 이미지는 float\_to\_image 함수를 이용하여 생성되며, 이미지의 크기는 가중치의 크기와 동일하다.



### rgbgr\_weights

```c
void rgbgr_weights(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}
```

함수 이름: rgbgr\_weights&#x20;

입력:&#x20;

* convolutional\_layer l: 컨볼루션 레이어

동작:

* 각 컨볼루션 레이어의 가중치 이미지를 가져온다.
* 이미지의 채널이 3인 경우, rgbgr\_image 함수를 사용하여 RGB 이미지를 BGR 이미지로 변환한다.
* 모든 가중치 이미지에 대해 위 동작을 수행한다.&#x20;

설명:

* 이 함수는 컨볼루션 레이어의 가중치 이미지를 BGR 채널 순서로 변환하는 기능을 수행한다.
* 컬러 이미지에서는 색상 채널의 순서가 RGB가 일반적이지만, OpenCV 라이브러리에서는 BGR 채널 순서를 사용하므로 이러한 변환이 필요하다.
* 이 함수는 주로 딥러닝 모델을 OpenCV와 같은 라이브러리에서 사용할 때 유용하게 사용된다. 하위 설명:
* for 문에서 i 변수를 초기화하고, l.n만큼 반복한다.
* get\_convolutional\_weight 함수를 사용하여 i번째 가중치 이미지를 가져온다.
* 이미지의 채널이 3인 경우에만 rgbgr\_image 함수를 사용하여 이미지를 변환한다.



### rescale\_weights

```c
void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}
```

함수 이름: rescale\_weights

입력:

* convolutional\_layer l : rescale할 레이어
* float scale : 가중치를 곱할 스케일 값
* float trans : 편향(bias) 값을 조절할 값

동작:&#x20;

* convolutional layer에서 가중치(weight)를 가져와서 이미지를 rescale하고, 이미지의 총 합(sum)을 구한 후, 편향(bias) 값을 trans만큼 더해줌

설명:

* convolutional\_layer l에서 가중치(weight) 이미지를 가져와서 이미지의 채널 수(c)가 3일 때(rescale하고자 하는 이미지가 RGB 이미지일 때), 이미지를 scale 값만큼 곱해줌
* 이미지의 모든 픽셀 값의 합(sum)을 구함
* l.biases\[i]에 sum \* trans 값을 더해줌으로써, 편향(bias) 값을 trans만큼 더해줌



### get\_weights

```c
image *get_weights(convolutional_layer l)
{
    image *weights = calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
           char buff[256];
           sprintf(buff, "filter%d", i);
           save_image(weights[i], buff);
         */
    }
    //error("hey");
    return weights;
}
```

함수 이름: get\_weights

입력:

* convolutional\_layer l: 합성곱 레이어 객체

동작:

* 합성곱 레이어의 가중치를 가져와서, 가중치 이미지들을 생성
* 생성된 가중치 이미지들을 정규화(normalize)
* 가중치 이미지들을 반환

설명:

* 함수는 입력으로 받은 합성곱 레이어의 가중치를 가져와 가중치 이미지들을 생성하고, 생성된 가중치 이미지들을 정규화(normalize)합니다.
* 정규화(normalize)란, 이미지 픽셀 값들을 0과 1사이의 값으로 조정하는 것을 말합니다.
* 생성된 가중치 이미지들은 포인터 배열로 반환됩니다.



### visualize\_convolutional\_layer

```c
image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}
```

함수 이름: visualize\_convolutional\_layer

입력:

* convolutional\_layer l: 시각화할 합성곱 레이어
* char \*window: 시각화할 윈도우 이름
* image \*prev\_weights: 이전 가중치 이미지 포인터

동작:

* 합성곱 레이어의 가중치 이미지와 출력 이미지를 시각화하고,
* 시각화된 가중치 이미지를 반환한다.

설명:

* convolutional\_layer 구조체에 저장된 가중치 값들을 이미지로 변환한다.
* 변환된 이미지들을 윈도우에 시각화하여 보여준다.
* 합성곱 레이어의 출력 이미지를 이미지 collapse를 통해 한 장으로 만들어서 보여준다.
* 반환되는 single\_weights 포인터는 가중치 이미지를 담고 있는 이미지 배열이다.

