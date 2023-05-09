# batchnorm\_layer

## Batch Normalization 이란?

Paper : [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)

* Gradient Vanishing, Gradient Exploding 문제점
* `internal covariate shift` : weight의 변화가 중첩되어 가중되는 크기 변화가 크다는 문제점
  * `careful initialization` : `Difficult`
  * `small learning rate` : `Slow`

위에 문제를 해결하기 위한 하나의 방법 입니다.

보통 `internal covariate shift`를 줄이기 위한 대표적인 방법은 각 layer의 입력에 whitening을 하는 것 입니다. 여기에서 whitening 이란 평균 0 분산 1로 바꾸어 주는 것(정규화)을 말합니다. 하지만 이러한 연산은 문제가 있습니다.

* bias의 영향이 무시 됩니다.

만약 $$y = WX + b$$ 연산을 한 뒤에 정규화하기 위해서 평균을 빼주는 경우 $$\hat{y} = y - mean(y)$$ bias $$b$$의 영향이 사라지게 됩니다.(bias는 고정 스칼라 값이기 때문에 평균을 구해도 같은 값이 나옵니다.)

* 비선형성이 없어질 수 있습니다.

만약 sigmoid를 통과하는 경우 대부분의 입력값은 sigmoid의 중간 부분에 속합니다. sigmoid에서 중간은 선형이기 때문에 비선형성이 사라질 수 있다는 것입니다.

이러한 문제를 해결하기 위해 `batch Normalization`이 나왔습니다.

* $$m$$ : mini-batch의 크기
* $$\mu$$ : mean
* $$\sigma$$ : std
* $$\gamma$$ : scale
* $$\beta$$ : shifts
* $$\gamma, \beta$$는 학습 가능한 파라미터 입니다. 이것이 비선형성을 완화시키기 위한 파라미터 입니다.

배치 정규화는 학습 하는 경우에는 미니 배치의 평균과 분산을 구할 수 있지만 추론을 하는 경우는 미니 배치가 없기 때문에 학습 하는 동안 계산 된 `이동 평균`을 사용 합니다.

* 이동 평균 : 각 미니 배치 평균의 평균
* 이동 분산 : 각 미니 배치 분산의 평균 \* m/(m-1) \[Bessel’s Correction]

CNN의 경우 bias의 역할을 $$\beta$$가 대신 하기 때문에 bias를 제거합니다. 그리고 컨볼루션 연산을 통해 출력되는 특징 맵으로 각 채널마다 평균과 분산을 계산하고 $$\gamma, \beta$$를 만듭니다. 즉, 채널의 개수 만큼 $$\gamma, \beta$$가 생겨납니다.

### 장점

* internal covariate shift 문제를 해결한다.
* learning rate를 크게 해도 된다.
* 신중하게 초기값을 정할 필요가 없다.
* dropout을 대체 할 수 있다.



### forward\_batchnorm\_layer

```c
void forward_batchnorm_layer(layer l, network net)
{
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    copy_cpu(l.outputs*l.batch, l.output, 1, l.x, 1);
    if(net.train){
        mean_cpu(l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);
        variance_cpu(l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance);

        scal_cpu(l.out_c, .99, l.rolling_mean, 1);
        axpy_cpu(l.out_c, .01, l.mean, 1, l.rolling_mean, 1);
        scal_cpu(l.out_c, .99, l.rolling_variance, 1);
        axpy_cpu(l.out_c, .01, l.variance, 1, l.rolling_variance, 1);

        normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);   
        copy_cpu(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
    } else {
        normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
    }
    scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
    add_bias(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
}
```

\
함수 이름: forward\_batchnorm\_layer

입력:&#x20;

* l: layer 구조체
* net: network 구조체

동작:&#x20;

* Batch normalization 레이어를 수행합니다.

설명:&#x20;

* 이 함수는 입력 데이터에 대해 Batch normalization을 수행합니다. 입력으로는 layer 구조체와 network 구조체가 필요합니다.
* 함수는 입력 데이터를 복사한 후, 학습 모드와 추론 모드를 구분하여 처리합니다. 학습 모드에서는 현재 배치에 대한 평균과 분산을 계산하고, 이를 사용하여 입력 데이터를 정규화합니다. 그 다음, 정규화된 데이터에 스케일과 바이어스를 적용합니다. 스케일과 바이어스는 layer 구조체 내의 scales 및 biases 필드에서 가져옵니다.
* 반면, 추론 모드에서는 배치에 대한 이동 평균과 이동 분산을 사용하여 입력 데이터를 정규화합니다. 이동 평균과 이동 분산은 layer 구조체 내의 rolling\_mean 및 rolling\_variance 필드에서 가져옵니다.
* 결과적으로, Batch normalization은 입력 데이터를 정규화하여 모델 학습을 안정화하고, 훨씬 빠르게 수렴하도록 도와줍니다.



### backward\_batchnorm\_layer

```c
void backward_batchnorm_layer(layer l, network net)
{
    if(!net.train){
        l.mean = l.rolling_mean;
        l.variance = l.rolling_variance;
    }
    backward_bias(l.bias_updates, l.delta, l.batch, l.out_c, l.out_w*l.out_h);
    backward_scale_cpu(l.x_norm, l.delta, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates);

    scale_bias(l.delta, l.scales, l.batch, l.out_c, l.out_h*l.out_w);

    mean_delta_cpu(l.delta, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta);
    variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta);
    normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.out_c, l.out_w*l.out_h, l.delta);
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}
```

함수 이름: backward\_batchnorm\_layer

입력:&#x20;

* l: layer 구조체
* net: network 구조체

동작:

* net.train이 false이면 현재 층의 rolling mean과 rolling variance로 l.mean과 l.variance를 대체한다.
* bias 업데이트와 scale 업데이트를 수행한다.
* delta를 scale로 곱하고 bias를 더해준다.
* delta에 대한 mean delta를 계산한다.
* delta에 대한 variance delta를 계산한다.
* delta를 정규화한다.
* 만약 l.type이 BATCHNORM이면 delta를 net.delta로 복사한다.

설명:&#x20;

* 이 함수는 배치 정규화 층의 역전파(backpropagation)를 수행한다. 배치 정규화 층의 역전파는 순전파(forward propagation)와는 다르게 여러 단계로 구성되어 있어 복잡하다. 이 함수는 이러한 단계들을 수행하여 역전파를 구현한다.
* 우선, net.train이 false인 경우 현재 층의 rolling mean과 rolling variance로 l.mean과 l.variance를 대체한다. rolling mean과 rolling variance는 현재 mini-batch 이전의 모든 데이터셋에서 계산된 평균과 분산을 저장하고 있다. 따라서 이전 데이터셋의 통계량을 사용하여 현재 mini-batch의 정규화를 수행한다.
* 다음으로, bias 업데이트와 scale 업데이트를 수행한다. 이전 층에서 업데이트한 bias와 scale을 사용하여 현재 층의 bias와 scale을 업데이트한다.
* 그 후, delta를 scale로 곱하고 bias를 더해준다. 이는 순전파에서 delta를 정규화하기 전에 수행한 scale과 bias의 연산을 역전파하는 것이다.
* 그 다음, delta에 대한 mean delta를 계산한다. 이는 순전파에서 정규화된 입력 값에 대한 delta를 계산하기 위해 사용한 mean을 역전파하는 것이다.
* delta에 대한 variance delta를 계산한 후, 이를 사용하여 delta를 정규화한다. 이는 순전파에서 정규화된 입력 값에 대한 delta를 계산하기 위해 사용한 variance를 역전파하는 것이다.
* 마지막으로, 만약 l.type이 BATCHNORM인 경우 delta를 net.delta로 복사한다. 이는 이전 층의 delta에 대한 역전파를 위한 작업이다.



### make\_batchnorm\_layer

```c
layer make_batchnorm_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "Batch Normalization Layer: %d x %d x %d image\n", w,h,c);
    layer l = {0};
    l.type = BATCHNORM;
    l.batch = batch;
    l.h = l.out_h = h;
    l.w = l.out_w = w;
    l.c = l.out_c = c;
    l.output = calloc(h * w * c * batch, sizeof(float));
    l.delta  = calloc(h * w * c * batch, sizeof(float));
    l.inputs = w*h*c;
    l.outputs = l.inputs;

    l.scales = calloc(c, sizeof(float));
    l.scale_updates = calloc(c, sizeof(float));
    l.biases = calloc(c, sizeof(float));
    l.bias_updates = calloc(c, sizeof(float));
    int i;
    for(i = 0; i < c; ++i){
        l.scales[i] = 1;
    }

    l.mean = calloc(c, sizeof(float));
    l.variance = calloc(c, sizeof(float));

    l.rolling_mean = calloc(c, sizeof(float));
    l.rolling_variance = calloc(c, sizeof(float));

    l.forward = forward_batchnorm_layer;
    l.backward = backward_batchnorm_layer;

    return l;
}
```

함수 이름: make\_batchnorm\_layer

입력:&#x20;

* batch: 배치 크기
* w: 이미지 너비
* h: 이미지 높이
* c: 채널 수

동작:&#x20;

* 배치 정규화 레이어를 생성하고 초기화합니다.&#x20;
* 배치 정규화의 스케일과 바이어스 값을 1과 0으로 초기화하고, 평균과 분산, 그리고 롤링 평균과 롤링 분산 값을 0으로 초기화합니다.&#x20;
* 입력과 출력의 크기를 설정하고, forward\_batchnorm\_layer와 backward\_batchnorm\_layer 함수를 설정합니다.

설명:&#x20;

* 이 함수는 배치 정규화 레이어를 생성하고 초기화합니다.&#x20;
* 배치 정규화 레이어는 인공 신경망에서 입력값을 정규화하는 레이어로, 입력값의 분포를 안정화하여 학습을 더욱 안정적으로 만듭니다. 이 함수에서는 배치 정규화 레이어를 생성하고 필요한 변수들을 초기화합니다.&#x20;
* 입력값과 출력값의 크기는 인자로 받은 값에 따라 설정하며, 스케일, 바이어스, 평균, 분산 등의 변수도 초기화합니다. 이 함수에서는 초기 스케일 값을 1로, 바이어스 값을 0으로 설정합니다.&#x20;
* 평균과 분산, 롤링 평균과 롤링 분산은 모두 0으로 초기화됩니다.&#x20;
* 이 함수에서는 forward\_batchnorm\_layer와 backward\_batchnorm\_layer 함수를 설정하며, 이 함수를 통해 생성된 배치 정규화 레이어는 인공 신경망 모델의 구성 요소로 활용됩니다.



### backward\_scale\_cpu

```c
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    int i,b,f;
    for(f = 0; f < n; ++f){
        float sum = 0;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < size; ++i){
                int index = i + size*(f + n*b);
                sum += delta[index] * x_norm[index];
            }
        }
        scale_updates[f] += sum;
    }
}
```

함수 이름: backward\_scale\_cpu

입력:

* x\_norm: normalization된 입력값을 가리키는 포인터(float 배열)
* delta: 출력값에 대한 손실의 미분값을 가리키는 포인터(float 배열)
* batch: 미니배치 크기(int)
* n: 필터 수(int)
* size: 필터 크기(int)
* scale\_updates: 스케일 매개변수의 업데이트 값을 저장할 포인터(float 배열)

동작:&#x20;

* 입력값을 normalization한 후, 출력값에 대한 손실의 미분값과 곱한 결과를 미니배치와 필터, 크기별로 합하여 스케일 매개변수의 업데이트 값을 계산합니다.

설명:&#x20;

* Convolutional Neural Network에서 Batch Normalization 계층에서 사용되는 함수 중 하나로, 스케일 매개변수의 업데이트 값을 계산하는 함수입니다.&#x20;
* 스케일 매개변수는 정규화된 입력값에 대해 곱해지는 매개변수로, 업데이트는 이 매개변수가 손실을 줄이는 방향으로 조정됩니다.&#x20;
* 이 함수는 backward propagation 단계에서 호출되며, 출력값에 대한 손실의 미분값과 normalization된 입력값의 곱을 미니배치와 필터, 크기별로 합하여 스케일 매개변수의 업데이트 값을 계산합니다.



### mean\_delta\_cpu

```c
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean_delta[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j*filters*spatial + i*spatial + k;
                mean_delta[i] += delta[index];
            }
        }
        mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
    }
}
```

함수 이름: mean\_delta\_cpu

입력:

* delta: 이전 층의 델타 값 (float 형태의 1차원 배열)
* variance: 현재 층의 분산 값 (float 형태의 1차원 배열)
* batch: 배치 사이즈 (int)
* filters: 필터 개수 (int)
* spatial: 공간 차원의 크기 (int)
* mean\_delta: 현재 층의 평균 델타 값 (float 형태의 1차원 배열)

동작:

* 현재 층의 평균 델타 값을 계산하는 함수
* 각 필터별로 델타 값의 합을 구하고, 분산의 제곱근 값으로 나누어 평균 델타 값을 구함

설명:

* mean\_delta\_cpu 함수는 Batch Normalization의 학습 과정 중 현재 층의 평균 델타 값을 계산하는 함수이다.
* 각 필터별로 델타 값의 합을 구한 후, 해당 필터의 분산 값의 제곱근으로 나누어 평균 델타 값을 계산한다.
* 이때 분산 값에 0으로 나누는 것을 방지하기 위해 작은 상수값(.00001f)을 더해준다.
* 계산된 평균 델타 값은 mean\_delta 배열에 저장된다.



### variance\_delta\_cpu

```c
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance_delta[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance_delta[i] += delta[index]*(x[index] - mean[i]);
            }
        }
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
    }
}
```

함수 이름: variance\_delta\_cpu

입력:

* x: 현재 층의 입력
* delta: 현재 층의 델타
* mean: 현재 층의 평균
* variance: 현재 층의 분산
* batch: 배치 크기
* filters: 필터 개수
* spatial: 공간 크기
* variance\_delta: 분산 델타

동작:&#x20;

* 현재 층의 분산 델타를 계산하는 함수입니다.&#x20;
* 입력으로 현재 층의 입력(x), 델타(delta), 평균(mean), 분산(variance), 배치 크기(batch), 필터 개수(filters), 공간 크기(spatial)가 주어집니다.&#x20;
* 각 필터별로 분산 델타를 계산하며, 이를 위해 각 배치에서 현재 필터와 공간 위치에 따른 인덱스를 계산합니다.&#x20;
* 분산 델타는 delta와 (x-mean)의 곱의 합으로 계산되며, 이 값에 -(variance+0.00001f)^(3/2)를 곱한 값의 반대값을 저장합니다.

설명:&#x20;

* 배치 정규화(batch normalization)에서는 각 층에서 입력(x)에 대한 평균(mean)과 분산(variance)을 계산합니다.&#x20;
* 그리고 분산 델타(variance delta)는 이전 층의 출력값과 현재 층의 델타를 이용하여 계산됩니다.&#x20;
* 이 함수는 분산 델타를 계산하기 위한 함수 중 하나로, 현재 층의 입력, 델타, 평균, 분산을 이용하여 분산 델타를 계산합니다.



### normalize\_delta\_cpu

```c
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    int f, j, k;
    for(j = 0; j < batch; ++j){
        for(f = 0; f < filters; ++f){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + f*spatial + k;
                delta[index] = delta[index] * 1./(sqrt(variance[f] + .00001f)) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
}
```

함수 이름: normalize\_delta\_cpu

입력:

* float \*x: 입력값 포인터
* float \*mean: 평균값 포인터
* float \*variance: 분산값 포인터
* float \*mean\_delta: 평균값 변화량 포인터
* float \*variance\_delta: 분산값 변화량 포인터
* int batch: 배치 크기
* int filters: 필터 개수
* int spatial: 공간 크기
* float \*delta: 델타값 포인터

동작:

* 입력값 x, 평균값 mean, 분산값 variance, 평균값 변화량 mean\_delta, 분산값 변화량 variance\_delta, 배치 크기 batch, 필터 개수 filters, 공간 크기 spatial, 델타값 delta를 받아서 델타값 delta를 정규화(normalize)한다.
* 정규화를 하기 위해 델타값 delta를 평균(mean)과 분산(variance)을 이용하여 표준화(standardize)한다.
* 또한, 평균값 변화량 mean\_delta, 분산값 변화량 variance\_delta를 이용하여 평균과 분산의 변화량을 추가로 반영한다.

설명:

* 입력값과 델타값은 모두 (batch \* filters \* spatial) 크기의 1차원 배열로 표현된다.
* 이 함수는 CPU에서 동작하며, GPU에서 동작하는 버전도 존재한다.
* 일반적으로 딥러닝에서 입력값을 정규화하는 것은 학습의 안정성과 성능 향상을 도모하기 위한 방법 중 하나이다.
* 입력값 x를 평균 mean과 분산 variance를 이용하여 표준화한 후, 델타값 delta에 다시 곱해줌으로써 입력값을 정규화한다.
* 평균값 변화량 mean\_delta와 분산값 변화량 variance\_delta는 이전 배치에서의 값들을 고려하여 새로운 배치에서의 평균과 분산이 어떻게 변화하는지 추정하기 위한 값이다.

