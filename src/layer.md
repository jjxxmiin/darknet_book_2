# layer

```c
// darknet.h

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    ISEG,
    REORG,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    BLANK
} LAYER_TYPE;
```

이 코드는 열거형(enum)으로 LAYER\_TYPE이라는 타입을 정의하고 있습니다. LAYER\_TYPE은 다양한 레이어 유형을 정의하고 있으며, 각 레이어 유형은 해당하는 이름으로 정의되어 있습니다.

다음은 각 레이어 유형과 그에 해당하는 이름입니다.

* CONVOLUTIONAL: 컨볼루션(Convolution) 레이어
* DECONVOLUTIONAL: 디컨볼루션(Deconvolution) 레이어
* CONNECTED: 완전 연결(Fully Connected) 레이어
* MAXPOOL: 맥스 풀링(Max Pooling) 레이어
* SOFTMAX: 소프트맥스(Softmax) 레이어
* DETECTION: 객체 검출(Detection) 레이어
* DROPOUT: 드롭아웃(Dropout) 레이어
* CROP: 크롭(Crop) 레이어
* ROUTE: 루트(Route) 레이어
* COST: 비용(Cost) 레이어
* NORMALIZATION: 정규화(Normalization) 레이어
* AVGPOOL: 평균 풀링(Average Pooling) 레이어
* LOCAL: 로컬(Local) 레이어
* SHORTCUT: 숏컷(Shortcut) 레이어
* ACTIVE: 활성화(Activation) 레이어
* RNN: 순환 신경망(Recurrent Neural Network) 레이어
* GRU: 게이트 순환 유닛(Gated Recurrent Unit) 레이어
* LSTM: 장단기 메모리(Long Short-Term Memory) 레이어
* CRNN: 합성곱 순환 신경망(Convolutional Recurrent Neural Network) 레이어
* BATCHNORM: 배치 정규화(Batch Normalization) 레이어
* NETWORK: 네트워크(Network) 레이어
* XNOR: 이진화(Binary) 레이어
* REGION: 지역(Region) 레이어
* YOLO: YOLO(You Only Look Once) 레이어
* ISEG: 인스턴스 분할(Instance Segmentation) 레이어
* REORG: 리오그(Reorg) 레이어
* UPSAMPLE: 업샘플(Upsample) 레이어
* LOGXENT: 로그-엔트로피(Log-entropy) 레이어
* L2NORM: L2 노름(L2 Norm) 레이어
* BLANK: 빈(Blank) 레이어

이 함수는 LAYER\_TYPE이라는 열거형을 정의한 것이므로 입력값과 동작은 없습니다.



## free\_layer

```c
void free_layer(layer l)
{
    if(l.type == DROPOUT){
        if(l.rand)           free(l.rand);
        return;
    }
    if(l.cweights)           free(l.cweights);
    if(l.indexes)            free(l.indexes);
    if(l.input_layers)       free(l.input_layers);
    if(l.input_sizes)        free(l.input_sizes);
    if(l.map)                free(l.map);
    if(l.rand)               free(l.rand);
    if(l.cost)               free(l.cost);
    if(l.state)              free(l.state);
    if(l.prev_state)         free(l.prev_state);
    if(l.forgot_state)       free(l.forgot_state);
    if(l.forgot_delta)       free(l.forgot_delta);
    if(l.state_delta)        free(l.state_delta);
    if(l.concat)             free(l.concat);
    if(l.concat_delta)       free(l.concat_delta);
    if(l.binary_weights)     free(l.binary_weights);
    if(l.biases)             free(l.biases);
    if(l.bias_updates)       free(l.bias_updates);
    if(l.scales)             free(l.scales);
    if(l.scale_updates)      free(l.scale_updates);
    if(l.weights)            free(l.weights);
    if(l.weight_updates)     free(l.weight_updates);
    if(l.delta)              free(l.delta);
    if(l.output)             free(l.output);
    if(l.squared)            free(l.squared);
    if(l.norms)              free(l.norms);
    if(l.spatial_mean)       free(l.spatial_mean);
    if(l.mean)               free(l.mean);
    if(l.variance)           free(l.variance);
    if(l.mean_delta)         free(l.mean_delta);
    if(l.variance_delta)     free(l.variance_delta);
    if(l.rolling_mean)       free(l.rolling_mean);
    if(l.rolling_variance)   free(l.rolling_variance);
    if(l.x)                  free(l.x);
    if(l.x_norm)             free(l.x_norm);
    if(l.m)                  free(l.m);
    if(l.v)                  free(l.v);
    if(l.z_cpu)              free(l.z_cpu);
    if(l.r_cpu)              free(l.r_cpu);
    if(l.h_cpu)              free(l.h_cpu);
    if(l.binary_input)       free(l.binary_input);
}

```

함수 이름: free\_layer

입력:&#x20;

* layer 구조체 (layer 타입 포인터 변수 l)

동작:&#x20;

* layer 구조체에서 동적으로 할당한 모든 메모리를 해제하는 함수.&#x20;
* DROPOUT 레이어인 경우 l.rand 변수만 해제하고 함수를 종료한다.

설명:&#x20;

* 이 함수는 입력으로 전달된 layer 구조체에서 동적으로 할당된 모든 메모리를 해제한다.&#x20;
* 할당된 메모리가 없는 경우 아무런 동작도 하지 않는다. DROPOUT 레이어인 경우 l.rand 변수만 해제하고 함수를 종료한다.&#x20;
* 나머지 레이어의 경우, layer 구조체에서 사용하는 모든 변수를 순회하며 할당된 메모리가 있는 경우 메모리를 해제한다.&#x20;
* 각 변수에 대한 메모리 해제는 malloc 함수를 사용하여 할당된 것과 동일한 방식으로 이루어진다.

