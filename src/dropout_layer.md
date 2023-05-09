# dropout\_layer

## Dropout Layer란?

Dropout Layer는 딥러닝에서 오버피팅을 방지하기 위한 regularization 기법 중 하나입니다. 이 레이어는 학습 중에 일부 뉴런을 무작위로 선택하여 출력을 0으로 만드는 과정을 수행합니다. 이렇게 함으로써 네트워크가 특정 뉴런에 과도하게 의존하지 않도록 하고, 뉴런의 가중치가 전체 데이터셋에 대해 일반화되도록 합니다.

Dropout Layer는 일반적으로 fully connected layer나 convolutional layer 다음에 추가됩니다. 학습 중에 Dropout Layer를 통과한 출력값은 실제로 학습에 사용되지 않습니다. 대신, 학습이 완료된 후에는 모든 뉴런을 사용하여 출력을 계산합니다. 이는 일종의 앙상블 학습과 유사한 효과를 가지며, 오버피팅을 줄이고 일반화 성능을 향상시킵니다.

Dropout Layer의 사용 여부와 dropout 비율은 하이퍼파라미터로써 조절됩니다. 이 값은 신경망의 복잡성과 데이터셋의 크기에 따라 조정될 수 있습니다. 또한, Dropout Layer의 사용 여부와 비율은 네트워크의 일반화 성능과 학습 속도에 큰 영향을 미칩니다.



## forward\_dropout\_layer

```c
void forward_dropout_layer(dropout_layer l, network net)
{
    int i;
    if (!net.train) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = rand_uniform(0, 1);
        l.rand[i] = r;
        if(r < l.probability) net.input[i] = 0;
        else net.input[i] *= l.scale;
    }
}
```

함수 이름: forward\_dropout\_layer

입력:

* dropout\_layer l: dropout layer의 구조체
* network net: neural network의 구조체

동작:

* neural network에서 dropout layer를 수행하는 forward propagation 함수
* dropout layer는 입력값의 일부를 0으로 만들어주는 역할을 한다.
* 만약 현재가 학습 모드인 경우, 각 입력값에 대해 확률 p(주어진 확률값)보다 작은 값인 경우 해당 입력값을 0으로 설정한다. 그렇지 않은 경우 해당 입력값을 (1-p)배 해준다.
* 이때, dropout layer는 입력값이 0으로 바뀐 비율만큼의 scale factor를 유지한다. (이후 backpropagation 시 활용)

설명:

* dropout layer는 overfitting을 방지하기 위한 regularization 방법 중 하나로, 특히 deep neural network에서 효과적이다.
* 학습 시, dropout layer는 입력값 중 일부를 무작위로 선택하여 0으로 만들어준다. 이는 모델이 특정 feature에 과도하게 의존하지 않도록 하여, generalization 능력을 향상시켜준다.
* 테스트 시, dropout layer는 사용되지 않는다. 대신 학습 시 사용된 확률 p를 이용하여 입력값에 (1-p)를 곱해줌으로써, 학습 시 dropout이 적용된 모델의 예측 결과를 보정해준다.
* dropout layer는 fully connected layer와 convolutional layer 모두에서 사용될 수 있다.



## backward\_dropout\_layer

```c
void backward_dropout_layer(dropout_layer l, network net)
{
    int i;
    if(!net.delta) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = l.rand[i];
        if(r < l.probability) net.delta[i] = 0;
        else net.delta[i] *= l.scale;
    }
}
```

함수 이름: backward\_dropout\_layer

입력:&#x20;

* dropout\_layer l: 드롭아웃 레이어 구조체
* network net: 신경망 구조체

동작:&#x20;

* 드롭아웃 레이어의 역전파(forward pass)를 수행한다.&#x20;
* 역전파 시, 랜덤하게 선택된 입력 값에 대해서만 그래디언트(gradient)를 계산하여 출력값을 갱신한다.

설명:

* 네트워크가 학습 상태인 경우에만 드롭아웃 레이어의 역전파를 수행한다.
* 랜덤하게 선택된 입력 값에 대한 그래디언트는 계산하지 않고 0으로 설정하여 출력값을 갱신한다.
* 그 외의 입력 값에 대해서는 scale 값에 따라 그래디언트를 계산하여 출력값을 갱신한다.



## resize\_dropout\_layer

```c
void resize_dropout_layer(dropout_layer *l, int inputs)
{
    l->rand = realloc(l->rand, l->inputs*l->batch*sizeof(float));
    #ifdef GPU
    cuda_free(l->rand_gpu);

    l->rand_gpu = cuda_make_array(l->rand, inputs*l->batch);
    #endif
}
```

함수 이름: resize\_dropout\_layer

입력:&#x20;

* dropout\_layer 구조체 포인터 l
* int inputs

동작:&#x20;

* dropout 레이어의 랜덤 드롭아웃 마스크를 입력 수에 맞게 조절한다.&#x20;
* 입력 수가 이전에 설정된 입력 수보다 작은 경우, 마스크를 새로운 크기에 맞게 조정한다.&#x20;
* GPU 버전의 경우 CUDA 메모리를 다시 할당하고, 업데이트된 랜덤 드롭아웃 마스크를 복사한다.

설명:&#x20;

* 입력으로 받은 dropout\_layer 구조체 포인터 l의 rand 배열의 크기를 inputs_l->batch_sizeof(float)로 재할당한다.&#x20;
* GPU가 활성화되어 있는 경우, 이전에 할당된 CUDA 메모리를 해제하고 새로운 크기에 맞게 다시 할당한다.&#x20;
* 이후, 새로운 rand 배열을 CUDA 메모리로 복사한다.



## make\_dropout\_layer

```c
dropout_layer make_dropout_layer(int batch, int inputs, float probability)
{
    dropout_layer l = {0};
    l.type = DROPOUT;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    l.rand = calloc(inputs*batch, sizeof(float));
    l.scale = 1./(1.-probability);
    l.forward = forward_dropout_layer;
    l.backward = backward_dropout_layer;
    #ifdef GPU
    l.forward_gpu = forward_dropout_layer_gpu;
    l.backward_gpu = backward_dropout_layer_gpu;
    l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
    #endif
    fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    return l;
}
```

함수 이름: make\_dropout\_layer&#x20;

* 입력: batch(int): 배치 크기
* inputs(int): 입력 크기
* probability(float): 드롭아웃 확률&#x20;

동작:&#x20;

* 드롭아웃 레이어를 생성하고 초기화한다.&#x20;

설명:

* 드롭아웃 레이어 구조체를 선언하고 초기화한다.
* 드롭아웃 레이어의 타입을 DROPOUT으로 설정한다.
* 드롭아웃 확률, 입력 크기, 출력 크기, 배치 크기를 설정한다.
* 입력 크기와 출력 크기가 같으므로 l.outputs = l.inputs로 설정한다.
* 배치 크기와 입력 크기를 곱한 만큼의 크기를 갖는 난수 배열 l.rand를 생성하고 초기화한다.
* 스케일링 파라미터 l.scale을 계산한다.
* forward\_dropout\_layer와 backward\_dropout\_layer 함수를 설정한다.
* GPU를 사용하는 경우, forward\_dropout\_layer\_gpu와 backward\_dropout\_layer\_gpu 함수도 설정하고, 난수 배열 l.rand\_gpu를 생성하고 초기화한다.
* 생성한 드롭아웃 레이어의 정보를 출력한다.
* 생성한 드롭아웃 레이어 구조체를 반환한다.

