# cost\_layer

Loss를 구하기 위한 layer입니다.

## COST\_TYPE

```c
typedef enum{
    SSE, MASKED, L1, SEG, SMOOTH,WGAN
} COST_TYPE;
```

위 코드는 COST\_TYPE라는 열거형(enum)을 정의하는 코드입니다. COST\_TYPE은 네트워크의 손실 함수(loss function)를 지정하기 위해 사용됩니다.

이 열거형은 다섯 가지의 상수(constant)를 정의하고 있습니다:

* SSE : 평균 제곱 오차(Mean Squared Error, MSE) 손실 함수를 나타냅니다. 실제 값과 예측 값의 차이를 제곱한 후 모든 입력에 대해 평균을 취한 값으로, 예측이 정확할수록 값이 작아집니다.
* MASKED : 마스크(mask)를 적용한 평균 제곱 오차 손실 함수입니다. 마스크는 일부 입력을 무시하고 손실을 계산하는 데 사용됩니다.
* L1 : 절대 오차(Absolute Error) 손실 함수를 나타냅니다. 실제 값과 예측 값의 차이의 절댓값에 대해 모든 입력에 대해 평균을 취한 값으로, 예측이 정확할수록 값이 작아집니다.
* SEG : 세그멘테이션(segmentation) 문제에 사용되는 교차 엔트로피(Cross-Entropy) 손실 함수입니다. 입력 이미지의 각 픽셀이 클래스(class)에 속할 확률을 예측하고, 이 예측 값과 실제 클래스 값의 차이에 대해 손실을 계산합니다.
* SMOOTH : 세그멘테이션 문제에 사용되는 스무딩(smoothing)된 교차 엔트로피 손실 함수입니다. SEG와 유사하지만, 예측값과 실제값 간의 오차를 평활화(smoothing)하여 세그멘테이션 결과를 보다 부드럽게 만듭니다.
* WGAN : 생성적 적대 신경망(Generative Adversarial Network, GAN)에서 사용되는 손실 함수인 Wasserstein GAN 손실 함수입니다. GAN은 이미지 생성에 활용되는 딥러닝 모델로, WGAN 손실 함수는 생성된 이미지와 실제 이미지 간의 거리를 최소화하는 방식으로 모델을 학습합니다.



## get\_cost\_type

```c
COST_TYPE get_cost_type(char *s)
{
    if (strcmp(s, "seg")==0) return SEG;
    if (strcmp(s, "sse")==0) return SSE;
    if (strcmp(s, "masked")==0) return MASKED;
    if (strcmp(s, "smooth")==0) return SMOOTH;
    if (strcmp(s, "L1")==0) return L1;
    if (strcmp(s, "wgan")==0) return WGAN;
    fprintf(stderr, "Couldn't find cost type %s, going with SSE\n", s);
    return SSE;
}
```

함수 이름: get\_cost\_type

입력:&#x20;

* s: 문자열 포인터&#x20;

동작:&#x20;

* 입력된 문자열 s와 COST\_TYPE 열거형 상수를 비교하여 일치하는 COST\_TYPE을 반환합니다.&#x20;
* 입력된 문자열과 일치하는 COST\_TYPE이 없으면 "Couldn't find cost type %s, going with SSE" 오류 메시지를 출력하고 SSE를 반환합니다.

설명:

* get\_cost\_type 함수는 문자열 s를 입력으로 받아 이에 대응하는 COST\_TYPE을 반환합니다.
* 입력된 문자열 s를 SEG, SSE, MASKED, SMOOTH, L1, WGAN과 차례대로 비교하면서 일치하는 COST\_TYPE 상수를 반환합니다.
* 일치하는 문자열이 없을 경우 fprintf 함수를 사용하여 "Couldn't find cost type %s, going with SSE" 오류 메시지를 출력하고 SSE를 반환합니다.
* 함수가 반환하는 값은 COST\_TYPE 열거형 상수입니다.



## get\_cost\_string

```c
char *get_cost_string(COST_TYPE a)
{
    switch(a){
        case SEG:
            return "seg";
        case SSE:
            return "sse";
        case MASKED:
            return "masked";
        case SMOOTH:
            return "smooth";
        case L1:
            return "L1";
        case WGAN:
            return "wgan";
    }
    return "sse";
}
```

함수 이름: get\_cost\_string

입력:&#x20;

* a: COST\_TYPE 타입의 변수&#x20;

동작:&#x20;

* a에 해당하는 COST\_TYPE에 대한 문자열을 반환한다.

설명:

* 입력으로 받은 COST\_TYPE a에 해당하는 문자열을 반환하는 함수이다.
* switch 문을 사용하여 a가 각각의 COST\_TYPE에 해당하는 경우에 해당하는 문자열을 반환한다.
* 만약 a가 어떠한 COST\_TYPE에도 해당하지 않는 경우 "sse" 문자열을 반환한다.



## forward\_cost\_layer

```c
void forward_cost_layer(cost_layer l, network net)
{
    if (!net.truth) return;
    if(l.cost_type == MASKED){
        int i;
        for(i = 0; i < l.batch*l.inputs; ++i){
            if(net.truth[i] == SECRET_NUM) net.input[i] = SECRET_NUM;
        }
    }
    if(l.cost_type == SMOOTH){
        smooth_l1_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    }else if(l.cost_type == L1){
        l1_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    } else {
        l2_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    }
    l.cost[0] = sum_array(l.output, l.batch*l.inputs);
}
```

함수 이름: forward\_cost\_layer

입력:

* cost\_layer l: 비용 계산 레이어 구조체
* network net: 신경망 구조체

동작:

* net.truth이 존재하면 실행
  * l.cost\_type이 MASKED일 경우, SECRET\_NUM으로 표시된 값이 있는 위치는 net.input 값도 SECRET\_NUM으로 변경
  * l.cost\_type이 SMOOTH일 경우, smooth L1 함수를 이용하여 예측 값과 실제 값의 차이를 계산하여 l.delta와 l.output에 저장
  * l.cost\_type이 L1일 경우, L1 함수를 이용하여 예측 값과 실제 값의 차이를 계산하여 l.delta와 l.output에 저장
  * l.cost\_type이 그 외일 경우, L2 함수를 이용하여 예측 값과 실제 값의 차이를 계산하여 l.delta와 l.output에 저장
* l.output의 모든 원소의 합을 l.cost\[0]에 저장

설명:&#x20;

* 비용 계산 레이어는 신경망의 예측 결과와 실제 결과의 차이를 계산하여 비용을 구하는 역할을 합니다.&#x20;
* 이 함수는 주어진 비용 계산 레이어와 신경망을 이용하여 비용을 계산하고, 계산된 비용을 l.cost\[0]에 저장합니다.&#x20;
* 또한, l.cost\_type에 따라서 예측 값과 실제 값의 차이를 계산하는 함수를 호출하여 l.delta와 l.output에 저장합니다.&#x20;
* 이때, l.cost\_type이 MASKED일 경우, SECRET\_NUM으로 표시된 값이 있는 위치는 net.input 값도 SECRET\_NUM으로 변경하여 비용 계산에서 제외합니다.

.

## backward\_cost\_layer

```c
void backward_cost_layer(const cost_layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, l.scale, l.delta, 1, net.delta, 1);
}
```

함수 이름: backward\_cost\_layer

입력:&#x20;

* cost\_layer l
* network net

동작:&#x20;

* cost\_layer의 gradient를 계산하고, 이를 network의 delta값에 더해준다.&#x20;
* 이 때, l.scale은 gradient의 크기를 제어하기 위한 스케일링 인자이다.

설명:

* cost\_layer의 gradient는 delta 배열에 저장된다.
* axpy\_cpu 함수를 통해 net.delta 배열에 l.delta 배열을 l.scale만큼 스케일링하여 더해준다.
* 이 때, 두 배열의 크기는 l.batch\*l.inputs이다.
* 즉, cost\_layer를 통해 구한 gradient는 network의 다음 layer로 전달되며, 이후에 backward propagation이 이어져서 gradient가 역전파되게 된다.



## resize\_cost\_layer

```c
void resize_cost_layer(cost_layer *l, int inputs)
{
    l->inputs = inputs;
    l->outputs = inputs;
    l->delta = realloc(l->delta, inputs*l->batch*sizeof(float));
    l->output = realloc(l->output, inputs*l->batch*sizeof(float));
}
```

함수 이름: resize\_cost\_layer

입력:&#x20;

* l: cost\_layer 구조체 포인터
* inputs: int

동작:&#x20;

* cost\_layer 구조체 포인터 l의 inputs와 outputs 멤버 변수를 입력값으로 변경하고, delta와 output 배열의 크기를 realloc 함수를 사용하여 재할당한다.

설명:

* 함수는 cost\_layer 구조체를 받아서 해당 구조체의 멤버 변수를 조정하는 역할을 한다.
* l의 inputs와 outputs 멤버 변수를 입력값으로 변경한다.
* realloc 함수를 사용하여 l->delta 배열과 l->output 배열의 크기를 inputs_l->batch_sizeof(float)으로 재할당한다.
* 이 함수는 resize\_network 함수에서 호출되며, 신경망을 재조정할 때 cost\_layer도 함께 조정해야 하기 때문에 필요하다.



## make\_cost\_layer

```c
cost_layer make_cost_layer(int batch, int inputs, COST_TYPE cost_type, float scale)
{
    fprintf(stderr, "cost                                           %4d\n",  inputs);
    cost_layer l = {0};
    l.type = COST;

    l.scale = scale;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.cost_type = cost_type;
    l.delta = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.forward = forward_cost_layer;
    l.backward = backward_cost_layer;

    return l;
}
```

함수 이름: make\_cost\_layer

입력:

* int batch: batch size
* int inputs: layer의 input dimension
* COST\_TYPE cost\_type: cost function type
* float scale: cost의 크기 조절을 위한 스케일 값

동작:

* cost\_layer 구조체를 생성하고, 필드값들을 초기화한다.
* 입력받은 cost\_type에 따라서 l.cost\_type을 설정한다.
* l.delta, l.output, l.cost 배열을 초기화한다.
* forward와 backward 함수를 설정한다.

설명:

* 이 함수는 cost layer를 생성하는 함수로, 입력값들을 받아서 cost layer의 구조체를 생성하고 초기화하는 역할을 한다. 이 함수를 통해서 생성된 cost layer는 neural network에서 사용된다.

