# activations\_1

## Activation Function 이란?

Activation Function(활성화 함수)은 인공신경망에서 입력 신호를 처리한 후, 출력 신호를 만들어내는 함수입니다. 즉, 입력값에 대한 결과값을 결정하는 함수입니다.

활성화 함수는 비선형 함수(non-linear function)이어야 합니다. 이는 인공신경망이 복잡한 데이터를 처리하고, 다양한 패턴을 학습할 수 있도록 하기 위함입니다. 만약 활성화 함수가 선형 함수(linear function)이라면, 신경망이 깊어질수록 입력값과 출력값이 선형적인 관계를 갖게 되어, 효과적인 학습이 불가능해집니다.

대표적인 활성화 함수로는 시그모이드 함수, ReLU 함수, tanh 함수 등이 있습니다. 이러한 활성화 함수는 입력값에 대해 다양한 변화를 주어, 신경망이 복잡한 패턴을 학습할 수 있도록 합니다.



```c
// darknet.h

typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
} ACTIVATION;
```

활성화 함수의 종류를 정의하는 열거형(enum)입니다.

각각의 활성화 함수는 해당 함수의 특성에 따라 선택되어 사용됩니다. 이러한 활성화 함수들은 다양한 비선형성(non-linearity)을 가지고 있기 때문에, 인공신경망 모델이 복잡한 패턴을 학습할 수 있도록 돕습니다.

각 활성화 함수의 특징은 다음과 같습니다.

* LOGISTIC : 로지스틱 함수
* RELU : ReLU 함수
* RELIE : ReLU 함수의 변형
* LINEAR : 선형 함수
* RAMP : RAMP 함수
* TANH : 하이퍼볼릭 탄젠트 함수
* PLSE : 플립-선형 함수
* LEAKY : Leaky ReLU 함수
* ELU : Exponential Linear Units 함수
* LOGGY : 로지스틱 함수의 변형
* STAIR : 계단 함수
* HARDTAN : 하드 탄젠트 함수
* LHTAN : LeCun 하이퍼볼릭 탄젠트 함수
* SELU : Scaled Exponential Linear Units 함수

이러한 활성화 함수들은 모델의 입력값에 대해 적합한 비선형 변환을 수행하여, 모델이 입력 데이터의 패턴을 파악하고 예측을 수행할 수 있도록 합니다.



### get\_activation\_string

```c
char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case ELU:
            return "elu";
        case SELU:
            return "selu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        case STAIR:
            return "stair";
        case HARDTAN:
            return "hardtan";
        case LHTAN:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}
```

함수 이름: get\_activation\_string

입력:&#x20;

* a: 활성화 함수(enum 값)

동작:&#x20;

* 입력된 활성화 함수(enum 값)에 대응되는 문자열을 반환합니다.

설명:&#x20;

* 이 함수는 활성화 함수를 입력하면 해당 함수에 대응되는 문자열을 반환합니다.&#x20;
* 문자열은 해당 함수의 이름과 동일합니다.&#x20;
* 만약 입력된 함수에 대응하는 문자열이 없는 경우, 기본값으로 "relu"를 반환합니다.



### get\_activation

```c
ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC;
    if (strcmp(s, "loggy")==0) return LOGGY;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "selu")==0) return SELU;
    if (strcmp(s, "relie")==0) return RELIE;
    if (strcmp(s, "plse")==0) return PLSE;
    if (strcmp(s, "hardtan")==0) return HARDTAN;
    if (strcmp(s, "lhtan")==0) return LHTAN;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "stair")==0) return STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}
```

함수 이름: get\_activation

입력:&#x20;

* s: 활성화 함수를 나타내는 문자열

동작:&#x20;

* 입력된 문자열에 대응하는 활성화 함수(enum 값)을 반환합니다.

설명:&#x20;

* 이 함수는 문자열로 표현된 활성화 함수의 이름을 입력하면 해당 함수에 대응하는 enum 값(정수)을 반환합니다.&#x20;
* 함수 내부에서는 입력된 문자열을 활성화 함수 이름들과 비교하여 대응하는 enum 값을 반환하며, 입력된 문자열이 어떠한 활성화 함수와도 대응되지 않는 경우에는 기본값으로 RELU를 반환합니다.&#x20;
* 이 때, 함수는 stderr을 이용하여 에러 메시지를 출력합니다.



### activate

<pre class="language-c"><code class="lang-c"><strong>float activate(float x, ACTIVATION a)
</strong>{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case SELU:
            return selu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}
</code></pre>

함수 이름: activate

입력:&#x20;

* x: 활성화 함수에 대한 입력 값
* a: 적용할 활성화 함수

동작:&#x20;

* 입력된 활성화 함수(enum 값)에 따라 x 값을 활성화합니다.

설명:&#x20;

* 이 함수는 입력된 실수값 x와 활성화 함수(enum 값)을 입력받아, 해당 활성화 함수에 따라 x 값을 활성화합니다. 이 함수는 float 형 값을 반환합니다.&#x20;
* 활성화 함수는 선형(linear), 로지스틱(logistic), 로그(loggy), ReLU(relu), ELU(elu), SELU(selu), RELIE(relie), RAMP(ramp), LeakyReLU(leaky), 하이퍼볼릭 탄젠트(tanh), PLSE(plse), STAIR(stair), HardTanh(hardtan), LHTan(lhtan) 등이 가능합니다.&#x20;
* 해당 함수는 입력된 활성화 함수(enum 값)에 따라 적절한 활성화 함수를 호출하여 실수값 x를 처리하고 결과를 반환합니다.



### activate\_array

```c
void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = activate(x[i], a);
    }
}
```

함수 이름: activate\_array

입력:&#x20;

* x: 입력값 배열
* n: 배열 크기
* a: 활성화 함수

동작:&#x20;

* 입력값 배열 x에 활성화 함수 a를 적용하여 각 원소를 활성화합니다.

설명:&#x20;

* 이 함수는 입력값 배열 x의 각 원소에 활성화 함수 a를 적용합니다.&#x20;
* 입력값 배열 x와 배열 크기 n, 그리고 활성화 함수 a를 입력으로 받습니다.
* 배열 x의 각 원소에 대해 activate 함수를 호출하여 활성화된 값을 다시 배열 x의 해당 원소에 저장합니다.&#x20;
* 이 과정을 배열 x의 모든 원소에 대해 반복하면, 입력값 배열 x에 활성화 함수 a를 적용한 결과를 얻을 수 있습니다.

### gradient

```c
float gradient(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:
            return loggy_gradient(x);
        case RELU:
            return relu_gradient(x);
        case ELU:
            return elu_gradient(x);
        case SELU:
            return selu_gradient(x);
        case RELIE:
            return relie_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case LEAKY:
            return leaky_gradient(x);
        case TANH:
            return tanh_gradient(x);
        case PLSE:
            return plse_gradient(x);
        case STAIR:
            return stair_gradient(x);
        case HARDTAN:
            return hardtan_gradient(x);
        case LHTAN:
            return lhtan_gradient(x);
    }
    return 0;
}
```

함수 이름: gradient

입력:

* x: 활성화 함수에 대한 입력 값
* a: 적용할 활성화 함수

동작:

* 입력 값 x와 적용할 활성화 함수 a에 따라 해당 활성화 함수의 도함수를 계산하여 반환하는 함수

설명:

* 신경망에서 역전파(backpropagation) 알고리즘을 적용할 때, 오차(error)를 최소화하기 위해 가중치(weight)를 조절해야 하는데, 이를 위해 각 노드의 입력 값이 활성화 함수를 거쳐 출력 값으로 변환되는 과정에서 해당 활성화 함수의 도함수(gradient)를 구해야 한다.
* gradient 함수는 입력 값 x와 적용할 활성화 함수 a를 받아 해당 활성화 함수의 도함수를 계산하여 반환하는 함수로, switch문을 사용하여 입력으로 받은 활성화 함수 a에 따라 해당 도함수를 계산하여 반환한다.

### gradient\_array

```c
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);
    }
}
```

함수 이름: gradient\_array

입력:&#x20;

* x: 입력 배열 포인터&#x20;
* n: 입력 배열의 크기&#x20;
* a: 활성화 함수&#x20;
* delta: 출력 배열 포인터

동작:&#x20;

* 입력 배열 x의 각 요소에 대한 활성화 함수의 미분 값을 delta에 곱하여 출력 배열을 계산한다.&#x20;
* 이 함수는 역전파(backpropagation) 알고리즘에서 사용되며, 미분 값(delta)을 입력으로 받아, 이전 층에서 전달된 미분 값에 대한 현재 층의 미분 값을 계산하여 이전 층으로 전달하는 역할을 한다.

설명:&#x20;

* 역전파 알고리즘은 딥러닝 학습에서 사용되는 기법으로, 출력 값과 실제 값 사이의 오차를 최소화하는 방향으로 가중치와 편향을 업데이트한다.&#x20;
* 이 과정에서 gradient\_array 함수는 각 층에서 계산된 미분 값과 활성화 함수의 미분 값을 곱하여 이전 층으로 전달하며, 이전 층에서 전달된 미분 값을 현재 층에서 곱하여 출력 배열의 미분 값을 계산한다. 이렇게 계산된 미분 값은 가중치와 편향을 업데이트할 때 사용된다.
