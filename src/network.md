# network

## get\_base\_args

```c
load_args get_base_args(network *net)
{
    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.size = net->w;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.center = net->center;
    args.saturation = net->saturation;
    args.hue = net->hue;
    return args;
}
```

함수 이름: get\_base\_args&#x20;

입력:&#x20;

* network \*net (신경망 구조체 포인터)&#x20;

동작:&#x20;

* 입력으로 받은 신경망 구조체 포인터의 정보를 이용하여 load\_args 구조체를 초기화하고 반환함.&#x20;

설명:&#x20;

* load\_args 구조체는 데이터 로딩 시 필요한 다양한 인자들을 담고 있는 구조체이다.&#x20;
* 이 함수는 입력으로 받은 신경망 구조체 포인터에서 데이터 로딩 시 필요한 인자들을 추출하여 load\_args 구조체를 초기화한 후 반환한다.&#x20;
* 초기화하는 인자들로는 데이터 크기, 이미지 전처리에 사용되는 값들 (crop, angle, aspect, exposure, center, saturation, hue) 등이 있다.



## load\_network

```c
network *load_network(char *cfg, char *weights, int clear)
{
    network *net = parse_network_cfg(cfg);
    if(weights && weights[0] != 0){
        load_weights(net, weights);
    }
    if(clear) (*net->seen) = 0;
    return net;
}
```

함수 이름: load\_network&#x20;

입력:

* char \*cfg: 네트워크 구성 파일의 경로를 나타내는 문자열 포인터
* char \*weights: 학습된 가중치 파일의 경로를 나타내는 문자열 포인터
* int clear: 네트워크의 seen 값을 0으로 설정할 지 여부를 나타내는 정수

동작:

* parse\_network\_cfg 함수를 사용하여 cfg 파일에서 네트워크 구성을 읽어와 네트워크를 초기화
* weights가 주어지면 load\_weights 함수를 사용하여 가중치를 로드
* clear가 1이면 네트워크의 seen 값을 0으로 설정
* 초기화된 네트워크 포인터를 반환

설명:

* 이 함수는 구성 파일과 학습된 가중치 파일로부터 네트워크를 로드하는 함수이다.
* cfg와 weights 인자는 파일 경로를 나타내는 문자열 포인터이다.
* clear 인자는 네트워크의 seen 값을 초기화할 지 여부를 결정한다.
* 이 함수는 먼저 parse\_network\_cfg 함수를 사용하여 cfg 파일에서 네트워크 구성을 읽어와 네트워크를 초기화한다.
* weights가 주어졌다면 load\_weights 함수를 사용하여 가중치를 로드한다.
* clear가 1이면 네트워크의 seen 값을 0으로 설정한다.
* 초기화된 네트워크 포인터를 반환한다.



## get\_current\_batch

```c
size_t get_current_batch(network *net)
{
    size_t batch_num = (*net->seen)/(net->batch*net->subdivisions);
    return batch_num;
}
```

함수 이름: get\_current\_batch&#x20;

입력:&#x20;

* network 구조체 포인터 변수 net&#x20;

동작:&#x20;

* 현재까지 처리된 데이터 샘플 수와 배치 크기, 서브디비전 값으로부터 현재 배치의 번호를 계산하여 반환한다.&#x20;

설명:&#x20;

* get\_current\_batch 함수는 현재까지 처리된 데이터 샘플 수와 배치 크기, 서브디비전 값을 이용하여 현재 배치의 번호를 계산하고, 그 값을 size\_t 타입으로 반환한다.&#x20;
* 이 함수는 학습 중인 신경망에서 현재 몇 번째 배치인지 확인할 때 사용된다.



## get\_current\_rate

```c
float get_current_rate(network *net)
{
    size_t batch_num = get_current_batch(net);
    int i;
    float rate;
    if (batch_num < net->burn_in) return net->learning_rate * pow((float)batch_num / net->burn_in, net->power);
    switch (net->policy) {
        case CONSTANT:
            return net->learning_rate;
        case STEP:
            return net->learning_rate * pow(net->scale, batch_num/net->step);
        case STEPS:
            rate = net->learning_rate;
            for(i = 0; i < net->num_steps; ++i){
                if(net->steps[i] > batch_num) return rate;
                rate *= net->scales[i];
            }
            return rate;
        case EXP:
            return net->learning_rate * pow(net->gamma, batch_num);
        case POLY:
            return net->learning_rate * pow(1 - (float)batch_num / net->max_batches, net->power);
        case RANDOM:
            return net->learning_rate * pow(rand_uniform(0,1), net->power);
        case SIG:
            return net->learning_rate * (1./(1.+exp(net->gamma*(batch_num - net->step))));
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net->learning_rate;
    }
}
```

함수 이름: get\_current\_rate

입력:&#x20;

* network 구조체 포인터(net)

동작:&#x20;

* 현재 배치(batch\_num)와 네트워크의 학습률(learning\_rate) 및 학습 정책(policy)에 따라 현재 학습 속도(rate)를 계산한다.

설명:&#x20;

* 현재 배치(batch\_num)는 (\*net->seen) / (net->batch \* net->subdivisions)로 계산된다.&#x20;
* 이때 (\*net->seen)은 네트워크가 지금까지 본 이미지 수를 나타내고, net->batch와 net->subdivisions는 네트워크가 한 번의 역전파 단계에서 처리할 이미지의 개수와 서브 배치(sub-batch)의 수를 의미한다.&#x20;
* 학습 속도는 네트워크의 학습 정책(policy)에 따라 계산된다.&#x20;
* 여러 가지 학습 정책을 지원한다. CONSTANT(고정), STEP(단계), STEPS(단계 여러 개), EXP(지수), POLY(다항), RANDOM(랜덤), SIG(시그모이드) 등이 있다.&#x20;
* 각 학습 정책에 대한 자세한 설명은 darknet 공식 문서를 참조하면 된다.
* 반환값: 현재 학습 속도(rate)



## get\_layer\_string

```c
char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return "convolutional";
        case ACTIVE:
            return "activation";
        case LOCAL:
            return "local";
        case DECONVOLUTIONAL:
            return "deconvolutional";
        case CONNECTED:
            return "connected";
        case RNN:
            return "rnn";
        case GRU:
            return "gru";
        case LSTM:
	    return "lstm";
        case CRNN:
            return "crnn";
        case MAXPOOL:
            return "maxpool";
        case REORG:
            return "reorg";
        case AVGPOOL:
            return "avgpool";
        case SOFTMAX:
            return "softmax";
        case DETECTION:
            return "detection";
        case REGION:
            return "region";
        case YOLO:
            return "yolo";
        case DROPOUT:
            return "dropout";
        case CROP:
            return "crop";
        case COST:
            return "cost";
        case ROUTE:
            return "route";
        case SHORTCUT:
            return "shortcut";
        case NORMALIZATION:
            return "normalization";
        case BATCHNORM:
            return "batchnorm";
        default:
            break;
    }
    return "none";
}
```

함수 이름: get\_layer\_string

입력:&#x20;

* LAYER\_TYPE a (열거형)

동작:&#x20;

* 입력으로 받은 레이어 유형에 해당하는 문자열을 반환한다.&#x20;
* switch 문을 사용하여 입력된 유형에 따라 문자열을 반환한다.&#x20;
* 만약에 해당하는 유형이 없을 경우 "none" 문자열을 반환한다.

설명:&#x20;

* 이 함수는 Darknet 프레임워크에서 사용되는 레이어 유형을 문자열로 변환해주는 함수이다.&#x20;
* Darknet에서 사용되는 모든 레이어 유형을 열거형으로 정의하고 있으며, 이 함수는 해당 열거형 변수를 입력받아 문자열을 반환한다.&#x20;
* Darknet의 코드에서 자주 사용되는 함수 중 하나이다.



## make\_network

```c
network *make_network(int n)
{
    network *net = calloc(1, sizeof(network));
    net->n = n;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
    return net;
}
```

함수 이름: make\_network

입력:&#x20;

* n (int): 네트워크가 가질 레이어 수

동작:&#x20;

* 네트워크 구조체를 할당하고 초기화한다. layers, seen, t, cost 등의 구조체 변수도 초기화한다.

설명:&#x20;

* 인자로 받은 n을 레이어 수로 가지는 네트워크 구조체를 생성하고, 필요한 구조체 변수들을 초기화하는 함수이다.&#x20;
* 함수가 호출되면, calloc()을 사용하여 메모리를 할당하고, 구조체 변수들을 0으로 초기화한 후, 구조체 포인터를 반환한다.



## forward\_network

```c
void forward_network(network *netp)
{
    network net = *netp;
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta){
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        l.forward(l, net);
        net.input = l.output;
        if(l.truth) {
            net.truth = l.output;
        }
    }
    calc_network_cost(netp);
}
```

함수 이름: forward\_network

입력:&#x20;

* network \*netp (포인터 형태의 network 구조체)

동작:&#x20;

* 전방향 계산을 수행하고 네트워크 비용을 계산한다.&#x20;
* 함수 내부에서는 전달된 netp를 복사하여 net 구조체를 만든 후, 네트워크 내의 모든 레이어를 순차적으로 실행하면서 결과값을 계산한다.&#x20;
* 또한 현재 레이어의 출력값을 다음 레이어의 입력값으로 사용한다.

설명:&#x20;

* 네트워크의 전방향 계산(forward propagation)을 수행하는 함수이다. 입력 데이터가 네트워크를 통과하면서 각 레이어에서 연산을 수행하고 최종 출력값을 계산한다.&#x20;
* 이때 입력 데이터는 처음 레이어에 입력되고, 각 레이어는 이전 레이어의 출력값을 입력으로 받는다.&#x20;
* 함수가 실행되면, 전달된 netp 구조체를 복사하여 net 구조체를 만든다.&#x20;
* 그 후, 모든 레이어를 순차적으로 실행하면서 출력값을 계산하고, 현재 레이어의 출력값을 다음 레이어의 입력값으로 사용한다.&#x20;
* 이때, 레이어에서는 forward 함수를 호출하여 출력값을 계산하고, 이 출력값은 다시 net 구조체의 input 필드에 저장된다.&#x20;
* 마지막으로, calc\_network\_cost 함수를 호출하여 네트워크 비용을 계산한다.



## update\_network

```c
void update_network(network *netp)
{
    network net = *netp;
    int i;
    update_args a = {0};
    a.batch = net.batch*net.subdivisions;
    a.learning_rate = get_current_rate(netp);
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.adam = net.adam;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = *net.t;

    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update){
            l.update(l, a);
        }
    }
}
```

함수 이름: update\_network

입력:&#x20;

* network 포인터(netp)

동작:&#x20;

* 네트워크의 레이어들을 업데이트합니다.&#x20;
* update\_args 구조체를 생성하여 각 레이어의 update 함수를 호출하고, 이전 레이어의 출력을 다음 레이어의 입력으로 전달합니다.

설명:

* net: network 구조체 변수로, 입력으로 받은 netp를 역참조하여 생성합니다.
* i: 반복문에서 사용할 인덱스 변수입니다.
* a: update\_args 구조체로, 업데이트 함수에 필요한 인자들을 담고 있습니다.
* a.batch: 한 번에 처리할 이미지의 수(batch size)입니다.
* a.learning\_rate: 현재 학습률(learning rate)을 가져옵니다.
* a.momentum: 모멘텀(momentum) 값을 가져옵니다.
* a.decay: 가중치 감소(weight decay) 값을 가져옵니다.
* a.adam: Adam 최적화(Optimization) 알고리즘을 사용할 지 여부를 나타냅니다.
* a.B1: Adam 알고리즘의 첫 번째 모멘트 계수입니다.
* a.B2: Adam 알고리즘의 두 번째 모멘트 계수입니다.
* a.eps: Adam 알고리즘의 엡실론(epsilon) 값입니다.
* \*net.t: 현재까지 수행한 반복 횟수입니다. 이 값을 1 증가시키고, a.t에 저장합니다.
* l: 현재 레이어를 나타내는 layer 구조체 변수입니다.
* l.update: 현재 레이어의 update 함수 포인터입니다. 이 값이 NULL이 아니면 l.update 함수를 호출하여 레이어를 업데이트합니다.



## calc\_network\_cost

```c
void calc_network_cost(network *netp)
{
    network net = *netp;
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    *net.cost = sum/count;
}
```

함수 이름: calc\_network\_cost

입력:&#x20;

* network 구조체 포인터(netp)

동작:&#x20;

* 네트워크 내의 모든 레이어의 비용(cost) 값을 합산하여 평균값을 계산하고, 그 결과를 net 구조체의 cost 변수에 저장한다.

설명:

* 입력으로 받은 네트워크 구조체 포인터(netp)로부터 네트워크 구조체(net)를 생성한다.
* for 루프를 사용하여 모든 레이어에 대해 비용(cost) 값을 확인한다.
* 비용이 존재하는 레이어의 cost 값을 합산(sum)하고, 개수(count)를 카운트한다.
* 모든 레이어의 비용 값의 평균값을 계산하여, net 구조체의 cost 변수에 저장한다.



## get\_predicted\_class\_network

```c
int get_predicted_class_network(network *net)
{
    return max_index(net->output, net->outputs);
}
```

함수 이름: get\_predicted\_class\_network&#x20;

입력:&#x20;

* network \*net (신경망 포인터)&#x20;

동작:&#x20;

* 신경망의 출력 벡터에서 예측된 클래스의 인덱스를 반환합니다.&#x20;
* max\_index() 함수를 사용하여 가장 큰 값의 인덱스를 찾습니다.&#x20;

설명:&#x20;

* 이 함수는 분류 작업에서 신경망의 출력값에서 가장 큰 값을 가진 클래스의 인덱스를 반환하는데 사용됩니다.&#x20;
* 출력 벡터는 확률 분포와 유사하며, 가장 큰 값은 신경망이 예측한 클래스의 확률입니다.&#x20;
* max\_index() 함수는 배열에서 가장 큰 값을 가진 원소의 인덱스를 찾아 반환합니다.



## backward\_network

```c
void backward_network(network *netp)
{
    network net = *netp;
    int i;
    network orig = net;
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
        }
        net.index = i;
        l.backward(l, net);
    }
}
```

함수 이름: backward\_network

입력:&#x20;

* network 구조체 포인터 (network \*netp)

동작:&#x20;

* 네트워크의 각 레이어의 backward 함수를 호출하여 역전파(backpropagation)를 수행합니다.&#x20;
* 출력값(output)에 대한 오차를 계산하고 각 레이어의 delta값을 업데이트합니다.

설명:&#x20;

* 입력으로 받은 포인터로부터 network 구조체를 가져온 뒤, 각 레이어의 backward 함수를 역순으로 실행합니다.&#x20;
* 레이어를 거꾸로 실행하는 이유는, 각 레이어에서 계산된 delta값을 이전 레이어로 전달해주어야 하기 때문입니다.&#x20;
* i가 0일 때는 첫 번째 레이어이므로, 입력값을 다시 원래대로 되돌려 놓습니다.&#x20;
* l.stopbackward가 true이면 해당 레이어의 backward 함수를 실행하지 않고, 이전 레이어로 넘어갑니다.
* 출력: 없음 (void)



## train\_network\_datum

```c
float train_network_datum(network *net)
{
    *net->seen += net->batch;
    net->train = 1;
    forward_network(net);
    backward_network(net);
    float error = *net->cost;
    if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
    return error;
}
```

함수 이름: train\_network\_datum

입력:&#x20;

* (network\*) net: 학습할 신경망

동작:&#x20;

* 입력 데이터에 대해 학습을 수행하고, 에러를 반환한다.

설명:

* \*net->seen 값에 net->batch 값을 더한다.
* net->train 값을 1로 설정하여 신경망이 학습 중임을 나타낸다.
* forward\_network 함수를 호출하여 순전파를 수행한다.
* backward\_network 함수를 호출하여 역전파를 수행한다.
* \*net->cost 값을 error 변수에 대입한다.
* (\*net->seen)/net->batch 값이 net->subdivisions로 나누어 떨어지면, update\_network 함수를 호출하여 가중치를 업데이트 한다.
* 에러(error)를 반환한다.



## train\_network\_sgd

```c
float train_network_sgd(network *net, data d, int n)
{
    int batch = net->batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_random_batch(d, batch, net->input, net->truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}
```

함수 이름: train\_network\_sgd

입력:

* network \*net: 신경망을 가리키는 포인터
* data d: 학습 데이터셋
* int n: 학습 데이터셋에서 랜덤하게 선택할 데이터 수

동작:&#x20;

* 주어진 학습 데이터셋에서 n개의 데이터를 랜덤하게 선택해 신경망을 학습시킵니다.&#x20;
* 각 데이터는 train\_network\_datum 함수를 사용해 학습하며, 에러를 누적하여 평균 에러를 계산하고 반환합니다.

설명:

* batch: 한 번에 처리될 데이터의 수입니다.
* err: train\_network\_datum 함수에서 계산된 에러입니다.
* sum: n개의 데이터에서 계산된 총 에러입니다.



## train\_network

```c
float train_network(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}
```

함수 이름: train\_network&#x20;

입력:&#x20;

* network \*net (신경망 모델 포인터)
* data d (학습용 데이터셋)&#x20;

동작:&#x20;

* 주어진 데이터셋을 이용하여 신경망 모델을 학습시킴.&#x20;
* 배치 크기를 고려하여 데이터셋을 미니배치로 나누어 각각을 학습 데이터로 사용하고, 각 미니배치마다 train\_network\_datum 함수를 호출하여 모델을 학습시킴.&#x20;
* 학습 데이터 전체에 대한 오차의 평균값을 반환함.&#x20;

설명:

* assert 문을 사용하여 데이터셋의 크기가 배치 크기의 배수인지 확인함.
* 입력으로 주어진 데이터셋 d 를 배치 크기로 나누어 미니배치의 개수 n 을 구함.
* n 만큼 반복하며, i 번째 미니배치를 가져와서 get\_next\_batch 함수를 호출하여 net->input, net->truth 배열에 입력 데이터와 정답 데이터를 설정함.
* train\_network\_datum 함수를 호출하여 모델을 학습시킴. 이 때, 각 미니배치마다의 오차를 sum 변수에 더해줌.
* 전체 학습 데이터에 대한 오차의 평균값을 계산하여 반환함.



## set\_temp\_network

```c
void set_temp_network(network *net, float t)
{
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].temperature = t;
    }
}
```

함수 이름: set\_temp\_network

입력:&#x20;

* network 구조체 포인터(net)
* 부동 소수점 값(t)

동작:&#x20;

* 네트워크의 모든 레이어의 온도를 주어진 값으로 설정한다.

설명:&#x20;

* 입력으로 받은 네트워크(net)의 각 레이어의 온도(temperature)를 주어진 값(t)으로 설정한다.&#x20;
* 이 함수는 각 레이어의 온도를 제어하여, 훈련 중 네트워크가 수렴하기 전에 지역 최적점에 빠지지 않도록 한다.



## set\_batch\_network

```c
void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
    }
}
```

함수 이름: set\_batch\_network

입력:

* net (network\* 타입): 배치 크기를 설정할 신경망 구조체 포인터
* b (int 타입): 설정할 배치 크기 값

동작:

* 입력으로 받은 배치 크기 값을 신경망 구조체의 배치 크기에 설정하고, 모든 레이어의 배치 크기도 입력값으로 설정한 값으로 변경한다.

설명:

* 해당 함수는 신경망의 배치 크기를 설정하는 함수이다.
* 입력으로 받은 신경망 구조체 포인터의 배치 크기를 입력으로 받은 배치 크기 값으로 설정하고, 모든 레이어의 배치 크기도 같은 값으로 변경한다.
* 배치 크기란 한 번에 처리할 데이터의 개수를 의미한다.
* 배치 크기를 설정하는 것은 신경망 학습 시 한 번에 처리할 데이터의 개수를 결정하는 중요한 요소 중 하나이다.
* 배치 크기를 적절히 설정하면 학습 속도와 정확도를 개선할 수 있다.



## resize\_network

```c
int resize_network(network *net, int w, int h)
{
    int i;
    //if(w == net->w && h == net->h) return 0;
    net->w = w;
    net->h = h;
    int inputs = 0;
    size_t workspace_size = 0;
    //fprintf(stderr, "Resizing to %d x %d...\n", w, h);
    //fflush(stderr);
    for (i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            resize_convolutional_layer(&l, w, h);
        }else if(l.type == CROP){
            resize_crop_layer(&l, w, h);
        }else if(l.type == MAXPOOL){
            resize_maxpool_layer(&l, w, h);
        }else if(l.type == REGION){
            resize_region_layer(&l, w, h);
        }else if(l.type == YOLO){
            resize_yolo_layer(&l, w, h);
        }else if(l.type == ROUTE){
            resize_route_layer(&l, net);
        }else if(l.type == SHORTCUT){
            resize_shortcut_layer(&l, w, h);
        }else if(l.type == UPSAMPLE){
            resize_upsample_layer(&l, w, h);
        }else if(l.type == REORG){
            resize_reorg_layer(&l, w, h);
        }else if(l.type == AVGPOOL){
            resize_avgpool_layer(&l, w, h);
        }else if(l.type == NORMALIZATION){
            resize_normalization_layer(&l, w, h);
        }else if(l.type == COST){
            resize_cost_layer(&l, inputs);
        }else{
            error("Cannot resize this type of layer");
        }
        if(l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        if(l.workspace_size > 2000000000) assert(0);
        inputs = l.outputs;
        net->layers[i] = l;
        w = l.out_w;
        h = l.out_h;
        if(l.type == AVGPOOL) break;
    }
    layer out = get_network_output_layer(net);
    net->inputs = net->layers[0].inputs;
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    free(net->input);
    free(net->truth);
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    net->truth = calloc(net->truths*net->batch, sizeof(float));

    free(net->workspace);
    net->workspace = calloc(1, workspace_size);

    //fprintf(stderr, " Done!\n");
    return 0;
}
```

함수 이름: resize\_network

입력:&#x20;

* network \*net (신경망 모델 포인터)
* int w (변경할 너비)
* int h (변경할 높이)

동작:&#x20;

* 입력된 너비와 높이를 기반으로 네트워크 모델을 재조정하고, 새로운 크기에 맞게 각 레이어를 재설정합니다.&#x20;
* 이 함수는 너비와 높이를 조정하는 것 외에도, 새로운 입력과 출력 크기를 계산하고, 입력 및 출력 메모리를 할당하며, 작업 공간의 크기를 조정합니다.

설명:

* network \*net: 딥러닝 모델을 나타내는 포인터
* int w: 변경할 입력 이미지의 너비
* int h: 변경할 입력 이미지의 높이
* int i: 반복문에서 사용되는 인덱스 변수
* int inputs: 입력의 크기를 저장하는 변수
* size\_t workspace\_size: 작업 공간의 크기를 저장하는 변수
* layer l: 현재 처리 중인 레이어
* layer out: 네트워크 출력 레이어
* float \*input: 입력 데이터를 저장하는 배열
* float \*truth: 실제 출력 데이터를 저장하는 배열
* float \*output: 네트워크의 출력 데이터를 저장하는 배열
* 이 함수는 입력된 너비와 높이를 기반으로 네트워크 모델을 재조정합니다.&#x20;
* 그러면서, 모든 레이어에 대해 새로운 크기에 맞게 각 레이어를 재설정합니다.&#x20;
* 이 때, 반복문을 사용하여 모든 레이어를 처리하며, 레이어 유형에 따라 해당하는 resize 함수를 호출합니다.&#x20;
* 각 레이어의 작업 공간 크기를 계산하고, 최대 크기를 workspace\_size 변수에 저장합니다.&#x20;
* 이후, 새로운 입력 및 출력 크기를 계산하고, 입력 및 출력 배열을 할당합니다.&#x20;
* 마지막으로, 작업 공간의 크기를 조정하고, 작업 공간을 할당합니다.



## get\_network\_detection\_layer

```c
layer get_network_detection_layer(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        if(net->layers[i].type == DETECTION){
            return net->layers[i];
        }
    }
    fprintf(stderr, "Detection layer not found!!\n");
    layer l = {0};
    return l;
}
```

함수 이름: get\_network\_detection\_layer

입력:&#x20;

* network \*net (신경망 구조체)

동작:&#x20;

* 주어진 신경망에서 탐지(Detection) 레이어를 찾아 해당 레이어를 반환한다.&#x20;
* 탐지 레이어가 없는 경우 오류 메시지를 출력하고 빈 layer 구조체를 반환한다.

설명:&#x20;

* 이 함수는 YOLO(Object Detection) 알고리즘에서 사용된다.&#x20;
* YOLO 신경망에서는 출력값을 생성하는 마지막 레이어가 탐지 레이어이다.&#x20;
* 이 함수는 신경망의 레이어들을 반복하여 DETECTION 타입의 레이어를 찾는다.&#x20;
* 탐지 레이어를 찾으면 해당 레이어를 반환하고, 찾지 못하면 오류 메시지를 출력하고 빈 layer 구조체를 반환한다.



## get\_network\_image\_layer

```c
image get_network_image_layer(network *net, int i)
{
    layer l = net->layers[i];

    if (l.out_w && l.out_h && l.out_c){
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}
```

함수 이름: get\_network\_image\_layer&#x20;

입력:&#x20;

* network 구조체 포인터
* int 형 변수 i&#x20;

동작:&#x20;

* 주어진 network에서 i번째 레이어의 출력 이미지를 가져온다. 만약 해당 레이어의 출력 이미지가 없으면 비어있는 이미지를 반환한다.&#x20;

설명:

* 주어진 network에서 i번째 레이어를 가져온다.
* 해당 레이어의 출력 이미지가 존재하면 해당 이미지를 float\_to\_image 함수를 사용하여 생성한다.
* 해당 레이어의 출력 이미지가 없으면 (out\_w, out\_h, out\_c가 0) 비어있는 이미지를 반환한다.



## get\_network\_image

```c
image get_network_image(network *net)
{
    int i;
    for(i = net->n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    image def = {0};
    return def;
}
```

함수 이름: get\_network\_image&#x20;

입력:&#x20;

* network \*net (네트워크 모델)&#x20;

동작:&#x20;

* 네트워크 모델의 출력 레이어들 중에서 마지막 출력 레이어부터 시작하여 출력 이미지를 가져옴&#x20;

설명:&#x20;

* YOLO 알고리즘에서 출력 이미지를 가져오기 위해 사용되는 함수로, 네트워크 모델의 출력 레이어들 중에서 마지막 출력 레이어부터 시작하여 출력 이미지를 가져오는 역할을 수행한다.&#x20;
* 가져온 이미지는 image 구조체로 반환되며, 만약 출력 이미지가 존재하지 않는다면 크기가 0인 image 구조체가 반환된다.



## visualize\_network

```c
void visualize_network(network *net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net->n; ++i){
        sprintf(buff, "Layer %d", i);
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    }
}
```

함수 이름: visualize\_network

입력:&#x20;

* network 구조체 포인터(net)

동작:&#x20;

* 네트워크의 각 레이어를 시각화하여 출력합니다.&#x20;
* CONVOLUTIONAL 레이어인 경우 visualize\_convolutional\_layer 함수를 호출하여 시각화합니다.

설명:&#x20;

* 입력으로 네트워크 구조체 포인터를 받아 각 레이어를 시각화합니다.&#x20;
* 시각화한 결과는 이전 레이어의 출력 이미지를 이어서 출력됩니다.&#x20;
* 또한, CONVOLUTIONAL 레이어인 경우 visualize\_convolutional\_layer 함수를 호출하여 해당 레이어를 시각화합니다.&#x20;
* 각 레이어의 이름은 "Layer i"로 지정되며, i는 해당 레이어의 인덱스입니다.

##

## top\_predictions

```c
void top_predictions(network *net, int k, int *index)
{
    top_k(net->output, net->outputs, k, index);
}
```

함수 이름: top\_predictions&#x20;

입력:&#x20;

* network \*net (신경망 구조체)
* int k (상위 예측값의 개수)
* int \*index (상위 예측값들의 인덱스를 저장할 배열 포인터)&#x20;

동작:&#x20;

* 입력으로 들어온 신경망의 출력값을 기반으로, 상위 k개의 예측값과 해당 예측값들의 인덱스를 계산하여 index 배열에 저장한다.&#x20;

설명:&#x20;

* top\_k() 함수를 호출하여, 신경망의 출력값에서 상위 k개의 값을 찾고, 해당 값들의 인덱스를 index 배열에 저장한다.



## network\_predict

```c
float *network_predict(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    forward_network(net);
    float *out = net->output;
    *net = orig;
    return out;
}
```

함수 이름: network\_predict&#x20;

입력:

* network \*net: 신경망을 가리키는 포인터
* float \*input: 입력 데이터를 가리키는 포인터

동작:

* orig 변수에 net을 복사
* net->input을 입력 데이터로 설정
* net->truth, net->train, net->delta를 0으로 설정
* forward\_network 함수를 호출하여 순전파 수행
* net->output을 반환
* \*net을 orig로 되돌림

설명:&#x20;

* 주어진 신경망 net과 입력 데이터 input을 이용하여 예측을 수행하는 함수입니다.&#x20;
* 입력 데이터는 net->input에 설정되며, forward\_network 함수를 호출하여 신경망을 순전파시킵니다.&#x20;
* 최종 출력값인 net->output을 반환하고, 신경망 net은 원래 상태로 되돌립니다.



## num\_detections

```c
int num_detections(network *net, float thresh)
{
    int i;
    int s = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO){
            s += yolo_num_detections(l, thresh);
        }
        if(l.type == DETECTION || l.type == REGION){
            s += l.w*l.h*l.n;
        }
    }
    return s;
}
```

함수 이름: num\_detections&#x20;

입력:&#x20;

* network \*net (YOLO 또는 Detection 레이어가 포함된 네트워크)
* float thresh (임계값)&#x20;

동작:&#x20;

* 입력된 네트워크에서 YOLO 및 Detection 레이어에서 임계값을 넘는 검출 수를 셉니다.&#x20;

설명:&#x20;

* 입력된 네트워크의 모든 레이어를 확인하면서 YOLO 레이어에서 yolo\_num\_detections 함수를 호출하여 임계값을 넘는 검출 수를 측정하고, DETECTION 또는 REGION 레이어의 너비, 높이 및 채널 정보를 기반으로 총 검출 수를 계산합니다.



## make\_network\_boxes

```c
detection *make_network_boxes(network *net, float thresh, int *num)
{
    layer l = net->layers[net->n - 1];
    int i;
    int nboxes = num_detections(net, thresh);
    if(num) *num = nboxes;
    detection *dets = calloc(nboxes, sizeof(detection));
    for(i = 0; i < nboxes; ++i){
        dets[i].prob = calloc(l.classes, sizeof(float));
        if(l.coords > 4){
            dets[i].mask = calloc(l.coords-4, sizeof(float));
        }
    }
    return dets;
}
```

함수 이름: make\_network\_boxes&#x20;

입력:

* network \*net : YOLO 네트워크
* float thresh : 검출 임계값
* int \*num : 검출된 바운딩 박스 개수를 저장할 포인터 변수

동작:

* YOLO 네트워크의 출력에서 검출된 바운딩 박스 개수를 계산하여 num 변수에 저장하고, 검출된 바운딩 박스를 저장할 detection 구조체 배열을 생성하고 반환함
* 생성된 detection 구조체 배열의 각 원소마다 클래스별 확률(prob)과 마스크(mask)를 위한 메모리를 할당함

설명:

* YOLO 네트워크의 출력에서 검출된 바운딩 박스 개수를 계산하기 위해 num\_detections 함수를 호출함
* 검출된 바운딩 박스 개수에 해당하는 크기의 detection 구조체 배열을 동적으로 할당함
* 할당된 detection 구조체 배열의 각 원소마다 클래스별 확률(prob)을 저장할 float 배열을 동적으로 할당함
* 만약 YOLO 레이어의 coords 값이 4보다 크다면, 각 원소마다 마스크(mask)를 저장할 float 배열도 동적으로 할당함
* 생성된 detection 구조체 배열을 반환함



## fill\_network\_boxes

```c
void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
    int j;
    for(j = 0; j < net->n; ++j){
        layer l = net->layers[j];
        if(l.type == YOLO){
            int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets);
            dets += count;
        }
        if(l.type == REGION){
            get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
        if(l.type == DETECTION){
            get_detection_detections(l, w, h, thresh, dets);
            dets += l.w*l.h*l.n;
        }
    }
}
```

함수 이름: fill\_network\_boxes

입력:

* network \*net: YOLO, Region, Detection 레이어를 가지고 있는 네트워크
* int w: 입력 이미지의 너비
* int h: 입력 이미지의 높이
* float thresh: 박스 확률 임계값
* float hier: YOLO 레이어에서 사용하는 hierachical softmax 임계값
* int \*map: 이미지의 너비와 높이에 따라 셀의 인덱스를 계산하기 위한 인덱스 맵
* int relative: YOLO 레이어의 경우 박스 좌표를 상대적인 좌표로 계산할 지 여부
* detection \*dets: 네트워크에서 예측한 박스 정보를 저장하는 detection 구조체 배열

동작:&#x20;

* 네트워크에서 예측한 박스 정보를 detection 구조체 배열에 저장하는 함수로, YOLO, Region, Detection 레이어를 차례로 순회하면서 각 레이어에서 예측한 박스 정보를 detection 구조체 배열에 저장합니다.&#x20;
* YOLO 레이어는 get\_yolo\_detections 함수를 사용하여 예측한 박스 정보를, Region 레이어는 get\_region\_detections 함수를, Detection 레이어는 get\_detection\_detections 함수를 사용하여 예측한 박스 정보를 detection 구조체 배열에 저장합니다.

설명:

* get\_yolo\_detections: YOLO 레이어에서 예측한 박스 정보를 detection 구조체 배열에 저장하는 함수
* get\_region\_detections: Region 레이어에서 예측한 박스 정보를 detection 구조체 배열에 저장하는 함수
* get\_detection\_detections: Detection 레이어에서 예측한 박스 정보를 detection 구조체 배열에 저장하는 함수



## get\_network\_boxes

```c
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
    detection *dets = make_network_boxes(net, thresh, num);
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets);
    return dets;
}
```

함수 이름: get\_network\_boxes

입력:

* network \*net: YOLO 신경망
* int w: 입력 이미지의 너비
* int h: 입력 이미지의 높이
* float thresh: 객체 탐지 임계값
* float hier: YOLO 계층 간 히어러키(threshold)
* int \*map: 클래스 매핑
* int relative: 박스 좌표를 상대적인 값으로 가져올 지 여부
* int \*num: 탐지된 객체의 개수를 저장할 포인터

동작:

* 객체 탐지 후 해당 객체에 대한 detection 구조체 배열을 반환한다.
* make\_network\_boxes 함수를 통해 detection 구조체 배열을 생성하고, fill\_network\_boxes 함수를 통해 객체 탐지를 수행한다.

설명:

* YOLO 신경망에서 객체 탐지를 수행하고, detection 구조체 배열을 반환하는 함수이다.
* 입력 이미지의 크기, 탐지 임계값 등을 인자로 받는다.
* make\_network\_boxes 함수를 호출하여 detection 구조체 배열을 할당한 후, fill\_network\_boxes 함수를 호출하여 해당 배열에 탐지 결과를 채운다.
* 탐지된 객체의 개수는 num 포인터를 통해 반환된다.



## free\_detections

```c
void free_detections(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(dets[i].prob);
        if(dets[i].mask) free(dets[i].mask);
    }
    free(dets);
}
```

함수 이름: free\_detections

입력:&#x20;

* dets: detection 구조체 배열의 포인터
* n: 배열의 크기

동작:&#x20;

* detection 구조체 배열과 그 안에 있는 동적 할당된 배열들을 해제한다.

설명:&#x20;

* detection 구조체 배열은 객체 감지(object detection) 알고리즘에서 얻은 결과물을 담고 있다.&#x20;
* 이 함수는 이러한 detection 구조체 배열과 그 안에 있는 확률(prob) 배열과 마스크(mask) 배열을 해제한다.&#x20;
* 이는 메모리 누수(memory leak)를 방지하고 메모리를 효율적으로 사용하기 위함이다.



## network\_predict\_image

```c
float *network_predict_image(network *net, image im)
{
    image imr = letterbox_image(im, net->w, net->h);
    set_batch_network(net, 1);
    float *p = network_predict(net, imr.data);
    free_image(imr);
    return p;
}
```

함수 이름: network\_predict\_image&#x20;

입력:

* network \*net: 예측할 네트워크 포인터
* image im: 예측할 이미지

동작:

* 입력 이미지를 지정된 네트워크 입력 크기로 조정하는 letterbox\_image 함수를 호출하여 새로운 이미지 생성
* 네트워크 배치 크기를 1로 설정하는 set\_batch\_network 함수 호출
* network\_predict 함수를 호출하여 예측 실행
* 새로운 이미지를 해제한 후, 결과 출력 포인터를 반환

설명:&#x20;

* 입력으로 주어진 이미지를 예측할 네트워크의 입력 크기로 맞추고, 배치 크기를 1로 설정하여 네트워크를 실행하여 예측 결과를 출력하는 함수입니다.&#x20;
* 이 함수는 이미지를 직접 입력으로 사용하지 않고, 입력 이미지를 새로운 이미지로 변환하여 사용합니다.



## network\_width, network\_height

```c
int network_width(network *net){return net->w;}
int network_height(network *net){return net->h;}
```

함수: network\_width, network\_height

입력:&#x20;

* 네트워크 구조체 포인터 (network \*)

동작:&#x20;

* 네트워크의 입력 이미지 가로 크기 (네트워크 w)를 반환합니다.
* 네트워크의 입력 이미지 세로 크기 (네트워크 h)를 반환합니다.

설명:&#x20;

* 입력으로 받은 네트워크 구조체 포인터를 이용해 네트워크의 w 값을 반환합니다.
* 입력으로 받은 네트워크 구조체 포인터를 이용해 네트워크의 h 값을 반환합니다.



## network\_predict\_data\_multi

```c
matrix network_predict_data_multi(network *net, data test, int n)
{
    int i,j,b,m;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch*test.X.rows, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        for(m = 0; m < n; ++m){
            float *out = network_predict(net, X);
            for(b = 0; b < net->batch; ++b){
                if(i+b == test.X.rows) break;
                for(j = 0; j < k; ++j){
                    pred.vals[i+b][j] += out[j+b*k]/n;
                }
            }
        }
    }
    free(X);
    return pred;   
}
```

함수 이름: network\_predict\_data\_multi

입력:

* network \*net: 예측할 신경망 구조
* data test: 예측할 데이터셋
* int n: 예측할 때 신경망을 몇 번 실행할 것인지 지정하는 변수

동작:

* 예측할 데이터셋을 배치로 나누어 신경망에 입력으로 전달
* 입력된 신경망을 n번 실행하여 예측 값을 추정
* 추정한 예측 값을 평균하여 최종 예측 값으로 반환

설명:&#x20;

* 이 함수는 입력된 데이터셋에 대한 신경망 예측 값을 평균하여 반환하는 함수입니다.&#x20;
* 입력된 데이터셋은 행렬(matrix) 형태로 주어집니다.&#x20;
* 이때, 입력된 데이터셋이 한 번에 처리하기에 너무 크다면, 배치(batch) 단위로 데이터셋을 나누어 신경망에 입력으로 전달합니다.&#x20;
* 이후, 입력된 신경망을 n번 실행하여 예측 값을 추정하고, 추정한 예측 값들을 평균하여 최종 예측 값을 계산합니다.&#x20;
* 최종 예측 값은 행렬(matrix) 형태로 반환됩니다.



## network\_predict\_data

```c
matrix network_predict_data(network *net, data test)
{
    int i,j,b;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;   
}
```

함수 이름: network\_predict\_data

입력:

* network \*net: 예측에 사용되는 뉴럴 네트워크
* data test: 예측할 데이터가 포함된 data 구조체

동작:

* test의 X 데이터를 net의 batch 크기로 분할하여 예측 수행
* 모든 예측 결과를 한 번에 반환

설명:

* 입력으로 주어진 네트워크(net)와 데이터(test)를 사용하여 예측(prediction) 수행
* test.X.rows는 데이터의 총 행(row) 수
* net->batch는 미리 지정된 batch 크기
* X는 test 데이터의 일부를 저장하는 포인터로, float형으로 초기화됨
* for 루프를 사용하여 test 데이터를 batch 크기로 분할하고, X에 저장
* network\_predict 함수를 사용하여 X를 예측하고, 결과를 out에 저장
* pred 구조체에 예측 결과를 저장하고, 반환
* pred 구조체는 행렬(matrix) 구조체이며, 예측 결과를 담는 2차원 배열을 가짐
* pred.vals\[i]\[j]는 i행 j열의 예측 결과에 해당함
* 메모리를 해제하고, pred 구조체를 반환



## print\_network

```c
void print_network(network *net)
{
    int i,j;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}
```

함수 이름: print\_network

입력:&#x20;

* network 구조체 포인터(net)

동작:&#x20;

* 네트워크의 각 레이어의 출력 평균과 분산을 계산하고, 각 레이어의 출력 값과 함께 출력하여 네트워크의 출력 상태를 디버깅용으로 출력한다.

설명:

* 각 레이어의 출력 평균과 분산을 계산한다.
* 각 레이어의 출력 값을 출력한다. 출력 값이 100개를 넘으면 처음 100개와 마지막 값만 출력한다.
* 디버깅 정보를 표준 에러(stderr)로 출력한다.



## compare\_networks

```c
void compare_networks(network *n1, network *n2, data test)
{
    matrix g1 = network_predict_data(n1, test);
    matrix g2 = network_predict_data(n2, test);
    int i;
    int a,b,c,d;
    a = b = c = d = 0;
    for(i = 0; i < g1.rows; ++i){
        int truth = max_index(test.y.vals[i], test.y.cols);
        int p1 = max_index(g1.vals[i], g1.cols);
        int p2 = max_index(g2.vals[i], g2.cols);
        if(p1 == truth){
            if(p2 == truth) ++d;
            else ++c;
        }else{
            if(p2 == truth) ++b;
            else ++a;
        }
    }
    printf("%5d %5d\n%5d %5d\n", a, b, c, d);
    float num = pow((abs(b - c) - 1.), 2.);
    float den = b + c;
    printf("%f\n", num/den);
}
```

함수 이름: compare\_networks

입력:

* network \*n1: 비교할 첫 번째 신경망
* network \*n2: 비교할 두 번째 신경망
* data test: 비교할 데이터셋

동작:&#x20;

* 두 개의 신경망 모델 n1과 n2를 통해 주어진 데이터셋 test를 예측하고, 예측 결과를 비교하여 오분류율을 계산한다.&#x20;
* 이를 위해, 두 모델이 예측한 결과를 토대로 4분할표(confusion matrix)를 구하고, 각각의 값에 따라 오분류율을 계산한다.

설명:

* a,b,c,d: 4분할표에서 각각 오분류된 샘플의 개수
* truth: 실제 클래스 레이블
* p1: 첫 번째 신경망 n1의 예측 결과 클래스 레이블
* p2: 두 번째 신경망 n2의 예측 결과 클래스 레이블
* num: 오분류된 샘플 수의 차이에 대한 제곱
* den: 오분류된 샘플 수의 합
* 결과로는 4분할표와 오분류율이 출력된다.



## network\_accuracy

```c
float network_accuracy(network *net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}
```

함수 이름: network\_accuracy

입력:

* network \*net: 신경망 모델 포인터
* data d: 테스트 데이터

동작:&#x20;

* 입력된 신경망 모델을 사용하여 테스트 데이터의 예측 결과를 구하고, 실제 레이블과 비교하여 정확도를 계산합니다.

설명:

* matrix\_topk\_accuracy 함수를 사용하여 계산된 정확도를 반환합니다.
* guess 행렬은 함수 내부에서 생성되고, 반환되기 전에 메모리가 해제됩니다.



## network\_accuracies

```c
float *network_accuracies(network *net, data d, int n)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}
```

함수 이름: network\_accuracies&#x20;

입력:

* network \*net : 평가할 신경망
* data d : 평가할 데이터
* int n : 상위 n개 예측의 정확도를 계산 (1 이상의 정수)&#x20;

동작:

* 입력된 신경망과 데이터를 사용하여 예측을 수행하고, 상위 1개와 상위 n개 예측의 정확도를 계산
* 계산된 정확도를 배열로 저장하고, 이를 반환
* 반환된 배열의 첫 번째 요소는 상위 1개 예측의 정확도, 두 번째 요소는 상위 n개 예측의 정확도&#x20;

설명:

* 입력된 데이터로 입력된 신경망의 예측을 수행하고, 상위 1개와 상위 n개 예측의 정확도를 계산하는 함수입니다.
* 계산된 정확도는 정적으로 선언된 배열에 저장되며, 이 배열이 반환됩니다.
* 배열의 첫 번째 요소는 상위 1개 예측의 정확도, 두 번째 요소는 상위 n개 예측의 정확도입니다.
* 함수 내부에서는 network\_predict\_data 함수를 사용하여 입력된 데이터의 예측 결과를 계산하고, 계산된 결과를 matrix\_topk\_accuracy 함수를 사용하여 정확도를 계산합니다. 계산된 결과는 먼저 정적으로 선언된 배열에 저장되며, 이 배열이 반환됩니다.



## get\_network\_output\_layer

```c
layer get_network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}
```

함수 이름: get\_network\_output\_layer

입력:&#x20;

* network \*net (신경망 포인터)

동작:&#x20;

* 신경망의 출력 레이어를 찾아 해당 레이어를 반환합니다. 출력 레이어는 COST 레이어가 아닌 마지막 레이어입니다.

설명:

* net: 신경망 포인터
* i: 반복문을 위한 정수 변수
* 반환값: 출력 레이어 (layer 타입)



## network\_accuracy\_multi

```c
float network_accuracy_multi(network *net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}
```

함수 이름: network\_accuracy\_multi&#x20;

입력:&#x20;

* network \*net (신경망 구조체 포인터)
* data d (테스트 데이터)
* int n (클래스 수)&#x20;

동작:&#x20;

* 입력된 테스트 데이터 d와 신경망 net를 이용하여 n개의 클래스 중 가장 높은 예측값을 갖는 클래스를 예측하고, 이를 실제 클래스와 비교하여 정확도를 계산한다.&#x20;

설명:&#x20;

* 이 함수는 멀티 클래스 분류 문제에서 신경망의 정확도를 계산하는 데 사용된다.&#x20;
* 입력된 테스트 데이터와 신경망을 이용하여 n개의 클래스 중 가장 높은 예측값을 갖는 클래스를 예측하고, 이를 실제 클래스와 비교하여 정확도를 계산한다.&#x20;
* 이 때, matrix\_topk\_accuracy 함수를 이용하여 정확도를 계산한다.



## free\_network

```c
void free_network(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        free_layer(net->layers[i]);
    }
    free(net->layers);
    if(net->input) free(net->input);
    if(net->truth) free(net->truth);

    free(net);
}
```

함수 이름: free\_network

입력:&#x20;

* network 구조체 포인터 (network \*)

동작:&#x20;

* 주어진 network 구조체와 그 안의 모든 레이어, 입력 데이터, 정답 데이터 등을 해제(free)합니다.

설명:&#x20;

* 이 함수는 deep learning 모델을 만드는 데 사용되는 network 구조체와 그 안에 포함된 레이어들을 해제하는 함수입니다.&#x20;
* 이 함수는 모든 레이어, 입력 데이터, 정답 데이터 등을 메모리에서 해제하며, 이를 통해 메모리 누수(memory leak)를 방지할 수 있습니다.&#x20;
* 이 함수가 호출되면, 해당 네트워크와 관련된 모든 자원을 해제하므로, 더 이상 해당 네트워크를 사용할 수 없게 됩니다.



## network\_output\_layer

```c
layer network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}
```

함수 이름: network\_output\_layer

입력:&#x20;

* network 구조체 포인터 (net)

동작:&#x20;

* 해당 네트워크 구조체의 마지막 레이어부터 순서대로 탐색하며, COST 레이어를 만날 때까지 탐색을 반복하고 COST 레이어를 찾으면 해당 레이어를 반환함.

설명:&#x20;

* 딥러닝 모델의 출력값을 담고 있는 레이어를 반환하는 함수입니다.&#x20;
* 네트워크 구조체에서 마지막 레이어부터 역순으로 탐색하며, COST 레이어를 만날 때까지 탐색합니다.&#x20;
* COST 레이어는 신경망 모델에서 마지막 레이어로 사용되며, 이전 레이어의 출력값을 입력으로 받아 손실 함수 값을 계산합니다.&#x20;
* 따라서 COST 레이어는 모델의 출력값을 담고 있습니다. COST 레이어를 찾으면 해당 레이어를 반환합니다.



## network\_inputs

```c
int network_inputs(network *net)
{
    return net->layers[0].inputs;
}
```

함수 이름: network\_inputs&#x20;

입력:&#x20;

* network \*net (신경망 구조체)&#x20;

동작:&#x20;

* 주어진 신경망의 첫 번째 레이어의 입력 수를 반환합니다.&#x20;

설명:&#x20;

* 이 함수는 주어진 신경망의 첫 번째 레이어의 입력 수를 반환합니다.&#x20;
* 첫 번째 레이어는 입력 레이어이므로 입력 수는 입력 이미지의 크기를 결정하는 값입니다.



## network\_outputs

```c
int network_outputs(network *net)
{
    return network_output_layer(net).outputs;
}
```

함수 이름: network\_outputs

입력:&#x20;

* network \*net (Neural Network 모델)

동작:&#x20;

* Neural Network의 출력 레이어의 출력값 개수를 반환한다.

설명:&#x20;

* 이 함수는 Neural Network 모델의 출력 레이어의 출력값 개수를 반환하는 함수이다.&#x20;
* 반환되는 값은 int 타입이다.



## network\_output

```c
float *network_output(network *net)
{
    return network_output_layer(net).output;
}
```

함수 이름: network\_output&#x20;

입력:&#x20;

* network 구조체 포인터 (훈련된 신경망 모델)&#x20;

동작:&#x20;

* 훈련된 신경망 모델의 최종 출력을 반환하는 함수.&#x20;
* 내부적으로 network\_output\_layer 함수를 호출하여 출력 레이어의 출력을 반환함.&#x20;

설명:&#x20;

* 입력된 네트워크 모델에 대해, 최종 출력을 담은 배열의 포인터를 반환하는 함수입니다.&#x20;
* 이 함수는 내부적으로 network\_output\_layer 함수를 호출하여 출력 레이어의 출력을 반환합니다.

