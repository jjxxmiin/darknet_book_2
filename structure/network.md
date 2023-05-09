---
description: network 구조체
---

# network

Network 구조체 입니다.

```c
typedef struct network{
    int n;                            
    int batch;                        
    size_t *seen;
    int *t;
    float epoch;                     
    int subdivisions;
    layer *layers;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;

    int gpu_index;
    tree *hierarchy;

    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;
    float clip;
} network;
```

* `n`: 레이어의 개수입니다.
* `batch`: 하나의 batch에 있는 이미지의 개수
* `seen`: 학습 데이터 셋에서 본 이미지의 개수
* `t`: 학습 중인 epoch의 번호
* `epoch`: 학습 중인 epoch의 번호(소수점 표시 가능)
* `subdivisions`: 입력 이미지의 크기를 조절하기 위한 파라미터
* `layers`: 네트워크에 사용되는 레이어 배열
* `output`: 출력값을 저장할 배열
* `learning_rate_policy`: 학습률 조절을 위한 정책
* `learning_rate`: 초기 학습률
* `momentum`: 모멘텀
* `decay`: 가중치 감소
* `gamma`: 학습률 감소 비율
* `scale`: 입력 이미지의 크기 조절 비율
* `power`: 학습률 감소를 조절하는 파라미터
* `time_steps`: RNN 모델에서 사용되는 파라미터
* `step`: 현재 학습 중인 step의 번호
* `max_batches`: 최대 batch 개수
* `scales`: 각 레이어의 학습률 조절을 위한 파라미터
* `steps`: 학습률 조절을 위한 step 배열
* `num_steps`: 학습률 조절을 위한 step의 개수
* `burn_in`: 초기 학습률 감소에 사용되는 파라미터
* `adam`: Adam 알고리즘 사용 여부
* `B1, B2, eps`: Adam 알고리즘에 사용되는 파라미터
* `inputs`: 입력 이미지의 채널 수
* `outputs`: 출력 레이어의 개수
* `truths`: 학습 데이터 셋에서 정답 label의 개수
* `notruth`: 학습 데이터 셋에서 정답 label이 없는 이미지의 개수
* `h, w, c`: 이미지의 높이(height), 너비(width), 채널 수(channel)를 나타내는 변수
* `max_crop, min_crop` : 이미지를 무작위로 자를 때, 최대 및 최소 자를 크기
* `max_ratio, min_ratio` : 이미지를 무작위로 자를 때, 최대 및 최소 자를 비율
* `center` : 이미지를 중앙으로 정렬할지 여부를 나타내는 변수
* `angle` : 이미지를 회전할 각도
* `aspect` : 이미지의 종횡비(aspect ratio)를 무작위로 변화시키는 비율
* `exposure, saturation, hue`   : 이미지를 밝기, 채도, 색상으로 무작위로 변화시키는 비율
* `random` : 무작위 데이터 증강 (augmentation) 을 사용 여부를 나타내는플래그
* `gpu_index` : GPU 인덱스
* `hierarchy` : 사용할 YOLO의 계층 구조
* `input` : 네트워크에 입력되는 데이터
* `truth` : 학습 시, 실제 정답 데이터
* `delta` : 학습 시, 역전파 시 사용될 오차
* `workspace` : 메모리를 동적으로 할당
* `train` : 현재 모드가 학습 모드인지, 추론 모드인지 결정하는 플래그
* `index` : 네트워크의 인덱스
* `cost` : 현재 배치에서의 손실값
* `clip` : 그래디언트 폭주 방지를 위해 클리핑, 일정 범위를 넘어서는 그래디언트 값을 자르는데, 이 때 잘라낼 최대 크기



#### + momentum을 추가하면 어떻게 되나요?

```c
# origin

weight = weight + learning rate * dL / dw

# momentum

velocity = momentum * velocity - learning rate * dL / dw
weight = weight + velocity
```

만약 momentum이 0.9이고 2번을 한다고 가정하면, $$velocity = 0.9 * (-\frac{\partial L}{\partial W_1}) - \lambda \frac{\partial L}{\partial W_2}$$

* `decay` : weight decay (L2 regularization)

L2 regularization은 가중치가 크면 클수록 큰 페널티를 줘서 overfitting을 방지하는 방법입니다.

Loss함수에 $$\frac{1}{2} \lambda W W^T$$를 더합니다. 여기서 $$\lambda$$가 `decay`입니다. 클수록 가중치에 큰 페널티를 줍니다.

$$\frac{1}{2}$$가 있는 이유는 미분시 $$W^2$$의 2가 내려와서 값을 1로 만들기 위함입니다.

$$W = W - learning rate * (\frac{\partial L}{\partial W} + \lambda W)$$
