---
description: layer 구조체
---

# layer

Layer 구조체 입니다.

```c
struct layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    void (*forward)   (struct layer, struct network);
    void (*backward)  (struct layer, struct network);
    void (*update)    (struct layer, update_args);
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h,w,c;                              /// input height, width, channel
    int out_h, out_w, out_c;                /// output height, width, channel
    int n;                                  /// number of filter(= output channel)
    int max_boxes;
    int groups;
    int size;                               /// kernel size
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int noloss;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    int random;
    float ignore_thresh;
    float truth_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontload;
    int dontsave;
    int dontloadscales;
    int numload;

    float temperature;
    float probability;
    float scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers;
    int   * input_sizes;
    int   * map;
    int   * counts;
    float ** sums;
    float * rand;
    float * cost;
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;
    float * combine_cpu;
    float * combine_delta_cpu;

    float * concat;
    float * concat_delta;

    float * binary_weights;

    float * biases;
    float * bias_updates;

    float * scales;
    float * scale_updates;

    float * weights;
    float * weight_updates;

    float * delta;
    float * output;
    float * loss;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    float * m;
    float * v;

    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;


    float *z_cpu;
    float *r_cpu;
    float *h_cpu;
    float * prev_state_cpu;

    float *temp_cpu;
    float *temp2_cpu;
    float *temp3_cpu;

    float *dh_cpu;
    float *hh_cpu;
    float *prev_cell_cpu;
    float *cell_cpu;
    float *f_cpu;
    float *i_cpu;
    float *g_cpu;
    float *o_cpu;
    float *c_cpu;
    float *dc_cpu;

    float * binary_input;

    struct layer *input_layer;                      /// rnn input layer
    struct layer *self_layer;                       /// rnn self layer
    struct layer *output_layer;                     /// rnn output layer

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;

    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;

    tree *softmax_tree;

    size_t workspace_size;
};
```

* `type`: 레이어의 종류를 나타내는 열거형(enum) 변수
* `activation`: 활성화 함수를 나타내는 열거형 변수
* `cost_type`: 비용 함수를 나타내는 열거형 변수
* `forward`: 레이어의 순전파(forward propagation) 연산을 수행하는 함수 포인터
* `backward`: 레이어의 역전파(backward propagation) 연산을 수행하는 함수 포인터
* `update`: 레이어의 가중치(weight)와 편향(bias) 값을 업데이트하는 함수 포인터
* `batch_normalize`: 배치 정규화(batch normalization)를 수행할지 여부를 나타내는 정수 변수
* `shortcut`: shortcut 연결을 사용할지 여부를 나타내는 정수 변수
* `batch`: 미니배치 크기를 나타내는 정수 변수
* `forced`: 강제로 레이어를 수행할지 여부를 나타내는 정수 변수
* `flipped`: 입력 데이터를 뒤집어서 사용할지 여부를 나타내는 정수 변수
* `inputs`: 레이어의 입력 개수를 나타내는 정수 변수
* `outputs`: 레이어의 출력 개수를 나타내는 정수 변수
* `nweights`: 가중치(weight)의 개수를 나타내는 정수 변수
* `nbiases`: 편향(bias)의 개수를 나타내는 정수 변수
* `extra`: 추가 파라미터 값의 개수를 나타내는 정수 변수
* `truths`: 레이블(label) 데이터의 개수를 나타내는 정수 변수
* `h,w,c`: 입력 데이터의 높이, 너비, 채널(channel)을 나타내는 정수 변수
* `out_h, out_w, out_c`: 출력 데이터의 높이, 너비, 채널을 나타내는 정수 변수
* `n`: 필터(filter)의 개수(출력 채널 수)를 나타내는 정수 변수
* `max_boxes`: bounding box의 최대 개수를 나타내는 정수 변수
* `groups`: 그룹 수를 나타내는 정수 변수
* `size`: 컨볼루션(kernel)의 크기를 나타내는 정수 변수
* `side`: max pooling 연산의 윈도우 크기를 나타내는 정수 변수
* `stride`: 스트라이드(stride) 크기를 나타내는 정수 변수
* `reverse`: RNN 레이어에서 시퀀스의 역순으로 처리할지 여부를 나타내는 플래그
* `flatten`: 입력 데이터를 1차원 벡터로 평탄화할지 여부를 나타내는 플래그
* `spatial`: 컨볼루션 레이어에서 공간적인 정보를 유지할지 여부를 나타내는 플래그



* `pad`: 입력 데이터 주변에 패딩을 추가할 크기를 나타내는 값
* `sqrt`: 배치 정규화 계층에서 분산에 더해줄 작은 값으로, 분산이 0이 되는 것을 방지
* `flip`: 이미지를 수평으로 뒤집을지 여부를 나타내는 플래그
* `index`: 컨볼루션 레이어에서 사용할 필터의 인덱스
* `binary`: 이진 분류를 위한 활성화 함수 사용 여부를 나타내는 플래그
* `xnor`: XNOR-Networks를 위한 플래그
* `steps`: RNN 레이어에서 시퀀스의 길이
* `hidden`: RNN 레이어에서 숨겨진 상태의 차원 수
* `truth`: YOLO 계열의 레이어에서 참값의 크기
* `smooth`: 경사 하강법에서 사용할 스무딩 계수
* `dot`: 배치 정규화 계층에서 공분산을 계산할 때 사용되는 값
* `angle`: RoI Pooling 레이어에서 RoI를 회전시키는 각도
* `jitter`: 데이터 확장을 위한 이미지의 크기 변화 폭을 나타내는 값
* `saturation`: 데이터 확장을 위한 채도 변화 폭을 나타내는 값
* `exposure`: 데이터 확장을 위한 밝기 변화 폭을 나타내는 값
* `shift`: 데이터 확장을 위한 이미지 이동 변화 폭을 나타내는 값
* `ratio`: 데이터 확장을 위한 이미지 비율 변화 폭을 나타내는 값
* `learning_rate_scale`: 가중치 업데이트 시 학습률을 곱할 스케일 값을 나타내는 값
* `clip`: 경사 하강법에서 사용할 기울기 클리핑 임계값
* `noloss`: 손실이 없는 레이어인지 여부
* `softmax`: 소프트맥스 레이어인지 여부
* `classes`: 분류해야 할 클래스의 개수
* `coords`: 바운딩 박스 좌표의 개수
* `background`: 배경 클래스가 존재하는지 여부
* `rescore`: bbox의 신뢰도를 다시 계산하는지 여부
* `objectness`: bbox의 objectness를 계산하는지 여부
* `joint`: 다른 레이어와 연결되는지 여부
* `noadjust`: 학습 가능한 파라미터가 있는지 여부
* `reorg`: Reorg 레이어인지 여부
* `log`: 로그 레이어인지 여부
* `tanh`: 하이퍼볼릭 탄젠트 레이어인지 여부
* `mask`: 마스크 배열
* `total`: 마스크 배열의 총 길이



* `alpha`: 경사 하강법의 학습률
* `beta`: 경사 하강법의 모멘텀 계수
* `kappa`: RPN 레이어의 균형 매개변수



* `coord_scale`: bbox 좌표의 손실 가중치
* `object_scale`: bbox objectness의 손실 가중치
* `noobject_scale`: bbox non-objectness의 손실 가중치
* `mask_scale`: 마스크의 손실 가중치
* `class_scale`: 클래스의 손실 가중치
* `bias_match`: 바이어스를 맞추는지 여부
* `random`: 무작위로 초기화하는지 여부
* `ignore_thresh`: 무시 임계값
* `truth_thresh`: 진실 임계값
* `thresh`: 임계값
* `focus`: focal loss 매개변수
* `classfix`: 클래스 인덱스 매핑 매개변수
* `absolute`: 절대값을 사용하는지 여부



* `onlyforward`: 순전파만 수행하는지 여부
* `stopbackward`: 역전파를 중지하는지 여부
* `dontload`: 레이어의 가중치를 로드하지 않는지 여부
* `dontsave`: 레이어의 가중치를 저장하지 않는지 여부
* `dontloadscales`: 레이어의 스케일을 로드하지 않는지 여부
* `numload`: 가중치를 로드한 횟수



* `temperature`: Softmax 온도
* `probability`: 드롭아웃 확률
* `scale`: 배치 정규화 매개변수
* `cweights`: 클래스 가중치 파일 이름
* `indexes`: 인덱스 배열
* `input_layers`: 현재 레이어에 입력으로 들어오는 레이어의 인덱스 배열
* `input_sizes`: 입력 레이어의 크기 배열
* `map`: 이전 레이어와 현재 레이어 간의 매핑 정보
* `counts`: 이전 레이어와 현재 레이어 간의 매핑 정보에 따른 각각의 노드 수
* `sums`: 이전 레이어와 현재 레이어 간의 매핑 정보에 따른 가중치 합
* `rand`: 무작위 값
* `cost`: 비용
* `state`: 현재 레이어의 상태 값
* `prev_state`: 이전 레이어의 상태 값
* `forgot_state`: 현재 레이어에서 사용되는 이전 레이어의 상태 값
* `forgot_delta`: 현재 레이어에서 사용되는 이전 레이어의 상태 값의 변화량
* `state_delta`: 현재 레이어의 상태 값의 변화량
* `combine_cpu`: 두 레이어를 결합하기 위한 값
* `combine_delta_cpu`: 결합한 레이어의 변화량



* `concat`: 레이어를 연결하기 위한 값
* `concat_delta`: 연결한 레이어의 변화량



* `binary_weights`: 이진 가중치
* `biases`: 편향 값
* `bias_updates`: 편향 값의 업데이트
* `scales`: 스케일 값
* `scale_updates`: 스케일 값의 업데이트
* `weights`: 가중치 값
* `weight_updates`: 가중치 값의 업데이트
* `delta`: 오차 값
* `output`: 출력 값
* `loss`: 손실 값
* `squared`: 제곱 값
* `norms`: 노름 값
* `spatial_mean`: 공간 평균 값
* `mean`: 평균 값
* `variance`: 분산 값
* `mean_delta`: 평균 값의 변화량
* `variance_delta`: 분산 값의 변화량
* `rolling_mean`: 이동 평균 값
* `rolling_variance`: 이동 분산 값
* `x`: 입력 값
* `x_norm`: 정규화된 입력 값
* `m`: 이동 평균
* `v`: 이동 분산
* `bias_m`: 편향 이동 평균
* `bias_v`: 편향 이동 분산
* `scale_m`: 배치 정규화(Batch Normalization)에서 사용되는 moving mean
* `scale_v`: 배치 정규화에서 사용되는 moving variance



* `z_cpu, r_cpu, h_cpu, prev_state_cpu`: LSTM 네트워크의 게이트를 구성하는 변수들



* `temp_cpu, temp2_cpu, temp3_cpu`: 임시 저장용 변수들



* `dh_cpu, hh_cpu, prev_cell_cpu, cell_cpu, f_cpu, i_cpu, g_cpu, o_cpu, c_cpu, dc_cpu`: LSTM 네트워크의 셀(Cell)을 구성하는 변수들
* `binary_input`: 이진(Binary) 입력값
* `input_layer, self_layer, output_layer`: RNN(Recurrent Neural Network) 구조에서 사용되는 입력 레이어, 자기 순환 레이어, 출력 레이어
* `reset_layer, update_layer, state_layer`: LSTM 네트워크에서 사용되는 reset, update, state 게이트를 구성하는 레이어
* `input_gate_layer, state_gate_layer, input_save_layer, state_save_layer, input_state_layer, state_state_layer`: LSTM 네트워크에서 사용되는 게이트와 셀을 구성하는 레이어
* `input_z_layer, state_z_layer, input_r_layer, state_r_layer, input_h_layer, state_h_layer`: LSTM 네트워크의 게이트를 구성하는 레이어



* `wz, uz, wr, ur, wh, uh, uo, wo, uf, wf, ui, wi, ug, wg`: LSTM 네트워크에서 사용되는 가중치와 편향



* `softmax_tree`: Softmax 함수에서 사용되는 트리(Tree) 구조



* `workspace_size`: 사용 가능한 메모리 공간의 크기





