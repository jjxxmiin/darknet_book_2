# region\_layer

## forward\_region\_layer

```c
void forward_region_layer(const layer l, network net)
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    float avg_iou = 0;
    float recall = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        if(l.softmax_tree){
            int onlyclass = 0;
            for(t = 0; t < 30; ++t){
                box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
                if(!truth.x) break;
                int class = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];
                float maxp = 0;
                int maxi = 0;
                if(truth.x > 100000 && truth.y > 100000){
                    for(n = 0; n < l.n*l.w*l.h; ++n){
                        int class_index = entry_index(l, b, n, l.coords + 1);
                        int obj_index = entry_index(l, b, n, l.coords);
                        float scale =  l.output[obj_index];
                        l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                        float p = scale*get_hierarchy_probability(l.output + class_index, l.softmax_tree, class, l.w*l.h);
                        if(p > maxp){
                            maxp = p;
                            maxi = n;
                        }
                    }
                    int class_index = entry_index(l, b, maxi, l.coords + 1);
                    int obj_index = entry_index(l, b, maxi, l.coords);
                    delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
                    if(l.output[obj_index] < .3) l.delta[obj_index] = l.object_scale * (.3 - l.output[obj_index]);
                    else  l.delta[obj_index] = 0;
                    l.delta[obj_index] = 0;
                    ++class_count;
                    onlyclass = 1;
                    break;
                }
            }
            if(onlyclass) continue;
        }
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                    float best_iou = 0;
                    for(t = 0; t < 30; ++t){
                        box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                        }
                    }
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, l.coords);
                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                    if(l.background) l.delta[obj_index] = l.noobject_scale * (1 - l.output[obj_index]);
                    if (best_iou > l.thresh) {
                        l.delta[obj_index] = 0;
                    }

                    if(*(net.seen) < 12800){
                        box truth = {0};
                        truth.x = (i + .5)/l.w;
                        truth.y = (j + .5)/l.h;
                        truth.w = l.biases[2*n]/l.w;
                        truth.h = l.biases[2*n+1]/l.h;
                        delta_region_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, l.delta, .01, l.w*l.h);
                    }
                }
            }
        }
        for(t = 0; t < 30; ++t){
            box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            for(n = 0; n < l.n; ++n){
                int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                if(l.bias_match){
                    pred.w = l.biases[2*n]/l.w;
                    pred.h = l.biases[2*n+1]/l.h;
                }
                pred.x = 0;
                pred.y = 0;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }

            int box_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 0);
            float iou = delta_region_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth.w*truth.h), l.w*l.h);
            if(l.coords > 4){
                int mask_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 4);
                delta_region_mask(net.truth + t*(l.coords + 1) + b*l.truths + 5, l.output, l.coords - 4, mask_index, l.delta, l.w*l.h, l.mask_scale);
            }
            if(iou > .5) recall += 1;
            avg_iou += iou;

            int obj_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords);
            avg_obj += l.output[obj_index];
            l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index]);
            if (l.rescore) {
                l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index]);
            }
            if(l.background){
                l.delta[obj_index] = l.object_scale * (0 - l.output[obj_index]);
            }

            int class = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];
            if (l.map) class = l.map[class];
            int class_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords + 1);
            delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
            ++count;
            ++class_count;
        }
    }
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, count);
}
```

함수 이름: forward\_region\_layer

입력:

* layer: region\_layer 구조체 포인터
* net: network 구조체 포인터

동작:

* 입력으로 받은 region\_layer 구조체 포인터를 사용하여 region layer를 순전파(forward) 진행
* 각각의 입력 이미지에 대해 region layer의 출력(feature map)을 계산
* 계산된 feature map을 region\_layer 구조체의 output 변수에 저장

설명:

* 이 함수는 YOLO 객체 검출 알고리즘의 region layer의 순전파를 수행하는 함수입니다.
* region layer는 입력 이미지의 여러 영역(region)을 검출하고 각 영역에 대해 객체 클래스 확률과 위치 정보를 예측합니다.
* 입력으로는 region\_layer 구조체 포인터와 network 구조체 포인터를 받습니다.
* region\_layer 구조체는 layer 구조체를 상속하며, 필요한 정보들을 포함합니다.
* 이 함수는 입력 이미지를 받아서 region layer의 출력(feature map)을 계산하고, 이를 region\_layer 구조체의 output 변수에 저장합니다.
* 계산된 feature map은 다음 단계에서 YOLO 알고리즘의 다른 layer들과 함께 사용됩니다.



## backward\_region\_layer

```c
void backward_region_layer(const layer l, network net)
{
    /*
       int b;
       int size = l.coords + l.classes + 1;
       for (b = 0; b < l.batch*l.n; ++b){
       int index = (b*size + 4)*l.w*l.h;
       gradient_array(l.output + index, l.w*l.h, LOGISTIC, l.delta + index);
       }
       axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
     */
}
```

함수 이름: backward\_region\_layer

입력:&#x20;

* layer 구조체 l
* network 구조체 net

동작:

* Region Layer의 역전파(backpropagation)를 수행하는 함수입니다.

설명:&#x20;

* 이 함수는 Region Layer의 역전파를 수행하는데, 이를 위해 먼저 입력으로 받은 layer와 network 구조체를 사용합니다.&#x20;
* 그리고 해당 layer의 출력값과 delta값을 이용하여 gradient\_array 함수를 호출하여 미분값(gradient)을 구합니다.&#x20;
* 그 후, axpy\_cpu 함수를 이용하여 미분값을 누적시켜 네트워크의 전체적인 delta값을 구합니다.



## resize\_reorg\_layer

```c
void resize_region_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + l->coords + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));
}
```

함수 이름: resize\_region\_layer

입력:

* layer \*l: resize를 수행할 region layer의 포인터
* int w: layer의 새로운 width
* int h: layer의 새로운 height

동작:

* region layer의 크기를 재조정한다.
* l->w, l->h 값을 새로운 width와 height로 업데이트한다.
* l->outputs와 l->inputs 값을 재계산한다.
* l->output과 l->delta의 메모리 크기를 재할당한다.

설명:&#x20;

* 이 함수는 Darknet neural network library에서 사용되는 함수로, region layer의 크기를 재조정하는 역할을 한다.&#x20;
* region layer는 객체 검출을 위해 사용되는 layer 중 하나이며, 여러 개의 bounding box를 예측하고 클래스별 확률을 출력하는 역할을 한다.&#x20;
* 이 함수는 region layer의 크기가 변경될 때마다 호출되어, output과 delta 배열의 메모리 크기를 재할당하고, outputs과 inputs 값을 새로운 크기로 업데이트한다.&#x20;
* 이를 통해, network가 새로운 크기의 region layer를 처리할 수 있도록 한다.



## make\_region\_layer

```c
layer make_region_layer(int batch, int w, int h, int n, int classes, int coords)
{
    layer l = {0};
    l.type = REGION;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + coords + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.coords = coords;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(n*2, sizeof(float));
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + coords + 1);
    l.inputs = l.outputs;
    l.truths = 30*(l.coords + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    int i;
    for(i = 0; i < n*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_region_layer;
    l.backward = backward_region_layer;

    fprintf(stderr, "detection\n");
    srand(0);

    return l;
}
```

함수 이름: make\_region\_layer

입력:

* int batch: layer의 batch size
* int w: layer의 width
* int h: layer의 height
* int n: layer의 bounding box 개수
* int classes: 분류하고자 하는 클래스의 개수
* int coords: 각 bounding box의 x, y, width, height를 나타내는 좌표 개수

동작:

* region layer를 생성하고 초기화한다.
* layer의 type을 REGION으로 설정한다.
* layer의 n, batch, h, w, c, out\_w, out\_h, out\_c, classes, coords, cost, biases, bias\_updates, outputs, inputs, truths, delta, output 등의 값을 초기화한다.
* layer의 biases 배열을 0.5로 초기화한다.
* layer의 forward와 backward 함수 포인터를 설정한다.
* detection 메시지를 출력한다.
* 생성된 layer를 반환한다.

설명:&#x20;

* 이 함수는 Darknet neural network library에서 사용되는 함수로, region layer를 생성하고 초기화하는 역할을 한다.&#x20;
* region layer는 객체 검출을 위해 사용되는 layer 중 하나이며, 여러 개의 bounding box를 예측하고 클래스별 확률을 출력하는 역할을 한다.&#x20;
* 이 함수는 region layer를 생성하고 초기화하기 위해, layer의 필요한 값들을 초기화한다.&#x20;
* 이를 통해, network가 region layer를 사용하여 객체 검출을 수행할 수 있게 된다.&#x20;
* 초기화된 layer는 반환되어, 이후 다른 layer와 결합하여 network를 구성할 수 있다.



## get\_region\_box

```c
box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}
```

함수 이름: get\_region\_box

입력:

* float \*x: region layer의 출력값
* float \*biases: bounding box의 prior값
* int n: bounding box의 개수
* int index: bounding box의 index
* int i: bounding box의 x좌표 index
* int j: bounding box의 y좌표 index
* int w: region layer의 width
* int h: region layer의 height
* int stride: region layer의 출력값의 stride 값

동작:

* region layer의 출력값과 prior값을 이용하여 bounding box를 계산한다.
* 계산된 bounding box를 반환한다.

설명:&#x20;

* 이 함수는 region layer에서 출력된 값을 이용하여 bounding box를 계산하는 역할을 한다.&#x20;
* region layer는 객체 검출을 위해 사용되는 layer 중 하나이며, 여러 개의 bounding box를 예측하고 클래스별 확률을 출력하는 역할을 한다.&#x20;
* 이 함수는 region layer에서 출력된 값 x와 bounding box의 prior값 biases, 그리고 bounding box의 정보를 나타내는 n, index, i, j, w, h, stride를 이용하여 bounding box를 계산한다.&#x20;
* 계산된 bounding box는 box 구조체로 반환되어, 객체 검출을 수행하는 다른 함수에서 활용된다.



## delta\_region\_box

```c
float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}
```

함수 이름: delta\_region\_box

입력:

* box truth: ground truth box 정보를 담고 있는 구조체
* float \*x: 모델의 출력값
* float \*biases: 모델의 bias
* int n: anchor box 개수
* int index: 현재 처리 중인 box에 대한 index
* int i: 현재 처리 중인 box의 좌측 상단 x좌표
* int j: 현재 처리 중인 box의 좌측 상단 y좌표
* int w: 전체 이미지의 너비
* int h: 전체 이미지의 높이
* float \*delta: 현재 box의 delta 값을 저장할 배열
* float scale: delta 값에 곱해줄 스케일링 인자
* int stride: 모델의 출력값 중 현재 box의 시작 인덱스

동작:&#x20;

* 현재 처리 중인 box와 그에 대응하는 ground truth box 간의 IoU를 계산하고, 이를 반환한다.&#x20;
* 그리고 현재 box의 delta 값을 계산하고 delta 배열에 저장한다.

설명:&#x20;

* 이 함수는 YOLOv3 모델에서 region layer에서 box에 대한 delta 값을 계산하기 위해 사용된다.&#x20;
* 이를 위해 현재 처리 중인 box와 그에 대응하는 ground truth box 간의 IoU를 계산하고, 이를 반환한다.&#x20;
* 그리고 현재 box의 delta 값을 계산하고 delta 배열에 저장한다.&#x20;
* 이 함수는 모델의 학습 과정에서 사용된다.



## delta\_region\_mask

```c
void delta_region_mask(float *truth, float *x, int n, int index, float *delta, int stride, int scale)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[index + i*stride] = scale*(truth[i] - x[index + i*stride]);
    }
}
```

함수 이름: delta\_region\_mask

입력:

* float \*truth: 실제 값
* float \*x: 예측 값
* int n: 마스크 개수
* int index: 시작 인덱스
* float \*delta: 역전파 시 사용될 델타값
* int stride: 데이터의 스트라이드(stride)
* int scale: 스케일 값

동작:

* 마스크(mask) 개수만큼 반복하며, 역전파 시 사용될 델타 값을 계산한다.
* 델타 값은 예측 값에서 실제 값의 차이에 스케일 값을 곱한 값이다.

설명:

* 이 함수는 region\_layer에서 마스크 값에 대한 역전파를 수행하는 함수이다.&#x20;
* 예측 값과 실제 값 사이의 차이에 스케일 값을 곱한 값을 델타 값으로 사용하여 역전파를 수행한다.&#x20;
* 이 함수는 region\_layer에서 사용되며, 마스크 값에 대한 역전파를 처리하는 과정에서 호출된다.



## delta\_region\_class

```c
void delta_region_class(float *output, float *delta, int index, int class, int classes, tree *hier, float scale, int stride, float *avg_cat, int tag)
{
    int i, n;
    if(hier){
        float pred = 1;
        while(class >= 0){
            pred *= output[index + stride*class];
            int g = hier->group[class];
            int offset = hier->group_offset[g];
            for(i = 0; i < hier->group_size[g]; ++i){
                delta[index + stride*(offset + i)] = scale * (0 - output[index + stride*(offset + i)]);
            }
            delta[index + stride*class] = scale * (1 - output[index + stride*class]);

            class = hier->parent[class];
        }
        *avg_cat += pred;
    } else {
        if (delta[index] && tag){
            delta[index + stride*class] = scale * (1 - output[index + stride*class]);
            return;
        }
        for(n = 0; n < classes; ++n){
            delta[index + stride*n] = scale * (((n == class)?1 : 0) - output[index + stride*n]);
            if(n == class) *avg_cat += output[index + stride*n];
        }
    }
}
```

함수 이름: delta\_region\_class

입력:

* output: 모델의 출력값
* delta: 가중치 갱신에 사용되는 출력값의 변화량
* index: 현재 예측값이 저장된 인덱스
* class: 예측된 클래스 인덱스
* classes: 클래스의 총 개수
* hier: 클래스가 계층 구조를 가지는 경우 그 구조 정보를 담고 있는 트리
* scale: 출력값의 변화량에 곱해지는 스케일 값
* stride: 출력값의 차원
* avg\_cat: 출력값의 평균 카테고리
* tag: 미사용

동작:

* hier 가 null이 아닌 경우:
  * class에서 시작하여 hier를 따라 부모 클래스로 이동하면서, 해당 클래스의 출력값을 예측값으로 사용하고, 계층 구조에서 같은 그룹에 속하는 다른 클래스의 출력값은 0으로 만든다.
* hier 가 null인 경우:
  * 출력값을 클래스별로 순회하면서, 예측된 클래스와 일치하는 경우 출력값에 1을 할당하고, 그 외의 경우 0을 할당한다. 이때, 출력값의 변화량은 스케일과 차이에 비례한다.
* tag 가 1인 경우:
  * 출력값이 0이 아니면 예측된 클래스에 해당하는 delta 값을 갱신한다.
* avg\_cat 에서는 hier 가 null인 경우, 출력값에 예측된 클래스의 값을 더해준다.



## logit

```c
float logit(float x)
{
    return log(x/(1.-x));
}
```

함수 이름: logit

입력:&#x20;

* 실수 x

동작:&#x20;

* 로짓(logit) 함수는 0에서 1사이의 값을 가지는 x를 입력받아, 로그(odds) 변환을 수행하여 출력합니다.&#x20;
* 로그 변환은 확률값을 odds값으로 변환하는 과정으로, odds값은 해당 사건이 발생할 확률과 발생하지 않을 확률의 비율을 나타냅니다.&#x20;
* 로짓 함수는 odds값을 실수 범위 전체에서 정의하기 위해 사용됩니다.&#x20;
* 수식으로는 log(x/(1.-x))로 표현됩니다.

설명:

* 로짓 함수는 확률 값을 odds값으로 변환하여 해당 값을 실수 범위 전체에서 정의합니다.
* 확률값 x는 0에서 1사이의 값을 가지며, 로짓 함수의 분모에서 1-x는 해당 사건이 발생하지 않을 확률을 의미합니다.
* 로짓 함수의 출력 값은 해당 사건이 발생할 확률(p)에 대해 log(p/(1-p))와 같이 표현됩니다.
* 로짓 함수는 딥러닝에서 sigmoid 함수와 함께 많이 사용됩니다.



## tisnan

```c
float tisnan(float x)
{
    return (x != x);
}
```

함수 이름: tisnan

입력:&#x20;

* float x

동작:&#x20;

* 인자로 주어진 x가 NaN(Not a Number)인지 여부를 확인하는 함수입니다.

설명:

* C/C++의 isnan 함수는 NaN인 경우에만 true를 반환합니다.&#x20;
* 하지만 tisnan 함수는 x가 NaN이면 true, 그렇지 않으면 false를 반환합니다.&#x20;
* 이는 x가 NaN이 아닌 경우 x와 자기 자신을 비교한 결과가 false가 되기 때문입니다.&#x20;
* x가 NaN인 경우, 어떤 값과 비교해도 false가 아닌 특징을 이용합니다.



## entry\_index

```c
int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(l.coords+l.classes+1) + entry*l.w*l.h + loc;
}
```

함수 이름: entry\_index

입력:

* layer l: YOLO 레이어
* int batch: 미니배치에서 처리할 이미지의 인덱스
* int location: 예측할 그리드 셀의 인덱스
* int entry: 예측하려는 속성(좌표, 클래스 등)의 인덱스

동작:

* YOLO 레이어의 출력 배열에서 지정한 미니배치, 그리드 셀, 속성의 인덱스에 해당하는 요소의 인덱스를 반환한다.

설명:

* YOLO는 그리드 셀을 사용하여 이미지를 분할하고, 각 그리드 셀에서 bounding box와 objectness, 클래스 확률 등을 예측한다.
* 출력 배열은 미니배치, 그리드 셀, 속성의 순서로 이루어져 있으며, 각 요소의 인덱스는 entry\_index 함수를 사용하여 계산된다.
* 미니배치 내에서 각 이미지는 병렬로 처리되므로, 요소의 인덱스는 batch\*l.outputs에서 시작한다.
* 그리드 셀은 2차원 형태이므로, 인덱스를 계산하기 위해 location을 그리드 셀의 너비(l.w)와 높이(l.h)로 나누어서 행과 열의 인덱스를 계산한다.
* 해당 속성의 인덱스와 그리드 셀의 인덱스를 곱하여 해당 속성의 첫 번째 요소의 인덱스를 계산하고, 그리드 셀의 인덱스를 더하여 해당 요소의 인덱스를 구한다.



## correct\_region\_boxes

```c
void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}
```

함수 이름: correct\_region\_boxes

입력:

* detection \*dets: detection 구조체 배열 포인터
* int n: detection 구조체 배열의 크기
* int w: 입력 이미지의 가로 크기
* int h: 입력 이미지의 세로 크기
* int netw: 신경망의 입력 이미지 가로 크기
* int neth: 신경망의 입력 이미지 세로 크기
* int relative: bounding box 좌표를 상대 좌표로 계산할지 절대 좌표로 계산할지 여부 (0: 절대 좌표, 1: 상대 좌표)

동작:

* 입력 이미지와 신경망 입력 이미지의 비율을 고려하여 bounding box 좌표를 수정하는 함수

설명:

* 이 함수는 YOLO (You Only Look Once) 객체 검출 알고리즘에서 사용하는 함수이다.
* 입력 이미지와 신경망 입력 이미지의 비율을 고려하여 bounding box 좌표를 수정한다.
* 입력 이미지와 신경망 입력 이미지의 가로, 세로 비율이 다르면 입력 이미지를 신경망 입력 이미지에 맞게 resize하고, 그 비율에 맞게 bounding box 좌표를 수정한다.
* 수정된 bounding box 좌표는 입력 이미지에 대한 상대 좌표 또는 절대 좌표로 계산할 수 있다.
* 상대 좌표로 계산하려면 relative 인자를 1로 설정하고, 절대 좌표로 계산하려면 relative 인자를 0으로 설정한다.



## get\_region\_detections

```c
void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets)
{
    int i,j,n,z;
    float *predictions = l.output;
    if (l.batch == 2) {
        float *flip = l.output + l.outputs;
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w/2; ++i) {
                for (n = 0; n < l.n; ++n) {
                    for(z = 0; z < l.classes + l.coords + 1; ++z){
                        int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                        int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                        float swap = flip[i1];
                        flip[i1] = flip[i2];
                        flip[i2] = swap;
                        if(z == 0){
                            flip[i1] = -flip[i1];
                            flip[i2] = -flip[i2];
                        }
                    }
                }
            }
        }
        for(i = 0; i < l.outputs; ++i){
            l.output[i] = (l.output[i] + flip[i])/2.;
        }
    }
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = n*l.w*l.h + i;
            for(j = 0; j < l.classes; ++j){
                dets[index].prob[j] = 0;
            }
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            int mask_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            float scale = l.background ? 1 : predictions[obj_index];
            dets[index].bbox = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w*l.h);
            dets[index].objectness = scale > thresh ? scale : 0;
            if(dets[index].mask){
                for(j = 0; j < l.coords - 4; ++j){
                    dets[index].mask[j] = l.output[mask_index + j*l.w*l.h];
                }
            }

            int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + !l.background);
            if(l.softmax_tree){

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0, l.w*l.h);
                if(map){
                    for(j = 0; j < 200; ++j){
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + map[j]);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                } else {
                    int j =  hierarchy_top_prediction(predictions + class_index, l.softmax_tree, tree_thresh, l.w*l.h);
                    dets[index].prob[j] = (scale > thresh) ? scale : 0;
                }
            } else {
                if(dets[index].objectness){
                    for(j = 0; j < l.classes; ++j){
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + j);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                }
            }
        }
    }
    correct_region_boxes(dets, l.w*l.h*l.n, w, h, netw, neth, relative);
}
```

함수 이름: get\_region\_detections

입력:

* layer l: YOLO 레이어 객체
* int w: 입력 이미지의 너비
* int h: 입력 이미지의 높이
* int netw: 네트워크 입력 이미지의 너비
* int neth: 네트워크 입력 이미지의 높이
* float thresh: 객체를 탐지하기 위한 threshold 값
* int \*map: softmax 트리 매핑 값
* float tree\_thresh: softmax 트리를 사용할 때 threshold 값
* int relative: 상대적인 좌표를 사용할 것인지 여부
* detection \*dets: 검출된 객체의 배열

동작:&#x20;

* 입력 이미지에서 객체를 탐지하고 detection 객체 배열에 결과를 저장한다. 입력 레이어에서 출력을 가져오고, 필요한 경우 레이어 출력을 뒤집는다. 그 후, 각 픽셀마다 bounding box와 클래스 확률을 계산하여 detection 객체 배열에 저장한다.
  * 레이어 출력을 가져온다.
  * 레이어 배치(batch) 수가 2인 경우, 레이어 출력을 뒤집는다.
  * 각 픽셀마다 bounding box와 클래스 확률을 계산하여 detection 객체 배열에 저장한다.
  * 상대적인 좌표를 사용할 경우, 좌표를 절대 좌표로 변환한다.

설명:

* layer l: YOLO 레이어 객체
* int w: 입력 이미지의 너비
* int h: 입력 이미지의 높이
* int netw: 네트워크 입력 이미지의 너비
* int neth: 네트워크 입력 이미지의 높이
* float thresh: 객체를 탐지하기 위한 threshold 값
* int \*map: softmax 트리 매핑 값. 이 값이 NULL이 아니면, softmax 트리가 사용된다.
* float tree\_thresh: softmax 트리를 사용할 때 threshold 값
* int relative: 상대적인 좌표를 사용할 것인지 여부. 1이면 상대적인 좌표를 사용하고, 0이면 절대 좌표를 사용한다.
* detection \*dets: 검출된 객체의 배열. 각 객체는 bounding box, 클래스 확률, objectness 및 mask 값을 가진다.
* float \*predictions = l.output: 레이어 출력을 가져온다.
* if (l.batch == 2): 레이어 배치(batch) 수가 2인 경우, 레이어 출력을 뒤집는다. 이것은 이미지 증강(augmentation)을 위한 것이다.
* for (i = 0; i < l.w\*l.h; ++i): 각 픽셀마다 detection 객체를 계산한다.
  * int row = i / l.w; int col = i % l.w: 현재 픽셀의 행과 열을 계산한다.
  * for(n = 0; n < l.n;



## zero\_objectness

```c
void zero_objectness(layer l)
{
    int i, n;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            l.output[obj_index] = 0;
        }
    }
}
```

함수 이름: zero\_objectness

입력:

* layer l: region layer

동작:

* region layer에서 objectness 값 중, bbox가 아닌 부분을 0으로 초기화한다.

설명:&#x20;

* 이 함수는 region layer에서 objectness 값 중 bbox가 아닌 부분을 0으로 초기화하는 역할을 한다.&#x20;
* region layer는 객체 검출을 위해 사용되는 layer 중 하나이며, 여러 개의 bounding box를 예측하고 클래스별 확률을 출력하는 역할을 한다.&#x20;
* 이 함수는 region layer에서 출력된 값 중, bbox가 아닌 부분의 objectness 값을 0으로 초기화한다.&#x20;
* 이는 객체가 없는 부분에 대한 확률 값을 0으로 설정하는 것이며, 이를 통해 객체 검출 정확도를 향상시킬 수 있다.&#x20;
* 이 함수는 bbox가 아닌 부분을 0으로 초기화하는 역할을 하며, 다른 함수에서 호출되어 사용된다.

