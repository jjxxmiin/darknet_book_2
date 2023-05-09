# box

## nms\_comparator

```c
int nms_comparator(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if(b.sort_class >= 0){
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}
```

함수 이름: nms\_comparator

입력:&#x20;

* void 포인터형 pa
* void 포인터형 pb

동작:&#x20;

* pa와 pb가 가리키는 detection 구조체 a와 b를 가져와, a와 b의 확률(prob)이나 개체(objectness)의 차이(diff)를 계산하여 오름차순으로 정렬한다.&#x20;
* sort\_class가 0보다 크거나 같은 경우에는 prob의 차이를, 그렇지 않은 경우에는 objectness의 차이를 이용하여 정렬한다.

설명:&#x20;

* Non-Maximum Suppression(NMS) 알고리즘을 적용하기 위한 comparator 함수이다.&#x20;
* NMS는 객체 탐지(Object Detection)에서 중복되는 박스(Box)를 제거하는 알고리즘이다.&#x20;
* NMS를 적용하기 전에, 객체 검출 모델이 예측한 여러 개의 박스들을 확률 등의 기준으로 내림차순으로 정렬하는데, 이때 사용되는 함수이다.&#x20;
* a와 b가 가리키는 detection 구조체의 요소 중 sort\_class번째 클래스에 대한 확률(prob) 혹은 개체(objectness)의 차이(diff)를 계산하여 정렬한다.&#x20;
* 함수 내에서 qsort() 함수와 함께 사용되어 정렬을 수행한다.



## do\_nms\_obj

```c
void do_nms_obj(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i){
        if(dets[i].objectness == 0){
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;

    for(i = 0; i < total; ++i){
        dets[i].sort_class = -1;
    }

    qsort(dets, total, sizeof(detection), nms_comparator);
    for(i = 0; i < total; ++i){
        if(dets[i].objectness == 0) continue;
        box a = dets[i].bbox;
        for(j = i+1; j < total; ++j){
            if(dets[j].objectness == 0) continue;
            box b = dets[j].bbox;
            if (box_iou(a, b) > thresh){
                dets[j].objectness = 0;
                for(k = 0; k < classes; ++k){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}
```

함수 이름: do\_nms\_obj

입력:

* detection \*dets: detection 구조체 배열 포인터
* int total: detection 배열의 총 원소 개수
* int classes: 분류(classification) 수
* float thresh: IOU(Intersection over Union) 임계값

동작:

* 입력으로 받은 detection 배열에서 objectness가 0인 detection을 배열의 끝으로 이동시키고 해당 원소를 배열에서 제거함
* 정렬되지 않은 detection 배열을 class의 확률값(prob) 또는 objectness 값(objectness)을 기준으로 내림차순으로 정렬
* 정렬된 detection 배열에서 IOU가 임계값(thresh) 이상인 detection 쌍을 찾아서 낮은 objectness 값과 해당 쌍의 모든 class의 확률값을 0으로 설정함

설명:

* non-maximum suppression(NMS) 알고리즘을 수행하는 함수로, 입력으로 받은 detection 배열에서 겹치는 박스를 제거하는 역할을 함
* 입력으로 받은 detection 구조체 배열 포인터(dets)는 박스(bounding box)의 위치, 크기, confidence(확률), class 확률 등을 포함하는 구조체 배열임
* 함수 내부에서는 객체를 검출한 detection 구조체 배열(dets)을 objectness 값으로 내림차순으로 정렬하고, 겹치는 박스를 제거함
* 객체를 검출한 detection 배열에서 objectness가 0인 detection을 제거함으로써, 겹치는 박스 중에서 objectness 값이 낮은 박스를 제거함
* objectness가 0이 되면 해당 박스는 객체가 아니라고 판단하고, 그에 해당하는 class 확률값을 0으로 설정함
* 이러한 처리를 거친 detection 배열은 최종적으로 겹치는 박스를 제거한 후의 객체 검출 결과를 나타냄



## do\_nms\_sort

```c
void do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i){
        if(dets[i].objectness == 0){
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator);
        for(i = 0; i < total; ++i){
            if(dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for(j = i+1; j < total; ++j){
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}
```

함수 이름: do\_nms\_sort

입력:

* detection \*dets: 각 클래스의 detection 결과를 담고 있는 배열
* int total: 총 detection 개수
* int classes: 클래스 개수
* float thresh: IoU 임계값

동작:

* 입력으로 받은 detection 배열에서 objectness 값이 0인 detection은 제거한다.
* classes 개수만큼 반복하며, 각 클래스에 대한 detection 결과를 오름차순으로 정렬한다.
* 정렬된 detection 결과 중, 확률값이 0인 detection은 제외한다.
* 이후, 각 detection 결과의 bounding box와 다른 detection 결과의 bounding box의 IoU를 계산하여, 임계값(thresh)보다 큰 경우 해당 detection의 확률값을 0으로 만든다.

설명:&#x20;

* 이 함수는 Object Detection에서 NMS(Non-Maximum Suppression)을 수행하는 함수로, 클래스별로 detection 결과를 정렬하여 IoU 임계값을 넘는 detection 결과를 제거한다.&#x20;
* 이를 통해, 겹치는 박스를 제거하고 최종 detection 결과를 얻게 된다.



## float\_to\_box

```c
box float_to_box(float *f, int stride)
{
    box b = {0};
    b.x = f[0];
    b.y = f[1*stride];
    b.w = f[2*stride];
    b.h = f[3*stride];
    return b;
}
```

함수 이름: float\_to\_box

입력:

* float \*f: 4개의 값으로 이루어진 1차원 float 배열 포인터
* int stride: 배열에서 값 사이의 간격

동작:

* 주어진 float 배열 포인터를 사용하여 새로운 box 구조체를 만들고 반환합니다.
* box 구조체는 x, y, w, h 값을 갖습니다.
* x, y, w, h 값은 각각 배열의 첫 번째, stride 번째, 2_stride 번째, 3_stride 번째 값으로 설정됩니다.

설명:

* 이 함수는 주로 YOLO와 같은 객체 검출 모델에서 사용됩니다.
* 객체 검출 모델은 bounding box를 출력값으로 반환하는데, 이 bounding box는 (x,y,w,h) 형태의 4개의 값을 가집니다.
* 그러나 이러한 출력값은 float 배열의 형태로 반환되기 때문에, 이를 적절한 형태의 구조체로 변환해주는 함수가 필요합니다.
* float\_to\_box 함수는 이러한 변환을 수행하는 함수 중 하나입니다.



## overlap

```c
float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}
```

함수 이름: overlap

입력:&#x20;

* 네 개의 실수(x1, w1, x2, w2)를 인자로 받습니다.

동작:&#x20;

* 두 개의 박스가 겹치는 영역의 너비를 계산합니다.&#x20;
* 박스의 중심 좌표와 폭을 이용하여 왼쪽 끝과 오른쪽 끝을 계산한 뒤 겹치는 영역의 너비를 반환합니다.

설명:&#x20;

* Object Detection에서 NMS(Non-Maximum Suppression) 과정에서 사용됩니다.&#x20;
* 박스가 서로 얼마나 겹치는지를 계산하여 IoU(Intersection over Union) 값을 구합니다.&#x20;
* 겹치는 영역이 작을수록 IoU 값은 작아지며, 일정 값을 기준으로 IoU 값이 이하인 박스들을 제거하여 중복 탐지를 방지합니다.



## box\_intersection

```c
float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}
```

함수 이름: box\_intersection

입력:&#x20;

* box a, box b (bounding box 정보를 나타내는 구조체)

동작:&#x20;

* 두 개의 bounding box a와 b가 주어졌을 때, 이들의 교차 면적을 계산하는 함수이다.

설명:&#x20;

* overlap 함수를 이용하여 두 bounding box의 x, y 좌표에 대한 교차 부분의 길이를 계산한 후, 이들 길이를 곱하여 교차 면적을 계산한다.&#x20;
* 만약 교차하는 부분이 없다면 면적은 0이 된다.



## box\_union

```c
float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}
```

함수 이름: box\_union

입력:&#x20;

* 두 개의 box(a, b)

동작:&#x20;

* 입력으로 받은 두 개의 box의 union을 계산하여 반환한다. union은 두 box가 차지하는 면적을 합친 값이다.

설명:&#x20;

* 입력으로 받은 두 개의 box a, b에 대해서, 각 box의 면적(w\*h)을 더한 후에 box\_intersection() 함수를 호출하여 교차하는 영역의 면적(i)을 구한다.&#x20;
* 그리고 두 box의 면적의 합에서 교차하는 영역(i)을 뺀 값을 반환한다.&#x20;
* 이 값이 입력으로 받은 두 box의 union 값이 된다.



## box\_iou

```c
float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}
```

* IOU를 계산합니다.

## box\_rmse

```c
float box_rmse(box a, box b)
{
    return sqrt(pow(a.x-b.x, 2) +
                pow(a.y-b.y, 2) +
                pow(a.w-b.w, 2) +
                pow(a.h-b.h, 2));
}
```

함수 이름: box\_rmse

입력:&#x20;

* box 구조체인 a와 b

동작:&#x20;

* 두 개의 box 구조체 a와 b의 좌표값을 이용하여 root mean square error(RMSE)을 계산한다.

설명:&#x20;

* box 구조체는 바운딩 박스를 나타내는 구조체이다. 이 함수는 두 개의 box 구조체 a와 b가 주어졌을 때, 각 좌표값(x, y, w, h)을 이용하여 RMSE를 계산한다.&#x20;
* RMSE는 예측값과 실제값의 차이를 계산할 때 많이 사용되며, 차이의 제곱을 평균내고 다시 루트를 씌운 값이다.&#x20;
* 따라서 이 함수에서는 각 좌표값의 차이를 제곱하여 평균을 내고 다시 루트를 취한 값을 반환한다.



## derivative

```c
dbox derivative(box a, box b)
{
    dbox d;
    d.dx = 0;
    d.dw = 0;
    float l1 = a.x - a.w/2;
    float l2 = b.x - b.w/2;
    if (l1 > l2){
        d.dx -= 1;
        d.dw += .5;
    }
    float r1 = a.x + a.w/2;
    float r2 = b.x + b.w/2;
    if(r1 < r2){
        d.dx += 1;
        d.dw += .5;
    }
    if (l1 > r2) {
        d.dx = -1;
        d.dw = 0;
    }
    if (r1 < l2){
        d.dx = 1;
        d.dw = 0;
    }

    d.dy = 0;
    d.dh = 0;
    float t1 = a.y - a.h/2;
    float t2 = b.y - b.h/2;
    if (t1 > t2){
        d.dy -= 1;
        d.dh += .5;
    }
    float b1 = a.y + a.h/2;
    float b2 = b.y + b.h/2;
    if(b1 < b2){
        d.dy += 1;
        d.dh += .5;
    }
    if (t1 > b2) {
        d.dy = -1;
        d.dh = 0;
    }
    if (b1 < t2){
        d.dy = 1;
        d.dh = 0;
    }
    return d;
}
```

함수 이름: dbox derivative(box a, box b)

입력:&#x20;

* 박스 a와 박스 b

동작:&#x20;

* 입력된 두 상자의 교차 영역(intersection)에 대한 미분값(도함수)을 계산합니다.

설명:&#x20;

* 상자 a와 b의 교차 영역에 대한 미분값(도함수)을 계산하기 위해 입력된 두 상자의 위치, 너비, 높이를 이용하여 l1, l2, r1, r2, t1, t2, b1, b2 변수를 계산합니다.
* l1, l2는 상자 a와 b의 왼쪽 끝 좌표, r1, r2는 상자 a와 b의 오른쪽 끝 좌표, t1, t2는 상자 a와 b의 위쪽 끝 좌표, b1, b2는 상자 a와 b의 아래쪽 끝 좌표입니다.
* 계산된 변수를 이용하여 교차 영역에 대한 미분값(도함수)을 계산합니다. 상자 a와 b의 교차 영역이 완전히 겹치지 않는 경우, 미분값은 0이 됩니다.
* 함수는 교차 영역에 대한 미분값(도함수)을 dbox 구조체에 저장하여 반환합니다. 반환된 미분값은 dintersect 함수에서 교차 영역의 너비와 높이를 이용하여 계산된 교차 영역에 대한 미분값을 계산하는데 사용됩니다.



## dintersect

```c
dbox dintersect(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    dbox dover = derivative(a, b);
    dbox di;

    di.dw = dover.dw*h;
    di.dx = dover.dx*h;
    di.dh = dover.dh*w;
    di.dy = dover.dy*w;

    return di;
}
```

함수 이름: dintersect

입력:&#x20;

* 박스 a와 박스 b

동작:&#x20;

* 입력된 두 상자의 교차 영역(intersection)을 계산하고, 교차 영역의 너비와 높이를 이용하여 미분값(도함수)을 계산합니다.

설명:&#x20;

* 입력된 두 상자의 교차 영역을 계산하기 위해 overlap 함수를 호출합니다. overlap 함수는 두 개의 선분이 겹치는 길이(너비 또는 높이)를 계산해주는 함수입니다. overlap 함수를 이용하여 교차 영역의 너비(w)와 높이(h)를 구하고, 상자 a와 b의 교차 영역에 대한 미분값(도함수)을 계산하는 derivative 함수를 호출합니다.
* derivative 함수는 상자 a와 b의 교차 영역에 대한 미분값(도함수)을 계산하는 함수로, 입력된 두 상자의 위치, 너비, 높이를 이용하여 계산합니다. 계산된 미분값을 이용하여 상자 a와 b의 교차 영역에 대한 미분값을 계산하고, 이 값을 dbox 구조체에 저장하여 반환합니다.
* 반환되는 dbox 구조체에는 교차 영역에 대한 너비와 높이에 대한 미분값이 저장되어 있습니다. 이 값을 이용하여 역전파(backpropagation)를 수행하거나, 최적화(optimization) 알고리즘에서 미분값을 이용하여 경사하강법(gradient descent) 등을 수행할 수 있습니다.



## dunion

```c
dbox dunion(box a, box b)
{
    dbox du;

    dbox di = dintersect(a, b);
    du.dw = a.h - di.dw;
    du.dh = a.w - di.dh;
    du.dx = -di.dx;
    du.dy = -di.dy;

    return du;
}
```

함수 이름: dunion

입력: 박스 a와 박스 b

동작:&#x20;

* 박스 a와 박스 b의 유니온을 계산하여, 두 박스를 모두 포함하는 가장 작은 박스를 반환합니다.

설명:&#x20;

* 입력으로 주어진 두 박스 a, b에 대해서, 두 박스가 공통으로 가지는 영역의 넓이를 계산하고, 이를 이용하여 두 박스를 모두 포함하는 가장 작은 박스의 넓이를 구합니다.&#x20;
* 이후, du 구조체에 이 작은 박스의 폭과 높이의 차이, 그리고 중심 좌표의 차이를 저장하여 반환합니다.&#x20;
* 이때, dintersect 함수를 이용하여 두 박스의 인터섹션을 계산합니다.



## test

```c
void test_dunion()
{
    box a = {0, 0, 1, 1};
    box dxa= {0+.0001, 0, 1, 1};
    box dya= {0, 0+.0001, 1, 1};
    box dwa= {0, 0, 1+.0001, 1};
    box dha= {0, 0, 1, 1+.0001};

    box b = {.5, .5, .2, .2};
    dbox di = dunion(a,b);
    printf("Union: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter =  box_union(a, b);
    float xinter = box_union(dxa, b);
    float yinter = box_union(dya, b);
    float winter = box_union(dwa, b);
    float hinter = box_union(dha, b);
    xinter = (xinter - inter)/(.0001);
    yinter = (yinter - inter)/(.0001);
    winter = (winter - inter)/(.0001);
    hinter = (hinter - inter)/(.0001);
    printf("Union Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}

void test_dintersect()
{
    box a = {0, 0, 1, 1};
    box dxa= {0+.0001, 0, 1, 1};
    box dya= {0, 0+.0001, 1, 1};
    box dwa= {0, 0, 1+.0001, 1};
    box dha= {0, 0, 1, 1+.0001};

    box b = {.5, .5, .2, .2};
    dbox di = dintersect(a,b);
    printf("Inter: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter =  box_intersection(a, b);
    float xinter = box_intersection(dxa, b);
    float yinter = box_intersection(dya, b);
    float winter = box_intersection(dwa, b);
    float hinter = box_intersection(dha, b);
    xinter = (xinter - inter)/(.0001);
    yinter = (yinter - inter)/(.0001);
    winter = (winter - inter)/(.0001);
    hinter = (hinter - inter)/(.0001);
    printf("Inter Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}

void test_box()
{
    test_dintersect();
    test_dunion();
    box a = {0, 0, 1, 1};
    box dxa= {0+.00001, 0, 1, 1};
    box dya= {0, 0+.00001, 1, 1};
    box dwa= {0, 0, 1+.00001, 1};
    box dha= {0, 0, 1, 1+.00001};

    box b = {.5, 0, .2, .2};

    float iou = box_iou(a,b);
    iou = (1-iou)*(1-iou);
    printf("%f\n", iou);
    dbox d = diou(a, b);
    printf("%f %f %f %f\n", d.dx, d.dy, d.dw, d.dh);

    float xiou = box_iou(dxa, b);
    float yiou = box_iou(dya, b);
    float wiou = box_iou(dwa, b);
    float hiou = box_iou(dha, b);
    xiou = ((1-xiou)*(1-xiou) - iou)/(.00001);
    yiou = ((1-yiou)*(1-yiou) - iou)/(.00001);
    wiou = ((1-wiou)*(1-wiou) - iou)/(.00001);
    hiou = ((1-hiou)*(1-hiou) - iou)/(.00001);
    printf("manual %f %f %f %f\n", xiou, yiou, wiou, hiou);
}
```

```
Inter: 0.100000 0.100000 0.050000 0.050000
Inter Manual 0.100015 0.100015 0.050012 0.050012

Union: -0.100000 -0.100000 0.950000 0.950000
Union Manual -0.100136 -0.100136 0.950098 0.950098

IOU : 0.500000 0.000000 -0.800000 -0.800000
IOU manual : -0.390865 0.001386 -0.156757 0.039369
```

실행 결과 입니다. 뭔가 Inter, Union은 미분 값이 들어 맞는데 IOU가 다른 것 같습니다. 문제를 찾아보니 test\_iou와 diou는 darknet에서 동작하지 않는 코드라고 합니다. 그래서 diou에 낭비 코드가 있나봅니다.

* 출처 : [https://github.com/pjreddie/darknet/issues/199](https://github.com/pjreddie/darknet/issues/199)
* `|| 1`이 부분을 지우고 다시 실행하면 아래와 같이 나옵니다.

```
IOU: 0.392006 0.000000 0.158310 -0.037693
IOU manual -0.390865 0.001386 -0.156757 0.039369
```

1. 함수 이름: test\_dunion()

* 입력: 없음
* 동작: dunion() 함수를 사용하여 두 박스의 합집합을 계산하고, 수동으로 계산한 값과 비교하여 정확성을 검증합니다.
* 설명: 두 개의 박스를 생성하고, dunion() 함수를 사용하여 합집합을 계산합니다. 이후, 수동으로 생성한 값과 비교하여 정확성을 검증합니다.

2. 함수 이름: test\_dintersect()

* 입력: 없음
* 동작: dintersect() 함수를 사용하여 두 박스의 교집합을 계산하고, 수동으로 계산한 값과 비교하여 정확성을 검증합니다.
* 설명: 두 개의 박스를 생성하고, dintersect() 함수를 사용하여 교집합을 계산합니다. 이후, 수동으로 생성한 값과 비교하여 정확성을 검증합니다.

3. 함수 이름: test\_box()

* 입력: 없음
* 동작: box\_iou() 함수와 diou() 함수를 사용하여 두 박스의 IoU(Intersection over Union) 값을 계산하고, 수동으로 계산한 값과 비교하여 정확성을 검증합니다.
* 설명: 두 개의 박스를 생성하고, box\_iou() 함수와 diou() 함수를 사용하여 IoU 값을 계산합니다. 이후, 수동으로 생성한 값과 비교하여 정확성을 검증합니다. 또한, 앞서 정의한 test\_dunion() 함수와 test\_dintersect() 함수를 호출하여 해당 함수들의 정확성도 검증합니다.



## diou

```c
dbox diou(box a, box b)
{
    float u = box_union(a,b);
    float i = box_intersection(a,b);
    dbox di = dintersect(a,b);
    dbox du = dunion(a,b);
    dbox dd = {0,0,0,0};

    if(i <= 0 || 1) {         // 이상한 낭비 코드
        dd.dx = b.x - a.x;
        dd.dy = b.y - a.y;
        dd.dw = b.w - a.w;
        dd.dh = b.h - a.h;
        return dd;
    }

    dd.dx = 2*pow((1-(i/u)),1)*(di.dx*u - du.dx*i)/(u*u);
    dd.dy = 2*pow((1-(i/u)),1)*(di.dy*u - du.dy*i)/(u*u);
    dd.dw = 2*pow((1-(i/u)),1)*(di.dw*u - du.dw*i)/(u*u);
    dd.dh = 2*pow((1-(i/u)),1)*(di.dh*u - du.dh*i)/(u*u);
    return dd;
}
```

함수 이름: dbox\_diou

입력:&#x20;

* 박스 a와 박스 b

동작:&#x20;

* 두 상자의 교차 부피(dbox\_intersection)와 합집합 부피(dbox\_union)를 계산한 후, 교차한 영역의 미분(dbox)을 구합니다.&#x20;
* 교차 부피가 0 이하거나 1 이상인 경우, 두 상자의 위치 차이를 미분한 값을 반환합니다. 그렇지 않으면, 두 상자의 교차한 영역의 비율에 따라 미분 값을 계산합니다.

설명:&#x20;

* 이 함수는 두 개의 상자의 위치와 크기 차이에 대한 미분 값을 계산합니다.&#x20;
* 상자의 위치와 크기 차이가 작은 경우, 이 미분 값은 상자를 이동하거나 크기를 조정할 때 필요한 정보가 될 수 있습니다.&#x20;
* 이 함수는 YOLOv3와 같은 딥러닝 모델에서 물체 검출과 같은 작업을 수행할 때 사용됩니다.



## do\_nms

```c
void do_nms(box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    for(i = 0; i < total; ++i){
        int any = 0;
        for(k = 0; k < classes; ++k) any = any || (probs[i][k] > 0);
        if(!any) {
            continue;
        }
        for(j = i+1; j < total; ++j){
            if (box_iou(boxes[i], boxes[j]) > thresh){
                for(k = 0; k < classes; ++k){
                    if (probs[i][k] < probs[j][k]) probs[i][k] = 0;
                    else probs[j][k] = 0;
                }
            }
        }
    }
}
```

함수 이름: do\_nms

입력:

* boxes: 박스 좌표 정보를 담은 배열
* probs: 박스에 대한 각 클래스의 확률 정보를 담은 2차원 배열
* total: 전체 박스의 개수
* classes: 분류할 클래스의 개수
* thresh: IoU 임계값

동작:

* 박스들을 하나씩 검사하면서 해당 박스가 어떤 클래스에도 속하지 않으면 다음 박스를 검사한다.
* 속한 클래스가 존재하는 박스들 중 IoU 값이 임계값(thresh)보다 크다면, 확률값이 작은 박스는 해당 클래스에 대한 확률값을 0으로 만든다.

설명:

* Non-maximum suppression(NMS) 알고리즘을 구현한 함수로, 같은 객체를 나타내는 중복된 박스를 제거하는 역할을 한다.
* 입력으로 받은 박스와 클래스별 확률 정보를 이용해 IoU 값을 계산하고, IoU 값이 임계값을 넘는 박스 중 확률값이 작은 박스는 제거한다.
* 이 과정을 거쳐 최종적으로 남은 박스들은 중복이 제거된 후보 객체들이 된다.



## encode\_box

```c
box encode_box(box b, box anchor)
{
    box encode;
    encode.x = (b.x - anchor.x) / anchor.w;
    encode.y = (b.y - anchor.y) / anchor.h;
    encode.w = log2(b.w / anchor.w);
    encode.h = log2(b.h / anchor.h);
    return encode;
}
```

함수 이름: encode\_box

입력:&#x20;

* 박스 a와 박스 b

동작:&#x20;

* 입력으로 받은 box b와 anchor를 이용하여 bounding box 정보를 인코딩합니다.

설명:&#x20;

* encode\_box 함수는 bounding box regression을 위해 입력으로 받은 bounding box 정보와 anchor 정보를 이용하여 bounding box 정보를 인코딩합니다. 입력으로 받은 b의 (x, y, w, h) 값과 anchor의 (x, y, w, h) 값을 이용하여 새로운 bounding box 정보를 계산합니다. 새로운 bounding box 정보 encode의 (x, y, w, h) 값은 다음과 같이 계산됩니다.
  * encode.x = (b.x - anchor.x) / anchor.w
  * encode.y = (b.y - anchor.y) / anchor.h
  * encode.w = log2(b.w / anchor.w)
  * encode.h = log2(b.h / anchor.h)
* 이렇게 인코딩된 bounding box 정보는 모델 학습 시 bounding box regression을 수행하는 데 사용됩니다.



## decode\_box

```c
box decode_box(box b, box anchor)
{
    box decode;
    decode.x = b.x * anchor.w + anchor.x;
    decode.y = b.y * anchor.h + anchor.y;
    decode.w = pow(2., b.w) * anchor.w;
    decode.h = pow(2., b.h) * anchor.h;
    return decode;
}
```

함수 이름: decode\_box

입력:&#x20;

* 박스 a와 박스 b

동작:&#x20;

* anchor를 기준으로 b에서 디코딩된 box를 반환합니다. x, y, w, h 값을 새로 계산하여 반환합니다.

설명:&#x20;

* 디코딩은 인코딩된 상자와 앵커 상자를 기준으로 예측된 상자의 좌표와 크기를 복원하는 작업입니다.&#x20;
* 이 함수는 입력으로 받은 인코딩된 box b와 앵커 상자 anchor를 이용하여 예측된 상자의 실제 값을 계산하여 box 구조체로 반환합니다.&#x20;
* x, y, w, h 값을 계산하기 위해 인코딩된 값 b와 anchor 값을 사용하고 있습니다. 반환된 값은 디코딩된 상자의 좌표와 크기를 나타냅니다.

