# yolo\_layer

### forward\_yolo\_layer

```c
void forward_yolo_layer(const layer l, network net)
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h);
                    float best_iou = 0;
                    int best_t = 0;
                    for(t = 0; t < l.max_boxes; ++t){
                        box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = 0 - l.output[obj_index];
                    if (best_iou > l.ignore_thresh) {
                        l.delta[obj_index] = 0;
                    }
                    if (best_iou > l.truth_thresh) {
                        l.delta[obj_index] = 1 - l.output[obj_index];

                        int class = net.truth[best_t*(4 + 1) + b*l.truths + 4];
                        if (l.map) class = l.map[class];
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0);
                        box truth = float_to_box(net.truth + best_t*(4 + 1) + b*l.truths, 1);
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
                    }
                }
            }
        }
        for(t = 0; t < l.max_boxes; ++t){
            box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for(n = 0; n < l.total; ++n){
                box pred = {0};
                pred.w = l.biases[2*n]/net.w;
                pred.h = l.biases[2*n+1]/net.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }

            int mask_n = int_index(l.mask, best_n, l.n);
            if(mask_n >= 0){
                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);

                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                avg_obj += l.output[obj_index];
                l.delta[obj_index] = 1 - l.output[obj_index];

                int class = net.truth[t*(4 + 1) + b*l.truths + 4];
                if (l.map) class = l.map[class];
                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat);

                ++count;
                ++class_count;
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);
}
```

함수 이름: forward\_yolo\_layer

입력:

* l : YOLO 레이어 객체
* net : 네트워크 객체

동작:

* YOLO 레이어의 forward 연산을 수행하고, 검출된 bounding box와 확률값 등을 반환한다.

설명:

* YOLO(You Only Look Once) 알고리즘은 이미지 내 객체 검출(object detection) 알고리즘 중 하나로, 실시간 객체 검출에 적합한 알고리즘이다.
* forward\_yolo\_layer 함수는 YOLO 레이어의 forward 연산을 수행하고, 검출된 bounding box와 확률값 등을 반환하는 함수이다.
* 함수의 입력으로는 YOLO 레이어 객체와 네트워크 객체가 전달된다.
* YOLO 레이어 객체는 YOLO 레이어의 설정 정보를 담고 있으며, 네트워크 객체는 YOLO 모델의 네트워크 구조를 담고 있다.
* forward 연산을 수행하면, 입력 이미지 내에서 객체의 위치를 예측하고 bounding box를 그려낸다. 이때, 예측된 bounding box는 anchor box와 조합되어 계산된다.
* 함수는 검출된 bounding box의 좌표, 클래스 정보, 확률값 등을 반환한다.



### backward\_yolo\_layer

```c
void backward_yolo_layer(const layer l, network net)
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}
```

함수 이름: backward\_yolo\_layer 입력:

* l: yolo\_layer 형식의 구조체 변수
* net: network 형식의 구조체 변수

동작:

* YOLO (You Only Look Once) layer의 backpropagation을 수행하는 함수입니다.
* l.delta에 저장된 YOLO layer의 gradient를 사용하여, l.batch \* l.inputs 크기의 벡터를 계산합니다.
* 계산된 벡터를 net.delta에 더합니다.

설명:

* YOLO layer는 object detection에 자주 사용되는 layer 중 하나입니다.
* 이 함수는 YOLO layer의 backpropagation을 위해 사용됩니다.
* l.delta는 YOLO layer에서 각 grid cell에서 예측한 bounding box들과 실제 bounding box의 차이를 나타내는 값입니다.
* 이 함수는 l.delta를 사용하여, 이전 layer의 gradient를 계산합니다.
* 계산된 gradient는 net.delta에 저장되며, 이전 layer의 gradient 계산에 사용됩니다.



### make\_yolo\_layer

```c
layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)
{
    int i;
    layer l = {0};
    l.type = YOLO;

    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(total*2, sizeof(float));
    if(mask) l.mask = mask;
    else{
        l.mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    l.truths = 90*(4 + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;

    fprintf(stderr, "yolo\n");
    srand(0);

    return l;
}
```

함수 이름: make\_yolo\_layer

입력:

* int batch: 한 번에 처리할 이미지의 개수
* int w: 입력 이미지의 가로 길이
* int h: 입력 이미지의 세로 길이
* int n: 각 층에서 사용될 앵커 박스(anchor box)의 개수
* int total: 모든 층에서 사용될 앵커 박스(anchor box)의 총 개수
* int \*mask: 사용할 앵커 박스(anchor box)의 인덱스를 나타내는 배열. NULL인 경우 모든 앵커 박스(anchor box)를 사용
* int classes: 분류할 클래스의 개수

동작:

* 입력으로 받은 batch, w, h, n, total, mask, classes 값을 이용해 YOLO 레이어를 생성한다.
* 생성된 레이어에 대한 초기화를 수행한다.
* 생성된 레이어의 forward와 backward 함수를 forward\_yolo\_layer와 backward\_yolo\_layer로 설정한다.
* 생성된 레이어를 반환한다.

설명:

* YOLO 레이어를 생성하고 초기화하는 함수이다.
* YOLO 레이어는 이미지 분류 및 객체 검출에 사용된다.
* batch, w, h, n, total, mask, classes 값을 입력으로 받아 해당하는 YOLO 레이어를 생성하고 초기화한다.
* biases, bias\_updates, output, delta, cost 등의 변수들을 초기화한다.
* 생성된 레이어의 forward와 backward 함수를 forward\_yolo\_layer와 backward\_yolo\_layer로 설정한다.
* 생성된 레이어를 반환한다.



### correct\_yolo\_boxes

```c
void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
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

함수 이름: correct\_yolo\_boxes

입력:

* dets: detection 구조체 배열 포인터
* n: detection 구조체 배열의 크기
* w: 입력 이미지의 너비
* h: 입력 이미지의 높이
* netw: 네트워크 입력 이미지의 너비
* neth: 네트워크 입력 이미지의 높이
* relative: 상대적인 좌표계를 사용할지 여부를 나타내는 정수값

동작:&#x20;

* YOLO 레이어에서 출력된 bounding box들을 입력 이미지 좌표계에서 상대적인 좌표계에서 절대적인 좌표계로 변환시키는 함수이다.

설명:&#x20;

* 입력 이미지와 네트워크 입력 이미지의 크기를 이용하여 bounding box 좌표를 변환시키는 과정을 수행한다.&#x20;
* 네트워크 입력 이미지의 크기와 입력 이미지의 크기를 비교하여 더 긴 쪽에 맞추어 비율을 유지하며, 좌표값을 변환한다.&#x20;
* relative 값이 0일 경우, 변환된 좌표를 입력 이미지 좌표계에서 절대적인 좌표계로 변환한다.



### yolo\_num\_detections

```c
int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}
```

함수 이름: yolo\_num\_detections

입력:

* layer l: YOLO 레이어 정보
* float thresh: Objectness score threshold 값

동작:

* 입력으로 들어온 YOLO 레이어에서, Objectness score 값이 threshold(thresh)보다 큰 detection의 개수를 세는 함수이다.
* Objectness score 값이란, 해당 bounding box 안에 물체가 있는지 여부를 나타내는 값으로, YOLO에서는 이 값이 threshold보다 작으면 해당 detection이 무시된다.

설명:

* for문을 두번 돌며, l.w\*l.h 크기의 그리드 내에서 l.n 개의 anchor box를 가진 detection들을 검사한다.
* obj\_index 를 계산하여 해당 detection의 Objectness score 값이 threshold보다 크면 count 값을 증가시킨다.
* 최종적으로 count 값을 반환한다.



### avg\_flipped\_yolo

```c
void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
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
```

함수 이름: avg\_flipped\_yolo

입력:

* layer l: YOLO 레이어 구조체

동작:

* YOLO 레이어의 출력값을 수평으로 대칭(flip)한 값과 평균을 내어 새로운 출력값으로 대체하는 함수
* 입력으로 들어온 layer l 구조체의 output 배열과 output + outputs 배열(수평 대칭한 값)을 사용함
* 수평 대칭한 값과의 평균값을 구하여 `output` 배열의 각 원소에 저장함

설명:

* YOLO 레이어에서는 입력 이미지를 여러 그리드(cell)로 분할하고, 각 그리드마다 bounding box들의 정보를 예측함
* 이때 bounding box들의 위치와 크기는 입력 이미지에 대한 상대적인 값으로 저장되어 있으므로, 이미지가 수평 대칭되면 예측한 bounding box들도 수평 대칭됨
* 따라서 이미지를 수평 대칭으로 뒤집어 예측한 bounding box들도 대칭시키는 작업이 필요함
* 이 함수는 입력으로 들어온 layer l 구조체의 output 배열과 output + outputs 배열(수평 대칭한 값)을 사용하여, 각 bounding box에 대해 수평 대칭된 값을 계산한 후, 원래 값과의 평균을 새로운 값으로 대체함
* bounding box의 위치와 크기는 x, y, w, h 4가지 값으로 표현됨. 이 함수는 수평 대칭한 값을 계산할 때, x 값만 부호를 바꿔주고 나머지 값은 그대로 유지함
* output 배열의 각 원소에는 최종 예측값이 저장되어 있으므로, 이 함수가 실행된 이후에는 대칭된 이미지에 대한 예측값과 원래 이미지에 대한 예측값의 평균값이 output 배열에 저장됨



### get\_yolo\_detections

```c
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j = 0; j < l.classes; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}
```

함수 이름: get\_yolo\_detections

입력:

* layer l: YOLO 레이어
* int w: 입력 이미지의 가로 크기
* int h: 입력 이미지의 세로 크기
* int netw: YOLO 모델 입력 이미지 가로 크기
* int neth: YOLO 모델 입력 이미지 세로 크기
* float thresh: 객체 탐지에 사용되는 임계값
* int \*map: 클래스 매핑 정보를 담고 있는 포인터
* int relative: 출력 결과를 상대적인 좌표로 변환할지 여부를 결정하는 변수 (1 or 0)
* detection \*dets: 객체 탐지 결과를 저장할 detection 구조체 배열

동작:

* YOLO 레이어의 출력을 사용하여 객체 탐지를 수행한다.
* 각각의 바운딩 박스에 대해, 해당 객체의 objectness 값이 임계값보다 큰 경우에만 탐지 결과로 인정한다.
* 탐지된 객체의 바운딩 박스 정보와 클래스별 확률 값을 detection 구조체에 저장한다.
* 상대적인 좌표를 사용하도록 relative 변수가 1로 설정된 경우, 바운딩 박스 좌표와 크기를 입력 이미지 크기로 변환한다.
* 탐지된 객체 수를 반환한다.

설명:&#x20;

* 이 함수는 YOLO 레이어의 출력을 사용하여 입력 이미지에서 객체를 탐지하는 함수이다.&#x20;
* 입력으로는 YOLO 레이어와 입력 이미지의 크기, 탐지에 사용되는 임계값, 클래스 매핑 정보, 출력 결과를 상대적인 좌표로 변환할지 여부 등이 들어온다.&#x20;
* 이 함수는 입력 이미지에서 객체를 탐지한 결과를 detection 구조체 배열로 반환한다.&#x20;
* 반환된 detection 구조체에는 각 객체의 바운딩 박스 정보와 클래스별 확률 값이 저장되어 있다.

