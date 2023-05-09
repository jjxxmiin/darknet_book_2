# detection\_layer

## forward\_detection\_layer

```c
void forward_detection_layer(const detection_layer l, network net)
{
    int locations = l.side*l.side;
    int i,j;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    //if(l.reorg) reorg(l.output, l.w*l.h, size*l.n, l.batch, 1);
    int b;
    if (l.softmax){
        for(b = 0; b < l.batch; ++b){
            int index = b*l.inputs;
            for (i = 0; i < locations; ++i) {
                int offset = i*l.classes;
                softmax(l.output + index + offset, l.classes, 1, 1,
                        l.output + index + offset);
            }
        }
    }
    if(net.train){
        float avg_iou = 0;
        float avg_cat = 0;
        float avg_allcat = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        int count = 0;
        *(l.cost) = 0;
        int size = l.inputs * l.batch;
        memset(l.delta, 0, size * sizeof(float));
        for (b = 0; b < l.batch; ++b){
            int index = b*l.inputs;
            for (i = 0; i < locations; ++i) {
                int truth_index = (b*locations + i)*(1+l.coords+l.classes);
                int is_obj = net.truth[truth_index];
                for (j = 0; j < l.n; ++j) {
                    int p_index = index + locations*l.classes + i*l.n + j;
                    l.delta[p_index] = l.noobject_scale*(0 - l.output[p_index]);
                    *(l.cost) += l.noobject_scale*pow(l.output[p_index], 2);
                    avg_anyobj += l.output[p_index];
                }

                int best_index = -1;
                float best_iou = 0;
                float best_rmse = 20;

                if (!is_obj){
                    continue;
                }

                int class_index = index + i*l.classes;
                for(j = 0; j < l.classes; ++j) {
                    l.delta[class_index+j] = l.class_scale * (net.truth[truth_index+1+j] - l.output[class_index+j]);
                    *(l.cost) += l.class_scale * pow(net.truth[truth_index+1+j] - l.output[class_index+j], 2);
                    if(net.truth[truth_index + 1 + j]) avg_cat += l.output[class_index+j];
                    avg_allcat += l.output[class_index+j];
                }

                box truth = float_to_box(net.truth + truth_index + 1 + l.classes, 1);
                truth.x /= l.side;
                truth.y /= l.side;

                for(j = 0; j < l.n; ++j){
                    int box_index = index + locations*(l.classes + l.n) + (i*l.n + j) * l.coords;
                    box out = float_to_box(l.output + box_index, 1);
                    out.x /= l.side;
                    out.y /= l.side;

                    if (l.sqrt){
                        out.w = out.w*out.w;
                        out.h = out.h*out.h;
                    }

                    float iou  = box_iou(out, truth);
                    //iou = 0;
                    float rmse = box_rmse(out, truth);
                    if(best_iou > 0 || iou > 0){
                        if(iou > best_iou){
                            best_iou = iou;
                            best_index = j;
                        }
                    }else{
                        if(rmse < best_rmse){
                            best_rmse = rmse;
                            best_index = j;
                        }
                    }
                }

                if(l.forced){
                    if(truth.w*truth.h < .1){
                        best_index = 1;
                    }else{
                        best_index = 0;
                    }
                }
                if(l.random && *(net.seen) < 64000){
                    best_index = rand()%l.n;
                }

                int box_index = index + locations*(l.classes + l.n) + (i*l.n + best_index) * l.coords;
                int tbox_index = truth_index + 1 + l.classes;

                box out = float_to_box(l.output + box_index, 1);
                out.x /= l.side;
                out.y /= l.side;
                if (l.sqrt) {
                    out.w = out.w*out.w;
                    out.h = out.h*out.h;
                }
                float iou  = box_iou(out, truth);

                //printf("%d,", best_index);
                int p_index = index + locations*l.classes + i*l.n + best_index;
                *(l.cost) -= l.noobject_scale * pow(l.output[p_index], 2);
                *(l.cost) += l.object_scale * pow(1-l.output[p_index], 2);
                avg_obj += l.output[p_index];
                l.delta[p_index] = l.object_scale * (1.-l.output[p_index]);

                if(l.rescore){
                    l.delta[p_index] = l.object_scale * (iou - l.output[p_index]);
                }

                l.delta[box_index+0] = l.coord_scale*(net.truth[tbox_index + 0] - l.output[box_index + 0]);
                l.delta[box_index+1] = l.coord_scale*(net.truth[tbox_index + 1] - l.output[box_index + 1]);
                l.delta[box_index+2] = l.coord_scale*(net.truth[tbox_index + 2] - l.output[box_index + 2]);
                l.delta[box_index+3] = l.coord_scale*(net.truth[tbox_index + 3] - l.output[box_index + 3]);
                if(l.sqrt){
                    l.delta[box_index+2] = l.coord_scale*(sqrt(net.truth[tbox_index + 2]) - l.output[box_index + 2]);
                    l.delta[box_index+3] = l.coord_scale*(sqrt(net.truth[tbox_index + 3]) - l.output[box_index + 3]);
                }

                *(l.cost) += pow(1-iou, 2);
                avg_iou += iou;
                ++count;
            }
        }

        if(0){
            float *costs = calloc(l.batch*locations*l.n, sizeof(float));
            for (b = 0; b < l.batch; ++b) {
                int index = b*l.inputs;
                for (i = 0; i < locations; ++i) {
                    for (j = 0; j < l.n; ++j) {
                        int p_index = index + locations*l.classes + i*l.n + j;
                        costs[b*locations*l.n + i*l.n + j] = l.delta[p_index]*l.delta[p_index];
                    }
                }
            }
            int indexes[100];
            top_k(costs, l.batch*locations*l.n, 100, indexes);
            float cutoff = costs[indexes[99]];
            for (b = 0; b < l.batch; ++b) {
                int index = b*l.inputs;
                for (i = 0; i < locations; ++i) {
                    for (j = 0; j < l.n; ++j) {
                        int p_index = index + locations*l.classes + i*l.n + j;
                        if (l.delta[p_index]*l.delta[p_index] < cutoff) l.delta[p_index] = 0;
                    }
                }
            }
            free(costs);
        }


        *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);


        printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou/count, avg_cat/count, avg_allcat/(count*l.classes), avg_obj/count, avg_anyobj/(l.batch*locations*l.n), count);
        //if(l.reorg) reorg(l.delta, l.w*l.h, size*l.n, l.batch, 0);
    }
}
```

함수 이름: forward\_detection\_layer

입력

* detection\_layer 구조체 l
* network 구조체 net

동작

* detection\_layer 구조체는 객체 탐지에 필요한 다양한 매개변수와 데이터를 포함하며, network 구조체는 레이어에 대한 입력 데이터를 포함합니다.
* 함수는 먼저 몇 가지 변수를 초기화하고, net에서 입력 데이터를 레이어의 출력 데이터로 복사합니다.
* 만약 l의 softmax 매개변수가 true인 경우, 함수는 각 이미지의 각 위치에 대한 출력 데이터에 softmax 활성화 함수를 적용합니다.
* 만약 네트워크가 학습 중인 경우(net.train이 true), 함수는 출력 데이터와 ground-truth 어노테이션을 사용하여 detection layer의 손실을 계산합니다. 구체적으로, 각 이미지의 각 위치에서 객체와 관련된 손실과 객체와 무관한 손실, 예측된 경계 상자와 관련된 손실, 예측된 클래스와 관련된 손실을 계산합니다. 이러한 손실은 누적되어 l.cost에 저장됩니다. 함수는 또한 출력 데이터에 대한 손실의 그래디언트를 계산하고 l.delta에 저장합니다.
* 손실과 그래디언트를 계산한 후, 함수는 훈련 과정을 모니터링하기 위해 손실과 관련된 여러 통계(평균 objectness 점수 및 평균 카테고리 점수 등)를 업데이트합니다.
* 마지막으로, 함수는 값을 반환합니다.

설명

* 객체 탐지용 신경망에서 detection layer의 forward pass를 수행하는 함수입니다.
* 함수는 detection\_layer 구조체와 network 구조체를 입력으로 받습니다.
* 함수는 detection\_layer 구조체의 매개변수와 데이터를 사용하여 출력 데이터를 계산하고, 네트워크가 훈련 중인 경우 손실과 그래디언트를 계산합니다.
* 손실과 그래디언트를 계산한 후, 함수는 통계를 업데이트하고 값을 반환합니다.



## backward\_detection\_layer

```c
void backward_detection_layer(const detection_layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}
```

함수 이름: backward\_detection\_layer

입력:

* const detection\_layer l: detection\_layer 구조체 변수로, detection layer의 설정과 상태를 저장합니다.
* network net: neural network를 나타내는 구조체 변수로, detection layer가 속한 network의 상태를 저장합니다.

동작:

* detection layer의 backward propagation을 수행합니다.
* axpy\_cpu 함수를 사용하여, detection layer의 delta 값과 network의 delta 값을 더합니다.

설명:

* detection layer는 입력 이미지에서 object detection을 수행하는 레이어입니다.
* backward\_detection\_layer 함수는 이러한 detection layer의 backward propagation을 수행합니다.
* backward propagation은 gradient를 역방향으로 전파하여, 각각의 가중치(weight)와 bias에 대한 gradient를 계산합니다.
* axpy\_cpu 함수는 BLAS 라이브러리 함수로, 벡터 덧셈 연산을 수행합니다.
* l.delta와 net.delta는 각각 detection layer와 network의 gradient 값을 저장하는 배열입니다.
* backward\_detection\_layer 함수는 l.delta와 net.delta를 더하여, network의 delta 값을 갱신합니다.



## make\_detection\_layer

```c
detection_layer make_detection_layer(int batch, int inputs, int n, int side, int classes, int coords, int rescore)
{
    detection_layer l = {0};
    l.type = DETECTION;

    l.n = n;
    l.batch = batch;
    l.inputs = inputs;
    l.classes = classes;
    l.coords = coords;
    l.rescore = rescore;
    l.side = side;
    l.w = side;
    l.h = side;
    assert(side*side*((1 + l.coords)*l.n + l.classes) == inputs);
    l.cost = calloc(1, sizeof(float));
    l.outputs = l.inputs;
    l.truths = l.side*l.side*(1+l.coords+l.classes);
    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.outputs, sizeof(float));

    l.forward = forward_detection_layer;
    l.backward = backward_detection_layer;

    fprintf(stderr, "Detection Layer\n");
    srand(0);

    return l;
}
```

함수 이름: make\_detection\_layer

입력:

* batch: 배치 크기
* inputs: 입력값 크기
* n: anchor box 개수
* side: feature map 크기
* classes: 분류할 클래스 수
* coords: 각 anchor box의 좌표 개수
* rescore:

동작:

* detection\_layer 구조체를 초기화하고, 입력값에 대한 필요한 정보를 설정함
* cost, output, delta를 할당하고, truths와 outputs을 설정함
* forward와 backward 함수를 설정함
* "Detection Layer"라는 메시지를 출력함

설명:

* make\_detection\_layer 함수는 detection\_layer 구조체를 초기화하고, 입력값에 대한 필요한 정보를 설정한 후, 초기화된 구조체를 반환합니다. 이 함수는 YOLO 신경망에서 사용되는 detection layer를 생성하는 데 사용됩니다.
* batch는 한 번에 처리할 데이터의 개수이며, inputs는 이전 레이어의 출력값의 크기입니다. n은 anchor box의 개수이고, side는 feature map의 가로, 세로 크기입니다. classes는 분류할 클래스의 개수이며, coords는 각 anchor box의 좌표 개수입니다. rescore는 YOLOv2에서 사용되는 값으로, bbox의 정확도를 측정하는 데 사용됩니다.
* 이 함수는 detection\_layer 구조체를 초기화하고, 입력값에 대한 필요한 정보를 설정합니다. 그리고 cost, output, delta를 할당하고, truths와 outputs을 설정합니다. 마지막으로 forward와 backward 함수를 설정하고, "Detection Layer"라는 메시지를 출력합니다.



## get\_detection\_detections

```c
void get_detection_detections(layer l, int w, int h, float thresh, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    //int per_cell = 5*num+classes;
    for (i = 0; i < l.side*l.side; ++i){
        int row = i / l.side;
        int col = i % l.side;
        for(n = 0; n < l.n; ++n){
            int index = i*l.n + n;
            int p_index = l.side*l.side*l.classes + i*l.n + n;
            float scale = predictions[p_index];
            int box_index = l.side*l.side*(l.classes + l.n) + (i*l.n + n)*4;
            box b;
            b.x = (predictions[box_index + 0] + col) / l.side * w;
            b.y = (predictions[box_index + 1] + row) / l.side * h;
            b.w = pow(predictions[box_index + 2], (l.sqrt?2:1)) * w;
            b.h = pow(predictions[box_index + 3], (l.sqrt?2:1)) * h;
            dets[index].bbox = b;
            dets[index].objectness = scale;
            for(j = 0; j < l.classes; ++j){
                int class_index = i*l.classes;
                float prob = scale*predictions[class_index+j];
                dets[index].prob[j] = (prob > thresh) ? prob : 0;
            }
        }
    }
}
```

함수 이름: get\_detection\_detections

입력:

* layer l: YOLO 네트워크에서 출력 레이어
* int w: 입력 이미지의 너비
* int h: 입력 이미지의 높이
* float thresh: objectness score의 최소 임계값
* detection \*dets: 각 검출 객체의 정보를 저장할 detection 구조체 배열

동작:

* YOLO 네트워크 출력값을 받아서 객체 검출 수행
* 객체 검출 결과를 detection 구조체 배열 dets에 저장

설명:

* YOLO 네트워크에서 출력 레이어의 출력값(predictions)을 받아서, 객체 검출 수행
* l.side는 출력 레이어의 가로 세로 크기, l.n은 각 셀마다 예측한 bounding box의 개수, l.classes는 클래스 개수
* 레이어의 출력값을 이용하여 bounding box와 objectness score, 클래스 확률값 계산
* 계산한 정보를 detection 구조체 배열 dets에 저장하고 반환

