# demo

## parameter

```c
#define DEMO 1

#ifdef OPENCV

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static network *net;
static image buff [3];
static image buff_letter[3];
static int buff_index = 0;
static void * cap;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_frame = 3;
static int demo_index = 0;
static float **predictions;
static float *avg;
static int demo_done = 0;
static int demo_total = 0;
double demo_time;

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
```

* demo\_names: 클래스 이름을 담고 있는 문자열 배열
* demo\_alphabet: 이미지 출력 시 사용되는 폰트 이미지 배열
* demo\_classes: 클래스의 개수
* net: 딥러닝 모델을 담고 있는 네트워크 구조체
* buff: 카메라 또는 비디오 스트림에서 읽어들인 이미지를 담고 있는 이미지 배열
* buff\_letter: 이미지 출력 시 사용되는 글자 이미지 배열
* buff\_index: 현재 사용 중인 이미지 배열의 인덱스
* cap: 카메라 또는 비디오 스트림을 담고 있는 포인터
* fps: 현재 프레임 속도
* demo\_thresh: 객체 탐지에 사용되는 임계값
* demo\_hier: 객체 탐지 시 사용되는 IoU 임계값
* running: 프로그램이 실행 중인지 나타내는 플래그
* demo\_frame: 현재 프레임 인덱스
* demo\_index: 현재 클래스 인덱스
* predictions: 네트워크가 예측한 객체의 정보를 담고 있는 이차원 배열
* avg: 예측한 객체 정보의 평균값
* demo\_done: 객체 탐지가 완료되었는지 나타내는 플래그
* demo\_total: 객체 탐지된 전체 개수
* demo\_time: 객체 탐지에 소요된 시간



함수 이름: detection \*get\_network\_boxes

입력:

* network \*net : 사용할 네트워크
* int w : 이미지의 가로 크기
* int h : 이미지의 세로 크기
* float thresh : 객체 검출을 위한 최소 확률 임계값
* float hier : 객체 검출을 위한 최소 IOU 임계값
* int \*map : 사용하지 않음
* int relative : 좌표 계산에 사용
* int \*num : 검출된 객체의 수를 저장할 포인터







## size\_network

```c
int size_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            count += l.outputs;
        }
    }
    return count;
}
```

함수 이름: size\_network

입력:&#x20;

* network 구조체 포인터 (neural network 모델)

동작:&#x20;

* 입력으로 받은 neural network 모델에서 YOLO, REGION, DETECTION 레이어의 출력 크기를 합산하여 총 출력 크기를 계산한다.

설명:&#x20;

* 이 함수는 neural network 모델의 출력 크기를 계산하는 함수이다.&#x20;
* 입력으로 받은 모델의 모든 레이어를 순회하면서 YOLO, REGION, DETECTION 레이어의 출력 크기를 합산하여 반환한다.&#x20;
* 이 함수는 예를 들어 모델의 출력 크기를 계산하는 데 사용될 수 있으며, 예측이나 추론 결과를 처리하는 데 유용하게 사용될 수 있다.



## remember\_network

```c
void remember_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}
```

함수 이름: remember\_network

입력:&#x20;

* network \*net: 뉴럴 네트워크 구조체 포인터

동작:&#x20;

* 네트워크의 출력 값을 예측 값으로 복사하여 기억합니다.&#x20;
* 이 함수는 YOLO, REGION 또는 DETECTION 레이어에서 나온 출력 값만 복사합니다.

설명:&#x20;

* 뉴럴 네트워크에서는 입력 데이터를 이용하여 출력 값을 예측합니다.&#x20;
* 이 함수는 해당 네트워크의 예측 값을 복사하여 기억합니다.&#x20;
* 이 기억된 예측 값은 나중에 다양한 목적으로 사용될 수 있습니다.&#x20;
* 이 함수는 demo\_index와 predictions 배열을 사용합니다.&#x20;
* demo\_index는 현재 데모에서 사용되는 이미지의 인덱스를 나타내며, predictions 배열은 예측 값을 기억하기 위한 배열입니다.&#x20;
* 이 함수는 각 레이어의 출력 값의 크기를 count 변수에 누적하여 predictions 배열의 적절한 위치에 복사합니다.



## avg\_predictions

```c
detection *avg_predictions(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(demo_total, 0, avg, 1);
    for(j = 0; j < demo_frame; ++j){
        axpy_cpu(demo_total, 1./demo_frame, predictions[j], 1, avg, 1);
    }
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(l.output, avg + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}
```

함수 이름: avg\_predictions

입력:

* network \*net: 네트워크 모델
* int \*nboxes: 감지된 bounding box의 개수를 담을 포인터 변수

동작:

* 이전에 저장된 예측(predictions)을 이용하여 각 클래스에 대한 확률값의 평균을 계산한다.
* 계산된 평균값을 이용하여 다시 네트워크를 실행하고, 감지된 bounding box를 반환한다.

설명:

* 이전에 저장된 predictions은 remember\_network 함수를 통해 저장된 예측값을 말한다.
* 이 함수에서는 저장된 예측값들의 평균값을 계산하고, 이를 이용하여 네트워크를 실행한다.
* 이전 예측값들의 평균을 이용함으로써 일시적인 예측값의 변동성을 줄이고, 보다 안정적인 예측 결과를 얻을 수 있다.
* 계산된 예측값을 이용하여 get\_network\_boxes 함수를 호출하여 bounding box를 감지하고, 이를 반환한다.



## detect\_in\_thread

```c
void *detect_in_thread(void *ptr)
{
    running = 1;
    float nms = .4;

    layer l = net->layers[net->n-1];
    float *X = buff_letter[(buff_index+2)%3].data;
    network_predict(net, X);

    /*
       if(l.type == DETECTION){
       get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
       } else */
    remember_network(net);
    detection *dets = 0;
    int nboxes = 0;
    dets = avg_predictions(net, &nboxes);


    /*
       int i,j;
       box zero = {0};
       int classes = l.classes;
       for(i = 0; i < demo_detections; ++i){
       avg[i].objectness = 0;
       avg[i].bbox = zero;
       memset(avg[i].prob, 0, classes*sizeof(float));
       for(j = 0; j < demo_frame; ++j){
       axpy_cpu(classes, 1./demo_frame, dets[j][i].prob, 1, avg[i].prob, 1);
       avg[i].objectness += dets[j][i].objectness * 1./demo_frame;
       avg[i].bbox.x += dets[j][i].bbox.x * 1./demo_frame;
       avg[i].bbox.y += dets[j][i].bbox.y * 1./demo_frame;
       avg[i].bbox.w += dets[j][i].bbox.w * 1./demo_frame;
       avg[i].bbox.h += dets[j][i].bbox.h * 1./demo_frame;
       }
    //copy_cpu(classes, dets[0][i].prob, 1, avg[i].prob, 1);
    //avg[i].objectness = dets[0][i].objectness;
    }
     */

    if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");
    image display = buff[(buff_index+2) % 3];
    draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);
    free_detections(dets, nboxes);

    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
}
```

함수 이름: detect\_in\_thread

입력:&#x20;

* void\* ptr: 포인터 타입의 인자, 사용하지 않음

동작:&#x20;

* YOLO 신경망으로 객체를 탐지하고, 결과를 출력 이미지에 표시한다.&#x20;
* 평균 예측 값을 계산하고, 이를 바탕으로 객체를 탐지하고, 비최대 억제(NMS)를 수행한다.&#x20;
* 마지막으로, 탐지된 객체들을 이미지에 그리고 출력한다.

설명:

* 먼저, running 변수를 1로 설정하여 스레드가 실행중임을 표시한다.
* nms 변수에 0.4를 할당하여 비최대 억제에 사용될 임계값을 설정한다.
* 다음으로, YOLO 신경망의 출력층(layer)을 가져와서, 현재 처리할 이미지 데이터인 buff\_letter\[(buff\_index+2)%3].data를 입력값으로 전달하여 객체를 예측한다.
* 예측된 결과를 평균 예측 값으로 기억하고, avg\_predictions() 함수를 사용하여 평균 예측 값을 바탕으로 객체를 탐지한다.
* 비최대 억제를 수행하여 중복으로 탐지된 객체를 제거한다.
* 마지막으로, 출력 이미지에 탐지된 객체들을 그리고 출력한다. 스레드가 실행을 완료하면 running 변수를 0으로 설정하여 스레드가 종료되었음을 표시한다.



## fetch\_in\_thread

```c
void *fetch_in_thread(void *ptr)
{
    free_image(buff[buff_index]);
    buff[buff_index] = get_image_from_stream(cap);
    if(buff[buff_index].data == 0) {
        demo_done = 1;
        return 0;
    }
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    return 0;
}
```

함수 이름: fetch\_in\_thread&#x20;

입력:&#x20;

* void \*ptr: 포인터

동작:&#x20;

* 카메라로부터 이미지를 가져와서 해당 이미지를 신경망 모델의 입력 크기에 맞게 리사이징하고, 이전 이미지를 해제하고 새 이미지로 대체함.&#x20;

설명:&#x20;

* 이 함수는 쓰레드에서 실행되며, 카메라 스트림에서 이미지를 가져와서 이전 버퍼 이미지를 해제하고, 새 이미지를 신경망 모델의 입력 크기에 맞게 리사이징하여 현재 버퍼에 할당하는 역할을 합니다.&#x20;
* 이전에 할당된 이미지 메모리를 해제하여 메모리 누수를 방지합니다.



## display\_in\_thread

```c
void *display_in_thread(void *ptr)
{
    int c = show_image(buff[(buff_index + 1)%3], "Demo", 1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}
```

함수 이름: display\_in\_thread

입력:&#x20;

* void \*ptr: 포인터

동작:

* 현재 버퍼 중 뒤의 두 번째 이미지를 화면에 표시한다.
* 키보드 입력을 받아 해당하는 기능을 수행한다. (ESC: 종료, R: detection 임계값 증가, F: detection 임계값 감소, T: hierarchy 임계값 증가, G: hierarchy 임계값 감소)

설명:

* 딥러닝 모델이 예측한 결과를 화면에 보여주는 역할을 담당하는 함수이다.
* 버퍼 중 뒤의 두 번째 이미지를 화면에 표시하여 실시간으로 영상을 확인할 수 있도록 한다.
* 키보드 입력을 받아 해당하는 기능을 수행한다. ESC를 누르면 프로그램이 종료되고, R, F, T, G 키를 누르면 각각 detection 임계값을 증가시키거나 감소시키고, hierarchy 임계값을 증가시키거나 감소시킨다.



## display\_loop

```c
void *display_loop(void *ptr)
{
    while(1){
        display_in_thread(0);
    }
}
```

함수 이름: display\_loop

입력:&#x20;

* ptr: void 포인터

동작:&#x20;

* display\_in\_thread 함수를 무한 루프로 실행하여 영상 출력 창을 유지시키고, 사용자 입력에 따라 demo\_thresh와 demo\_hier 등의 변수 값을 변경할 수 있도록 한다.

설명:&#x20;

* 이 함수는 사용자에게 영상 출력 창을 제공하고, 사용자 입력을 받아들여 변수 값들을 조절할 수 있도록 한다.&#x20;
* display\_in\_thread 함수는 이 함수 내에서 무한히 반복되며, 영상 출력 창이 종료되거나 프로그램이 종료될 때까지 유지된다.&#x20;
* 이 함수는 쓰레드로 실행되기 때문에, 메인 프로그램과 별개로 동작하며 영상 출력 창이 종료되더라도 메인 프로그램은 계속해서 실행될 수 있다.



## detect\_loop

```c

void *detect_loop(void *ptr)
{
    while(1){
        detect_in_thread(0);
    }
}
```

함수 이름: detect\_loop\
입력:&#x20;

* ptr: void 포인터 (사용하지 않음)\


동작:&#x20;

* 무한 루프를 돌면서 detect\_in\_thread 함수를 호출하여 객체 탐지를 수행\


설명:&#x20;

* 객체 탐지 루프를 돌며 영상에서 객체를 탐지하고, 그 결과를 화면에 표시하는 함수입니다.



## demo

```c
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    //demo_frame = avg_frames;
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;

    srand(2222222);

    int i;
    demo_total = size_network(net);
    predictions = calloc(demo_frame, sizeof(float*));
    for (i = 0; i < demo_frame; ++i){
        predictions[i] = calloc(demo_total, sizeof(float));
    }
    avg = calloc(demo_total, sizeof(float));

    if(filename){
        printf("video file: %s\n", filename);
        cap = open_video_stream(filename, 0, 0, 0, 0);
    }else{
        cap = open_video_stream(0, cam_index, w, h, frames);
    }

    if(!cap) error("Couldn't connect to webcam.\n");

    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);

    int count = 0;
    if(!prefix){
        make_window("Demo", 1352, 1013, fullscreen);
    }

    demo_time = what_time_is_it_now();

    while(!demo_done){
        buff_index = (buff_index + 1) %3;
        if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
        if(!prefix){
            fps = 1./(what_time_is_it_now() - demo_time);
            demo_time = what_time_is_it_now();
            display_in_thread(0);
        }else{
            char name[256];
            sprintf(name, "%s_%08d", prefix, count);
            save_image(buff[(buff_index + 1)%3], name);
        }
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
        ++count;
    }
}

```

함수 이름: demo

입력:

* char \*cfgfile: YOLO 모델의 설정 파일 경로
* char \*weightfile: 학습된 YOLO 모델의 가중치 파일 경로
* float thresh: Object detection 결과의 임계값
* int cam\_index: 사용할 웹캠의 인덱스 (0부터 시작)
* const char \*filename: Object detection을 수행할 동영상 파일 경로 (웹캠을 사용하지 않을 경우에만 사용)
* char \*\*names: Object detection 대상 클래스명 배열
* int classes: Object detection 대상 클래스 수
* int delay: Object detection 프레임 간의 딜레이
* char \*prefix: Object detection 결과 저장시 사용할 파일 이름 prefix
* int avg\_frames: Object detection 프레임의 평균화 수 (최근 몇 개의 프레임을 평균화하여 Object detection 수행)
* float hier: YOLO 모델의 hier 파라미터 값
* int w: 동영상 또는 웹캠 프레임의 너비
* int h: 동영상 또는 웹캠 프레임의 높이
* int frames: Object detection 수행할 프레임 수 (동영상에서 사용)
* int fullscreen: Object detection 결과를 풀스크린으로 표시할지 여부

동작:

* YOLO 모델을 로드하고, 웹캠 또는 동영상을 캡처하기 위한 초기화 작업을 수행한다.
* Object detection 결과를 저장하기 위한 배열과 변수를 초기화한다.
* Object detection을 위한 fetch, detect, display 스레드를 생성하고, 결과를 출력한다.
* prefix가 설정되어 있으면 Object detection 결과를 이미지 파일로 저장한다.
* demo\_done 플래그가 설정되면 프로그램을 종료한다.

설명:

* 이 코드는 YOLO 알고리즘을 사용하여 object detection을 수행하는 데모 프로그램이다.
* 프로그램은 웹캠 또는 동영상을 입력으로 받아서 Object detection을 수행하고, 결과를 실시간으로 출력한다.
* 프로그램은 fetch, detect, display 스레드를 생성하여 Object detection 처리를 병렬화하고, 최적화된 성능을 보인다.
* 프로그램에서 사용하는 fetch\_in\_thread, detect\_in\_thread, display\_in\_thread 함수들은 각각 fetch, detect, display 스레드에서 실행되는 함수이다.
* demo 함수는 이들 스레드를 생성하고, Object detection 결과를 출력하는 메인 루프 역할을 수행한다.



## demo error

```c
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif
```

함수 이름: demo

입력:

* char \*cfgfile: YOLO 모델의 구성 파일 경로
* char \*weightfile: 학습된 YOLO 모델의 가중치 파일 경로
* float thresh: 객체 탐지 임계값
* int cam\_index: 사용할 카메라의 인덱스
* const char \*filename: 사용할 비디오 파일 경로
* char \*\*names: 클래스 이름 배열
* int classes: 클래스 수
* int delay: 비디오 재생 프레임 간 딜레이 (밀리초 단위)
* char \*prefix: 결과 이미지 파일 이름의 prefix
* int avg: YOLO 모델에서 사용되는 평균화 프레임 수
* float hier: YOLO 모델에서 사용되는 Hierarchy 임계값
* int w: 입력 이미지의 너비
* int h: 입력 이미지의 높이
* int frames: 비디오에서 읽을 프레임 수
* int fullscreen: 전체 화면 모드 여부

동작:

* YOLO 모델을 로드하고 입력 이미지를 처리하며 객체 탐지 결과를 표시하는 데모를 수행한다.
* OpenCV를 사용하여 웹캠 또는 비디오 파일에서 입력 이미지를 가져온다.
* YOLO 모델에서 객체 탐지를 위해 fetch\_in\_thread 및 detect\_in\_thread 함수를 실행하는 스레드를 생성한다.
* display\_in\_thread 함수를 사용하여 객체 탐지 결과를 화면에 표시한다.
* 비디오에서 프레임을 읽고 이미지를 처리한 후 결과 이미지를 파일로 저장할 수 있다.
* 프로그램 종료 조건인 demo\_done이 true가 될 때까지 무한 루프를 실행한다.

설명:&#x20;

* 이 함수는 OpenCV를 사용하여 웹캠 또는 비디오 파일에서 입력 이미지를 가져와 YOLO 모델을 사용하여 객체를 탐지하는 데모를 수행한다.&#x20;
* 만약 OpenCV가 설치되어 있지 않은 경우에는 "Demo needs OpenCV for webcam images." 메시지가 출력된다.&#x20;
* 이 함수에서는 fetch\_in\_thread 및 detect\_in\_thread 함수를 실행하는 스레드를 생성하여 객체 탐지 속도를 향상시키고, display\_in\_thread 함수를 사용하여 객체 탐지 결과를 화면에 표시한다.&#x20;
* 또한, 비디오에서 프레임을 읽고 이미지를 처리한 후 결과 이미지를 파일로 저장할 수 있다.

