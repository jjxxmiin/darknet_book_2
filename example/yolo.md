# yolo

```c
#include "darknet.h"

char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
```

## train\_yolo

```c
void train_yolo(char *cfgfile, char *weightfile)
{
    char *train_images = "/data/voc/train.txt";
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    int imgs = net->batch*net->subdivisions;
    int i = *net->seen/imgs;
    data train, buffer;


    layer l = net->layers[net->n - 1];

    int side = l.side;
    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA;

    args.angle = net->angle;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}
```

함수 이름: train\_yolo

입력:

* char \*cfgfile: YOLO 네트워크 구성 파일 경로
* char \*weightfile: 사전 학습된 가중치 파일 경로

동작:

* YOLO 네트워크를 사용하여 객체 검출을 학습하는 함수이다.
* 주어진 구성 파일(cfgfile)과 가중치 파일(weightfile)을 사용하여 네트워크를 초기화한다.
* 학습 이미지 경로(train\_images)와 백업 디렉토리 경로(backup\_directory)를 설정한다.
* 네트워크의 학습 관련 설정을 출력한다(학습률, 모멘텀, 감쇠).
* 네트워크를 학습하기 위한 데이터(train)와 버퍼(buffer)를 초기화한다.
* 주어진 이미지 경로에서 학습 데이터를 로드하고, 데이터를 배치 단위로 가져오는 스레드를 생성한다.
* 최대 배치 수(net->max\_batches)에 도달할 때까지 반복문을 실행하여 네트워크를 학습한다.
* 데이터를 로드하고 학습을 수행한 후 손실(loss)를 계산하고, 이전 손실값(avg\_loss)을 업데이트한다.
* 현재 학습 속도(get\_current\_rate(net)), 경과 시간, 처리한 이미지 수 등의 정보를 출력한다.
* 일정 주기마다 또는 처음 1000회 학습 후 100회마다 가중치를 백업 디렉토리에 저장한다.
* 반복문이 끝난 후 최종 가중치를 저장한다.

설명:

* 이 함수는 YOLO (You Only Look Once) 객체 검출 모델을 사용하여 객체 검출을 학습하는 기능을 수행한다.
* 학습은 네트워크 구성 파일(cfgfile)과 사전 학습된 가중치 파일(weightfile)을 로드하여 초기화된 네트워크를 사용한다.
* train\_images 변수에는 학습에 사용할 이미지 경로가 설정되어 있다.
* backup\_directory 변수에는 학습 중 백업 파일을 저장할 디렉토리 경로가 설정되어 있다.
* 학습은 반복적으로 수행되며, 네트워크를 학습하기 위해 데이터를 로드하고 해당 데이터로 네트워크를 업데이트한다.
* 학습은 최대 배치 수(net->max\_batches)에 도달할 때까지 계속된다.
* 학습 과정에서 손실(loss)을 계산하고, 평균 손실값(avg\_loss)을 업데이트하여 학습 진행 상황을 모니터링한다.
* 주기적으로 가중치를 백업 디렉토리에 저장하여 학습 중간 결과를 보관할 수 있다.



## print\_yolo\_detections

```c
void print_yolo_detections(FILE **fps, char *id, int total, int classes, int w, int h, detection *dets)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                    xmin, ymin, xmax, ymax);
        }
    }
}
```

함수 이름: print\_yolo\_detections

입력:

* FILE \*\*fps: 출력 파일 포인터 배열
* char \*id: 이미지 id
* int total: 인식된 객체의 총 개수
* int classes: 객체 클래스의 수
* int w: 이미지의 너비
* int h: 이미지의 높이
* detection \*dets: 객체 인식 결과 배열 포인터

동작:&#x20;

* 객체 인식 결과 배열에서 추론된 bounding box 정보를 기반으로 출력 파일에 객체 감지 정보를 쓴다.&#x20;
* 출력 파일은 클래스 수 만큼의 개수를 가지며, 클래스에 따라 해당 클래스 파일에 쓰여진다.

설명:

* 함수는 YOLO (You Only Look Once) 객체 검출 모델에서 감지된 객체를 파일에 출력하는 기능을 한다.
* 함수는 각 클래스에 해당하는 파일 포인터를 담은 파일 포인터 배열과, 이미지의 id, 객체 총 개수, 객체 클래스 수, 이미지의 너비와 높이, 객체 인식 결과 배열 포인터를 입력으로 받는다.
* 객체 인식 결과 배열에서 추론된 bounding box 정보를 기반으로 각 클래스에 대해 적절한 출력을 파일에 쓴다.
* 출력 파일은 클래스 수 만큼의 개수를 가지며, 클래스에 따라 해당 클래스 파일에 쓰여진다.
* 출력 형식은 "이미지 id 확률 x\_min y\_min x\_max y\_max" 이며, bbox 좌표는 정규화(normalized) 된 상태로 출력된다.



## validate\_yolo

```c
void validate_yolo(char *cfg, char *weights)
{
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("/home/pjreddie/data/voc/2007_test.txt");
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 8;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, 0, 0, 0, &nboxes);
            if (nms) do_nms_sort(dets, l.side*l.side*l.n, classes, iou_thresh);
            print_yolo_detections(fps, id, l.side*l.side*l.n, classes, w, h, dets);
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}
```

함수 이름: validate\_yolo

입력:

* char \*cfg: YOLO 네트워크 구성 파일 경로
* char \*weights: 사전 학습된 가중치 파일 경로

동작:

* YOLO 네트워크를 사용하여 객체 검출의 유효성을 검증하는 함수이다.
* 주어진 구성 파일(cfg)과 가중치 파일(weights)을 사용하여 네트워크를 초기화한다.
* 네트워크의 배치 크기를 1로 설정한다.
* 네트워크의 학습률(learning rate), 모멘텀(momentum), 감쇠(decay) 값을 출력한다.
* 시드 값을 설정한다.
* 결과를 저장할 파일 경로와 리스트를 초기화한다.
* VOC 데이터셋의 테스트 이미지 경로를 가져온다.
* VOC 클래스의 개수, 네트워크의 마지막 레이어 정보를 가져온다.
* 클래스별로 결과를 저장할 파일 포인터 배열을 할당하고 초기화한다.
* 이미지 개수(m)를 가져온다.
* 반복문을 통해 다음을 수행한다:
  * 이미지 경로(path)를 가져오고, 이미지와 크기가 조정된 이미지를 로드한다.
  * 이미지에 대한 식별자(id)를 생성한다.
  * 크기가 조정된 이미지를 네트워크에 입력하여 객체를 예측한다.
  * 네트워크 출력으로부터 객체 박스(dets)를 가져온다. 임계값(thresh)을 적용하여 객체를 필터링한다.
  * 객체 검출 결과를 파일에 출력한다.
  * 할당된 리소스를 해제한다.
* 총 검출 시간을 출력한다.

설명:

* 이 함수는 YOLO (You Only Look Once) 객체 검출 모델을 사용하여 객체 검출의 유효성을 검증하는 기능을 수행한다.
* 검증은 주어진 구성 파일(cfg)과 사전 학습된 가중치 파일(weights)을 로드하여 초기화된 네트워크를 사용한다.
* VOC 데이터셋의 테스트 이미지를 사용하여 객체 검출 결과를 평가한다.
* 객체 박스를 출력하는 파일 포인터 배열을 할당하고 초기화한다.
* 이미지를 여러 스레드로 처리하여 병렬로 실행하며, 객체 검출 결과를 출력한다.
* 최종 검출 시간을 출력하여 전체 수행 시간을 측정한다.



## validate\_yolo\_recall

```c
void validate_yolo_recall(char *cfg, char *weights)
{
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    list *plist = get_paths("data/voc.2007.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }

    int m = plist->size;
    int i=0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = 0;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net->w, net->h);
        char *id = basecfg(path);
        network_predict(net, sized.data);

        int nboxes = 0;
        detection *dets = get_network_boxes(net, orig.w, orig.h, thresh, 0, 0, 1, &nboxes);
        if (nms) do_nms_obj(dets, side*side*l.n, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < side*side*l.n; ++k){
            if(dets[k].objectness > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < side*side*l.n; ++k){
                float iou = box_iou(dets[k].bbox, t);
                if(dets[k].objectness > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free_detections(dets, nboxes);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}
```

함수 이름: validate\_yolo\_recall

입력:

* char \*cfg: YOLO 네트워크 구성 파일 경로
* char \*weights: 사전 학습된 가중치 파일 경로

동작:

* YOLO 네트워크를 사용하여 객체 검출의 재현율(recall)을 검증하는 함수이다.
* 주어진 구성 파일(cfg)과 가중치 파일(weights)을 사용하여 네트워크를 초기화한다.
* 네트워크의 배치 크기를 1로 설정한다.
* 네트워크의 학습률(learning rate), 모멘텀(momentum), 감쇠(decay) 값을 출력한다.
* 난수 시드를 설정한다.
* 결과를 저장할 파일 경로와 리스트를 초기화한다.
* VOC 데이터셋의 테스트 이미지 경로를 가져온다.
* VOC 클래스의 개수, 네트워크의 마지막 레이어 정보, 그리고 네트워크의 입력 크기 정보를 가져온다.
* 클래스별로 결과를 저장할 파일 포인터 배열을 할당하고 초기화한다.
* 이미지 개수(m)를 가져온다.
* 총 예측 수(total), 정확한 예측 수(correct), 제안 수(proposals), 평균 IOU(avg\_iou)를 초기화한다.
* 이미지 개수(m)만큼 반복하면서 다음을 수행한다:
  * 이미지 경로(path)를 가져온다.
  * 원본 이미지를 로드하고, 네트워크의 입력 크기에 맞게 크기를 조정한다.
  * 이미지의 식별자(id)를 생성한다.
  * 크기가 조정된 이미지를 네트워크에 입력하여 객체를 예측한다.
  * 네트워크 출력으로부터 객체 박스(dets)를 가져온다. 임계값(thresh)을 적용하여 객체를 필터링한다.
  * 해당 이미지에 대한 정답(label) 파일 경로(labelpath)를 생성한다.
  * 정답(label) 파일을 읽어 객체 박스(truth)를 가져온다.
  * 객체 제안 수(proposals)를 계산한다.
  * 정답(label)과 예측 결과를 비교하여 재현율을 계산한다.
  * 결과를 출력한다.

설명:

* 이 함수는 YOLO (You Only Look Once) 객체 검출 모델을 사용하여 객체 검출의 재현율(recall)을 검증하는 기능을 수행한다.
* 검증은 주어진 구성 파일(cfg)과 사전 학습된 가중치 파일(weights)을 로드하여 초기화된 네트워크를 사용한다.
* VOC 데이터셋의 테스트 이미지를 사용하여 객체 검출 결과를 평가한다.



## test\_yolo

```c
void test_yolo(char *cfgfile, char *weightfile, char *filename, float thresh)
{
    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    layer l = net->layers[net->n-1];
    set_batch_network(net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    float nms=.4;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net->w, net->h);
        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));

        int nboxes = 0;
        detection *dets = get_network_boxes(net, 1, 1, thresh, 0, 0, 0, &nboxes);
        if (nms) do_nms_sort(dets, l.side*l.side*l.n, l.classes, nms);

        draw_detections(im, dets, l.side*l.side*l.n, thresh, voc_names, alphabet, 20);
        save_image(im, "predictions");
        show_image(im, "predictions", 0);
        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}
```

함수 이름: test\_yolo

입력:

* char \*cfgfile: YOLO 네트워크 구성 파일 경로
* char \*weightfile: 사전 학습된 가중치 파일 경로
* char \*filename: 테스트할 이미지 파일 경로 (선택적)
* float thresh: 객체 검출 임계값

동작:

* YOLO 네트워크를 사용하여 객체 검출을 테스트하는 함수이다.
* 주어진 구성 파일(cfgfile)과 가중치 파일(weightfile)을 사용하여 네트워크를 초기화한다.
* 객체 검출 결과를 시각화하기 위해 알파벳 이미지(alphabet)를 로드한다.
* 네트워크의 마지막 레이어(layer) 정보를 가져온다.
* 네트워크의 배치 크기를 1로 설정한다.
* 난수 시드를 설정한다.
* 테스트할 이미지 파일 경로(filename)가 주어진 경우 해당 파일로 설정한다. 그렇지 않으면 사용자로부터 이미지 파일 경로를 입력받는다.
* 입력된 이미지를 로드하고, 네트워크의 입력 크기에 맞게 크기를 조정한다.
* 크기가 조정된 이미지를 네트워크에 입력하여 객체를 예측한다.
* 예측 시간을 측정하고 출력한다.
* 네트워크의 출력으로부터 객체 박스(dets)를 가져온다. 임계값(thresh)을 적용하여 객체를 필터링한다.
* 객체를 시각화하여 원본 이미지에 그린다.
* 객체 검출 결과 이미지를 "predictions" 이름으로 저장한다.
* 결과 이미지를 보여준다.
* 메모리를 해제한다.
* 테스트할 이미지 파일 경로(filename)가 주어진 경우 반복문을 종료한다.

설명:

* 이 함수는 YOLO (You Only Look Once) 객체 검출 모델을 사용하여 이미지의 객체를 테스트하는 기능을 수행한다.
* 테스트는 주어진 구성 파일(cfgfile)과 사전 학습된 가중치 파일(weightfile)을 로드하여 초기화된 네트워크를 사용한다.
* 테스트할 이미지 파일 경로(filename)가 주어지면 해당 이미지로 테스트를 수행하고, 그렇지 않으면 사용자로부터 이미지 파일 경로를 입력받아 테스트한다.
* 객체 검출 결과는 객체 박스(dets)를 시각화하여 원본 이미지에 그리고, "predictions"라는 이름으로 결과 이미지를 저장한다.
* 테스트는 사용자가 종료하길 원할 때까지 계속해서 이미지를 입력받아 테스트할 수 있다.



## run\_yolo

```c
void run_yolo(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    int avg = find_int_arg(argc, argv, "-avg", 1);
    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) test_yolo(cfg, weights, filename, thresh);
    else if(0==strcmp(argv[2], "train")) train_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "recall")) validate_yolo_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, 20, frame_skip, prefix, avg, .5, 0,0,0,0);
}
```

함수 이름: run\_yolo

입력:

* argc: 정수형 매개변수. 명령줄에서 전달된 인수(argument)의 개수.
* argv: 문자열 배열 매개변수. 명령줄에서 전달된 인수의 배열.

동작:

1. prefix: 문자열 포인터 변수. "-prefix" 옵션의 값을 찾아 할당한다.
2. thresh: 부동 소수점 변수. "-thresh" 옵션의 값을 찾아 할당한다. 기본값은 0.2이다.
3. cam\_index: 정수형 변수. "-c" 옵션의 값을 찾아 할당한다. 기본값은 0이다.
4. frame\_skip: 정수형 변수. "-s" 옵션의 값을 찾아 할당한다. 기본값은 0이다.
5. argc가 4보다 작으면, 오류 메시지를 출력하고 함수를 종료한다.
6. avg: 정수형 변수. "-avg" 옵션의 값을 찾아 할당한다. 기본값은 1이다.
7. cfg: 문자열 포인터 변수. argv\[3]의 값을 할당한다.
8. weights: 문자열 포인터 변수. argc가 4보다 크면 argv\[4]의 값을 할당한다. 그렇지 않으면 0을 할당한다.
9. filename: 문자열 포인터 변수. argc가 5보다 크면 argv\[5]의 값을 할당한다. 그렇지 않으면 0을 할당한다.
10. argv\[2]의 값이 "test"와 동일하면 test\_yolo(cfg, weights, filename, thresh) 함수를 호출한다.
11. argv\[2]의 값이 "train"과 동일하면 train\_yolo(cfg, weights) 함수를 호출한다.
12. argv\[2]의 값이 "valid"와 동일하면 validate\_yolo(cfg, weights) 함수를 호출한다.
13. argv\[2]의 값이 "recall"과 동일하면 validate\_yolo\_recall(cfg, weights) 함수를 호출한다.
14. argv\[2]의 값이 "demo"와 동일하면 demo(cfg, weights, thresh, cam\_index, filename, voc\_names, 20, frame\_skip, prefix, avg, .5, 0,0,0,0) 함수를 호출한다.

설명:&#x20;

* 이 함수는 명령줄 인수를 분석하고 해당하는 동작을 수행하는 함수입니다.&#x20;
* 입력으로 주어진 `argc`와 `argv`를 사용하여 옵션과 값들을 찾고, 해당하는 동작을 수행하는 다른 함수들을 호출합니다.&#x20;
* 동작은 `argv[2]`의 값에 따라 결정되며, 각 동작에는 추가적인 인수들이 필요할 수 있습니다.&#x20;
* 만약 `argc`가 4보다 작으면 오류 메시지를 출력하고 함수를 종료합니다.

