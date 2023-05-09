# coco

```c
#include "darknet.h"

#include <stdio.h>

char *coco_classes[] = {"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};
```

## train\_coco

```c
void train_coco(char *cfgfile, char *weightfile)
{
    //char *train_images = "/home/pjreddie/data/voc/test/train.txt";
    //char *train_images = "/home/pjreddie/data/coco/train.txt";
    char *train_images = "data/coco.trainval.txt";
    //char *train_images = "data/bags.train.list";
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

        /*
           image im = float_to_image(net->w, net->h, 3, train.X.vals[113]);
           image copy = copy_image(im);
           draw_coco(copy, train.y.vals[113], 7, "truth");
           cvWaitKey(0);
           free_image(copy);
         */

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
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}
```

함수 이름: train\_coco

입력:

* cfgfile: 학습을 위한 YOLO 모델 구성 파일 경로 (문자열)
* weightfile: 학습을 시작할 때 사용할 가중치 파일 경로 (문자열)

동작:&#x20;

* COCO 데이터셋을 사용하여 YOLO 모델을 학습하는 함수입니다.&#x20;
* 주어진 cfgfile과 weightfile을 사용하여 YOLO 모델을 불러온 후, train\_images에서 이미지를 로드하고 해당 이미지와 레이블 데이터를 사용하여 모델을 학습합니다.&#x20;
* 학습 중에는 주기적으로 가중치를 저장하고, 학습 속도, 평균 손실값 등의 정보를 출력합니다.

설명:

* train\_images: 학습에 사용될 이미지와 레이블 데이터 파일 경로 (문자열)
* backup\_directory: 가중치 파일 백업을 위한 디렉토리 경로 (문자열)
* base: cfgfile에서 모델 구성 파일 이름 (문자열)
* avg\_loss: 현재까지의 평균 손실값 (실수)
* net: YOLO 모델 (네트워크)
* imgs: 배치 크기 (정수)
* i: 현재까지 학습한 배치 수 (정수)
* train: 학습에 사용될 데이터 (데이터 구조체)
* buffer: 데이터를 불러올 때 사용될 버퍼 (데이터 구조체)
* l: 모델의 마지막 레이어 (레이어 구조체)
* side: 출력 그리드 한 변의 길이 (정수)
* classes: 객체 종류 수 (정수)
* jitter: 이미지 자르기에 사용되는 임의의 값 (실수)
* plist: 이미지 파일 경로 리스트 (리스트 구조체)
* paths: 이미지 파일 경로 배열 (문자열 포인터 배열)
* args: load\_data\_in\_thread 함수로 전달되는 인자들 (load\_args 구조체)
* load\_thread: 이미지 및 레이블 데이터를 로드하는 스레드 (pthread\_t)
* time: 시간 측정을 위한 변수 (clock\_t)
* loss: 현재 배치의 손실값 (실수)
* buff: 가중치 파일 이름 등을 저장하는 문자열 버퍼 (문자열)



## print\_cocos

```c
static void print_cocos(FILE *fp, int image_id, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < num_boxes; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
        }
    }
}
```

함수 이름: print\_cocos&#x20;

입력:

* fp (FILE \*) : 출력 파일 포인터
* image\_id (int) : 이미지 ID
* dets (detection \*) : 객체 감지 정보 배열 포인터
* num\_boxes (int) : 객체 감지 정보 배열의 크기
* classes (int) : 클래스 수
* w (int) : 이미지 너비
* h (int) : 이미지 높이&#x20;

동작:&#x20;

* 객체 감지 정보를 COCO 형식의 JSON 파일로 출력하는 함수입니다.&#x20;
* 각 객체에 대한 정보는 "image\_id", "category\_id", "bbox", "score"의 4가지 정보로 구성됩니다.&#x20;
* 출력 파일은 입력으로 받은 출력 파일 포인터인 fp에 출력됩니다.&#x20;
* 입력으로 받은 객체 감지 정보 배열 포인터 dets를 이용하여 bbox 정보를 계산하고, 각 클래스에 대한 확률 정보를 이용하여 COCO 형식의 JSON 파일 형태로 출력합니다.&#x20;

설명:&#x20;

* COCO(Common Objects in Context) 데이터셋은 객체 감지 분야에서 널리 사용되는 데이터셋 중 하나입니다.&#x20;
* 이 데이터셋에서 사용되는 객체 감지 정보의 출력 형식인 COCO 형식의 JSON 파일을 출력하기 위한 함수입니다.



## get\_coco\_image\_id

```c
int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '_');
    return atoi(p+1);
}
```

함수 이름: get\_coco\_image\_id&#x20;

입력:&#x20;

* char \*filename: COCO 데이터셋 이미지 파일의 경로를 가리키는 문자열 포인터&#x20;

동작:&#x20;

* 입력된 이미지 파일 경로에서 이미지 ID를 파싱하여 정수형으로 반환합니다.&#x20;

설명:&#x20;

* COCO 데이터셋에서 이미지 파일의 이름은 'COCO\_\[category]_\[image\_id].jpg'_와 같은 형태로 구성됩니다.&#x20;
* get\_coco\_image\_id 함수는 입력된 파일 경로에서 _'_'\_" 기호를 기준으로 마지막 문자열을 추출한 뒤, 이를 정수형으로 변환하여 반환합니다.&#x20;
* 이렇게 추출된 숫자는 해당 이미지의 고유한 ID를 나타냅니다.



## validate\_coco

```c
void validate_coco(char *cfg, char *weights)
{
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    char *base = "results/";
    list *plist = get_paths("data/coco_val_5k.list");
    //list *plist = get_paths("/home/pjreddie/data/people-art/test.txt");
    //list *plist = get_paths("/home/pjreddie/data/voc/test/2007_test.txt");
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    snprintf(buff, 1024, "%s/coco_results.json", base);
    FILE *fp = fopen(buff, "w");
    fprintf(fp, "[\n");

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .01;
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
            int image_id = get_coco_image_id(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, 0, 0, 0, &nboxes);
            if (nms) do_nms_sort(dets, l.side*l.side*l.n, classes, iou_thresh);
            print_cocos(fp, image_id, dets, l.side*l.side*l.n, classes, w, h);
            free_detections(dets, nboxes);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fseek(fp, -2, SEEK_CUR);
    fprintf(fp, "\n]\n");
    fclose(fp);

    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}
```

함수 이름: validate\_coco

입력:

* cfg: YOLO 모델의 설정 파일 경로를 나타내는 문자열 포인터
* weights: YOLO 모델의 가중치 파일 경로를 나타내는 문자열 포인터

동작:&#x20;

* 이 함수는 COCO 데이터셋에서 검증을 수행합니다. 주어진 cfg와 weights 파일로부터 YOLO 네트워크를 로드하고, COCO 데이터셋의 이미지를 읽어와 이를 이용해 객체 검출을 수행합니다.&#x20;
* 이 함수는 검출된 객체들의 위치와 클래스 정보를 coco\_results.json 파일에 저장합니다.

설명:&#x20;

* 이 함수는 YOLO 네트워크를 사용하여 COCO 데이터셋에서 객체 검출을 수행합니다.&#x20;
* 입력으로는 YOLO 모델의 설정 파일(cfg)과 가중치 파일(weights)의 경로를 받습니다.&#x20;
* COCO 데이터셋에서 검증 이미지의 경로를 가진 리스트(plist)를 얻은 후, 해당 경로의 이미지들을 읽어들여 YOLO 네트워크로 객체 검출을 수행합니다.&#x20;
* 이 함수는 검출된 객체들의 위치와 클래스 정보를 coco\_results.json 파일에 저장합니다.
* 이 함수에서는 다양한 변수들을 설정할 수 있습니다. threshold(thresh) 변수는 객체 검출의 임계값을 나타내며, nms 변수는 Non-Maximum Suppression(NMS)을 수행할 지 여부를 결정합니다.&#x20;
* iou\_thresh 변수는 NMS에서 사용할 IoU 임계값을 나타냅니다. nthreads 변수는 쓰레드 수를 결정합니다.
* 이 함수에서는 COCO 데이터셋에 대한 정보도 사용됩니다. COCO 데이터셋에서는 이미지마다 고유한 ID가 존재하는데, 해당 ID를 사용하여 검출된 객체들의 정보를 coco\_results.json 파일에 저장합니다. 클래스 정보를 얻기 위해서는 YOLO 네트워크의 마지막 레이어(l)를 사용합니다.



## validate\_coco\_recall

```c
void validate_coco_recall(char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    list *plist = get_paths("/home/pjreddie/data/voc/test/2007_test.txt");
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, coco_classes[j]);
        fps[j] = fopen(buff, "w");
    }

    int m = plist->size;
    int i=0;

    float thresh = .001;
    int nms = 0;
    float iou_thresh = .5;

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
        free_detections(dets, nboxes);
        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}
```

함수 이름: validate\_coco\_recall

입력:

* cfgfile: YOLO 모델의 구성 파일 경로
* weightfile: YOLO 모델의 가중치 파일 경로

동작:

* 주어진 cfgfile과 weightfile을 사용하여 YOLO 모델을 로드한다.
* 배치 크기를 1로 설정한다.
* 무작위 시드를 초기화한다.
* 테스트 이미지 경로가 포함된 파일을 읽어온다.
* YOLO 모델에서 출력 계층을 가져온다.
* 출력 계층에서 클래스 수와 네트워크 출력 크기를 가져온다.
* coco\_classes라는 배열을 사용하여 각 클래스의 결과를 저장할 파일을 열고 파일 포인터를 fps 배열에 저장한다.
* 테스트 이미지의 수를 계산한다.
* YOLO 모델을 사용하여 테스트 이미지를 예측한다.
* 예측된 결과를 토대로 NMS를 실행하여 중복된 검출 결과를 제거한다.
* 이미지에 대한 정답 레이블 파일 경로를 생성하고 레이블 파일을 읽어들인다.
* 네트워크 출력과 레이블 파일을 비교하여 평균 IOU와 정확도를 계산한다.

설명:&#x20;

* 이 함수는 COCO 데이터셋에서 YOLO 모델의 검출 결과를 검증하는 기능을 수행한다. 주어진 cfgfile과 weightfile을 사용하여 YOLO 모델을 로드하고, 각 클래스마다 검출 결과를 저장할 파일을 열고 파일 포인터를 저장한다.&#x20;
* 그리고 테스트 이미지 경로가 포함된 파일을 읽어온 후, YOLO 모델을 사용하여 예측을 실행하고, NMS를 사용하여 중복된 결과를 제거한 후, 정답 레이블과 비교하여 평균 IOU와 정확도를 계산한다.



## test\_coco

```c
void test_coco(char *cfgfile, char *weightfile, char *filename, float thresh)
{
    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    layer l = net->layers[net->n-1];
    set_batch_network(net, 1);
    srand(2222222);
    float nms = .4;
    clock_t time;
    char buff[256];
    char *input = buff;
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

        draw_detections(im, dets, l.side*l.side*l.n, thresh, coco_classes, alphabet, 80);
        save_image(im, "prediction");
        show_image(im, "predictions", 0);
        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}
```

함수 이름: run\_coco&#x20;

입력:

* argc (int): 명령행 인수(argument)의 수
* argv (char \*\*): 명령행 인수 배열

동작:&#x20;

* coco 데이터셋을 사용하여 객체 검출(object detection)을 실행하는 함수입니다. 함수의 인수로는 명령행 인수를 받아와서 필요한 인수들을 추출하고, 해당하는 검출 함수를 호출합니다.

설명:

* prefix (char \*): "-prefix" 옵션으로 주어진 문자열
* thresh (float): "-thresh" 옵션으로 주어진 실수값 (기본값 0.2)
* cam\_index (int): "-c" 옵션으로 주어진 정수값 (기본값 0)
* frame\_skip (int): "-s" 옵션으로 주어진 정수값 (기본값 0)
* cfg (char \*): coco 검출을 위한 darknet configuration 파일 경로
* weights (char \*): coco 검출을 위한 darknet 가중치 파일 경로 (옵션)
* filename (char \*): coco 검출 대상 이미지/비디오 파일 경로 (옵션)
* avg (int): "-avg" 옵션으로 주어진 정수값 (기본값 1)
* coco\_classes (char \*\*): coco 클래스 이름 배열
* 80: coco 데이터셋의 클래스 수
* .5: NMS (Non-Maximum Suppression) 임계값
* 0,0,0,0: yolo\_eval 함수에 전달되는 인자 (사용하지 않음)



## run\_coco

```c
void run_coco(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);

    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    int avg = find_int_arg(argc, argv, "-avg", 1);
    if(0==strcmp(argv[2], "test")) test_coco(cfg, weights, filename, thresh);
    else if(0==strcmp(argv[2], "train")) train_coco(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_coco(cfg, weights);
    else if(0==strcmp(argv[2], "recall")) validate_coco_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, coco_classes, 80, frame_skip, prefix, avg, .5, 0,0,0,0);
}
```

함수 이름: run\_coco 입력:

* argc: int형 변수. 명령행 인자의 개수를 나타낸다.
* argv: char형 포인터 배열. 명령행 인자를 가리키는 포인터 배열이다.&#x20;

동작:&#x20;

* coco 데이터셋을 이용하여 YOLOv3 모델을 학습하거나 검증하거나, 테스트하거나, 데모를 실행한다.&#x20;

설명:&#x20;

* 이 함수는 argc와 argv를 통해 전달된 인자들을 처리하여 coco 데이터셋을 이용하여 YOLOv3 모델을 학습하거나 검증하거나, 테스트하거나, 데모를 실행한다.&#x20;
* 함수 내부에서는 입력으로 전달된 인자들을 처리하기 위해 find\_char\_arg(), find\_float\_arg(), find\_int\_arg()와 같은 함수들이 사용되며, 인자들의 값에 따라 다양한 기능을 수행하게 된다.&#x20;
* 만약 인자의 개수가 부족하면 오류 메시지를 출력하고 함수를 종료한다.

