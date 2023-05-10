# detector

```c
#include "darknet.h"

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};
```

## train\_detector

```c
void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    data train, buffer;

    layer l = net->layers[net->n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    //args.type = INSTANCE_DATA;
    args.threads = 64;

    pthread_t load_thread = load_data(args);
    double time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32;
            if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            #pragma omp parallel for
            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        /*
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           if(!b.x) break;
           printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
           }
         */
        /*
           int zz;
           for(zz = 0; zz < train.X.cols; ++zz){
           image im = float_to_image(net->w, net->h, 3, train.X.vals[zz]);
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[zz] + k*5, 1);
           printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
           draw_bbox(im, b, 1, 1,0,0);
           }
           show_image(im, "truth11");
           cvWaitKey(0);
           save_image(im, "truth11");
           }
         */

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);

        time=what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);
        if(i%100==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        if(i%10000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}
```

함수 이름: train\_detector&#x20;

입력:

* datacfg: 문자열 포인터, 데이터 파일 경로를 지정하는 cfg 파일 경로
* cfgfile: 문자열 포인터, 모델의 구조를 지정하는 .cfg 파일 경로
* weightfile: 문자열 포인터, 사전 학습된 모델의 가중치 파일 경로
* gpus: 정수 포인터, 사용할 GPU 디바이스 번호 배열
* ngpus: 정수, 사용할 GPU 디바이스 수
* clear: 정수, 1이면 네트워크의 이전 상태를 지우고 새로운 상태로 시작, 0이면 계속 이어서 학습

동작:&#x20;

* 입력으로 주어진 데이터(cfg, cfgfile, weightfile)를 이용하여 네트워크 모델을 학습시키는 함수입니다.&#x20;
* 입력으로 받은 ngpus 개수만큼의 GPU 디바이스를 사용하여 병렬 학습을 수행합니다.&#x20;
* 학습 중에는 입력 이미지를 불러와 네트워크에 입력으로 제공하고, 그에 따른 손실 값을 계산하여 가중치를 업데이트합니다.&#x20;
* 학습 중에는 중간중간에 네트워크의 가중치를 저장할 수도 있습니다.

설명:&#x20;

* 이 함수는 darknet 프레임워크에서 사용되는 함수로, YOLO 객체 검출 알고리즘을 학습시키는데 사용됩니다.&#x20;
* 이 함수는 입력으로 받은 cfg, cfgfile, weightfile을 이용하여 네트워크 모델을 초기화한 후, ngpus 개수만큼의 GPU를 사용하여 네트워크를 병렬 학습시킵니다.&#x20;
* 이 함수에서 사용하는 load\_data 함수는 데이터를 비동기적으로 로드하며, 그에 따른 속도 향상을 가져옵니다.&#x20;
* 함수 내부에서는 네트워크의 학습률, 모멘텀, 가중치 감소 계수 등을 출력하며, 학습이 진행됨에 따라 현재까지의 평균 손실 값을 출력합니다.&#x20;
* 학습 중에는 네트워크의 가중치를 저장하여, 학습을 중단하고 다시 시작할 때 이전 상태에서 이어서 학습할 수 있도록 합니다.



## get\_coco\_image\_id

```c
static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if(c) p = c;
    return atoi(p+1);
}
```

함수 이름: get\_coco\_image\_id&#x20;

입력:

* filename (문자열 포인터)&#x20;

동작:&#x20;

* 입력된 파일 경로에서 마지막으로 '/'가 등장하는 위치와 '\__'_가 등장하는 위치를 찾아, 그 중 더 나중에 등장한 위치를 p 변수에 저장한다. 만약 _'\__'가 존재한다면, p 변수를 c 변수로 대체한다.&#x20;
* 그리고 p 포인터가 가리키는 문자열에서 숫자 부분을 추출하여 정수형으로 반환한다.&#x20;

설명:&#x20;

* COCO 데이터셋에서 이미지 파일 이름의 일부에는 해당 이미지의 ID 값이 포함되어 있다.&#x20;
* 이 함수는 파일 이름에서 ID 값을 추출하기 위해 사용된다.&#x20;
* 입력으로는 파일 이름을 전달하며, 파일 이름에 포함된 ID 값을 정수형으로 반환한다.



## print\_cocos

```c
static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
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

* FILE \*fp: 출력 파일 포인터
* char \*image\_path: 이미지 파일 경로
* detection \*dets: 객체 검출 결과 배열 포인터
* int num\_boxes: 검출된 객체의 수
* int classes: 클래스의 수
* int w: 이미지의 폭
* int h: 이미지의 높이

동작:

* COCO 형식으로 검출된 객체를 출력 파일에 쓰는 함수이다.
* get\_coco\_image\_id 함수를 이용하여 이미지의 ID를 가져온다.
* 각 객체에 대해 경계 상자(bounding box)를 COCO 형식으로 변환하여 출력 파일에 쓴다.

설명:

* 함수는 COCO 형식으로 검출된 객체를 출력 파일에 쓴다.
* 먼저 이미지 경로를 이용하여 이미지 ID를 가져온다.
* 각 객체의 경계 상자를 COCO 형식으로 변환하여 출력 파일에 쓴다.
* 출력 파일에는 "image\_id", "category\_id", "bbox", "score"의 정보가 포함된다.
* 출력 파일은 JSON 형식으로 작성된다.
* 각 객체의 확률(prob)이 0보다 큰 경우에만 출력된다.



## print\_detector\_detections

```c
void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                    xmin, ymin, xmax, ymax);
        }
    }
}
```

함수 이름: print\_detector\_detections

입력:

* FILE \*\*fps: 출력 파일 포인터의 배열
* char \*id: 이미지 ID
* detection \*dets: 객체 탐지 결과 배열
* int total: 객체 탐지 결과 배열의 크기
* int classes: 클래스 수
* int w: 이미지 가로 길이
* int h: 이미지 세로 길이

동작:

* 객체 탐지 결과(detection) 배열을 가져와서, 이미지의 ID, 클래스, 박스 좌표(x, y, w, h) 및 클래스 확률을 포함하는 텍스트 파일을 출력한다.

설명:

* 함수는 객체 탐지 결과(detection)를 가져와서, 해당 객체가 속한 클래스의 텍스트 파일 포인터를 가져온다.
* 그리고 해당 객체의 클래스 확률이 0이 아닌 경우, 객체의 이미지 ID, 클래스 ID, 박스 좌표(x, y, w, h) 및 클래스 확률을 해당 클래스의 텍스트 파일에 출력한다.
* 박스 좌표는 이미지의 경계를 벗어나지 않도록 조정된다.
* 이 함수는 YOLO 알고리즘의 출력 결과를 기반으로 하는 객체 탐지 알고리즘에서 사용된다.



## print\_imagenet\_detections

```c
void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
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
            int class = j;
            if (dets[i].prob[class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j+1, dets[i].prob[class],
                    xmin, ymin, xmax, ymax);
        }
    }
}
```

함수 이름: print\_imagenet\_detections

입력:

* FILE\* fp: 출력 파일 포인터
* int id: 이미지 ID
* detection\* dets: 객체 검출 결과
* int total: 객체 개수
* int classes: 클래스 개수
* int w: 이미지 가로 크기
* int h: 이미지 세로 크기

동작:&#x20;

* 이미지넷 포맷으로 객체 검출 결과를 출력하는 함수이다.&#x20;
* 각 객체의 좌표와 클래스별 확률을 출력 파일 포인터에 쓴다.

설명:

* 함수 내부에서 for문을 돌면서, 모든 객체에 대해 아래의 작업을 수행한다.
  * 객체의 bounding box 좌표를 xmin, ymin, xmax, ymax 변수에 저장한다.
  * 이미지 경계를 벗어나는 경우, 경계 내부로 조정한다.
  * 클래스 개수만큼 for문을 돌며, 해당 클래스의 확률이 0이 아닌 경우, 출력 파일 포인터에 객체 정보를 쓴다.
  * 이미지 ID, 클래스 ID, 확률, xmin, ymin, xmax, ymax 순으로 출력한다.



## validate\_detector\_flip

```c
void validate_detector_flip(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 2);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    image input = make_image(net->w, net->h, net->c*2);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
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
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data, 1);
            flip_image(val_resized[t]);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data + net->w*net->h*net->c, 1);

            network_predict(net, input.data);
            int w = val[t].w;
            int h = val[t].h;
            int num = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &num);
            if (nms) do_nms_sort(dets, num, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, num, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, num, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, num, classes, w, h);
            }
            free_detections(dets, num);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR);
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}
```

함수 이름: validate\_detector\_flip

입력:

* datacfg: 데이터 파일 경로를 지정하는 문자열 포인터
* cfgfile: 모델 구성 파일 경로를 지정하는 문자열 포인터
* weightfile: 모델 가중치 파일 경로를 지정하는 문자열 포인터
* outfile: 출력 파일 이름을 지정하는 문자열 포인터

동작:

* 주어진 모델 파일(cfgfile, weightfile)과 데이터 파일(datacfg)을 사용하여 네트워크를 로드합니다.
* valid\_images에 지정된 경로에서 이미지 리스트를 읽어들입니다.
* name\_list에 지정된 경로에서 클래스 이름 리스트를 읽어들입니다.
* coco 또는 imagenet 평가 방식이 지정되면 각각 coco\_results 또는 imagenet-detection 파일에 출력합니다. 그렇지 않으면 comp4\_det\_test\_ 이름으로 클래스마다 개별 출력 파일을 생성합니다.
* 이미지를 불러들이고 네트워크에서 예측합니다.
* 다중 스레드를 사용하여 이미지를 비동기적으로 불러들입니다.
* 예측된 결과에 대해 nms(non-maximum suppression) 및 클래스별로 정렬을 수행합니다.

설명:&#x20;

* 이 함수는 Darknet 프레임워크의 YOLO 객체 검출 모델의 성능을 평가하는 함수입니다.&#x20;
* validate\_detector 함수와 유사하지만, 이미지를 수평 방향으로 뒤집어서 다시 예측하는 "flip" 기능이 추가되었습니다.&#x20;
* 다중 스레드를 사용하여 이미지를 비동기적으로 처리하여 처리 속도를 높입니다.&#x20;
* 출력 파일의 형식은 coco 또는 imagenet 방식을 따르거나 클래스별로 개별 파일로 출력할 수 있습니다.



## validate\_detector

```c
void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
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
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &nboxes);
            if (nms) do_nms_sort(dets, nboxes, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, nboxes, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, nboxes, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, nboxes, classes, w, h);
            }
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR);
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}
```

함수 이름: validate\_detector&#x20;

입력:

* char \*datacfg: 데이터 구성 파일 경로를 가리키는 문자열 포인터
* char \*cfgfile: 모델 구성 파일 경로를 가리키는 문자열 포인터
* char \*weightfile: 모델 가중치 파일 경로를 가리키는 문자열 포인터
* char \*outfile: 출력 파일 경로를 가리키는 문자열 포인터

동작:&#x20;

* 주어진 모델과 데이터를 이용하여 검출기(valid detector)를 검증하는 함수입니다.&#x20;
* 함수는 입력된 데이터 구성 파일, 모델 구성 파일, 모델 가중치 파일, 출력 파일 경로를 기반으로 모델을 로드하고, 검증 데이터를 가져옵니다.&#x20;
* 가져온 검증 데이터를 이용하여 모델을 평가하고, 각 객체 검출의 예측 결과를 출력 파일에 저장합니다.

설명:

* options: 데이터 구성 파일에서 읽어온 옵션 리스트
* valid\_images: 검증 데이터 리스트 파일 경로를 가리키는 문자열 포인터
* name\_list: 객체 클래스 이름 리스트 파일 경로를 가리키는 문자열 포인터
* prefix: 출력 파일 경로의 prefix를 가리키는 문자열 포인터
* names: 객체 클래스 이름 배열
* mapf: 클래스 이름 매핑 파일 경로를 가리키는 문자열 포인터
* map: 클래스 이름 매핑 배열
* net: 로드된 모델
* l: 모델의 마지막 레이어
* classes: 객체 클래스의 개수
* buff: 문자열 버퍼
* type: 검증 데이터의 형식(voc, coco, imagenet)
* fp: 출력 파일의 파일 포인터
* fps: 클래스별 출력 파일의 파일 포인터 배열
* coco: coco 데이터인지 여부
* imagenet: imagenet 데이터인지 여부
* thresh: 객체 검출을 위한 임계값
* nms: 비최대 억제 임계값
* nthreads: 사용할 스레드의 개수
* val, val\_resized, buf, buf\_resized: 이미지와 크기가 조정된 이미지의 배열
* thr: 이미지 로딩을 위한 스레드 배열
* args: 이미지 로딩 인자 구조체
* i, t: 반복문 인덱스
* start: 함수 시작 시간
* plist: 검증 데이터 리스트
* paths: 검증 데이터 경로 배열
* id: 검증 데이터의 기본 파일 이름 (확장자 제외)
* X: 모델에 입력될 이미지 데이터 배열
* w, h: 입력 이미지의 가로, 세로 크기
* nboxes: 검출된 객체의 개수
* dets: 객체 검출 정보를 담은 배열



## validate\_detector\_recall

```c
void validate_detector_recall(char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths("data/coco_val_5k.list");
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];

    int j, k;

    int m = plist->size;
    int i=0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = .4;

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
        detection *dets = get_network_boxes(net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes);
        if (nms) do_nms_obj(dets, nboxes, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < nboxes; ++k){
            if(dets[k].objectness > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < l.w*l.h*l.n; ++k){
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
        free(id);
        free_image(orig);
        free_image(sized);
    }
}
```

함수 이름: validate\_detector\_recall

입력:

* cfgfile: YOLO 모델 구성 파일 경로
* weightfile: YOLO 모델 가중치 파일 경로

동작:&#x20;

* 주어진 cfgfile과 weightfile로 YOLO 모델을 로드하고, coco\_val\_5k.list 파일에 있는 이미지를 이용하여 검출 결과를 검증한다.&#x20;
* 검출 결과와 실제 라벨 사이의 IoU(IoU threshold는 0.5로 고정)를 계산하고, 이를 이용하여 Recall을 계산한다.

설명:&#x20;

* 주어진 YOLO 모델(cfgfile, weightfile)을 로드하고, coco\_val\_5k.list 파일에 있는 이미지들을 이용하여 검출 결과를 검증한다.&#x20;
* 모델을 이용하여 이미지에서 객체를 검출한 후, 검출된 객체와 실제 라벨 사이의 IoU(IoU threshold는 0.5로 고정)를 계산하고, 이를 이용하여 Recall 값을 계산한다.&#x20;
* 이 과정을 모든 이미지에 대해 반복하며, 총 검출된 객체 수, 정확하게 검출된 객체 수, 전체 라벨 수, 그리고 RPs/Img(이미지 당 평균 추론 수), 평균 IoU, Recall 값을 출력한다.



## test\_detector

```c
void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
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
        image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1];


        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            make_window("predictions", 512, 512, 0);
            show_image(im, "predictions", 0);
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}
```

함수 이름: test\_detector&#x20;

입력:

* datacfg: 데이터 설정 파일 경로 (문자열)
* cfgfile: 네트워크 설정 파일 경로 (문자열)
* weightfile: 학습된 가중치 파일 경로 (문자열)
* filename: 입력 이미지 파일 경로 (문자열 또는 NULL)
* thresh: 객체 탐지를 위한 임계값 (실수)
* hier\_thresh: 계층적 임계값 (실수)
* outfile: 출력 이미지 파일 경로 (문자열 또는 NULL)
* fullscreen: 전체 화면 모드 여부 (정수)

동작: 입력 이미지를 객체 탐지하기 위해 Darknet 라이브러리를 사용하여 처리하고, 결과 이미지를 출력하는 함수입니다.

* read\_data\_cfg 함수를 이용하여 데이터 설정 파일을 읽어들입니다.
* option\_find\_str 함수를 이용하여 데이터 설정 파일에서 names 키 값을 찾아서 name\_list 변수에 저장합니다.
* get\_labels 함수를 이용하여 names 파일에서 클래스 이름을 가져와 names 변수에 저장합니다.
* load\_alphabet 함수를 이용하여 알파벳 이미지 데이터를 가져와 alphabet 변수에 저장합니다.
* load\_network 함수를 이용하여 네트워크 설정 파일과 학습된 가중치 파일을 로드하고, net 변수에 저장합니다.
* set\_batch\_network 함수를 이용하여 배치 크기를 1로 설정합니다.
* srand 함수를 이용하여 시드값을 설정합니다.
* 입력 이미지 파일 경로가 주어졌을 경우, input 변수에 파일 경로를 복사합니다.
* 입력 이미지 파일 경로가 주어지지 않았을 경우, 표준 입력으로부터 입력 이미지 파일 경로를 받아 input 변수에 저장합니다.
* load\_image\_color 함수를 이용하여 입력 이미지를 로드하고, im 변수에 저장합니다.
* letterbox\_image 함수를 이용하여 입력 이미지를 네트워크 입력 크기로 변환한 후, sized 변수에 저장합니다.
* network->layers\[net->n-1]을 이용하여 출력 레이어의 정보를 l 변수에 저장합니다.
* sized.data를 이용하여 네트워크에 입력할 데이터 X를 생성합니다.
* network\_predict 함수를 이용하여 객체 탐지를 수행합니다.
* get\_network\_boxes 함수를 이용하여 탐지된 객체 정보를 가져와 dets 변수에 저장합니다.
* nms가 0보다 큰 경우, do\_nms\_sort 함수를 이용하여 non-maximum suppression을 수행합니다.
* draw\_detections 함수를 이용하여 탐지된 객체에 대한 경계 상자를 이미지에 그립니다.
* free\_detections 함수를 이용하여 탐지된 객체 정보를 메모리에서 해제합니다.
* outfile이 주어졌을 경우, save\_image 함수를 이용하여 출력 이미지를 파일로 저장합니다.
* outfile이 주어지지 않았을 경우, save\_image 함수를 이용하여 출력 이미지를 "predictions"라는 파일 이름으로 저장합니다.



## run\_detector

```c
void run_detector(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .5);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int avg = find_int_arg(argc, argv, "-avg", 3);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    int width = find_int_arg(argc, argv, "-w", 0);
    int height = find_int_arg(argc, argv, "-h", 0);
    int fps = find_int_arg(argc, argv, "-fps", 0);
    //int class = find_int_arg(argc, argv, "-class", 0);

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "valid2")) validate_detector_flip(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "recall")) validate_detector_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
    }
    //else if(0==strcmp(argv[2], "extract")) extract_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
    //else if(0==strcmp(argv[2], "censor")) censor_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
}
```

함수 이름: run\_detector

입력:

* argc: 정수 값으로, 전달된 명령행 인수(argument)의 개수를 나타냅니다.
* argv: 문자열 배열로, 전달된 명령행 인수의 값들을 저장합니다.

동작:

1. argc와 argv를 사용하여 여러 인수들을 추출하고 저장합니다.
2. argc가 4보다 작을 경우, 사용법을 출력하고 함수를 종료합니다.
3. \-gpus 옵션에 대한 값을 추출하고, ,를 기준으로 분리하여 각 GPU 번호를 저장합니다.
4. GPU 번호가 지정되지 않은 경우, 기본 GPU 번호를 사용합니다.
5. \-clear, -fullscreen, -w, -h, -fps 등의 인수들을 추출하고 저장합니다.
6. datacfg, cfg, weights, filename 등의 파일 경로를 추출하고 저장합니다.
7. argv\[2] 값에 따라 다른 동작을 수행합니다:
   * "test": test\_detector 함수를 호출하여 디텍터를 테스트합니다.
   * "train": train\_detector 함수를 호출하여 디텍터를 훈련합니다.
   * "valid": validate\_detector 함수를 호출하여 디텍터를 검증합니다.
   * "valid2": validate\_detector\_flip 함수를 호출하여 디텍터를 뒤집어서 검증합니다.
   * "recall": validate\_detector\_recall 함수를 호출하여 디텍터의 리콜(recall) 값을 검증합니다.
   * "demo": 다양한 옵션들을 설정한 후 demo 함수를 호출하여 실시간 데모를 실행합니다.

설명:&#x20;

* 위의 코드는 run\_detector라는 함수를 정의한 것으로 보입니다.&#x20;
* 이 함수는 전달된 명령행 인수를 분석하고, 해당 인수에 따라 다른 동작을 수행합니다.&#x20;
* 함수의 동작은 인수들을 추출하여 필요한 값들을 초기화한 후, 해당하는 동작 함수를 호출합니다.

