# segmenter

```c
#include "darknet.h"
#include <sys/time.h>
#include <assert.h>
```

## train\_segmenter

```c
void train_segmenter(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int display)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
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
    image pred = get_network_image(net);

    int div = net->w/pred.w;
    assert(pred.w * div == net->w);
    assert(pred.h * div == net->h);

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *train_list = option_find_str(options, "train", "data/train.list");

    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.threads = 32;
    args.scale = div;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;
    args.classes = 80;

    args.paths = paths;
    args.n = imgs;
    args.m = N;
    args.type = SEGMENTATION_DATA;

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        double time = what_time_is_it_now();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time = what_time_is_it_now();

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
        if(display){
            image tr = float_to_image(net->w/div, net->h/div, 80, train.y.vals[net->batch*(net->subdivisions-1)]);
            image im = float_to_image(net->w, net->h, net->c, train.X.vals[net->batch*(net->subdivisions-1)]);
            image mask = mask_to_rgb(tr);
            image prmask = mask_to_rgb(pred);
            show_image(im, "input", 1);
            show_image(prmask, "pred", 1);
            show_image(mask, "truth", 100);
            free_image(mask);
            free_image(prmask);
        }
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, *net->seen);
        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    free_network(net);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}
```

함수 이름: train\_segmenter

입력:

* datacfg: 데이터 구성 파일의 경로를 나타내는 문자열 포인터
* cfgfile: 구성 파일의 경로를 나타내는 문자열 포인터
* weightfile: 가중치 파일의 경로를 나타내는 문자열 포인터
* gpus: GPU 번호 배열을 나타내는 정수 포인터
* ngpus: GPU 수를 나타내는 정수
* clear: 네트워크를 지울지 여부를 나타내는 정수 (0 또는 1)
* display: 이미지를 화면에 표시할지 여부를 나타내는 정수 (0 또는 1)

동작:

* 데이터 구성 파일, 구성 파일, 가중치 파일, GPU 번호, GPU 수, 네트워크 초기화 여부, 이미지 출력 여부를 입력으로 받아 세그멘테이션 네트워크를 학습하는 함수
* 함수 내부에서는 다중 GPU를 사용할 수 있도록 CUDA 함수를 사용하며, 데이터를 읽어들이고 세그멘테이션 네트워크를 훈련한다.
* epoch 마다 가중치 파일을 저장한다.

설명:&#x20;

* train\_segmenter 함수는 입력으로 데이터 구성 파일, 구성 파일, 가중치 파일, GPU 번호 배열, GPU 수, 네트워크 초기화 여부, 이미지 출력 여부를 받아 세그멘테이션 네트워크를 학습하는 함수이다.
* 함수 내부에서는 다중 GPU를 사용할 수 있도록 CUDA 함수를 사용하며, 데이터를 읽어들이고 세그멘테이션 네트워크를 훈련한다. 또한 epoch 마다 가중치 파일을 저장한다.



## predict\_segmenter

```c
void predict_segmenter(char *datafile, char *cfg, char *weights, char *filename)
{
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);
    srand(2222222);

    clock_t time;
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        image sized = letterbox_image(im, net->w, net->h);

        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(net, X);
        image pred = get_network_image(net);
        image prmask = mask_to_rgb(pred);
        printf("Predicted: %f\n", predictions[0]);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        show_image(sized, "orig", 1);
        show_image(prmask, "pred", 0);
        free_image(im);
        free_image(sized);
        free_image(prmask);
        if (filename) break;
    }
}
```

함수 이름: predict\_segmenter

입력:

* datafile: 사용되지 않음
* cfg: YOLO 모델의 설정 파일 경로
* weights: YOLO 모델의 가중치 파일 경로
* filename: 입력 이미지 파일 경로. 이 값이 존재하면 사용자가 이미지 파일 경로를 입력하지 않고도 함수를 호출할 수 있습니다.

동작:&#x20;

* 이 함수는 YOLO 모델을 사용하여 이미지의 객체를 인식하고, 객체 영역을 segmentation하여 시각화합니다.&#x20;
* 함수는 무한 루프를 실행하며, 사용자가 입력 이미지 파일 경로를 직접 제공하거나 filename 매개변수를 통해 미리 지정한 파일 경로를 사용합니다.&#x20;
* 함수는 입력 이미지를 로드하고, 크기를 조정하며, YOLO 모델을 사용하여 객체의 위치를 예측하고 시각화합니다.&#x20;
* 함수는 예측된 객체의 확률과 실행 시간을 출력합니다.

설명:

* char \*datafile: 사용되지 않는 매개변수입니다.
* char \*cfg: YOLO 모델의 설정 파일 경로를 나타내는 문자열 포인터입니다.
* char \*weights: YOLO 모델의 가중치 파일 경로를 나타내는 문자열 포인터입니다.
* char \*filename: 입력 이미지 파일 경로를 나타내는 문자열 포인터입니다. 이 값이 존재하면 사용자가 이미지 파일 경로를 입력하지 않아도 됩니다.
* 함수는 입력 이미지를 로드하고 크기를 조정합니다.
* YOLO 모델을 사용하여 객체 위치를 예측합니다.
* 예측된 객체의 확률과 실행 시간을 출력합니다.
* 함수는 입력 이미지, segmentation 결과 이미지 및 사용된 메모리를 해제합니다.



## demo\_segmenter

```c
void demo_segmenter(char *datacfg, char *cfg, char *weights, int cam_index, const char *filename)
{
#ifdef OPENCV
    printf("Classifier Demo\n");
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);

    srand(2222222);
    void * cap = open_video_stream(filename, cam_index, 0,0,0);

    if(!cap) error("Couldn't connect to webcam.\n");
    float fps = 0;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream(cap);
        image in_s = letterbox_image(in, net->w, net->h);

        network_predict(net, in_s.data);

        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nFPS:%.0f\n",fps);

        image pred = get_network_image(net);
        image prmask = mask_to_rgb(pred);
        show_image(prmask, "Segmenter", 10);

        free_image(in_s);
        free_image(in);
        free_image(prmask);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}
```

함수 이름: demo\_segmenter&#x20;

입력:

* char \*datacfg : 데이터 설정 파일 경로
* char \*cfg : 모델 설정 파일 경로
* char \*weights : 모델 가중치 파일 경로
* int cam\_index : 카메라 인덱스
* const char \*filename : 입력 동영상 파일 경로 (NULL일 수도 있음)

동작:&#x20;

* 입력으로 받은 이미지나 동영상을 이용하여 segmentation 모델의 성능을 실시간으로 보여줌

설명:

* OpenCV가 설치되어 있어야 함
* 모델 파일과 입력 데이터 파일을 이용하여 segmentation 모델을 불러옴
* 입력으로 카메라 인덱스나 동영상 파일을 받음
* while 루프를 통해 입력된 데이터를 지속적으로 처리하며, 처리 속도에 따라 FPS를 계산하여 출력함
* 입력 이미지를 모델 입력 크기에 맞게 resize한 후, 모델에 입력하여 segmentation 결과를 얻음
* 결과를 시각화하여 보여줌
* 메모리를 해제하고, 다음 입력 데이터를 처리함



## run\_segmenter

```c
void run_segmenter(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
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

    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int clear = find_arg(argc, argv, "-clear");
    int display = find_arg(argc, argv, "-display");
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) predict_segmenter(data, cfg, weights, filename);
    else if(0==strcmp(argv[2], "train")) train_segmenter(data, cfg, weights, gpus, ngpus, clear, display);
    else if(0==strcmp(argv[2], "demo")) demo_segmenter(data, cfg, weights, cam_index, filename);
}
```

함수 이름: run\_segmenter

입력:

* argc: int, 명령줄에서 입력된 인자의 개수
* argv: char\*\*, 명령줄에서 입력된 인자들의 배열

동작:

* 입력된 인자들을 파싱하여 train, test, demo에 맞게 각각의 함수를 호출하거나, 에러 메시지를 출력한다.
* gpu\_list, gpus, gpu, ngpus, cam\_index, clear, display, data, cfg, weights, filename 등의 변수를 초기화하고 각 함수에 인자로 전달한다.

설명:

* argc가 4보다 작으면 사용법 메시지를 출력하고 함수를 종료한다.
* gpu\_list가 있는 경우, 쉼표로 구분된 gpu 인덱스를 파싱하여 gpus에 저장한다.
* gpu\_list가 없는 경우, 전역 변수로 설정된 gpu\_index를 사용한다.
* cam\_index, clear, display, data, cfg, weights, filename은 각각 인자에서 추출한다.
* argv\[2]의 값에 따라 train, test, demo 함수 중 하나를 호출한다.

