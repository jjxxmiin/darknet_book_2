# instance-segmenter

```c
#include "darknet.h"
#include <sys/time.h>
#include <assert.h>

void normalize_image2(image p);
```

## train\_isegmenter

```c
void train_isegmenter(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int display)
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

    image embed = pred;
    embed.c = 3;
    embed.data += embed.w*embed.h*80;

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
    args.num_boxes = 90;

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
    args.type = ISEG_DATA;

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
            pred.c = 80;
            image mask = mask_to_rgb(tr);
            image prmask = mask_to_rgb(pred);
            image ecopy = copy_image(embed);
            normalize_image2(ecopy);
            show_image(ecopy, "embed", 1);
            free_image(ecopy);

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

함수 이름: train\_isegmenter

입력:

* char \*datacfg: 데이터 설정 파일 경로
* char \*cfgfile: 모델 설정 파일 경로
* char \*weightfile: 사전 훈련된 가중치 파일 경로
* int \*gpus: GPU 인덱스 배열
* int ngpus: 사용할 GPU 개수
* int clear: 네트워크를 초기화할지 여부를 나타내는 플래그 (0 또는 1)
* int display: 중간 결과를 표시할지 여부를 나타내는 플래그 (0 또는 1)

동작:

* 주어진 데이터와 설정으로 이미지 세그멘테이션 네트워크를 훈련하는 함수이다.
* 네트워크를 초기화하고 설정에 따라 데이터를 로드하여 훈련을 수행한다.
* 다음 과정을 수행한다:
  1. 훈련에 필요한 변수와 설정 값을 초기화한다.
  2. 네트워크를 로드하고 학습률을 GPU 개수로 조정한다.
  3. 데이터 설정 파일에서 백업 디렉토리 경로와 훈련 데이터 리스트 파일 경로를 가져온다.
  4. 훈련 데이터의 경로를 가져온다.
  5. 훈련 데이터를 로드하기 위해 load\_data 함수를 호출한다.
  6. epoch 수를 계산하고, 최대 배치 수에 도달할 때까지 훈련을 반복한다.
  7. 데이터를 로드하고 네트워크를 훈련하여 손실 값을 계산한다.
  8. display 플래그가 설정된 경우 중간 결과를 표시한다.
  9. 평균 손실 값을 업데이트한다.
  10. 훈련 데이터를 해제한다.
  11. epoch이 변경되었을 때마다 가중치를 백업한다.
  12. 현재 배치가 100의 배수일 때마다 가중치를 백업한다.
  13. 훈련이 완료되면 최종 가중치를 저장한다.
  14. 메모리를 해제한다.

설명:

* 이 함수는 주어진 데이터와 설정으로 이미지 세그멘테이션 네트워크를 훈련하는 기능을 수행한다.
* 훈련 데이터를 로드하고 네트워크를 업데이트하여 손실을 최소화하는 것이 목표이다.
* 훈련 도중 중간 결과를 표시할 수 있으며, 훈련이 완료되면 최종 가중치를 저장한다.



## predict\_isegmenter

```c
void predict_isegmenter(char *datafile, char *cfg, char *weights, char *filename)
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

함수 이름: predict\_isegmenter

입력:

* char \*datafile: 데이터 파일 경로
* char \*cfg: 네트워크 설정 파일 경로
* char \*weights: 네트워크 가중치 파일 경로
* char \*filename: 이미지 파일 경로 (옵션)

동작:

* 이미지 세그멘테이션 네트워크를 사용하여 이미지를 예측하는 함수이다.
* 주어진 네트워크 설정 파일과 가중치 파일을 사용하여 네트워크를 로드하고 설정한다.
* 반복적으로 이미지를 입력받아 예측 결과를 출력한다.
* 입력 이미지 경로는 프로그램 실행 시 인자로 전달되거나 사용자에게 입력받을 수 있다.
* 다음과 같은 과정을 반복한다:
  1. 파일명이 주어진 경우 해당 파일을 입력 이미지로 사용한다. 파일명이 주어지지 않은 경우 사용자에게 이미지 파일 경로를 입력받는다.
  2. 입력 이미지를 컬러 이미지로 로드한다.
  3. 입력 이미지를 네트워크의 입력 크기에 맞게 리사이징한다.
  4. 리사이징된 이미지 데이터를 가져와서 네트워크로 예측을 수행한다.
  5. 네트워크의 출력을 시각화하여 표시한다.
  6. 예측 결과와 수행 시간을 출력한다.
  7. 사용한 이미지와 시각화 이미지를 메모리에서 해제한다.
  8. 파일명이 주어진 경우 반복문을 종료한다.

설명:

* 이 함수는 이미지 세그멘테이션 네트워크를 사용하여 이미지를 예측하는 함수이다.
* 주어진 네트워크 설정 파일과 가중치 파일을 사용하여 네트워크를 로드하고 설정한다.
* 사용자에게 이미지 파일 경로를 입력받거나 프로그램 실행 시 파일 경로를 인자로 전달받아 입력 이미지로 사용한다.
* 입력 이미지를 네트워크의 입력 크기에 맞게 리사이징한 후 네트워크로 예측을 수행한다.
* 예측 결과를 출력하고 수행 시간을 계산하여 표시한다.
* 이 함수는 반복적으로 이미지를 입력받아 예측하는데 사용되며, 입력 파일명이 주어진 경우에는 단일 이미지에 대한 예측만 수행한다.



## demo\_isegmenter

```c
void demo_isegmenter(char *datacfg, char *cfg, char *weights, int cam_index, const char *filename)
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

함수 이름: demo\_isegmenter

입력:

* char \*datacfg: 데이터 설정 파일 경로
* char \*cfg: 네트워크 설정 파일 경로
* char \*weights: 네트워크 가중치 파일 경로
* int cam\_index: 카메라 인덱스
* const char \*filename: 비디오 파일 경로 (옵션)

동작:

* 이미지 세그멘테이션 네트워크의 데모 실행을 제어하는 함수이다.
* 주어진 데이터 설정 파일, 네트워크 설정 파일, 가중치 파일을 사용하여 네트워크를 로드하고 설정한다.
* 비디오 스트림 또는 웹캠을 열어 데모를 실행한다.
* 다음과 같은 과정을 반복한다:
  1. 시간 측정을 시작한다.
  2. 스트림으로부터 이미지를 가져온다.
  3. 가져온 이미지를 네트워크의 입력 크기에 맞게 리사이징한다.
  4. 네트워크로 이미지를 전달하여 예측을 수행한다.
  5. 화면을 지우고 FPS 값을 출력한다.
  6. 네트워크의 출력을 시각화하여 "Segmenter"라는 창에 표시한다.
  7. 사용한 이미지와 시각화 이미지를 메모리에서 해제한다.
  8. 시간 측정을 종료하고 FPS 값을 계산한다.

설명:

* 이 함수는 이미지 세그멘테이션 네트워크의 데모 실행을 제어한다.
* 주어진 데이터 설정 파일, 네트워크 설정 파일, 가중치 파일을 사용하여 네트워크를 로드하고 설정한다.
* 비디오 스트림 또는 웹캠에서 프레임을 가져와서 네트워크를 통해 이미지 세그멘테이션을 수행하고 결과를 시각화하여 보여준다.
* 이 함수는 OpenCV가 사용 가능한 경우에만 컴파일된다.



## run\_isegmenter

```c
void run_isegmenter(int argc, char **argv)
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
    if(0==strcmp(argv[2], "test")) predict_isegmenter(data, cfg, weights, filename);
    else if(0==strcmp(argv[2], "train")) train_isegmenter(data, cfg, weights, gpus, ngpus, clear, display);
    else if(0==strcmp(argv[2], "demo")) demo_isegmenter(data, cfg, weights, cam_index, filename);
}
```

함수 이름: run\_isegmenter

입력:

* int argc: 명령행 인수의 개수
* char \*\*argv: 명령행 인수의 배열

동작:

* 이미지 세그멘테이션 네트워크의 실행을 제어하는 함수이다.
* 명령행 인수를 기반으로 훈련, 테스트, 데모 중 어떤 동작을 수행할지 결정한다.
* 다음 과정을 수행한다:
  1. 명령행 인수가 부족한 경우 사용법을 출력하고 종료한다.
  2. GPU 인덱스를 가져온다. "-gpus" 플래그가 제공된 경우 해당 GPU 인덱스를 가져온다.
  3. 훈련, 테스트 또는 데모를 수행할 GPU의 개수와 인덱스를 설정한다.
  4. 카메라 인덱스를 가져온다.
  5. "-clear" 플래그를 찾아 네트워크를 초기화할지 여부를 결정한다.
  6. "-display" 플래그를 찾아 중간 결과를 표시할지 여부를 결정한다.
  7. 데이터, 설정 및 가중치 파일의 경로를 가져온다.
  8. "test" 명령이 주어진 경우 이미지 세그멘테이션 네트워크를 테스트하는 predict\_isegmenter 함수를 호출한다.
  9. "train" 명령이 주어진 경우 이미지 세그멘테이션 네트워크를 훈련하는 train\_isegmenter 함수를 호출한다.
  10. "demo" 명령이 주어진 경우 이미지 세그멘테이션 네트워크를 데모하는 demo\_isegmenter 함수를 호출한다.

설명:

* 이 함수는 이미지 세그멘테이션 네트워크의 실행을 제어하는 기능을 수행한다.
* 주어진 명령행 인수를 기반으로 훈련, 테스트, 데모 중 어떤 동작을 수행할지 결정한다.
* 훈련, 테스트 또는 데모에 필요한 데이터와 설정 파일 경로, 가중치 파일 경로 등을 인수로 받아 해당 기능을 호출한다.

