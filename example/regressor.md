# regressor

```c
#include "darknet.h"
#include <sys/time.h>
#include <assert.h>
```

## train\_regressor

```c
void train_regressor(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
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

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *train_list = option_find_str(options, "train", "data/train.list");
    int classes = option_find_int(options, "classes", 1);

    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    clock_t time;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.threads = 32;
    args.classes = classes;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.n = imgs;
    args.m = N;
    args.type = REGRESSION_DATA;

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        time=clock();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();

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
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net->seen);
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

함수 이름: train\_regressor&#x20;

입력:

* datacfg: char 형식의 데이터 파일 경로
* cfgfile: char 형식의 네트워크 설정 파일 경로
* weightfile: char 형식의 미리 학습된 가중치 파일 경로
* gpus: int 배열 형식의 GPU 번호
* ngpus: int 형식의 GPU 개수
* clear: int 형식의 클리어 여부

동작:&#x20;

* 주어진 데이터 파일, 네트워크 설정 파일, 미리 학습된 가중치 파일, GPU 번호 및 개수를 사용하여 회귀 모델을 학습시킵니다.&#x20;
* 이 함수는 YOLO 알고리즘을 사용하여 학습을 수행합니다.&#x20;
* 학습 중에는 네트워크의 학습률, 모멘텀, 감쇠 및 데이터 경로 등의 정보가 표시됩니다.

입력을 설명할 때:

* char 형식의 데이터 파일 경로: 학습에 사용될 데이터 파일의 경로
* char 형식의 네트워크 설정 파일 경로: 네트워크 설정 파일의 경로
* char 형식의 미리 학습된 가중치 파일 경로: 미리 학습된 가중치 파일의 경로
* int 배열 형식의 GPU 번호: 사용할 GPU의 번호
* int 형식의 GPU 개수: 사용할 GPU의 개수
* int 형식의 클리어 여부: 1이면 이전의 학습 결과를 지우고 새로운 학습을 시작합니다. 0이면 이전의 학습 결과를 유지하고 새로운 학습을 시작합니다.



## predict\_regressor

```c
void predict_regressor(char *cfgfile, char *weightfile, char *filename)
{
    network *net = load_network(cfgfile, weightfile, 0);
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
        printf("Predicted: %f\n", predictions[0]);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}
```

함수 이름: predict\_regressor&#x20;

입력:

* cfgfile: 신경망 구성 파일 경로를 가리키는 문자열 포인터
* weightfile: 학습된 가중치 파일 경로를 가리키는 문자열 포인터
* filename: 입력 이미지 파일 경로를 가리키는 문자열 포인터 (선택적)

동작:&#x20;

* 이 함수는 cfgfile과 weightfile을 사용하여 학습된 회귀 신경망을 로드합니다.&#x20;
* 사용자가 이미지 파일 경로를 입력하거나 파일 이름이 이미 함수 호출 시 제공되었다면, 함수는 해당 이미지를 로드하고 신경망을 통해 예측을 수행합니다. 예측 결과는 콘솔에 출력됩니다.

설명:&#x20;

* 이 함수는 Darknet 라이브러리를 사용하여 작성된 함수입니다.&#x20;
* 이 함수는 회귀 문제를 푸는 데 사용되는 신경망 모델을 로드하고, 이미지 파일 경로를 입력받아 해당 이미지를 예측합니다.&#x20;
* 이 함수는 이미지 파일 경로를 입력으로 받을 수도 있고, 함수를 호출하는 시점에서 이미 파일 이름이 제공된 경우에는 파일 이름을 인자로 전달할 수 있습니다.&#x20;
* 이 함수는 입력 이미지를 로드하고, letterbox\_image 함수를 사용하여 이미지를 모델에 맞는 크기로 변환한 다음, 네트워크를 통해 예측을 수행합니다.&#x20;
* 마지막으로, 예측 결과를 콘솔에 출력합니다.



## demo\_regressor

```c
void demo_regressor(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
    printf("Regressor Demo\n");
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    list *options = read_data_cfg(datacfg);
    int classes = option_find_int(options, "classes", 1);
    char *name_list = option_find_str(options, "names", 0);
    char **names = get_labels(name_list);

    void * cap = open_video_stream(filename, cam_index, 0,0,0);
    if(!cap) error("Couldn't connect to webcam.\n");
    float fps = 0;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream(cap);
        image crop = center_crop_image(in, net->w, net->h);
        grayscale_image_3c(crop);

        float *predictions = network_predict(net, crop.data);

        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nFPS:%.0f\n",fps);

        int i;
        for(i = 0; i < classes; ++i){
            printf("%s: %f\n", names[i], predictions[i]);
        }

        show_image(crop, "Regressor", 10);
        free_image(in);
        free_image(crop);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}
```

함수 이름: demo\_regressor&#x20;

입력:

* datacfg: char 형식의 데이터 파일 경로
* cfgfile: char 형식의 네트워크 구성 파일 경로
* weightfile: char 형식의 가중치 파일 경로
* cam\_index: int 형식의 카메라 인덱스 (0 이상)
* filename: char 형식의 동영상 파일 경로 또는 NULL

동작:

* OpenCV 라이브러리를 사용하여 카메라 또는 동영상에서 입력 이미지를 받아와서,
* 받아온 이미지를 네트워크 입력 크기에 맞게 중앙을 기준으로 자르고 그레이스케일로 변환하여,
* 네트워크를 사용하여 이미지를 예측하고 결과를 출력하며,
* 예측 속도를 계산하여 화면에 출력한다.

설명:

* 입력된 파일 경로로부터 데이터, 구성 및 가중치 파일을 읽어서 네트워크를 생성한다.
* 카메라 인덱스가 0보다 큰 경우는 해당 인덱스의 카메라로부터 이미지를 가져오고, filename 인자가 NULL이 아닌 경우는 해당 파일로부터 이미지를 가져온다.
* 받아온 이미지를 중앙을 기준으로 자르고 그레이스케일로 변환한다.
* 네트워크를 사용하여 이미지를 예측하고, 결과를 출력한다.
* 결과는 클래스 레이블과 해당 클래스에 대한 예측 값으로 이루어져 있다.
* 예측 속도를 계산하여 화면에 출력한다.



## run\_regressor

```c
void run_regressor(int argc, char **argv)
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
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) predict_regressor(data, cfg, weights);
    else if(0==strcmp(argv[2], "train")) train_regressor(data, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "demo")) demo_regressor(data, cfg, weights, cam_index, filename);
}
```

함수 이름: run\_regressor

입력:

* int argc: 프로그램 실행 시 전달된 인자의 개수
* char \*\*argv: 프로그램 실행 시 전달된 인자들의 배열

동작:

* 입력으로 받은 인자들을 기반으로, regressor를 테스트, 훈련, 데모 실행하는 함수
* 인자로 "test"가 전달된 경우 predict\_regressor() 함수를 실행하여 regressor 모델을 테스트
* 인자로 "train"이 전달된 경우 train\_regressor() 함수를 실행하여 regressor 모델을 훈련
* 인자로 "demo"가 전달된 경우 demo\_regressor() 함수를 실행하여 regressor 모델의 데모 실행
* "-gpus" 인자를 통해 GPU ID를 설정할 수 있으며, 쉼표로 구분된 문자열 형태로 전달
* "-c" 인자를 통해 카메라 장치의 인덱스를 설정할 수 있음
* "-clear" 인자를 전달하면, 훈련 시 이전에 저장된 모든 기록 삭제
* 인자로 전달된 데이터 파일, 설정 파일, 가중치 파일, 파일 이름 등을 적절히 파싱하여 사용

설명:

* run\_regressor() 함수는 Darknet 프레임워크의 regressor 모델을 테스트, 훈련, 데모 실행하는 함수입니다.
* 입력으로 받은 인자들을 기반으로, 각각의 동작을 수행합니다.
* "-gpus" 인자를 통해 GPU ID를 설정할 수 있으며, 쉼표로 구분된 문자열 형태로 전달할 수 있습니다. 만약 이 인자가 전달되지 않은 경우, 기본적으로 GPU 0번을 사용합니다.
* "-c" 인자를 통해 카메라 장치의 인덱스를 설정할 수 있습니다. 이 인자가 전달되지 않은 경우, 기본값 0을 사용합니다.
* "-clear" 인자를 전달하면, 훈련 시 이전에 저장된 모든 기록을 삭제합니다.
* 인자로 전달된 데이터 파일, 설정 파일, 가중치 파일, 파일 이름 등을 적절히 파싱하여 사용합니다. 이 함수는 predict\_regressor(), train\_regressor(), demo\_regressor() 함수를 호출하여 regressor 모델을 테스트, 훈련, 데모 실행합니다.

