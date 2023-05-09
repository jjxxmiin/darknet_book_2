# writing

```c
#include "darknet.h"
```

## train\_writing

```c
void train_writing(char *cfgfile, char *weightfile)
{
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    list *plist = get_paths("figures.list");
    char **paths = (char **)list_to_array(plist);
    clock_t time;
    int N = plist->size;
    printf("N: %d\n", N);
    image out = get_network_image(net);

    data train, buffer;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.out_w = out.w;
    args.out_h = out.h;
    args.paths = paths;
    args.n = imgs;
    args.m = N;
    args.d = &buffer;
    args.type = WRITING_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    int epoch = (*net.seen)/N;
    while(get_current_batch(net) < net.max_batches || net.max_batches == 0){
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);
        printf("Loaded %lf seconds\n",sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);

        /*
           image pred = float_to_image(64, 64, 1, out);
           print_image(pred);
         */

        /*
           image im = float_to_image(256, 256, 3, train.X.vals[0]);
           image lab = float_to_image(64, 64, 1, train.y.vals[0]);
           image pred = float_to_image(64, 64, 1, out);
           show_image(im, "image");
           show_image(lab, "label");
           print_image(lab);
           show_image(pred, "pred");
           cvWaitKey(0);
         */

        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
        free_data(train);
        if(get_current_batch(net)%100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s_batch_%ld.weights", backup_directory, base, get_current_batch(net));
            save_weights(net, buff);
        }
        if(*net.seen/N > epoch){
            epoch = *net.seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
    }
}
```

함수 이름: train\_writing

입력:

* cfgfile: YOLO 모델의 구성 파일 경로를 나타내는 문자열 포인터
* weightfile: 사전 학습된 모델 가중치 파일 경로를 나타내는 문자열 포인터

동작:&#x20;

* 주어진 cfgfile을 사용하여 YOLO 모델을 구성하고, weightfile이 제공되는 경우 사전 학습된 가중치를 로드합니다.&#x20;
* 그 후, 명시된 배치 크기로 데이터를 불러와 네트워크를 학습시킵니다.&#x20;
* 매 에포크 끝에 가중치를 저장하고, 매 100 배치마다 중간 결과를 백업 디렉토리에 저장합니다.

설명:

* basecfg(cfgfile) 함수를 사용하여 모델의 구성 파일에서 base 이름을 가져옵니다.
* YOLO 모델을 구성하기 위해 parse\_network\_cfg(cfgfile) 함수를 사용합니다.
* weightfile이 제공된 경우, load\_weights(\&net, weightfile) 함수를 사용하여 사전 학습된 가중치를 로드합니다.
* 네트워크의 학습률, 모멘텀, 감쇠값을 출력합니다.
* get\_paths("figures.list") 함수를 사용하여 이미지 파일 경로가 포함 된 리스트를 가져옵니다.
* load\_data\_in\_thread(args) 함수를 사용하여 학습 데이터를 비동기적으로 로드합니다.
* train\_network(net, train) 함수를 사용하여 네트워크를 학습시키고 손실을 반환합니다.
* get\_current\_batch(net) 함수를 사용하여 현재 배치를 가져오고, max\_batches를 초과하지 않는 한 네트워크를 계속 학습합니다.
* save\_weights(net, buff) 함수를 사용하여 매 100 배치마다 중간 결과를 백업 디렉토리에 저장합니다.
* epoch를 저장하고 매 에포크 끝에 가중치를 저장합니다.



## test\_writing

```c
void test_writing(char *cfgfile, char *weightfile, char *filename)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
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
        resize_network(&net, im.w, im.h);
        printf("%d %d %d\n", im.h, im.w, im.c);
        float *X = im.data;
        time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        image pred = get_network_image(net);

        image upsampled = resize_image(pred, im.w, im.h);
        image thresh = threshold_image(upsampled, .5);
        pred = thresh;

        show_image(pred, "prediction");
        show_image(im, "orig");
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif

        free_image(upsampled);
        free_image(thresh);
        free_image(im);
        if (filename) break;
    }
}
```

함수 이름: test\_writing

입력:

* cfgfile (char\*): YOLO 모델의 구성 파일 경로
* weightfile (char\*): YOLO 모델의 가중치 파일 경로 (optional)
* filename (char\*): 테스트할 이미지 파일 경로 (optional)

동작:&#x20;

* 주어진 YOLO 모델(cfgfile)과 가중치(weightfile)를 사용하여 이미지 파일(filename)에서 손으로 쓴 문자를 인식하는 테스트를 수행합니다.&#x20;
* 만약 filename이 주어지지 않으면 사용자로부터 이미지 경로를 입력받습니다.&#x20;
* 이미지를 처리하여 모델의 예측을 출력하고, OpenCV를 사용하여 예측된 이미지와 원본 이미지를 보여줍니다.

설명:&#x20;

* 이 함수는 YOLO 모델을 사용하여 손글씨 인식을 테스트하는 코드를 담고 있습니다.&#x20;
* 이미지를 입력으로 받아 모델을 실행하여 손글씨가 어떤 문자를 나타내는지 예측합니다.&#x20;
* 이 예측된 이미지와 원본 이미지를 비교하여 출력합니다.&#x20;
* 모델 구성 파일(cfgfile)과 가중치 파일(weightfile)이 필요하며, 필요한 경우 테스트할 이미지 파일(filename)도 지정할 수 있습니다.



## run\_writing

```c
void run_writing(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5] : 0;
    if(0==strcmp(argv[2], "train")) train_writing(cfg, weights);
    else if(0==strcmp(argv[2], "test")) test_writing(cfg, weights, filename);
}
```

함수 이름: run\_writing

입력:

* argc: int형 변수, 명령행 매개변수의 개수
* argv: char형 포인터 배열, 명령행 매개변수

동작:

* 입력된 매개변수에 따라서 train\_writing() 또는 test\_writing() 함수를 호출함
* "train" 매개변수가 입력된 경우, train\_writing() 함수를 호출하고 학습을 수행함
* "test" 매개변수가 입력된 경우, test\_writing() 함수를 호출하고 이미지 파일을 분류함

설명:

* 입력된 매개변수의 개수가 4보다 작은 경우, 사용 방법을 출력하고 종료함
* cfg: char형 포인터 변수, YOLOv3 모델의 설정 파일 경로
* weights: char형 포인터 변수, 미리 학습된 가중치 파일 경로 (선택적)
* filename: char형 포인터 변수, 이미지 파일 경로 (선택적)

