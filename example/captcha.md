# captcha

```c
#include "darknet.h"
```

## fix\_data\_captcha

```c
void fix_data_captcha(data d, int mask)
{
    matrix labels = d.y;
    int i, j;
    for(i = 0; i < d.y.rows; ++i){
        for(j = 0; j < d.y.cols; j += 2){
            if (mask){
                if(!labels.vals[i][j]){
                    labels.vals[i][j] = SECRET_NUM;
                    labels.vals[i][j+1] = SECRET_NUM;
                }else if(labels.vals[i][j+1]){
                    labels.vals[i][j] = 0;
                }
            } else{
                if (labels.vals[i][j]) {
                    labels.vals[i][j+1] = 0;
                } else {
                    labels.vals[i][j+1] = 1;
                }
            }
        }
    }
}
```

함수 이름: fix\_data\_captcha

입력:

* data d: 학습 데이터를 나타내는 구조체
* int mask: 데이터 마스킹 여부를 결정하는 플래그 값

동작:&#x20;

* 주어진 데이터 d에서 레이블을 수정하여 반환하는 함수이다.&#x20;
* 이 함수는 레이블을 2개씩 묶어서, 첫 번째 레이블이 0일 경우에는 두 번째 레이블을 1로 바꾸고, 첫 번째 레이블이 1일 경우에는 두 번째 레이블을 0으로 바꾼다.&#x20;
* 마스킹 플래그가 설정되어 있을 경우에는 첫 번째 레이블이 0이면 두 번째 레이블도 0으로, 첫 번째 레이블이 0이 아니면 두 번째 레이블을 0으로 설정하고 첫 번째 레이블은 SECRET\_NUM 값으로 바꾼다.

설명:&#x20;

* 이 함수는 머신러닝 모델에서 사용되는 학습 데이터의 레이블을 수정하는 함수이다. 레이블은 학습 데이터의 정답을 나타내며, 이 함수에서는 주어진 데이터에서 레이블을 2개씩 묶어서 처리한다.&#x20;
* 각각의 묶음은 이미지에 있는 문자 1개씩을 나타낸다. 첫 번째 레이블이 0인 경우, 두 번째 레이블을 1로 설정하여 해당 문자가 존재함을 나타내고, 첫 번째 레이블이 1인 경우, 두 번째 레이블을 0으로 설정하여 해당 문자가 존재하지 않음을 나타낸다.
* 또한, 마스킹 플래그가 설정되어 있을 경우, 첫 번째 레이블이 0이면 두 번째 레이블도 0으로 설정하고, 첫 번째 레이블이 0이 아니면 두 번째 레이블을 0으로 설정하고 첫 번째 레이블은 SECRET\_NUM 값으로 바꾼다.&#x20;
* SECRET\_NUM 값은 숨겨진 값을 의미하는데, 이는 모델이 레이블을 이용하여 학습할 때 레이블이 0인 경우와 레이블이 1인 경우를 구분할 수 있도록 돕는다.



## train\_captcha

```c
void train_captcha(char *cfgfile, char *weightfile)
{
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network *net = load_network(cfgfile, weightfile, 0);
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    int imgs = 1024;
    int i = *net->seen/imgs;
    int solved = 1;
    list *plist;
    char **labels = get_labels("/data/captcha/reimgs.labels.list");
    if (solved){
        plist = get_paths("/data/captcha/reimgs.solved.list");
    }else{
        plist = get_paths("/data/captcha/reimgs.raw.list");
    }
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    clock_t time;
    pthread_t load_thread;
    data train;
    data buffer;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.paths = paths;
    args.classes = 26;
    args.n = imgs;
    args.m = plist->size;
    args.labels = labels;
    args.d = &buffer;
    args.type = CLASSIFICATION_DATA;

    load_thread = load_data_in_thread(args);
    while(1){
        ++i;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        fix_data_captcha(train, solved);

        /*
           image im = float_to_image(256, 256, 3, train.X.vals[114]);
           show_image(im, "training");
           cvWaitKey(0);
         */

        load_thread = load_data_in_thread(args);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d: %f, %f avg, %lf seconds, %ld images\n", i, loss, avg_loss, sec(clock()-time), *net->seen);
        free_data(train);
        if(i%100==0){
            char buff[256];
            sprintf(buff, "/home/pjreddie/imagenet_backup/%s_%d.weights",base, i);
            save_weights(net, buff);
        }
    }
}
```

함수 이름: train\_captcha

입력:&#x20;

* cfgfile: 모델 설정 파일 경로
* weightfile: 학습된 가중치 파일 경로

동작:&#x20;

* 이 함수는 지정된 cfgfile 및 weightfile을 사용하여 captcha 이미지를 학습하는 데 사용됩니다. 학습 중에는 이미지를 일괄 처리하고 이미지의 라벨을 수정하는 작업이 이루어집니다.&#x20;
* 이미지와 라벨은 비동기적으로 로드되고 학습 데이터가 버퍼에 로드됩니다. 이후 train\_network 함수를 사용하여 네트워크를 학습하고 학습된 가중치를 저장합니다.

설명:

1. srand(time(0)): 랜덤 시드를 초기화합니다.
2. basecfg(cfgfile): cfgfile에서 기본 설정을 가져와 base.cfg 파일에 쓰고 base.cfg 파일 이름을 반환합니다.
3. load\_network(cfgfile, weightfile, 0): cfgfile과 weightfile을 사용하여 네트워크를 로드합니다.
4. get\_labels("/data/captcha/reimgs.labels.list"): 라벨 목록 파일에서 라벨을 가져옵니다.
5. get\_paths("/data/captcha/reimgs.solved.list"): 이미지 파일 목록을 가져옵니다.
6. list\_to\_array(plist): 목록에서 배열을 가져옵니다.
7. pthread\_join(load\_thread, 0): 이미지 데이터가 로드될 때까지 기다립니다.
8. load\_data\_in\_thread(args): 이미지 데이터를 비동기적으로 로드합니다.
9. fix\_data\_captcha(train, solved): 학습 데이터의 라벨을 수정합니다.
10. train\_network(net, train): 네트워크를 학습합니다.
11. save\_weights(net, buff): 학습된 가중치를 지정된 경로에 저장합니다.

## test\_captcha

```c
void test_captcha(char *cfgfile, char *weightfile, char *filename)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    int i = 0;
    char **names = get_labels("/data/captcha/reimgs.labels.list");
    char buff[256];
    char *input = buff;
    int indexes[26];
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            //printf("Enter Image Path: ");
            //fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, net->w, net->h);
        float *X = im.data;
        float *predictions = network_predict(net, X);
        top_predictions(net, 26, indexes);
        //printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < 26; ++i){
            int index = indexes[i];
            if(i != 0) printf(", ");
            printf("%s %f", names[index], predictions[index]);
        }
        printf("\n");
        fflush(stdout);
        free_image(im);
        if (filename) break;
    }
}
```

함수 이름: test\_captcha

입력:&#x20;

* cfgfile: YOLO 모델의 configuration 파일 경로&#x20;
* weightfile: YOLO 모델의 가중치 파일 경로&#x20;
* filename - 입력 이미지 파일 경로 (선택적)

동작:&#x20;

* YOLO 모델을 사용하여 입력 이미지에서 문자 인식을 수행하고 결과를 출력한다.

설명:&#x20;

* 이 함수는 YOLO 모델을 사용하여 입력 이미지에서 문자 인식을 수행하는 함수이다. 함수는 먼저 YOLO 모델을 로드하고, 이미지를 로드하고, YOLO 모델을 사용하여 예측을 수행한다. 그리고 나서 예측 결과를 출력한다.
* 만약 filename이 제공되면, 함수는 해당 파일에서 이미지를 읽고 예측을 수행한다. 그렇지 않으면, 함수는 사용자로부터 이미지 파일 경로를 입력받는다.
* 결과는 예측된 문자와 그 확률로 구성된다. 예측된 문자는 라벨 파일에서 참조하며, top\_predictions 함수를 사용하여 예측된 확률이 높은 상위 26개의 인덱스를 찾는다.
* 함수는 입력 이미지의 예측 결과를 출력하고, 출력 버퍼를 비운다.



## valid\_captcha

```c
void valid_captcha(char *cfgfile, char *weightfile, char *filename)
{
    char **labels = get_labels("/data/captcha/reimgs.labels.list");
    network *net = load_network(cfgfile, weightfile, 0);
    list *plist = get_paths("/data/captcha/reimgs.fg.list");
    char **paths = (char **)list_to_array(plist);
    int N = plist->size;
    int outputs = net->outputs;

    set_batch_network(net, 1);
    srand(2222222);
    int i, j;
    for(i = 0; i < N; ++i){
        if (i%100 == 0) fprintf(stderr, "%d\n", i);
        image im = load_image_color(paths[i], net->w, net->h);
        float *X = im.data;
        float *predictions = network_predict(net, X);
        //printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        int truth = -1;
        for(j = 0; j < 13; ++j){
            if (strstr(paths[i], labels[j])) truth = j;
        }
        if (truth == -1){
            fprintf(stderr, "bad: %s\n", paths[i]);
            return;
        }
        printf("%d, ", truth);
        for(j = 0; j < outputs; ++j){
            if (j != 0) printf(", ");
            printf("%f", predictions[j]);
        }
        printf("\n");
        fflush(stdout);
        free_image(im);
        if (filename) break;
    }
}
```

함수 이름: valid\_captcha

입력:

* cfgfile: 모델 구성 파일 경로
* weightfile: 모델 가중치 파일 경로
* filename: 출력 결과를 저장할 파일 이름 (옵션)

동작:

* 주어진 모델 구성과 가중치 파일을 사용하여 캡차 이미지 데이터셋을 이용해 모델의 성능을 평가하고, 결과를 출력함
* 이미지 데이터셋과 레이블 정보는 고정된 경로에서 가져옴
* 각 이미지마다 예측 결과와 실제 레이블을 비교하여 정확도를 측정하고, 출력 결과에 추가함
* 출력 결과는 터미널에 출력되며, 옵션으로 주어진 파일 이름에 저장됨

설명:

* valid\_captcha 함수는 주어진 모델 구성과 가중치 파일을 이용하여 캡차 이미지 데이터셋을 평가하는 함수입니다.
* 먼저, get\_labels 함수를 이용하여 이미지 레이블 정보를 가져옵니다.
* 그리고 load\_network 함수를 이용하여 모델을 불러옵니다.
* 이후 get\_paths 함수를 이용하여 이미지 데이터셋 경로를 가져옵니다.
* 이미지마다 예측 결과와 실제 레이블을 비교하여 정확도를 측정하고, 출력 결과에 추가합니다.
* 출력 결과는 터미널에 출력되며, 옵션으로 주어진 파일 이름에 저장됩니다.



## run\_captcha

```c
void run_captcha(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "train")) train_captcha(cfg, weights);
    else if(0==strcmp(argv[2], "test")) test_captcha(cfg, weights, filename);
    else if(0==strcmp(argv[2], "valid")) valid_captcha(cfg, weights, filename);
    //if(0==strcmp(argv[2], "test")) test_captcha(cfg, weights);
    //else if(0==strcmp(argv[2], "encode")) encode_captcha(cfg, weights);
    //else if(0==strcmp(argv[2], "decode")) decode_captcha(cfg, weights);
    //else if(0==strcmp(argv[2], "valid")) validate_captcha(cfg, weights);
}
```

함수 이름: run\_captcha

입력:

* argc: 인자 개수
* argv: 인자 배열

동작:

* 입력된 인자를 기반으로 다른 함수들을 호출하며, Captcha 이미지 인식 모델의 학습, 테스트, 검증 등의 작업을 수행함.

설명:

* Captcha 이미지 인식 모델의 학습, 테스트, 검증 등의 작업을 수행하기 위해 사용되는 함수.
* argv\[2]의 값에 따라 다른 함수들을 호출함.
* argv\[2] 값이 "train"일 경우, train\_captcha 함수를 호출하여 모델을 학습함.
* argv\[2] 값이 "test"일 경우, test\_captcha 함수를 호출하여 모델의 성능을 평가함.
* argv\[2] 값이 "valid"일 경우, valid\_captcha 함수를 호출하여 검증 데이터셋을 이용해 모델의 성능을 검증함.
* argv\[3]은 모델 구성 파일(.cfg)의 경로를 나타냄.
* argv\[4]는 모델 가중치 파일(.weights)의 경로를 나타냄. (선택 사항)
* argv\[5]는 검증 데이터셋 파일의 경로를 나타냄. (선택 사항)
