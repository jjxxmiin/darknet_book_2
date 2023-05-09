# voxel

```c
#include "darknet.h"
```

## extract\_voxel

```c
void extract_voxel(char *lfile, char *rfile, char *prefix)
{
#ifdef OPENCV
    int w = 1920;
    int h = 1080;
    int shift = 0;
    int count = 0;
    CvCapture *lcap = cvCaptureFromFile(lfile);
    CvCapture *rcap = cvCaptureFromFile(rfile);
    while(1){
        image l = get_image_from_stream(lcap);
        image r = get_image_from_stream(rcap);
        if(!l.w || !r.w) break;
        if(count%100 == 0) {
            shift = best_3d_shift_r(l, r, -l.h/100, l.h/100);
            printf("%d\n", shift);
        }
        image ls = crop_image(l, (l.w - w)/2, (l.h - h)/2, w, h);
        image rs = crop_image(r, 105 + (r.w - w)/2, (r.h - h)/2 + shift, w, h);
        char buff[256];
        sprintf(buff, "%s_%05d_l", prefix, count);
        save_image(ls, buff);
        sprintf(buff, "%s_%05d_r", prefix, count);
        save_image(rs, buff);
        free_image(l);
        free_image(r);
        free_image(ls);
        free_image(rs);
        ++count;
    }

#else
    printf("need OpenCV for extraction\n");
#endif
}
```

함수 이름: extract\_voxel&#x20;

입력:

* lfile: 왼쪽 카메라 비디오 파일 경로
* rfile: 오른쪽 카메라 비디오 파일 경로
* prefix: 저장할 이미지 파일 이름에 붙을 접두사

동작:&#x20;

* 이 함수는 왼쪽 카메라와 오른쪽 카메라로부터 비디오를 읽어서 이미지를 추출합니다.&#x20;
* 추출된 이미지는 3D 볼륨을 만들기 위한 왼쪽 카메라와 오른쪽 카메라의 이미지 쌍입니다.&#x20;
* 추출된 이미지는 파일로 저장됩니다.

설명:

1. OpenCV 라이브러리가 필요합니다.
2. 1920x1080 해상도로 이미지를 추출합니다.
3. 이미지 추출 시 왼쪽 카메라와 오른쪽 카메라의 이미지 위치를 조정합니다.
4. 이미지 추출 시 왼쪽 카메라와 오른쪽 카메라의 이미지 쌍을 생성합니다.
5. 이미지는 파일로 저장됩니다.
6. 이미지 파일 이름에는 prefix와 번호가 붙습니다.



## train\_voxel

```c
void train_voxel(char *cfgfile, char *weightfile)
{
    char *train_images = "/data/imagenet/imagenet1k.train.list";
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;
    data train, buffer;


    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.scale = 4;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;
    args.type = SUPER_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
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
        if(i%1000==0){
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

함수 이름: train\_voxel

입력:

* cfgfile: 모델 구성 파일의 경로를 가리키는 문자열 포인터
* weightfile: 모델 가중치 파일의 경로를 가리키는 문자열 포인터

동작:&#x20;

* 주어진 모델 구성 파일(cfgfile)을 바탕으로 3D 객체 인식 모델을 학습하는 함수.&#x20;
* 이미지 경로가 담긴 리스트(train\_images)에서 학습 데이터를 로드하고, 이를 바탕으로 모델을 업데이트한다.&#x20;
* 매 1000 에폭마다 가중치 파일을 백업 디렉토리(backup\_directory)에 저장한다.

설명:

1. 모델 구성 파일(cfgfile)로부터 모델을 파싱(parse\_network\_cfg)하여 생성한다.
2. 만약 모델 가중치 파일(weightfile)이 주어졌다면, 해당 가중치를 로드(load\_weights)한다.
3. 학습 이미지 경로가 담긴 리스트(train\_images)를 로드하고, 이를 기반으로 데이터를 로드하는 스레드(load\_data\_in\_thread)를 생성한다.
4. 학습 데이터를 이용하여 모델을 학습(train\_network)하고, 손실(loss)을 계산한다.
5. 1000 에폭마다 가중치 파일을 백업 디렉토리(backup\_directory)에 저장한다(save\_weights).
6. 100 에폭마다 모델 가중치를 백업 디렉토리에 저장한다.
7. 모델 학습이 완료되면 최종 가중치를 백업 디렉토리에 저장한다.



## test\_voxel

```c
void test_voxel(char *cfgfile, char *weightfile, char *filename)
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
        printf("%d %d\n", im.w, im.h);

        float *X = im.data;
        time=clock();
        network_predict(net, X);
        image out = get_network_image(net);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        save_image(out, "out");

        free_image(im);
        if (filename) break;
    }
}
```

함수 이름: test\_voxel

입력:

* cfgfile: char\* 타입. YOLO 모델의 구성 파일 경로.
* weightfile: char\* 타입. YOLO 모델의 가중치 파일 경로.
* filename: char\* 타입. 테스트하려는 이미지 파일 경로. 사용자 입력으로 받거나 프로그램 인자로 전달 받을 수 있음.

동작:&#x20;

* 입력으로 주어진 YOLO 모델을 이용하여 이미지를 테스트하고 결과 이미지를 저장하는 함수.&#x20;
* 입력으로 이미지 파일 경로가 주어지면 해당 이미지에 대한 예측을 수행하고, 입력이 없는 경우 사용자로부터 이미지 파일 경로를 입력받아 예측 수행.&#x20;
* 예측 결과는 out.png 파일로 저장됨.

설명:

1. YOLO 모델 구성 파일(cfgfile)을 파싱하여 네트워크 모델 생성.
2. YOLO 모델 가중치(weightfile)를 로드하여 모델에 적용.
3. 배치 크기를 1로 설정.
4. 시드(seed) 값을 고정하여 난수 생성기 초기화.
5. 이미지 파일 경로(filename)가 주어진 경우 해당 이미지 파일을 로드하고, 입력이 없는 경우 사용자로부터 이미지 파일 경로를 입력받음.
6. 입력 이미지를 로드하고, YOLO 모델의 입력 크기에 맞게 조정(resize) 함.
7. 예측 수행(network\_predict) 및 예측 시간 측정.
8. 예측 결과 이미지(out)를 파일로 저장.
9. 이미지 메모리 해제 및 다음 입력 대기.
10. filename 값이 주어진 경우 예측 수행 후 종료.



## run\_voxel

```c
void run_voxel(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5] : 0;
    if(0==strcmp(argv[2], "train")) train_voxel(cfg, weights);
    else if(0==strcmp(argv[2], "test")) test_voxel(cfg, weights, filename);
    else if(0==strcmp(argv[2], "extract")) extract_voxel(argv[3], argv[4], argv[5]);
    /*
       else if(0==strcmp(argv[2], "valid")) validate_voxel(cfg, weights);
     */
}
```

함수 이름: run\_voxel

입력:&#x20;

* argc (int): 명령행 인자 개수
* argv (char \*\*): 명령행 인자 배열

동작:&#x20;

* 주어진 명령행 인자에 따라 train\_voxel, test\_voxel, extract\_voxel, validate\_voxel 중 하나를 실행

설명:&#x20;

* run\_voxel 함수는 주어진 명령행 인자에 따라 train\_voxel, test\_voxel, extract\_voxel, validate\_voxel 중 하나를 실행하는 역할을 합니다.&#x20;
* argc와 argv는 명령행에서 입력받은 인자의 개수와 배열을 전달받습니다.&#x20;
* 실행할 함수와 해당 함수에 필요한 인자들은 argv 배열에서 가져와 실행됩니다.&#x20;
* 예를 들어, argv\[2]가 "train"이면 train\_voxel 함수가 실행되며, cfg 파일 경로와 weight 파일 경로가 필요합니다.&#x20;
* 만약 argv\[2]가 "test"라면 test\_voxel 함수가 실행되며, cfg 파일 경로와 weight 파일 경로, 테스트할 이미지 파일 경로가 필요합니다.&#x20;
* "extract" 옵션의 경우에는 extract\_voxel 함수가 실행되며, cfg 파일 경로, weight 파일 경로, 추출할 feature map이 있는 레이어 이름이 필요합니다.&#x20;
* 마지막으로 "valid" 옵션은 주석 처리되어 있습니다.

