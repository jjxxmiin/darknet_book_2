# cifar

```c
#include "darknet.h"
```

## train\_cifar

```c
void train_cifar(char *cfgfile, char *weightfile)
{
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network *net = load_network(cfgfile, weightfile, 0);
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);

    char *backup_directory = "/home/pjreddie/backup/";
    int classes = 10;
    int N = 50000;

    char **labels = get_labels("data/cifar/labels.txt");
    int epoch = (*net->seen)/N;
    data train = load_all_cifar10();
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        clock_t time=clock();

        float loss = train_network_sgd(net, train, 1);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.95 + loss*.05;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net->seen);
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
    free_ptrs((void**)labels, classes);
    free(base);
    free_data(train);
}
```

함수 이름: train\_cifar&#x20;

입력:&#x20;

* char \*cfgfile (설정 파일 경로)
* char \*weightfile (가중치 파일 경로)&#x20;

동작:&#x20;

* CIFAR-10 데이터셋에 대한 신경망을 훈련시키는 함수이다.&#x20;
* 지정된 구성 파일과 가중치 파일을 사용하여 네트워크를 로드하고, SGD(확률적 경사 하강법)를 사용하여 훈련을 수행한다.&#x20;
* 주기적으로 현재 손실, 평균 손실, 학습률, 경과 시간, 이미지 수 등을 출력하고, 지정된 배치 수 또는 최대 배치 수에 도달하면 훈련을 중지한다.&#x20;
* 또한 훈련 중에 지정된 백업 디렉토리에 모델 가중치를 저장한다.&#x20;

설명:

* backup\_directory: 모델 가중치를 저장할 백업 디렉토리 경로
* classes: 분류 클래스 수 (CIFAR-10의 경우 10)
* N: 학습 세트 이미지 수 (CIFAR-10의 경우 50000)
* labels: 클래스 레이블 배열 포인터
* epoch: 현재 학습 epoch 수
* train: CIFAR-10 데이터셋의 학습 데이터를 포함하는 데이터 구조체
* avg\_loss: 현재까지의 평균 손실값
* time: 현재 배치의 훈련 시간
* loss: 현재 배치의 손실값
* buff: 모델 가중치를 저장할 파일 이름 및 경로
* net: 로드된 신경망 구조체



## train\_cifar\_distill

```c
void train_cifar_distill(char *cfgfile, char *weightfile)
{
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network *net = load_network(cfgfile, weightfile, 0);
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);

    char *backup_directory = "/home/pjreddie/backup/";
    int classes = 10;
    int N = 50000;

    char **labels = get_labels("data/cifar/labels.txt");
    int epoch = (*net->seen)/N;

    data train = load_all_cifar10();
    matrix soft = csv_to_matrix("results/ensemble.csv");

    float weight = .9;
    scale_matrix(soft, weight);
    scale_matrix(train.y, 1. - weight);
    matrix_add_matrix(soft, train.y);

    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        clock_t time=clock();

        float loss = train_network_sgd(net, train, 1);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.95 + loss*.05;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net->seen);
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
    free_ptrs((void**)labels, classes);
    free(base);
    free_data(train);
}
```

함수 이름: train\_cifar\_distill

입력:

* cfgfile: char 포인터. 네트워크 설정 파일의 경로를 지정한다.
* weightfile: char 포인터. 학습된 네트워크 가중치 파일의 경로를 지정한다.

동작:

* CIFAR-10 데이터셋을 사용하여 네트워크를 학습시킨다.
* 지도학습(supervised learning)된 모델의 예측 확률값(소프트맥스 출력)을 앙상블(ensemble) 모델의 예측 확률값과 결합하여 손실을 계산하고 역전파 알고리즘을 사용하여 네트워크 가중치를 업데이트한다.
* 지정된 배치(batch) 수(max\_batches) 또는 무제한 반복(iteration)을 실행하며, 학습 과정에서 네트워크의 가중치를 주기적으로 저장하고, 학습 속도(learning rate), 모멘텀(momentum), 가중치 감쇠(decay) 등의 정보를 출력한다.

설명:&#x20;

* 이 함수는 CIFAR-10 데이터셋을 사용하여 distillation 기법을 사용하여 네트워크를 학습시키는 함수이다.&#x20;
* distillation 기법은 작은 모델(학생)을 학습시키는데, 큰 모델(선생)의 예측값을 사용하여 학습시키는 방법이다.&#x20;
* 이 함수는 지도학습된 모델(선생)과 앙상블 모델(선생)의 예측값을 결합하여 distillation 기법을 사용하여 작은 모델(학생)을 학습시킨다.&#x20;
* 이 함수는 주어진 네트워크 설정 파일과 학습된 네트워크 가중치 파일을 사용하여 네트워크를 로드하고, CIFAR-10 데이터셋을 불러와서 네트워크를 학습시킨다.&#x20;
* 학습 중에는 지정된 배치 수(max\_batches) 또는 무제한 반복(iteration)을 실행하며, 네트워크의 가중치를 주기적으로 저장하고, 학습 속도(learning rate), 모멘텀(momentum), 가중치 감쇠(decay) 등의 정보를 출력한다.&#x20;
* 이 함수는 학습 중에 손실을 계산할 때 앙상블 모델의 예측 확률값과 지도학습된 모델의 예측 확률값을 결합하여 사용한다.&#x20;
* 이 함수는 학습이 완료된 후, 사용한 메모리를 모두 해제한다.



## test\_cifar\_multi

```c
void test_cifar_multi(char *filename, char *weightfile)
{
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    float avg_acc = 0;
    data test = load_cifar10_data("data/cifar/cifar-10-batches-bin/test_batch.bin");

    int i;
    for(i = 0; i < test.X.rows; ++i){
        image im = float_to_image(32, 32, 3, test.X.vals[i]);

        float pred[10] = {0};

        float *p = network_predict(net, im.data);
        axpy_cpu(10, 1, p, 1, pred, 1);
        flip_image(im);
        p = network_predict(net, im.data);
        axpy_cpu(10, 1, p, 1, pred, 1);

        int index = max_index(pred, 10);
        int class = max_index(test.y.vals[i], 10);
        if(index == class) avg_acc += 1;
        free_image(im);
        printf("%4d: %.2f%%\n", i, 100.*avg_acc/(i+1));
    }
}
```

함수 이름: test\_cifar\_multi

입력:

* filename: 테스트할 모델의 설정 파일 경로
* weightfile: 테스트할 모델의 가중치 파일 경로

동작:

* 설정 파일과 가중치 파일을 이용해 모델을 로드한다.
* 배치 크기를 1로 설정한다.
* CIFAR-10 데이터셋의 테스트 데이터를 로드한다.
* 각각의 테스트 이미지에 대해 다음을 수행한다:
  * 테스트 이미지를 네트워크에 입력으로 넣고, 출력값을 가져온다.
  * 이미지를 좌우로 뒤집어 다시 한 번 네트워크에 입력으로 넣고, 출력값을 가져온다.
  * 두 번의 출력값을 평균내어 예측값을 계산한다.
  * 예측값과 실제 레이블을 비교하여 정확도를 계산하고, 이를 누적한다.
  * 이미지 메모리를 해제한다.
  * 현재까지의 정확도를 출력한다.

설명:&#x20;

* 이 함수는 로드한 모델을 이용해 CIFAR-10 데이터셋의 테스트 데이터를 평가하는 역할을 한다.&#x20;
* 각각의 테스트 이미지에 대해 두 번의 예측값을 계산한 뒤 평균을 내어 최종 예측값을 계산하고, 이를 실제 레이블과 비교하여 정확도를 계산한다.&#x20;
* 이렇게 계산된 정확도는 각각의 이미지에서 누적되어 전체 테스트 데이터셋에 대한 평균 정확도를 계산하게 된다.



## test\_cifar

```c
void test_cifar(char *filename, char *weightfile)
{
    network *net = load_network(filename, weightfile, 0);
    srand(time(0));

    clock_t time;
    float avg_acc = 0;
    float avg_top5 = 0;
    data test = load_cifar10_data("data/cifar/cifar-10-batches-bin/test_batch.bin");

    time=clock();

    float *acc = network_accuracies(net, test, 2);
    avg_acc += acc[0];
    avg_top5 += acc[1];
    printf("top1: %f, %lf seconds, %d images\n", avg_acc, sec(clock()-time), test.X.rows);
    free_data(test);
}
```

함수 이름: test\_cifar&#x20;

입력:

* filename: char\* 타입의 파일 이름 (네트워크 구조가 저장된 파일)
* weightfile: char\* 타입의 파일 이름 (네트워크 가중치가 저장된 파일)&#x20;

동작:

* 입력으로 받은 파일에서 네트워크 구조와 가중치를 로드하고, 이를 이용해 cifar-10 데이터셋의 정확도를 평가합니다.
* 평가 방식은 top-1과 top-5 정확도를 측정합니다.
* 측정 결과를 출력하고, 마지막으로 사용한 메모리를 해제합니다.&#x20;

설명:

* 네트워크를 로드하고 srand를 이용해 난수 발생기 초기화합니다.
* 시간을 측정하기 위해 clock() 함수를 사용합니다.
* load\_cifar10\_data 함수를 사용해 cifar-10 데이터셋을 로드합니다.
* network\_accuracies 함수를 사용해 cifar-10 데이터셋의 top-1과 top-5 정확도를 계산합니다.
* 측정 결과를 출력하고, 사용한 메모리를 해제합니다.



## extract\_cifar

```c
void extract_cifar()
{
char *labels[] = {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
    int i;
    data train = load_all_cifar10();
    data test = load_cifar10_data("data/cifar/cifar-10-batches-bin/test_batch.bin");
    for(i = 0; i < train.X.rows; ++i){
        image im = float_to_image(32, 32, 3, train.X.vals[i]);
        int class = max_index(train.y.vals[i], 10);
        char buff[256];
        sprintf(buff, "data/cifar/train/%d_%s",i,labels[class]);
        save_image_options(im, buff, PNG, 0);
    }
    for(i = 0; i < test.X.rows; ++i){
        image im = float_to_image(32, 32, 3, test.X.vals[i]);
        int class = max_index(test.y.vals[i], 10);
        char buff[256];
        sprintf(buff, "data/cifar/test/%d_%s",i,labels[class]);
        save_image_options(im, buff, PNG, 0);
    }
}
```

함수 이름: extract\_cifar&#x20;

입력:&#x20;

* 없음&#x20;

동작:&#x20;

* CIFAR-10 데이터셋에서 이미지를 추출하여 클래스 레이블과 함께 저장합니다.&#x20;
* 훈련 이미지는 'data/cifar/train' 폴더에 저장되고, 테스트 이미지는 'data/cifar/test' 폴더에 저장됩니다.&#x20;
* 이미지 파일 이름은 각각 '인덱스\_클래스명' 형식으로 지정됩니다.&#x20;

설명:

* labels: 클래스 레이블을 저장하는 문자열 배열
* train: 모든 CIFAR-10 훈련 데이터를 로드하는 데 사용되는 data 구조체
* test: CIFAR-10 테스트 데이터를 로드하는 데 사용되는 data 구조체
* for 루프를 사용하여 훈련 및 테스트 데이터셋에서 이미지를 추출하고, 해당 이미지의 클래스 레이블을 가져와서 이미지를 저장합니다.
* sprintf 함수를 사용하여 이미지 파일 이름을 지정합니다.



## test\_cifar\_csv

```c
void test_cifar_csv(char *filename, char *weightfile)
{
    network *net = load_network(filename, weightfile, 0);
    srand(time(0));

    data test = load_cifar10_data("data/cifar/cifar-10-batches-bin/test_batch.bin");

    matrix pred = network_predict_data(net, test);

    int i;
    for(i = 0; i < test.X.rows; ++i){
        image im = float_to_image(32, 32, 3, test.X.vals[i]);
        flip_image(im);
    }
    matrix pred2 = network_predict_data(net, test);
    scale_matrix(pred, .5);
    scale_matrix(pred2, .5);
    matrix_add_matrix(pred2, pred);

    matrix_to_csv(pred);
    fprintf(stderr, "Accuracy: %f\n", matrix_topk_accuracy(test.y, pred, 1));
    free_data(test);
}
```

함수 이름: test\_cifar\_csv

입력:

* filename (char\*): 네트워크 모델 파일 경로
* weightfile (char\*): 학습된 가중치 파일 경로

동작:

* 지정된 네트워크 모델 파일과 가중치 파일을 로드하여 네트워크를 생성합니다.
* 시드 값을 현재 시간으로 설정하여 난수 생성기를 초기화합니다.
* CIFAR-10 데이터셋의 테스트 데이터를 로드합니다.
* 네트워크 모델을 사용하여 테스트 데이터를 예측하고, 예측 결과를 matrix 형식으로 반환합니다.
* 테스트 데이터의 이미지를 반전시키고, 다시 한 번 네트워크 모델을 사용하여 예측 결과를 반환합니다.
* 첫 번째 예측 결과와 두 번째 예측 결과를 합산하고, 이를 csv 파일로 저장합니다.
* 테스트 데이터의 실제 레이블과 예측 결과를 비교하여 top-1 정확도를 계산하고, 표준 오류 스트림(stderr)에 출력합니다.
* 메모리 할당 해제

설명:&#x20;

* 이 함수는 CIFAR-10 데이터셋을 사용하여 로드한 테스트 데이터에 대해 네트워크 모델의 예측 결과를 csv 파일로 저장하고, 이를 기반으로 정확도를 계산하는 역할을 합니다.&#x20;
* 함수는 load\_network(), load\_cifar10\_data(), network\_predict\_data(), matrix\_topk\_accuracy(), free\_data() 함수 등을 사용하여 동작합니다.



## test\_cifar\_csvtrain

```c
void test_cifar_csvtrain(char *cfg, char *weights)
{
    network *net = load_network(cfg, weights, 0);
    srand(time(0));

    data test = load_all_cifar10();

    matrix pred = network_predict_data(net, test);

    int i;
    for(i = 0; i < test.X.rows; ++i){
        image im = float_to_image(32, 32, 3, test.X.vals[i]);
        flip_image(im);
    }
    matrix pred2 = network_predict_data(net, test);
    scale_matrix(pred, .5);
    scale_matrix(pred2, .5);
    matrix_add_matrix(pred2, pred);

    matrix_to_csv(pred);
    fprintf(stderr, "Accuracy: %f\n", matrix_topk_accuracy(test.y, pred, 1));
    free_data(test);
}
```

함수 이름: test\_cifar\_csvtrain

입력:

* cfg (char\*): 네트워크 모델 설정 파일 경로
* weights (char\*): 학습된 가중치 파일 경로

동작:

* 지정된 네트워크 모델 설정 파일과 가중치 파일을 로드하여 네트워크를 생성합니다.
* 시드 값을 현재 시간으로 설정하여 난수 생성기를 초기화합니다.
* CIFAR-10 데이터셋의 전체 데이터를 로드합니다.
* 네트워크 모델을 사용하여 전체 데이터를 예측하고, 예측 결과를 matrix 형식으로 반환합니다.
* 전체 데이터의 이미지를 반전시키고, 다시 한 번 네트워크 모델을 사용하여 예측 결과를 반환합니다.
* 첫 번째 예측 결과와 두 번째 예측 결과를 합산하고, 이를 csv 파일로 저장합니다.
* 전체 데이터의 실제 레이블과 예측 결과를 비교하여 top-1 정확도를 계산하고, 표준 오류 스트림(stderr)에 출력합니다.
* 메모리 할당 해제

설명:&#x20;

* 이 함수는 CIFAR-10 데이터셋을 사용하여 전체 데이터에 대해 네트워크 모델의 예측 결과를 csv 파일로 저장하고, 이를 기반으로 정확도를 계산하는 역할을 합니다.&#x20;
* 함수는 load\_network(), load\_all\_cifar10(), network\_predict\_data(), matrix\_topk\_accuracy(), free\_data() 함수 등을 사용하여 동작합니다.



## eval\_cifar\_csv

```c
void eval_cifar_csv()
{
    data test = load_cifar10_data("data/cifar/cifar-10-batches-bin/test_batch.bin");

    matrix pred = csv_to_matrix("results/combined.csv");
    fprintf(stderr, "%d %d\n", pred.rows, pred.cols);

    fprintf(stderr, "Accuracy: %f\n", matrix_topk_accuracy(test.y, pred, 1));
    free_data(test);
    free_matrix(pred);
}
```

함수 이름: eval\_cifar\_csv

입력:&#x20;

* 없음

동작:

* CIFAR-10 데이터셋의 테스트 데이터를 로드합니다.
* csv 파일로 저장된 예측 결과를 matrix 형식으로 로드합니다.
* 예측 결과의 행과 열 개수를 출력합니다.
* 예측 결과와 실제 레이블을 비교하여 top-1 정확도를 계산하고, 표준 오류 스트림(stderr)에 출력합니다.
* 메모리 할당 해제

설명:&#x20;

* 이 함수는 CIFAR-10 데이터셋을 사용하여 저장된 csv 파일의 예측 결과를 로드하고, 이를 기반으로 top-1 정확도를 계산하여 출력하는 역할을 합니다.&#x20;
* 함수는 load\_cifar10\_data(), csv\_to\_matrix(), matrix\_topk\_accuracy(), free\_data(), free\_matrix() 함수 등을 사용하여 동작합니다.



## run\_cifar

```c
void run_cifar(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    if(0==strcmp(argv[2], "train")) train_cifar(cfg, weights);
    else if(0==strcmp(argv[2], "extract")) extract_cifar();
    else if(0==strcmp(argv[2], "distill")) train_cifar_distill(cfg, weights);
    else if(0==strcmp(argv[2], "test")) test_cifar(cfg, weights);
    else if(0==strcmp(argv[2], "multi")) test_cifar_multi(cfg, weights);
    else if(0==strcmp(argv[2], "csv")) test_cifar_csv(cfg, weights);
    else if(0==strcmp(argv[2], "csvtrain")) test_cifar_csvtrain(cfg, weights);
    else if(0==strcmp(argv[2], "eval")) eval_cifar_csv();
}
```

함수 이름: run\_cifar

입력:

* int argc: 입력 인수의 개수
* char \*\*argv: 입력 인수의 배열 포인터

동작:

* 입력 인수의 개수가 4보다 작으면 사용 방법을 출력하고 함수를 종료합니다.
* 3번째 입력 인수를 cfg 변수에 저장합니다.
* 4번째 입력 인수가 존재하면 weights 변수에 저장합니다.
* 2번째 입력 인수에 따라 다음 함수 중 하나를 호출합니다.
  * train\_cifar()
  * extract\_cifar()
  * train\_cifar\_distill()
  * test\_cifar()
  * test\_cifar\_multi()
  * test\_cifar\_csv()
  * test\_cifar\_csvtrain()
  * eval\_cifar\_csv()
* 각 함수는 CIFAR-10 데이터셋을 사용하여 모델을 훈련하고, 예측 결과를 출력하거나, 예측 결과를 csv 파일로 저장하거나, 저장된 csv 파일을 로드하여 정확도를 출력하는 등의 동작을 수행합니다.

설명:&#x20;

* 이 함수는 CIFAR-10 데이터셋을 사용하여 다양한 동작을 수행하는 함수들을 호출하는 역할을 합니다.&#x20;
* 함수는 인수로 받은 입력 인수의 개수와 배열 포인터를 사용하여 각 동작에 필요한 cfg 파일과 weights 파일을 결정하고, 이를 이용하여 train\_cifar(), extract\_cifar(), train\_cifar\_distill(), test\_cifar(), test\_cifar\_multi(), test\_cifar\_csv(), test\_cifar\_csvtrain(), eval\_cifar\_csv() 함수 중 적절한 함수를 호출합니다.&#x20;
* 이 함수는 명령행 인수를 처리하는 데에 사용되는 main() 함수에서 호출됩니다.

