# art

```c
#include "darknet.h"

#include <sys/time.h>
```

## demo\_art

```c
void demo_art(char *cfgfile, char *weightfile, int cam_index)
{
#ifdef OPENCV
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);

    void * cap = open_video_stream(0, cam_index, 0,0,0);

    char *window = "ArtJudgementBot9000!!!";
    if(!cap) error("Couldn't connect to webcam.\n");
    int i;
    int idx[] = {37, 401, 434};
    int n = sizeof(idx)/sizeof(idx[0]);

    while(1){
        image in = get_image_from_stream(cap);
        image in_s = resize_image(in, net->w, net->h);

        float *p = network_predict(net, in_s.data);

        printf("\033[2J");
        printf("\033[1;1H");

        float score = 0;
        for(i = 0; i < n; ++i){
            float s = p[idx[i]];
            if (s > score) score = s;
        }
        score = score;
        printf("I APPRECIATE THIS ARTWORK: %10.7f%%\n", score*100);
        printf("[");
	int upper = 30;
        for(i = 0; i < upper; ++i){
            printf("%c", ((i+.5) < score*upper) ? 219 : ' ');
        }
        printf("]\n");

        show_image(in, window, 1);
        free_image(in_s);
        free_image(in);
    }
#endif
}
```

함수 이름: demo\_art

입력:

* cfgfile: YOLO 모델 구성 파일의 경로
* weightfile: 학습된 가중치 파일의 경로
* cam\_index: 사용할 카메라의 인덱스

동작:

* 주어진 cfgfile과 weightfile을 사용하여 YOLO 모델을 로드한다.
* 입력 이미지를 모델의 크기에 맞게 조절한다.
* 모델을 이용하여 이미지에서 물체를 검출하고 각 물체의 클래스 확률을 계산한다.
* 계산된 클래스 확률을 이용하여 작품의 평가 점수를 출력하고, 그래프로 나타낸다.
* 입력 이미지를 화면에 표시한다.
* 프로그램이 종료될 때까지 위의 과정을 반복한다.

설명:&#x20;

* 이 함수는 주어진 cfgfile과 weightfile을 사용하여 YOLO 모델을 로드하고, 이 모델을 사용하여 카메라로부터 입력되는 이미지에서 물체를 검출하고 평가 점수를 계산하는 예제 코드이다.&#x20;
* 이 함수는 OpenCV를 사용하여 화면에 이미지를 출력하며, 사용자의 입력이나 다른 이벤트를 처리하지는 않는다.&#x20;
* 따라서 이 함수는 YOLO 모델을 사용하여 물체 검출을 수행하는 예제 코드로 사용될 수 있다.



## run\_art

```c
void run_art(int argc, char **argv)
{
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    char *cfg = argv[2];
    char *weights = argv[3];
    demo_art(cfg, weights, cam_index);
}
```

함수 이름: run\_art&#x20;

입력:&#x20;

* argc와 argv&#x20;

동작:&#x20;

* 주어진 인수들을 기반으로 demo\_art 함수를 호출합니다.&#x20;

설명:&#x20;

* 주어진 argc와 argv를 통해 cam\_index, cfg, weights를 설정하고 demo\_art 함수를 실행합니다.&#x20;
* demo\_art 함수는 실시간 웹캠 비디오 스트림에서 예술 필터를 적용한 비디오를 생성합니다.&#x20;
* cam\_index가 주어지면 해당 인덱스의 카메라를 사용하고, 그렇지 않으면 기본 카메라를 사용합니다.

