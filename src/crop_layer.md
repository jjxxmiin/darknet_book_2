# crop\_layer

input(image, feature map)을 crop하기 위한 layer입니다.

## get\_crop\_image

```c
image get_crop_image(crop_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;
    return float_to_image(w,h,c,l.output);
}
```

함수 이름: get\_crop\_image

입력:&#x20;

* crop\_layer 타입 변수 l

동작:&#x20;

* 입력으로 받은 crop\_layer 타입 변수 l의 output 배열에서 데이터를 가져와서 float\_to\_image 함수를 통해 이미지 형태로 변환하여 반환합니다.

설명:&#x20;

* 입력으로 받은 crop\_layer 타입 변수 l은 이미지 데이터를 잘라내기 위한 정보들을 가지고 있습니다.&#x20;
* 이 함수는 그 정보를 기반으로 잘라낸 이미지 데이터를 가져와서 float\_to\_image 함수를 통해 이미지 형태로 변환하고 반환합니다.&#x20;
* 이 함수는 이미지 인식 분야에서 많이 사용됩니다.



## forward\_crop\_layer

```c
void forward_crop_layer(const crop_layer l, network net)
{
    int i,j,c,b,row,col;
    int index;
    int count = 0;
    int flip = (l.flip && rand()%2);
    int dh = rand()%(l.h - l.out_h + 1);
    int dw = rand()%(l.w - l.out_w + 1);
    float scale = 2;
    float trans = -1;
    if(l.noadjust){
        scale = 1;
        trans = 0;
    }
    if(!net.train){
        flip = 0;
        dh = (l.h - l.out_h)/2;
        dw = (l.w - l.out_w)/2;
    }
    for(b = 0; b < l.batch; ++b){
        for(c = 0; c < l.c; ++c){
            for(i = 0; i < l.out_h; ++i){
                for(j = 0; j < l.out_w; ++j){
                    if(flip){
                        col = l.w - dw - j - 1;    
                    }else{
                        col = j + dw;
                    }
                    row = i + dh;
                    index = col+l.w*(row+l.h*(c + l.c*b));
                    l.output[count++] = net.input[index]*scale + trans;
                }
            }
        }
    }
}
```

함수 이름: forward\_crop\_layer

입력:

* const crop\_layer l: crop\_layer 타입의 l 변수. 크롭 레이어의 정보를 담고 있음.
* network net: network 타입의 net 변수. 신경망 정보를 담고 있음.

동작:

* 랜덤으로 좌우 반전과 이미지를 잘라낸 후, 크롭 레이어의 출력값을 계산하여 l.output 배열에 저장함.

설명:

* 크롭 레이어는 입력 이미지를 랜덤으로 자른 후, 출력 이미지를 생성함.
* 입력 이미지의 크기는 l.h x l.w x l.c이고, 출력 이미지의 크기는 l.out\_h x l.out\_w x l.out\_c임.
* 좌우 반전 여부는 l.flip 값에 따라 랜덤으로 결정됨.
* 잘라낸 이미지의 위치는 l.h, l.w에서 각각 l.out\_h, l.out\_w 크기만큼 랜덤하게 선택됨.
* 크롭 레이어는 네트워크가 학습 중일 때만 좌우 반전을 하고 이미지를 잘라냄. 학습이 아닐 때는 좌우 반전을 하지 않고 이미지 중앙에서 잘라냄.
* scale과 trans 값은 이미지를 정규화하기 위해 사용됨.



## backward\_crop\_layer

```c
void backward_crop_layer(const crop_layer l, network net){}
```

함수 이름: backward\_crop\_layer

입력:&#x20;

* crop\_layer l
* network net

동작:&#x20;

* crop\_layer의 역전파를 수행합니다.&#x20;
* 이 함수는 backward 연산을 수행하지 않습니다.

설명:&#x20;

* crop\_layer는 입력 이미지의 일부분을 무작위로 잘라내어 출력으로 내보내는 레이어입니다.&#x20;
* 이 함수는 해당 레이어에서의 역전파를 구현합니다.&#x20;
* 그러나 이 레이어는 역전파를 위한 학습 가능한 매개변수를 가지고 있지 않습니다.&#x20;
* 따라서 이 함수는 빈 함수로 남겨둡니다.



## resize\_crop\_layer

```c
void resize_crop_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->out_w =  l->scale*w;
    l->out_h =  l->scale*h;

    l->inputs = l->w * l->h * l->c;
    l->outputs = l->out_h * l->out_w * l->out_c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
}
```

함수 이름: resize\_crop\_layer

입력:&#x20;

* layer \*l: 크롭 레이어 구조체 포인터
* int w: 새로운 가로 크기
* int h: 새로운 세로 크기

동작:&#x20;

* 크롭 레이어의 가로 세로 크기를 변경하고, 출력 크기를 다시 계산하고, 출력 배열의 크기를 재할당한다.

설명:&#x20;

* 입력으로 받은 크롭 레이어 구조체 포인터를 사용하여 가로와 세로 크기를 변경한다.&#x20;
* 그 후, 새로운 가로와 세로 크기를 사용하여 출력 크기를 다시 계산하고, 입력 크기와 출력 크기를 사용하여 출력 배열의 크기를 재할당한다.



## make\_crop\_layer

```c
crop_layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure)
{
    fprintf(stderr, "Crop Layer: %d x %d -> %d x %d x %d image\n", h,w,crop_height,crop_width,c);
    crop_layer l = {0};
    l.type = CROP;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.scale = (float)crop_height / h;
    l.flip = flip;
    l.angle = angle;
    l.saturation = saturation;
    l.exposure = exposure;
    l.out_w = crop_width;
    l.out_h = crop_height;
    l.out_c = c;
    l.inputs = l.w * l.h * l.c;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.output = calloc(l.outputs*batch, sizeof(float));
    l.forward = forward_crop_layer;
    l.backward = backward_crop_layer;

    return l;
}
```

함수 이름: make\_crop\_layer&#x20;

입력:

* batch: int, 미니배치 크기
* h: int, 입력 이미지의 높이
* w: int, 입력 이미지의 너비
* c: int, 입력 이미지의 채널 수
* crop\_height: int, 자를 영역의 높이
* crop\_width: int, 자를 영역의 너비
* flip: int, 자르기 전 이미지를 수평으로 뒤집을지 여부
* angle: float, 이미지 회전 각도
* saturation: float, 이미지 포화도 조절 값
* exposure: float, 이미지 노출 조절 값

동작:

* 입력된 값을 바탕으로 crop\_layer 구조체를 생성하고 초기화한다.
* 출력되는 영상의 크기를 계산한다.
* 필요한 입력 및 출력 공간을 할당한다.
* forward\_crop\_layer 함수와 backward\_crop\_layer 함수를 설정한다.
* 초기화된 crop\_layer 구조체를 반환한다.

설명:&#x20;

* 이 함수는 crop\_layer를 생성하고 초기화하는 함수이다.&#x20;
* crop\_layer는 입력된 이미지에서 주어진 크기의 영역을 잘라내는 역할을 한다.&#x20;
* 입력된 인자를 바탕으로 crop\_layer 구조체를 생성하고 초기화한 후 반환한다.&#x20;
* 이때 forward\_crop\_layer 함수와 backward\_crop\_layer 함수를 설정해 주어야 한다.

