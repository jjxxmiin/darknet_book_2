# image\_opencv

## image\_to\_ipl

```c
#ifdef OPENCV

using namespace cv;

extern "C" {

IplImage *image_to_ipl(image im)
{
    int x,y,c;
    IplImage *disp = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(c= 0; c < im.c; ++c){
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val*255);
            }
        }
    }
    return disp;
}
```

함수 이름: image\_to\_ipl&#x20;

입력:&#x20;

* image im (이미지 구조체)&#x20;

동작:&#x20;

* 입력으로 들어온 이미지 구조체를 OpenCV의 IplImage 구조체로 변환하여 반환합니다.&#x20;
* 변환 과정에서는 이미지 데이터의 크기 및 채널에 맞게 IplImage 구조체를 생성한 뒤, 입력 이미지 데이터를 0\~255 범위로 스케일링하여 IplImage에 저장합니다.&#x20;

설명:&#x20;

* YOLO같은 딥러닝 모델에서는 이미지를 다루어야 하기 때문에, 이러한 이미지를 각 프레임마다 OpenCV의 IplImage 구조체로 변환하여 출력하기 위한 함수입니다.&#x20;
* IplImage 구조체는 OpenCV에서 이미지를 다루기 위해 사용되는 구조체로, 채널, 크기, 데이터 타입 등 이미지 정보를 담고 있습니다.&#x20;
* 이 함수에서는 입력으로 들어온 이미지 구조체를 IplImage 구조체로 변환하여 반환합니다.



## ipl\_to\_image

```c
image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return im;
}
```

함수 이름: ipl\_to\_image&#x20;

입력:&#x20;

* IplImage\* src (IplImage 포인터)&#x20;

동작:&#x20;

* OpenCV 라이브러리의 IplImage 포맷으로 저장된 이미지를 Darknet의 image 포맷으로 변환한다.&#x20;
* 주어진 IplImage 포인터로부터 이미지의 높이, 너비, 채널 수를 가져와서 이를 기반으로 새로운 image를 만든다.&#x20;
* 그리고 나서 IplImage의 imageData 포인터를 이용하여 이미지 데이터를 가져와서 새로 만든 image의 데이터 포인터에 할당한다.&#x20;

설명:

* OpenCV의 IplImage 포맷으로 저장된 이미지를 Darknet의 image 포맷으로 변환하는 함수이다.
* 주어진 IplImage 포인터로부터 이미지의 높이, 너비, 채널 수를 가져온다.
* 가져온 높이, 너비, 채널 수를 이용하여 make\_image 함수를 호출하여 새로운 image를 만든다.
* 이후, IplImage의 imageData 포인터를 이용하여 이미지 데이터를 가져와서 새로 만든 image의 데이터 포인터에 할당한다.
* 이미지 데이터의 픽셀 값 범위가 0~~255이므로, 255로 나누어서 0~~1 범위로 정규화한다. 반환값: 변환된 image 구조체



## image\_to\_mat

```c
Mat image_to_mat(image im)
{
    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy);

    IplImage *ipl = image_to_ipl(copy);
    Mat m = cvarrToMat(ipl, true);
    cvReleaseImage(&ipl);
    free_image(copy);
    return m;
}
```

함수 이름: image\_to\_mat

입력:&#x20;

* image im (입력 이미지)

동작:&#x20;

* 입력 이미지를 OpenCV의 Mat 형식으로 변환합니다.&#x20;
* 먼저 입력 이미지를 복사하고 제약 조건을 적용한 후, 입력 이미지가 3채널(RGB)인 경우 rgbgr 변환을 수행합니다.&#x20;
* 그 후, 변환된 이미지를 IplImage 형식으로 변환하고 cvarrToMat 함수를 사용하여 Mat 형식으로 변환합니다. 마지막으로 IplImage 메모리를 해제하고 이미지 복사본을 해제합니다.

설명:

* copy\_image(im): 입력 이미지의 복사본을 생성합니다.
* constrain\_image(copy): 이미지를 0\~1 값으로 제한합니다.
* rgbgr\_image(copy): 이미지 색상 채널을 RGB에서 BGR 순으로 변경합니다.
* image\_to\_ipl(copy): 이미지를 IplImage 형식으로 변환합니다.
* cvarrToMat(ipl, true): IplImage를 Mat 형식으로 변환합니다.
* cvReleaseImage(\&ipl): IplImage 메모리를 해제합니다.
* free\_image(copy): 이미지 복사본 메모리를 해제합니다.



## mat\_to\_image

```c
image mat_to_image(Mat m)
{
    IplImage ipl = m;
    image im = ipl_to_image(&ipl);
    rgbgr_image(im);
    return im;
}
```

함수 이름: mat\_to\_image

입력:&#x20;

* Mat m (OpenCV에서 제공하는 이미지 포맷 Mat 형식의 이미지)

동작:&#x20;

* OpenCV의 Mat 형식의 이미지를 Darknet의 image 형식으로 변환하는 함수입니다.&#x20;
* 먼저, Mat 형식의 이미지를 IplImage 형식으로 변환한 후 ipl\_to\_image 함수를 사용하여 Darknet의 image 형식으로 변환합니다.&#x20;
* 이후, rgbgr\_image 함수를 사용하여 이미지의 색상을 변환합니다.

설명:&#x20;

* 입력으로 받은 Mat 형식의 이미지를 Darknet의 image 형식으로 변환하여 반환하는 함수입니다.



## open\_video\_stream

```c
void *open_video_stream(const char *f, int c, int w, int h, int fps)
{
    VideoCapture *cap;
    if(f) cap = new VideoCapture(f);
    else cap = new VideoCapture(c);
    if(!cap->isOpened()) return 0;
    if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
    if(h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, w);
    if(fps) cap->set(CV_CAP_PROP_FPS, w);
    return (void *) cap;
}
```

함수 이름: open\_video\_stream

입력:

* const char \*f: 비디오 파일의 경로 (비디오 스트림이 아닌 경우 0으로 설정)
* int c: 비디오 캡처 장치의 인덱스 (비디오 파일이 아닌 경우 0으로 설정)
* int w: 비디오 프레임의 너비 (설정하지 않은 경우 0)
* int h: 비디오 프레임의 높이 (설정하지 않은 경우 0)
* int fps: 비디오의 초당 프레임 수 (설정하지 않은 경우 0)

동작:

* 입력으로 주어진 비디오 파일 또는 캡처 장치에서 비디오 스트림을 열고, 비디오 스트림을 캡처하는 데 사용되는 VideoCapture 객체를 생성한다.
* VideoCapture 객체가 정상적으로 열리지 않은 경우 0을 반환한다.
* 너비, 높이 및 FPS 값이 설정되었다면, 해당 값을 VideoCapture 객체에 설정한다.
* VideoCapture 객체의 포인터를 반환한다.

설명:

* 입력된 비디오 파일 경로나 카메라 장치 번호에 해당하는 비디오 스트림을 열고, 프레임의 너비, 높이, 속도를 설정한다.&#x20;
* 이때, 프레임 너비와 높이, 속도가 0이면 기본값으로 설정된다. 반환되는 VideoCapture 객체는 이후 비디오 프레임을 읽어오기 위해 사용된다.



## get\_image\_from\_stream

```c
image get_image_from_stream(void *p)
{
    VideoCapture *cap = (VideoCapture *)p;
    Mat m;
    *cap >> m;
    if(m.empty()) return make_empty_image(0,0,0);
    return mat_to_image(m);
}
```

함수 이름: get\_image\_from\_stream

입력:&#x20;

* void 포인터 p (영상 스트림 객체)

동작:&#x20;

* 입력으로 받은 영상 스트림 객체에서 현재 프레임을 읽어와 OpenCV의 Mat 형식으로 저장하고, 이를 Darknet의 image 형식으로 변환하여 반환한다.&#x20;
* 만약 현재 프레임이 없는 경우, 크기가 0인 빈 image를 반환한다.

설명:

* VideoCapture: OpenCV에서 영상 스트림을 처리하기 위한 클래스
* Mat: OpenCV에서 이미지를 처리하기 위한 클래스
* make\_empty\_image(w,h,c): Darknet에서 빈 image를 생성하는 함수 (너비 w, 높이 h, 채널 수 c)
* mat\_to\_image(m): OpenCV의 Mat 형식의 이미지를 Darknet의 image 형식으로 변환하는 함수



## load\_image\_cv

```c
image load_image_cv(char *filename, int channels)
{
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    Mat m;
    m = imread(filename, flag);
    if(!m.data){
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename);
        system(buff);
        return make_image(10,10,3);
        //exit(0);
    }
    image im = mat_to_image(m);
    return im;
}
```

함수 이름: load\_image\_cv&#x20;

입력:

* filename (char \*) : 이미지 파일 이름
* channels (int) : 채널 수 (0, 1, 3 중 하나)

동작:&#x20;

* OpenCV를 사용하여 이미지 파일을 로드하고, 채널 수를 지정할 수 있음.&#x20;
* 이미지를 Mat 형식으로 읽은 다음, mat\_to\_image() 함수를 사용하여 image 형식으로 변환하여 반환함.

설명:

* OpenCV를 사용하여 이미지 파일을 읽어옴
* channels 값에 따라 이미지를 grayscale 또는 color 이미지로 읽어옴 (channels=0인 경우 grayscale, channels=1인 경우 color, channels=3인 경우 RGB 이미지를 읽어옴)
* 이미지 파일이 없을 경우, 콘솔에 에러 메시지를 출력하고, 크기가 10x10이고 채널 수가 3인 빈 이미지를 생성하여 반환함.
* mat\_to\_image() 함수를 사용하여 Mat 형식의 이미지를 image 형식으로 변환하여 반환함.



## show\_image\_cv

```c
int show_image_cv(image im, const char* name, int ms)
{
    Mat m = image_to_mat(im);
    imshow(name, m);
    int c = waitKey(ms);
    if (c != -1) c = c%256;
    return c;
}
```

함수 이름: show\_image\_cv&#x20;

입력:&#x20;

* image im (표시할 이미지)
* const char\* name (윈도우 창 이름)
* int ms (윈도우가 열린 상태로 유지할 시간)

동작:&#x20;

* 입력으로 받은 이미지를 OpenCV Mat 형식으로 변환하고, 해당 이미지를 윈도우 창에 표시한다.&#x20;
* 그리고 윈도우가 열린 상태로 ms (입력으로 받은 시간) 밀리초만큼 대기한 후, 키보드 입력이 있으면 해당 입력의 아스키 코드 값을 반환하고, 그렇지 않으면 -1을 반환한다.&#x20;

설명:

* Darknet에서 표시할 이미지를 OpenCV의 imshow 함수를 사용하여 윈도우 창에 표시하는 함수이다.&#x20;
* 이미지를 OpenCV의 Mat 형식으로 변환하여 imshow 함수에 전달하고, 입력된 시간(ms)만큼 대기하다가 키보드 입력이 있으면 해당 입력의 아스키 코드 값을 반환한다.&#x20;
* 반환된 값이 -1이면 아무 입력도 없었다는 뜻이다.



## make\_window

```c
void make_window(char *name, int w, int h, int fullscreen)
{
    namedWindow(name, WINDOW_NORMAL);
    if (fullscreen) {
        setWindowProperty(name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    } else {
        resizeWindow(name, w, h);
        if(strcmp(name, "Demo") == 0) moveWindow(name, 0, 0);
    }
}

}
```

함수 이름: make\_window

입력:&#x20;

* name (char\*): 창의 이름
* w(int): 창의 너비
* h(int): 창의 높이
* fullscreen(int): 전체화면 여부(0 또는 1)

동작:&#x20;

* OpenCV 라이브러리의 namedWindow() 함수를 사용하여 이름이 name인 창을 생성한다.&#x20;
* fullscreen이 1인 경우 창을 전체화면으로 표시하고, 0인 경우 창의 크기를 w x h로 조정한다.&#x20;
* Demo라는 이름의 창인 경우, (0,0) 위치로 이동시킨다.

설명:&#x20;

* OpenCV를 사용하여 이미지를 보여주는 창을 만드는 함수이다.&#x20;
* 창의 이름과 크기, 전체화면 여부를 입력으로 받고, namedWindow()과 setWindowProperty() 또는 resizeWindow()와 moveWindow() 함수를 사용하여 창을 생성하거나 크기와 위치를 조정한다.

