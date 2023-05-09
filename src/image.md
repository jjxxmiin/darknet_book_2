# image

## get\_color

```c
float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} }; // BGR순서


float get_color(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    //printf("%f\n", r);
    return r;
}
```

함수 이름: get\_color

입력:&#x20;

* int c (색상 채널 인덱스, 0\~2)
* int x (현재 위치)
* int max (전체 범위)

동작:&#x20;

* 입력된 색상 채널 인덱스에 해당하는 값을 colors 배열에서 가져와, 현재 위치(x)와 전체 범위(max)의 비율을 이용하여 6개의 색상 중 2개의 색상을 선형 보간(linear interpolation)한 값을 반환한다.

설명:&#x20;

* colors 배열은 6개의 색상값(표현식에서는 BGR 순서로 저장되어 있음)을 가지고 있다.&#x20;
* get\_color 함수는 입력된 색상 채널 인덱스(c)에 해당하는 값(0\~2)을 이용하여 colors 배열에서 해당 색상 채널의 값을 가져오고, 현재 위치(x)와 전체 범위(max)의 비율을 계산하여 6개의 색상 중 어느 두 색상을 선형 보간할지 결정한다.&#x20;
* 그 후, 선형 보간한 결과를 반환한다.



## mask\_to\_rgb

```c
image mask_to_rgb(image mask)
{
    int n = mask.c;
    image im = make_image(mask.w, mask.h, 3);
    int i, j;
    for(j = 0; j < n; ++j){
        int offset = j*123457 % n;
        float red = get_color(2,offset,n);
        float green = get_color(1,offset,n);
        float blue = get_color(0,offset,n);
        for(i = 0; i < im.w*im.h; ++i){
            im.data[i + 0*im.w*im.h] += mask.data[j*im.h*im.w + i]*red;
            im.data[i + 1*im.w*im.h] += mask.data[j*im.h*im.w + i]*green;
            im.data[i + 2*im.w*im.h] += mask.data[j*im.h*im.w + i]*blue;
        }
    }
    return im;
}
```

함수 이름: mask\_to\_rgb

입력: image mask (이진 이미지)

동작: 입력으로 들어온 이진 이미지 mask를 RGB 이미지로 변환한다. 변환된 RGB 이미지는 색상이 다른 여러 개의 이진 이미지를 겹쳐서 만든 것처럼 보인다. 즉, mask에 있는 1의 위치에 해당하는 픽셀은 다양한 색상으로 색칠되며, 0의 위치에 해당하는 픽셀은 검은색으로 처리된다.

설명:

* n: mask의 채널 수
* im: mask와 같은 크기를 갖는 3채널 이미지
* i, j: 반복문을 위한 변수
* offset: j를 이용한 일종의 랜덤값. 여러 채널에서 겹쳐진 픽셀에 대해 서로 다른 색상이 적용되도록 하기 위함.
* red, green, blue: offset 값을 기반으로 6가지 색상 중 3가지를 선택하여 결정. (BGR 순서로)
* im.data: 이미지 데이터의 포인터를 가리키는 포인터
* im.w: 이미지의 너비
* im.h: 이미지의 높이
* mask.data: mask 이미지 데이터의 포인터를 가리키는 포인터
* i + 0 \* im.w \* im.h, i + 1 \* im.w \* im.h, i + 2 \* im.w \* im.h: RGB 채널 각각에 대해 해당되는 인덱스
* 겹쳐진 채널 간에 해당되는 픽셀에 대해 색상을 합산한다.
* mask의 값이 0일 경우 해당 픽셀은 색상이 변하지 않는다. 1일 경우, RGB 각 채널에 대해 위에서 결정된 값(red, green, blue)이 mask 값에 비례하여 곱해진다.



## get\_pixel

```c
static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}
```

함수 이름: get\_pixel\
입력:

* m: 이미지 데이터를 포함하는 image 구조체 변수
* x: 가로 좌표 값 (int 타입)
* y: 세로 좌표 값 (int 타입)
* c: 컬러 채널 값 (int 타입)

동작:&#x20;

* 이미지 데이터에서 지정된 좌표의 컬러값을 반환한다.

설명:

* 인덱싱은 0부터 시작하며, 입력받은 좌표와 컬러 채널 값에 해당하는 픽셀의 실수 값을 반환한다.&#x20;
* 함수 내부에서는 입력 받은 좌표와 채널 값이 이미지의 크기와 채널 수 내에 있는지 확인하고, 그렇지 않을 경우 assert 함수를 통해 프로그램을 강제 종료시킨다.



## get\_pixel\_extend

```c
static float get_pixel_extend(image m, int x, int y, int c)
{
    if(x < 0 || x >= m.w || y < 0 || y >= m.h) return 0;

    if(c < 0 || c >= m.c) return 0;
    return get_pixel(m, x, y, c);
}
```

함수 이름: get\_pixel\_extend&#x20;

입력:&#x20;

* image m (이미지 구조체)
* int x (가로 좌표)
* int y (세로 좌표)
* int c (채널 인덱스)

동작:&#x20;

* 입력으로 받은 좌표와 채널 인덱스에 해당하는 픽셀값을 가져온다.&#x20;
* 만약 좌표가 이미지의 범위를 벗어나면 0을 반환한다.&#x20;

설명:&#x20;

* 이미지에서 해당 좌표와 채널 인덱스에 해당하는 픽셀값을 가져오는 함수이다.&#x20;
* 범위를 벗어나는 경우 0을 반환한다.&#x20;
* 이 함수는 이미지를 확장하는 데 사용된다.



## set\_pixel

```c
static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}
```

함수 이름: set\_pixel

입력:&#x20;

* image m (이미지 구조체)
* int x (픽셀의 x 좌표)
* int y (픽셀의 y 좌표)
* int c (채널 번호)
* float val (설정할 값)

동작:&#x20;

* 입력으로 받은 이미지 m의 (x, y) 좌표에 있는 c번째 채널의 값을 val로 설정한다.

설명:&#x20;

* 입력으로 받은 이미지 m의 (x, y) 좌표에 있는 c번째 채널의 값을 val로 설정하는 함수이다.&#x20;
* 이미지 m의 크기를 벗어나는 좌표나 채널 번호가 주어진 경우 함수는 즉시 반환된다.&#x20;
* 함수 내부에서는 assert 함수를 사용하여 입력 좌표와 채널 번호가 이미지의 크기를 벗어나지 않는지 검사한다.



## add\_pixel

```c
static void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}
```

함수 이름: add\_pixel

입력:&#x20;

* image m (이미지 구조체 포인터)
* int x (x 좌표값)
* int y (y 좌표값)
* int c (색상 채널)
* float val (더할 값)

동작:&#x20;

* 입력으로 주어진 이미지에서 (x, y, c) 좌표의 픽셀값에 val을 더합니다.

설명:&#x20;

* 이미지 구조체 m에서 (x, y, c) 좌표의 픽셀값에 val을 더하는 함수입니다.&#x20;
* 이미지의 가로, 세로, 채널 정보는 이미지 구조체 내에 저장되어 있으며, 이미지 데이터는 1차원 배열 형태로 저장됩니다.&#x20;
* 따라서 이미지 내의 특정 픽셀에 접근하려면, 1차원 배열에서의 인덱스를 계산해야 합니다.&#x20;
* 이 함수는 해당 인덱스를 계산하여 해당 위치의 픽셀값에 val을 더합니다.



## bilinear\_interpolate

```c
static float bilinear_interpolate(image im, float x, float y, int c)
{
    int ix = (int) floorf(x);
    int iy = (int) floorf(y);

    float dx = x - ix;
    float dy = y - iy;

    float val = (1-dy) * (1-dx) * get_pixel_extend(im, ix, iy, c) +
        dy     * (1-dx) * get_pixel_extend(im, ix, iy+1, c) +
        (1-dy) *   dx   * get_pixel_extend(im, ix+1, iy, c) +
        dy     *   dx   * get_pixel_extend(im, ix+1, iy+1, c);
    return val;
}
```

함수 이름: bilinear\_interpolate 입력:

* image im : 보간을 수행할 이미지 데이터를 가리키는 포인터
* float x : 보간을 수행할 좌표값 x
* float y : 보간을 수행할 좌표값 y
* int c : 보간을 수행할 이미지 채널

동작:&#x20;

* 입력으로 주어진 이미지의 (x, y) 좌표에서의 채널 c에 대한 보간값을 계산한다.&#x20;
* 보간은 4개의 꼭짓점에 대한 bilinear interpolation 방식을 사용한다.

설명:&#x20;

* bilinear interpolation은 이미지 보간 기법 중 하나로, 4개의 인접한 픽셀 값을 사용하여 주어진 좌표에서의 값을 계산하는 방식이다.&#x20;
* 입력으로 주어진 이미지에서 (x, y) 좌표에서의 채널 c에 대한 보간값을 계산하기 위해 4개의 인접한 픽셀 값을 가져온다.&#x20;
* 이때, x와 y의 정수 부분은 좌표 인덱스로 사용되고, 소수 부분은 보간 계수로 사용된다.&#x20;
* 이렇게 계산된 보간값이 반환된다.



## composite\_image

```c
void composite_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float val = get_pixel(source, x, y, k);
                float val2 = get_pixel_extend(dest, dx+x, dy+y, k);
                set_pixel(dest, dx+x, dy+y, k, val * val2);
            }
        }
    }
}
```

함수 이름: composite\_image

입력:

* source: 합성할 이미지 (image)
* dest: 합성 대상 이미지 (image)
* dx: dest 이미지에서 source 이미지가 시작될 x좌표 (int)
* dy: dest 이미지에서 source 이미지가 시작될 y좌표 (int)

동작:&#x20;

* source 이미지를 dest 이미지의 (dx, dy) 좌표부터 합성하는 함수이다.&#x20;
* source 이미지와 dest 이미지의 같은 위치의 픽셀값을 곱하여 dest 이미지에 덮어쓴다.

설명:&#x20;

* source 이미지와 dest 이미지의 같은 위치의 픽셀을 곱한 값을 dest 이미지에 덮어쓰므로, 합성 결과는 source 이미지가 dest 이미지 위에 덮여진 이미지가 된다.&#x20;
* 함수 내부에서는 이중 for 루프를 이용하여 source 이미지의 모든 픽셀에 대해 dest 이미지에 덮어쓰는 동작을 수행한다.



## border\_image

```c

image border_image(image a, int border)
{
    image b = make_image(a.w + 2*border, a.h + 2*border, a.c);
    int x,y,k;
    for(k = 0; k < b.c; ++k){
        for(y = 0; y < b.h; ++y){
            for(x = 0; x < b.w; ++x){
                float val = get_pixel_extend(a, x - border, y - border, k);
                if(x - border < 0 || x - border >= a.w || y - border < 0 || y - border >= a.h) val = 1;
                set_pixel(b, x, y, k, val);
            }
        }
    }
    return b;
}
```

함수 이름: border\_image&#x20;

입력:&#x20;

* image a (이미지)
* int border (테두리 크기)&#x20;

동작:&#x20;

* 입력 이미지 주위에 지정된 크기의 검은색 테두리를 추가한 이미지를 만든다.&#x20;
* 검은색 테두리의 크기는 입력된 border 값에 의해 결정된다.&#x20;
* 입력 이미지의 가장자리 픽셀은 테두리를 만들기 위해 사용되며, 검은색으로 설정된다.&#x20;

설명:&#x20;

* 입력된 이미지 a를 복제하고 입력된 border 값에 따라 이미지 크기를 늘린다.&#x20;
* 그런 다음 각 채널과 픽셀의 값을 테두리 안으로 이동시켜 설정하면서, 테두리에 검은색 픽셀을 추가한다.



## tile\_images

```c
image tile_images(image a, image b, int dx)
{
    if(a.w == 0) return copy_image(b);
    image c = make_image(a.w + b.w + dx, (a.h > b.h) ? a.h : b.h, (a.c > b.c) ? a.c : b.c);
    fill_cpu(c.w*c.h*c.c, 1, c.data, 1);
    embed_image(a, c, 0, 0);
    composite_image(b, c, a.w + dx, 0);
    return c;
}
```

함수 이름: tile\_images

입력:

* image a: 이미지 a
* image b: 이미지 b
* int dx: a와 b 사이의 간격

동작:

* 이미지 a와 b를 하나의 이미지로 합친다.
* 이미지 a와 b 중에서 크기가 더 큰 쪽을 기준으로 이미지 c를 만든다.
* 이미지 a를 (0, 0) 좌표에 삽입한다.
* 이미지 b를 (a.w + dx, 0) 좌표에 삽입한다.
* 이미지 c를 반환한다.

설명:&#x20;

* 두 개의 이미지를 이어 붙여서 하나의 이미지로 만드는 함수이다.&#x20;
* 두 이미지의 크기는 달라도 된다. a와 b 사이에는 dx만큼의 간격이 생긴다.&#x20;
* 이미지 a와 b 중에서 크기가 더 큰 쪽을 기준으로 이미지 c를 만들고, 이미지 a와 b를 합친다.&#x20;
* 이미지 a는 (0, 0) 좌표에, 이미지 b는 (a.w + dx, 0) 좌표에 삽입된다. 만들어진 이미지 c를 반환한다.



## get\_label

```c
image get_label(image **characters, char *string, int size)
{
    size = size/10;
    if(size > 7) size = 7;
    image label = make_empty_image(0,0,0);
    while(*string){
        image l = characters[size][(int)*string];
        image n = tile_images(label, l, -size - 1 + (size+1)/2);
        free_image(label);
        label = n;
        ++string;
    }
    image b = border_image(label, label.h*.25);
    free_image(label);
    return b;
}
```

함수 이름: get\_label&#x20;

입력:

* characters: image 구조체의 이중포인터
* string: 문자열(char 배열)
* size: 폰트 크기(int)

동작:&#x20;

* 주어진 문자열을 폰트 크기에 따라 이미지로 변환한다.&#x20;
* 이 때, characters 배열은 미리 생성된 문자 이미지 배열을 저장하고 있으며, 해당 문자열을 이루는 각각의 문자에 대해 적절한 이미지를 찾아 이어 붙여 최종적으로 전체 문자열을 나타내는 이미지를 만든다.&#x20;
* 만들어진 이미지에는 경계를 가지도록 border\_image 함수를 사용하여 경계선을 추가하고, 최종적으로 경계선이 추가된 이미지를 반환한다.

설명:

* characters: 각 문자에 해당하는 이미지가 저장된 이중 포인터
* string: 문자열(char 배열)
* size: 폰트 크기(int)로 주어진 값을 10으로 나눈 후, 7 이상이면 7로 설정한다.
* label: 이미지 구조체로 초기화되어있는 빈 이미지를 생성한다.
* while 루프: 문자열에 포함된 문자를 하나씩 처리하며 label에 해당 문자의 이미지를 추가한다.
* l: 현재 처리 중인 문자에 해당하는 이미지
* n: label과 l을 이어붙인 이미지
* free\_image(label): 이전 루프에서 사용되었던 label 이미지를 해제한다.
* label = n: 이전 루프에서 생성된 이미지와 현재 문자의 이미지를 이어 붙인 결과를 label에 대입한다.
* \++string: 문자열 포인터를 다음 문자로 이동시킨다.
* b: 경계가 추가된 이미지를 저장하는 변수로, border\_image 함수를 이용하여 경계를 추가한다.
* free\_image(label): label 이미지를 메모리에서 해제한다.
* return b: 경계가 추가된 이미지를 반환한다.



## draw\_label

```c
void draw_label(image a, int r, int c, image label, const float *rgb)
{
    int w = label.w;
    int h = label.h;
    if (r - h >= 0) r = r - h;

    int i, j, k;
    for(j = 0; j < h && j + r < a.h; ++j){
        for(i = 0; i < w && i + c < a.w; ++i){
            for(k = 0; k < label.c; ++k){
                float val = get_pixel(label, i, j, k);
                set_pixel(a, i+c, j+r, k, rgb[k] * val);
            }
        }
    }
}
```

함수 이름: draw\_label

입력:

* image a: 라벨이 그려질 이미지
* int r: 라벨이 그려질 행 번호
* int c: 라벨이 그려질 열 번호
* image label: 그려질 라벨 이미지
* const float \*rgb: 라벨 색상을 결정하는 RGB 값 배열

동작:

* 주어진 이미지 a의 특정 위치에 주어진 라벨 이미지를 그린다.
* 주어진 RGB 값 배열을 사용하여 라벨을 색상화한다.

설명:

* 입력으로 주어진 라벨 이미지를 주어진 위치에 그리는 함수이다.
* 라벨 이미지가 이미지 a의 범위를 벗어나면 그리지 않는다.
* 그려질 라벨 이미지의 크기는 이미지 a 내에서의 위치를 기준으로 결정된다.
* 라벨 색상은 주어진 RGB 값 배열을 사용하여 결정된다.



## draw\_box

```c
void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    //normalize_image(a);
    int i;
    if(x1 < 0) x1 = 0;
    if(x1 >= a.w) x1 = a.w-1;
    if(x2 < 0) x2 = 0;
    if(x2 >= a.w) x2 = a.w-1;

    if(y1 < 0) y1 = 0;
    if(y1 >= a.h) y1 = a.h-1;
    if(y2 < 0) y2 = 0;
    if(y2 >= a.h) y2 = a.h-1;

    for(i = x1; i <= x2; ++i){
        a.data[i + y1*a.w + 0*a.w*a.h] = r;
        a.data[i + y2*a.w + 0*a.w*a.h] = r;

        a.data[i + y1*a.w + 1*a.w*a.h] = g;
        a.data[i + y2*a.w + 1*a.w*a.h] = g;

        a.data[i + y1*a.w + 2*a.w*a.h] = b;
        a.data[i + y2*a.w + 2*a.w*a.h] = b;
    }
    for(i = y1; i <= y2; ++i){
        a.data[x1 + i*a.w + 0*a.w*a.h] = r;
        a.data[x2 + i*a.w + 0*a.w*a.h] = r;

        a.data[x1 + i*a.w + 1*a.w*a.h] = g;
        a.data[x2 + i*a.w + 1*a.w*a.h] = g;

        a.data[x1 + i*a.w + 2*a.w*a.h] = b;
        a.data[x2 + i*a.w + 2*a.w*a.h] = b;
    }
}
```

함수 이름: draw\_box 입력:

* image a: 그림을 그릴 이미지
* int x1: 상자의 왼쪽 상단 모서리 x좌표
* int y1: 상자의 왼쪽 상단 모서리 y좌표
* int x2: 상자의 오른쪽 하단 모서리 x좌표
* int y2: 상자의 오른쪽 하단 모서리 y좌표
* float r: 상자 선 색상의 R 채널 값
* float g: 상자 선 색상의 G 채널 값
* float b: 상자 선 색상의 B 채널 값

동작:&#x20;

* 입력으로 주어진 이미지 a에 상자를 그린다.&#x20;
* 상자는 입력으로 주어진 좌표 x1, y1을 왼쪽 상단 모서리로 하고, x2, y2를 오른쪽 하단 모서리로 하는 사각형이며, 선 색상은 입력으로 주어진 r, g, b 값으로 결정된다.

설명:&#x20;

* 입력으로 주어진 이미지 a에 상자를 그리는 함수이다.&#x20;
* x1, y1, x2, y2로 주어진 좌표로 사각형의 모서리를 결정하고, r, g, b로 주어진 값으로 사각형의 선 색상을 결정한다.&#x20;
* 그리는 과정에서 이미지의 경계를 벗어나는 경우, 해당 좌표를 경계 값으로 조정한다.



## draw\_box\_width

```c
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
    int i;
    for(i = 0; i < w; ++i){
        draw_box(a, x1+i, y1+i, x2-i, y2-i, r, g, b);
    }
}
```

함수 이름: draw\_box\_width&#x20;

입력:

* image a: 사각형이 그려질 이미지
* int x1: 사각형의 왼쪽 위 모서리의 x 좌표
* int y1: 사각형의 왼쪽 위 모서리의 y 좌표
* int x2: 사각형의 오른쪽 아래 모서리의 x 좌표
* int y2: 사각형의 오른쪽 아래 모서리의 y 좌표
* int w: 그려질 사각형의 두께
* float r: 사각형의 빨간색 채널 값 (0\~1)
* float g: 사각형의 초록색 채널 값 (0\~1)
* float b: 사각형의 파란색 채널 값 (0\~1)

동작:&#x20;

* 입력으로 받은 이미지 a에 x1, y1 좌표에서부터 x2, y2 좌표까지 두께가 w인 사각형을 그립니다. 색은 r, g, b 값으로 결정됩니다.

설명:&#x20;

* draw\_box 함수를 이용하여 입력으로 받은 이미지 a에 여러 개의 선을 그리면서 두께가 w인 사각형을 그리는 함수입니다.&#x20;
* 이 함수는 draw\_box 함수를 호출하여 두께가 1인 사각형을 그리는 것을 w번 반복하여 두께가 w인 사각형을 그립니다.



## draw\_bbox

```c
void draw_bbox(image a, box bbox, int w, float r, float g, float b)
{
    int left  = (bbox.x-bbox.w/2)*a.w;
    int right = (bbox.x+bbox.w/2)*a.w;
    int top   = (bbox.y-bbox.h/2)*a.h;
    int bot   = (bbox.y+bbox.h/2)*a.h;

    int i;
    for(i = 0; i < w; ++i){
        draw_box(a, left+i, top+i, right-i, bot-i, r, g, b);
    }
}
```

함수 이름: draw\_bbox

입력:

* image a: 바운딩 박스가 그려질 이미지
* box bbox: 그려질 바운딩 박스의 정보가 담긴 box 구조체
* int w: 그려질 바운딩 박스의 선 두께
* float r: 그려질 바운딩 박스의 빨간색(R) 성분 값 (0 \~ 1)
* float g: 그려질 바운딩 박스의 녹색(G) 성분 값 (0 \~ 1)
* float b: 그려질 바운딩 박스의 파란색(B) 성분 값 (0 \~ 1)

동작:&#x20;

* 입력으로 주어진 이미지 a에, 입력으로 주어진 box bbox의 정보를 이용하여 바운딩 박스를 그린다.
* 그려진 바운딩 박스의 선 두께는 입력으로 주어진 w이고, 선의 색은 입력으로 주어진 RGB 성분 값 r, g, b로 지정된다.

설명:

* 바운딩 박스의 정보를 담고 있는 box 구조체는 x, y, w, h의 4개 필드를 가지며, 각각은 바운딩 박스의 중심 x좌표, 중심 y좌표, 너비, 높이를 나타낸다.
* 입력으로 주어진 바운딩 박스의 정보를 이용하여, 이미지 a 상에 바운딩 박스를 그리는 함수이다.
* 바운딩 박스의 좌측 상단 모서리의 좌표(left, top)와 우측 하단 모서리의 좌표(right, bot)를 계산한다.
* draw\_box 함수를 이용하여, 계산된 좌표를 이용하여 선 두께만큼 여러번 그려줌으로써 두꺼운 선의 바운딩 박스를 그린다.



## load\_alphabet

```c
image **load_alphabet()
{
    int i, j;
    const int nsize = 8;
    image **alphabets = calloc(nsize, sizeof(image));
    for(j = 0; j < nsize; ++j){
        alphabets[j] = calloc(128, sizeof(image));
        for(i = 32; i < 127; ++i){
            char buff[256];
            sprintf(buff, "data/labels/%d_%d.png", i, j);
            alphabets[j][i] = load_image_color(buff, 0, 0);
        }
    }
    return alphabets;
}
```

함수 이름: load\_alphabet&#x20;

입력:&#x20;

* 없음&#x20;

동작:&#x20;

* 128개의 이미지로 이루어진 8 x 128 크기의 이차원 배열(alphabets)을 생성하고, 이미지 경로에 따라 128개의 이미지를 불러와 배열에 저장한다.&#x20;

설명:

* 이 함수는 텍스트 인식을 위한 영문 대소문자와 숫자를 나타내는 이미지들을 불러와서 이차원 배열에 저장하는 함수이다.
* 8개의 알파벳 이미지들(nsize)을 저장할 8 x 128 크기의 이차원 배열(alphabets)을 동적으로 할당한다.
* 이차원 배열을 루프를 돌며, 이미지 경로를 문자열로 생성하고 load\_image\_color() 함수를 사용하여 이미지를 불러온 후 배열에 저장한다.
* 각 이미지는 ASCII 코드의 32부터 127까지의 문자에 대응된다.



## draw\_detections

```c
void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes)
{
    int i,j;

    for(i = 0; i < num; ++i){
        char labelstr[4096] = {0};
        int class = -1;
        for(j = 0; j < classes; ++j){                                       
            if (dets[i].prob[j] > thresh){                                      // probability가 thresh보다 큰 경우의 클래스를 출력합니다.
                if (class < 0) {
                    strcat(labelstr, names[j]);
                    class = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                printf("%s: %.0f%%\n", names[j], dets[i].prob[j]*100);
            }
        }
        if(class >= 0){
            int width = im.h * .006;

            /*
               if(0){
               width = pow(prob, 1./2.)*10+1;
               alphabet = 0;
               }
             */

            //printf("%d %s: %.0f%%\n", i, names[class], prob*100);
            int offset = class*123457 % classes;
            float red = get_color(2,offset,classes);
            float green = get_color(1,offset,classes);
            float blue = get_color(0,offset,classes);
            float rgb[3];

            //width = prob*20+2;

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b = dets[i].bbox;
            //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

            draw_box_width(im, left, top, right, bot, width, red, green, blue); // box를 그립니다.
            if (alphabet) {                                                     // label을 표시합니다.
                image label = get_label(alphabet, labelstr, (im.h*.03));
                draw_label(im, top + width, left, label, rgb);
                free_image(label);
            }
            if (dets[i].mask){                                                  //
                image mask = float_to_image(14, 14, 1, dets[i].mask);
                image resized_mask = resize_image(mask, b.w*im.w, b.h*im.h);
                image tmask = threshold_image(resized_mask, .5);
                embed_image(tmask, im, left, top);
                free_image(mask);
                free_image(resized_mask);
                free_image(tmask);
            }
        }
    }
}
```

함수 이름: draw\_detections

입력:

* im: detection 결과를 그릴 이미지 (image 타입)
* dets: detection 결과 (detection 타입 배열)
* num: detection 결과의 개수 (int 타입)
* thresh: detection 결과의 확률 임계값 (float 타입)
* names: 클래스 이름 (char 타입 포인터 배열)
* alphabet: label 표시에 사용될 이미지 (image 타입 포인터 배열)
* classes: 클래스의 개수 (int 타입)

동작:

* detection 결과를 바탕으로 box와 label을 그리는 함수입니다.
* 각 detection 결과마다 해당 클래스의 이름과 확률을 출력합니다.
* 클래스마다 색상을 다르게 지정하여 box를 그리고, label을 표시합니다.
* 만약 mask 정보가 있다면, 해당 mask 정보도 이미지에 표시합니다.

설명:&#x20;

* 이 함수는 detection 결과를 바탕으로 이미지에 box와 label을 그리는 함수입니다. 입력으로는 detection 결과를 그릴 이미지(im), detection 결과(dets), detection 결과의 개수(num), detection 결과의 확률 임계값(thresh), 클래스 이름(names), label 표시에 사용될 이미지(alphabet), 클래스의 개수(classes)를 받습니다.
* 각 detection 결과마다 해당 클래스의 이름과 확률을 출력하고, 클래스마다 색상을 다르게 지정하여 box를 그리고, label을 표시합니다. 만약 mask 정보가 있다면, 해당 mask 정보도 이미지에 표시합니다.



## transpose\_image

```c
void transpose_image(image im)
{
    assert(im.w == im.h);
    int n, m;
    int c;
    for(c = 0; c < im.c; ++c){
        for(n = 0; n < im.w-1; ++n){
            for(m = n + 1; m < im.w; ++m){
                float swap = im.data[m + im.w*(n + im.h*c)];
                im.data[m + im.w*(n + im.h*c)] = im.data[n + im.w*(m + im.h*c)];
                im.data[n + im.w*(m + im.h*c)] = swap;
            }
        }
    }
}
```

```
-------------                   -------------
| 1 | 2 | 3 |                   | 1 | 4 | 7 |
-------------                   -------------
| 4 | 5 | 6 |        ->         | 2 | 5 | 8 |
-------------                   -------------
| 7 | 8 | 9 |                   | 3 | 6 | 9 |
-------------                   -------------
```

함수 이름: transpose\_image

입력:

* image im: 전치할 이미지

동작:&#x20;

* 이미지를 전치하는 함수입니다. 이미지의 가로와 세로가 같아야하며, 입력된 이미지의 채널 수에 따라 모든 채널에 대해 전치를 수행합니다.

설명:&#x20;

* 주어진 이미지의 가로와 세로가 같은지 확인하고, 모든 채널에 대해 전치를 수행합니다.&#x20;
* 전치는 행렬에서 행과 열을 바꾸는 작업입니다.&#x20;
* 전치된 이미지는 입력 이미지의 가로와 세로가 서로 바뀐 이미지가 됩니다.



## flip\_images

```c
void flip_image(image a)
{
    int i,j,k;
    for(k = 0; k < a.c; ++k){
        for(i = 0; i < a.h; ++i){
            for(j = 0; j < a.w/2; ++j){
                int index = j + a.w*(i + a.h*(k));
                int flip = (a.w - j - 1) + a.w*(i + a.h*(k));
                float swap = a.data[flip];
                a.data[flip] = a.data[index];
                a.data[index] = swap;
            }
        }
    }
}
```

함수 이름: flip\_image

입력:&#x20;

* image a (이미지 구조체 포인터)

동작:&#x20;

* 주어진 이미지를 좌우 반전시킵니다.

설명:&#x20;

* 입력으로 주어진 이미지의 각 픽셀 값을 좌우 반전시킵니다.&#x20;
* 이미지는 채널마다 따로 처리됩니다.&#x20;
* 예를 들어, RGB 채널이 있는 경우 R, G, B 채널이 각각 독립적으로 좌우 반전됩니다.



## rotate\_image\_cw

```c
void rotate_image_cw(image im, int times)
{
    assert(im.w == im.h);
    times = (times + 400) % 4;
    int i, x, y, c;
    int n = im.w;
    for(i = 0; i < times; ++i){
        for(c = 0; c < im.c; ++c){
            for(x = 0; x < n/2; ++x){
                for(y = 0; y < (n-1)/2 + 1; ++y){
                    float temp = im.data[y + im.w*(x + im.h*c)];
                    im.data[y + im.w*(x + im.h*c)] = im.data[n-1-x + im.w*(y + im.h*c)];
                    im.data[n-1-x + im.w*(y + im.h*c)] = im.data[n-1-y + im.w*(n-1-x + im.h*c)];
                    im.data[n-1-y + im.w*(n-1-x + im.h*c)] = im.data[x + im.w*(n-1-y + im.h*c)];
                    im.data[x + im.w*(n-1-y + im.h*c)] = temp;
                }
            }
        }
    }
}
```

함수 이름: rotate\_image\_cw&#x20;

입력:&#x20;

* image im (회전할 이미지)
* int times (회전할 횟수)

동작:&#x20;

* 입력된 이미지를 시계 방향으로 지정된 횟수만큼 회전시킵니다.&#x20;
* 회전은 90도 단위로 이루어지며, 회전 횟수는 0, 1, 2, 3 중 하나의 값을 가져야합니다.&#x20;
* 회전은 입력된 이미지의 중심을 기준으로 이루어집니다.&#x20;

설명:&#x20;

* 함수는 입력된 이미지를 시계 방향으로 회전시키는 동작을 수행합니다.&#x20;
* 회전할 횟수는 times 매개 변수로 지정됩니다.&#x20;
* 회전은 90도 단위로 이루어지며, times 값이 0, 1, 2, 3 중 하나인지 확인합니다.&#x20;
* 회전은 입력된 이미지의 중심을 기준으로 이루어지며, 회전된 이미지는 입력된 이미지를 대체합니다.&#x20;
* 함수는 입력된 이미지를 수정하므로, 원본 이미지의 백업을 만드는 것이 좋습니다.



## image\_distance

```c
image image_distance(image a, image b)
{
    int i,j;
    image dist = make_image(a.w, a.h, 1);
    for(i = 0; i < a.c; ++i){
        for(j = 0; j < a.h*a.w; ++j){
            dist.data[j] += pow(a.data[i*a.h*a.w+j]-b.data[i*a.h*a.w+j],2);
        }
    }
    for(j = 0; j < a.h*a.w; ++j){
        dist.data[j] = sqrt(dist.data[j]);
    }
    return dist;
}
```

함수 이름: image\_distance

입력:&#x20;

* image a (첫 번째 이미지)
* image b (두 번째 이미지)

동작:&#x20;

* 입력으로 받은 두 이미지 a와 b 사이의 거리(distance)를 계산합니다.&#x20;
* 거리 계산은 유클리드 거리를 사용하며, 각 픽셀의 차이를 제곱한 값들의 합을 계산하고, 이 값을 루트 씌워줍니다.&#x20;
* 이렇게 계산된 거리 값을 가지고 새로운 이미지 dist를 만들어 반환합니다.

설명:&#x20;

* 입력으로 받은 두 이미지 a와 b의 크기는 같아야 합니다.&#x20;
* 거리 값이 작을수록 두 이미지가 서로 비슷한 이미지입니다.&#x20;
* 이 함수는 주로 이미지 검색(image retrieval)에서 사용됩니다.



## ghost\_image

```c
void ghost_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    float max_dist = sqrt((-source.w/2. + .5)*(-source.w/2. + .5));
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float dist = sqrt((x - source.w/2. + .5)*(x - source.w/2. + .5) + (y - source.h/2. + .5)*(y - source.h/2. + .5));
                float alpha = (1 - dist/max_dist);
                if(alpha < 0) alpha = 0;
                float v1 = get_pixel(source, x,y,k);
                float v2 = get_pixel(dest, dx+x,dy+y,k);
                float val = alpha*v1 + (1-alpha)*v2;
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}
```

함수 이름: ghost\_image

입력:

* source: ghost 이미지의 소스 이미지 (image 타입)
* dest: ghost 이미지가 적용될 대상 이미지 (image 타입)
* dx: dest 이미지에서 ghost 이미지가 시작될 x 좌표
* dy: dest 이미지에서 ghost 이미지가 시작될 y 좌표

동작:&#x20;

* 소스 이미지를 기반으로 ghost 이미지를 생성하고, dest 이미지에 적용하는 함수입니다.&#x20;
* 소스 이미지의 각 픽셀과 대상 이미지의 해당 위치의 픽셀을 비교하여, 두 픽셀 사이의 거리에 따라 alpha 값을 계산합니다.&#x20;
* alpha 값을 사용하여 ghost 이미지의 해당 위치의 픽셀 값을 계산하고, dest 이미지에 ghost 이미지를 적용합니다.

설명:

* max\_dist: ghost 이미지의 중심으로부터 가장 먼 거리
* dist: 소스 이미지의 해당 픽셀과 ghost 이미지의 중심 사이의 거리
* alpha: ghost 이미지에서 소스 이미지의 픽셀이 차지하는 가중치. 거리가 작을수록 가중치가 큽니다.
* v1: 소스 이미지의 해당 픽셀 값
* v2: 대상 이미지의 해당 위치의 픽셀 값
* val: ghost 이미지의 해당 위치의 픽셀 값



## blocky\_image

```c
void blocky_image(image im, int s)
{
    int i,j,k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < im.h; ++j){
            for(i = 0; i < im.w; ++i){
                im.data[i + im.w*(j + im.h*k)] = im.data[i/s*s + im.w*(j/s*s + im.h*k)];
            }
        }
    }
}
```

함수 이름: blocky\_image

입력:&#x20;

* image im (입력 이미지)
* int s (블록 크기)

동작:&#x20;

* 입력 이미지를 블록 단위로 잘라서 블록 내의 모든 픽셀값을 블록 내 첫번째 픽셀값으로 대체하는 함수입니다.&#x20;
* 즉, 블록 크기 s로 입력 이미지를 잘랐을 때, 한 블록 내의 픽셀값은 첫번째 픽셀값으로 대체됩니다.&#x20;
* 이는 블록 내 픽셀값들이 크게 차이나지 않도록 하여 이미지의 "blocky"한 효과를 내는 데 사용됩니다.

설명:&#x20;

* 입력 이미지 im의 가로, 세로, 채널 수를 각각 im.w, im.h, im.c라고 할 때, 이 함수는 총 im.w \* im.h \* im.c 개의 픽셀을 가진 이미지를 입력받습니다.&#x20;
* 입력받은 이미지를 블록 크기 s로 나눠서 각 블록의 첫번째 픽셀값을 제외한 모든 픽셀값을 첫번째 픽셀값으로 대체합니다.&#x20;
* 이러한 과정을 거치면 블록 내 픽셀값이 첫번째 픽셀값과 같아지므로, 입력 이미지의 모든 블록이 뚜렷한 경계로 구분되는 "blocky"한 효과를 낼 수 있습니다.



## censor\_image

```c
void censor_image(image im, int dx, int dy, int w, int h)
{
    int i,j,k;
    int s = 32;
    if(dx < 0) dx = 0;
    if(dy < 0) dy = 0;

    for(k = 0; k < im.c; ++k){
        for(j = dy; j < dy + h && j < im.h; ++j){
            for(i = dx; i < dx + w && i < im.w; ++i){
                im.data[i + im.w*(j + im.h*k)] = im.data[i/s*s + im.w*(j/s*s + im.h*k)];
                //im.data[i + j*im.w + k*im.w*im.h] = 0;
            }
        }
    }
}
```

함수 이름: censor\_image

입력:

* im: 처리할 이미지 (image 타입)
* dx: 가릴 영역의 왼쪽 상단 모서리 x좌표 (int 타입)
* dy: 가릴 영역의 왼쪽 상단 모서리 y좌표 (int 타입)
* w: 가릴 영역의 너비 (int 타입)
* h: 가릴 영역의 높이 (int 타입)

동작:

* im 이미지의 (dx, dy) 좌표를 시작으로 w×h 크기의 영역을 검정색으로 가린다.
* 단, 이 때, 가리는 영역은 32×32 크기의 작은 블록들로 나누어져 있으며, 검정색 블록들로 이루어진 가림막이 적용된다.

설명:&#x20;

* 이미지(im)의 특정 영역(dx, dy)를 검정색 블록으로 가림으로써 해당 영역을 숨기는 함수이다.&#x20;
* 만약 dx, dy가 음수인 경우에는 좌측 상단 모서리가 이미지 바깥으로 벗어나지 않도록 보정된다.&#x20;
* 가릴 영역은 32×32 크기의 작은 블록으로 이루어진 가림막으로 처리되어, 검정색 블록으로 가려진 영역의 크기는 32의 배수가 된다. 따라서 w와 h 값이 32의 배수가 아닌 경우, 마지막 블록의 일부분만 가릴 수 있다.



## embed\_image

```c
void embed_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float val = get_pixel(source, x,y,k);
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}
```

함수 이름: embed\_image

입력:

* source: 이미지 타입(image)의 소스 이미지
* dest: 이미지 타입(image)의 목적 이미지
* dx: 목적 이미지에서 소스 이미지가 삽입될 x좌표
* dy: 목적 이미지에서 소스 이미지가 삽입될 y좌표

동작:&#x20;

* 이 함수는 소스 이미지를 목적 이미지에 삽입하는 기능을 수행합니다.&#x20;
* 삽입될 위치는 목적 이미지에서 지정된 dx, dy 위치입니다.&#x20;
* 소스 이미지의 모든 픽셀을 돌면서 목적 이미지에 값을 복사하여 삽입합니다.

설명:&#x20;

* 이 함수는 이미지를 다루는 작업을 수행하는 함수입니다.&#x20;
* 소스 이미지와 목적 이미지는 이미지 타입(image)으로 입력되며, dx와 dy는 삽입될 위치를 나타내는 정수형(int) 변수입니다.&#x20;
* 소스 이미지와 목적 이미지의 가로, 세로, 채널 정보는 이미지 타입(image)으로부터 추출됩니다.&#x20;
* 소스 이미지에서는 모든 픽셀을 돌면서 채널별로 목적 이미지에 값을 복사하여 삽입합니다.&#x20;
* 이 함수를 사용하면 이미지의 일부분을 잘라내어 다른 이미지에 삽입하는 기능을 구현할 수 있습니다.



## collapse\_image\_layers

```c
image collapse_image_layers(image source, int border)
{
    int h = source.h;
    h = (h+border)*source.c - border;
    image dest = make_image(source.w, h, 1);
    int i;
    for(i = 0; i < source.c; ++i){
        image layer = get_image_layer(source, i);
        int h_offset = i*(source.h+border);
        embed_image(layer, dest, 0, h_offset);
        free_image(layer);
    }
    return dest;
}
```

함수 이름: collapse\_image\_layers

입력:

* source: 이미지 타입(image)의 소스 이미지
* border: 이미지 층(layer) 간에 추가할 픽셀 경계(border) 크기

동작:&#x20;

* 이 함수는 여러 개의 이미지 층(layer)을 하나의 이미지로 병합하는 기능을 수행합니다.&#x20;
* 이를 위해 각 층의 이미지를 불러와서 목적 이미지에 삽입합니다.&#x20;
* 각 층과 층 사이에는 border 크기만큼 간격을 둡니다.&#x20;
* 최종적으로 병합된 이미지를 반환합니다.

설명:&#x20;

* 이 함수는 이미지 층(layer)을 다루는 작업을 수행하는 함수입니다.&#x20;
* 소스 이미지와 border는 각각 이미지 타입(image)과 정수형(int)으로 입력됩니다.&#x20;
* 함수는 소스 이미지의 높이와 border 크기를 이용하여 목적 이미지(dest)의 높이를 계산하고, 이를 이용하여 목적 이미지를 생성합니다.&#x20;
* 목적 이미지의 가로 길이는 소스 이미지와 동일하고, 높이는 각 층의 높이와 border 크기를 이용하여 계산합니다.&#x20;
* 소스 이미지에는 여러 개의 이미지 층(layer)이 포함되어 있으며, 각 층을 하나씩 불러와서 목적 이미지(dest)에 삽입합니다.&#x20;
* 층과 층 사이에는 border 크기만큼 간격을 두고 삽입합니다. 이미지를 병합하고 난 후에는 free\_image() 함수를 이용하여 각 층의 메모리를 해제합니다.
* 이 함수를 사용하면 여러 개의 이미지를 하나의 이미지로 병합하여 사용할 수 있습니다.&#x20;
* 이미지 층(layer)을 이용하여 딥러닝 모델을 구현할 때, 이미지의 전처리 과정에서 여러 개의 이미지를 병합하여 하나의 입력으로 만들 수 있습니다.



## constrain\_image

```c
void constrain_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h*im.c; ++i){
        if(im.data[i] < 0) im.data[i] = 0;
        if(im.data[i] > 1) im.data[i] = 1;
    }
}
```

함수 이름: constrain\_image

입력:

* im: 이미지 타입(image)의 입력 이미지

동작:&#x20;

* 이 함수는 이미지의 각 픽셀 값이 0과 1 사이의 값을 가지도록 제한(constrain)합니다.&#x20;
* 입력 이미지의 모든 픽셀 값을 확인하면서 0보다 작으면 0으로, 1보다 크면 1로 값을 제한합니다.

설명:&#x20;

* 이 함수는 이미지 처리를 위한 함수 중 하나입니다.&#x20;
* 이미지 타입(image)의 입력 이미지(im)를 받아서 각 픽셀의 값을 확인하면서 0보다 작거나 1보다 크면 값을 0 또는 1로 조정합니다.&#x20;
* 이를 통해 이미지의 픽셀 값이 0과 1 사이의 값을 가지도록 제한됩니다.
* 이 함수는 이미지 처리에서 사용될 때, 이미지의 픽셀 값 범위를 0과 1 사이의 값으로 제한하여 다른 연산을 수행할 때의 안정성을 보장할 수 있습니다.&#x20;
* 또한, 이미지 처리의 결과물을 출력할 때, 이미지의 각 픽셀 값이 0과 1 사이의 값을 가지도록 조정하여 정확한 시각화를 보장할 수 있습니다.



## normalize\_image

```c
void normalize_image(image p)
{
    int i;
    float min = 9999999;
    float max = -999999;

    for(i = 0; i < p.h*p.w*p.c; ++i){
        float v = p.data[i];
        if(v < min) min = v;
        if(v > max) max = v;
    }
    if(max - min < .000000001){
        min = 0;
        max = 1;
    }
    for(i = 0; i < p.c*p.w*p.h; ++i){
        p.data[i] = (p.data[i] - min)/(max-min);
    }
}
```

함수 이름: normalize\_image

입력:

* p: 이미지 타입(image)의 입력 이미지

동작:&#x20;

* 이 함수는 입력 이미지의 각 픽셀 값을 0과 1 사이의 값으로 정규화(normalize)합니다.&#x20;
* 입력 이미지의 모든 픽셀 값을 확인하면서 최소(min)값과 최대(max)값을 찾은 후, 모든 픽셀 값을 최소와 최대값의 범위에 맞게 조정합니다.

설명:&#x20;

* 이 함수는 이미지 처리를 위한 함수 중 하나입니다. 이미지 타입(image)의 입력 이미지(p)를 받아서 각 픽셀 값을 확인하면서 최소값과 최대값을 찾습니다.&#x20;
* 그리고 모든 픽셀 값을 최소와 최대값의 범위에 맞게 조정하여 0과 1 사이의 값을 가지도록 정규화합니다.
* 이 함수는 이미지 처리에서 사용될 때, 이미지의 픽셀 값 범위를 0과 1 사이의 값으로 조정하여 다른 연산을 수행할 때의 안정성을 보장할 수 있습니다.&#x20;
* 또한, 이미지 처리의 결과물을 출력할 때, 이미지의 각 픽셀 값이 0과 1 사이의 값을 가지도록 조정하여 정확한 시각화를 보장할 수 있습니다.



## normalize\_image2

```c
void normalize_image2(image p)
{
    float *min = calloc(p.c, sizeof(float));
    float *max = calloc(p.c, sizeof(float));
    int i,j;
    for(i = 0; i < p.c; ++i) min[i] = max[i] = p.data[i*p.h*p.w];

    for(j = 0; j < p.c; ++j){
        for(i = 0; i < p.h*p.w; ++i){
            float v = p.data[i+j*p.h*p.w];
            if(v < min[j]) min[j] = v;
            if(v > max[j]) max[j] = v;
        }
    }
    for(i = 0; i < p.c; ++i){
        if(max[i] - min[i] < .000000001){
            min[i] = 0;
            max[i] = 1;
        }
    }
    for(j = 0; j < p.c; ++j){
        for(i = 0; i < p.w*p.h; ++i){
            p.data[i+j*p.h*p.w] = (p.data[i+j*p.h*p.w] - min[j])/(max[j]-min[j]);
        }
    }
    free(min);
    free(max);
}
```

함수 이름: normalize\_image2

입력:&#x20;

* image p (이미지 구조체)

동작:&#x20;

* 이미지 p의 모든 채널에 대해 최솟값과 최댓값을 찾아 정규화를 수행합니다.&#x20;
* 먼저 최솟값과 최댓값을 찾기 위해 각 채널마다 p의 데이터를 순회하면서 최솟값과 최댓값을 찾습니다.&#x20;
* 만약 최댓값과 최솟값이 같으면, 최솟값을 0으로, 최댓값을 1로 설정합니다.&#x20;
* 그렇지 않으면, 최솟값을 0으로, 최댓값을 1로 정규화합니다.

설명:&#x20;

* normalize\_image2 함수는 이미지를 정규화하는 함수입니다.&#x20;
* 이미지를 정규화하는 이유는 이미지의 픽셀값이 서로 다른 범위를 가지고 있을 때, 학습 성능이 저하될 수 있기 때문입니다.&#x20;
* 이 함수는 이미지의 모든 채널에 대해 정규화를 수행하므로, 각 채널의 픽셀값 범위를 동일하게 만들어줍니다.
* 이 함수는 이미지 p의 모든 채널에 대해 정규화를 수행합니다.&#x20;
* 먼저 최솟값과 최댓값을 찾기 위해 각 채널마다 p의 데이터를 순회하면서 최솟값과 최댓값을 찾습니다.&#x20;
* 그리고 최솟값과 최댓값의 차이가 0.000000001보다 작으면 최솟값을 0으로, 최댓값을 1로 설정합니다.&#x20;
* 그렇지 않으면, 최솟값을 0으로, 최댓값을 1로 정규화합니다.
* 최종적으로 normalize\_image2 함수는 이미지 p의 모든 채널에 대해 정규화된 이미지를 반환합니다.



## copy\_image\_into

```c
void copy_image_into(image src, image dest)
{
    memcpy(dest.data, src.data, src.h*src.w*src.c*sizeof(float));
}
```

함수 이름: copy\_image\_into&#x20;

입력:&#x20;

* (image) src - 복사하려는 이미지
* (image) dest - 복사 대상 이미지&#x20;

동작:&#x20;

* src 이미지의 데이터를 dest 이미지에 복사&#x20;

설명:&#x20;

* src 이미지의 데이터를 dest 이미지로 복사하여 dest 이미지를 src 이미지와 동일하게 만드는 함수입니다.&#x20;
* src와 dest는 모두 image 구조체입니다. memcpy 함수를 사용하여 src 이미지의 데이터를 dest 이미지로 복사합니다.



## copy\_image

```c
image copy_image(image p)
{
    image copy = p;
    copy.data = calloc(p.h*p.w*p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));
    return copy;
}
```

함수 이름: copy\_image

입력:&#x20;

* image p (복사하고자 하는 이미지)

동작:&#x20;

* 입력으로 들어온 이미지 p를 복사하여 새로운 이미지 객체를 생성한다.&#x20;
* 그리고 새로 생성한 이미지 객체의 data 포인터가 가리키는 곳에 p.data 포인터가 가리키는 데이터를 복사하여 저장한다.

설명:

* 함수는 입력으로 들어온 이미지 p를 복사한 새로운 이미지 객체를 생성하여 반환한다.
* 이미지 복사를 위해선 새로운 이미지 객체를 생성하고, 데이터를 복사해야 한다. 따라서 copy\_image 함수는 먼저 입력 이미지 객체 p와 동일한 속성을 가지는 새로운 이미지 객체 copy를 생성한다. (여기서 copy는 p와 동일한 w, h, c를 가지지만, data는 새로운 메모리 공간을 가리키게 된다.)
* 그리고 copy 객체의 data 메모리 공간에 p 객체의 data 메모리 공간에 있는 데이터를 복사한다. 이를 위해 memcpy 함수를 사용한다.
* 이후에는 복사한 이미지 객체 copy를 반환한다.



## rgbgr\_image

```c
void rgbgr_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}
```

함수 이름: rgbgr\_image

입력:&#x20;

* image im (이미지 데이터를 담고 있는 구조체)

동작:&#x20;

* 주어진 이미지의 RGB 채널 값을 BGR 채널 값으로 바꾼다.

설명:&#x20;

* 입력으로 주어진 이미지의 각 픽셀은 float 형식으로 구성되어 있다.&#x20;
* 이 함수는 이러한 이미지에서 RGB 채널 값을 BGR 채널 값으로 바꾼다.&#x20;
* 구체적으로는, 주어진 이미지의 너비와 높이를 곱한 값만큼 반복하면서, 각 픽셀의 RGB 채널 값을 BGR 채널 값으로 교체한다.&#x20;
* 이때, R과 B 채널의 위치를 바꾸면 된다. 이 함수는 입력 이미지를 직접 변경하며 반환 값은 없다.



## show\_image

```c
int show_image(image p, const char *name, int ms)
{
#ifdef OPENCV
    int c = show_image_cv(p, name, ms);
    return c;
#else
    fprintf(stderr, "Not compiled with OpenCV, saving to %s.png instead\n", name);
    save_image(p, name);
    return -1;
#endif
}
```

함수 이름: show\_image&#x20;

입력:

* image p: 출력할 이미지
* const char \*name: 윈도우 창 이름 또는 이미지 파일 이름
* int ms: 이미지가 윈도우 창에 출력되는 시간(밀리초)

동작:

* OPENCV 매크로가 정의되어 있다면, OpenCV를 사용하여 이미지를 윈도우 창에 출력하고 키보드 입력을 반환한다.
* OPENCV 매크로가 정의되어 있지 않으면, 이미지를 PNG 파일로 저장한다.

설명:&#x20;

* 이미지를 윈도우 창에 출력하거나 파일로 저장하는 함수이다.&#x20;
* OPENCV 매크로가 정의되어 있다면 OpenCV를 사용하여 이미지를 출력하고 키보드 입력을 반환한다.&#x20;
* OPENCV 매크로가 정의되어 있지 않으면 이미지를 PNG 파일로 저장하며, 출력된 파일 이름은 name 매개 변수에 지정된 이름에 .png 확장자가 추가된 것이다.&#x20;
* 반환값은 OPENCV를 사용할 때 윈도우 창에서 사용자 입력을 받은 키보드 코드이다.&#x20;
* OPENCV가 사용되지 않은 경우 -1을 반환한다.



## save\_image\_options

```c
void save_image_options(image im, const char *name, IMTYPE f, int quality)
{
    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    if(f == PNG)       sprintf(buff, "%s.png", name);
    else if (f == BMP) sprintf(buff, "%s.bmp", name);
    else if (f == TGA) sprintf(buff, "%s.tga", name);
    else if (f == JPG) sprintf(buff, "%s.jpg", name);
    else               sprintf(buff, "%s.png", name);
    unsigned char *data = calloc(im.w*im.h*im.c, sizeof(char));
    int i,k;
    for(k = 0; k < im.c; ++k){
        for(i = 0; i < im.w*im.h; ++i){
            data[i*im.c+k] = (unsigned char) (255*im.data[i + k*im.w*im.h]);
        }
    }
    int success = 0;
    if(f == PNG)       success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
    else if (f == BMP) success = stbi_write_bmp(buff, im.w, im.h, im.c, data);
    else if (f == TGA) success = stbi_write_tga(buff, im.w, im.h, im.c, data);
    else if (f == JPG) success = stbi_write_jpg(buff, im.w, im.h, im.c, data, quality);
    free(data);
    if(!success) fprintf(stderr, "Failed to write image %s\n", buff);
}
```

함수 이름: save\_image\_options

입력:&#x20;

* image im (이미지 데이터)
* const char \*name (파일 이름)
* IMTYPE f (저장할 이미지 파일 형식)
* int quality (JPEG 파일 형식으로 저장할 때 이미지 품질 설정)

동작:&#x20;

* 입력된 이미지 데이터를 지정된 파일 형식으로 저장하는 함수입니다.&#x20;
* 입력된 파일 형식에 따라 파일 확장자를 결정하고, 입력된 이미지 데이터를 파일 형식에 맞게 인코딩하여 파일로 저장합니다.

설명:&#x20;

* 함수 내부에서는 입력된 이미지 데이터를 unsigned char 타입으로 변환한 후, 지정된 파일 형식에 따라 인코딩하여 파일로 저장합니다.&#x20;
* 이미지 파일 형식은 PNG, BMP, TGA, JPG 형식을 지원하며, 기본적으로 PNG 형식으로 저장합니다.&#x20;
* 만약 OpenCV가 설치되어 있다면, OpenCV를 사용하여 이미지를 화면에 출력할 수도 있습니다.&#x20;
* 저장된 파일 이름은 입력된 파일 이름에 파일 확장자가 추가된 형태로 저장됩니다. 파일 저장에 실패하면, 에러 메시지가 출력됩니다.



## save\_image

```c
void save_image(image im, const char *name)
{
    save_image_options(im, name, JPG, 80);
}
```

함수 이름: save\_image

입력:&#x20;

* image im (저장할 이미지)
* const char \*name (저장할 파일 이름)

동작:&#x20;

* 입력된 이미지를 JPG 형식으로 지정된 파일 이름으로 저장한다.&#x20;
* 내부적으로는 save\_image\_options 함수를 호출하며, 이미지를 unsigned char 배열로 변환하여 파일에 쓴다.

설명:&#x20;

* 입력된 이미지를 지정된 파일 이름으로 저장하는 함수이다.&#x20;
* 이미지 파일의 형식은 기본적으로 JPG 형식이며, 저장할 때 압축률은 80으로 지정된다.&#x20;
* 내부적으로는 save\_image\_options 함수를 호출하여 이미지 파일을 저장하며, 파일의 형식을 변경하려면 save\_image\_options 함수를 직접 호출하여야 한다.



## show\_image\_layers

```c
void show_image_layers(image p, char *name)
{
    int i;
    char buff[256];
    for(i = 0; i < p.c; ++i){
        sprintf(buff, "%s - Layer %d", name, i);
        image layer = get_image_layer(p, i);
        show_image(layer, buff, 1);
        free_image(layer);
    }
}
```

함수 이름: show\_image\_layers&#x20;

입력:&#x20;

* image p (입력 이미지)
* char \*name (창 제목)

동작:&#x20;

* 입력 이미지의 각 레이어를 분리하여 창에 따로 표시&#x20;

설명:&#x20;

* 입력 이미지의 각 레이어를 분리하여 각각 따로 창으로 표시하는 함수입니다.&#x20;
* 레이어 별로 창 제목은 입력으로 받은 name 뒤에 " - Layer i" (i는 레이어 인덱스)를 붙여 구성합니다.&#x20;
* 이 때, 각 레이어는 get\_image\_layer 함수를 사용하여 추출하고, 해당 레이어를 표시한 뒤, 메모리를 해제합니다.



## show\_image\_collapsed

```c
void show_image_collapsed(image p, char *name)
{
    image c = collapse_image_layers(p, 1);
    show_image(c, name, 1);
    free_image(c);
}
```

함수 이름: show\_image\_collapsed

입력:&#x20;

* image p (출력할 이미지)
* char \*name (윈도우 창에 표시할 이미지의 이름)

동작:&#x20;

* 입력 이미지의 모든 레이어를 하나의 이미지로 축소한 다음, 이를 윈도우 창에 표시함

설명:&#x20;

* 입력으로 주어진 이미지 p의 모든 레이어를 하나의 이미지로 축소한 다음, 이를 윈도우 창에 표시하는 함수입니다.&#x20;
* 축소된 이미지는 collapse\_image\_layers 함수를 사용하여 생성하며, 윈도우 창에 표시하는 부분은 show\_image 함수를 사용합니다.&#x20;
* 최종적으로 생성된 축소된 이미지는 free\_image 함수를 사용하여 메모리에서 해제됩니다.



## make\_empty\_image

```c
image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}
```

함수 이름: make\_empty\_image

입력:&#x20;

* int w (이미지의 너비)
* int h (이미지의 높이)
* int c (이미지의 채널 수)

동작:&#x20;

* 너비, 높이, 채널 수를 지정하여 비어있는(image.data = 0) 이미지를 생성하고 반환합니다.

설명:&#x20;

* 이 함수는 입력받은 이미지의 너비, 높이, 채널 수를 기반으로 새로운 비어있는 이미지를 생성하고 반환합니다.&#x20;
* 이 함수는 이미지 데이터를 초기화하지 않습니다.&#x20;
* 따라서 새 이미지의 모든 픽셀 값은 0입니다.



## make\_image

```c
image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = calloc(h*w*c, sizeof(float));
    return out;
}
```

함수 이름: make\_image

입력:

* int w: 이미지의 가로 크기
* int h: 이미지의 세로 크기
* int c: 이미지의 채널 수

동작:&#x20;

* 입력받은 가로, 세로, 채널 수를 가지는 빈 이미지를 생성하고, 해당 이미지의 데이터를 저장할 메모리 공간을 할당한다.

설명:&#x20;

* make\_empty\_image 함수를 호출하여 가로, 세로, 채널 수를 가지는 빈 이미지를 생성하고, 해당 이미지의 데이터를 저장할 메모리 공간을 calloc 함수를 사용하여 할당한다.&#x20;
* 할당된 메모리는 0으로 초기화된다. 최종적으로 생성된 이미지를 반환한다.



## make\_random\_image

```c
image make_random_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = calloc(h*w*c, sizeof(float));
    int i;
    for(i = 0; i < w*h*c; ++i){
        out.data[i] = (rand_normal() * .25) + .5;
    }
    return out;
}
```

함수 이름: make\_random\_image

입력:

* w: 생성될 이미지의 가로 크기
* h: 생성될 이미지의 세로 크기
* c: 생성될 이미지의 채널 수

동작:&#x20;

* 가로, 세로, 채널 수를 입력 받아 난수로 구성된 새로운 이미지를 생성한다.

설명:&#x20;

* 새로운 이미지를 만들기 위해 입력된 크기(w, h, c)로 make\_empty\_image() 함수를 호출하고, 그 결과로 반환된 이미지(out)의 데이터를 난수로 초기화한다.&#x20;
* 여기서 사용된 rand\_normal() 함수는 평균이 0이고 표준편차가 1인 정규분포를 따르는 난수를 생성하는 함수이다. 반환된 out 이미지는 생성된 새로운 이미지이다.



## float\_to\_image

```c
image float_to_image(int w, int h, int c, float *data)
{
    image out = make_empty_image(w,h,c);
    out.data = data;
    return out;
}
```

함수 이름: float\_to\_image

입력:&#x20;

* (int) w: 이미지의 너비
* (int) h: 이미지의 높이
* (int) c: 이미지의 채널 수
* (float \*) data: 이미지 데이터가 들어있는 포인터

동작:&#x20;

* 주어진 이미지 데이터 포인터를 사용하여 w, h, c 크기의 새로운 이미지를 만듭니다.

설명:&#x20;

* 이 함수는 주어진 이미지 데이터 포인터를 사용하여 w, h, c 크기의 새로운 이미지를 만들고 반환합니다.&#x20;
* 반환 된 이미지는 주어진 데이터를 참조하므로, 반환 된 이미지를 수정하면 입력 데이터도 수정됩니다.



## place\_image

```c
void place_image(image im, int w, int h, int dx, int dy, image canvas)
{
    int x, y, c;
    for(c = 0; c < im.c; ++c){
        for(y = 0; y < h; ++y){
            for(x = 0; x < w; ++x){
                float rx = ((float)x / w) * im.w;
                float ry = ((float)y / h) * im.h;
                float val = bilinear_interpolate(im, rx, ry, c);
                set_pixel(canvas, x + dx, y + dy, c, val);
            }
        }
    }
}
```

함수 이름: place\_image

입력:

* image im: 복사할 이미지
* int w: 복사할 이미지의 가로 크기
* int h: 복사할 이미지의 세로 크기
* int dx: 캔버스 내에서 이미지가 배치될 x좌표
* int dy: 캔버스 내에서 이미지가 배치될 y좌표
* image canvas: 이미지가 배치될 캔버스

동작:&#x20;

* 주어진 이미지 im을 가로 크기 w, 세로 크기 h로 조정한 후, 캔버스 canvas 내에서 좌표 (dx, dy)에 위치시킵니다.&#x20;
* bilinear\_interpolate 함수를 사용하여 이미지의 픽셀 값을 가져와 캔버스에 복사합니다.

설명:&#x20;

* 이미지를 다른 이미지나 캔버스 위에 배치할 때 사용합니다.&#x20;
* 이미지를 가로, 세로 크기에 맞게 조정한 후, 지정된 좌표에 위치시킵니다.&#x20;
* bilinear\_interpolate 함수를 사용하여 이미지의 값을 가져와 캔버스에 복사합니다.



## center\_crop\_image

```c
image center_crop_image(image im, int w, int h)
{
    int m = (im.w < im.h) ? im.w : im.h;   
    image c = crop_image(im, (im.w - m) / 2, (im.h - m)/2, m, m);
    image r = resize_image(c, w, h);
    free_image(c);
    return r;
}
```

함수 이름: center\_crop\_image

입력:

* im: 자를 이미지
* w: 출력할 이미지의 너비
* h: 출력할 이미지의 높이

동작:

* 입력된 이미지를 중앙에서부터 가장 작은 차원을 기준으로 정사각형으로 자른다.
* 자른 이미지를 출력할 너비와 높이로 리사이즈한다.

설명:&#x20;

* 입력된 이미지를 중앙에서부터 가장 작은 차원을 기준으로 정사각형으로 자르고, 그 결과 이미지를 주어진 출력 크기로 리사이즈하여 반환하는 함수이다. 이 함수는 이미지 분류 등에서 자주 사용된다.



## rotate\_crop\_image

```c
image rotate_crop_image(image im, float rad, float s, int w, int h, float dx, float dy, float aspect)
{
    int x, y, c;
    float cx = im.w/2.;
    float cy = im.h/2.;
    image rot = make_image(w, h, im.c);
    for(c = 0; c < im.c; ++c){
        for(y = 0; y < h; ++y){
            for(x = 0; x < w; ++x){
                float rx = cos(rad)*((x - w/2.)/s*aspect + dx/s*aspect) - sin(rad)*((y - h/2.)/s + dy/s) + cx;
                float ry = sin(rad)*((x - w/2.)/s*aspect + dx/s*aspect) + cos(rad)*((y - h/2.)/s + dy/s) + cy;
                float val = bilinear_interpolate(im, rx, ry, c);
                set_pixel(rot, x, y, c, val);
            }
        }
    }
    return rot;
}
```

함수 이름: rotate\_crop\_image

입력:

* image im : 회전 및 크롭할 원본 이미지
* float rad : 회전할 각도 (라디안)
* float s : 크기 조절 비율
* int w : 출력 이미지의 가로 크기
* int h : 출력 이미지의 세로 크기
* float dx : x축 방향 이동량
* float dy : y축 방향 이동량
* float aspect : 가로 세로 비율

동작:&#x20;

* 입력으로 받은 원본 이미지를 주어진 각도로 회전하고, 주어진 크기 비율로 조절하며, 주어진 위치로 이동시킨 후, 주어진 가로 세로 비율에 맞게 크롭하여 출력 이미지를 생성한다.

설명:&#x20;

* 이미지 처리에서 회전, 크기 조절, 이동, 크롭은 매우 기본적인 작업 중 하나이다. 이 함수는 이러한 작업을 수행하는 함수 중 하나로, 입력으로 받은 이미지를 주어진 각도와 비율, 위치에 따라 회전, 크기 조절, 이동을 수행한 후, 주어진 가로 세로 비율에 맞게 크롭하여 출력 이미지를 생성한다. 회전 및 크롭에 필요한 보간(interpolation)은 bilinear\_interpolate 함수를 사용한다.



## rotate\_image

```c
image rotate_image(image im, float rad)
{
    int x, y, c;
    float cx = im.w/2.;
    float cy = im.h/2.;
    image rot = make_image(im.w, im.h, im.c);
    for(c = 0; c < im.c; ++c){
        for(y = 0; y < im.h; ++y){
            for(x = 0; x < im.w; ++x){
                float rx = cos(rad)*(x-cx) - sin(rad)*(y-cy) + cx;
                float ry = sin(rad)*(x-cx) + cos(rad)*(y-cy) + cy;
                float val = bilinear_interpolate(im, rx, ry, c);
                set_pixel(rot, x, y, c, val);
            }
        }
    }
    return rot;
}
```

함수 이름: rotate\_image

입력:&#x20;

* image im (입력 이미지)
* float rad (회전 각도)

동작:&#x20;

* 입력 이미지를 주어진 각도(rad)만큼 회전시킨 이미지를 생성합니다.&#x20;
* 회전된 이미지의 크기는 입력 이미지와 동일합니다.

설명:&#x20;

* 입력 이미지의 각 픽셀 위치를 회전 변환하여 회전된 이미지에서 해당 위치의 값을 bilinear\_interpolate 함수를 사용하여 보간합니다.&#x20;
* 회전 변환된 위치에서 보간된 값은 회전된 이미지의 해당 위치의 값으로 설정됩니다.&#x20;
* 회전된 이미지를 반환합니다.



## fill\_image

```c
void fill_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] = s;
}
```

함수 이름: fill\_image&#x20;

입력:&#x20;

* image m (이미지 구조체 포인터)
* float s (초기화 값)&#x20;

동작:&#x20;

* 이미지의 모든 픽셀 값을 초기화 값 s로 설정함&#x20;

설명:&#x20;

* 입력으로 받은 이미지 구조체 포인터 m이 가리키는 이미지의 모든 픽셀 값을 초기화 값 s로 설정합니다.&#x20;
* 초기화 값 s는 float형으로 입력 받으며, 이미지의 채널 수(c), 높이(h), 너비(w)에 따라 이미지 전체의 크기(m.h_m.w_m.c)만큼 반복하여 각 픽셀을 초기화합니다.&#x20;
* 따라서 입력 이미지의 모든 픽셀 값을 동일한 값으로 채우고자 할 때 이 함수를 사용할 수 있습니다.



## translate\_image

```c
void translate_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] += s;
}
```

함수 이름: translate\_image

입력:

* &#x20;image m (이미지)
* float s (이동량)

동작:&#x20;

* 입력으로 받은 이미지 m의 모든 픽셀값을 s만큼 이동시킵니다.

설명:&#x20;

* 입력으로 받은 이미지의 너비, 높이, 채널 수를 모두 곱한 크기만큼 모든 픽셀값을 s만큼 증가시키는 함수입니다.&#x20;
* 이를 통해 이미지를 s만큼 오른쪽 또는 아래쪽으로 이동시킬 수 있습니다.



## scale\_image

```c
void scale_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] *= s;
}
```

함수 이름: scale\_image&#x20;

입력:&#x20;

* image m (이미지 구조체 포인터)
* float s (스케일링 비율)&#x20;

동작:&#x20;

* 입력된 이미지의 픽셀값들을 스케일링 비율에 맞게 조정&#x20;

설명:&#x20;

* 입력된 이미지의 각 픽셀값들에 대해 스케일링 비율을 곱하여 이미지 전체를 조정함.&#x20;
* 스케일링 비율이 1보다 작으면 이미지가 축소되고, 1보다 크면 이미지가 확대됨.



## crop\_image

```c
image crop_image(image im, int dx, int dy, int w, int h)
{
    image cropped = make_image(w, h, im.c);
    int i, j, k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int r = j + dy;
                int c = i + dx;
                float val = 0;
                r = constrain_int(r, 0, im.h-1);
                c = constrain_int(c, 0, im.w-1);
                val = get_pixel(im, c, r, k);
                set_pixel(cropped, i, j, k, val);
            }
        }
    }
    return cropped;
}
```

함수 이름: crop\_image&#x20;

입력:

* image im: 크롭 대상 이미지
* int dx: x축으로 이동할 픽셀 수
* int dy: y축으로 이동할 픽셀 수
* int w: 출력할 이미지의 너비
* int h: 출력할 이미지의 높이

동작:&#x20;

* 입력으로 받은 이미지에서 dx와 dy만큼 이동한 후, w와 h 크기만큼의 영역을 자른 이미지를 생성하여 반환합니다.&#x20;
* 이미지의 채널 수만큼 반복문을 돌면서, 출력할 이미지에 픽셀 값을 복사합니다.

설명:&#x20;

* 입력으로 받은 이미지의 특정 영역을 잘라내는 기능을 수행하는 함수입니다.&#x20;
* 크롭을 위해 x와 y 축으로 각각 이동할 픽셀 수와 출력할 이미지의 너비와 높이를 인자로 받습니다.&#x20;
* 함수 내부에서는 출력할 이미지 크기만큼 메모리를 할당하고, 입력 이미지에서 영역을 잘라내어 복사합니다.&#x20;
* 픽셀 좌표는 dx와 dy만큼 이동한 후, 입력 이미지의 범위를 벗어나지 않도록 제한합니다.



## best\_3d\_shift\_r

```c
int best_3d_shift_r(image a, image b, int min, int max)
{
    if(min == max) return min;
    int mid = floor((min + max) / 2.);
    image c1 = crop_image(b, 0, mid, b.w, b.h);
    image c2 = crop_image(b, 0, mid+1, b.w, b.h);
    float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 10);
    float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 10);
    free_image(c1);
    free_image(c2);
    if(d1 < d2) return best_3d_shift_r(a, b, min, mid);
    else return best_3d_shift_r(a, b, mid+1, max);
}
```

함수 이름: best\_3d\_shift\_r

입력:

* image a: 비교할 이미지 데이터
* image b: 이미지 데이터
* int min: 이동 거리의 최소값
* int max: 이동 거리의 최대값

동작:

* 이진 검색을 사용하여 이미지 b의 수직 이동 거리를 조정하여 이미지 a와 가장 일치하는 이동 거리를 찾음
* 이미지 b를 두 개의 이미지로 자르고, 자른 이미지들과 이미지 a와의 거리를 계산하여 최적의 이동 거리를 찾음

설명:

* 이미지 a와 b는 같은 크기와 채널 수를 가지고 있어야 함
* 이 함수는 재귀적으로 호출되며, 이동 거리의 최소값과 최대값을 사용하여 이미지 b를 자르고 거리를 계산하여 최적의 이동 거리를 찾음
* 이미지 b를 수직으로 자른 두 개의 이미지(c1, c2)를 생성하여 이미지 a와의 거리(d1, d2)를 계산하고, 이동 거리를 반으로 나누어 재귀적으로 호출하여 최적의 이동 거리를 찾음
* 재귀 호출을 할 때 거리(d1, d2)가 작은 쪽을 선택하여 이동 거리를 좁혀나가며 최적의 이동 거리를 찾음



## best\_3d\_shift

```c
int best_3d_shift(image a, image b, int min, int max)
{
    int i;
    int best = 0;
    float best_distance = FLT_MAX;
    for(i = min; i <= max; i += 2){
        image c = crop_image(b, 0, i, b.w, b.h);
        float d = dist_array(c.data, a.data, a.w*a.h*a.c, 100);
        if(d < best_distance){
            best_distance = d;
            best = i;
        }
        printf("%d %f\n", i, d);
        free_image(c);
    }
    return best;
}
```

함수 이름: best\_3d\_shift

입력:&#x20;

* image a (비교할 이미지)
* image b (이동시킬 대상 이미지)
* int min (이동 범위 최소값)
* int max (이동 범위 최대값)

동작:&#x20;

* 이미지 b를 min부터 max까지 2씩 증가시키며 이동시킨 후, 이미지 a와의 거리를 측정하여 가장 작은 거리를 가지는 이동값을 반환한다.

설명:&#x20;

* best\_3d\_shift 함수는 이미지 a와 b를 비교하여 최적의 이동 값을 찾아내는 함수이다.&#x20;
* 이미지 b를 min부터 max까지 2씩 증가시키며 이동시킨 후, 이동된 이미지와 이미지 a와의 거리를 측정하여 가장 작은 거리를 가지는 이동값을 반환한다.&#x20;
* 이미지 이동은 crop\_image 함수를 이용하여 구현하였으며, 거리 측정은 dist\_array 함수를 이용하여 구현하였다.&#x20;
* 함수 실행 결과를 printf 함수를 이용하여 출력한다.



## composite\_3d

```c
void composite_3d(char *f1, char *f2, char *out, int delta)
{
    if(!out) out = "out";
    image a = load_image(f1, 0,0,0);
    image b = load_image(f2, 0,0,0);
    int shift = best_3d_shift_r(a, b, -a.h/100, a.h/100);

    image c1 = crop_image(b, 10, shift, b.w, b.h);
    float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 100);
    image c2 = crop_image(b, -10, shift, b.w, b.h);
    float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 100);

    if(d2 < d1 && 0){
        image swap = a;
        a = b;
        b = swap;
        shift = -shift;
        printf("swapped, %d\n", shift);
    }
    else{
        printf("%d\n", shift);
    }

    image c = crop_image(b, delta, shift, a.w, a.h);
    int i;
    for(i = 0; i < c.w*c.h; ++i){
        c.data[i] = a.data[i];
    }
    save_image(c, out);
}
```

함수 이름: composite\_3d&#x20;

입력:

* char \*f1: 합성할 첫 번째 이미지 파일 경로
* char \*f2: 합성할 두 번째 이미지 파일 경로
* char \*out: 결과 이미지를 저장할 파일 경로 (default: "out")
* int delta: 이미지 합성 시 가로 방향으로 이동할 픽셀 수

동작:

1. f1과 f2 이미지를 불러옴
2. best\_3d\_shift\_r 함수를 이용하여 두 이미지를 수평으로 정렬
3. delta 값에 따라 f2 이미지를 가로 방향으로 이동시킴
4. f2 이미지와 f1 이미지를 합성하여 결과 이미지를 만들고 저장

설명:&#x20;

* composite\_3d 함수는 두 개의 이미지를 합성하여 결과 이미지를 만들어주는 함수입니다.&#x20;
* 합성할 두 이미지 파일의 경로와 결과 이미지를 저장할 파일 경로를 인자로 받으며, 결과 이미지 파일 경로는 입력하지 않으면 "out"으로 설정됩니다.&#x20;
* delta 값은 이미지 합성 시 f2 이미지를 가로 방향으로 이동시키는 픽셀 수를 나타냅니다.
* 먼저, f1과 f2 이미지를 불러온 다음 best\_3d\_shift\_r 함수를 이용하여 두 이미지를 수평으로 정렬합니다. 이후 delta 값에 따라 f2 이미지를 가로 방향으로 이동시키고, f2 이미지와 f1 이미지를 합성하여 결과 이미지를 만듭니다.
* 결과 이미지는 save\_image 함수를 이용하여 out 인자에 입력된 파일 경로에 저장됩니다.



## letterbox\_image\_into

```c
void letterbox_image_into(image im, int w, int h, image boxed)
{
    int new_w = im.w;
    int new_h = im.h;
    if (((float)w/im.w) < ((float)h/im.h)) {
        new_w = w;
        new_h = (im.h * w)/im.w;
    } else {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }
    image resized = resize_image(im, new_w, new_h);
    embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2);
    free_image(resized);
}

```

함수 이름: letterbox\_image\_into

입력:&#x20;

* (image im): 원본 이미지
* (int w): 박스의 가로 길이
* (int h): 박스의 세로 길이
* (image boxed): 박스 이미지

동작:&#x20;

* 입력된 원본 이미지를 주어진 박스 크기에 맞게 리사이징하고, 중앙에 위치시켜 박스 이미지에 삽입하는 함수

설명:

* 입력된 원본 이미지를 주어진 박스의 가로와 세로 길이 중 어느 쪽에 맞출 것인지를 판단
* 새로 조정된 이미지의 크기를 계산하고, 리사이징
* 리사이징된 이미지를 중앙에 위치시켜 박스 이미지에 삽입
* 사용된 이미지는 메모리에서 해제



## letterbox\_image

```c
image letterbox_image(image im, int w, int h)
{
    int new_w = im.w;
    int new_h = im.h;
    if (((float)w/im.w) < ((float)h/im.h)) {
        new_w = w;
        new_h = (im.h * w)/im.w;
    } else {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }
    image resized = resize_image(im, new_w, new_h);
    image boxed = make_image(w, h, im.c);
    fill_image(boxed, .5);
    //int i;
    //for(i = 0; i < boxed.w*boxed.h*boxed.c; ++i) boxed.data[i] = 0;
    embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2);
    free_image(resized);
    return boxed;
}
```

함수 이름: letterbox\_image&#x20;

입력:

* image im: letterbox 처리할 대상 이미지
* int w: 결과 이미지의 폭
* int h: 결과 이미지의 높이

동작:&#x20;

* 입력으로 주어진 이미지 im을 w x h 크기로 letterbox 처리한 이미지를 반환한다.&#x20;
* letterbox 처리란 원본 이미지를 새로운 크기의 이미지 안쪽에 더 큰 배경 이미지를 만들어 원본 이미지를 중앙에 위치시키는 것이다.

설명:

1. 입력으로 주어진 이미지 im의 가로, 세로 비율과 결과 이미지의 가로, 세로 비율을 비교하여, 새로운 이미지의 가로와 세로를 계산한다.
2. 입력 이미지를 새로운 가로와 세로로 resize\_image 함수를 이용해 크기를 조절한다.
3. 새로운 가로와 세로 크기를 가지는 새로운 이미지를 생성한다.
4. 새로운 이미지를 0.5로 초기화한다.
5. 원본 이미지를 중앙에 위치시킨 후, 결과 이미지를 반환한다.



## resize\_max

```c
image resize_max(image im, int max)
{
    int w = im.w;
    int h = im.h;
    if(w > h){
        h = (h * max) / w;
        w = max;
    } else {
        w = (w * max) / h;
        h = max;
    }
    if(w == im.w && h == im.h) return im;
    image resized = resize_image(im, w, h);
    return resized;
}
```

함수 이름: resize\_max&#x20;

입력:&#x20;

* image im (입력 이미지)
* int max (리사이징할 이미지의 최대 크기)&#x20;

동작:&#x20;

* 입력 이미지를 최대 크기(max)에 맞추어 리사이징한다.&#x20;
* 이미지의 가로와 세로 중 더 긴 쪽을 max에 맞게 조정하고, 비율에 맞게 다른 쪽 길이를 조정한다.&#x20;
* 조정한 크기로 이미지를 리사이징하고, 리사이징한 이미지를 반환한다.&#x20;
* 만약 이미지 크기가 이미 최대 크기(max)에 맞는 경우 입력 이미지를 그대로 반환한다.&#x20;

설명:&#x20;

* 입력 이미지를 최대 크기에 맞게 조정하여 리사이징하는 함수이다.&#x20;
* 이미지 크기를 비롯한 정보를 담은 image 타입을 입력으로 받고, 리사이징할 최대 크기를 int 타입으로 입력으로 받는다.



## resize\_min

```c
image resize_min(image im, int min)
{
    int w = im.w;
    int h = im.h;
    if(w < h){
        h = (h * min) / w;
        w = min;
    } else {
        w = (w * min) / h;
        h = min;
    }
    if(w == im.w && h == im.h) return im;
    image resized = resize_image(im, w, h);
    return resized;
}
```

함수 이름: resize\_min&#x20;

입력:&#x20;

* image im (변환할 이미지)
* int min (가로와 세로의 크기 중 작은 값을 설정)&#x20;

동작:&#x20;

* 입력 이미지의 가로와 세로 중 작은 값을 기준으로, min 값에 맞추어 이미지 크기를 조정하는 함수.&#x20;
* 이미지 비율은 유지됨.&#x20;

설명:&#x20;

* 입력 이미지의 가로와 세로 중 작은 값을 min 값으로 맞추어 이미지 크기를 조정한다.&#x20;
* 이미지 비율은 유지되며, 이미지가 min 값보다 작으면 원본 이미지를 반환한다.&#x20;
* 그렇지 않은 경우 입력 이미지를 min 값에 맞게 조정한 후 반환한다.



## random\_crop\_image

```c
image random_crop_image(image im, int w, int h)
{
    int dx = rand_int(0, im.w - w);
    int dy = rand_int(0, im.h - h);
    image crop = crop_image(im, dx, dy, w, h);
    return crop;
}
```

함수 이름: random\_crop\_image&#x20;

입력:&#x20;

* image im (자르기를 수행할 입력 이미지)
* int w (자를 이미지의 폭)
* int h (자를 이미지의 높이)&#x20;

동작:&#x20;

* 입력 이미지에서 임의로 선택한 위치에서 지정한 크기(w,h)로 이미지를 자른 후, 자른 이미지를 반환한다.&#x20;

설명:&#x20;

* 입력 이미지에서 임의의 위치에서 지정한 크기(w,h)로 이미지를 자른 후, 자른 이미지를 반환하는 함수이다.&#x20;
* 이때, 자르는 위치는 입력 이미지 내에서 랜덤으로 선택하며, 자를 이미지의 폭과 높이는 인자로 지정된 값으로 설정된다.



## random\_augment\_args

```c
augment_args random_augment_args(image im, float angle, float aspect, int low, int high, int w, int h)
{
    augment_args a = {0};
    aspect = rand_scale(aspect);
    int r = rand_int(low, high);
    int min = (im.h < im.w*aspect) ? im.h : im.w*aspect;
    float scale = (float)r / min;

    float rad = rand_uniform(-angle, angle) * TWO_PI / 360.;

    float dx = (im.w*scale/aspect - w) / 2.;
    float dy = (im.h*scale - w) / 2.;
    //if(dx < 0) dx = 0;
    //if(dy < 0) dy = 0;
    dx = rand_uniform(-dx, dx);
    dy = rand_uniform(-dy, dy);

    a.rad = rad;
    a.scale = scale;
    a.w = w;
    a.h = h;
    a.dx = dx;
    a.dy = dy;
    a.aspect = aspect;
    return a;
}
```

함수 이름: random\_augment\_args&#x20;

입력:

* im: 이미지 데이터를 담은 구조체
* angle: 회전 각도 (도 단위)
* aspect: 이미지 비율을 랜덤으로 조절하기 위한 비율
* low: 이미지 크기를 랜덤으로 결정하기 위한 최소 값
* high: 이미지 크기를 랜덤으로 결정하기 위한 최대 값
* w: 결과 이미지의 너비
* h: 결과 이미지의 높이&#x20;

동작:&#x20;

* 입력으로 받은 이미지를 회전, 크기 조절, 자르기 등 다양한 형태로 변환하여 새로운 이미지를 생성한다.&#x20;
* 랜덤으로 결정되는 회전 각도, 이미지 비율, 크기, 위치 등을 이용하여 다양한 변환을 수행한다.&#x20;
* 변환된 이미지의 크기는 w와 h로 지정된 값에 맞추어 자르거나 패딩하여 생성된다.&#x20;

설명:&#x20;

* 이미지 데이터를 변환하여 데이터 증강(augmentation)을 수행하는 함수로, 딥러닝 모델 학습에 사용될 수 있다.&#x20;
* 함수는 augmentation 인자를 랜덤으로 결정하여 입력 이미지를 변환하고, 그 결과로 새로운 이미지 데이터를 담은 구조체를 반환한다.



## random\_augment\_image

```c
image random_augment_image(image im, float angle, float aspect, int low, int high, int w, int h)
{
    augment_args a = random_augment_args(im, angle, aspect, low, high, w, h);
    image crop = rotate_crop_image(im, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);
    return crop;
}
```

함수 이름: random\_augment\_image&#x20;

입력:

* im: Augment를 적용할 이미지를 포함하는 image 구조체
* angle: Augment를 적용할 때 회전시킬 각도 범위
* aspect: Augment를 적용할 때 조정할 가로세로비 범위
* low, high: Augment를 적용할 때 임의의 크기 범위
* w, h: Augment를 적용할 때 임의의 이미지 크기 범위

동작:&#x20;

* 입력 이미지에 다양한 Augment를 적용하여 변환된 이미지를 반환하는 함수

설명:&#x20;

* 이 함수는 입력 이미지에 무작위로 다양한 Augment 기법을 적용하여 이미지를 변환한 후, 변환된 이미지를 반환합니다.&#x20;
* Augment를 적용하는 방법에는 회전, 크기 조절, 자르기 등이 있습니다.&#x20;
* 이 함수에서는 random\_augment\_args 함수를 사용하여 Augment에 필요한 인자들을 무작위로 설정한 후, rotate\_crop\_image 함수를 사용하여 이미지를 변환합니다.&#x20;
* 마지막으로, 변환된 이미지를 반환합니다.



## three\_way\_max

```c
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}
```

함수 이름: three\_way\_max

입력:

* a: float 타입의 변수
* b: float 타입의 변수
* c: float 타입의 변수

동작:

* 입력된 a, b, c 세 개의 변수 중에서 가장 큰 값을 반환한다.

설명:

* 세 개의 값을 비교하는데, 우선 a와 b를 비교한 후 그 중 큰 값을 max로 설정한다.
* 그리고 max와 c를 비교하여 큰 값을 반환한다.



## three\_way\_min

```c
float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}
```

함수 이름: three\_way\_min

입력:

* a: 비교하고자 하는 실수값
* b: 비교하고자 하는 실수값
* c: 비교하고자 하는 실수값

동작:&#x20;

* 주어진 세 개의 실수값 a, b, c 중 가장 작은 값을 반환합니다.

설명:&#x20;

* 이 함수는 주어진 세 개의 실수값 a, b, c 중 가장 작은 값을 반환하는 함수입니다.&#x20;
* 먼저 a와 b를 비교하여 더 작은 값을 선택하고, 그 선택된 값과 c를 비교하여 가장 작은 값을 반환합니다.



## yuv\_to\_rgb

```c
void yuv_to_rgb(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float y, u, v;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            y = get_pixel(im, i , j, 0);
            u = get_pixel(im, i , j, 1);
            v = get_pixel(im, i , j, 2);

            r = y + 1.13983*v;
            g = y + -.39465*u + -.58060*v;
            b = y + 2.03211*u;

            set_pixel(im, i, j, 0, r);
            set_pixel(im, i, j, 1, g);
            set_pixel(im, i, j, 2, b);
        }
    }
}
```

함수 이름: yuv\_to\_rgb&#x20;

입력:&#x20;

* image 타입의 이미지(im) 포인터&#x20;

동작:&#x20;

* YUV 색상 공간으로 표현된 이미지를 RGB 색상 공간으로 변환합니다.&#x20;

설명:&#x20;

* 이 함수는 이미지(im) 포인터를 입력으로 받아서 YUV 색상 공간으로 표현된 이미지를 RGB 색상 공간으로 변환합니다.&#x20;
* 변환된 이미지는 입력 이미지와 같은 이미지(im) 포인터에 저장됩니다.&#x20;
* YUV와 RGB 간의 변환은 일련의 수식으로 이루어집니다.&#x20;
* 이 함수는 이미지의 각 픽셀마다 YUV 값을 계산하고, 이를 RGB 값으로 변환하여 이미지에 저장합니다.



## rgb\_to\_yuv

```c
void rgb_to_yuv(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float y, u, v;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            r = get_pixel(im, i , j, 0);
            g = get_pixel(im, i , j, 1);
            b = get_pixel(im, i , j, 2);

            y = .299*r + .587*g + .114*b;
            u = -.14713*r + -.28886*g + .436*b;
            v = .615*r + -.51499*g + -.10001*b;

            set_pixel(im, i, j, 0, y);
            set_pixel(im, i, j, 1, u);
            set_pixel(im, i, j, 2, v);
        }
    }
}
```

함수 이름: rgb\_to\_yuv&#x20;

입력:&#x20;

* image im (RGB 이미지)&#x20;

동작:&#x20;

* 입력으로 주어진 RGB 이미지를 YUV 색 공간으로 변환한다.&#x20;
* 각각의 픽셀의 RGB 값을 이용하여 YUV 값을 계산하고, 새로운 이미지에 해당 값을 저장한다.&#x20;

설명:

* YUV 색 공간은 밝기(Y)와 색차(U, V)로 구성된다.
* Y 값은 입력 이미지의 RGB 값으로부터 계산된다.
* U와 V 값은 입력 이미지의 RGB 값을 이용하여 각각의 식을 통해 계산된다.
* 계산된 Y, U, V 값은 각각의 픽셀에 대해 set\_pixel 함수를 통해 저장된다.



## rgb\_to\_hsv

```c
void rgb_to_hsv(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float h, s, v;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            r = get_pixel(im, i , j, 0);
            g = get_pixel(im, i , j, 1);
            b = get_pixel(im, i , j, 2);
            float max = three_way_max(r,g,b);
            float min = three_way_min(r,g,b);
            float delta = max - min;
            v = max;
            if(max == 0){
                s = 0;
                h = 0;
            }else{
                s = delta/max;
                if(r == max){
                    h = (g - b) / delta;
                } else if (g == max) {
                    h = 2 + (b - r) / delta;
                } else {
                    h = 4 + (r - g) / delta;
                }
                if (h < 0) h += 6;
                h = h/6.;
            }
            set_pixel(im, i, j, 0, h);
            set_pixel(im, i, j, 1, s);
            set_pixel(im, i, j, 2, v);
        }
    }
}
```

함수 이름: rgb\_to\_hsv&#x20;

입력:&#x20;

* image im (RGB 이미지)&#x20;

동작:&#x20;

* 입력으로 주어진 RGB 이미지를 HSV 색 공간으로 변환한다.&#x20;

설명:&#x20;

* 이 함수는 입력으로 주어진 RGB 이미지를 HSV 색 공간으로 변환하는 함수이다.&#x20;
* 각 픽셀의 RGB 값을 읽어들인 후, 최댓값과 최솟값을 이용하여 V 값을 계산하고, S 값을 계산한다.&#x20;
* H 값을 계산하기 위해 R, G, B 중 최댓값을 찾고, 이를 기반으로 각 채널 값들을 이용하여 H 값을 계산한다.&#x20;
* 계산된 H, S, V 값을 이용하여 해당 픽셀의 새로운 색상 값을 설정하고, 이미지를 수정한다.



## hsv\_to\_rgb

```c
void hsv_to_rgb(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float h, s, v;
    float f, p, q, t;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            h = 6 * get_pixel(im, i , j, 0);
            s = get_pixel(im, i , j, 1);
            v = get_pixel(im, i , j, 2);
            if (s == 0) {
                r = g = b = v;
            } else {
                int index = floor(h);
                f = h - index;
                p = v*(1-s);
                q = v*(1-s*f);
                t = v*(1-s*(1-f));
                if(index == 0){
                    r = v; g = t; b = p;
                } else if(index == 1){
                    r = q; g = v; b = p;
                } else if(index == 2){
                    r = p; g = v; b = t;
                } else if(index == 3){
                    r = p; g = q; b = v;
                } else if(index == 4){
                    r = t; g = p; b = v;
                } else {
                    r = v; g = p; b = q;
                }
            }
            set_pixel(im, i, j, 0, r);
            set_pixel(im, i, j, 1, g);
            set_pixel(im, i, j, 2, b);
        }
    }
}
```

함수 이름: hsv\_to\_rgb&#x20;

입력:&#x20;

* image 타입의 이미지 데이터 (세 개의 채널을 가져야 함)&#x20;

동작:&#x20;

* HSV(Hue, Saturation, Value) 색상 모델로 표현된 이미지를 RGB(Red, Green, Blue) 색상 모델로 변환함&#x20;

설명:&#x20;

* 입력으로 주어진 이미지 데이터는 HSV 색상 모델로 표현되며, 각 픽셀은 Hue, Saturation, Value의 세 가지 요소로 구성됩니다.&#x20;
* 이 함수는 입력 이미지를 순회하면서 각 픽셀의 HSV 값을 이용하여 RGB 값을 계산하고, 계산된 RGB 값을 이용하여 입력 이미지를 RGB 색상 모델로 변환합니다.&#x20;
* 계산 방법은 HSV 색상 모델과 RGB 색상 모델 간의 변환식을 이용합니다.



## grayscale\_image\_3c

```c
void grayscale_image_3c(image im)
{
    assert(im.c == 3);
    int i, j, k;
    float scale[] = {0.299, 0.587, 0.114};
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            float val = 0;
            for(k = 0; k < 3; ++k){
                val += scale[k]*get_pixel(im, i, j, k);
            }
            im.data[0*im.h*im.w + im.w*j + i] = val;
            im.data[1*im.h*im.w + im.w*j + i] = val;
            im.data[2*im.h*im.w + im.w*j + i] = val;
        }
    }
}
```

함수 이름: grayscale\_image\_3c&#x20;

입력:&#x20;

* image im (그레이스케일로 변환할 입력 이미지)&#x20;

동작:&#x20;

* 입력 이미지를 그레이스케일로 변환하여 입력 이미지의 3채널 값을 모두 같은 값으로 만듦&#x20;

설명:&#x20;

* 입력 이미지의 채널 수가 3이 아닐 경우 에러를 발생시키고, 입력 이미지의 각각의 픽셀 값을 RGB 채널의 가중치 값을 곱하여 더한 후, 결과 값을 입력 이미지의 3채널 모두에 대입하여 그레이스케일 이미지로 변환합니다.



## grayscale\_image

```c
image grayscale_image(image im)
{
    assert(im.c == 3);
    int i, j, k;
    image gray = make_image(im.w, im.h, 1);
    float scale[] = {0.299, 0.587, 0.114};
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < im.h; ++j){
            for(i = 0; i < im.w; ++i){
                gray.data[i+im.w*j] += scale[k]*get_pixel(im, i, j, k);
            }
        }
    }
    return gray;
}
```

함수 이름: grayscale\_image&#x20;

입력:&#x20;

* image im (3채널 컬러 이미지)&#x20;

동작:&#x20;

* 입력으로 들어온 3채널 컬러 이미지를 그레이스케일 이미지로 변환한다.&#x20;
* 변환 방법은 RGB 각각의 채널에 대해 0.299, 0.587, 0.114의 가중치를 적용하여 합한 값을 하나의 픽셀 값으로 사용한다.&#x20;

설명:&#x20;

* 입력 이미지의 채널 수가 3이 아닐 경우 함수가 종료된다.&#x20;
* 변환된 그레이스케일 이미지는 1채널을 가지며, make\_image 함수를 사용하여 생성된다.&#x20;
* 이미지의 픽셀 값을 가져오거나 설정하기 위해서는 get\_pixel, set\_pixel 함수를 사용한다.&#x20;
* 변환된 그레이스케일 이미지가 반환된다.



## threshold\_image

```c
image threshold_image(image im, float thresh)
{
    int i;
    image t = make_image(im.w, im.h, im.c);
    for(i = 0; i < im.w*im.h*im.c; ++i){
        t.data[i] = im.data[i]>thresh ? 1 : 0;
    }
    return t;
}
```

함수 이름: threshold\_image

입력:&#x20;

* image im (이미지)
* float thresh (임계값)

동작:&#x20;

* 입력 이미지에서 임계값보다 큰 값은 1로, 작거나 같은 값은 0으로 변환한 이미지를 반환합니다.

설명:&#x20;

* 입력으로 들어온 이미지(im)의 너비, 높이, 채널 수와 같은 크기의 새로운 이미지(t)를 만들고, 입력 이미지의 모든 픽셀 값에 대해 임계값(thresh)과 비교합니다.&#x20;
* 만약 픽셀 값이 임계값보다 크면 1로, 그렇지 않으면 0으로 설정하여 t에 저장합니다.&#x20;
* 최종적으로 변환된 이미지(t)를 반환합니다.



## blend\_image

```c
image blend_image(image fore, image back, float alpha)
{
    assert(fore.w == back.w && fore.h == back.h && fore.c == back.c);
    image blend = make_image(fore.w, fore.h, fore.c);
    int i, j, k;
    for(k = 0; k < fore.c; ++k){
        for(j = 0; j < fore.h; ++j){
            for(i = 0; i < fore.w; ++i){
                float val = alpha * get_pixel(fore, i, j, k) +
                    (1 - alpha)* get_pixel(back, i, j, k);
                set_pixel(blend, i, j, k, val);
            }
        }
    }
    return blend;
}
```

함수 이름: blend\_image

입력:&#x20;

* (image) fore: 전경 이미지
* (image) back: 배경 이미지
* (float) alpha - 전경 이미지의 가중치 값

동작:&#x20;

* fore와 back 이미지의 크기와 채널 수가 같은지 확인한 후, fore와 back 이미지를 alpha 값에 따라 섞은 새로운 이미지를 만들어 반환한다.

설명:

* fore와 back 이미지는 같은 크기와 채널 수를 가지고 있어야 한다.
* fore 이미지의 각 픽셀 값에 alpha 값을 곱하고, back 이미지의 해당 픽셀 값에 (1 - alpha)를 곱한 후, 더해서 새로운 이미지의 해당 픽셀 값으로 설정한다.
* 반환되는 blend 이미지는 fore와 back 이미지를 alpha 값에 따라 섞은 결과이다.



## scale\_image\_channel

```c
void scale_image_channel(image im, int c, float v)
{
    int i, j;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            float pix = get_pixel(im, i, j, c);
            pix = pix*v;
            set_pixel(im, i, j, c, pix);
        }
    }
}
```

함수 이름: scale\_image\_channel

입력:&#x20;

* image 타입의 이미지 m
* 정수 타입의 채널 번호 c
* 실수 타입의 배율 값 v

동작:&#x20;

* 이미지의 특정 채널 값을 주어진 배율 값으로 스케일링(scale)합니다.

설명:&#x20;

* 입력으로 들어온 이미지 m의 특정 채널 c의 각 픽셀 값을 v 배율 값으로 곱하여 스케일링합니다.&#x20;
* 이미지의 너비와 높이에 대해 각각 반복문을 수행하여 이미지의 모든 픽셀에 대해 채널 c의 값을 스케일링합니다.



## translate\_image\_channel

```c
void translate_image_channel(image im, int c, float v)
{
    int i, j;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            float pix = get_pixel(im, i, j, c);
            pix = pix+v;
            set_pixel(im, i, j, c, pix);
        }
    }
}
```

함수 이름: translate\_image\_channel&#x20;

입력:&#x20;

* image im (이미지)
* int c (채널)
* float v (이동할 값)&#x20;

동작:&#x20;

* 입력된 이미지의 지정된 채널을 v만큼 이동합니다.&#x20;

설명:&#x20;

* 입력된 이미지 im의 c번째 채널의 모든 픽셀 값을 v만큼 이동하여 변경합니다.&#x20;
* 각 픽셀의 값을 get\_pixel()로 가져와서 v만큼 더하고, 변경된 값을 set\_pixel()로 설정합니다.



## binarize\_image

```c
image binarize_image(image im)
{
    image c = copy_image(im);
    int i;
    for(i = 0; i < im.w * im.h * im.c; ++i){
        if(c.data[i] > .5) c.data[i] = 1;
        else c.data[i] = 0;
    }
    return c;
}
```

함수 이름: binarize\_image&#x20;

입력:&#x20;

* image im (이진화할 이미지)&#x20;

동작:&#x20;

* 입력 이미지의 각 픽셀 값을 확인하여 0.5보다 크면 1로, 작으면 0으로 바꾸어 새로운 이미지를 만든 후 반환합니다.&#x20;

설명:&#x20;

* 입력으로 받은 이미지를 이진화하여 새로운 이미지를 생성합니다.&#x20;
* 이진화란, 이미지를 흑백으로 변환하고, 경계값(threshold) 이상의 픽셀을 1로, 이하의 픽셀을 0으로 바꾸는 것을 의미합니다.&#x20;
* 반환되는 이미지는 원래 이미지와 같은 크기를 가지며, 각 픽셀은 0 또는 1의 값을 가집니다.



## saturate\_image

```c
void saturate_image(image im, float sat)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    hsv_to_rgb(im);
    constrain_image(im);
}
```

함수 이름: saturate\_image&#x20;

입력:&#x20;

* image im (이미지)
* float sat (포화도 조절 값)&#x20;

동작:&#x20;

* 입력으로 받은 이미지의 색 공간을 RGB에서 HSV로 변환한 후, HSV에서 Saturation(포화도) 채널 값을 주어진 값(sat)만큼 조절합니다.&#x20;
* 그리고 다시 HSV에서 RGB로 변환하여 이미지를 반환합니다.&#x20;

설명:&#x20;

* 이 함수는 입력된 이미지의 색상을 포화도를 조절하여 변환시키는 함수입니다.&#x20;
* 포화도는 색의 진하고 연함의 정도를 나타내며, 이 값을 조절하여 이미지에 풍부한 색감을 부여하거나 흑백 이미지로 변환하는 등의 다양한 응용이 가능합니다.



## hue\_image

```c
void hue_image(image im, float hue)
{
    rgb_to_hsv(im);
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        im.data[i] = im.data[i] + hue;
        if (im.data[i] > 1) im.data[i] -= 1;
        if (im.data[i] < 0) im.data[i] += 1;
    }
    hsv_to_rgb(im);
    constrain_image(im);
}
```

함수 이름: hue\_image&#x20;

입력:&#x20;

* image im (이미지)
* float hue (색조 값)&#x20;

동작:&#x20;

* 입력 이미지의 색조를 지정된 값만큼 조정합니다.&#x20;
* 입력 이미지를 RGB에서 HSV 색 공간으로 변환한 다음, 픽셀의 색조 값을 조정합니다.&#x20;
* 이후 다시 RGB 색 공간으로 변환하고, 최대/최소값을 벗어나는 픽셀 값을 제한합니다.&#x20;

설명:&#x20;

* 이미지의 색조를 변경하는 함수입니다.&#x20;
* 색조 값은 -1과 1 사이의 값이며, 음수 값은 색조를 반대 방향으로 변경합니다.&#x20;
* 예를 들어, hue=-0.1이면 색조가 0.1 만큼 감소하고, hue=0.2이면 색조가 0.2 만큼 증가합니다.



## exposure\_image

```c
void exposure_image(image im, float sat)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 2, sat);
    hsv_to_rgb(im);
    constrain_image(im);
}
```

함수 이름: exposure\_image&#x20;

입력:&#x20;

* image im (이미지 데이터)
* float sat (밝기 조절 비율)&#x20;

동작:&#x20;

* 입력 이미지를 HSV 색 공간으로 변환한 후, 밝기 채널에 sat 비율을 곱한 값을 설정하고 다시 RGB 색 공간으로 변환합니다.&#x20;
* 변환 후 픽셀 값이 0과 1 사이를 벗어나는 경우, 최대값과 최소값으로 조정합니다.&#x20;

설명:&#x20;

* 이미지의 밝기를 조절합니다.&#x20;
* 입력으로 들어온 이미지 데이터의 픽셀 값들을 밝기 조절 비율인 sat로 조정합니다.



## distort\_image

```c
void distort_image(image im, float hue, float sat, float val)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    scale_image_channel(im, 2, val);
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        im.data[i] = im.data[i] + hue;
        if (im.data[i] > 1) im.data[i] -= 1;
        if (im.data[i] < 0) im.data[i] += 1;
    }
    hsv_to_rgb(im);
    constrain_image(im);
}
```

함수 이름: distort\_image

입력:

* im: 왜곡을 적용할 이미지 (image 타입)
* hue: 색조 왜곡 정도 (float 타입)
* sat: 채도 왜곡 정도 (float 타입)
* val: 명도 왜곡 정도 (float 타입)

동작:&#x20;

* 입력 이미지에 색조, 채도, 명도 왜곡을 랜덤하게 적용합니다.&#x20;
* 우선 RGB 색 공간에서 HSV 색 공간으로 변환한 후, 채도와 명도를 각각 주어진 값만큼 조정합니다.&#x20;
* 그 후, 각 픽셀의 색조를 주어진 값만큼 변화시키고, 0\~1 사이의 값으로 제한합니다. 마지막으로 HSV 색 공간에서 다시 RGB 색 공간으로 변환한 후, 픽셀 값이 0\~1 사이에 위치하도록 값을 제한합니다.

설명:&#x20;

* 입력 이미지에 왜곡을 적용하는 함수입니다.&#x20;
* 색상, 채도, 명도 각각의 값을 랜덤하게 조절하여 입력 이미지에 변화를 줍니다.&#x20;
* 변환된 이미지는 다시 RGB 색 공간으로 변환되고, 픽셀 값이 0\~1 사이에 위치하도록 제한합니다.



## random\_distort\_image

```c
void random_distort_image(image im, float hue, float saturation, float exposure)
{
    float dhue = rand_uniform(-hue, hue);
    float dsat = rand_scale(saturation);
    float dexp = rand_scale(exposure);
    distort_image(im, dhue, dsat, dexp);
}
```

함수 이름: random\_distort\_image

입력:

* im: 왜곡(distortion)을 적용할 이미지를 나타내는 image 구조체 포인터
* hue: 이미지에 적용할 색조(hue) 왜곡의 최댓값(float)
* saturation: 이미지에 적용할 채도(saturation) 왜곡의 최댓값(float)
* exposure: 이미지에 적용할 노출(exposure) 왜곡의 최댓값(float)

동작:&#x20;

* 입력 이미지에 hue, saturation, exposure 왜곡을 무작위로 적용하여 왜곡된 이미지를 생성합니다.

설명:

* dhue: 무작위로 생성된 hue 왜곡의 정도(float)
* dsat: 무작위로 생성된 saturation 왜곡의 정도(float)
* dexp: 무작위로 생성된 exposure 왜곡의 정도(float)
* distort\_image(): hue, saturation, exposure 왜곡을 입력 이미지에 적용하는 함수



## saturate\_exposure\_image

```c
void saturate_exposure_image(image im, float sat, float exposure)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    scale_image_channel(im, 2, exposure);
    hsv_to_rgb(im);
    constrain_image(im);
}
```

함수 이름: saturate\_exposure\_image

입력:

* im: 조정할 이미지를 나타내는 image 구조체 포인터
* sat: 적용할 채도(scale) 값
* exposure: 적용할 노출(exposure) 값

동작:

* 입력으로 받은 이미지의 RGB 값을 HSV 값으로 변환
* 이미지의 채도 채널을 주어진 값으로 조정
* 이미지의 노출 채널을 주어진 값으로 조정
* 이미지의 HSV 값을 RGB 값으로 다시 변환
* 값이 0과 1 사이를 벗어나는 픽셀 값을 0 또는 1로 잘라내어 이미지를 제한

설명:&#x20;

* 입력으로 받은 이미지의 채도와 노출 값을 주어진 값으로 조정하는 함수이다.&#x20;
* 이미지의 RGB 값으로부터 HSV 값으로 변환하여 채도와 노출 값을 조정한 뒤, 다시 RGB 값으로 변환한다.&#x20;
* 이 때, 픽셀 값이 0과 1 사이를 벗어나는 경우 이를 0 또는 1로 잘라내어 이미지를 제한한다.



## resize\_image

```c
image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);   
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < im.h; ++r){
            for(c = 0; c < w; ++c){
                float val = 0;
                if(c == w-1 || im.w == 1){
                    val = get_pixel(im, im.w-1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                float val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if(r == h-1 || im.h == 1) continue;
            for(c = 0; c < w; ++c){
                float val = dy * get_pixel(part, c, iy+1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}

```

함수 이름: resize\_image

입력:&#x20;

* image im (원본 이미지)
* int w (변경할 너비)
* int h (변경할 높이)

동작:&#x20;

* 입력된 원본 이미지를 주어진 w와 h 크기로 변경하고, 새로운 크기에 맞게 이미지의 픽셀 값을 보간하여 조절합니다.

설명:

* make\_image 함수를 사용하여 새로운 크기의 이미지를 만들고, 이를 resized 변수에 저장합니다.
* make\_image 함수를 사용하여 너비가 w이고 원래 이미지의 높이와 채널 수를 가진 이미지를 만들고, 이를 part 변수에 저장합니다.
* 주어진 너비(w)와 높이(h)로부터 이미지 크기를 변경하기 위해 보간(interpolation)을 수행합니다. w\_scale과 h\_scale 변수는 원본 이미지에서 변환된 이미지로의 스케일링 값을 나타냅니다.
* for 루프를 사용하여 원본 이미지의 픽셀 값을 보간하여 part 이미지에 저장합니다. 보간(interpolation) 방법으로는 이전 픽셀과 다음 픽셀의 가중평균을 사용합니다.
* for 루프를 사용하여 part 이미지에서 변환된 이미지의 각 픽셀 값을 계산하고, resized 이미지에 저장합니다. 이 때, 현재 위치와 다음 위치 사이의 보간(interpolation)을 수행합니다.



## test\_resize

```c
void test_resize(char *filename)
{
    image im = load_image(filename, 0,0, 3);
    float mag = mag_array(im.data, im.w*im.h*im.c);
    printf("L2 Norm: %f\n", mag);
    image gray = grayscale_image(im);

    image c1 = copy_image(im);
    image c2 = copy_image(im);
    image c3 = copy_image(im);
    image c4 = copy_image(im);
    distort_image(c1, .1, 1.5, 1.5);
    distort_image(c2, -.1, .66666, .66666);
    distort_image(c3, .1, 1.5, .66666);
    distort_image(c4, .1, .66666, 1.5);


    show_image(im,   "Original", 1);
    show_image(gray, "Gray", 1);
    show_image(c1, "C1", 1);
    show_image(c2, "C2", 1);
    show_image(c3, "C3", 1);
    show_image(c4, "C4", 1);
#ifdef OPENCV
    while(1){
        image aug = random_augment_image(im, 0, .75, 320, 448, 320, 320);
        show_image(aug, "aug", 1);
        free_image(aug);


        float exposure = 1.15;
        float saturation = 1.15;
        float hue = .05;

        image c = copy_image(im);

        float dexp = rand_scale(exposure);
        float dsat = rand_scale(saturation);
        float dhue = rand_uniform(-hue, hue);

        distort_image(c, dhue, dsat, dexp);
        show_image(c, "rand", 1);
        printf("%f %f %f\n", dhue, dsat, dexp);
        free_image(c);
    }
#endif
}

```

함수 이름: test\_resize

입력:

* filename: char 포인터 타입. 리사이즈할 이미지 파일명을 지정하는 문자열 포인터.

동작:

* 입력된 filename을 이용하여 이미지 파일을 불러온다.
* 이미지 데이터의 L2 Norm을 계산하여 출력한다.
* 불러온 이미지를 grayscale로 변환한 이미지 데이터를 생성한다.
* 불러온 이미지 데이터를 복사하여 왜곡(distortion)된 이미지 데이터를 생성한다.
* 왜곡된 이미지 데이터를 각각 다른 유형의 왜곡값으로 변환한다.
* 변환된 이미지들과 원본 이미지, grayscale로 변환된 이미지를 차례대로 화면에 출력한다.
* OPENCV 매크로가 정의되어 있을 경우, 이미지 데이터를 무작위로 변환하여 화면에 출력한다. 변환에 사용되는 값들은 무작위로 결정되며, 화면 출력 후에는 해당 이미지 데이터를 해제한다.

설명:&#x20;

* 이 함수는 입력된 filename을 이용하여 이미지 파일을 불러와 여러 가지 방식으로 변환하여 화면에 출력하는 역할을 한다.&#x20;
* 먼저 이미지 데이터의 L2 Norm을 계산하여 출력하고, grayscale로 변환한 이미지 데이터와 원본 이미지 데이터를 화면에 출력한다.&#x20;
* 그리고 왜곡된 이미지 데이터를 생성하여 각각 다른 유형의 왜곡값으로 변환한 다음, 이를 차례대로 화면에 출력한다.&#x20;
* OPENCV 매크로가 정의되어 있을 경우, 이미지 데이터를 무작위로 변환하여 화면에 출력하며, 변환에 사용되는 값들은 무작위로 결정된다.&#x20;
* 출력 후에는 해당 이미지 데이터를 해제하여 메모리 누수를 방지한다.



## load\_image\_stb

```c
image load_image_stb(char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
    if(channels) c = channels;
    int i,j,k;
    image im = make_image(w, h, c);
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index]/255.;
            }
        }
    }
    free(data);
    return im;
}
```

함수 이름: load\_image\_stb

입력:&#x20;

* filename(char\*): 이미지 파일 경로
* channels(int): 채널 수

동작:&#x20;

* STB 라이브러리를 이용하여 입력한 이미지 파일을 읽어들이고, 픽셀 데이터를 메모리에 할당한 후 이미지 구조체(image)로 변환하여 반환합니다.&#x20;
* 변환된 이미지 데이터는 \[0, 1] 범위로 정규화되어 있습니다.

설명:&#x20;

* 입력한 이미지 파일 경로(filename)와 채널 수(channels)를 인자로 받습니다.&#x20;
* 입력된 이미지 파일을 STB 라이브러리를 이용하여 읽어들입니다.&#x20;
* 읽어들인 이미지의 픽셀 데이터를 메모리에 할당하고, 이미지 구조체(image)로 변환합니다.&#x20;
* 변환된 이미지 데이터는 \[0, 1] 범위로 정규화되어 있습니다.



## load\_image

```c
image load_image(char *filename, int w, int h, int c)
{
#ifdef OPENCV
    image out = load_image_cv(filename, c);
#else
    image out = load_image_stb(filename, c);
#endif

    if((h && w) && (h != out.h || w != out.w)){
        image resized = resize_image(out, w, h);
        free_image(out);
        out = resized;
    }
    return out;
}
```

함수 이름: load\_image

입력:

* filename: 이미지 파일 경로
* w: 로드한 이미지의 너비 (0이면 원본 이미지의 너비 사용)
* h: 로드한 이미지의 높이 (0이면 원본 이미지의 높이 사용)
* c: 로드한 이미지의 채널 수

동작:

* OpenCV 라이브러리가 사용 가능한 경우, load\_image\_cv 함수를 사용하여 이미지를 로드합니다. 그렇지 않으면 load\_image\_stb 함수를 사용합니다.
* 만약 w와 h가 0이 아니고 로드한 이미지의 크기와 다르다면, resize\_image 함수를 사용하여 이미지를 크기를 w × h로 조정합니다.
* 최종적으로 로드한 이미지를 반환합니다.

설명:&#x20;

* 이 함수는 이미지 파일을 로드하여 image 구조체를 반환하는 함수입니다.&#x20;
* 로드할 이미지 파일의 경로와 크기, 채널 수를 입력으로 받습니다.&#x20;
* OpenCV 라이브러리가 사용 가능한 경우, OpenCV 함수를 사용하여 이미지를 로드합니다.&#x20;
* 그렇지 않으면 STB 라이브러리를 사용합니다.&#x20;
* 로드한 이미지의 크기가 w × h와 다르다면 resize\_image 함수를 사용하여 이미지 크기를 조정합니다.&#x20;
* 최종적으로 로드한 이미지를 반환합니다.



## load\_image\_color

```c
image load_image_color(char *filename, int w, int h)
{
    return load_image(filename, w, h, 3);
}
```

함수 이름: load\_image\_color

입력:

* filename: 이미지 파일 경로
* w: 이미지의 가로 크기
* h: 이미지의 세로 크기

동작:&#x20;

* 이미지 파일을 로드하여, 컬러 이미지로 변환한 후 반환한다.

설명:&#x20;

* 이 함수는 주어진 이미지 파일 경로에서 이미지를 로드하고, 가로 크기와 세로 크기를 지정하여 이미지를 리사이징한 후, 컬러 이미지로 변환하여 반환한다.&#x20;
* 반환되는 이미지는 'image' 타입이며, R, G, B 세 개의 채널을 갖는 컬러 이미지이다.&#x20;
* 이때, 이미지 파일이 로드되지 않거나, 지정된 경로에 파일이 존재하지 않는 경우, 프로그램이 오류를 반환할 수 있다.



## get\_image\_layer

```c
image get_image_layer(image m, int l)
{
    image out = make_image(m.w, m.h, 1);
    int i;
    for(i = 0; i < m.h*m.w; ++i){
        out.data[i] = m.data[i+l*m.h*m.w];
    }
    return out;
}
```

함수 이름: get\_image\_layer

입력:&#x20;

* image m (원본 이미지)
* int l (가져올 레이어 인덱스)

동작:&#x20;

* 입력으로 받은 원본 이미지에서 레이어 인덱스에 해당하는 레이어를 가져와서 1채널 이미지로 만들어 반환합니다.

설명:&#x20;

* 입력으로 받은 이미지 m은 여러 채널을 가지고 있습니다.&#x20;
* 이 함수는 이 중에서 l번째 레이어를 가져와서 1채널 이미지로 만듭니다.&#x20;
* 이를 위해서는 이미지의 높이와 너비만큼 반복하면서 레이어 인덱스에 해당하는 값을 가져와서 새로 만든 1채널 이미지 out의 data 배열에 넣어주면 됩니다.&#x20;
* 최종적으로 1채널 이미지 out을 반환합니다.



## print\_image

```c
void print_image(image m)
{
    int i, j, k;
    for(i =0 ; i < m.c; ++i){
        for(j =0 ; j < m.h; ++j){
            for(k = 0; k < m.w; ++k){
                printf("%.2lf, ", m.data[i*m.h*m.w + j*m.w + k]);
                if(k > 30) break;
            }
            printf("\n");
            if(j > 30) break;
        }
        printf("\n");
    }
    printf("\n");
}
```

함수 이름: print\_image

입력:&#x20;

* image m (출력할 이미지)

동작:&#x20;

* 입력된 이미지의 데이터 값을 출력하는 함수이다.&#x20;
* 출력되는 값은 세 개의 for 루프를 통해 이미지의 가로, 세로 및 채널에 대한 데이터 값이다.

설명:&#x20;

* 입력된 이미지의 데이터 값을 출력하는 함수이다.&#x20;
* 이미지의 가로, 세로 및 채널에 대한 데이터 값을 반복문을 통해 출력하며, 각 값은 소수점 이하 둘째 자리까지 출력된다.&#x20;
* 출력되는 값은 총 세 개의 for 루프를 통해 이미지의 가로, 세로 및 채널에 대한 데이터 값이며, 가로와 세로 값이 각각 30보다 큰 경우 해당 줄에서 출력이 중단된다.



## collapse\_images\_vert

```c
image collapse_images_vert(image *ims, int n)
{
    int color = 1;
    int border = 1;
    int h,w,c;
    w = ims[0].w;
    h = (ims[0].h + border) * n - border;
    c = ims[0].c;
    if(c != 3 || !color){
        w = (w+border)*c - border;
        c = 1;
    }

    image filters = make_image(w, h, c);
    int i,j;
    for(i = 0; i < n; ++i){
        int h_offset = i*(ims[0].h+border);
        image copy = copy_image(ims[i]);
        //normalize_image(copy);
        if(c == 3 && color){
            embed_image(copy, filters, 0, h_offset);
        }
        else{
            for(j = 0; j < copy.c; ++j){
                int w_offset = j*(ims[0].w+border);
                image layer = get_image_layer(copy, j);
                embed_image(layer, filters, w_offset, h_offset);
                free_image(layer);
            }
        }
        free_image(copy);
    }
    return filters;
}
```

함수 이름: collapse\_images\_vert&#x20;

입력:&#x20;

* image \*ims (이미지 배열 포인터)
* int n (이미지 배열의 원소 개수)&#x20;

동작:&#x20;

* ims 배열에 있는 이미지들을 수직으로 합친 하나의 이미지를 생성하고, 이를 반환함. 수직으로 합친 이미지는 ims 배열 원소들의 높이를 모두 더한 값에 각 이미지 사이에 border 값을 더한 것이 높이이고, ims 배열 원소들의 폭 중 가장 큰 값에 c 값(3 또는 1)에 따라 폭을 결정함.&#x20;

설명:

* color와 border는 이미지 생성에 필요한 변수로 각각 1과 1로 초기화됨.
* w, h, c는 반환될 이미지의 폭, 높이, 채널 값으로 ims\[0]의 값들로 초기화됨.
* ims\[0]의 채널 값이 3이 아니거나 color가 0일 경우 w값은 ims\[0]의 폭과 c 값을 이용하여 결정됨.
* make\_image 함수를 이용하여 반환될 이미지 filters를 생성함.
* ims 배열 원소들을 반복문으로 돌면서, 각 원소의 이미지를 copy\_image 함수를 이용하여 copy에 복사함.
* 이미지 채널 값이 3이고 color가 1일 경우, ims 배열 원소 이미지를 embed\_image 함수를 이용하여 filters 이미지에 추가함.
* 이미지 채널 값이 3이 아니거나 color가 0일 경우, 이미지 채널 수 만큼 반복문을 돌면서, get\_image\_layer 함수를 이용하여 이미지의 채널을 가져와 layer에 저장하고, embed\_image 함수를 이용하여 filters 이미지에 추가함.
* 반복문이 끝나면, filters 이미지를 반환하고, 이때 copy 변수와 각 layer들은 free\_image 함수를 이용하여 메모리 해제됨.



## collapse\_images\_horz

```c
image collapse_images_horz(image *ims, int n)
{
    int color = 1;
    int border = 1;
    int h,w,c;
    int size = ims[0].h;
    h = size;
    w = (ims[0].w + border) * n - border;
    c = ims[0].c;
    if(c != 3 || !color){
        h = (h+border)*c - border;
        c = 1;
    }

    image filters = make_image(w, h, c);
    int i,j;
    for(i = 0; i < n; ++i){
        int w_offset = i*(size+border);
        image copy = copy_image(ims[i]);
        //normalize_image(copy);
        if(c == 3 && color){
            embed_image(copy, filters, w_offset, 0);
        }
        else{
            for(j = 0; j < copy.c; ++j){
                int h_offset = j*(size+border);
                image layer = get_image_layer(copy, j);
                embed_image(layer, filters, w_offset, h_offset);
                free_image(layer);
            }
        }
        free_image(copy);
    }
    return filters;
}
```

함수 이름: collapse\_images\_horz

입력:&#x20;

* image \*ims (이미지 배열 포인터)
* int n (이미지 개수)

동작:&#x20;

* 이미지 배열을 수평으로 합쳐서 새로운 이미지를 만듭니다.&#x20;
* 입력된 이미지 배열에서 이미지의 높이를 구한 후, 해당 높이와 같은 크기를 가지는 새로운 이미지를 만듭니다.&#x20;
* 그 다음, 입력된 이미지 배열에서 이미지를 하나씩 꺼내어 복사하고, 만든 새로운 이미지에 붙여넣습니다.&#x20;
* 이미지들은 각각 border 픽셀 만큼의 공간을 띄어서 붙여넣으며, 새로운 이미지가 color 이미지일 경우 각 이미지를 RGB 채널별로 붙여넣습니다.

설명:

* size: 이미지 높이
* color: 입력된 이미지 배열의 이미지가 컬러 이미지인지 여부
* border: 이미지 사이의 간격
* h, w, c: 새로 만들 이미지의 높이, 너비, 채널 수
* filters: 수평으로 합쳐진 이미지를 저장할 이미지 구조체
* copy: 입력된 이미지 배열에서 꺼낸 이미지를 복사한 이미지
* layer: 입력된 이미지의 RGB 채널 중 하나인 이미지



## show\_image\_normalized

```c
void show_image_normalized(image im, const char *name)
{
    image c = copy_image(im);
    normalize_image(c);
    show_image(c, name, 1);
    free_image(c);
}
```

함수 이름: show\_image\_normalized

입력:

* image im: 보여줄 이미지
* const char \*name: 창의 이름

동작:

* 입력된 이미지를 복사하여 정규화(normalize)합니다.
* 정규화된 이미지를 창에 보여줍니다.
* 복사한 이미지와 정규화된 이미지를 모두 해제합니다.

설명:&#x20;

* 이 함수는 입력된 이미지를 복사하여 정규화(normalize)한 뒤, 그 결과를 창에 보여주는 함수입니다.&#x20;
* 정규화된 이미지를 보여주는 이유는, 이미지의 픽셀 값이 0에서 1사이의 값으로 스케일링 되어 있기 때문입니다.&#x20;
* 이를 그대로 보여주면 어두운 이미지는 거의 보이지 않고 밝은 이미지만 보일 수 있습니다.&#x20;
* 따라서 정규화된 이미지를 보여줌으로써 더 자세하고 균일한 이미지를 보여줄 수 있습니다.&#x20;
* 이 함수는 보통 디버깅이나 시각화 등의 용도로 사용됩니다.



## show\_images

```c
void show_images(image *ims, int n, char *window)
{
    image m = collapse_images_vert(ims, n);

    normalize_image(m);
    save_image(m, window);
    show_image(m, window, 1);
    free_image(m);
}
```

함수 이름: show\_images

입력:

* image \*ims: 이미지 배열의 포인터
* int n: 이미지 배열의 길이
* char \*window: 이미지 창의 이름

동작:

* 여러 이미지를 수직 방향으로 합치고, 정규화(normalize)합니다.
* 정규화된 이미지를 파일로 저장하고, 이미지 창에 띄웁니다.
* 이미지 합치기에 사용된 메모리를 해제합니다.

설명:&#x20;

* 이미지 배열을 수직 방향으로 합친 뒤, 그 결과를 하나의 이미지로 만듭니다.&#x20;
* 이 이미지는 정규화(normalization)됩니다.&#x20;
* 이후, 정규화된 이미지는 파일로 저장되며, 이미지 창에 띄워집니다.&#x20;
* 이미지 창의 이름은 인자로 전달된 문자열을 사용합니다. 마지막으로, 이미지 합치기에 사용된 메모리는 해제됩니다.



## free\_image

```c
void free_image(image m)
{
    if(m.data){
        free(m.data);
    }
}
```

함수 이름: free\_image

입력:&#x20;

* image 타입의 변수 m

동작:&#x20;

* 입력으로 받은 이미지 데이터 m의 메모리를 해제함

설명:&#x20;

* YOLO의 이미지 데이터 구조체 image의 메모리를 해제하는 함수입니다.&#x20;
* 이미지 데이터의 포인터를 가지고 있는 m.data가 NULL이 아닌 경우에 해당 메모리를 해제합니다.

