# im2col

* 이미지를 columns으로 변환해주는 것을 말합니다.

## im2col\_get\_pixel

```c
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}
```

함수 이름: im2col\_get\_pixel

입력:

* im: 이미지 데이터를 가리키는 포인터(float \*)
* height: 이미지 높이(int)
* width: 이미지 너비(int)
* channels: 이미지 채널 수(int)
* row: 픽셀의 세로 위치(int)
* col: 픽셀의 가로 위치(int)
* channel: 픽셀이 위치한 채널(int)
* pad: 패딩 크기(int)

동작:&#x20;

* 입력으로 주어진 위치(row, col, channel)에 해당하는 픽셀 값을 반환한다.&#x20;
* 단, 패딩(pad)이 적용되어 있는 경우, 패딩 크기만큼 위치를 조정하여 이미지 데이터를 가져온다.&#x20;
* 만약 가져올 데이터가 이미지의 범위를 벗어나는 경우, 0을 반환한다.

설명:&#x20;

* 이미지 데이터를 2차원 배열 형태로 변환하는 im2col 연산에서, 주어진 위치에 대응하는 데이터를 가져오기 위해 사용된다.&#x20;
* 이 함수는 해당 위치(row, col, channel)에서의 값을 가져오는 역할을 한다.&#x20;
* 이미지 데이터는 1차원 배열로 저장되어 있기 때문에, row, col, channel 정보를 이용하여 인덱스를 계산하여 값을 가져온다.&#x20;
* 단, 패딩이 적용된 경우, row와 col에서 패딩의 크기(pad)를 뺀 값을 사용하여 인덱스를 계산한다.



## im2col\_cpu

```c
//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}
```

함수 이름: im2col\_cpu&#x20;

입력:

* data\_im: float 포인터, 입력 이미지 데이터 포인터
* channels: int, 입력 이미지 채널 수
* height: int, 입력 이미지 높이
* width: int, 입력 이미지 너비
* ksize: int, 필터(커널) 크기
* stride: int, 스트라이드 크기
* pad: int, 패딩 크기
* data\_col: float 포인터, 변환된 이미지 데이터 포인터

동작:&#x20;

* 입력 이미지를 필터 크기, 스트라이드, 패딩 정보를 기반으로 im2col 방식으로 변환하여 data\_col에 저장하는 함수입니다.&#x20;
* Caffe라는 머신러닝 프레임워크에서 가져온 코드로, 입력 이미지의 각 채널에서 필터의 크기만큼 움직여가며 슬라이딩 윈도우를 만들고, 윈도우 내의 값을 일렬로 늘어놓은 다음, data\_col에 저장합니다.

설명:

* height\_col: int, im2col 변환 후 출력 이미지 높이
* width\_col: int, im2col 변환 후 출력 이미지 너비
* channels\_col: int, im2col 변환 후 출력 이미지 채널 수
* c: int, channels\_col 내 현재 채널 인덱스
* w\_offset: int, 현재 채널에서 필터의 너비 오프셋
* h\_offset: int, 현재 채널에서 필터의 높이 오프셋
* c\_im: int, 현재 채널에서 입력 이미지 채널 인덱스
* h: int, 출력 이미지의 현재 높이 인덱스
* w: int, 출력 이미지의 현재 너비 인덱스
* im\_row: int, 입력 이미지 내 현재 위치의 높이 인덱스
* im\_col: int, 입력 이미지 내 현재 위치의 너비 인덱스
* col\_index: int, 출력 이미지 내 현재 위치의 인덱스
* im2col\_get\_pixel: im2col 변환시 현재 위치에서의 픽셀 값을 가져오는 함수

