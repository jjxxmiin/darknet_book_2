# col2im

* columns을 이미지로 변환해주는 것을 말합니다.

## col2im\_add\_pixel

```c
void col2im_add_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return;
    im[col + width*(row + height*channel)] += val;
}
```

함수 이름: col2im\_add\_pixel

입력:

* im: 1차원 배열 형태의 이미지 데이터
* height: 이미지 높이
* width: 이미지 너비
* channels: 이미지 채널 수
* row: 적용할 픽셀의 y 좌표
* col: 적용할 픽셀의 x 좌표
* channel: 적용할 채널 번호
* pad: 패딩 크기
* val: 추가할 값

동작:

* col2im 함수 내부에서 사용되며, 이미지의 특정 위치에 값을 더합니다.
* 주어진 row, col, channel 값을 이용해 이미지의 해당 위치에 val 값을 더합니다.
* 패딩이 적용된 이미지에서는 row, col 값에 pad 값을 뺀 위치에 값을 더합니다.
* 이미지를 1차원 배열로 저장한 경우, 해당 위치에 있는 값을 수정합니다.

설명:

* 이미지 처리에서 특정 위치에 값을 더하는 연산은 다양한 곳에서 사용됩니다.
* 이 함수는 col2im 함수 내부에서 사용되며, 패딩이 적용된 이미지의 이미지 값을 복원하는 과정에서 호출됩니다.
* 패딩이 적용된 이미지는 출력 이미지의 크기와 다르기 때문에, col2im 함수에서는 입력 이미지의 값을 출력 이미지의 각 위치에 매핑해야 합니다.
* 이를 위해 출력 이미지의 위치에 해당하는 입력 이미지의 위치를 계산하는 과정에서, 패딩이 적용된 입력 이미지의 특정 위치에 값을 더해주어야 합니다.
* 이 함수는 이미지를 1차원 배열로 저장한 경우, 주어진 row, col, channel 값을 이용해 해당 위치의 값을 수정합니다.



## col2im\_cpu

```c
//This one might be too, can't remember.
void col2im_cpu(float* data_col,
         int channels,  int height,  int width,
         int ksize,  int stride, int pad, float* data_im)
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
                double val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad, val);
            }
        }
    }
}
```

함수 이름: col2im\_cpu

입력:

* data\_col: 1D 배열 형태로 입력된 이미지 데이터 (channels\_col x height\_col x width\_col)
* channels: 이미지의 채널 수
* height: 이미지의 높이
* width: 이미지의 너비
* ksize: 커널의 크기
* stride: 스트라이드의 크기
* pad: 패딩의 크기
* data\_im: 3D 배열 형태로 변환된 이미지 데이터 (channels x height x width)

동작:

* 주어진 data\_col을 col2im 변환하여 data\_im에 저장함
* col2im 변환은 convolution 연산 시 입력 데이터를 커널의 모양에 맞게 재배열하는 과정을 말함
* col2im 변환을 통해 커널과의 행렬 곱셈 연산을 대신할 수 있어 계산 비용을 줄일 수 있음

설명:

* channels\_col은 커널의 크기와 입력 이미지의 채널 수를 곱한 값임
* height\_col과 width\_col은 커널의 크기와 스트라이드, 패딩 정보를 이용하여 계산됨
* for문을 이용하여 channels\_col, height\_col, width\_col에 대해 반복하며 col2im 변환 수행
* channels\_col을 기준으로 w\_offset과 h\_offset을 계산하여 커널 내에서의 위치 정보를 획득함
* c\_im을 계산하여 현재 위치가 입력 이미지의 어느 채널인지 판단함
* for문을 이용하여 height\_col과 width\_col에 대해 반복하며 col2im\_add\_pixel 함수를 호출하여 data\_im에 값을 저장함
* col2im\_add\_pixel 함수는 이미지를 col2im 변환한 결과인 data\_im에 값을 저장하는 함수임
* col2im\_add\_pixel 함수는 입력 이미지의 위치 정보와 값을 이용하여 data\_im에 값을 저장함

