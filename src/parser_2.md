# parser\_2

### free\_section

```c
void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}
```

함수 이름: free\_section

입력:&#x20;

* section \*s (설정 파일에서 파싱된 섹션을 나타내는 section 구조체)

동작:&#x20;

* 설정 파일에서 파싱된 섹션을 해제(free)한다.&#x20;
* 이 함수는 section 구조체와 해당 구조체의 options 리스트에 저장된 모든 kvp(key-value pair) 요소를 해제한다.

설명:

* s->type: 섹션의 이름을 저장하는 문자열을 해제한다.
* s->options: 섹션 내부의 모든 옵션(kvp) 요소들을 해제한다. 각 요소는 kvp 구조체에 저장되어 있으며, key와 value를 저장하는 문자열을 포함하고 있다.
* s->options->front: options 리스트의 처음 노드를 가리키는 포인터.
* pair->key: kvp 구조체 내부에서 저장된 key 문자열을 해제한다.
* pair: kvp 구조체를 해제한다.
* n: options 리스트에서 현재 처리 중인 노드를 가리키는 포인터.
* next: options 리스트에서 현재 노드의 다음 노드를 가리키는 포인터.
* s: 최종적으로 section 구조체를 해제한다.



### parse\_data

```c
void parse_data(char *data, float *a, int n)
{
    int i;
    if(!data) return;
    char *curr = data;

    char *next = data;
    int done = 0;
    for(i = 0; i < n && !done; ++i){
        while(*++next !='\0' && *next != ',');
        if(*next == '\0') done = 1;
        *next = '\0';
        sscanf(curr, "%g", &a[i]);
        curr = next+1;
    }
}
```

함수 이름: parse\_data

입력:

* char \*data: 분석할 데이터를 가리키는 문자열 포인터
* float \*a: 분석한 데이터를 저장할 float 배열 포인터
* int n: 분석할 데이터 개수

동작:&#x20;

* 주어진 문자열 포인터 data에서 쉼표로 구분된 각 데이터를 분석하여 float 배열 포인터 a에 저장한다.&#x20;
* n은 분석할 데이터 개수로, n보다 적은 개수의 데이터가 있을 경우 남은 배열 요소는 0으로 초기화된다.

설명:&#x20;

* parse\_data 함수는 CSV(comma-separated values) 형식의 문자열 데이터를 분석하여 float 배열로 저장하는 함수이다. 입력으로는 분석할 데이터가 담긴 문자열 포인터 data와 이를 저장할 float 배열 포인터 a, 그리고 분석할 데이터 개수를 의미하는 정수형 변수 n을 받는다.
* 함수는 문자열 포인터 curr을 처음에는 data로 초기화하고, 다음으로 분석할 데이터를 가리키는 문자열 포인터 next를 curr로 초기화한다. done 변수는 모든 데이터를 분석했는지 여부를 나타내는 변수로, 처음에는 false로 초기화한다.
* 반복문을 이용하여 n개의 데이터를 분석한다. 분석할 데이터가 더 이상 없으면 done 변수를 true로 변경하고 반복문을 빠져나온다. 현재 분석하고 있는 데이터가 끝나는 위치(next)를 찾아, 해당 위치에 널 문자('\0')를 넣어 문자열을 종료시킨다. curr에는 분석한 데이터를 저장하기 위해 해당 데이터의 시작 위치를 가리키는 next의 다음 위치를 저장한다.
* sscanf 함수를 이용하여 curr이 가리키는 위치에서부터 다음 널 문자 전까지의 문자열을 float 값으로 변환하여 a 배열에 저장한다. 분석한 데이터의 개수가 n개보다 적을 경우 남은 배열 요소는 0으로 초기화된다.



### parse\_local

```c
local_layer parse_local(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int(options, "pad",0);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before local layer must output image.");

    local_layer layer = make_local_layer(batch,h,w,c,n,size,stride,pad,activation);

    return layer;
}
```

함수 이름: parse\_local&#x20;

입력:

* options: 리스트 포인터, 레이어의 설정 정보
* params: size\_params 구조체, 레이어 입력 이미지의 크기 정보를 가지고 있음

동작:

* options에서 필터 수(n), 필터 크기(size), 스트라이드(stride), 패딩(pad), 활성화 함수(activation) 정보를 파싱하여 local\_layer 구조체를 생성하고 반환함
* batch, h, w, c는 params에서 가져옴
* h, w, c가 0이면 에러를 발생시킴

설명:

* local\_layer 구조체는 로컬 레이어를 표현하기 위한 구조체로, 필터 수(n), 필터 크기(size), 스트라이드(stride), 패딩(pad), 활성화 함수(activation) 등의 정보를 가지고 있음
* make\_local\_layer 함수를 이용하여 local\_layer 구조체를 생성함
* option\_find\_int, option\_find\_str 함수를 이용하여 options에서 필요한 정보를 파싱함
* get\_activation 함수를 이용하여 activation\_s 문자열에 해당하는 활성화 함수를 가져옴
* local\_layer 이전 레이어가 이미지를 출력하는 레이어인지 확인하기 위해 params에서 h, w, c 정보를 가져옴
* h, w, c가 0이면 이미지를 출력하는 레이어가 아니므로 에러를 발생시킴



### parse\_deconvolutional

```c
layer parse_deconvolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before deconvolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    if(pad) padding = size/2;

    layer l = make_deconvolutional_layer(batch,h,w,c,n,size,stride,padding, activation, batch_normalize, params.net->adam);

    return l;
}
```

함수 이름: parse\_deconvolutional&#x20;

입력:&#x20;

* list \*options (설정 리스트)
* size\_params params (크기 정보 구조체)&#x20;

동작:&#x20;

* 설정 리스트로부터 디컨볼루션 레이어를 파싱하여 생성한다.&#x20;

설명:

* 설정 리스트로부터 필터 수, 커널 크기, 스트라이드, 활성화 함수 등을 파싱한다.
* 크기 정보 구조체에서 입력 이미지의 높이, 너비, 채널, 배치 크기 등을 가져온다.
* 입력 이미지의 높이, 너비, 채널이 정상적으로 출력되었는지 확인한다.
* 배치 정규화, 패딩, 패딩 크기 등의 옵션을 파싱한다.
* make\_deconvolutional\_layer 함수를 호출하여 디컨볼루션 레이어를 생성하고 반환한다.



### parse\_convolutional

```c
convolutional_layer parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    int groups = option_find_int_quiet(options, "groups", 1);
    if(pad) padding = size/2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);

    convolutional_layer layer = make_convolutional_layer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, params.net->adam);
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);

    return layer;
}
```

함수 이름: parse\_convolutional

입력:

* options: 컨볼루션 레이어를 파싱하기 위한 설정 값들이 들어있는 링크드 리스트
* params: 네트워크의 사이즈 매개변수를 저장한 구조체

동작:

* 설정값들을 파싱하여 컨볼루션 레이어를 생성합니다.
* 파싱된 설정값들을 이용하여 컨볼루션 레이어의 필터 수, 필터 크기, 스트라이드, 패딩 등을 설정합니다.
* 컨볼루션 레이어에 적용할 활성화 함수를 설정합니다.
* 컨볼루션 레이어를 생성하고 반환합니다.

설명:

* n: 필터의 개수입니다.
* size: 필터의 크기입니다.
* stride: 필터의 스트라이드입니다.
* pad: 필터 패딩을 의미합니다. 이 값이 1이면 padding 값은 필터 크기의 절반으로 설정됩니다.
* padding: 패딩 값입니다.
* groups: 그룹 수입니다.
* activation\_s: 레이어의 활성화 함수입니다. "logistic"을 디폴트로 합니다.
* activation: 활성화 함수의 구조체입니다.
* batch: 현재 배치의 크기입니다.
* h, w, c: 현재 레이어의 인풋 이미지의 높이, 너비, 채널입니다.
* batch\_normalize: 배치 정규화 여부입니다.
* binary: 이진 분류 여부입니다.
* xnor: xnor 연산 여부입니다.
* layer.flipped: 필터를 뒤집는 여부입니다.
* layer.dot: dot 연산 여부입니다.





### parse\_crnn

```c
layer parse_crnn(list *options, size_params params)
{
    int output_filters = option_find_int(options, "output_filters",1);
    int hidden_filters = option_find_int(options, "hidden_filters",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_crnn_layer(params.batch, params.w, params.h, params.c, hidden_filters, output_filters, params.time_steps, activation, batch_normalize);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
}
```

함수 이름: parse\_crnn&#x20;

입력:

* options: 파싱할 옵션 리스트
* params: 크기 매개변수(size\_params) 구조체

동작:&#x20;

* CRNN(Convolutional Recurrent Neural Network) 레이어를 파싱하여 초기화하고 반환합니다.&#x20;
* 파싱할 옵션으로는 output\_filters, hidden\_filters, activation, batch\_normalize, shortcut 등이 있습니다.

설명:

* output\_filters: 출력 필터 수
* hidden\_filters: hidden state의 필터 수
* activation\_s: 활성화 함수 문자열
* activation: get\_activation 함수를 통해 가져온 활성화 함수
* batch\_normalize: 배치 정규화 여부(0 또는 1)
* l.shortcut: shortcut 연결 여부(0 또는 1)



### parse\_rnn

```c
layer parse_rnn(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_rnn_layer(params.batch, params.inputs, output, params.time_steps, activation, batch_normalize, params.net->adam);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
}
```

함수 이름: parse\_rnn&#x20;

입력:

* options: 네트워크 레이어의 설정을 담은 리스트 포인터
* params: 입력 데이터의 사이즈 정보를 담은 size\_params 구조체

동작:&#x20;

* RNN (Recurrent Neural Network) 레이어를 파싱하여 생성한다.

설명:

* RNN 레이어는 주어진 시퀀스 데이터를 처리하는 데 사용된다.
* output: 출력 벡터의 차원 수
* activation: 활성화 함수 (default: logistic)
* batch\_normalize: 배치 정규화 사용 여부 (0 또는 1)
* shortcut: RNN 레이어에서 shortcut을 사용할지 여부 (0 또는 1)
* time\_steps: 시퀀스 데이터의 길이 (타임 스텝 수)
* inputs: RNN 레이어의 입력 벡터 차원 수 (이전 타임 스텝의 출력 벡터 차원 수)



### parse\_gru

```c
layer parse_gru(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_gru_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->adam);
    l.tanh = option_find_int_quiet(options, "tanh", 0);

    return l;
}
```

함수 이름: parse\_gru&#x20;

입력:&#x20;

* options (옵션 리스트)
* params (크기 파라미터)&#x20;

동작:&#x20;

* GRU 레이어를 파싱하여 생성하고 반환&#x20;

설명:

* options: GRU 레이어에 대한 옵션들을 담은 리스트
* params: 레이어의 크기 파라미터(batch, input size, time steps, adam optimizer)
* output: 출력 크기(output size)
* batch\_normalize: 배치 정규화 여부(1 또는 0)
* tanh: tanh 활성화 함수 사용 여부(1 또는 0)
* l: 생성된 GRU 레이어



### parse\_lstm

```c
layer parse_lstm(list *options, size_params params)
{
    int output = option_find_int(options, "output", 1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_lstm_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->adam);

    return l;
}
```

함수 이름: parse\_lstm&#x20;

입력:&#x20;

* options(옵션 리스트)
* params(크기 매개변수)&#x20;

동작:&#x20;

* LSTM 레이어를 파싱하여 생성한 후 반환합니다.&#x20;

설명:

* output: 출력 차원 수를 나타내는 정수 값
* batch\_normalize: 배치 정규화를 사용할지 여부를 나타내는 0 또는 1의 정수 값
* batch: 배치 크기를 나타내는 정수 값
* inputs: 입력 차원 수를 나타내는 정수 값
* time\_steps: 시간 단계 수를 나타내는 정수 값
* adam: ADAM 최적화 알고리즘을 사용할지 여부를 나타내는 0 또는 1의 정수 값 반환: LSTM 레이어



### parse\_connected

```c
layer parse_connected(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_connected_layer(params.batch, params.inputs, output, activation, batch_normalize, params.net->adam);
    return l;
}
```

함수 이름: parse\_connected&#x20;

입력:&#x20;

* options (list 포인터) 파싱된 옵션 리스트
* params: (size\_params) 네트워크 입력 크기

동작:&#x20;

* 입력으로 받은 옵션 리스트에서 "output", "activation", "batch\_normalize" 등의 옵션 값을 파싱하여, make\_connected\_layer 함수를 사용하여 연결층(layer)을 생성하고, 설정된 옵션 값을 적용하여 반환한다.

설명:

* output: 출력 뉴런의 개수를 정의하는 int 형태의 옵션 값
* activation\_s: 활성화 함수를 정의하는 문자열 형태의 옵션 값. "logistic", "relu", "leaky", "linear" 중 하나를 선택할 수 있다.
* activation: activation\_s에 해당하는 활성화 함수를 저장하는 ACTIVATION 타입 변수
* batch\_normalize: 배치 정규화(batch normalization)를 적용할지 여부를 정의하는 int 형태의 옵션 값. 적용할 경우 1, 적용하지 않을 경우 0으로 설정한다.
* l: make\_connected\_layer 함수를 사용하여 생성된 연결층(layer)을 저장하는 layer 타입 변수. 설정된 옵션 값들이 적용된 상태이다.



### parse\_softmax

```c
layer parse_softmax(list *options, size_params params)
{
    int groups = option_find_int_quiet(options, "groups",1);
    layer l = make_softmax_layer(params.batch, params.inputs, groups);
    l.temperature = option_find_float_quiet(options, "temperature", 1);
    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) l.softmax_tree = read_tree(tree_file);
    l.w = params.w;
    l.h = params.h;
    l.c = params.c;
    l.spatial = option_find_float_quiet(options, "spatial", 0);
    l.noloss =  option_find_int_quiet(options, "noloss", 0);
    return l;
}
```

함수 이름: parse\_softmax&#x20;

입력:&#x20;

* options (옵션 리스트)
* params (크기 매개변수 구조체)&#x20;

동작:&#x20;

* 옵션 리스트에서 해당 옵션 값들을 파싱하여 softmax 레이어를 생성하고 반환한다.&#x20;
* 그룹 수, 온도, 트리 파일, 공간, 무손실 등의 옵션을 지원한다.&#x20;

설명:&#x20;

* 파싱된 값을 기반으로 softmax 레이어를 생성하고 반환한다.&#x20;
* 소프트맥스 트리를 사용하는 경우 트리 파일을 읽어들인다.&#x20;
* noloss 옵션은 softmax 손실 계산에 영향을 미치며, spatial 옵션은 출력 크기에 대한 공간 가중치를 적용한다.



### parse\_yolo\_mask

```c
int *parse_yolo_mask(char *a, int *num)
{
    int *mask = 0;
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',')+1;
        }
        *num = n;
    }
    return mask;
}
```

함수 이름: parse\_yolo\_mask&#x20;

입력:

* char\* a: YOLO 마스크 문자열
* int\* num: 마스크 배열의 요소 수를 저장할 포인터

동작:

* YOLO 마스크 문자열을 구문 분석하여 정수 배열을 생성하고 반환
* 문자열에서 구분 기호(쉼표)를 사용하여 정수 배열의 각 요소를 구성
* num 포인터를 통해 생성된 배열의 요소 수를 반환

설명:

* 이 함수는 YOLO 마스크 문자열을 입력 받아 해당 문자열에서 구분 기호(쉼표)를 사용하여 정수 배열을 생성하고 반환합니다.
* 이 함수는 문자열을 구문 분석하여 배열의 각 요소를 구성하기 위해 atoi 함수를 사용합니다.
* 문자열에서 구분 기호를 사용하여 각 요소를 구분합니다.
* 함수는 생성된 배열의 요소 수를 num 포인터를 통해 반환합니다.



### parse\_yolo

```c
layer parse_yolo(list *options, size_params params)
{
    int classes = option_find_int(options, "classes", 20);
    int total = option_find_int(options, "num", 1);
    int num = total;

    char *a = option_find_str(options, "mask", 0);
    int *mask = parse_yolo_mask(a, &num);
    layer l = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes);
    assert(l.outputs == params.inputs);

    l.max_boxes = option_find_int_quiet(options, "max",90);
    l.jitter = option_find_float(options, "jitter", .2);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
    l.random = option_find_int_quiet(options, "random", 0);

    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    a = option_find_str(options, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}
```

함수 이름: parse\_yolo&#x20;

입력:&#x20;

* options (list 포인터)
* params (size\_params 타입)&#x20;

동작:&#x20;

* YOLO 레이어를 파싱하여 레이어를 만들고 반환합니다.&#x20;

설명:

* classes (int 타입): 클래스 수를 나타냅니다.
* total (int 타입): Anchor Box의 수를 나타냅니다.
* num (int 타입): Anchor Box의 수를 나타냅니다.
* a (char 포인터): Mask 값으로, Anchor Box를 적용할 레이어를 선택합니다.
* mask (int 포인터): Mask 값으로 선택된 Anchor Box를 저장합니다.
* l (layer 타입): 만들어진 YOLO 레이어입니다.
* max\_boxes (int 타입): 최대 박스 수를 나타냅니다.
* jitter (float 타입): Jitter 값으로, 이미지의 위치를 무작위로 이동시킵니다.
* ignore\_thresh (float 타입): Ignore Threshold 값으로, 박스와의 IoU 값이 이 값보다 작으면 무시합니다.
* truth\_thresh (float 타입): Truth Threshold 값으로, ground truth와 예측 값의 IoU 값이 이 값보다 크면 해당 예측 값을 ground truth로 취급합니다.
* random (int 타입): Random 값으로, 이미지를 무작위로 변환할 때 사용하는 시드(seed) 값입니다.
* map\_file (char 포인터): Map 파일로, 클래스 이름을 저장한 파일입니다.
* a (char 포인터): Anchor 값을 저장한 문자열입니다.
* biases (float 배열): Anchor 값을 저장한 배열입니다.



### parse\_iseg

```c
layer parse_iseg(list *options, size_params params)
{
    int classes = option_find_int(options, "classes", 20);
    int ids = option_find_int(options, "ids", 32);
    layer l = make_iseg_layer(params.batch, params.w, params.h, classes, ids);
    assert(l.outputs == params.inputs);
    return l;
}
```

함수 이름: parse\_iseg

입력:

* list \*options: 옵션 리스트 포인터
* size\_params params: 모델 사이즈 매개변수

동작:

* 주어진 옵션과 모델 사이즈 매개변수를 이용해 iseg(layered semantic segmentation) 레이어를 생성하고 반환한다.
* "classes" 옵션을 이용해 클래스 개수를 설정한다.
* "ids" 옵션을 이용해 id 개수를 설정한다.
* 레이어의 출력이 모델 입력과 일치하는지 확인한다.

설명:

* iseg 레이어는 이미지를 입력으로 받아 각 픽셀을 클래스별로 분류하는 모델이다.
* classes: 분류할 클래스 개수
* ids: 각 클래스별로 할당할 고유한 id 개수



### parse\_region

```c
layer parse_region(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 4);
    int classes = option_find_int(options, "classes", 20);
    int num = option_find_int(options, "num", 1);

    layer l = make_region_layer(params.batch, params.w, params.h, num, classes, coords);
    assert(l.outputs == params.inputs);

    l.log = option_find_int_quiet(options, "log", 0);
    l.sqrt = option_find_int_quiet(options, "sqrt", 0);

    l.softmax = option_find_int(options, "softmax", 0);
    l.background = option_find_int_quiet(options, "background", 0);
    l.max_boxes = option_find_int_quiet(options, "max",30);
    l.jitter = option_find_float(options, "jitter", .2);
    l.rescore = option_find_int_quiet(options, "rescore",0);

    l.thresh = option_find_float(options, "thresh", .5);
    l.classfix = option_find_int_quiet(options, "classfix", 0);
    l.absolute = option_find_int_quiet(options, "absolute", 0);
    l.random = option_find_int_quiet(options, "random", 0);

    l.coord_scale = option_find_float(options, "coord_scale", 1);
    l.object_scale = option_find_float(options, "object_scale", 1);
    l.noobject_scale = option_find_float(options, "noobject_scale", 1);
    l.mask_scale = option_find_float(options, "mask_scale", 1);
    l.class_scale = option_find_float(options, "class_scale", 1);
    l.bias_match = option_find_int_quiet(options, "bias_match",0);

    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) l.softmax_tree = read_tree(tree_file);
    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    char *a = option_find_str(options, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}
```

함수 이름: parse\_region

입력:

* options: YOLO의 설정값을 담은 리스트
* params: 네트워크의 입력 이미지 크기 정보를 담은 size\_params 구조체

동작:&#x20;

* YOLO 네트워크에서 사용되는 region 레이어를 파싱하여 생성하는 함수

설명:

* coords: 각 bounding box의 좌표 정보 개수 (보통 4)
* classes: 인식하려는 클래스 수
* num: 각 grid cell마다 예측하려는 bounding box 개수
* log: bounding box 좌표값에 로그 함수를 적용할지 여부
* sqrt: bounding box 좌표값에 제곱근 함수를 적용할지 여부
* softmax: 소프트맥스 함수 사용 여부
* background: 배경 클래스 여부
* max\_boxes: 예측할 최대 bounding box 개수
* jitter: 이미지에 랜덤한 조작을 가하는 정도
* rescore: 예측한 bounding box를 재점수화할지 여부
* thresh: bounding box 예측 결과 중 confidence가 이 값보다 낮은 경우 제거
* classfix: 클래스 인덱스 보정값
* absolute: bounding box 좌표값을 0\~1이 아닌 절대값으로 지정할지 여부
* random: bounding box 좌표값에 랜덤한 값을 더할지 여부
* coord\_scale: bounding box 좌표값의 scale
* object\_scale: object가 있을 때의 confidence score에 대한 scale
* noobject\_scale: object가 없을 때의 confidence score에 대한 scale
* mask\_scale: mask 값에 대한 scale
* class\_scale: class score에 대한 scale
* bias\_match: bias 값을 일치시키는지 여부
* tree\_file: 클래스 트리 파일 경로
* map\_file: 클래스 이름-인덱스 매핑 파일 경로
* anchors: 각 bounding box에 사용되는 anchor box 값
* 반환값: 생성된 region 레이어



### parse\_detection

```c
detection_layer parse_detection(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 1);
    int classes = option_find_int(options, "classes", 1);
    int rescore = option_find_int(options, "rescore", 0);
    int num = option_find_int(options, "num", 1);
    int side = option_find_int(options, "side", 7);
    detection_layer layer = make_detection_layer(params.batch, params.inputs, num, side, classes, coords, rescore);

    layer.softmax = option_find_int(options, "softmax", 0);
    layer.sqrt = option_find_int(options, "sqrt", 0);

    layer.max_boxes = option_find_int_quiet(options, "max",90);
    layer.coord_scale = option_find_float(options, "coord_scale", 1);
    layer.forced = option_find_int(options, "forced", 0);
    layer.object_scale = option_find_float(options, "object_scale", 1);
    layer.noobject_scale = option_find_float(options, "noobject_scale", 1);
    layer.class_scale = option_find_float(options, "class_scale", 1);
    layer.jitter = option_find_float(options, "jitter", .2);
    layer.random = option_find_int_quiet(options, "random", 0);
    layer.reorg = option_find_int_quiet(options, "reorg", 0);
    return layer;
}
```

함수 이름: parse\_detection&#x20;

입력:&#x20;

* options (list 포인터)
* params (size\_params 구조체)&#x20;

동작:&#x20;

* detection\_layer를 파싱하고 초기화하며, 파싱된 값들을 detection\_layer에 저장하여 반환한다.&#x20;

설명:

* coords: 각 bounding box에 대한 x, y, w, h값의 개수 (int)
* classes: 분류할 클래스의 수 (int)
* rescore: YOLOv2에서 사용되었던 값으로, 객체에 대한 예측 신뢰도(rescore)를 사용할지 여부를 결정 (int)
* num: 각 층에서 예측하는 bounding box의 수 (int)
* side: 층의 너비 또는 높이 (int)
* layer.softmax: softmax 함수를 사용할지 여부 (int)
* layer.sqrt: sqrt 함수를 사용할지 여부 (int)
* layer.max\_boxes: 예측할 bounding box의 최대 개수 (int)
* layer.coord\_scale: 좌표에 대한 가중치 값 (float)
* layer.forced: 강제 예측 수행 여부 (int)
* layer.object\_scale: 객체에 대한 가중치 값 (float)
* layer.noobject\_scale: 객체가 없는 영역에 대한 가중치 값 (float)
* layer.class\_scale: 클래스에 대한 가중치 값 (float)
* layer.jitter: 이미지 jittering 적용 여부 (float)
* layer.random: 이미지를 무작위로 변환할지 여부 (int)
* layer.reorg: reorg 레이어 사용 여부 (int)
* 반환값: detection\_layer



### parse\_cost

```c
cost_layer parse_cost(list *options, size_params params)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    float scale = option_find_float_quiet(options, "scale",1);
    cost_layer layer = make_cost_layer(params.batch, params.inputs, type, scale);
    layer.ratio =  option_find_float_quiet(options, "ratio",0);
    layer.noobject_scale =  option_find_float_quiet(options, "noobj", 1);
    layer.thresh =  option_find_float_quiet(options, "thresh",0);
    return layer;
}
```

함수 이름: parse\_cost

입력:&#x20;

* list \*options
* size\_params params

동작:&#x20;

* cost\_layer를 파싱하여 생성하고 반환함.

설명:

* type\_s: cost\_layer의 type을 설정하며, "type" option을 통해 string으로 입력 받음.
* type: type\_s를 바탕으로, COST\_TYPE enum 값으로 변환하여 cost\_layer의 type을 설정함.
* scale: cost\_layer의 scale 값을 설정함. "scale" option을 통해 float 값으로 입력 받음.
* layer: make\_cost\_layer() 함수를 호출하여 cost\_layer를 생성함.
* ratio: cost\_layer의 ratio 값을 설정함. "ratio" option을 통해 float 값으로 입력 받음.
* noobject\_scale: cost\_layer의 noobject\_scale 값을 설정함. "noobj" option을 통해 float 값으로 입력 받음.
* thresh: cost\_layer의 thresh 값을 설정함. "thresh" option을 통해 float 값으로 입력 받음.
* 반환값: 생성된 cost\_layer를 반환함.



### parse\_crop

```c
crop_layer parse_crop(list *options, size_params params)
{
    int crop_height = option_find_int(options, "crop_height",1);
    int crop_width = option_find_int(options, "crop_width",1);
    int flip = option_find_int(options, "flip",0);
    float angle = option_find_float(options, "angle",0);
    float saturation = option_find_float(options, "saturation",1);
    float exposure = option_find_float(options, "exposure",1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before crop layer must output image.");

    int noadjust = option_find_int_quiet(options, "noadjust",0);

    crop_layer l = make_crop_layer(batch,h,w,c,crop_height,crop_width,flip, angle, saturation, exposure);
    l.shift = option_find_float(options, "shift", 0);
    l.noadjust = noadjust;
    return l;
}
```

함수 이름: parse\_crop&#x20;

입력:&#x20;

* options(list 포인터)
* params(size\_params 구조체)&#x20;

동작:&#x20;

* options에서 crop layer의 옵션값을 파싱하여 crop\_layer 구조체를 생성하고 반환한다.&#x20;

설명:

* crop\_height(int): crop할 영상의 높이
* crop\_width(int): crop할 영상의 너비
* flip(int): crop 영상을 수평으로 뒤집을 것인지 여부 (0: 안 뒤집음, 1: 뒤집음)
* angle(float): crop 영상 회전 각도 (라디안 단위)
* saturation(float): crop 영상의 채도 조절 (1이면 변경 없음)
* exposure(float): crop 영상의 노출 조절 (1이면 변경 없음)
* noadjust(int): crop 영상의 라벨과 경계상자를 자동으로 조절하지 않음 (0: 자동 조절, 1: 자동 조절 안 함)
* shift(float): crop 영상의 RGB 색상 채널 값에 더해지는 값 (RGB 값이 0\~1 범위를 벗어나지 않도록 제한해야 함)



### parse\_reorg

```c
layer parse_reorg(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int reverse = option_find_int_quiet(options, "reverse",0);
    int flatten = option_find_int_quiet(options, "flatten",0);
    int extra = option_find_int_quiet(options, "extra",0);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before reorg layer must output image.");

    layer layer = make_reorg_layer(batch,w,h,c,stride,reverse, flatten, extra);
    return layer;
}
```

함수 이름: parse\_reorg&#x20;

입력:&#x20;

* options (list 포인터)
* params (size\_params 구조체)&#x20;

동작:&#x20;

* options에서 "stride", "reverse", "flatten", "extra"에 해당하는 정수 값을 찾아서, make\_reorg\_layer 함수를 사용하여 reorg layer를 만들고 반환합니다.&#x20;

설명:&#x20;

* 입력으로 options와 params를 받아, reorg layer를 생성하여 반환하는 함수입니다.&#x20;
* options에서는 stride, reverse, flatten, extra에 대한 값을 찾아서 해당 값들을 이용하여 reorg layer를 생성합니다.&#x20;
* params에서는 입력 이미지의 높이, 너비, 채널 수, 배치 크기를 받아옵니다.



### parse\_maxpool

```c
maxpool_layer parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int_quiet(options, "padding", size-1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before maxpool layer must output image.");

    maxpool_layer layer = make_maxpool_layer(batch,h,w,c,size,stride,padding);
    return layer;
}
```

함수 이름: parse\_maxpool&#x20;

입력:&#x20;

* options (list 포인터): 파싱할 maxpool 레이어의 옵션 리스트&#x20;
* params (size\_params): maxpool 레이어의 입력 이미지 크기 및 배치 크기 정보가 담긴 size\_params 구조체&#x20;

동작:&#x20;

* 파싱된 옵션을 기반으로 maxpool 레이어를 생성하고 반환함&#x20;

설명:&#x20;

* 입력 이미지에 대한 maxpool 연산을 수행하는 레이어를 생성하고 반환하는 함수입니다.&#x20;
* 옵션 리스트에서 stride, size, padding 값을 읽어와 maxpool 레이어를 생성합니다.&#x20;
* 입력 이미지 크기와 배치 크기 정보를 포함하는 size\_params 구조체를 사용합니다.&#x20;
* 반환값은 maxpool\_layer 구조체입니다.



### parse\_avgpool

```c
avgpool_layer parse_avgpool(list *options, size_params params)
{
    int batch,w,h,c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before avgpool layer must output image.");

    avgpool_layer layer = make_avgpool_layer(batch,w,h,c);
    return layer;
}
```

함수 이름: parse\_avgpool&#x20;

입력:

* options (list): 옵션 리스트
* params (size\_params): 사이즈 파라미터 (w, h, c, batch)

동작:

* 입력으로 받은 사이즈 파라미터를 이용하여 avgpool\_layer를 생성하고 반환한다.

설명:

* 이 함수는 options와 params를 입력으로 받아 avgpool\_layer를 생성하는 함수이다.
* 함수 내부에서는 사이즈 파라미터를 이용하여 avgpool\_layer를 만들고, 이를 반환한다.
* 만약 h, w, c가 0이라면 "Layer before avgpool layer must output image." 에러를 발생시킨다.
* 생성된 avgpool\_layer는 이후의 계산에서 사용된다.



### parse\_dropout

```c
dropout_layer parse_dropout(list *options, size_params params)
{
    float probability = option_find_float(options, "probability", .5);
    dropout_layer layer = make_dropout_layer(params.batch, params.inputs, probability);
    layer.out_w = params.w;
    layer.out_h = params.h;
    layer.out_c = params.c;
    return layer;
}
```

함수 이름: parse\_dropout

입력:

* options (list): 옵션 리스트
* params (size\_params): 사이즈 파라미터 (batch, inputs, w, h, c)

동작:

* options에서 "probability"를 찾아 확률을 설정하여 dropout\_layer를 생성하고 반환한다.

설명:

* 이 함수는 options와 params를 입력으로 받아 dropout\_layer를 생성하는 함수이다.
* options에서 "probability" 값을 찾아 이를 이용하여 dropout\_layer를 생성한다.
* 생성된 dropout\_layer의 out\_w, out\_h, out\_c는 params의 w, h, c 값을 가진다.
* 생성된 dropout\_layer는 이후의 계산에서 사용된다.



### parse\_normalization

```c
layer parse_normalization(list *options, size_params params)
{
    float alpha = option_find_float(options, "alpha", .0001);
    float beta =  option_find_float(options, "beta" , .75);
    float kappa = option_find_float(options, "kappa", 1);
    int size = option_find_int(options, "size", 5);
    layer l = make_normalization_layer(params.batch, params.w, params.h, params.c, size, alpha, beta, kappa);
    return l;
}
```

함수 이름: parse\_normalization

입력:

* options (list): 옵션 리스트
* params (size\_params): 사이즈 파라미터 (batch, w, h, c)

동작:

* options에서 alpha, beta, kappa, size 값을 찾아 정규화(normalization) 레이어를 생성하고 반환한다.

설명:

* 이 함수는 options와 params를 입력으로 받아 정규화 레이어를 생성하는 함수이다.
* options에서 "alpha", "beta", "kappa", "size" 값을 찾아 이를 이용하여 정규화 레이어를 생성한다.
* 생성된 정규화 레이어는 이후의 계산에서 사용된다.



### parse\_batchnorm

```c
layer parse_batchnorm(list *options, size_params params)
{
    layer l = make_batchnorm_layer(params.batch, params.w, params.h, params.c);
    return l;
}
```

함수 이름: parse\_batchnorm

입력:

* options (list): 옵션 리스트
* params (size\_params): 사이즈 파라미터 (batch, w, h, c)

동작:

* batchnorm\_layer를 생성하고 반환한다.

설명:

* 이 함수는 options와 params를 입력으로 받아 batchnorm\_layer를 생성하는 함수이다.
* make\_batchnorm\_layer() 함수를 이용하여 batchnorm\_layer를 생성하고 이를 반환한다.
* 생성된 batchnorm\_layer는 이후의 계산에서 사용된다.



### parse\_shortcut

```c
layer parse_shortcut(list *options, size_params params, network *net)
{
    char *l = option_find(options, "from");
    int index = atoi(l);
    if(index < 0) index = params.index + index;

    int batch = params.batch;
    layer from = net->layers[index];

    layer s = make_shortcut_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);

    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    s.activation = activation;
    s.alpha = option_find_float_quiet(options, "alpha", 1);
    s.beta = option_find_float_quiet(options, "beta", 1);
    return s;
}
```

함수 이름: parse\_shortcut

입력:

* options (list): 옵션 리스트
* params (size\_params): 사이즈 파라미터 (batch, w, h, c)
* net (network): 네트워크

동작:

* options에서 "from" 값을 찾아 해당 인덱스의 레이어와 shortcut\_layer를 생성하고 반환한다.

설명:

* 이 함수는 options와 params, net을 입력으로 받아 shortcut\_layer를 생성하는 함수이다.
* options에서 "from" 값을 찾아 해당 인덱스의 레이어를 가져온다.
* 가져온 레이어와 make\_shortcut\_layer() 함수를 이용하여 shortcut\_layer를 생성하고 이를 반환한다.
* 생성된 shortcut\_layer는 이후의 계산에서 사용된다.



### parse\_l2norm

```c
layer parse_l2norm(list *options, size_params params)
{
    layer l = make_l2norm_layer(params.batch, params.inputs);
    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;
    return l;
}
```

함수 이름: parse\_l2norm

입력:

* options (list): 옵션 리스트
* params (size\_params): 사이즈 파라미터 (batch, w, h, c)

동작:

* make\_l2norm\_layer() 함수를 호출하여 l2norm\_layer를 생성하고 이를 반환한다.

설명:

* 이 함수는 options와 params를 입력으로 받아 l2norm\_layer를 생성하는 함수이다.
* make\_l2norm\_layer() 함수를 호출하여 l2norm\_layer를 생성하고, 이후에는 해당 layer의 크기를 params의 크기로 설정한 후 이를 반환한다.
* 생성된 l2norm\_layer는 이후의 계산에서 사용된다.



### parse\_logistic

```c
layer parse_logistic(list *options, size_params params)
{
    layer l = make_logistic_layer(params.batch, params.inputs);
    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;
    return l;
}
```

함수 이름: parse\_logistic

입력:

* options (list): 옵션 리스트
* params (size\_params): 사이즈 파라미터 (batch, w, h, c)

동작:

* make\_logistic\_layer() 함수를 호출하여 logistic\_layer를 생성하고 이를 반환한다.

설명:

* 이 함수는 options와 params를 입력으로 받아 logistic\_layer를 생성하는 함수이다.
* make\_logistic\_layer() 함수를 호출하여 logistic\_layer를 생성하고, 이후에는 해당 layer의 크기를 params의 크기로 설정한 후 이를 반환한다.
* 생성된 logistic\_layer는 이후의 계산에서 사용된다.



### parse\_activation

```c
layer parse_activation(list *options, size_params params)
{
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);

    layer l = make_activation_layer(params.batch, params.inputs, activation);

    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;

    return l;
}
```

함수 이름: parse\_activation&#x20;

입력:&#x20;

* options (list 포인터)
* params (size\_params 구조체)&#x20;

동작:&#x20;

* 주어진 options에서 "activation" 옵션을 찾아 해당 활성화 함수를 가져와 make\_activation\_layer 함수를 사용하여 새로운 활성화 레이어를 만들고, 이를 반환합니다.&#x20;

설명:

* options: 이 함수의 매개 변수로 주어진 옵션 리스트는 해당 레이어의 옵션 값들을 포함합니다.
* activation\_s: "activation" 옵션의 값을 가져옵니다. 만약 "activation" 옵션이 주어지지 않은 경우 "linear"로 초기화합니다.
* activation: activation\_s 문자열을 이용하여 해당 활성화 함수를 가져옵니다.
* make\_activation\_layer 함수를 사용하여 params.batch, params.inputs, activation을 사용하여 새로운 활성화 레이어를 만듭니다.
* 마지막으로, 새로운 레이어의 출력 크기를 params와 동일하게 설정하고 반환합니다.



### parse\_upsample

```c
layer parse_upsample(list *options, size_params params, network *net)
{

    int stride = option_find_int(options, "stride",2);
    layer l = make_upsample_layer(params.batch, params.w, params.h, params.c, stride);
    l.scale = option_find_float_quiet(options, "scale", 1);
    return l;
}
```

함수 이름: parse\_upsample

입력:

* options: 사용자가 정의한 옵션을 저장한 리스트 포인터
* params: 입력 레이어의 파라미터를 저장한 size\_params 구조체
* net: 네트워크 구조체

동작:

* 옵션에서 stride 값을 읽어서 업샘플링 레이어를 생성한다.
* 업샘플링 레이어의 scale 값을 설정한다.

설명:

* parse\_upsample 함수는 옵션에서 지정한 stride 값을 사용하여 입력 레이어를 업샘플링하는 레이어를 생성하고 반환하는 함수이다.
* options 리스트는 사용자가 지정한 옵션들을 저장하고 있으며, params는 입력 레이어의 파라미터를 저장한 구조체이다.
* net은 전체 네트워크 구조체를 가리키는 포인터이다.
* 업샘플링 레이어를 생성하기 위해 make\_upsample\_layer 함수를 사용한다. 이때, params에서 읽어온 입력 레이어의 파라미터와 사용자가 지정한 stride 값을 사용한다.
* l.scale은 사용자가 지정한 scale 값을 저장한다. 만약, 옵션에서 scale 값을 지정하지 않았으면 1.0으로 초기화된다.
* 생성된 업샘플링 레이어를 반환한다.



### parse\_route

```c
route_layer parse_route(list *options, size_params params, network *net)
{
    char *l = option_find(options, "layers");
    int len = strlen(l);
    if(!l) error("Route Layer must specify input layers");
    int n = 1;
    int i;
    for(i = 0; i < len; ++i){
        if (l[i] == ',') ++n;
    }

    int *layers = calloc(n, sizeof(int));
    int *sizes = calloc(n, sizeof(int));
    for(i = 0; i < n; ++i){
        int index = atoi(l);
        l = strchr(l, ',')+1;
        if(index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = net->layers[index].outputs;
    }
    int batch = params.batch;

    route_layer layer = make_route_layer(batch, n, layers, sizes);

    convolutional_layer first = net->layers[layers[0]];
    layer.out_w = first.out_w;
    layer.out_h = first.out_h;
    layer.out_c = first.out_c;
    for(i = 1; i < n; ++i){
        int index = layers[i];
        convolutional_layer next = net->layers[index];
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            layer.out_c += next.out_c;
        }else{
            layer.out_h = layer.out_w = layer.out_c = 0;
        }
    }

    return layer;
}
```

함수 이름: parse\_route&#x20;

입력:

* options: 파라미터 설정 정보를 담은 리스트
* params: size\_params 구조체 변수
* net: 네트워크 구조체 변수&#x20;

동작:&#x20;

* 라우팅 레이어를 파싱하여 레이어를 생성하고 반환한다.&#x20;

설명:

* "layers" 파라미터를 이용해 입력 레이어를 결정한다.
* 입력 레이어의 인덱스와 출력 크기를 저장하고, 이 정보를 이용하여 라우팅 레이어를 생성한다.
* 입력 레이어들 중 가장 처음 입력 레이어의 출력 크기로 라우팅 레이어의 출력 크기를 설정한다.
* 나머지 입력 레이어들의 출력 크기가 첫 번째 입력 레이어의 출력 크기와 일치하면, 출력 채널 수를 추가한다. 만약 일치하지 않으면, 출력 크기를 0으로 설정한다.
* 생성한 라우팅 레이어를 반환한다.



### get\_policy

```c
typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, "random")==0) return RANDOM;
    if (strcmp(s, "poly")==0) return POLY;
    if (strcmp(s, "constant")==0) return CONSTANT;
    if (strcmp(s, "step")==0) return STEP;
    if (strcmp(s, "exp")==0) return EXP;
    if (strcmp(s, "sigmoid")==0) return SIG;
    if (strcmp(s, "steps")==0) return STEPS;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return CONSTANT;
}
```

함수 이름: get\_policy

입력:&#x20;

* char\* s (문자열 포인터)

동작:&#x20;

* 주어진 문자열 s가 나타내는 학습률 스케줄링 정책을 반환함

설명:&#x20;

* 이 함수는 learning\_rate\_policy 열거형 타입을 사용하여 주어진 문자열 s가 나타내는 학습률 스케줄링 정책을 반환합니다.&#x20;
* 문자열 s가 "random", "poly", "constant", "step", "exp", "sigmoid", "steps" 중 하나인 경우 해당 정책을 반환하고, 그렇지 않으면 "Couldn't find policy %s, going with constant" 메시지를 출력하고 기본값으로 CONSTANT를 반환합니다.



### is\_network

```c
int is_network(section *s)
{
    return (strcmp(s->type, "[net]")==0 || strcmp(s->type, "[network]")==0);
}
```

함수 이름: is\_network&#x20;

입력:&#x20;

* section \*s (구조체 포인터)&#x20;

동작:&#x20;

* 주어진 섹션(s)이 네트워크를 나타내는 섹션인지 여부를 반환합니다.&#x20;

설명:&#x20;

* 입력된 섹션 구조체 포인터가 "\[net]" 또는 "\[network]" 문자열과 같은지를 확인하여 해당 섹션이 네트워크를 나타내는 섹션인지 여부를 반환합니다.&#x20;
* 1이 반환되면 네트워크를 나타내는 섹션입니다. 0이 반환되면 그렇지 않은 것입니다.



### load\_weights

```c
void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, 0, net->n);
}
```

```c
void load_weights_upto(network *net, char *filename, int start, int cutoff)
{
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
        fread(net->seen, sizeof(size_t), 1, fp);
    } else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }
    int transpose = (major > 1000) || (minor > 1000);

    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            load_convolutional_weights(l, fp);
        }
        if(l.type == CONNECTED){
            load_connected_weights(l, fp, transpose);
        }
        if(l.type == BATCHNORM){
            load_batchnorm_weights(l, fp);
        }
        if(l.type == CRNN){
            load_convolutional_weights(*(l.input_layer), fp);
            load_convolutional_weights(*(l.self_layer), fp);
            load_convolutional_weights(*(l.output_layer), fp);
        }
        if(l.type == RNN){
            load_connected_weights(*(l.input_layer), fp, transpose);
            load_connected_weights(*(l.self_layer), fp, transpose);
            load_connected_weights(*(l.output_layer), fp, transpose);
        }
        if (l.type == LSTM) {
            load_connected_weights(*(l.wi), fp, transpose);
            load_connected_weights(*(l.wf), fp, transpose);
            load_connected_weights(*(l.wo), fp, transpose);
            load_connected_weights(*(l.wg), fp, transpose);
            load_connected_weights(*(l.ui), fp, transpose);
            load_connected_weights(*(l.uf), fp, transpose);
            load_connected_weights(*(l.uo), fp, transpose);
            load_connected_weights(*(l.ug), fp, transpose);
        }
        if (l.type == GRU) {
            if(1){
                load_connected_weights(*(l.wz), fp, transpose);
                load_connected_weights(*(l.wr), fp, transpose);
                load_connected_weights(*(l.wh), fp, transpose);
                load_connected_weights(*(l.uz), fp, transpose);
                load_connected_weights(*(l.ur), fp, transpose);
                load_connected_weights(*(l.uh), fp, transpose);
            }else{
                load_connected_weights(*(l.reset_layer), fp, transpose);
                load_connected_weights(*(l.update_layer), fp, transpose);
                load_connected_weights(*(l.state_layer), fp, transpose);
            }
        }
        if(l.type == LOCAL){
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fread(l.biases, sizeof(float), l.outputs, fp);
            fread(l.weights, sizeof(float), size, fp);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}
```

함수 이름: load\_weights\_upto

입력:

* network \*net: 뉴럴 네트워크
* char \*filename: 가중치 파일 이름
* int start: 시작 레이어 인덱스
* int cutoff: 로드를 중지할 레이어 인덱스

동작:

* 지정된 가중치 파일에서 가중치를 로드하여 뉴럴 네트워크에 적용하는 함수
* 로드를 시작하는 레이어 인덱스와 로드를 중지하는 레이어 인덱스를 인자로 받음
* 뉴럴 네트워크에서 로드를 건너뛸 레이어에 대한 처리도 수행함

설명:

* 지정된 가중치 파일을 열고, 파일이 없으면 file\_error 함수를 호출하여 오류 메시지 출력
* 가중치 파일의 형식에 따라 파일에서 major, minor, revision 값을 읽어 들임
* major와 minor 값이 2 이상이면 net->seen 값을 파일에서 읽어 들임
* major와 minor 값이 2 미만이면 iseen 값을 파일에서 읽어들여 \*net->seen에 저장함
* major 값이 1000 이상이거나 minor 값이 1000 이상이면 전치(transpose) 플래그를 설정함
* start부터 cutoff까지의 레이어에 대해, 각 레이어의 타입에 따라 가중치를 로드함
* 로드를 건너뛰어야 하는 레이어의 경우, dontload 플래그가 true이므로 continue문을 통해 건너뛰어 처리함
* CONVOLUTIONAL 또는 DECONVOLUTIONAL 레이어의 경우 load\_convolutional\_weights 함수 호출
* CONNECTED 레이어의 경우 load\_connected\_weights 함수 호출
* BATCHNORM 레이어의 경우 load\_batchnorm\_weights 함수 호출
* CRNN, RNN, LSTM, GRU 레이어의 경우 각각 해당하는 가중치를 로드함
* LOCAL 레이어의 경우 biases와 weights를 파일에서 읽어 들임
* 로드가 끝나면 파일을 닫고 "Done!" 메시지를 출력함



### load\_convolutional\_weights

```c
void load_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //load_convolutional_weights_binary(l, fp);
        //return;
    }
    if(l.numload) l.n = l.numload;
    int num = l.c/l.groups*l.n*l.size*l.size;
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
        if(0){
            int i;
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_mean[i]);
            }
            printf("\n");
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_variance[i]);
            }
            printf("\n");
        }
        if(0){
            fill_cpu(l.n, 0, l.rolling_mean, 1);
            fill_cpu(l.n, 0, l.rolling_variance, 1);
        }
        if(0){
            int i;
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_mean[i]);
            }
            printf("\n");
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_variance[i]);
            }
            printf("\n");
        }
    }
    fread(l.weights, sizeof(float), num, fp);
    //if(l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
    if (l.flipped) {
        transpose_matrix(l.weights, l.c*l.size*l.size, l.n);
    }
    //if (l.binary) binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.weights);
}
```

함수 이름: load\_convolutional\_weights&#x20;

입력:&#x20;

* layer l(컨볼루션 레이어 구조체)
* FILE \*fp(파일 포인터)&#x20;

동작:&#x20;

* 컨볼루션 레이어의 가중치(weights)와 bias를 파일에서 읽어온다.&#x20;
* 이때, 배치 정규화(batch normalization)를 사용하는 경우, 스케일(scale), 롤링 평균(rolling mean), 롤링 분산(rolling variance)도 함께 읽어온다.&#x20;
* 가중치를 읽어온 후, 필터 크기와 채널 수 등의 정보를 기반으로 가중치를 조작하는 연산(transpose\_matrix, binarize\_weights)을 수행한다.&#x20;

설명:&#x20;

* 컨볼루션 레이어의 가중치와 bias를 파일에서 읽어오는 함수이다.&#x20;
* 이 함수는 darknet 딥러닝 프레임워크의 구현체에서 사용되며, 파라미터로 전달된 layer 구조체에 가중치와 bias 정보를 채운다.&#x20;
* 이 함수는 배치 정규화를 사용하는 경우, 스케일, 롤링 평균, 롤링 분산 정보도 함께 읽어온다.&#x20;
* 이 함수는 또한 가중치를 읽어온 후, 필터 크기와 채널 수 등의 정보를 기반으로 가중치를 조작하는 연산(transpose\_matrix, binarize\_weights)을 수행한다.



### load\_batchnorm\_weights

```c
void load_batchnorm_weights(layer l, FILE *fp)
{
    fread(l.scales, sizeof(float), l.c, fp);
    fread(l.rolling_mean, sizeof(float), l.c, fp);
    fread(l.rolling_variance, sizeof(float), l.c, fp);
}
```

함수 이름: load\_batchnorm\_weights&#x20;

입력:&#x20;

* layer l(배치 정규화 레이어 구조체)
* FILE \*fp(파일 포인터)&#x20;

동작:&#x20;

* 배치 정규화 레이어의 스케일(scale), 롤링 평균(rolling mean), 롤링 분산(rolling variance) 정보를 파일에서 읽어온다.

&#x20;설명:&#x20;

* 배치 정규화 레이어의 스케일, 롤링 평균, 롤링 분산 정보를 파일에서 읽어오는 함수이다.&#x20;
* 이 함수는 darknet 딥러닝 프레임워크의 구현체에서 사용되며, 파라미터로 전달된 layer 구조체에 스케일, 롤링 평균, 롤링 분산 정보를 채운다.&#x20;
* 이 함수는 load\_convolutional\_weights 함수에서 호출되어 배치 정규화를 사용하는 경우, 해당 정보를 함께 읽어오는 역할을 한다.



### load\_connected\_weights

```c
void load_connected_weights(layer l, FILE *fp, int transpose)
{
    fread(l.biases, sizeof(float), l.outputs, fp);
    fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if(transpose){
        transpose_matrix(l.weights, l.inputs, l.outputs);
    }
    //printf("Biases: %f mean %f variance\n", mean_array(l.biases, l.outputs), variance_array(l.biases, l.outputs));
    //printf("Weights: %f mean %f variance\n", mean_array(l.weights, l.outputs*l.inputs), variance_array(l.weights, l.outputs*l.inputs));
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.outputs, fp);
        fread(l.rolling_mean, sizeof(float), l.outputs, fp);
        fread(l.rolling_variance, sizeof(float), l.outputs, fp);
        //printf("Scales: %f mean %f variance\n", mean_array(l.scales, l.outputs), variance_array(l.scales, l.outputs));
        //printf("rolling_mean: %f mean %f variance\n", mean_array(l.rolling_mean, l.outputs), variance_array(l.rolling_mean, l.outputs));
        //printf("rolling_variance: %f mean %f variance\n", mean_array(l.rolling_variance, l.outputs), variance_array(l.rolling_variance, l.outputs));
    }
}
```

함수 이름: load\_connected\_weights&#x20;

입력:&#x20;

* layer l(완전 연결 레이어 구조체)
* FILE \*fp(파일 포인터)
* int transpose(전치 여부)&#x20;

동작:&#x20;

* 완전 연결 레이어의 편향(bias)과 가중치(weights) 정보를 파일에서 읽어온다.&#x20;
* 이 때, 전치 플래그(transpose)가 참이면 가중치를 전치한다. 또한 배치 정규화(batch normalization)를 사용하는 경우, 스케일(scale), 롤링 평균(rolling mean), 롤링 분산(rolling variance) 정보도 파일에서 읽어온다.&#x20;

설명:&#x20;

* 완전 연결 레이어의 편향과 가중치 정보를 파일에서 읽어오는 함수이다.&#x20;
* 이 함수는 darknet 딥러닝 프레임워크의 구현체에서 사용되며, 파라미터로 전달된 layer 구조체에 편향과 가중치 정보를 채운다.&#x20;
* 이 함수는 load\_network 함수에서 호출되어, 모델의 파라미터를 파일에서 읽어오는 역할을 한다.&#x20;
* 전치 플래그가 참이면 가중치를 전치하여 저장하며, 배치 정규화를 사용하는 경우, 해당 정보도 함께 읽어오는 역할을 한다.



### transpose\_matrix

```c
void transpose_matrix(float *a, int rows, int cols)
{
    float *transpose = calloc(rows*cols, sizeof(float));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}
```

* 행렬을 전치 시킵니다.

```
-------------                   -------------
| 1 | 2 | 3 |                   | 1 | 4 | 7 |
-------------                   -------------
| 4 | 5 | 6 |        ->         | 2 | 5 | 8 |
-------------                   -------------
| 7 | 8 | 9 |                   | 3 | 6 | 9 |
-------------                   -------------
```

함수 이름: transpose\_matrix

입력:

* float \*a: 전치할 행렬의 포인터
* int rows: 행렬의 행 수
* int cols: 행렬의 열 수

동작:&#x20;

* 주어진 행렬을 전치하여 다시 a에 저장하는 함수입니다. 전치란, 행렬의 행과 열을 서로 바꾸는 것을 의미합니다.

설명:&#x20;

* 행렬을 전치하기 위해서는 각 원소의 인덱스를 바꿔주면 됩니다.&#x20;
* 따라서, 이 함수에서는 행렬 a의 각 행과 열을 반복문을 통해 돌며 transpose 배열에 전치된 원소를 저장합니다.&#x20;
* 그리고나서, transpose 배열을 다시 a에 복사하여 전치된 행렬을 얻습니다.&#x20;
* 이때, 메모리 누수를 방지하기 위해 calloc으로 할당한 transpose 배열은 free 함수를 이용하여 해제합니다.



### save\_weights

```c
void save_weights(network *net, char *filename)
{
    save_weights_upto(net, filename, net->n);
}
```

```c
void save_weights_upto(network *net, char *filename, int cutoff)
{
    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);

    int major = 0;
    int minor = 2;
    int revision = 0;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(net->seen, sizeof(size_t), 1, fp);

    int i;
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontsave) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            save_convolutional_weights(l, fp);
        } if(l.type == CONNECTED){
            save_connected_weights(l, fp);
        } if(l.type == BATCHNORM){
            save_batchnorm_weights(l, fp);
        } if(l.type == RNN){
            save_connected_weights(*(l.input_layer), fp);
            save_connected_weights(*(l.self_layer), fp);
            save_connected_weights(*(l.output_layer), fp);
        } if (l.type == LSTM) {
            save_connected_weights(*(l.wi), fp);
            save_connected_weights(*(l.wf), fp);
            save_connected_weights(*(l.wo), fp);
            save_connected_weights(*(l.wg), fp);
            save_connected_weights(*(l.ui), fp);
            save_connected_weights(*(l.uf), fp);
            save_connected_weights(*(l.uo), fp);
            save_connected_weights(*(l.ug), fp);
        } if (l.type == GRU) {
            if(1){
                save_connected_weights(*(l.wz), fp);
                save_connected_weights(*(l.wr), fp);
                save_connected_weights(*(l.wh), fp);
                save_connected_weights(*(l.uz), fp);
                save_connected_weights(*(l.ur), fp);
                save_connected_weights(*(l.uh), fp);
            }else{
                save_connected_weights(*(l.reset_layer), fp);
                save_connected_weights(*(l.update_layer), fp);
                save_connected_weights(*(l.state_layer), fp);
            }
        }  if(l.type == CRNN){
            save_convolutional_weights(*(l.input_layer), fp);
            save_convolutional_weights(*(l.self_layer), fp);
            save_convolutional_weights(*(l.output_layer), fp);
        } if(l.type == LOCAL){
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fwrite(l.biases, sizeof(float), l.outputs, fp);
            fwrite(l.weights, sizeof(float), size, fp);
        }
    }
    fclose(fp);
}
```

함수 이름: save\_weights\_upto

입력:

* network \*net: 저장할 가중치를 포함하는 신경망 구조체
* char \*filename: 가중치를 저장할 파일 이름
* int cutoff: 저장할 가중치의 최대 레이어 수

동작:

* 주어진 파일 이름으로 가중치를 저장하는 함수
* 저장할 가중치를 포함하는 신경망 구조체와 파일 이름을 입력 받는다.
* cutoff을 이용해 저장할 레이어 수를 제한할 수 있다.
* major, minor, revision 및 net->seen 값을 파일에 저장한다.
* 각 레이어에 대해 해당 레이어 유형에 따라 적절한 가중치 저장 함수를 호출한다.
* 파일 포인터를 닫는다.

설명:

* 이 함수는 신경망의 가중치를 파일에 저장하는 기능을 한다.
* 저장할 가중치는 CONVOLUTIONAL, DECONVOLUTIONAL, CONNECTED, BATCHNORM, RNN, LSTM, GRU, CRNN, LOCAL 레이어 타입에 따라 적절한 가중치 저장 함수를 호출하여 저장한다.
* 저장된 파일은 추후 load\_weights() 함수를 사용하여 신경망에 로드할 수 있다.
* cutoff 값을 사용하여 신경망의 일부 레이어만 저장할 수 있다.



### save\_connected\_weights

```c
void save_connected_weights(layer l, FILE *fp)
{
    fwrite(l.biases, sizeof(float), l.outputs, fp);
    fwrite(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
    }
}
```

함수 이름: save\_connected\_weights

입력:

* layer l: 저장할 weights를 가진 connected layer
* FILE \*fp: 저장할 파일 포인터

동작:

* connected layer의 biases와 weights를 이진 파일로 저장
* batch normalization이 사용되었다면, scales, rolling\_mean, rolling\_variance도 저장

설명:&#x20;

* 주어진 connected layer의 biases와 weights를 이진 파일로 저장하는 함수이다.&#x20;
* batch normalization이 사용된 경우, scales, rolling\_mean, rolling\_variance도 저장한다.&#x20;
* 이진 파일로 저장하기 위해 C 표준 라이브러리의 fwrite 함수를 사용한다.



### save\_batchnorm\_weights

```c
void save_batchnorm_weights(layer l, FILE *fp)
{
    fwrite(l.scales, sizeof(float), l.c, fp);
    fwrite(l.rolling_mean, sizeof(float), l.c, fp);
    fwrite(l.rolling_variance, sizeof(float), l.c, fp);
}
```

함수 이름: save\_batchnorm\_weights

입력:&#x20;

* layer l (배치 정규화를 적용하는 레이어), FILE \*fp (파일 포인터)

동작:&#x20;

* 주어진 레이어 l의 배치 정규화 가중치를 파일에 저장합니다.

설명:

* 배치 정규화를 적용하는 레이어 l의 가중치를 파일에 저장합니다.
* l의 가중치에는 scales, rolling\_mean, rolling\_variance가 포함됩니다.
* 각각의 가중치는 float 형식으로 파일에 저장됩니다.



### save\_convolutional\_weights

```c
void save_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //save_convolutional_weights_binary(l, fp);
        //return;
    }

    int num = l.nweights;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights, sizeof(float), num, fp);
}
```

함수 이름: save\_convolutional\_weights&#x20;

입력:&#x20;

* layer l (layer: 컨볼루션 레이어)
* FILE \*fp (fp: 파일 포인터)&#x20;

동작:&#x20;

* 주어진 컨볼루션 레이어의 가중치를 파일에 저장하는 함수. 각 필터의 bias와 weight, batch normalization을 위한 scale, rolling\_mean, rolling\_variance를 저장한다.&#x20;

설명:

* l.binary가 true인 경우 이진 파일로 저장하고 함수를 종료한다.
* 필터 개수(num)를 계산하고 biases를 먼저 파일에 저장한다.
* batch\_normalize가 true인 경우 scales, rolling\_mean, rolling\_variance도 파일에 저장한다.
* 마지막으로 weights를 파일에 저장한다.

