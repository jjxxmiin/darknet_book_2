# rnn

```c
#include "darknet.h"
#include <math.h>

typedef struct {
    float *x;
    float *y;
} float_pair;
```

## load\_files

```c
unsigned char **load_files(char *filename, int *n)
{
    list *paths = get_paths(filename);
    *n = paths->size;
    unsigned char **contents = calloc(*n, sizeof(char *));
    int i;
    node *x = paths->front;
    for(i = 0; i < *n; ++i){
        contents[i] = read_file((char *)x->val);
        x = x->next;
    }
    return contents;
}
```

함수 이름: load\_files&#x20;

입력:

* filename (char \*) : 파일 경로를 나타내는 문자열
* n (int \*) : 파일 개수를 나타내는 포인터 변수&#x20;

동작:

* filename을 이용하여 파일들이 있는 디렉토리 경로를 찾아낸다.
* 디렉토리 경로에 있는 파일들을 모두 읽어서 메모리에 저장한다.
* 파일의 개수를 n에 저장한다.
* 읽어들인 파일들의 내용을 포인터 배열에 저장하고, 해당 배열을 반환한다.&#x20;

설명:&#x20;

* 이 함수는 입력받은 filename을 이용하여 해당 경로에 있는 파일들을 읽어들이는 함수이다.&#x20;
* 먼저 get\_paths 함수를 통해 파일 경로를 리스트로 얻어낸 후, 리스트의 크기를 이용하여 메모리 할당을 한다.&#x20;
* 그리고 for 루프를 이용하여 리스트에 저장된 경로들의 파일 내용을 읽어들여 배열에 저장하게 된다.&#x20;
* 마지막으로 배열을 반환하면서 읽어들인 파일 개수를 n에 저장하게 된다. 이 함수는 unsigned char 형식의 이중 포인터를 반환하기 때문에, 반환된 값에 접근하기 위해서는 이중 포인터를 이용해야 한다.



## read\_tokenized\_data

```c
int *read_tokenized_data(char *filename, size_t *read)
{
    size_t size = 512;
    size_t count = 0;
    FILE *fp = fopen(filename, "r");
    int *d = calloc(size, sizeof(int));
    int n, one;
    one = fscanf(fp, "%d", &n);
    while(one == 1){
        ++count;
        if(count > size){
            size = size*2;
            d = realloc(d, size*sizeof(int));
        }
        d[count-1] = n;
        one = fscanf(fp, "%d", &n);
    }
    fclose(fp);
    d = realloc(d, count*sizeof(int));
    *read = count;
    return d;
}
```

함수 이름: read\_tokenized\_data

입력:

* filename: 파일 이름을 나타내는 문자열 포인터
* read: 토큰 수를 저장할 size\_t 포인터

동작:

* 입력된 파일을 읽어들여 숫자 데이터를 배열에 저장한 뒤, 해당 배열을 반환함.
* 파일에서 숫자를 읽어들일 때는 공백으로 구분되는 여러 개의 정수들을 각각 하나씩 순서대로 읽어들임.
* 읽어들인 정수들은 배열에 저장됨.
* 배열의 크기는 초기에 512로 설정되며, 배열이 꽉 차게 되면 크기를 두 배로 확장함.
* 반환된 배열의 크기는 해당 파일에서 읽어들인 숫자의 개수와 같으며, 이 값은 read 포인터를 통해 출력됨.

설명:

* 이 함수는 파일에서 숫자 데이터를 읽어들여 배열에 저장하는 기능을 수행함.
* 파일에서 읽어들인 숫자들은 int 타입으로 처리됨.
* 파일에서 숫자를 읽어들일 때, fscanf 함수를 사용하여 파일에서 읽은 문자열을 정수로 변환함.
* 읽어들인 정수는 동적 할당된 배열에 저장됨.
* 배열의 크기가 부족할 경우, realloc 함수를 사용하여 크기를 두 배로 늘림.
* 읽어들인 숫자의 개수를 변수 count를 통해 추적함.
* 최종적으로, 배열의 크기를 실제로 읽어들인 숫자의 개수와 동일하게 줄이기 위해 realloc 함수를 한번 더 사용함.



## read\_tokens

```c
char **read_tokens(char *filename, size_t *read)
{
    size_t size = 512;
    size_t count = 0;
    FILE *fp = fopen(filename, "r");
    char **d = calloc(size, sizeof(char *));
    char *line;
    while((line=fgetl(fp)) != 0){
        ++count;
        if(count > size){
            size = size*2;
            d = realloc(d, size*sizeof(char *));
        }
        if(0==strcmp(line, "<NEWLINE>")) line = "\n";
        d[count-1] = line;
    }
    fclose(fp);
    d = realloc(d, count*sizeof(char *));
    *read = count;
    return d;
}
```

함수 이름: read\_tokens

입력:

* filename: 파일 이름을 나타내는 문자열 포인터
* read: 토큰 수를 저장할 size\_t 포인터

동작:

* 주어진 파일에서 문자열 토큰을 읽어들입니다.
* 각 토큰을 문자열 포인터의 배열에 저장합니다.
* 저장된 토큰 수를 read 포인터에 저장합니다.

설명:&#x20;

* 주어진 파일에서 문자열 토큰을 읽어들여서 이를 문자열 포인터의 배열에 저장하고, 저장된 토큰 수를 반환하는 함수입니다.&#x20;
* 파일을 열고 각 라인을 읽어들인 후, 문자열이 있는 경우 이를 개행 문자로 바꾸고 배열에 저장합니다.&#x20;
* 저장 공간이 부족한 경우 배열 크기를 2배로 늘려줍니다. 마지막으로 배열 크기를 실제 저장된 토큰 수에 맞게 조절하고, 이를 read 포인터에 저장합니다.



## get\_rnn\_token\_data

```c
float_pair get_rnn_token_data(int *tokens, size_t *offsets, int characters, size_t len, int batch, int steps)
{
    float *x = calloc(batch * steps * characters, sizeof(float));
    float *y = calloc(batch * steps * characters, sizeof(float));
    int i,j;
    for(i = 0; i < batch; ++i){
        for(j = 0; j < steps; ++j){
            int curr = tokens[(offsets[i])%len];
            int next = tokens[(offsets[i] + 1)%len];

            x[(j*batch + i)*characters + curr] = 1;
            y[(j*batch + i)*characters + next] = 1;

            offsets[i] = (offsets[i] + 1) % len;

            if(curr >= characters || curr < 0 || next >= characters || next < 0){
                error("Bad char");
            }
        }
    }
    float_pair p;
    p.x = x;
    p.y = y;
    return p;
}
```

함수 이름: get\_rnn\_token\_data

입력:

* tokens (int \*): 정수형 배열로서 토큰들의 시퀀스를 나타냄
* offsets (size\_t \*): 정수형 배열로서 각 배치(batch)의 시작 위치(offset)를 나타냄
* characters (int): 정수형으로서 가능한 문자(character)의 총 개수를 나타냄
* len (size\_t): 정수형으로서 토큰 시퀀스의 전체 길이를 나타냄
* batch (int): 정수형으로서 배치(batch)의 개수를 나타냄
* steps (int): 정수형으로서 RNN에서 처리할 스텝(step)의 개수를 나타냄

동작:

* 입력으로 받은 토큰 시퀀스를 RNN 학습을 위한 형태로 변환하여 반환함
* 입력으로 받은 토큰 시퀀스를 이용해 one-hot 인코딩(one-hot encoding)된 입력 데이터(x)와 출력 데이터(y)를 생성함
* x와 y는 각각 크기가 (batch \* steps \* characters)인 1차원 배열이며, 각 스텝(step)에서 현재 입력(input)의 one-hot 인코딩이 x에 저장되고, 다음 출력(output)의 one-hot 인코딩이 y에 저장됨
* 생성된 x와 y를 구조체(struct)에 담아 반환함

설명:

* 이 함수는 RNN 학습을 위한 토큰 데이터를 생성하는 함수이며, 이를 위해 토큰 시퀀스를 입력과 출력으로 변환함
* 입력으로 받은 토큰 시퀀스는 각 배치(batch)의 시작 위치(offset)를 나타내는 offsets 배열과 함께 사용됨
* 각 배치마다 steps 개수만큼의 스텝(step)을 처리하며, 현재 입력(input)과 다음 출력(output)을 one-hot 인코딩(one-hot encoding)하여 입력 데이터(x)와 출력 데이터(y)를 생성함
* 생성된 입력과 출력 데이터는 float\_pair 구조체에 담아 반환됨



## get\_seq2seq\_data

```c
float_pair get_seq2seq_data(char **source, char **dest, int n, int characters, size_t len, int batch, int steps)
{
    int i,j;
    float *x = calloc(batch * steps * characters, sizeof(float));
    float *y = calloc(batch * steps * characters, sizeof(float));
    for(i = 0; i < batch; ++i){
        int index = rand()%n;
        //int slen = strlen(source[index]);
        //int dlen = strlen(dest[index]);
        for(j = 0; j < steps; ++j){
            unsigned char curr = source[index][j];
            unsigned char next = dest[index][j];

            x[(j*batch + i)*characters + curr] = 1;
            y[(j*batch + i)*characters + next] = 1;

            if(curr > 255 || curr <= 0 || next > 255 || next <= 0){
                /*text[(index+j+2)%len] = 0;
                printf("%ld %d %d %d %d\n", index, j, len, (int)text[index+j], (int)text[index+j+1]);
                printf("%s", text+index);
                */
                error("Bad char");
            }
        }
    }
    float_pair p;
    p.x = x;
    p.y = y;
    return p;
}
```

함수 이름: get\_seq2seq\_data

입력:

* char \*\*source: 원문을 포함한 문자열 배열
* char \*\*dest: 번역문을 포함한 문자열 배열
* int n: 입력 데이터의 총 개수
* int characters: 문자 집합의 크기
* size\_t len: 데이터의 최대 길이
* int batch: 배치 크기
* int steps: 각 시퀀스에서의 타임 스텝 수

동작:

* 주어진 소스와 대상 문자열에서 무작위로 배치 크기만큼의 시퀀스를 추출하여 입력 데이터를 생성합니다.
* 각 시퀀스에서의 모든 타임 스텝에 대해 현재 문자와 다음 문자를 가져와서 x와 y 배열에 인코딩합니다.
* 문자가 문자 집합의 범위를 벗어나는 경우 오류를 발생시킵니다.

설명:

* 이 함수는 시퀀스-투-시퀀스 모델의 데이터를 생성하는 데 사용됩니다.
* 소스 문자열은 모델의 인코더의 입력이고, 대상 문자열은 모델의 디코더의 입력 및 출력입니다.
* 모델은 소스 문자열을 인코딩하여 고정 길이의 벡터로 변환한 다음, 이 벡터를 사용하여 대상 문자열을 디코딩합니다.
* 이 함수는 무작위로 선택된 시퀀스를 생성하여 입력 데이터를 만듭니다.



## get\_rnn\_data

```c
float_pair get_rnn_data(unsigned char *text, size_t *offsets, int characters, size_t len, int batch, int steps)
{
    float *x = calloc(batch * steps * characters, sizeof(float));
    float *y = calloc(batch * steps * characters, sizeof(float));
    int i,j;
    for(i = 0; i < batch; ++i){
        for(j = 0; j < steps; ++j){
            unsigned char curr = text[(offsets[i])%len];
            unsigned char next = text[(offsets[i] + 1)%len];

            x[(j*batch + i)*characters + curr] = 1;
            y[(j*batch + i)*characters + next] = 1;

            offsets[i] = (offsets[i] + 1) % len;

            if(curr > 255 || curr <= 0 || next > 255 || next <= 0){
                /*text[(index+j+2)%len] = 0;
                printf("%ld %d %d %d %d\n", index, j, len, (int)text[index+j], (int)text[index+j+1]);
                printf("%s", text+index);
                */
                error("Bad char");
            }
        }
    }
    float_pair p;
    p.x = x;
    p.y = y;
    return p;
}
```

함수 이름: get\_rnn\_data

입력:

* unsigned char \*text: RNN 모델에 입력될 텍스트 데이터 배열
* size\_t \*offsets: 텍스트 데이터 배열에서 batch 크기만큼의 데이터를 선택하기 위한 offset 값들이 저장된 배열
* int characters: 문자 집합의 크기
* size\_t len: 텍스트 데이터 배열의 길이
* int batch: 한 번의 학습에서 처리할 batch 크기
* int steps: RNN 모델의 time step 크기

동작:&#x20;

* RNN 모델 학습을 위한 데이터셋을 생성하는 함수로, 입력으로 받은 텍스트 데이터 배열에서 batch 크기만큼의 데이터를 선택하여 RNN 모델 학습에 필요한 x, y 데이터셋을 생성합니다.&#x20;
* 생성된 x, y 데이터셋은 각각 RNN 모델의 입력과 정답 데이터가 됩니다.

설명:

* calloc 함수를 이용하여 x, y 데이터셋을 batch 크기와 time step 크기에 맞게 할당합니다.
* for문을 이용하여 batch 크기만큼 데이터를 선택하고, steps 크기만큼 x, y 데이터셋에 값을 할당합니다.
* x 데이터셋은 curr 위치에 해당하는 문자의 인덱스 값을 1로 설정하고, y 데이터셋은 next 위치에 해당하는 문자의 인덱스 값을 1로 설정합니다.
* offsets 배열에는 각 batch의 시작 위치가 저장되어 있으며, 한 번 데이터를 선택하고 나면 offsets 값을 len으로 나눈 나머지로 갱신합니다.
* 각 선택된 데이터가 문자 집합의 크기를 초과하거나 0 이하일 경우 에러 메시지를 출력합니다.
* 생성된 x, y 데이터셋을 float\_pair 구조체에 저장하여 반환합니다.



## train\_char\_rnn

```c
void train_char_rnn(char *cfgfile, char *weightfile, char *filename, int clear, int tokenized)
{
    srand(time(0));
    unsigned char *text = 0;
    int *tokens = 0;
    size_t size;
    if(tokenized){
        tokens = read_tokenized_data(filename, &size);
    } else {
        text = read_file(filename);
        size = strlen((const char*)text);
    }

    char *backup_directory = "/home/pjreddie/backup/";
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);
    float avg_loss = -1;
    network *net = load_network(cfgfile, weightfile, clear);

    int inputs = net->inputs;
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g, Inputs: %d %d %d\n", net->learning_rate, net->momentum, net->decay, inputs, net->batch, net->time_steps);
    int batch = net->batch;
    int steps = net->time_steps;
    if(clear) *net->seen = 0;
    int i = (*net->seen)/net->batch;

    int streams = batch/steps;
    size_t *offsets = calloc(streams, sizeof(size_t));
    int j;
    for(j = 0; j < streams; ++j){
        offsets[j] = rand_size_t()%size;
    }

    clock_t time;
    while(get_current_batch(net) < net->max_batches){
        i += 1;
        time=clock();
        float_pair p;
        if(tokenized){
            p = get_rnn_token_data(tokens, offsets, inputs, size, streams, steps);
        }else{
            p = get_rnn_data(text, offsets, inputs, size, streams, steps);
        }

        copy_cpu(net->inputs*net->batch, p.x, 1, net->input, 1);
        copy_cpu(net->truths*net->batch, p.y, 1, net->truth, 1);
        float loss = train_network_datum(net) / (batch);
        free(p.x);
        free(p.y);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        size_t chars = get_current_batch(net)*batch;
        fprintf(stderr, "%d: %f, %f avg, %f rate, %lf seconds, %f epochs\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), (float) chars/size);

        for(j = 0; j < streams; ++j){
            //printf("%d\n", j);
            if(rand()%64 == 0){
                //fprintf(stderr, "Reset\n");
                offsets[j] = rand_size_t()%size;
                reset_network_state(net, j);
            }
        }

        if(i%10000==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}
```

함수 이름: train\_char\_rnn

입력:

* cfgfile: 문자열, 네트워크 설정 파일 경로
* weightfile: 문자열, 네트워크 가중치 파일 경로
* filename: 문자열, 학습 데이터 파일 경로
* clear: 정수, 네트워크 초기화 여부 (0 또는 1)
* tokenized: 정수, 토큰화된 데이터를 사용할지 여부 (0 또는 1)

동작:

* 학습 데이터를 읽어들이고, 네트워크를 초기화한 후 학습을 수행하는 함수
* 입력으로 주어진 학습 데이터를 사용하여 배치 크기(batch size)와 시간 스텝(time step)을 설정하고, 훈련 데이터에서 임의의 위치에서 시작하여 여러 시퀀스를 추출한다.
* 추출된 시퀀스를 이용하여 배치 데이터를 생성하고, 이를 사용하여 네트워크를 훈련한다.
* 네트워크의 학습이 완료될 때까지 반복적으로 학습 데이터를 읽어들이고, 배치 데이터를 생성하여 네트워크를 학습한다.
* 네트워크의 학습 과정에서 정기적으로 가중치를 저장하고 학습 상태를 출력한다.

설명:

* 네트워크 설정 파일(cfgfile)은 네트워크의 구조 및 하이퍼파라미터를 정의하는 파일이다.
* 네트워크 가중치 파일(weightfile)은 미리 학습된 네트워크의 가중치 값을 저장하는 파일이다.
* 학습 데이터 파일(filename)은 네트워크를 학습할 때 사용하는 데이터를 저장하는 파일이다.
* clear 인수가 1로 설정되면, 네트워크의 가중치를 재설정한다.
* tokenized 인수가 1로 설정되면, 학습 데이터가 이미 토큰화되어 있는 경우이다. 그렇지 않으면, 학습 데이터를 읽어들인 후 토큰화한다.
* 배치 크기(batch size)는 네트워크가 한 번에 처리하는 입력 데이터의 크기이다. 이 값은 네트워크 설정 파일(cfgfile)에서 정의된다.
* 시간 스텝(time step)은 시퀀스 데이터를 처리하는 데 필요한 네트워크 계산의 수를 나타내는 값이다. 이 값도 네트워크 설정 파일(cfgfile)에서 정의된다.
* 추출된 시퀀스는 각각 배치 크기(batch size)와 시간 스텝(time step)에 맞게 조정된다.
* 훈련 데이터에서 임의의 위치에서 시작하여 추출된 시퀀스는 배치 크기(batch size)와 시간 스텝(time step)에 맞게 잘

##

## print\_symbol

```c
void print_symbol(int n, char **tokens){
    if(tokens){
        printf("%s ", tokens[n]);
    } else {
        printf("%c", n);
    }
}
```

함수 이름: print\_symbol&#x20;

입력:

* n (int 타입): 출력할 심볼의 인덱스
* tokens (char \*\* 타입): 심볼을 저장한 문자열 배열

동작:

* 입력으로 받은 인덱스 n에 해당하는 심볼을 출력한다.
* tokens 배열이 주어진 경우, 해당 인덱스에 해당하는 문자열을 출력하고 뒤에 공백을 추가한다.
* tokens 배열이 주어지지 않은 경우, 해당 인덱스에 해당하는 아스키 문자를 출력한다.

설명:

* print\_symbol 함수는 주어진 인덱스에 해당하는 심볼을 출력하는 함수이다.
* tokens 배열이 주어지면, 해당 배열에서 인덱스 n에 해당하는 문자열을 출력하고, 배열이 주어지지 않으면, 해당 인덱스에 해당하는 아스키 문자를 출력한다.
* 출력된 심볼 뒤에는 tokens 배열이 주어진 경우에는 공백이 추가된다.



## test\_char\_rnn

```c
void test_char_rnn(char *cfgfile, char *weightfile, int num, char *seed, float temp, int rseed, char *token_file)
{
    char **tokens = 0;
    if(token_file){
        size_t n;
        tokens = read_tokens(token_file, &n);
    }

    srand(rseed);
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);

    network *net = load_network(cfgfile, weightfile, 0);
    int inputs = net->inputs;

    int i, j;
    for(i = 0; i < net->n; ++i) net->layers[i].temperature = temp;
    int c = 0;
    int len = strlen(seed);
    float *input = calloc(inputs, sizeof(float));

    /*
       fill_cpu(inputs, 0, input, 1);
       for(i = 0; i < 10; ++i){
       network_predict(net, input);
       }
       fill_cpu(inputs, 0, input, 1);
     */

    for(i = 0; i < len-1; ++i){
        c = seed[i];
        input[c] = 1;
        network_predict(net, input);
        input[c] = 0;
        print_symbol(c, tokens);
    }
    if(len) c = seed[len-1];
    print_symbol(c, tokens);
    for(i = 0; i < num; ++i){
        input[c] = 1;
        float *out = network_predict(net, input);
        input[c] = 0;
        for(j = 32; j < 127; ++j){
            //printf("%d %c %f\n",j, j, out[j]);
        }
        for(j = 0; j < inputs; ++j){
            if (out[j] < .0001) out[j] = 0;
        }
        c = sample_array(out, inputs);
        print_symbol(c, tokens);
    }
    printf("\n");
}
```

함수 이름: test\_char\_rnn

입력:

* cfgfile: 문자열 포인터. 신경망의 구성 파일 경로를 지정하는데 사용됨.
* weightfile: 문자열 포인터. 신경망 가중치 파일 경로를 지정하는데 사용됨.
* num: 정수. 생성할 문자 수를 지정하는데 사용됨.
* seed: 문자열 포인터. 생성 시작을 위한 시드 텍스트를 지정하는데 사용됨.
* temp: 부동 소수점. 예측된 다음 문자의 확률 분포를 제어하는데 사용됨.
* rseed: 정수. 무작위 생성을 위한 시드를 지정하는데 사용됨.
* token\_file: 문자열 포인터. 토큰 파일 경로를 지정하는데 사용됨.

동작:

* 텍스트 생성 신경망을 테스트하여 텍스트를 생성함.

설명:

* 주어진 cfgfile 및 weightfile로부터 신경망을 로드함.
* 지정된 시드 텍스트에서 시작하여 num 개의 새로운 문자를 생성함.
* 생성된 각 문자는 출력되어 표시됨.
* 예측된 다음 문자의 확률 분포를 제어하는 데 사용되는 온도를 설정할 수 있음.
* 지정된 token\_file이 있으면, 출력할 때 해당 토큰 파일의 토큰을 사용하여 출력함.
* 시드 텍스트의 길이는 입력 신경망의 크기로 잘라짐.
* 출력되는 문자 수(num)는 시드 텍스트 이후에 생성되는 문자의 수를 의미함.



## test\_tactic\_rnn\_multi

```c
void test_tactic_rnn_multi(char *cfgfile, char *weightfile, int num, float temp, int rseed, char *token_file)
{
    char **tokens = 0;
    if(token_file){
        size_t n;
        tokens = read_tokens(token_file, &n);
    }

    srand(rseed);
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);

    network *net = load_network(cfgfile, weightfile, 0);
    int inputs = net->inputs;

    int i, j;
    for(i = 0; i < net->n; ++i) net->layers[i].temperature = temp;
    int c = 0;
    float *input = calloc(inputs, sizeof(float));
    float *out = 0;

    while(1){
        reset_network_state(net, 0);
        while((c = getc(stdin)) != EOF && c != 0){
            input[c] = 1;
            out = network_predict(net, input);
            input[c] = 0;
        }
        for(i = 0; i < num; ++i){
            for(j = 0; j < inputs; ++j){
                if (out[j] < .0001) out[j] = 0;
            }
            int next = sample_array(out, inputs);
            if(c == '.' && next == '\n') break;
            c = next;
            print_symbol(c, tokens);

            input[c] = 1;
            out = network_predict(net, input);
            input[c] = 0;
        }
        printf("\n");
    }
}
```

함수 이름: test\_tactic\_rnn\_multi

입력:

* cfgfile: char pointer. RNN 네트워크의 구성 파일 경로
* weightfile: char pointer. 학습된 RNN 네트워크의 가중치 파일 경로
* num: int. 생성할 텍스트의 길이 (문자 수)
* temp: float. 생성할 때 사용할 softmax 온도 값
* rseed: int. 난수 생성을 위한 시드 값
* token\_file: char pointer. 텍스트 생성에 사용할 토큰 파일 경로. (옵션)

동작:

* RNN 네트워크를 불러오고, 입력으로부터 새로운 텍스트를 생성하는 함수
* 입력으로 받은 cfgfile, weightfile로부터 네트워크를 로드하고, 입력 크기(inputs)를 설정
* 생성된 텍스트를 출력하기 위한 출력용 버퍼(input)를 초기화하고, 출력용 버퍼에 저장된 값을 사용하여 다음 출력 문자를 결정
* 출력용 버퍼에 현재 출력 문자를 저장하고, 출력 문자를 출력
* 출력용 버퍼와 현재 출력 문자를 사용하여 네트워크에 입력을 전달하고, 출력값(out)을 얻음
* 현재 출력 문자와 출력값(out)을 사용하여 다음 출력 문자를 결정
* 출력 문자가 개행 문자일 경우, 생성을 중지하고 출력 종료

설명:

* 이 함수는 주어진 RNN 모델을 사용하여 새로운 텍스트를 생성하고 출력하는 역할을 수행합니다.
* 입력으로는 RNN 모델의 구성 파일 경로(cfgfile), 학습된 가중치 파일 경로(weightfile), 생성할 텍스트의 길이(num), softmax 온도 값(temp), 난수 생성 시드 값(rseed)을 받습니다.
* 또한, 선택적으로 사용할 수 있는 입력으로는 텍스트 생성에 사용할 토큰 파일 경로(token\_file)가 있습니다.
* RNN 모델을 로드하고 입력 크기(inputs)를 설정한 후, 출력용 버퍼(input)를 초기화합니다.
* 출력용 버퍼에 저장된 값을 사용하여 다음 출력 문자를 결정하고, 출력 문자를 출력합니다.
* 출력용 버퍼와 현재 출력 문자를 사용하여 네트워크에 입력을 전달하고, 출력값(out)을 얻습니다.
* 현재 출력 문자와 출력값(out)을 사용하여 다음 출력 문자를 결정합니다.
* 출력 문자가 개행 문자일 경우, 생성을 중지하고 출력을 종료합니다.



## test\_tactic\_rnn

```c
void test_tactic_rnn(char *cfgfile, char *weightfile, int num, float temp, int rseed, char *token_file)
{
    char **tokens = 0;
    if(token_file){
        size_t n;
        tokens = read_tokens(token_file, &n);
    }

    srand(rseed);
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);

    network *net = load_network(cfgfile, weightfile, 0);
    int inputs = net->inputs;

    int i, j;
    for(i = 0; i < net->n; ++i) net->layers[i].temperature = temp;
    int c = 0;
    float *input = calloc(inputs, sizeof(float));
    float *out = 0;

    while((c = getc(stdin)) != EOF){
        input[c] = 1;
        out = network_predict(net, input);
        input[c] = 0;
    }
    for(i = 0; i < num; ++i){
        for(j = 0; j < inputs; ++j){
            if (out[j] < .0001) out[j] = 0;
        }
        int next = sample_array(out, inputs);
        if(c == '.' && next == '\n') break;
        c = next;
        print_symbol(c, tokens);

        input[c] = 1;
        out = network_predict(net, input);
        input[c] = 0;
    }
    printf("\n");
}
```

함수 이름: test\_tactic\_rnn 입력:

* cfgfile: char\*: 모델 설정 파일 경로
* weightfile: char\*: 모델 가중치 파일 경로
* num: int: 생성할 기호 수
* temp: float: 생성에 사용할 softmax 온도
* rseed: int: 난수 시드
* token\_file: char\*: 토큰 파일 경로 (선택 사항)

동작:&#x20;

* 주어진 모델을 사용하여 입력 기호를 생성하고 출력하는 함수입니다.&#x20;
* 입력으로는 모델 설정 파일, 가중치 파일, 생성할 기호 수, softmax 온도, 난수 시드, 토큰 파일 경로 (선택 사항)을 받습니다.&#x20;
* 입력된 모델을 불러온 후에는 softmax 온도를 설정하고, 입력 기호를 받아들이고 예측 결과를 계산합니다.&#x20;
* 이후, 생성할 기호 수만큼 예측을 반복하면서 생성된 기호를 출력합니다. 출력 중 "."과 "\n"이 연속으로 등장하면 출력을 중지합니다.

설명:

* tokens: char\*\*: 토큰 배열
* base: char\*: 모델 설정 파일에서 불러온 베이스 설정
* net: network\*: 불러온 모델
* inputs: int: 모델 입력의 크기
* i, j: int: 반복문을 위한 변수
* c: int: 현재 입력 기호
* input: float\*: 모델 입력
* out: float\*: 모델 예측 결과
* next: int: 다음 생성 기호
* print\_symbol: 함수 포인터: 출력 기호를 토큰화하여 출력하는 함수



## valid\_tactic\_rnn

```c
void valid_tactic_rnn(char *cfgfile, char *weightfile, char *seed)
{
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);

    network *net = load_network(cfgfile, weightfile, 0);
    int inputs = net->inputs;

    int count = 0;
    int words = 1;
    int c;
    int len = strlen(seed);
    float *input = calloc(inputs, sizeof(float));
    int i;
    for(i = 0; i < len; ++i){
        c = seed[i];
        input[(int)c] = 1;
        network_predict(net, input);
        input[(int)c] = 0;
    }
    float sum = 0;
    c = getc(stdin);
    float log2 = log(2);
    int in = 0;
    while(c != EOF){
        int next = getc(stdin);
        if(next == EOF) break;
        if(next < 0 || next >= 255) error("Out of range character");

        input[c] = 1;
        float *out = network_predict(net, input);
        input[c] = 0;

        if(c == '.' && next == '\n') in = 0;
        if(!in) {
            if(c == '>' && next == '>'){
                in = 1;
                ++words;
            }
            c = next;
            continue;
        }
        ++count;
        sum += log(out[next])/log2;
        c = next;
        printf("%d %d Perplexity: %4.4f    Word Perplexity: %4.4f\n", count, words, pow(2, -sum/count), pow(2, -sum/words));
    }
}
```

함수 이름: valid\_tactic\_rnn&#x20;

입력:

* cfgfile: char 포인터 타입. Tactic 모델의 구성 파일 경로.
* weightfile: char 포인터 타입. Tactic 모델의 가중치 파일 경로.
* seed: char 포인터 타입. 모델을 검증하기 위한 시드 문자열.

동작:&#x20;

* 주어진 Tactic 모델을 사용하여 시드 문자열을 바탕으로 모델을 검증하고, 입력으로 주어지는 텍스트 파일의 perplexity와 단어 당 perplexity를 계산하여 출력한다.

설명:

* basecfg 함수를 사용하여 cfgfile에서 모델의 기본 설정을 가져온다.
* load\_network 함수를 사용하여 cfgfile과 weightfile에서 모델을 로드한다.
* seed 문자열을 모델에 입력으로 제공하여 모델을 초기화한다.
* getc(stdin)을 사용하여 입력 파일에서 문자를 하나씩 읽어들인다.
* 입력 파일에서 읽어들인 문자를 모델에 입력으로 제공하고, 다음 문자를 예측한다.
* perplexity와 단어 당 perplexity를 계산하여 출력한다.



## valid\_char\_rnn

```c
void valid_char_rnn(char *cfgfile, char *weightfile, char *seed)
{
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);

    network *net = load_network(cfgfile, weightfile, 0);
    int inputs = net->inputs;

    int count = 0;
    int words = 1;
    int c;
    int len = strlen(seed);
    float *input = calloc(inputs, sizeof(float));
    int i;
    for(i = 0; i < len; ++i){
        c = seed[i];
        input[(int)c] = 1;
        network_predict(net, input);
        input[(int)c] = 0;
    }
    float sum = 0;
    c = getc(stdin);
    float log2 = log(2);
    while(c != EOF){
        int next = getc(stdin);
        if(next == EOF) break;
        if(next < 0 || next >= 255) error("Out of range character");
        ++count;
        if(next == ' ' || next == '\n' || next == '\t') ++words;
        input[c] = 1;
        float *out = network_predict(net, input);
        input[c] = 0;
        sum += log(out[next])/log2;
        c = next;
        printf("%d BPC: %4.4f   Perplexity: %4.4f    Word Perplexity: %4.4f\n", count, -sum/count, pow(2, -sum/count), pow(2, -sum/words));
    }
}
```

함수 이름: valid\_char\_rnn&#x20;

입력:

* cfgfile: char 형 포인터. 네트워크 설정 파일 경로
* weightfile: char 형 포인터. 학습된 모델 가중치 파일 경로
* seed: char 형 포인터. 네트워크 시작 문자열

동작:&#x20;

* 주어진 네트워크 설정 파일(cfgfile)과 가중치 파일(weightfile)을 사용하여 문자 수준(character-level)의 언어 모델을 로드하고, 주어진 시작 문자열(seed)로 네트워크 상태를 초기화합니다.&#x20;
* 이후에는 표준 입력(stdin)으로부터 문자를 하나씩 읽어들이면서, 현재까지 읽어들인 문자열에 대한 예측값을 출력하고, 다음 문자에 대한 예측을 수행합니다.
* 이 과정을 통해 모델의 성능을 측정합니다. 최종적으로, 현재까지 읽어들인 문자열에 대한 엔트로피(entropy), 퍼플렉서티(perplexity) 및 단어 당 퍼플렉서티를 출력합니다.

설명:

* basecfg(cfgfile): cfgfile로부터 설정 파일의 기본 이름을 추출하는 함수입니다.
* load\_network(cfgfile, weightfile, clear): cfgfile 및 weightfile로부터 네트워크를 로드하는 함수입니다. clear 인자는 네트워크 상태를 초기화할지 여부를 결정합니다.
* calloc(n, size): n개의 size 바이트 메모리 블록을 할당하고 이를 모두 0으로 초기화합니다.
* network\_predict(net, input): 네트워크(net)에 대한 입력(input)에 대한 예측값을 계산하는 함수입니다.
* log(x): x의 자연로그 값을 계산하는 함수입니다.
* pow(x, y): x의 y 제곱 값을 계산하는 함수입니다.
* BPC(Bit Per Character): 문자 당 평균 비트 수로, 모델의 예측 성능을 나타내는 지표 중 하나입니다.
* Perplexity: 언어 모델의 예측 성능을 나타내는 지표 중 하나로, 다음 문자의 예측 확률의 역수에 로그를 취한 값을 평균한 값입니다. 이 값이 작을수록 모델의 성능이 우수합니다.
* 단어 당 퍼플렉서티: 퍼플렉서티를 단어 수로 나눈 값으로, 긴 문장에서의 모델 성능을 측정하는 지표입니다.



## vec\_char\_rnn

```c
void vec_char_rnn(char *cfgfile, char *weightfile, char *seed)
{
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);

    network *net = load_network(cfgfile, weightfile, 0);
    int inputs = net->inputs;

    int c;
    int seed_len = strlen(seed);
    float *input = calloc(inputs, sizeof(float));
    int i;
    char *line;
    while((line=fgetl(stdin)) != 0){
        reset_network_state(net, 0);
        for(i = 0; i < seed_len; ++i){
            c = seed[i];
            input[(int)c] = 1;
            network_predict(net, input);
            input[(int)c] = 0;
        }
        strip(line);
        int str_len = strlen(line);
        for(i = 0; i < str_len; ++i){
            c = line[i];
            input[(int)c] = 1;
            network_predict(net, input);
            input[(int)c] = 0;
        }
        c = ' ';
        input[(int)c] = 1;
        network_predict(net, input);
        input[(int)c] = 0;

        layer l = net->layers[0];
        #ifdef GPU
        cuda_pull_array(l.output_gpu, l.output, l.outputs);
        #endif
        printf("%s", line);
        for(i = 0; i < l.outputs; ++i){
            printf(",%g", l.output[i]);
        }
        printf("\n");
    }
}
```

함수 이름: vec\_char\_rnn

입력:

* cfgfile: 학습된 모델의 설정 파일 경로 (문자열)
* weightfile: 학습된 모델의 가중치 파일 경로 (문자열)
* seed: 초기 시퀀스 (문자열)

동작:&#x20;

* 주어진 seed를 이용해 학습된 모델을 초기화한 뒤, stdin으로부터 입력된 문자열을 하나씩 받아들이면서 해당 문자열의 다음 문자 예측 결과를 출력한다.&#x20;
* 출력 결과는 입력된 문자열과 함께, 예측된 다음 문자의 출력 확률 값을 콤마로 구분하여 출력한다.

설명:&#x20;

* 주어진 학습된 모델(cfgfile, weightfile)과 초기 시퀀스(seed)를 이용해, stdin으로부터 입력된 문자열을 하나씩 받아들이면서 다음 문자의 예측 결과를 출력하는 함수이다.&#x20;
* 입력된 문자열의 마지막에는 공백문자를 추가해야 하며, 출력 결과는 입력된 문자열과 함께, 예측된 다음 문자의 출력 확률 값을 콤마로 구분하여 출력한다.&#x20;
* 이 때 출력 확률 값은 해당 문자의 원-핫 인코딩 벡터에 대해 모델의 예측 결과 값이다.



## run\_char\_rnn

```c
void run_char_rnn(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *filename = find_char_arg(argc, argv, "-file", "data/shakespeare.txt");
    char *seed = find_char_arg(argc, argv, "-seed", "\n\n");
    int len = find_int_arg(argc, argv, "-len", 1000);
    float temp = find_float_arg(argc, argv, "-temp", .7);
    int rseed = find_int_arg(argc, argv, "-srand", time(0));
    int clear = find_arg(argc, argv, "-clear");
    int tokenized = find_arg(argc, argv, "-tokenized");
    char *tokens = find_char_arg(argc, argv, "-tokens", 0);

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    if(0==strcmp(argv[2], "train")) train_char_rnn(cfg, weights, filename, clear, tokenized);
    else if(0==strcmp(argv[2], "valid")) valid_char_rnn(cfg, weights, seed);
    else if(0==strcmp(argv[2], "validtactic")) valid_tactic_rnn(cfg, weights, seed);
    else if(0==strcmp(argv[2], "vec")) vec_char_rnn(cfg, weights, seed);
    else if(0==strcmp(argv[2], "generate")) test_char_rnn(cfg, weights, len, seed, temp, rseed, tokens);
    else if(0==strcmp(argv[2], "generatetactic")) test_tactic_rnn(cfg, weights, len, temp, rseed, tokens);
}
```

함수 이름: run\_char\_rnn

입력:

* int argc: 명령줄 인수의 수
* char \*\*argv: 명령줄 인수 배열

동작:

* 입력된 인수가 충분하지 않으면 사용 방법을 출력하고 함수를 종료한다.
* filename, seed, len, temp, rseed, clear, tokenized, tokens을 각각의 옵션에 맞게 설정한다.
* cfg와 weights를 설정한다.
* argv\[2]에 따라서 train\_char\_rnn, valid\_char\_rnn, valid\_tactic\_rnn, vec\_char\_rnn, test\_char\_rnn, test\_tactic\_rnn 함수를 호출한다.

설명:

* 이 함수는 char\_rnn 모델을 실행하기 위한 함수로, 명령줄에서 인수를 받아와서 모델을 학습하거나 생성하는 등의 작업을 수행한다.
* filename은 학습할 데이터 파일의 경로를 지정한다.
* seed는 모델의 초기 입력 시퀀스를 지정한다.
* len은 생성할 시퀀스의 길이를 지정한다.
* temp는 생성할 시퀀스의 품질을 조절하는 온도 매개변수이다.
* rseed는 시드 값으로 사용할 난수 발생기의 시드를 지정한다.
* clear는 학습 중 생성한 캐시 파일을 삭제할지 여부를 결정한다.
* tokenized는 입력 파일이 토큰화된 텍스트인지 여부를 결정한다.
* tokens는 토큰화된 텍스트 파일의 경로를 지정한다.
* cfg와 weights는 각각 모델 구성 파일과 가중치 파일의 경로를 지정한다.
* argv\[2]에 따라서 학습, 검증, 생성 등의 작업을 수행하는 함수를 호출한다.

