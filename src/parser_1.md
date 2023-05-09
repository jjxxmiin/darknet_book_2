# parser\_1

`parser`는 compiler, interpreter의 구성 요소 중 하나로 입력 token에 포함 된 자료 구조를 구성하고 문법을 검사합니다. `darknet`에서는 `cfg`로 네트워크의 구조를 구성하고 문법을 검사하는데 사용 됩니다. 매우 중요하고 잘 짜여진 알고리즘 입니다.

여기서 `cfg`파일은 모델의 구조를 담고 있는 파일입니다. 파일의 구조를 보면 대략 아래와 같습니다.

```
[net]
param1=value1
param2=value2
param3=value3

[layer type]
param1=value1
param2=value2

[layer type]
param1=value1
param2=value2
```

괄호안에는 layer의 타입이 설정되어 있고 그 밑에는 해당 layer에 필요한 파라미터 값을 설정하도록 되어있습니다.

이렇게 모델의 구조를 설정하면 하나의 네트워크의 자료구조를 만들게 됩니다.

* network list는 root
* section list는 layer
* option list는 layer의 매개변수

라고 이해하고 읽어봅시다.

## parse\_network\_cfg

```c
network *parse_network_cfg(char *filename)
{
    list *sections = read_cfg(filename);                    /// sections = cfg 파일 구조 리스트
    node *n = sections->front;                              /// n = sections의 맨 앞 노드
    if(!n) error("Config file has no sections");
    network *net = make_network(sections->size - 1);        
    net->gpu_index = gpu_index;                                      
    size_params params;                                     

    section *s = (section *)n->val;                         
    list *options = s->options;                            
    if(!is_network(s)) error("First section must be [net] or [network]");
    parse_net_options(options, net);                        

    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.time_steps = net->time_steps;
    params.net = net;

    size_t workspace_size = 0;
    n = n->next;                                             /// layer 부분 해석시작
    int count = 0;
    free_section(s);                                         /// section 할당 해제
    fprintf(stderr, "layer     filters    size              input                output\n");
    while(n){
        params.index = count;
        fprintf(stderr, "%5d ", count);
        s = (section *)n->val;                               /// section의 값
        options = s->options;                                /// option 불러오기
        layer l = {0};   
        LAYER_TYPE lt = string_to_layer_type(s->type);       /// LAYER_TYPE 찾기
        if(lt == CONVOLUTIONAL){
            l = parse_convolutional(options, params);
        }else if(lt == DECONVOLUTIONAL){
            l = parse_deconvolutional(options, params);
        }else if(lt == LOCAL){
            l = parse_local(options, params);
        }else if(lt == ACTIVE){
            l = parse_activation(options, params);
        }else if(lt == LOGXENT){
            l = parse_logistic(options, params);
        }else if(lt == L2NORM){
            l = parse_l2norm(options, params);
        }else if(lt == RNN){
            l = parse_rnn(options, params);
        }else if(lt == GRU){
            l = parse_gru(options, params);
        }else if (lt == LSTM) {
            l = parse_lstm(options, params);
        }else if(lt == CRNN){
            l = parse_crnn(options, params);
        }else if(lt == CONNECTED){
            l = parse_connected(options, params);
        }else if(lt == CROP){
            l = parse_crop(options, params);
        }else if(lt == COST){
            l = parse_cost(options, params);
        }else if(lt == REGION){
            l = parse_region(options, params);
        }else if(lt == YOLO){
            l = parse_yolo(options, params);
        }else if(lt == ISEG){
            l = parse_iseg(options, params);
        }else if(lt == DETECTION){
            l = parse_detection(options, params);
        }else if(lt == SOFTMAX){
            l = parse_softmax(options, params);
            net->hierarchy = l.softmax_tree;
        }else if(lt == NORMALIZATION){
            l = parse_normalization(options, params);
        }else if(lt == BATCHNORM){
            l = parse_batchnorm(options, params);
        }else if(lt == MAXPOOL){
            l = parse_maxpool(options, params);
        }else if(lt == REORG){
            l = parse_reorg(options, params);
        }else if(lt == AVGPOOL){
            l = parse_avgpool(options, params);
        }else if(lt == ROUTE){
            l = parse_route(options, params, net);
        }else if(lt == UPSAMPLE){
            l = parse_upsample(options, params, net);
        }else if(lt == SHORTCUT){
            l = parse_shortcut(options, params, net);
        }else if(lt == DROPOUT){
            l = parse_dropout(options, params);
            l.output = net->layers[count-1].output;
            l.delta = net->layers[count-1].delta;
        }else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        l.clip = net->clip;
        l.truth = option_find_int_quiet(options, "truth", 0);
        l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
        l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
        l.dontsave = option_find_int_quiet(options, "dontsave", 0);
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.numload = option_find_int_quiet(options, "numload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        l.smooth = option_find_float_quiet(options, "smooth", 0);
        option_unused(options);
        net->layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        free_section(s);
        n = n->next;
        ++count;
        if(n){
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }
    free_list(sections);
    layer out = get_network_output_layer(net);
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    net->truth = calloc(net->truths*net->batch, sizeof(float));

    if(workspace_size){
        //printf("%ld\n", workspace_size);

        net->workspace = calloc(1, workspace_size);
    }
    return net;
}
```

함수 이름: parse\_network\_cfg

입력:

* filename (str): 구성 파일의 경로

동작:

* filename에서 구성 파일을 읽고, 각 섹션을 나타내는 리스트를 반환합니다.
* 각 섹션에 대해 이 함수는 parse\_xxx 함수 중 적절한 함수를 호출하여 각 레이어를 생성합니다. (xxx는 레이어 유형에 대한 이름입니다.)
* 각 레이어는 출력차원, 필터 수, 크기, 입력 및 출력 차원 등과 같은 레이어의 세부 정보를 인쇄합니다.
* 레이어 생성 후, 레이어를 네트워크 구조에 저장합니다.

설명:&#x20;

* parse\_network\_cfg 함수는 Darknet의 네트워크를 구성 파일에서 생성하는 역할을 합니다.&#x20;
* 이 함수는 filename에서 구성 파일을 읽어들인 후, 각 섹션을 나타내는 리스트를 반환합니다. 그리고 각 섹션에 대해서 parse\_xxx 함수 중 적절한 함수를 호출하여 각 레이어를 생성합니다.&#x20;
* 이때 params 구조체에는 네트워크에 대한 입력 크기, 배치 크기 등의 여러 매개변수가 저장되어 있습니다.&#x20;
* 생성된 레이어는 출력차원, 필터 수, 크기, 입력 및 출력 차원 등의 레이어의 세부 정보를 인쇄하며, 생성된 레이어는 네트워크 구조에 저장됩니다.
* params (Params 구조체): 네트워크에 대한 입력 크기, 배치 크기 등과 같은 여러 매개변수



## read\_cfg

```c
list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");             
    if(file == 0) file_error(filename);             
    char *line;                                     
    int nu = 0;
    list *options = make_list();                    
    section *current = 0;                           
    while((line=fgetl(file)) != 0){                
        ++ nu;
        strip(line);                                
        switch(line[0]){                            
            case '[':                               
                current = malloc(sizeof(section));  
                list_insert(options, current);      
                current->options = make_list();     
                current->type = line;                           
                break;                              
            case '\0':                              
            case '#':                               
            case ';':                               
                free(line);                         
                break;
            default:                               
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}
```

함수 이름: read\_cfg&#x20;

입력:

* filename (char\*): 설정 파일의 경로와 이름을 가리키는 문자열 포인터

동작:

* filename으로 지정된 설정 파일을 열고, 파일을 한 줄씩 읽어들입니다.
* 읽어들인 줄의 첫 번째 문자에 따라 해당 줄이 section, 주석, 비어있는 줄, 또는 option인지를 판단합니다.
* section인 경우 새로운 section을 생성하고 options 리스트를 할당합니다.
* option인 경우 현재 section의 options 리스트에 해당 option을 추가합니다.
* 설정 파일을 모두 읽어들인 후, 생성된 모든 section과 option을 담은 리스트를 반환합니다.

설명:&#x20;

* 이 함수는 설정 파일을 읽어들이는 기능을 담당합니다. 설정 파일은 section과 option으로 구성되며, section은 대괄호로 둘러싸인 문자열로 구분됩니다. 각 section은 해당 section에 속한 option들을 가질 수 있습니다. 각 option은 option의 이름과 값을 가지며, 이름과 값을 구분하는 문자는 등호(=)입니다.
* 이 함수는 filename으로 지정된 설정 파일을 열고, 파일을 한 줄씩 읽어들입니다. 읽어들인 줄의 첫 번째 문자에 따라 해당 줄이 section, 주석, 비어있는 줄, 또는 option인지를 판단합니다. section인 경우 새로운 section을 생성하고 options 리스트를 할당합니다. option인 경우 현재 section의 options 리스트에 해당 option을 추가합니다. 설정 파일을 모두 읽어들인 후, 생성된 모든 section과 option을 담은 리스트를 반환합니다.
* 함수에서는 설정 파일에서 읽어들인 각 줄을 strip 함수를 이용하여 공백 문자를 제거한 후 처리합니다. 만약 option을 처리하는 과정에서 해당 줄을 올바르게 처리하지 못하면, 에러 메시지를 출력하고 해당 줄을 제외합니다.
* 함수에서는 options 리스트를 생성하고 이를 담은 section 구조체를 생성합니다. 생성된 section 구조체는 options 리스트와 section 이름을 가집니다. 모든 section 구조체와 option 구조체는 make\_list 함수를 이용하여 리스트로 구성됩니다.



## parse\_net\_options

```c
void parse_net_options(list *options, network *net)
{
    net->batch = option_find_int(options, "batch",1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions",1);
    net->time_steps = option_find_int_quiet(options, "time_steps",1);
    net->notruth = option_find_int_quiet(options, "notruth",0);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;
    net->random = option_find_int_quiet(options, "random", 0);

    net->adam = option_find_int_quiet(options, "adam", 0);
    if(net->adam){
        net->B1 = option_find_float(options, "B1", .9);
        net->B2 = option_find_float(options, "B2", .999);
        net->eps = option_find_float(options, "eps", .0000001);
    }

    net->h = option_find_int_quiet(options, "height",0);
    net->w = option_find_int_quiet(options, "width",0);
    net->c = option_find_int_quiet(options, "channels",0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, "max_crop",net->w*2);
    net->min_crop = option_find_int_quiet(options, "min_crop",net->w);
    net->max_ratio = option_find_float_quiet(options, "max_ratio", (float) net->max_crop / net->w);
    net->min_ratio = option_find_float_quiet(options, "min_ratio", (float) net->min_crop / net->w);
    net->center = option_find_int_quiet(options, "center",0);
    net->clip = option_find_float_quiet(options, "clip", 0);

    net->angle = option_find_float_quiet(options, "angle", 0);
    net->aspect = option_find_float_quiet(options, "aspect", 1);
    net->saturation = option_find_float_quiet(options, "saturation", 1);
    net->exposure = option_find_float_quiet(options, "exposure", 1);
    net->hue = option_find_float_quiet(options, "hue", 0);

    if(!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");

    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, "burn_in", 0);
    net->power = option_find_float_quiet(options, "power", 4);
    if(net->policy == STEP){
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
    } else if (net->policy == STEPS){
        char *l = option_find(options, "steps");
        char *p = option_find(options, "scales");
        if(!l || !p) error("STEPS policy must have steps and scales in cfg file");

        int len = strlen(l);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (l[i] == ',') ++n;
        }
        int *steps = calloc(n, sizeof(int));
        float *scales = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            int step    = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',')+1;
            p = strchr(p, ',')+1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    } else if (net->policy == EXP){
        net->gamma = option_find_float(options, "gamma", 1);
    } else if (net->policy == SIG){
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    } else if (net->policy == POLY || net->policy == RANDOM){
    }
    net->max_batches = option_find_int(options, "max_batches", 0);
}
```

함수 이름: parse\_net\_options&#x20;

입력:

* options: 리스트 포인터. 네트워크 설정 옵션을 담고 있다.
* net: 네트워크 구조체 포인터. 옵션에서 파싱한 값을 저장할 구조체이다.

&#x20;동작:&#x20;

* 네트워크 설정 옵션을 파싱하여 네트워크 구조체에 저장한다.&#x20;

설명:&#x20;

* 이 함수는 Darknet 프레임워크에서 네트워크 설정 옵션을 파싱하여 네트워크 구조체에 저장하는 역할을 한다.&#x20;
* 함수 내부에서는 option\_find 함수를 이용하여 옵션 값을 찾은 후, 이 값을 이용하여 네트워크 구조체에 값을 저장한다.&#x20;
* 저장하는 값은 배치 크기, 학습률, 모멘텀, 가중치 감소값 등이 있다.&#x20;
* 또한 옵션에 따라 데이터 증강 방식, 학습 정책 등도 설정할 수 있다. 함수 내부에는 if문과 switch문을 이용하여 각각의 옵션에 따른 처리를 수행한다.

