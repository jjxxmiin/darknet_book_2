# option\_list

## read\_data\_cfg

```c
list *read_data_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, options)){
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

함수 이름: read\_data\_cfg

입력:&#x20;

* filename (char \*): 읽을 파일의 이름

동작:&#x20;

* 지정된 파일에서 데이터 구성 파일을 읽어들이고 각 설정 옵션을 구문 분석하여 연결 리스트로 반환한다.

설명:

* 지정된 파일을 열고 파일을 성공적으로 열지 못한 경우 오류 메시지를 출력한다.
* 파일에서 한 줄씩 읽으며 각 줄의 첫 문자를 확인하여 옵션을 구문 분석한다.
* 읽은 옵션을 옵션 연결 리스트에 추가하고 이를 반환한다.



## get\_metadata

```c
metadata get_metadata(char *file)
{
    metadata m = {0};
    list *options = read_data_cfg(file);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", 0);
    if(!name_list) {
        fprintf(stderr, "No names or labels found\n");
    } else {
        m.names = get_labels(name_list);
    }
    m.classes = option_find_int(options, "classes", 2);
    free_list(options);
    return m;
}
```

함수 이름: get\_metadata

입력:&#x20;

* char \*file (메타데이터 파일 이름)

동작:&#x20;

* 지정된 메타데이터 파일을 읽고, 이름 또는 레이블 목록을 찾아서 가져와서 metadata 구조체를 반환함.

설명:

* 함수는 metadata 구조체를 반환하며, 이 구조체는 클래스 수와 레이블 이름을 저장함.
* 함수는 지정된 메타데이터 파일을 읽어들이고, "names" 또는 "labels" 필드에서 레이블 이름을 찾음.
* 레이블 이름은 쉼표(,)로 구분된 문자열로 구성되며, get\_labels() 함수를 사용하여 리스트로 변환함.
* 함수는 "classes" 필드에서 클래스 수를 찾음.
* 함수는 메타데이터 파일에서 읽은 모든 필드를 해제함.



## read\_option

```c
int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for(i = 0; i < len; ++i){
        if(s[i] == '='){
            s[i] = '\0';
            val = s+i+1;
            break;
        }
    }
    if(i == len-1) return 0;
    char *key = s;
    option_insert(options, key, val);
    return 1;
}
```

함수 이름: read\_option

입력:&#x20;

* char 포인터 s (설정 파일에서 읽은 한 줄의 문자열)
* list 포인터 options (설정 값을 저장하는 연결 리스트)

동작:&#x20;

* 입력으로 받은 문자열 s를 key-value 쌍으로 분리하고, key와 value를 options 리스트에 추가한다.

설명:

* s 문자열에서 '=' 문자를 찾아 그 위치를 기준으로 key와 value를 구분한다.
* key와 value를 options 리스트에 추가한다.
* 설정 파일에서 한 줄을 잘못 읽거나 '=' 문자가 없는 경우에는 0을 반환하여 오류를 나타낸다.



## option\_insert

```c
void option_insert(list *l, char *key, char *val)
{
    kvp *p = malloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    list_insert(l, p);
}
```

함수 이름: option\_insert&#x20;

입력:

* l: option\_insert를 수행할 list 구조체 포인터
* key: 삽입할 key 문자열 포인터
* val: 삽입할 value 문자열 포인터

동작:&#x20;

* 주어진 key와 val을 새로운 kvp 구조체에 저장하고, used는 0으로 초기화한 뒤, list l에 새로운 kvp 구조체를 삽입한다.

설명:&#x20;

* option\_insert 함수는 key와 value를 갖는 새로운 kvp 구조체를 생성하여, 입력받은 list l에 삽입하는 함수이다.&#x20;
* kvp 구조체는 key와 val, 그리고 이 kvp가 사용되었는지를 나타내는 used 필드로 이루어져 있다.&#x20;
* option\_insert 함수는 주어진 key와 val로 새로운 kvp 구조체를 생성하고, used를 0으로 초기화한 뒤, 이를 list l에 삽입한다.



## option\_unused

```c
void option_unused(list *l)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(!p->used){
            fprintf(stderr, "Unused field: '%s = %s'\n", p->key, p->val);
        }
        n = n->next;
    }
}
```

함수 이름: option\_unused&#x20;

입력:&#x20;

* list 포인터 l&#x20;

동작:&#x20;

* l 리스트에 있는 모든 kvp(key-value pair)들 중에 사용되지 않은 kvp들을 찾아서 stderr로 출력한다.&#x20;

설명:

* 이 함수는 list l에 있는 kvp들 중에 사용되지 않은 kvp들을 찾아서 출력하는 함수이다.
* l은 linked list 구조체의 포인터이다.
* kvp 구조체는 key-value pair를 나타내는 구조체로 key와 val로 이루어져 있다.
* n은 linked list에서 현재 검사 중인 노드를 가리키는 포인터이다.
* while문은 linked list의 모든 노드를 검사한다.
* p는 현재 노드의 kvp를 가리키는 포인터이다.
* 만약 현재 kvp가 사용되지 않았으면, 해당 kvp의 key와 val을 stderr로 출력한다.
* n은 다음 노드를 가리키는 포인터로 업데이트된다.



## option\_find

```c
char *option_find(list *l, char *key)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(strcmp(p->key, key) == 0){
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }
    return 0;
}
```

함수 이름: option\_find&#x20;

입력:&#x20;

* list 포인터 l
* char 포인터 key&#x20;

동작:&#x20;

* 주어진 key로 list l에서 kvp 구조체의 key와 비교하여 일치하는 key를 찾고 해당하는 kvp 구조체의 val 포인터를 반환하고, 사용된 kvp 구조체의 used 값을 1로 설정한다.&#x20;

설명:&#x20;

* option\_find 함수는 주어진 key에 해당하는 값(val)을 찾는 함수로, 이를 위해 key-value pair(kvp) 구조체를 활용한다.&#x20;
* l은 kvp 구조체를 모아둔 list를 가리키는 포인터이며, key는 찾고자 하는 값의 key를 가리키는 포인터이다.&#x20;
* 반환값은 찾은 값(val)의 포인터이며, 해당하는 key가 없을 경우 0을 반환한다.



## option\_find\_str

```c
char *option_find_str(list *l, char *key, char *def)
{
    char *v = option_find(l, key);
    if(v) return v;
    if(def) fprintf(stderr, "%s: Using default '%s'\n", key, def);
    return def;
}
```

함수 이름: option\_find\_str

입력:

* list \*l: 옵션 리스트
* char \*key: 검색할 옵션 키
* char \*def: 옵션이 없을 경우 반환할 기본값

동작:

* 주어진 리스트에서 주어진 키를 검색하고 해당하는 값이 있다면 반환한다.
* 값이 없는 경우, 기본값(def)을 반환하고 해당하는 키와 기본값을 에러 메시지로 출력한다.

설명:

* 이 함수는 주어진 리스트에서 특정 옵션의 값을 검색하는 함수이다.
* 만약 해당하는 옵션의 값이 있다면 문자열 형태로 반환한다.
* 옵션이 없는 경우, 기본값(def)을 반환하고 해당하는 키와 기본값을 에러 메시지로 출력한다.
* 이 함수는 YOLO와 같은 딥러닝 모델에서 사용되는 옵션 값을 가져오는 데 사용된다.



## option\_find\_int

```c
int option_find_int(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    fprintf(stderr, "%s: Using default '%d'\n", key, def);
    return def;
}
```

함수 이름: option\_find\_int

입력:

* list \*l: 연결 리스트 포인터
* char \*key: 찾으려는 옵션 키 문자열 포인터
* int def: 기본값

동작:

* 입력된 연결 리스트에서 주어진 옵션 키를 찾아 해당 값의 정수형을 반환한다.
* 해당 옵션 키가 없을 경우 기본값을 반환하고 표준 오류 출력에 해당 옵션 키와 기본값을 출력한다.

설명:

* 입력된 연결 리스트는 옵션 키와 값의 쌍을 저장하고 있다.
* option\_find 함수를 이용해 주어진 옵션 키에 해당하는 값 문자열 포인터를 찾는다.
* 찾은 문자열 포인터를 atoi 함수를 이용해 정수형으로 변환하고 반환한다.
* 해당 옵션 키가 없을 경우 표준 오류 출력에 해당 옵션 키와 기본값을 출력하고 기본값을 반환한다.



## option\_find\_int\_quiet

```c
int option_find_int_quiet(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    return def;
}
```

함수 이름: option\_find\_int\_quiet

입력:

* list \*l: 설정 파일에서 읽어온 설정들이 저장된 list 구조체 포인터
* char \*key: 읽어올 설정의 이름
* int def: 설정 파일에서 해당 key에 대한 값을 찾지 못했을 경우 사용할 기본값

동작:

* 입력으로 받은 key에 해당하는 값을 설정 파일에서 찾습니다.
* 해당 값이 존재할 경우 int 형태로 변환하여 반환합니다.
* 해당 값이 존재하지 않을 경우 기본값 def를 사용합니다.
* 출력을 하지 않습니다.

설명:

* 이 함수는 설정 파일에서 int 형태의 값을 읽어오기 위해 사용됩니다.
* 입력으로 받은 설정 파일(list 구조체)에서 key에 해당하는 값을 찾습니다.
* 찾은 값을 atoi 함수를 이용하여 int 형태로 변환합니다.
* 만약 key에 해당하는 값이 존재하지 않을 경우 기본값 def를 사용합니다.
* 이 함수는 설정 파일에서 읽어온 int 값을 반환합니다.
* 만약 설정 파일에서 해당 key에 대한 값을 찾지 못했을 경우, 기본값 def를 사용합니다.
* 이 함수는 출력을 하지 않습니다.



## option\_find\_float\_quiet

```c
float option_find_float_quiet(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    return def;
}
```

함수 이름: option\_find\_float\_quiet

입력:

* list \*l: 설정 파일에서 읽어온 설정들이 저장된 list 구조체 포인터
* char \*key: 읽어올 설정의 이름
* float def: 설정 파일에서 해당 key에 대한 값을 찾지 못했을 경우 사용할 기본값

동작:

* 입력으로 받은 key에 해당하는 값을 설정 파일에서 찾습니다.
* 해당 값이 존재할 경우 float 형태로 변환하여 반환합니다.
* 해당 값이 존재하지 않을 경우 기본값 def를 사용합니다.
* 출력을 하지 않습니다.

설명:

* 이 함수는 설정 파일에서 float 형태의 값을 읽어오기 위해 사용됩니다.
* 입력으로 받은 설정 파일(list 구조체)에서 key에 해당하는 값을 찾습니다.
* 찾은 값을 atof 함수를 이용하여 float 형태로 변환합니다.
* 만약 key에 해당하는 값이 존재하지 않을 경우 기본값 def를 사용합니다.
* 이 함수는 설정 파일에서 읽어온 float 값을 반환합니다.
* 만약 설정 파일에서 해당 key에 대한 값을 찾지 못했을 경우, 기본값 def를 사용합니다.
* 이 함수는 출력을 하지 않습니다.



## option\_find\_float

```c
float option_find_float(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    fprintf(stderr, "%s: Using default '%lf'\n", key, def);
    return def;
}
```

함수 이름: option\_find\_float

입력:

* list \*l: 설정 파일에서 읽어온 설정들이 저장된 list 구조체 포인터
* char \*key: 읽어올 설정의 이름
* float def: 설정 파일에서 해당 key에 대한 값을 찾지 못했을 경우 사용할 기본값

동작:

* 입력으로 받은 key에 해당하는 값을 설정 파일에서 찾습니다.
* 해당 값이 존재할 경우 float 형태로 변환하여 반환합니다.
* 해당 값이 존재하지 않을 경우 기본값 def를 사용합니다.

설명:

* 이 함수는 설정 파일에서 float 형태의 값을 읽어오기 위해 사용됩니다.
* 입력으로 받은 설정 파일(list 구조체)에서 key에 해당하는 값을 찾습니다.
* 찾은 값을 atof 함수를 이용하여 float 형태로 변환합니다.
* 만약 key에 해당하는 값이 존재하지 않을 경우 기본값 def를 사용합니다.
* 이 함수는 설정 파일에서 읽어온 float 값을 반환합니다.
* 만약 설정 파일에서 해당 key에 대한 값을 찾지 못했을 경우, 기본값 def를 사용하고 사용한 값을 stderr에 출력합니다.

