# tree

## change\_leaves

```c
void change_leaves(tree *t, char *leaf_list)
{
    list *llist = get_paths(leaf_list);
    char **leaves = (char **)list_to_array(llist);
    int n = llist->size;
    int i,j;
    int found = 0;
    for(i = 0; i < t->n; ++i){
        t->leaf[i] = 0;
        for(j = 0; j < n; ++j){
            if (0==strcmp(t->name[i], leaves[j])){
                t->leaf[i] = 1;
                ++found;
                break;
            }
        }
    }
    fprintf(stderr, "Found %d leaves.\n", found);
}
```

함수 이름: change\_leaves

입력:

* t: 트리를 나타내는 구조체 포인터
* leaf\_list: 새로운 leaf 노드의 이름들을 담고 있는 문자열

동작:

* leaf\_list에 있는 leaf 노드들을 t의 leaf 노드로 변경
* t 구조체의 leaf 배열을 업데이트하여 leaf 노드를 표시
* 변경된 leaf 노드의 수를 출력

설명:

* 이 함수는 YOLO 객체 검출에서 사용되는 트리 구조체를 변경하는 함수입니다. 이 함수는 leaf\_list에 있는 leaf 노드들을 t 구조체의 leaf 노드로 변경합니다.
* 먼저, leaf\_list에서 leaf 노드의 이름을 가져와서 배열에 저장합니다.
* 그런 다음, t 구조체의 모든 노드를 확인하면서, leaf\_list에 있는 leaf 노드의 이름과 일치하는 노드가 있는 경우, 해당 노드를 leaf 노드로 표시합니다.
* 마지막으로, 변경된 leaf 노드의 수를 출력합니다.



## get\_hierarchy\_probability

```c
float get_hierarchy_probability(float *x, tree *hier, int c, int stride)
{
    float p = 1;
    while(c >= 0){
        p = p * x[c*stride];
        c = hier->parent[c];
    }
    return p;
}
```

함수 이름: get\_hierarchy\_probability

입력:

* x: 신경망 출력값 (1차원 실수 배열)
* hier: 계층 구조를 표현하는 트리
* c: 예측 클래스 인덱스
* stride: x 배열에서 한 클래스를 표현하기 위해 사용되는 요소 수

동작:

* 계층 구조 트리를 사용하여 예측 클래스의 계층 확률을 계산한다.
* 예측 클래스의 계층 구조를 따라 상위 클래스의 확률을 하위 클래스 확률에 곱해 최종 확률 값을 구한다.

설명:

* 이 함수는 계층 구조를 사용하여 다단계 객체 인식에서 예측된 클래스에 대한 계층 확률을 계산하는 데 사용된다.
* 계층 구조는 트리 형태로 표현되며, 이 트리는 부모-자식 관계를 나타낸다.
* 입력으로 주어진 c 인덱스는 예측된 클래스의 인덱스를 나타낸다.
* 계층 구조를 따라서, c 인덱스의 클래스에 해당하는 노드에서 루트 노드까지의 경로 상의 각 노드의 값을 곱해 최종 확률 값을 구한다.
* 이때, 입력 배열 x에서 한 클래스를 표현하기 위해 사용되는 요소 수를 stride로 나타낸다.



## hierarchy\_predictions

```c
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride)
{
    int j;
    for(j = 0; j < n; ++j){
        int parent = hier->parent[j];
        if(parent >= 0){
            predictions[j*stride] *= predictions[parent*stride];
        }
    }
    if(only_leaves){
        for(j = 0; j < n; ++j){
            if(!hier->leaf[j]) predictions[j*stride] = 0;
        }
    }
}
```

함수 이름: hierarchy\_predictions

입력:

* float \*predictions: 예측값 배열
* int n: 예측값 배열의 길이
* tree \*hier: 계층 구조 정보를 담은 tree 구조체
* int only\_leaves: leaf 노드들만 사용할지 여부 (1: leaf 노드만 사용, 0: 전체 노드 사용)
* int stride: 예측값 배열에서 노드 하나를 표현하는데 필요한 원소 수

동작:

* 계층 구조를 이용하여 예측값을 보정한다.
* 예측값이 담긴 배열 predictions을 입력으로 받아, 계층 구조를 따라 예측값을 보정한다. 예를 들어, 만약 j번째 노드가 parent노드의 child 노드라면, j번째 노드에 해당하는 예측값은 j번째 노드에 대한 예측값과 parent 노드에 대한 예측값의 곱으로 계산된다.
* 만약 only\_leaves가 1로 설정되어 있으면, leaf 노드 이외의 노드들에 해당하는 예측값은 0으로 설정된다.

설명:&#x20;

* 이 함수는 계층 구조 정보를 이용하여 예측값을 보정하는 작업을 수행한다.&#x20;
* 이때 계층 구조 정보는 tree 구조체에 저장되어 있다.&#x20;
* 계층 구조를 고려하여 예측값을 보정하면, 예측 성능을 개선할 수 있다.&#x20;
* 보정된 예측값은 이후 후처리 과정에서 사용된다.



## hierarchy\_top\_prediction

```c
int hierarchy_top_prediction(float *predictions, tree *hier, float thresh, int stride)
{
    float p = 1;
    int group = 0;
    int i;
    while(1){
        float max = 0;
        int max_i = 0;

        for(i = 0; i < hier->group_size[group]; ++i){
            int index = i + hier->group_offset[group];
            float val = predictions[(i + hier->group_offset[group])*stride];
            if(val > max){
                max_i = index;
                max = val;
            }
        }
        if(p*max > thresh){
            p = p*max;
            group = hier->child[max_i];
            if(hier->child[max_i] < 0) return max_i;
        } else if (group == 0){
            return max_i;
        } else {
            return hier->parent[hier->group_offset[group]];
        }
    }
    return 0;
}
```

함수 이름: hierarchy\_top\_prediction

입력:

* predictions: 예측값을 담은 실수형 배열
* hier: tree 형태의 계층 구조
* thresh: 임계값
* stride: 배열의 간격

동작:

* 계층 구조를 따라 최상위 예측값을 찾아 반환하는 함수입니다.
* 예측값 배열과 계층 구조를 입력받아 계층 구조를 따라 최상위 예측값을 찾습니다.
* 계층 구조는 tree 형태로 표현되며, 각각의 노드는 자식 노드를 가질 수 있습니다.
* 임계값(thresh)보다 큰 값 중 가장 큰 값을 갖는 자식 노드를 찾습니다.
* 최상위 노드까지 찾은 경우 해당 자식 노드의 인덱스를 반환합니다.

설명:

* predictions 배열은 계층 구조를 고려한 예측값을 포함합니다.
* 계층 구조를 따라 최상위 예측값을 찾을 때는 각 노드의 값을 곱해가면서 탐색합니다.
* group\_size와 group\_offset을 사용하여 각각의 자식 노드를 그룹으로 나누어 최대값을 찾습니다.
* p \* max > thresh를 만족하는 경우 자식 노드로 이동합니다.
* group == 0인 경우 최상위 노드에 도달했음을 의미합니다.
* p \* max <= thresh를 만족하는 경우 현재 그룹의 부모 노드로 이동합니다.
* 마지막으로 계층 구조를 따라 찾은 최상위 노드의 인덱스를 반환합니다.



## read\_tree

```c
tree *read_tree(char *filename)
{
    tree t = {0};
    FILE *fp = fopen(filename, "r");

    char *line;
    int last_parent = -1;
    int group_size = 0;
    int groups = 0;
    int n = 0;
    while((line=fgetl(fp)) != 0){
        char *id = calloc(256, sizeof(char));
        int parent = -1;
        sscanf(line, "%s %d", id, &parent);
        t.parent = realloc(t.parent, (n+1)*sizeof(int));
        t.parent[n] = parent;

        t.child = realloc(t.child, (n+1)*sizeof(int));
        t.child[n] = -1;

        t.name = realloc(t.name, (n+1)*sizeof(char *));
        t.name[n] = id;
        if(parent != last_parent){
            ++groups;
            t.group_offset = realloc(t.group_offset, groups * sizeof(int));
            t.group_offset[groups - 1] = n - group_size;
            t.group_size = realloc(t.group_size, groups * sizeof(int));
            t.group_size[groups - 1] = group_size;
            group_size = 0;
            last_parent = parent;
        }
        t.group = realloc(t.group, (n+1)*sizeof(int));
        t.group[n] = groups;
        if (parent >= 0) {
            t.child[parent] = groups;
        }
        ++n;
        ++group_size;
    }
    ++groups;
    t.group_offset = realloc(t.group_offset, groups * sizeof(int));
    t.group_offset[groups - 1] = n - group_size;
    t.group_size = realloc(t.group_size, groups * sizeof(int));
    t.group_size[groups - 1] = group_size;
    t.n = n;
    t.groups = groups;
    t.leaf = calloc(n, sizeof(int));
    int i;
    for(i = 0; i < n; ++i) t.leaf[i] = 1;
    for(i = 0; i < n; ++i) if(t.parent[i] >= 0) t.leaf[t.parent[i]] = 0;

    fclose(fp);
    tree *tree_ptr = calloc(1, sizeof(tree));
    *tree_ptr = t;
    //error(0);
    return tree_ptr;
}
```

함수 이름: read\_tree

입력:&#x20;

* char \*filename: 트리 구조를 저장한 파일 이름

동작:&#x20;

* 입력 파일에서 트리 구조를 읽어들이고, 해당 트리 구조를 표현하는 tree 구조체를 생성한다.

설명:&#x20;

* 입력 파일에서 트리 구조를 읽어들이는 과정에서는, 각 노드의 이름과 부모 노드의 인덱스를 읽어들이고, 이를 이용하여 parent, child, name, group, group\_offset, group\_size, leaf 등의 필드를 초기화한다.&#x20;
* 이후 생성된 tree 구조체를 가리키는 포인터를 반환한다.

