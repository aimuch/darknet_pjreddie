#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "option_list.h"
#include "utils.h"

/*
**  读取数据配置文件（.data文件），包含数据所在的路径、名称，其中包含的物体类别数等等
**  返回：list指针，包含所有数据信息。函数中会创建options变量，并返回其指针（若文件打开失败，将直接退出程序，不会返空指针）
**  文件举例：
**      classes= 9418
**      #train  = /home/pjreddie/data/coco/trainvalno5k.txt
**      train  = data/combine9k.train.list
**      valid  = /home/pjreddie/data/imagenet/det.val.files
*/
list *read_data_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line; // 行内容
    int nu = 0; // 行号
    list *options = make_list(); // 返回值

    // 读取文件流中的一行数据：返回C风格字符数组指针，不为空有效
    while((line=fgetl(file)) != 0){
        ++ nu;

        strip(line); // 作者在utils.c中实现了跟Python中同样的删除line中的空白符功能的函数

        switch(line[0]){

            // 以下面三种字符开头的都是无效行，直接跳过（如注释等）
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



/*
**  解析一行数据的内容，为options赋值，主要调用option_insert()
**  输入：s        从文件读入的某行字符数组指针
**       options  实际输出，解析出的数据将为该变量赋值
**  返回：int类型数据，1表示成功读取有效数据，0表示未能读取有效数据（说明文件中数据格式有问题）
**  流程：从配置（.data或者.cfg，不管是数据配置文件还是神经网络结构数据文件，其读取都需要调用这个函数）
**       文件中读入的每行数据包括两部分，第一部分为变量名称，如learning_rate，
**       第二部分为值，如0.01，两部分由 = 隔开，因此，要分别读入两部分的值，首先识别出
**       等号，获取等号所在的指针，并将等号替换为terminating null-characteristic '\0'，
**       这样第一部分会自动识别到'\0'停止，而第二部分则从等号下一个地方开始
*/
int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for(i = 0; i < len; ++i){
        // 找出 = 的偏移位置
        if(s[i] == '='){
            // 将 = 替换为'\0', 作为第一部分的终止符
            s[i] = '\0';

            // 第二部分字符数组的起始指针
            val = s+i+1;
            break;
        }
    }

    // 如果i==len-1，说明没有找到等号这个符号，那么就直接返回0（文件中还有一些注释，此外还有用[]括起来的字符，这些是网络层的类别或者名字，比如[maxpool]表示这些是池化层的参数）
    if(i == len-1) return 0;
    char *key = s;

    // 调用option_insert为options赋值，类比C++中的map数据结构，key相当是键值（变量名），而val则是值（变量的值）
    option_insert(options, key, val);
    return 1;
}

/*
**  将输入key和val赋值给kvp结构体对象，最终调用list_insert()将kvp赋值给list对象l，
**  完成最后的赋值（此函数之后，文件中某行的数据真正读入进list变量）
**  说明： 这个函数有点类似C++中按键值插入元素值的功能
**  输入：l      输出，最终被赋值的list变量
**       key    变量的名称，C风格字符数组
**       value  变量的值，C风格字符数组（还未转换成float或者double数据类型）
*/
void option_insert(list *l, char *key, char *val)
{
    // kvp也是一个结构体，包含两个C风格字符数组指针：key和val，对应键值和值，
    // 此处key为变量名，val为变量的值（比如类别数，路径名称，注意都是字符类型数据）
    kvp *p = malloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;

    // 调用list_insert函数将kvp结构体插入list中
    list_insert(l, p);
}

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
char *option_find_str(list *l, char *key, char *def)
{
    char *v = option_find(l, key);
    if(v) return v;
    if(def) fprintf(stderr, "%s: Using default '%s'\n", key, def);
    return def;
}

int option_find_int(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    fprintf(stderr, "%s: Using default '%d'\n", key, def);
    return def;
}

int option_find_int_quiet(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    return def;
}

float option_find_float_quiet(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    return def;
}

float option_find_float(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    fprintf(stderr, "%s: Using default '%lf'\n", key, def);
    return def;
}
