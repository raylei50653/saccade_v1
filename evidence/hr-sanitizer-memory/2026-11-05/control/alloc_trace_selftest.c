#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
static void snap(void){ void (*f)(void)=dlsym(RTLD_DEFAULT,"race_alloc_trace_snapshot");if(!f)exit(3);f(); }
int main(void){
 volatile unsigned char *a=malloc(2u<<20),*b=calloc(1,3u<<20);
 if(!a||!b)return 2;memset((void*)a,1,2u<<20);memset((void*)b,2,3u<<20);snap();
 free((void*)a);snap();b=realloc((void*)b,4u<<20);if(!b)return 2;memset((void*)b,3,4u<<20);snap();free((void*)b);snap();
 void *p=mmap(0,5u<<20,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);if(p==MAP_FAILED)return 2;memset(p,4,5u<<20);snap();munmap(p,5u<<20);snap();
 puts("allocator tracer selftest PASS");return 0;
}
