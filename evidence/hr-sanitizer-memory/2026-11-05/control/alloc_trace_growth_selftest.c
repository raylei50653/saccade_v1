#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
__attribute__((noinline)) static void *nested(unsigned depth,size_t bytes) {
 if(depth){void *p=nested(depth-1,bytes);volatile unsigned keep=depth;(void)keep;return p;}
 return malloc(bytes);
}
int main(void){
 void (*snap)(void)=dlsym(RTLD_DEFAULT,"race_alloc_trace_snapshot");if(!snap)return 2;
 void *a=nested(24,129u<<20);if(!a)return 2;memset(a,1,129u<<20);
 void *b=nested(24,130u<<20);if(!b)return 2;memset(b,2,130u<<20);snap();
 free(a);free(b);snap();puts("allocator growth tracer selftest PASS");return 0;
}
