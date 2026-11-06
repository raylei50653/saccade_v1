#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <limits.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>

/* Diagnostic observer. Heap requests >= min_bytes are tracked exactly until
 * free/realloc. mmap figures are a separate API-level measure (not RSS and not
 * additive to heap). Partial/unmatched munmap makes that measure incomplete.
 * No symbol lookup/unwind is performed while the bookkeeping lock is held. */
extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t,size_t);
extern void *__libc_realloc(void*,size_t);
extern void __libc_free(void*);
#define CAP (1u<<19)
#define OWNERS 2048u
#define DEPTH 12
#define TOMBSTONE ((uintptr_t)1)
#define STEP (128ull<<20)
enum { HEAP=1, MMAP=2 };
typedef struct { uintptr_t ptr; uint64_t bytes; uint32_t owner,kind; } Entry;
typedef struct {
 uintptr_t caller, stack[DEPTH];
 uint64_t live,peak,allocated,freed,alloc_calls,free_calls,max_alloc;
 unsigned kind,depth,ready;
} Owner;
static Entry *table;
static Owner owners[OWNERS];
static atomic_flag guard=ATOMIC_FLAG_INIT;
static _Atomic int init_state;
static __thread int inside;
static int enabled, logfd=-1;
static pid_t trace_pid;
static size_t min_bytes=65536;
static uint64_t live[3],peak[3],allocated[3],freed_bytes[3],active,dropped,owner_dropped,unmatched_unmap,partial_unmap,next_snapshot=STEP;
static char trace_dir[PATH_MAX];
static void *(*next_malloc)(size_t);
static void *(*next_calloc)(size_t,size_t);
static void *(*next_realloc)(void*,size_t);
static void (*next_free)(void*);
static void *(*next_mmap)(void*,size_t,int,int,int,off_t);
static int (*next_munmap)(void*,size_t);
static void lock(void) { while(atomic_flag_test_and_set_explicit(&guard,memory_order_acquire)) {} }
static void unlock(void) { atomic_flag_clear_explicit(&guard,memory_order_release); }
static void write_raw(const char *s,size_t n) { if(logfd>=0) { while(n) { ssize_t k=syscall(SYS_write,logfd,s,n); if(k<=0)break;s+=k;n-=(size_t)k; } } }
static void maps_dump(void) {
 char path[PATH_MAX+64],buf[16384];
 snprintf(path,sizeof path,"%s/maps-%ld.txt",trace_dir,(long)trace_pid);
 int out=(int)syscall(SYS_openat,AT_FDCWD,path,O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC,0600);
 int in=(int)syscall(SYS_openat,AT_FDCWD,"/proc/self/maps",O_RDONLY|O_CLOEXEC,0);
 if(in>=0&&out>=0) { ssize_t n; while((n=syscall(SYS_read,in,buf,sizeof buf))>0) { ssize_t pos=0;while(pos<n){ssize_t w=syscall(SYS_write,out,buf+pos,(size_t)(n-pos));if(w<=0)break;pos+=w;} } }
 if(in>=0)syscall(SYS_close,in);if(out>=0)syscall(SYS_close,out);
}
static void snapshot_locked(const char *reason) {
 struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);uint64_t ns=(uint64_t)ts.tv_sec*1000000000ull+(uint64_t)ts.tv_nsec;
 char b[2048]; int n=snprintf(b,sizeof b,"{\"type\":\"snapshot\",\"reason\":\"%s\",\"pid\":%ld,\"monotonic_ns\":%llu,\"heap_live\":%llu,\"heap_peak\":%llu,\"heap_allocated\":%llu,\"heap_freed\":%llu,\"mmap_live\":%llu,\"mmap_peak\":%llu,\"mmap_allocated\":%llu,\"mmap_freed\":%llu,\"active\":%llu,\"dropped\":%llu,\"owner_dropped\":%llu,\"unmatched_munmap\":%llu,\"partial_munmap\":%llu}\n",reason,(long)trace_pid,(unsigned long long)ns,(unsigned long long)live[HEAP],(unsigned long long)peak[HEAP],(unsigned long long)allocated[HEAP],(unsigned long long)freed_bytes[HEAP],(unsigned long long)live[MMAP],(unsigned long long)peak[MMAP],(unsigned long long)allocated[MMAP],(unsigned long long)freed_bytes[MMAP],(unsigned long long)active,(unsigned long long)dropped,(unsigned long long)owner_dropped,(unsigned long long)unmatched_unmap,(unsigned long long)partial_unmap);
 if(n>0)write_raw(b,(size_t)n);
 unsigned top[16];memset(top,0,sizeof top);
 for(unsigned i=1;i<OWNERS;i++)if(owners[i].caller) {
  for(unsigned j=0;j<16;j++)if(!top[j]||owners[i].live>owners[top[j]].live) { for(unsigned k=15;k>j;k--)top[k]=top[k-1];top[j]=i;break; }
 }
 for(unsigned j=0;j<16&&top[j];j++) { unsigned i=top[j];Owner *o=&owners[i];n=snprintf(b,sizeof b,"{\"type\":\"owner\",\"id\":%u,\"kind\":\"%s\",\"caller\":\"0x%lx\",\"live\":%llu,\"peak\":%llu,\"allocated\":%llu,\"freed\":%llu,\"alloc_calls\":%llu,\"free_calls\":%llu,\"max_allocation\":%llu}\n",i,o->kind==HEAP?"heap":"mmap",(unsigned long)o->caller,(unsigned long long)o->live,(unsigned long long)o->peak,(unsigned long long)o->allocated,(unsigned long long)o->freed,(unsigned long long)o->alloc_calls,(unsigned long long)o->free_calls,(unsigned long long)o->max_alloc);if(n>0)write_raw(b,(size_t)n); }
 maps_dump();
}
static void child_after_fork(void) { enabled=0; atomic_flag_clear(&guard); inside=0; }
static void resolve(void) {
 if(atomic_load_explicit(&init_state,memory_order_acquire)==2)return;
 int expected=0;if(!atomic_compare_exchange_strong(&init_state,&expected,1))return;
 inside++;
 next_malloc=dlsym(RTLD_NEXT,"malloc");next_calloc=dlsym(RTLD_NEXT,"calloc");next_realloc=dlsym(RTLD_NEXT,"realloc");next_free=dlsym(RTLD_NEXT,"free");next_mmap=dlsym(RTLD_NEXT,"mmap");next_munmap=dlsym(RTLD_NEXT,"munmap");
 const char *dir=getenv("RACE_ALLOC_TRACE_DIR");const char *minimum=getenv("RACE_ALLOC_MIN_BYTES");
 if(minimum&&*minimum){unsigned long long v=strtoull(minimum,0,10);if(v>=4096&&v<=(1ull<<30))min_bytes=(size_t)v;}
 if(dir&&*dir&&strlen(dir)<sizeof trace_dir) {
  strcpy(trace_dir,dir);trace_pid=getpid();char path[PATH_MAX+64];snprintf(path,sizeof path,"%s/alloc-%ld.jsonl",trace_dir,(long)trace_pid);logfd=(int)syscall(SYS_openat,AT_FDCWD,path,O_WRONLY|O_CREAT|O_APPEND|O_CLOEXEC,0600);
  table=(Entry*)syscall(SYS_mmap,0,CAP*sizeof(Entry),PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  if(logfd>=0&&table!=MAP_FAILED) { enabled=1;char b[512];int n=snprintf(b,sizeof b,"{\"type\":\"header\",\"pid\":%ld,\"min_bytes\":%zu,\"entry_capacity\":%u,\"table_bytes\":%zu,\"owner_capacity\":%u,\"stack_depth\":%d,\"notes\":\"heap requested bytes >=threshold only; mmap API figures separate, not RSS; no direct-syscall coverage\"}\n",(long)trace_pid,min_bytes,CAP,CAP*sizeof(Entry),OWNERS,DEPTH);write_raw(b,(size_t)n);maps_dump(); }
 }
 pthread_atfork(0,0,child_after_fork);atomic_store_explicit(&init_state,2,memory_order_release);inside--;
}
static unsigned owner_get(uintptr_t pc,unsigned kind,int *new_site) {
 unsigned i=(unsigned)(((pc>>4)^(pc>>17)^kind*2654435761u)&(OWNERS-1));if(!i)i=1;
 for(unsigned probe=0;probe<OWNERS-1;probe++) { Owner *o=&owners[i];if(o->caller==pc&&o->kind==kind)return i;if(!o->caller){o->caller=pc;o->kind=kind;*new_site=1;return i;}i++;if(i==OWNERS)i=1; }
 owner_dropped++;return 0;
}
static size_t entry_hash(uintptr_t p,unsigned kind) { return (size_t)(((p>>4)*11400714819323198485ull+(uint64_t)kind*0x9e3779b9u)&(CAP-1)); }
static Entry *find(uintptr_t p,unsigned kind) {
 size_t i=entry_hash(p,kind);for(size_t k=0;k<CAP;k++,i=(i+1)&(CAP-1)) {Entry *e=&table[i];if(!e->ptr)return 0;if(e->ptr==p&&e->kind==kind)return e;}return 0;
}
static int insert(uintptr_t p,uint64_t bytes,unsigned owner,unsigned kind) {
 size_t i=entry_hash(p,kind);Entry *vacant=0;
 for(size_t k=0;k<CAP;k++,i=(i+1)&(CAP-1)) {Entry *e=&table[i];if(e->ptr==TOMBSTONE&&!vacant)vacant=e;if(!e->ptr){if(!vacant)vacant=e;break;} }
 if(!vacant){dropped++;return 0;}*vacant=(Entry){p,bytes,owner,kind};active++;return 1;
}
static void subtract_entry(Entry *e,uint64_t amount) {
 unsigned k=e->kind;Owner *o=&owners[e->owner];live[k]-=amount;freed_bytes[k]+=amount;o->live-=amount;o->freed+=amount;o->free_calls++;
}
static void remove_alloc(void *p,unsigned kind) {
 if(!enabled||!p)return;lock();Entry *e=find((uintptr_t)p,kind);if(e){subtract_entry(e,e->bytes);e->ptr=TOMBSTONE;active--;}unlock();
}
static void record_site(unsigned id) {
 if(!id)return;void *stack[DEPTH];int depth=backtrace(stack,DEPTH);char b[2048];size_t used=0;
 lock();Owner *o=&owners[id];if(!o->ready) {
  o->depth=(unsigned)depth;for(int j=0;j<depth;j++)o->stack[j]=(uintptr_t)stack[j];o->ready=1;
  used=(size_t)snprintf(b,sizeof b,"{\"type\":\"site\",\"id\":%u,\"kind\":\"%s\",\"caller\":\"0x%lx\",\"stack\":[",id,o->kind==HEAP?"heap":"mmap",(unsigned long)o->caller);
  for(int j=0;j<depth&&used+32<sizeof b;j++)used+=(size_t)snprintf(b+used,sizeof b-used,"%s\"0x%lx\"",j?",":"",(unsigned long)o->stack[j]);
  if(used+4<sizeof b){b[used++]=']';b[used++]='}';b[used++]='\n';write_raw(b,used);}
 }unlock();
}
static void record_alloc(void *p,size_t bytes,unsigned kind,uintptr_t pc) {
 if(!enabled||!p||p==MAP_FAILED||bytes<min_bytes)return;
 int fresh=0;lock();unsigned id=owner_get(pc,kind,&fresh);if(insert((uintptr_t)p,bytes,id,kind)) {
  Owner *o=&owners[id];live[kind]+=bytes;allocated[kind]+=bytes;if(live[kind]>peak[kind])peak[kind]=live[kind];o->live+=bytes;o->allocated+=bytes;o->alloc_calls++;if(o->live>o->peak)o->peak=o->live;if(bytes>o->max_alloc)o->max_alloc=bytes;
  uint64_t total=live[HEAP]+live[MMAP];if(total>=next_snapshot){next_snapshot=(total/STEP+1)*STEP;snapshot_locked("growth");}
 }unlock();if(fresh)record_site(id);
}
__attribute__((constructor)) static void start(void) { resolve(); }
__attribute__((destructor)) static void finish(void) { if(enabled&&!inside){inside++;lock();snapshot_locked("destructor");unlock();inside--;} }
void race_alloc_trace_snapshot(void) { if(enabled&&!inside){inside++;lock();snapshot_locked("explicit");unlock();inside--;} }
void *malloc(size_t n) { uintptr_t pc=(uintptr_t)__builtin_return_address(0);if(!inside)resolve();int ready=atomic_load_explicit(&init_state,memory_order_acquire)==2;void *p=(ready&&next_malloc?next_malloc:__libc_malloc)(n);if(!inside&&ready){inside++;record_alloc(p,n,HEAP,pc);inside--;}return p; }
void *calloc(size_t a,size_t b) { uintptr_t pc=(uintptr_t)__builtin_return_address(0);if(!inside)resolve();int ready=atomic_load_explicit(&init_state,memory_order_acquire)==2;void *p=(ready&&next_calloc?next_calloc:__libc_calloc)(a,b);if(!inside&&ready&&(!b||a<=SIZE_MAX/b)){inside++;record_alloc(p,a*b,HEAP,pc);inside--;}return p; }
void free(void *p) { if(!inside)resolve();int ready=atomic_load_explicit(&init_state,memory_order_acquire)==2;if(!inside&&ready){inside++;remove_alloc(p,HEAP);inside--;}if(ready&&next_free)next_free(p);else __libc_free(p); }
void *realloc(void *p,size_t n) {
 uintptr_t pc=(uintptr_t)__builtin_return_address(0);if(!inside)resolve();int ready=atomic_load_explicit(&init_state,memory_order_acquire)==2;void *q=(ready&&next_realloc?next_realloc:__libc_realloc)(p,n);
 if(!inside&&ready&&(q||n==0)){inside++;remove_alloc(p,HEAP);record_alloc(q,n,HEAP,pc);inside--;}return q;
}
void *mmap(void *p,size_t n,int prot,int flags,int fd,off_t off) {
 uintptr_t pc=(uintptr_t)__builtin_return_address(0);if(!inside)resolve();int ready=atomic_load_explicit(&init_state,memory_order_acquire)==2;void *q=ready&&next_mmap?next_mmap(p,n,prot,flags,fd,off):(void*)syscall(SYS_mmap,p,n,prot,flags,fd,off);
 if(!inside&&ready){inside++;record_alloc(q,n,MMAP,pc);inside--;}return q;
}
void *mmap64(void *p,size_t n,int prot,int flags,int fd,off64_t off) { return mmap(p,n,prot,flags,fd,(off_t)off); }
int munmap(void *p,size_t n) {
 if(!inside)resolve();int ready=atomic_load_explicit(&init_state,memory_order_acquire)==2;int rc=ready&&next_munmap?next_munmap(p,n):(int)syscall(SYS_munmap,p,n);
 if(!inside&&ready&&enabled&&rc==0){inside++;lock();Entry *e=find((uintptr_t)p,MMAP);
  if(e&&n<=e->bytes){uint64_t remain=e->bytes-n;unsigned owner=e->owner;subtract_entry(e,n);e->ptr=TOMBSTONE;active--;if(remain){partial_unmap++;insert((uintptr_t)p+n,remain,owner,MMAP);}}
  else unmatched_unmap++;
  unlock();inside--;}
 return rc;
}
