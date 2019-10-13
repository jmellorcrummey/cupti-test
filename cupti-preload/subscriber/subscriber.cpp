/*
 * Preloaded module that does nothing more than subscribe CUDA APIs
 *
 * This code will work on devices with compute capability 5.2
 * or 6.0 and higher.
 */

#include <cuda.h>
#include <cupti.h>
#include <dlfcn.h>

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include <atomic>


//************************************************************************
// macros
//************************************************************************

#define DRIVER_API_CALL(apiFuncCall)                                         \
  do {                                                                       \
    CUresult _status = apiFuncCall;                                          \
    if (_status != CUDA_SUCCESS) {                                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
        __FILE__, __LINE__, #apiFuncCall, _status);                          \
      exit(-1);                                                              \
    }                                                                        \
  } while (0)

#define RUNTIME_API_CALL(apiFuncCall)					                              \
  do {									                                                    \
    cudaError_t _status = apiFuncCall;					                            \
    if (_status != cudaSuccess) {					                                  \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
	      __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));     \
      exit(-1);								                                              \
    }									                                                      \
  } while (0)

#define CUPTI_CALL(call)						                                        \
  do {									                                                    \
    CUptiResult _status = call;						                                  \
    if (_status != CUPTI_SUCCESS) {					                                \
      const char *errstr;						                                        \
      cuptiGetResultString(_status, &errstr);				                        \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
	      __FILE__, __LINE__, #call, errstr);			                            \
      exit(-1);								                                              \
    }									                                                      \
  } while (0)


#define stringify(x) #x

#define DEBUGGER_WAIT_FLAG_DEFAULT 0
#define DEBUGGER_PRINT_FLAG_DEFAULT 1
  
//************************************************************************
// variables
//************************************************************************

// control flags
volatile int debugger_wait_flag = DEBUGGER_WAIT_FLAG_DEFAULT; 
volatile int doprint = DEBUGGER_PRINT_FLAG_DEFAULT;

CUpti_SubscriberHandle cupti_subscriber;
std::atomic<uint64_t> callback_count;

//************************************************************************
// forward declarations
//************************************************************************

#if 1
// ensure cupti_fini runs at library unload
static void cupti_fini() __attribute__((destructor));
// ensure cupti_init runs at library load
static void cupti_init() __attribute__((constructor));
#endif


//************************************************************************
// debugging support
//************************************************************************

static void
debugger_wait()
{
  while (debugger_wait_flag);
}


void
debugger_continue()
{
  debugger_wait_flag = 0;
}

//************************************************************************
// private operations
//************************************************************************

static void
cupti_subscriber_callback
(
 void *userdata,
 CUpti_CallbackDomain domain,
 CUpti_CallbackId cb_id,
 const void *cb_info
)
{
  callback_count.fetch_add(1);
}

//************************************************************************
// initialization 
//************************************************************************

void
cupti_init()
{
  debugger_wait();

  CUPTI_CALL(cuptiSubscribe(&cupti_subscriber,
    (CUpti_CallbackFunc) cupti_subscriber_callback,
    (void *) NULL));

  CUPTI_CALL(cuptiEnableDomain 
    (1, cupti_subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

  CUPTI_CALL(cuptiEnableDomain 
    (1, cupti_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));

  CUPTI_CALL(cuptiEnableDomain 
    (1, cupti_subscriber, CUPTI_CB_DOMAIN_RESOURCE));
}



//************************************************************************
// finalization 
//************************************************************************

void
cupti_fini()
{
  if (doprint) {
    uint64_t cnt = callback_count.load();
    printf("%llu cupti callbacks\n", cnt);
  }
}