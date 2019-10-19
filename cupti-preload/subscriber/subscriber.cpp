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
#define CUPTI_DRIVER_API_DEFAULT 1
#define CUPTI_RUNTIME_API_DEFAULT 1
#define CUPTI_RESOURCE_DEFAULT 1
  
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
  if (domain == CUPTI_CB_DOMAIN_RESOURCE && cb_id == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
    const CUpti_ResourceData *rd = (const CUpti_ResourceData *) cb_info;

    cuptiActivityEnableContext(rd->context, CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityEnableContext(rd->context, CUPTI_ACTIVITY_KIND_MEMCPY2);
    cuptiActivityEnableContext(rd->context, CUPTI_ACTIVITY_KIND_MEMSET);
    cuptiActivityEnableContext(rd->context, CUPTI_ACTIVITY_KIND_CONTEXT);
    cuptiActivityEnableContext(rd->context, CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityEnableContext(rd->context, CUPTI_ACTIVITY_KIND_DEVICE);
    cuptiActivityEnableContext(rd->context, CUPTI_ACTIVITY_KIND_DRIVER);
    cuptiActivityEnableContext(rd->context, CUPTI_ACTIVITY_KIND_RUNTIME);
    cuptiActivityEnableContext(rd->context, CUPTI_ACTIVITY_KIND_SYNCHRONIZATION);
    cuptiActivityEnableContext(rd->context, CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
  }
}


static int
cupti_get_env
(
 const char *env,
 int default_value
)
{
  const char *value = getenv(env);
  int ret = default_value;

  if (value) {
    ret = atoi(value);
  }

  return ret;
}

//************************************************************************
// initialization 
//************************************************************************

void
cupti_init()
{
  debugger_wait();

  int enable_driver_api = cupti_get_env("CUPTI_DRIVER_API", CUPTI_DRIVER_API_DEFAULT);
  int enable_runtime_api = cupti_get_env("CUPTI_RUNTIME_API", CUPTI_RUNTIME_API_DEFAULT);
  int enable_resource = cupti_get_env("CUPTI_RESOURCE", CUPTI_RESOURCE_DEFAULT);

  CUPTI_CALL(cuptiSubscribe(&cupti_subscriber,
    (CUpti_CallbackFunc) cupti_subscriber_callback,
    (void *) NULL));

  CUPTI_CALL(cuptiEnableDomain 
    (enable_driver_api, cupti_subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

  CUPTI_CALL(cuptiEnableDomain 
    (enable_runtime_api, cupti_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));

  CUPTI_CALL(cuptiEnableDomain 
    (enable_resource, cupti_subscriber, CUPTI_CB_DOMAIN_RESOURCE));
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
