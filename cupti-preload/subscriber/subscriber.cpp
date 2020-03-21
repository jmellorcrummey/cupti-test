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

uint64_t droppedRecords = 0;

CUpti_SubscriberHandle cupti_subscriber;
std::atomic<uint64_t> callback_count;

//************************************************************************
// forward declarations
//************************************************************************

// ensure cupti_fini runs at library unload
static void cupti_fini() __attribute__((destructor));
// ensure cupti_init runs at library load
static void cupti_init() __attribute__((constructor));

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
  } else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
    const CUpti_CallbackData *cd = (const CUpti_CallbackData *) cb_info;
    uint64_t id = 1;
    if (cd->callbackSite == CUPTI_API_ENTER) {
      id = callback_count.fetch_add(1);
      if (doprint) {
        printf("Runtime push id %lu\n", id);
      }
      cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, id);
    } else {
      cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id);
      if (doprint) {
        printf("Runtime pop id %lu\n", id);
      }
    }
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


static void
print_activity(CUpti_Activity *record)
{
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
      {
        CUpti_ActivityExternalCorrelation *external = (CUpti_ActivityExternalCorrelation *)record;
        printf("CUPTI external ID %lu, correlation ID %u\n", external->externalId, external->correlationId);
        break;
      }
    default:
      printf("Unknown activity kind %u\n", record->kind);
      break;
  }
}

//************************************************************************
// initialization 
//************************************************************************

void
cupti_buffer_alloc
(
 uint8_t **buffer,
 size_t *buffer_size,
 size_t *maxNumRecords
)
{
  // cupti client call this function
  int retval = posix_memalign((void **) buffer,
    (size_t) 8,
    (size_t) 16 * 1024 * 1024);

  *buffer_size = 16 * 1024 * 1024;

  *maxNumRecords = 0; 
}


void
cupti_buffer_completion_callback
(
 CUcontext ctx,
 uint32_t streamId,
 uint8_t *buffer,
 size_t size,
 size_t validSize
)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;
  do {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if(status == CUPTI_SUCCESS) {
      if (doprint) print_activity(record);
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      CUPTI_CALL(status);
    }
  } while (1);

  size_t dropped;
  CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
  if (dropped != 0) {
    droppedRecords += dropped;  
  }
  if (doprint) {
    printf("CUPTI dropped activities %zu\n", dropped);
  }
  free(buffer);
}


void
cupti_init()
{
  debugger_wait();

  int enable_driver_api = cupti_get_env("CUPTI_DRIVER_API", CUPTI_DRIVER_API_DEFAULT);
  int enable_runtime_api = cupti_get_env("CUPTI_RUNTIME_API", CUPTI_RUNTIME_API_DEFAULT);
  int enable_resource = cupti_get_env("CUPTI_RESOURCE", CUPTI_RESOURCE_DEFAULT);

  CUPTI_CALL(cuptiActivityRegisterCallbacks(cupti_buffer_alloc, cupti_buffer_completion_callback));

  CUPTI_CALL(cuptiSubscribe(&cupti_subscriber,
    (CUpti_CallbackFunc) cupti_subscriber_callback,
    (void *) NULL));

  CUPTI_CALL(cuptiEnableDomain 
    (enable_driver_api, cupti_subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

  CUPTI_CALL(cuptiEnableDomain 
    (enable_runtime_api, cupti_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));

  CUPTI_CALL(cuptiEnableDomain 
    (enable_resource, cupti_subscriber, CUPTI_CB_DOMAIN_RESOURCE));

  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION));
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
  CUPTI_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
}
