/*
 * Preloaded module that does nothing more than enable pc sampling.
 * All samples are scanned and dropped. This code is to measure
 * CUPTI overhead.
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
#include <map>


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


#define FORALL_PERIODS(macro)				\
  macro(CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN, 1)	\
  macro(CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_LOW, 2)	\
  macro(CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MID, 3)	\
  macro(CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_HIGH, 4)	\
  macro(CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MAX, 5)

#define stringify(x) #x
  
#define BUF_SIZE (4 * 1024 * 1024 - 128)
#define ALIGN_SIZE (8)

#define DEBUGGER_WAIT_FLAG_DEFAULT 0
#define DEBUGGER_PRINT_FLAG_DEFAULT 0
#define DEBUGGER_COUNT_FLAG_DEFAULT 1

//************************************************************************
// variables
//************************************************************************

// control flags
volatile int debugger_wait_flag = DEBUGGER_WAIT_FLAG_DEFAULT; 
volatile int doprint = DEBUGGER_PRINT_FLAG_DEFAULT;
volatile int docount = DEBUGGER_COUNT_FLAG_DEFAULT;
volatile int samplingEnabled = 0;

uint64_t totalSamples = 0;
uint64_t droppedSamples = 0;

uint64_t droppedRecords = 0;
uint64_t totalRecords = 0;

uint64_t bufferCount = 0;
uint64_t totalBufferSize = 0;


std::map<CUpti_ActivityKind, int> activityCount;
std::map<CUpti_ActivityPCSamplingStallReason, std::map<unsigned long long, int> > pcsamplingCount;
CUpti_SubscriberHandle cuptiSubscriber;

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

static const char *
getStallReasonString(CUpti_ActivityPCSamplingStallReason reason)
{
  switch (reason) {
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_INVALID:
    return "Invalid";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_NONE:
    return "Selected";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_INST_FETCH:
    return "Instruction fetch";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_EXEC_DEPENDENCY:
    return "Execution dependency";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_DEPENDENCY:
    return "Memory dependency";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_TEXTURE:
    return "Texture";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_SYNC:
    return "Sync";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_CONSTANT_MEMORY_DEPENDENCY:
    return "Constant memory dependency";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_PIPE_BUSY:
    return "Pipe busy";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_THROTTLE:
    return "Memory throttle";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_NOT_SELECTED:
    return "Not selected";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_OTHER:
    return "Other";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_SLEEPING:
    return "Sleeping";
  default:
    break;
  }

  return "<unknown>";
}


static void
printActivity(CUpti_Activity *record)
{
  switch (record->kind) {
  case CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR:
    {
      CUpti_ActivitySourceLocator *sourceLocator = 
        (CUpti_ActivitySourceLocator *)record;
      printf("Source Locator Id %d, File %s Line %d\n", 
	     sourceLocator->id, sourceLocator->fileName, 
	     sourceLocator->lineNumber);
      break;
    }
  case CUPTI_ACTIVITY_KIND_PC_SAMPLING:
    {
      CUpti_ActivityPCSampling3 *psRecord = (CUpti_ActivityPCSampling3 *)record;

      printf("source %u, functionId %u, pc 0x%llx, corr %u, " 
	     "samples %u, latencySamples %u, stallreason %s\n",
	     psRecord->sourceLocatorId,
	     psRecord->functionId,
	     (unsigned long long)psRecord->pcOffset,
	     psRecord->correlationId,
	     psRecord->samples,
	     psRecord->latencySamples,
	     getStallReasonString(psRecord->stallReason));
      break;
    }
  case CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO:
    {
      CUpti_ActivityPCSamplingRecordInfo *pcsriResult =
        (CUpti_ActivityPCSamplingRecordInfo *)(void *)record;

      printf("corr %u, totalSamples %llu, droppedSamples %llu, cycles %llu\n",
	     pcsriResult->correlationId,
	     (unsigned long long)pcsriResult->totalSamples,
	     (unsigned long long)pcsriResult->droppedSamples,
       (unsigned long long)pcsriResult->samplingPeriodInCycles);
      break;
    }
  case CUPTI_ACTIVITY_KIND_FUNCTION:
    {
      CUpti_ActivityFunction *fResult =
        (CUpti_ActivityFunction *)record;

      printf("id %u, ctx %u, moduleId %u, functionIndex %u, name %s\n",
	     fResult->id,
	     fResult->contextId,
	     fResult->moduleId,
	     fResult->functionIndex,
	     fResult->name);
      break;
    }
  default:
    printf("unknown\n");
    break;
  }
}


static void
countActivity(CUpti_Activity *record)
{
  activityCount[record->kind]++;
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_PC_SAMPLING:
      {
        CUpti_ActivityPCSampling3 *psRecord = (CUpti_ActivityPCSampling3 *)record;

        pcsamplingCount[psRecord->stallReason][psRecord->pcOffset] += psRecord->samples; 
        break;
      }
    default:
      break;
  }
}


static void CUPTIAPI
bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  *size = BUF_SIZE + ALIGN_SIZE;
  *buffer = (uint8_t*) calloc(1, *size);
  *maxNumRecords = 0;
  if (*buffer == NULL) {
    printf("CUPTI: buffer request -- out of memory\n");
    exit(-1);
  } else {
    if (doprint) {
      printf("CUPTI: buffer request -- allocated %" PRIu64 
	     " bytes\n", *size);
    }
  }
}


static void CUPTIAPI
bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, 
		size_t size, size_t validSize)
{
  bufferCount++;
  totalBufferSize += validSize;

  if (doprint) {
    printf("CUPTI: buffer completion -- %" PRIu64 " valid bytes\n", validSize);
  }

  CUptiResult status;
  CUpti_Activity *record = NULL;
  do {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if(status == CUPTI_SUCCESS) {
      if (doprint) printActivity(record);
      if (docount) countActivity(record);
      totalRecords++;
      if (record->kind == CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO) {
        CUpti_ActivityPCSamplingRecordInfo *pcsriResult = 
          (CUpti_ActivityPCSamplingRecordInfo *)(void *)record;
        totalSamples += (uint64_t) pcsriResult->totalSamples;
        droppedSamples += (uint64_t) pcsriResult->droppedSamples;
      }
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
  free(buffer);
}


static const char *
samplingPeriod(CUpti_ActivityPCSamplingPeriod p)
{
#define macro(enumName, val) case enumName: return #enumName; 
  switch (p) {
    FORALL_PERIODS(macro)
  default:
      return stringify(CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_INVALID); 
  }
#undef macro
  return stringify(CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_INVALID); 
}


static uint32_t
getPeriod()
{
  uint32_t period = 5; // default value
  const char *periodVal = getenv("CUPTI_SAMPLING_PERIOD");

  if (periodVal) {
    period = atoi(periodVal);
  } else {
    printf("usage: set environment variable CUPTI_SAMPLING_PERIOD " 
	   "to a value from 5-31\n");
    exit(-1);
  }

  return period;
}


static int
getDevice()
{
  int deviceNum = -1;
  const char *device = getenv("CUPTI_DEVICE_NUM");

  if (device) {
    deviceNum = atoi(device);
  } else {
    printf("usage: set environment variable CUPTI_DEVICE_NUM\n"); 
    exit(-1);
  }

  RUNTIME_API_CALL(cudaSetDevice(deviceNum));
  return deviceNum;
}

//************************************************************************
// initialization 
//************************************************************************

void
cupti_init()
{
  debugger_wait();

  int deviceNum = getDevice();

  cudaDeviceProp prop;
  RUNTIME_API_CALL(cudaGetDeviceProperties(&prop, deviceNum));

  uint32_t cuptiSamplingPeriod = getPeriod();

  if (cuptiSamplingPeriod > 31 || cuptiSamplingPeriod < 5) {
    printf("CUPTI: sampling disabled by setting sampling period to %u\n", cuptiSamplingPeriod);
    return;
  }
  samplingEnabled = 1;

  printf("CUPTI: initializing PC sampling for device %s " 
	 "with compute capability %d.%d. Sampling period is %u.\n", 
	 prop.name, prop.major, prop.minor, cuptiSamplingPeriod);

  CUpti_ActivityPCSamplingConfig configPC;
  configPC.size = sizeof(CUpti_ActivityPCSamplingConfig);
  configPC.samplingPeriod = (CUpti_ActivityPCSamplingPeriod) 0;
  configPC.samplingPeriod2 = cuptiSamplingPeriod;

  CUcontext cuCtx;
  CUdevice dev;
  DRIVER_API_CALL(cuDeviceGet(&dev, deviceNum));
  DRIVER_API_CALL(cuCtxCreate(&cuCtx, 0, dev));
  DRIVER_API_CALL(cuCtxSetCurrent(cuCtx));

  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING));
  CUPTI_CALL(cuptiActivityConfigurePCSampling(cuCtx, &configPC));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
}



//************************************************************************
// finalization 
//************************************************************************

void
cupti_fini()
{
  if (!samplingEnabled) return;

  RUNTIME_API_CALL(cudaDeviceSynchronize());
  CUPTI_CALL(cuptiActivityFlushAll(0));
  printf("CUPTI: total samples = %" PRIu64 ", dropped samples = %" PRIu64 
	 ", total records = %" PRIu64 ", dropped records = %" PRIu64 "\n",
	 totalSamples, droppedSamples, totalRecords, droppedRecords);
  printf("CUPTI: total buffer count = %" PRIu64 
	 ", total buffer valid size = %" PRIu64 "\n", 
	 bufferCount, totalBufferSize);

  if (doprint && docount) {
    printf("CUPTI: PC sampling count\n");
    for (auto iter : pcsamplingCount) {
      for (auto pc_iter : iter.second) {
        printf("%d : <%d, %d>\n", iter.first, pc_iter.first, pc_iter.second);
      }
    }
    printf("CUPTI: Activity count\n");
    for (auto iter : activityCount) {
      printf("%d : %d\n", iter.first, iter.second);
    }
  }
}
