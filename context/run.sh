export NUM_STREAMS_PER_CONTEXT=60
export NUM_CONTEXTS=3
export OMP_NUM_THREADS=180
export HPCTOOLKIT_GPU_TEST_REP=10000

SUBSCRIBE_DIR=../cupti-preload/subscriber

cd $SUBSCRIBE_DIR
make
cd -
make
$SUBSCRIBE_DIR/subscribe ./main
