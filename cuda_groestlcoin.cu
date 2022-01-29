// Auf Groestlcoin spezialisierte Version von Groestl inkl. Bitslice

#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

#ifdef __INTELLISENSE__
#define __CUDA_ARCH__ 500
#define __byte_perm(x,y,n) x
#endif

#include "miner.h"

__constant__ uint32_t pTarget[8]; // Single GPU
__constant__ uint32_t groestlcoin_gpu_msg[32];

static uint32_t *d_resultNonce[MAX_GPUS];

#if __CUDA_ARCH__ >= 300
// 64 Registers Variant for Compute 3.0+
#include "quark/groestl_functions_quad.h"
#include "quark/groestl_transf_quad.h"
#endif

#define SWAB32(x) cuda_swab32(x)

__global__ __launch_bounds__(256, 4)
void groestlcoin_gpu_hash_quad(uint32_t threads, uint32_t startNounce, uint32_t *resNounce)
{
}

__host__
void groestlcoin_cpu_init(int thr_id, uint32_t threads)
{
}

__host__
void groestlcoin_cpu_free(int thr_id)
{
}

__host__
void groestlcoin_cpu_setBlock(int thr_id, void *data, void *pTargetIn)
{
}

__host__
void groestlcoin_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *resNonce)
{
}
