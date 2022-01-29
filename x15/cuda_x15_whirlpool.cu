/**
* Whirlpool-512 CUDA implementation.
*
* ==========================(LICENSE BEGIN)============================
*
* Copyright (c) 2014-2016 djm34, tpruvot, SP, Provos Alexis
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
* ===========================(LICENSE END)=============================
* @author djm34 (initial draft)
* @author tpruvot (dual old/whirlpool modes, midstate)
* @author Provos Alexis (Applied partial shared memory utilization, precomputations, merging & tuning for 970/750ti under CUDA7.5 -> +93% increased throughput of whirlpool)
* @author SP (+70% faster on the rtx 2060 SUPER, smarter shared memory utilization)
*/

extern "C"
{
#include "sph/sph_whirlpool.h"
#include "miner.h"
}

#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"
#include "cuda_whirlpool_tables.cuh"

__constant__  __align__(64) static uint2 b0[256];
__constant__  __align__(64) static uint2 precomputed_round_key_64[72];
__constant__  __align__(64) static uint2 precomputed_round_key_80[80];
__constant__  __align__(64) uint2 InitVector_RC[10];

//--------START OF WHIRLPOOL DEVICE MACROS---------------------------------------------------------------------------
__device__ __forceinline__
void TRANSFER(uint2 *const __restrict__ dst, const uint2 *const __restrict__ src){
	dst[0] = src[0];
	dst[1] = src[1];
	dst[2] = src[2];
	dst[3] = src[3];
	dst[4] = src[4];
	dst[5] = src[5];
	dst[6] = src[6];
	dst[7] = src[7];
}

__device__ __forceinline__ uint2 d_ROUND_ELT(const uint32_t index,const uint2 sharedMemory[256][16], const uint2 *const __restrict__ in, const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7){

	uint2 ret = sharedMemory[__byte_perm(in[i0].x, 0, 0x4440)][threadIdx.x & index];
	ret ^= ROL8(sharedMemory[__byte_perm(in[i1].x, 0, 0x4441)][threadIdx.x & index]);
	ret ^= ROL16(sharedMemory[__byte_perm(in[i2].x, 0, 0x4442)][threadIdx.x &index]);
	ret ^= ROL24(sharedMemory[__byte_perm(in[i3].x, 0, 0x4443)][threadIdx.x &index]);
	ret ^= SWAPUINT2(sharedMemory[__byte_perm(in[i4].y, 0, 0x4440)][threadIdx.x &index]);
	ret ^= ROR24(sharedMemory[__byte_perm(in[i5].y, 0, 0x4441)][threadIdx.x &index]);
	ret ^= ROR16(sharedMemory[__byte_perm(in[i6].y, 0, 0x4442)][threadIdx.x &index]);
	ret ^= ROR8(sharedMemory[__byte_perm(in[i7].y, 0, 0x4443)][threadIdx.x &index]);
	return ret;
}

__device__ __forceinline__
uint2 d_ROUND_ELT1(const uint32_t index,const uint2 sharedMemory[256][16], const uint2 *const __restrict__ in, const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7, const uint2 c0){
	uint2 ret = sharedMemory[__byte_perm(in[i0].x, 0, 0x4440)][threadIdx.x & index];
	ret ^= ROL8(sharedMemory[__byte_perm(in[i1].x, 0, 0x4441)][threadIdx.x & index]);
	ret ^= ROL16(sharedMemory[__byte_perm(in[i2].x, 0, 0x4442)][threadIdx.x & index]);
	ret ^= ROL24(sharedMemory[__byte_perm(in[i3].x, 0, 0x4443)][threadIdx.x & index]);
	ret ^= SWAPUINT2(sharedMemory[__byte_perm(in[i4].y, 0, 0x4440)][threadIdx.x & index]);
	ret ^= ROR24(sharedMemory[__byte_perm(in[i5].y, 0, 0x4441)][threadIdx.x & index]);
	ret ^= ROR16(sharedMemory[__byte_perm(in[i6].y, 0, 0x4442)][threadIdx.x & index]);
	ret ^= ROR8(sharedMemory[__byte_perm(in[i7].y, 0, 0x4443)][threadIdx.x & index]);
	ret ^= c0;
	return ret;
}

//--------END OF WHIRLPOOL HOST MACROS-------------------------------------------------------------------------------

__host__
extern void x15_whirlpool_cpu_init(int thr_id, uint32_t threads, int mode){

	uint64_t* table0 = NULL;

	table0 = (uint64_t*)plain_T0;
	cudaMemcpyToSymbol(InitVector_RC, plain_RC, 10 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(precomputed_round_key_64, plain_precomputed_round_key_64, 72 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(b0, table0, 256 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}

void whirl_midstate(void *state, const void *input)
{
	sph_whirlpool_context ctx;
	sph_whirlpool1_init(&ctx);
	sph_whirlpool1(&ctx, input, 64);
	memcpy(state, ctx.state, 64);
}

__host__
extern void x15_whirlpool_cpu_free(int thr_id){
	cudaFree(InitVector_RC);
	cudaFree(b0);
//	cudaFree(b7);
}

__global__ __launch_bounds__(320, 2)
void x15_whirlpool_gpu_hash_64(uint32_t threads, uint64_t *g_hash)
{
	__shared__ uint2 sharedMemory[256][16];

	if (threadIdx.x < 256)
	{
		const uint2 tmp = b0[threadIdx.x];
		sharedMemory[threadIdx.x][0] = tmp ;
		sharedMemory[threadIdx.x][1] = tmp;
		sharedMemory[threadIdx.x][2] = tmp;
		sharedMemory[threadIdx.x][3] = tmp;
		sharedMemory[threadIdx.x][4] = tmp;
		sharedMemory[threadIdx.x][5] = tmp;
		sharedMemory[threadIdx.x][6] = tmp;
		sharedMemory[threadIdx.x][7] = tmp;
		sharedMemory[threadIdx.x][8] = tmp;
		sharedMemory[threadIdx.x][9] = tmp;
		sharedMemory[threadIdx.x][10] = tmp;
		sharedMemory[threadIdx.x][11] = tmp;
		sharedMemory[threadIdx.x][12] = tmp;
		sharedMemory[threadIdx.x][13] = tmp;
		sharedMemory[threadIdx.x][14] = tmp;
		sharedMemory[threadIdx.x][15] = tmp;

	}

	const uint32_t index = 15; //sharedindex;

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint2 hash[8], n[8], h[8];
		uint2 tmp[8] = {
			{ 0xC0EE0B30, 0x672990AF }, { 0x28282828, 0x28282828 }, { 0x28282828, 0x28282828 }, { 0x28282828, 0x28282828 },
			{ 0x28282828, 0x28282828 }, { 0x28282828, 0x28282828 }, { 0x28282828, 0x28282828 }, { 0x28282828, 0x28282828 }
		};

		*(uint2x4*)&hash[0] = __ldg4((uint2x4*)&g_hash[(thread << 3) + 0]);
		*(uint2x4*)&hash[4] = __ldg4((uint2x4*)&g_hash[(thread << 3) + 4]);

		__syncthreads();

#pragma unroll 8
		for (int i = 0; i<8; i++)
			n[i] = hash[i];

		tmp[0] ^= d_ROUND_ELT(index,sharedMemory, n, 0, 7, 6, 5, 4, 3, 2, 1);
		tmp[1] ^= d_ROUND_ELT(index, sharedMemory, n, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp[2] ^= d_ROUND_ELT(index, sharedMemory, n, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp[3] ^= d_ROUND_ELT(index, sharedMemory, n, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp[4] ^= d_ROUND_ELT(index, sharedMemory, n, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp[5] ^= d_ROUND_ELT(index, sharedMemory, n, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp[6] ^= d_ROUND_ELT(index, sharedMemory, n, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp[7] ^= d_ROUND_ELT(index, sharedMemory, n, 7, 6, 5, 4, 3, 2, 1, 0);

		//#pragma unroll
		for (int i = 1; i <10; i++)
		{
			TRANSFER(n, tmp);
			tmp[0] = d_ROUND_ELT1(index, sharedMemory, n, 0, 7, 6, 5, 4, 3, 2, 1, precomputed_round_key_64[(i - 1) * 8 + 0]);
			tmp[1] = d_ROUND_ELT1(index, sharedMemory, n, 1, 0, 7, 6, 5, 4, 3, 2, precomputed_round_key_64[(i - 1) * 8 + 1]);
			tmp[2] = d_ROUND_ELT1(index, sharedMemory, n, 2, 1, 0, 7, 6, 5, 4, 3, precomputed_round_key_64[(i - 1) * 8 + 2]);
			tmp[3] = d_ROUND_ELT1(index, sharedMemory, n, 3, 2, 1, 0, 7, 6, 5, 4, precomputed_round_key_64[(i - 1) * 8 + 3]);
			tmp[4] = d_ROUND_ELT1(index, sharedMemory, n, 4, 3, 2, 1, 0, 7, 6, 5, precomputed_round_key_64[(i - 1) * 8 + 4]);
			tmp[5] = d_ROUND_ELT1(index, sharedMemory, n, 5, 4, 3, 2, 1, 0, 7, 6, precomputed_round_key_64[(i - 1) * 8 + 5]);
			tmp[6] = d_ROUND_ELT1(index, sharedMemory, n, 6, 5, 4, 3, 2, 1, 0, 7, precomputed_round_key_64[(i - 1) * 8 + 6]);
			tmp[7] = d_ROUND_ELT1(index, sharedMemory, n, 7, 6, 5, 4, 3, 2, 1, 0, precomputed_round_key_64[(i - 1) * 8 + 7]);
		}

		TRANSFER(h, tmp);
#pragma unroll 8
		for (int i = 0; i<8; i++)
			hash[i] = h[i] = h[i] ^ hash[i];

#pragma unroll 6
		for (int i = 1; i<7; i++)
			n[i] = vectorize(0);

		n[0] = vectorize(0x80);
		n[7] = vectorize(0x2000000000000);

#pragma unroll 8
		for (int i = 0; i < 8; i++) {
			n[i] = n[i] ^ h[i];
		}

//		#pragma unroll
		for (int i = 0; i < 10; i++)
		{
			tmp[0] = InitVector_RC[i];
			tmp[0] ^= d_ROUND_ELT(index, sharedMemory, h, 0, 7, 6, 5, 4, 3, 2, 1);
			tmp[1] = d_ROUND_ELT(index, sharedMemory, h, 1, 0, 7, 6, 5, 4, 3, 2);
			tmp[2] = d_ROUND_ELT(index, sharedMemory, h, 2, 1, 0, 7, 6, 5, 4, 3);
			tmp[3] = d_ROUND_ELT(index, sharedMemory, h, 3, 2, 1, 0, 7, 6, 5, 4);
			tmp[4] = d_ROUND_ELT(index, sharedMemory, h, 4, 3, 2, 1, 0, 7, 6, 5);
			tmp[5] = d_ROUND_ELT(index, sharedMemory, h, 5, 4, 3, 2, 1, 0, 7, 6);
			tmp[6] = d_ROUND_ELT(index, sharedMemory, h, 6, 5, 4, 3, 2, 1, 0, 7);
			tmp[7] = d_ROUND_ELT(index, sharedMemory, h, 7, 6, 5, 4, 3, 2, 1, 0);
			TRANSFER(h, tmp);
			tmp[0] = d_ROUND_ELT1(index, sharedMemory, n, 0, 7, 6, 5, 4, 3, 2, 1, tmp[0]);
			tmp[1] = d_ROUND_ELT1(index, sharedMemory, n, 1, 0, 7, 6, 5, 4, 3, 2, tmp[1]);
			tmp[2] = d_ROUND_ELT1(index, sharedMemory, n, 2, 1, 0, 7, 6, 5, 4, 3, tmp[2]);
			tmp[3] = d_ROUND_ELT1(index, sharedMemory, n, 3, 2, 1, 0, 7, 6, 5, 4, tmp[3]);
			tmp[4] = d_ROUND_ELT1(index, sharedMemory, n, 4, 3, 2, 1, 0, 7, 6, 5, tmp[4]);
			tmp[5] = d_ROUND_ELT1(index, sharedMemory, n, 5, 4, 3, 2, 1, 0, 7, 6, tmp[5]);
			tmp[6] = d_ROUND_ELT1(index, sharedMemory, n, 6, 5, 4, 3, 2, 1, 0, 7, tmp[6]);
			tmp[7] = d_ROUND_ELT1(index, sharedMemory, n, 7, 6, 5, 4, 3, 2, 1, 0, tmp[7]);
			TRANSFER(n, tmp);
		}

		hash[0] = xor3x(hash[0], n[0], vectorize(0x80));
		hash[1] = hash[1] ^ n[1];
		hash[2] = hash[2] ^ n[2];
		hash[3] = hash[3] ^ n[3];
		hash[4] = hash[4] ^ n[4];
		hash[5] = hash[5] ^ n[5];
		hash[6] = hash[6] ^ n[6];
		hash[7] = xor3x(hash[7], n[7], vectorize(0x2000000000000));

		*(uint2x4*)&g_hash[(thread << 3) + 0] = *(uint2x4*)&hash[0];
		*(uint2x4*)&g_hash[(thread << 3) + 4] = *(uint2x4*)&hash[4];
	}
}

__global__ __launch_bounds__(320, 2)
void x15_whirlpool_gpu_hash_64_final(uint32_t threads, const uint64_t* __restrict__ g_hash, uint32_t* resNonce, const uint64_t target)
{
	__shared__ uint2 sharedMemory[256][16];

	if (threadIdx.x < 256) 
	{
		const uint2 tmp = b0[threadIdx.x];
		sharedMemory[threadIdx.x][0] = tmp;
		sharedMemory[threadIdx.x][1] = tmp;
		sharedMemory[threadIdx.x][2] = tmp;
		sharedMemory[threadIdx.x][3] = tmp;
		sharedMemory[threadIdx.x][4] = tmp;
		sharedMemory[threadIdx.x][5] = tmp;
		sharedMemory[threadIdx.x][6] = tmp;
		sharedMemory[threadIdx.x][7] = tmp;
		sharedMemory[threadIdx.x][8] = tmp;
		sharedMemory[threadIdx.x][9] = tmp;
		sharedMemory[threadIdx.x][10] = tmp;
		sharedMemory[threadIdx.x][11] = tmp;
		sharedMemory[threadIdx.x][12] = tmp;
		sharedMemory[threadIdx.x][13] = tmp;
		sharedMemory[threadIdx.x][14] = tmp;
		sharedMemory[threadIdx.x][15] = tmp;
	}
	uint32_t index = threadIdx.x & 15;

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads){

		uint2 hash[8], n[8], h[8], backup;
		uint2 tmp[8] = {
			{ 0xC0EE0B30, 0x672990AF }, { 0x28282828, 0x28282828 }, { 0x28282828, 0x28282828 }, { 0x28282828, 0x28282828 },
			{ 0x28282828, 0x28282828 }, { 0x28282828, 0x28282828 }, { 0x28282828, 0x28282828 }, { 0x28282828, 0x28282828 }
		};

		*(uint2x4*)&hash[0] = __ldg4((uint2x4*)&g_hash[(thread << 3) + 0]);
		*(uint2x4*)&hash[4] = __ldg4((uint2x4*)&g_hash[(thread << 3) + 4]);

		__syncthreads();

#pragma unroll 8
		for (int i = 0; i<8; i++)
			n[i] = hash[i];

		//		__syncthreads();

		tmp[0] ^= d_ROUND_ELT(index, sharedMemory, n, 0, 7, 6, 5, 4, 3, 2, 1);
		tmp[1] ^= d_ROUND_ELT(index, sharedMemory, n, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp[2] ^= d_ROUND_ELT(index, sharedMemory, n, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp[3] ^= d_ROUND_ELT(index, sharedMemory, n, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp[4] ^= d_ROUND_ELT(index, sharedMemory, n, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp[5] ^= d_ROUND_ELT(index, sharedMemory, n, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp[6] ^= d_ROUND_ELT(index, sharedMemory, n, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp[7] ^= d_ROUND_ELT(index, sharedMemory, n, 7, 6, 5, 4, 3, 2, 1, 0);

		for (int i = 1; i <10; i++){
			TRANSFER(n, tmp);
			tmp[0] = d_ROUND_ELT1(index, sharedMemory, n, 0, 7, 6, 5, 4, 3, 2, 1, precomputed_round_key_64[(i - 1) * 8 + 0]);
			tmp[1] = d_ROUND_ELT1(index, sharedMemory, n, 1, 0, 7, 6, 5, 4, 3, 2, precomputed_round_key_64[(i - 1) * 8 + 1]);
			tmp[2] = d_ROUND_ELT1(index, sharedMemory, n, 2, 1, 0, 7, 6, 5, 4, 3, precomputed_round_key_64[(i - 1) * 8 + 2]);
			tmp[3] = d_ROUND_ELT1(index, sharedMemory, n, 3, 2, 1, 0, 7, 6, 5, 4, precomputed_round_key_64[(i - 1) * 8 + 3]);
			tmp[4] = d_ROUND_ELT1(index, sharedMemory, n, 4, 3, 2, 1, 0, 7, 6, 5, precomputed_round_key_64[(i - 1) * 8 + 4]);
			tmp[5] = d_ROUND_ELT1(index, sharedMemory, n, 5, 4, 3, 2, 1, 0, 7, 6, precomputed_round_key_64[(i - 1) * 8 + 5]);
			tmp[6] = d_ROUND_ELT1(index, sharedMemory, n, 6, 5, 4, 3, 2, 1, 0, 7, precomputed_round_key_64[(i - 1) * 8 + 6]);
			tmp[7] = d_ROUND_ELT1(index, sharedMemory, n, 7, 6, 5, 4, 3, 2, 1, 0, precomputed_round_key_64[(i - 1) * 8 + 7]);
		}

		TRANSFER(h, tmp);
#pragma unroll 8
		for (int i = 0; i<8; i++)
			h[i] = h[i] ^ hash[i];

#pragma unroll 6
		for (int i = 1; i<7; i++)
			n[i] = vectorize(0);

		n[0] = vectorize(0x80);
		n[7] = vectorize(0x2000000000000);

#pragma unroll 8
		for (int i = 0; i < 8; i++) {
			n[i] = n[i] ^ h[i];
		}

		backup = h[3];

		//		#pragma unroll 8
		for (int i = 0; i < 8; i++) {
			tmp[0] = d_ROUND_ELT1(index, sharedMemory, h, 0, 7, 6, 5, 4, 3, 2, 1, InitVector_RC[i]);
			tmp[1] = d_ROUND_ELT(index, sharedMemory, h, 1, 0, 7, 6, 5, 4, 3, 2);
			tmp[2] = d_ROUND_ELT(index, sharedMemory, h, 2, 1, 0, 7, 6, 5, 4, 3);
			tmp[3] = d_ROUND_ELT(index, sharedMemory, h, 3, 2, 1, 0, 7, 6, 5, 4);
			tmp[4] = d_ROUND_ELT(index, sharedMemory, h, 4, 3, 2, 1, 0, 7, 6, 5);
			tmp[5] = d_ROUND_ELT(index, sharedMemory, h, 5, 4, 3, 2, 1, 0, 7, 6);
			tmp[6] = d_ROUND_ELT(index, sharedMemory, h, 6, 5, 4, 3, 2, 1, 0, 7);
			tmp[7] = d_ROUND_ELT(index, sharedMemory, h, 7, 6, 5, 4, 3, 2, 1, 0);
			TRANSFER(h, tmp);
			tmp[0] = d_ROUND_ELT1(index, sharedMemory, n, 0, 7, 6, 5, 4, 3, 2, 1, tmp[0]);
			tmp[1] = d_ROUND_ELT1(index, sharedMemory, n, 1, 0, 7, 6, 5, 4, 3, 2, tmp[1]);
			tmp[2] = d_ROUND_ELT1(index, sharedMemory, n, 2, 1, 0, 7, 6, 5, 4, 3, tmp[2]);
			tmp[3] = d_ROUND_ELT1(index, sharedMemory, n, 3, 2, 1, 0, 7, 6, 5, 4, tmp[3]);
			tmp[4] = d_ROUND_ELT1(index, sharedMemory, n, 4, 3, 2, 1, 0, 7, 6, 5, tmp[4]);
			tmp[5] = d_ROUND_ELT1(index, sharedMemory, n, 5, 4, 3, 2, 1, 0, 7, 6, tmp[5]);
			tmp[6] = d_ROUND_ELT1(index, sharedMemory, n, 6, 5, 4, 3, 2, 1, 0, 7, tmp[6]);
			tmp[7] = d_ROUND_ELT1(index, sharedMemory, n, 7, 6, 5, 4, 3, 2, 1, 0, tmp[7]);
			TRANSFER(n, tmp);
		}
		tmp[0] = d_ROUND_ELT1(index, sharedMemory, h, 0, 7, 6, 5, 4, 3, 2, 1, InitVector_RC[8]);
		tmp[1] = d_ROUND_ELT(index, sharedMemory, h, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp[2] = d_ROUND_ELT(index, sharedMemory, h, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp[3] = d_ROUND_ELT(index, sharedMemory, h, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp[4] = d_ROUND_ELT(index, sharedMemory, h, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp[5] = d_ROUND_ELT(index, sharedMemory, h, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp[6] = d_ROUND_ELT(index, sharedMemory, h, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp[7] = d_ROUND_ELT(index, sharedMemory, h, 7, 6, 5, 4, 3, 2, 1, 0);
		TRANSFER(h, tmp);
		tmp[0] = d_ROUND_ELT1(index, sharedMemory, n, 0, 7, 6, 5, 4, 3, 2, 1, tmp[0]);
		tmp[1] = d_ROUND_ELT1(index, sharedMemory, n, 1, 0, 7, 6, 5, 4, 3, 2, tmp[1]);
		tmp[2] = d_ROUND_ELT1(index, sharedMemory, n, 2, 1, 0, 7, 6, 5, 4, 3, tmp[2]);
		tmp[3] = d_ROUND_ELT1(index, sharedMemory, n, 3, 2, 1, 0, 7, 6, 5, 4, tmp[3]);
		tmp[4] = d_ROUND_ELT1(index, sharedMemory, n, 4, 3, 2, 1, 0, 7, 6, 5, tmp[4]);
		tmp[5] = d_ROUND_ELT1(index, sharedMemory, n, 5, 4, 3, 2, 1, 0, 7, 6, tmp[5]);
		tmp[6] = d_ROUND_ELT1(index, sharedMemory, n, 6, 5, 4, 3, 2, 1, 0, 7, tmp[6]);
		tmp[7] = d_ROUND_ELT1(index, sharedMemory, n, 7, 6, 5, 4, 3, 2, 1, 0, tmp[7]);

		n[3] = backup ^ d_ROUND_ELT(index, sharedMemory, h, 3, 2, 1, 0, 7, 6, 5, 4) ^ d_ROUND_ELT(index, sharedMemory, tmp, 3, 2, 1, 0, 7, 6, 5, 4);

		if (devectorize(n[3]) <= target){
			uint32_t tmp = atomicExch(&resNonce[0], thread);
			if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}
	}
}

extern void x15_whirlpool_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target)
{
	dim3 grid((threads + 320 - 1) / 320);
	dim3 block(320);

	x15_whirlpool_gpu_hash_64_final << <grid, block >> > (threads, (uint64_t*)d_hash, d_resNonce, target);
}

__host__
extern void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	dim3 grid((threads + 320 - 1) / 320);
	dim3 block(320);

	x15_whirlpool_gpu_hash_64 << <grid, block >> > (threads, (uint64_t*)d_hash);
}
__host__
void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	dim3 grid((threads + 256 - 1) / 256);
	dim3 block(256);

	x15_whirlpool_gpu_hash_64 << <grid, block >> > (threads, (uint64_t*)d_hash);
}
