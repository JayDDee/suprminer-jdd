/*
* sha-512 cuda kernel implementation.
*
* ==========================(LICENSE BEGIN)============================
*
* Copyright (c) 2014 djm34
*               2016 tpruvot
*				2017-2018 sp
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
*/
#include "cuda_helper_alexis.h"
#include "miner.h"
#include "cuda_vectors_alexis.h"

#define tpb 512

#define SWAP64(u64) cuda_swab64(u64)

static __constant__ uint64_t c_WB[80] = {
	0x428A2F98D728AE22, 0x7137449123EF65CD, 0xB5C0FBCFEC4D3B2F, 0xE9B5DBA58189DBBC,
	0x3956C25BF348B538, 0x59F111F1B605D019, 0x923F82A4AF194F9B, 0xAB1C5ED5DA6D8118,
	0xD807AA98A3030242, 0x12835B0145706FBE, 0x243185BE4EE4B28C, 0x550C7DC3D5FFB4E2,
	0x72BE5D74F27B896F, 0x80DEB1FE3B1696B1, 0x9BDC06A725C71235, 0xC19BF174CF692694,
	0xE49B69C19EF14AD2, 0xEFBE4786384F25E3, 0x0FC19DC68B8CD5B5, 0x240CA1CC77AC9C65,
	0x2DE92C6F592B0275, 0x4A7484AA6EA6E483, 0x5CB0A9DCBD41FBD4, 0x76F988DA831153B5,
	0x983E5152EE66DFAB, 0xA831C66D2DB43210, 0xB00327C898FB213F, 0xBF597FC7BEEF0EE4,
	0xC6E00BF33DA88FC2, 0xD5A79147930AA725, 0x06CA6351E003826F, 0x142929670A0E6E70,
	0x27B70A8546D22FFC, 0x2E1B21385C26C926, 0x4D2C6DFC5AC42AED, 0x53380D139D95B3DF,
	0x650A73548BAF63DE, 0x766A0ABB3C77B2A8, 0x81C2C92E47EDAEE6, 0x92722C851482353B,
	0xA2BFE8A14CF10364, 0xA81A664BBC423001, 0xC24B8B70D0F89791, 0xC76C51A30654BE30,
	0xD192E819D6EF5218, 0xD69906245565A910, 0xF40E35855771202A, 0x106AA07032BBD1B8,
	0x19A4C116B8D2D0C8, 0x1E376C085141AB53, 0x2748774CDF8EEB99, 0x34B0BCB5E19B48A8,
	0x391C0CB3C5C95A63, 0x4ED8AA4AE3418ACB, 0x5B9CCA4F7763E373, 0x682E6FF3D6B2B8A3,
	0x748F82EE5DEFB2FC, 0x78A5636F43172F60, 0x84C87814A1F0AB72, 0x8CC702081A6439EC,
	0x90BEFFFA23631E28, 0xA4506CEBDE82BDE9, 0xBEF9A3F7B2C67915, 0xC67178F2E372532B,
	0xCA273ECEEA26619C, 0xD186B8C721C0C207, 0xEADA7DD6CDE0EB1E, 0xF57D4F7FEE6ED178,
	0x06F067AA72176FBA, 0x0A637DC5A2C898A6, 0x113F9804BEF90DAE, 0x1B710B35131C471B,
	0x28DB77F523047D84, 0x32CAAB7B40C72493, 0x3C9EBE0A15C9BEBC, 0x431D67C49C100D4C,
	0x4CC5D4BECB3E42B6, 0x597F299CFC657E2A, 0x5FCB6FAB3AD6FAEC, 0x6C44198C4A475817
};

#define ROTR64_c(x, n)  (((x) >> (n)) | ((x) << (64 - (n))))

#define BSG5_0(x) xor3(ROTR64(x,28), ROTR64(x,34), ROTR64(x,39))
#define SSG5_0(x) xor3(ROTR64(x, 1), ROTR64(x ,8), shr_u64(x,7))
#define SSG5_1(x) xor3(ROTR64(x,19), ROTR64(x,61), shr_u64(x,6))


#define BSG5_0_c(x) ((ROTR64_c(x,28)^ ROTR64_c(x,34)^ ROTR64_c(x,39))
#define SSG5_0_c(x) ((ROTR64_c(x, 1)^ ROTR64_c(x ,8)^ x>>7))
#define SSG5_1_c(x) ((ROTR64_c(x,19)^ ROTR64_c(x,61)^ x>>6))


#define MAJ(X, Y, Z)   (((X) & (Y)) | (((X) | (Y)) & (Z)))
//#define MAJ(x, y, z)   andor(x,y,z)

__device__ __forceinline__
uint64_t Tone(const uint64_t* K, uint64_t* r, uint64_t* W, const uint8_t a, const uint8_t i)
{
	//asm("// TONE \n");
	const uint64_t e = r[(a + 4) & 7];
	const uint64_t BSG51 = xor3(ROTR64(e, 14), ROTR64(e, 18), ROTR64(e, 41));
	const uint64_t f = r[(a + 5) & 7];
	const uint64_t g = r[(a + 6) & 7];
	const uint64_t CHl = ((f ^ g) & e) ^ g; // xandx(e, f, g);
	return (r[(a + 7) & 7] + W[i] + BSG51 + CHl + K[i]);
}

#define SHA3_STEP(K, r, W, ord, i) { \
	const int a = (8 - ord) & 7; \
	uint64_t T1 = Tone(K, r, W, a, i); \
	r[(a+3) & 7]+= T1; \
	r[(a+7) & 7] = T1 + (BSG5_0(r[a]) + MAJ(r[a], r[(a+1) & 7], r[(a+2) & 7])); \
}

#define bsg5_0(x) (ROTR64(x,28) ^ ROTR64(x,34) ^ ROTR64(x,39))
#define bsg5_1(x) (ROTR64(x,14) ^ ROTR64(x,18) ^ ROTR64(x,41))
#define ssg5_0(x) (ROTR64(x, 1) ^ ROTR64(x, 8) ^ shr_u64(x,7))
#define ssg5_1(x) (ROTR64(x,19) ^ ROTR64(x,61) ^ shr_u64(x,6))


#define bsg5_0_c(x) (ROTR64_c(x,28) ^ ROTR64_c(x,34) ^ ROTR64_c(x,39))
#define bsg5_1_c(x) (ROTR64_c(x,14) ^ ROTR64_c(x,18) ^ ROTR64_c(x,41))

#define andor64(a,b,c) ((a & (b | c)) | (b & c))
#define xandx64(e,f,g) (g ^ (e & (g ^ f)))


//SHA512 MACROS ---------------------------------------------------------------------------
static __constant__ const uint64_t K_512[80] = {
	0x428A2F98D728AE22, 0x7137449123EF65CD, 0xB5C0FBCFEC4D3B2F, 0xE9B5DBA58189DBBC, 0x3956C25BF348B538, 0x59F111F1B605D019, 0x923F82A4AF194F9B, 0xAB1C5ED5DA6D8118,
	0xD807AA98A3030242, 0x12835B0145706FBE, 0x243185BE4EE4B28C, 0x550C7DC3D5FFB4E2, 0x72BE5D74F27B896F, 0x80DEB1FE3B1696B1, 0x9BDC06A725C71235, 0xC19BF174CF692694,
	0xE49B69C19EF14AD2, 0xEFBE4786384F25E3, 0x0FC19DC68B8CD5B5, 0x240CA1CC77AC9C65, 0x2DE92C6F592B0275, 0x4A7484AA6EA6E483, 0x5CB0A9DCBD41FBD4, 0x76F988DA831153B5,
	0x983E5152EE66DFAB, 0xA831C66D2DB43210, 0xB00327C898FB213F, 0xBF597FC7BEEF0EE4, 0xC6E00BF33DA88FC2, 0xD5A79147930AA725, 0x06CA6351E003826F, 0x142929670A0E6E70,
	0x27B70A8546D22FFC, 0x2E1B21385C26C926, 0x4D2C6DFC5AC42AED, 0x53380D139D95B3DF, 0x650A73548BAF63DE, 0x766A0ABB3C77B2A8, 0x81C2C92E47EDAEE6, 0x92722C851482353B,
	0xA2BFE8A14CF10364, 0xA81A664BBC423001, 0xC24B8B70D0F89791, 0xC76C51A30654BE30, 0xD192E819D6EF5218, 0xD69906245565A910, 0xF40E35855771202A, 0x106AA07032BBD1B8,
	0x19A4C116B8D2D0C8, 0x1E376C085141AB53, 0x2748774CDF8EEB99, 0x34B0BCB5E19B48A8, 0x391C0CB3C5C95A63, 0x4ED8AA4AE3418ACB, 0x5B9CCA4F7763E373, 0x682E6FF3D6B2B8A3,
	0x748F82EE5DEFB2FC, 0x78A5636F43172F60, 0x84C87814A1F0AB72, 0x8CC702081A6439EC, 0x90BEFFFA23631E28, 0xA4506CEBDE82BDE9, 0xBEF9A3F7B2C67915, 0xC67178F2E372532B,
	0xCA273ECEEA26619C, 0xD186B8C721C0C207, 0xEADA7DD6CDE0EB1E, 0xF57D4F7FEE6ED178, 0x06F067AA72176FBA, 0x0A637DC5A2C898A6, 0x113F9804BEF90DAE, 0x1B710B35131C471B,
	0x28DB77F523047D84, 0x32CAAB7B40C72493, 0x3C9EBE0A15C9BEBC, 0x431D67C49C100D4C, 0x4CC5D4BECB3E42B6, 0x597F299CFC657E2A, 0x5FCB6FAB3AD6FAEC, 0x6C44198C4A475817
};

__global__ __launch_bounds__(tpb, 2)
void x17_sha512_gpu_hash_64(const uint32_t threads, uint64_t *g_hash){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint64_t IV512[8] = {
		0x6A09E667F3BCC908, 0xBB67AE8584CAA73B, 0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
		0x510E527FADE682D1, 0x9B05688C2B3E6C1F, 0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
	};
	uint64_t r[8];
	uint64_t W[80];
	if (thread < threads){

		uint64_t *pHash = &g_hash[thread << 3];

		*(uint2x4*)&W[0] = *(uint2x4*)&pHash[0];
		*(uint2x4*)&W[4] = *(uint2x4*)&pHash[4];

#pragma unroll
		for (int i = 0; i < 8; i++) {
			W[i] = cuda_swab64(W[i]);
		}
		W[8] = 0x8000000000000000;

#pragma unroll
		for (int i = 9; i<15; i++) {
			W[i] = 0U;
		}
		W[15] = 0x0000000000000200;

#pragma unroll 64
		for (int i = 16; i < 80; i++) {
			W[i] = W[i - 7] + W[i - 16] + SSG5_0(W[i - 15]) + SSG5_1(W[i - 2]);
		}

		*(uint2x4*)&r[0] = *(uint2x4*)&IV512[0];
		*(uint2x4*)&r[4] = *(uint2x4*)&IV512[4];

		uint64_t t1;
#pragma unroll 16
		for (int i = 0; i < 16; i++)
		{
			t1 = W[i] + r[7] + bsg5_1_c(r[4]) + xandx64(r[4], r[5], r[6]) + K_512[i];
#pragma unroll
			for (int l = 6; l >= 0; l--) r[l + 1] = r[l];
			r[0] = t1 + andor64(r[1], r[2], r[3]) + bsg5_0_c(r[1]);
			r[4] += t1;
		}

#pragma unroll
		for (int i = 16; i < 80; i += 16)
		{
#pragma unroll
			for (uint32_t j = 0; j < 16; j++){
				W[j] = ssg5_1(W[(j + 14) & 15]) + ssg5_0(W[(j + 1) & 15]) + W[j] + W[(j + 9) & 15];
			}

#pragma unroll
			for (uint32_t j = 0; j < 16; j++){
				t1 = r[7] + W[j] + bsg5_1(r[4]) + xandx64(r[4], r[5], r[6]) + K_512[i + j];
				//				t1 = r[ 7] + W[j] + K_512[i+j] + xandx64(r[ 4], r[ 5], r[ 6]) + bsg5_1(r[ 4]);
#pragma unroll
				for (int l = 6; l >= 0; l--) r[l + 1] = r[l];
				r[0] = t1 + andor64(r[1], r[2], r[3]) + bsg5_0(r[1]);
				r[4] += t1;
			}
		}

#pragma unroll
		for (int u = 0; u < 8; u++) {
			r[u] = cuda_swab64(r[u] + IV512[u]);
		}

		*(uint2x4*)&pHash[0] = *(uint2x4*)&r[0];
		*(uint2x4*)&pHash[4] = *(uint2x4*)&r[4];

	}
}

__global__ __launch_bounds__(tpb, 2)
void x17_sha512_gpu_hash_64_final(const uint32_t threads, uint64_t *g_hash, uint32_t *resNonce, const uint64_t target)
{

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint64_t IV512[8] = {
		0x6A09E667F3BCC908, 0xBB67AE8584CAA73B, 0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
		0x510E527FADE682D1, 0x9B05688C2B3E6C1F, 0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
	};
	uint64_t r[8];
	uint64_t W[80];
	if (thread < threads){

		uint64_t *pHash = &g_hash[thread << 3];

		*(uint2x4*)&W[0] = *(uint2x4*)&pHash[0];
		*(uint2x4*)&W[4] = *(uint2x4*)&pHash[4];

#pragma unroll
		for (int i = 0; i < 8; i++) {
			W[i] = cuda_swab64(W[i]);
		}
		W[8] = 0x8000000000000000;

#pragma unroll
		for (int i = 9; i<15; i++) {
			W[i] = 0U;
		}
		W[15] = 0x0000000000000200;

#pragma unroll 64
		for (int i = 16; i < 80; i++) {
			W[i] = W[i - 7] + W[i - 16] + SSG5_0(W[i - 15]) + SSG5_1(W[i - 2]);
		}

		*(uint2x4*)&r[0] = *(uint2x4*)&IV512[0];
		*(uint2x4*)&r[4] = *(uint2x4*)&IV512[4];

		uint64_t t1;
#pragma unroll 16
		for (int i = 0; i < 16; i++)
		{
			t1 = W[i] + r[7] + bsg5_1_c(r[4]) + xandx64(r[4], r[5], r[6]) + K_512[i];
#pragma unroll
			for (int l = 6; l >= 0; l--) r[l + 1] = r[l];
			r[0] = t1 + andor64(r[1], r[2], r[3]) + bsg5_0_c(r[1]);
			r[4] += t1;
		}

#pragma unroll
		for (int i = 16; i < 80; i += 16)
		{
#pragma unroll
			for (uint32_t j = 0; j < 16; j++){
				W[j] = ssg5_1(W[(j + 14) & 15]) + ssg5_0(W[(j + 1) & 15]) + W[j] + W[(j + 9) & 15];
			}

#pragma unroll
			for (uint32_t j = 0; j < 16; j++){
				t1 = r[7] + W[j] + bsg5_1(r[4]) + xandx64(r[4], r[5], r[6]) + K_512[i + j];
				//				t1 = r[ 7] + W[j] + K_512[i+j] + xandx64(r[ 4], r[ 5], r[ 6]) + bsg5_1(r[ 4]);
#pragma unroll
				for (int l = 6; l >= 0; l--) r[l + 1] = r[l];
				r[0] = t1 + andor64(r[1], r[2], r[3]) + bsg5_0(r[1]);
				r[4] += t1;
			}
		}


		//		#pragma unroll
		//		for (int u = 0; u < 8; u++) 
		//		{
		//			r[u] = cuda_swab64(r[u] + IV512[u]);
		//		}

		r[3] = cuda_swab64(r[3] + IV512[3]);

		if (r[3] <= target)
		{
			uint32_t tmp = atomicExch(&resNonce[0], thread);
			if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}

		//		*(uint2x4*)&pHash[0] = *(uint2x4*)&r[0];
		//		*(uint2x4*)&pHash[4] = *(uint2x4*)&r[4];

	}
}


__host__
void x17_sha512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	const uint32_t threadsperblock = tpb;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	x17_sha512_gpu_hash_64 << <grid, block >> > (threads, (uint64_t*)d_hash);
}

__host__
void x17_sha512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *resNonce, const uint64_t target)
{
	const uint32_t threadsperblock = tpb;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	x17_sha512_gpu_hash_64_final << <grid, block >> > (threads, (uint64_t*)d_hash, resNonce, target);
}


//__constant__
//static uint64_t c_PaddedMessage80[10];
__constant__ static uint64_t c_W[64];
__constant__ static uint64_t c_r[8];
__constant__ static uint64_t c_PaddedMessage80[10];
__global__ __launch_bounds__(tpb, 2)
void x16_sha512_gpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint64_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint64_t W[80];
#pragma unroll
		for (int i = 0; i < 9; i++) {
			W[i] = SWAP64(c_PaddedMessage80[i]);
		}
		const uint32_t nonce = startNonce + thread;
		//((uint32_t*)W)[19] = cuda_swab32(nonce);
		W[9] = REPLACE_HIDWORD(c_PaddedMessage80[9], cuda_swab32(nonce));
		W[9] = cuda_swab64(W[9]);
		W[10] = 0x8000000000000000;

#pragma unroll
		for (int i = 11; i<15; i++) {
			W[i] = 0U;
		}
		W[15] = 0x0000000000000280;
		uint64_t r[8];

		/*		#pragma unroll
		for (int i = 16; i < 80; i++)
		{
		W[i] = (c_W[i]);
		}

		#pragma unroll
		for (int i = 0; i < 8; i++)
		{
		r[i] = (c_r[i]);
		}
		*/

		const uint64_t IV512[8] = {
			0x6A09E667F3BCC908, 0xBB67AE8584CAA73B,
			0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
			0x510E527FADE682D1, 0x9B05688C2B3E6C1F,
			0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
		};

		//		#pragma unroll 64
		//		for (int i = 16; i < 80; i++) {
		//			W[i] = SSG5_1(W[i - 2]) + W[i - 7];
		//			W[i] += SSG5_0(W[i - 15]) + W[i - 16];
		//		}

#pragma unroll 64
		for (int i = 16; i < 80; i++)
		{
			W[i] = c_W[i - 16];
		}

#pragma unroll
		for (int i = 0; i < 8; i++)
		{
			r[i] = c_r[i];//IV512[i];
		}

		uint64_t t1;

#pragma unroll
		for (int i = 9; i < 16; i++)
		{
			t1 = W[i] + r[7] + bsg5_1_c(r[4]) + xandx64(r[4], r[5], r[6]) + K_512[i];
#pragma unroll
			for (int l = 6; l >= 0; l--) r[l + 1] = r[l];
			r[0] = t1 + andor64(r[1], r[2], r[3]) + bsg5_0_c(r[1]);
			r[4] += t1;
		}

#pragma unroll
		for (int i = 16; i < 80; i += 16)
		{
#pragma unroll
			for (uint32_t j = 0; j < 16; j++){
				W[j] = ssg5_1(W[(j + 14) & 15]) + ssg5_0(W[(j + 1) & 15]) + W[j] + W[(j + 9) & 15];
			}

#pragma unroll
			for (uint32_t j = 0; j < 16; j++){
				t1 = r[7] + W[j] + bsg5_1(r[4]) + xandx64(r[4], r[5], r[6]) + K_512[i + j];
				//				t1 = r[ 7] + W[j] + K_512[i+j] + xandx64(r[ 4], r[ 5], r[ 6]) + bsg5_1(r[ 4]);
#pragma unroll
				for (int l = 6; l >= 0; l--) r[l + 1] = r[l];
				r[0] = t1 + andor64(r[1], r[2], r[3]) + bsg5_0(r[1]);
				r[4] += t1;
			}
		}

		/*
		#pragma unroll
		for (int i = 0; i < 80; i++) {
		SHA3_STEP(c_WB, r, W, i & 7, i);
		}
		*/
		const uint64_t hashPosition = thread;
		uint64_t *pHash = &g_hash[hashPosition << 3];
#pragma unroll
		for (int u = 0; u < 8; u++) {
			pHash[u] = SWAP64(r[u] + IV512[u]);
		}
	}
}

__host__
void x16_sha512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = tpb;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	x16_sha512_gpu_hash_80 << <grid, block >> > (threads, startNounce, (uint64_t*)d_hash);
}

__host__
void x16_sha512_setBlock_80(void *pdata)
{

	uint64_t W[80];

	for (int i = 0; i < 9; i++)
	{
		uint64_t temp = ((uint64_t *)pdata)[i];
		W[i] = SWAP64(temp);
	}
	W[9] = cuda_swab64(W[9]);
	W[10] = 0x8000000000000000;

#pragma unroll
	for (int i = 11; i<15; i++) {
		W[i] = 0U;
	}
	W[15] = 0x0000000000000280;

#pragma unroll 64
	for (int i = 16; i < 80; i++)
	{
		W[i] = SSG5_1_c(W[i - 2]) + W[i - 7];
		W[i] += SSG5_0_c(W[i - 15]) + W[i - 16];
	}

	const uint64_t IV512[8] = {
		0x6A09E667F3BCC908, 0xBB67AE8584CAA73B,
		0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
		0x510E527FADE682D1, 0x9B05688C2B3E6C1F,
		0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
	};

	uint64_t r[8];
#pragma unroll
	for (int i = 0; i < 8; i++) {
		r[i] = IV512[i];
	}

	uint64_t t1;

	for (int i = 0; i < 9; i++)
	{
		t1 = W[i] + r[7] + bsg5_1_c(r[4]) + xandx64(r[4], r[5], r[6]) + K_512[i];
#pragma unroll
		for (int l = 6; l >= 0; l--) r[l + 1] = r[l];
		r[0] = t1 + andor64(r[1], r[2], r[3]) + bsg5_0_c(r[1]);
		r[4] += t1;
	}
	cudaMemcpyToSymbol(c_r, r, sizeof(c_r), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_W, &W[16], sizeof(c_W), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_PaddedMessage80, pdata, sizeof(c_PaddedMessage80), 0, cudaMemcpyHostToDevice);
}