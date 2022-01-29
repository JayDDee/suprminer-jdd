/*
	Based on Tanguy Pruvot's repo
	Provos Alexis - 2016
*/

#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"

#define INTENSIVE_GMF
#include "cuda_x11_aes_sp.cuh"

__device__
static void echo_round_sp(const uint32_t sharedMemory[8 * 1024], uint32_t *W, uint32_t &k0){
	// Big Sub Words
#pragma unroll 16
	for (int idx = 0; idx < 16; idx++)
		AES_2ROUND_32(sharedMemory, W[(idx << 2) + 0], W[(idx << 2) + 1], W[(idx << 2) + 2], W[(idx << 2) + 3], k0);

	// Shift Rows
#pragma unroll 4
	for (int i = 0; i < 4; i++){
		uint32_t t[4];
		/// 1, 5, 9, 13
		t[0] = W[i + 4];
		t[1] = W[i + 8];
		t[2] = W[i + 24];
		t[3] = W[i + 60];
		W[i + 4] = W[i + 20];
		W[i + 8] = W[i + 40];
		W[i + 24] = W[i + 56];
		W[i + 60] = W[i + 44];

		W[i + 20] = W[i + 36];
		W[i + 40] = t[1];
		W[i + 56] = t[2];
		W[i + 44] = W[i + 28];

		W[i + 28] = W[i + 12];
		W[i + 12] = t[3];
		W[i + 36] = W[i + 52];
		W[i + 52] = t[0];
	}
	// Mix Columns
#pragma unroll 4
	for (int i = 0; i < 4; i++){ // Schleife über je 2*uint32_t
#pragma unroll 4
		for (int idx = 0; idx < 64; idx += 16){ // Schleife über die elemnte
			uint32_t a[4];
			a[0] = W[idx + i];
			a[1] = W[idx + i + 4];
			a[2] = W[idx + i + 8];
			a[3] = W[idx + i + 12];

			uint32_t ab = a[0] ^ a[1];
			uint32_t bc = a[1] ^ a[2];
			uint32_t cd = a[2] ^ a[3];

			uint32_t t, t2, t3;
			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			uint32_t abx = (t >> 7) * 27U ^ ((ab^t) << 1);
			uint32_t bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			uint32_t cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[idx + i] = (bc^ a[3] ^ abx);
			W[idx + i + 4] = xor3(a[0], cd, bcx);
			W[idx + i + 8] = xor3(ab, a[3], cdx);
			W[idx + i + 12] = xor3(ab, a[2], xor3(abx, bcx, cdx));
		}
	}
}

__global__ __launch_bounds__(256, 3) /* will force 80 registers */
void x11_echo512_gpu_hash_64_final_sp(uint32_t threads, uint64_t *g_hash, uint32_t* resNonce, const uint64_t target)
{
	__shared__ __align__(16) uint32_t sharedMemory[8 * 1024];

	aes_gpu_init256_32(sharedMemory);


	const uint32_t P[48] = {
		0xe7e9f5f5, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0xa4213d7e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		//8-12
		0x01425eb8, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0x65978b09, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		//21-25
		0x2cb6b661, 0x6b23b3b3, 0xcf93a7cf, 0x9d9d3751, 0x9ac2dea3, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		//34-38
		0x579f9f33, 0xfbfbfbfb, 0xfbfbfbfb, 0xefefd3c7, 0xdbfde1dd, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x34514d9e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0xb134347e, 0xea6f7e7e, 0xbd7731bd, 0x8a8a1968,
		0x14b8a457, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0x265f4382, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af
		//58-61
	};
	uint32_t k0;
	uint32_t h[16];

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){

		const uint32_t *hash = (uint32_t*)&g_hash[thread << 3];

		*(uint2x4*)&h[0] = __ldg4((uint2x4*)&hash[0]);
		*(uint2x4*)&h[8] = __ldg4((uint2x4*)&hash[8]);

		uint64_t backup = *(uint64_t*)&h[6];

		k0 = 512 + 8;

		__threadfence_block();

#pragma unroll 4
		for (uint32_t idx = 0; idx < 16; idx += 4)
			AES_2ROUND_32(sharedMemory, h[idx + 0], h[idx + 1], h[idx + 2], h[idx + 3], k0);

		k0 += 4;

		uint32_t W[64];

#pragma unroll 4
		for (uint32_t i = 0; i < 4; i++){
			uint32_t a = P[i];
			uint32_t b = P[i + 4];
			uint32_t c = h[i + 8];
			uint32_t d = P[i + 8];

			uint32_t ab = a ^ b;
			uint32_t bc = b ^ c;
			uint32_t cd = c ^ d;


			uint32_t t = ((a ^ b) & 0x80808080);
			uint32_t t2 = ((b ^ c) & 0x80808080);
			uint32_t t3 = ((c ^ d) & 0x80808080);

			uint32_t abx = ((t >> 7) * 27U) ^ ((ab^t) << 1);
			uint32_t bcx = ((t2 >> 7) * 27U) ^ ((bc^t2) << 1);
			uint32_t cdx = ((t3 >> 7) * 27U) ^ ((cd^t3) << 1);

			W[0 + i] = bc ^ d ^ abx;
			W[4 + i] = a ^ cd ^ bcx;
			W[8 + i] = ab ^ d ^ cdx;
			W[12 + i] = abx ^ bcx ^ cdx ^ ab ^ c;

			a = P[12 + i];
			b = h[i + 4];
			c = P[12 + i + 4];
			d = P[12 + i + 8];

			ab = a ^ b;
			bc = b ^ c;
			cd = c ^ d;


			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			abx = (t >> 7) * 27U ^ ((ab^t) << 1);
			bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[16 + i] = abx ^ bc ^ d;
			W[16 + i + 4] = bcx ^ a ^ cd;
			W[16 + i + 8] = cdx ^ ab ^ d;
			W[16 + i + 12] = abx ^ bcx ^ cdx ^ ab ^ c;

			a = h[i];
			b = P[24 + i];
			c = P[24 + i + 4];
			d = P[24 + i + 8];

			ab = a ^ b;
			bc = b ^ c;
			cd = c ^ d;


			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			abx = (t >> 7) * 27U ^ ((ab^t) << 1);
			bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[32 + i] = abx ^ bc ^ d;
			W[32 + i + 4] = bcx ^ a ^ cd;
			W[32 + i + 8] = cdx ^ ab ^ d;
			W[32 + i + 12] = abx ^ bcx ^ cdx ^ ab ^ c;

			a = P[36 + i];
			b = P[36 + i + 4];
			c = P[36 + i + 8];
			d = h[i + 12];

			ab = a ^ b;
			bc = b ^ c;
			cd = c ^ d;

			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			abx = (t >> 7) * 27U ^ ((ab^t) << 1);
			bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[48 + i] = abx ^ bc ^ d;
			W[48 + i + 4] = xor3(bcx , a , cd);
			W[48 + i + 8] = xor3(cdx , ab, d);
			W[48 + i + 12] = xor3(abx , bcx , xor3(cdx, ab, c));


		}

		for (int k = 1; k < 10; k++)
			echo_round_sp(sharedMemory, W, k0);

#pragma unroll 4
		for (int i = 0; i < 16; i += 4)
		{
			W[i] ^= W[32 + i] ^ 512;
			W[i + 1] ^= W[32 + i + 1];
			W[i + 2] ^= W[32 + i + 2];
			W[i + 3] ^= W[32 + i + 3];
		}
		uint64_t check = ((uint64_t*)hash)[3] ^ ((uint64_t*)W)[3];

		if (check <= target)
		{
			uint32_t tmp = atomicExch(&resNonce[0], thread);
			if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}
	}
}

__host__
void x11_echo512_cpu_hash_64_final_sp(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x11_echo512_gpu_hash_64_final_sp<<<grid, block>>>(threads, (uint64_t*)d_hash,d_resNonce,target);
}

__global__ __launch_bounds__(384, 2)
static void x11_echo512_gpu_hash_64_sp(uint32_t threads, uint32_t *g_hash)
{
	__shared__ uint32_t sharedMemory[8 * 1024];

	//	if (threadIdx.x < 256)
	//	{
	aes_gpu_init256_32(sharedMemory);
	//	}
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint32_t k0;
	uint32_t h[16];
	uint32_t hash[16];
	if (thread < threads){

		uint32_t *Hash = &g_hash[thread << 4];

		*(uint2x4*)&h[0] = __ldg4((uint2x4*)&Hash[0]);
		*(uint2x4*)&h[8] = __ldg4((uint2x4*)&Hash[8]);

		//		*(uint2x4*)&hash[0] = *(uint2x4*)&h[0];
		//		*(uint2x4*)&hash[8] = *(uint2x4*)&h[8];


		const uint32_t P[48] = {
			0xe7e9f5f5, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0xa4213d7e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
			//8-12
			0x01425eb8, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0x65978b09, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
			//21-25
			0x2cb6b661, 0x6b23b3b3, 0xcf93a7cf, 0x9d9d3751, 0x9ac2dea3, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
			//34-38
			0x579f9f33, 0xfbfbfbfb, 0xfbfbfbfb, 0xefefd3c7, 0xdbfde1dd, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
			0x34514d9e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0xb134347e, 0xea6f7e7e, 0xbd7731bd, 0x8a8a1968,
			0x14b8a457, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0x265f4382, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af
			//58-61
		};

		k0 = 520;
		__threadfence_block();

#pragma unroll 4
		for (uint32_t idx = 0; idx < 16; idx += 4)
		{
			AES_2ROUND_32(sharedMemory, h[idx + 0], h[idx + 1], h[idx + 2], h[idx + 3], k0);
		}
		k0 += 4;

		uint32_t W[64];

#pragma unroll 4
		for (uint32_t i = 0; i < 4; i++)
		{
			uint32_t a = P[i];
			uint32_t b = P[i + 4];
			uint32_t c = h[i + 8];
			uint32_t d = P[i + 8];

			uint32_t ab = a ^ b;
			uint32_t bc = b ^ c;
			uint32_t cd = c ^ d;


			uint32_t t = (ab & 0x80808080);
			uint32_t t2 = (bc & 0x80808080);
			uint32_t t3 = (cd & 0x80808080);

			uint32_t abx = (t >> 7) * 27U ^ ((ab^t) << 1);
			uint32_t bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			uint32_t cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[i] = abx ^ bc ^ d;
			W[i + 4] = bcx ^ a ^ cd;
			W[i + 8] = cdx ^ ab ^ d;
			W[i + 12] = abx ^ bcx ^ cdx ^ ab ^ c;

			a = P[i + 12];
			b = h[i + 4];
			c = P[i + 16];
			d = P[i + 20];

			ab = a ^ b;
			bc = b ^ c;
			cd = c ^ d;


			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			abx = (t >> 7) * 27U ^ ((ab^t) << 1);
			bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[16 + i] = bc ^ d ^ abx;
			W[16 + i + 4] = a ^ cd ^ bcx;
			W[16 + i + 8] = d ^ ab ^ cdx;
			W[16 + i + 12] = c ^ ab ^ abx ^ bcx ^ cdx;

			a = h[i];
			b = P[24 + i + 0];
			c = P[24 + i + 4];
			d = P[24 + i + 8];

			ab = a ^ b;
			bc = b ^ c;
			cd = c ^ d;


			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			abx = (t >> 7) * 27U ^ ((ab^t) << 1);
			bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[32 + i] = bc ^ d ^ abx;
			W[32 + i + 4] = a ^ cd ^ bcx;
			W[32 + i + 8] = d ^ ab ^ cdx;
			W[32 + i + 12] = c ^ ab ^ abx ^ bcx ^ cdx;

			a = P[36 + i];
			b = P[36 + i + 4];
			c = P[36 + i + 8];
			d = h[i + 12];

			ab = a ^ b;
			bc = b ^ c;
			cd = c ^ d;

			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			abx = (t >> 7) * 27U ^ ((ab^t) << 1);
			bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[48 + i] = (bc ^ d ^ abx);
			W[48 + i + 4] = (a ^ cd ^ bcx);
			W[48 + i + 8] = (d^ ab^ cdx);
			W[48 + i + 12] = (c ^ ab ^ (abx ^ bcx ^cdx));

		}

		for (int k = 1; k < 10; k++)
			echo_round_sp(sharedMemory, W, k0);

#pragma unroll 4
		for (int i = 0; i < 16; i += 4)
		{
			W[i] ^= W[32 + i] ^ 512;
			W[i + 1] ^= W[32 + i + 1];
			W[i + 2] ^= W[32 + i + 2];
			W[i + 3] ^= W[32 + i + 3];
		}
		*(uint2x4*)&Hash[0] = *(uint2x4*)&Hash[0] ^ *(uint2x4*)&W[0];
		*(uint2x4*)&Hash[8] = *(uint2x4*)&Hash[8] ^ *(uint2x4*)&W[8];
	}
}

__host__
void x11_echo512_cpu_hash_64_sp(int thr_id, uint32_t threads, uint32_t *d_hash){

	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x11_echo512_gpu_hash_64_sp<<<grid, block>>>(threads, d_hash);
}
