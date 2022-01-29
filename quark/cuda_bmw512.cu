/*
Based on SP's BMW kernel
Provos Alexis - 2016
Optimized for pascal sp - may 2018
*/

#include <stdio.h>
#include <memory.h>

#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"

#define CONST_EXP3d(i)   devectorize(ROL2(q[i+ 1], 5))    + devectorize(ROL2(q[i+ 3],11)) + devectorize(ROL2(q[i+5], 27)) + \
                         devectorize(SWAPDWORDS2(q[i+7])) + devectorize(ROL2(q[i+9], 37)) + devectorize(ROL2(q[i+11],43)) + \
                         devectorize(ROL2(q[i+13],53))    + devectorize(SHR2(q[i+14],1) ^ q[i+14]) + devectorize(SHR2(q[i+15],2) ^ q[i+15])

__device__ __forceinline__
static void bmw512_round1(uint2* q, uint2* h, const uint64_t* msg){
	const uint2 hash[16] =
	{
		{ 0x84858687, 0x80818283 }, { 0x8C8D8E8F, 0x88898A8B }, { 0x94959697, 0x90919293 }, { 0x9C9D9E9F, 0x98999A9B },
		{ 0xA4A5A6A7, 0xA0A1A2A3 }, { 0xACADAEAF, 0xA8A9AAAB }, { 0xB4B5B6B7, 0xB0B1B2B3 }, { 0xBCBDBEBF, 0xB8B9BABB },
		{ 0xC4C5C6C7, 0xC0C1C2C3 }, { 0xCCCDCECF, 0xC8C9CACB }, { 0xD4D5D6D7, 0xD0D1D2D3 }, { 0xDCDDDEDF, 0xD8D9DADB },
		{ 0xE4E5E6E7, 0xE0E1E2E3 }, { 0xECEDEEEF, 0xE8E9EAEB }, { 0xF4F5F6F7, 0xF0F1F2F3 }, { 0xFCFDFEFF, 0xF8F9FAFB }
	};

	const uint64_t hash2[16] =
	{
		0x8081828384858687, 0x88898A8B8C8D8E8F, 0x9091929394959697, 0x98999A9B9C9D9E9F,
		0xA0A1A2A3A4A5A6A7, 0xA8A9AAABACADAEAF, 0xB0B1B2B3B4B5B6B7, 0xB8B9BABBBCBDBEBF,
		0xC0C1C2C3C4C5C6C7 ^ 0x80, 0xC8C9CACBCCCDCECF, 0xD0D1D2D3D4D5D6D7, 0xD8D9DADBDCDDDEDF,
		0xE0E1E2E3E4E5E6E7, 0xE8E9EAEBECEDEEEF, 0xF0F1F2F3F4F5F6F7, 0xF8F9FAFBFCFDFEFF
	};

	const uint2 precalcf[9] =
	{
		{ 0x55555550, 0x55555555 }, { 0xAAAAAAA5, 0x5AAAAAAA }, { 0xFFFFFFFA, 0x5FFFFFFF }, { 0x5555554F, 0x65555555 },
		{ 0xAAAAAAA4, 0x6AAAAAAA }, { 0xFE00FFF9, 0x6FFFFFFF }, { 0xAAAAAAA1, 0x9AAAAAAA }, { 0xFFFEFFF6, 0x9FFFFFFF }, { 0x5755554B, 0xA5555555 }
	};

	uint2 tmp;
	uint64_t mxh[8];

	mxh[0] = msg[0] ^ hash2[0];
	mxh[1] = msg[1] ^ hash2[1];
	mxh[2] = msg[2] ^ hash2[2];
	mxh[3] = msg[3] ^ hash2[3];
	mxh[4] = msg[4] ^ hash2[4];
	mxh[5] = msg[5] ^ hash2[5];
	mxh[6] = msg[6] ^ hash2[6];
	mxh[7] = msg[7] ^ hash2[7];

	tmp = vectorize(mxh[5] - mxh[7]) + hash[10] + hash[13] + hash[14];
	q[0] = hash[1] + (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37));

	tmp = vectorize(mxh[6]) + hash[11] + hash[14] - (hash[15] ^ 512) - (hash[8] ^ 0x80);
	q[1] = hash[2] + (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43));

	tmp = vectorize(mxh[0] + mxh[7]) + hash[9] - hash[12] + (hash[15] ^ 0x200);
	q[2] = hash[3] + (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53));

	q[16] = (SHR2(q[0], 1) ^ SHL2(q[0], 2) ^ ROL2(q[0], 13) ^ ROL2(q[0], 43)) + (SHR2(q[1], 2) ^ SHL2(q[1], 1) ^ ROL2(q[1], 19) ^ ROL2(q[1], 53));
	q[17] = (SHR2(q[1], 1) ^ SHL2(q[1], 2) ^ ROL2(q[1], 13) ^ ROL2(q[1], 43)) + (SHR2(q[2], 2) ^ SHL2(q[2], 1) ^ ROL2(q[2], 19) ^ ROL2(q[2], 53));

	tmp = vectorize((mxh[0] - mxh[1]) + hash2[8] - hash2[10] + hash2[13]);
	q[3] = hash[4] + (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59));

	tmp = vectorize((mxh[1] + mxh[2]) + hash2[9] - hash2[11] - hash2[14]);
	q[4] = hash[5] + (SHR2(tmp, 1) ^ tmp);

	q[16] += (SHR2(q[2], 2) ^ SHL2(q[2], 2) ^ ROL2(q[2], 28) ^ ROL2(q[2], 59)) + (SHR2(q[3], 1) ^ SHL2(q[3], 3) ^ ROL2(q[3], 4) ^ ROL2(q[3], 37));
	q[17] += (SHR2(q[3], 2) ^ SHL2(q[3], 2) ^ ROL2(q[3], 28) ^ ROL2(q[3], 59)) + (SHR2(q[4], 1) ^ SHL2(q[4], 3) ^ ROL2(q[4], 4) ^ ROL2(q[4], 37));

	tmp = vectorize((mxh[3] - mxh[2] + hash2[10] - hash2[12] + (512 ^ hash2[15])));
	q[5] = hash[6] + (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37));

	tmp = vectorize((mxh[4]) - (mxh[0]) - (mxh[3]) + hash2[13] - hash2[11]);
	q[6] = hash[7] + (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43));

	q[16] += (SHR2(q[4], 1) ^ SHL2(q[4], 2) ^ ROL2(q[4], 13) ^ ROL2(q[4], 43)) + (SHR2(q[5], 2) ^ SHL2(q[5], 1) ^ ROL2(q[5], 19) ^ ROL2(q[5], 53));
	q[17] += (SHR2(q[5], 1) ^ SHL2(q[5], 2) ^ ROL2(q[5], 13) ^ ROL2(q[5], 43)) + (SHR2(q[6], 2) ^ SHL2(q[6], 1) ^ ROL2(q[6], 19) ^ ROL2(q[6], 53));

	tmp = vectorize((mxh[1]) - (mxh[4]) - (mxh[5]) - hash2[12] - hash2[14]);
	q[7] = hash[8] + (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53));

	tmp = vectorize((mxh[2]) - (mxh[5]) - (mxh[6]) + hash2[13] - (512 ^ hash2[15]));
	q[8] = hash[9] + (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59));

	q[16] += (SHR2(q[6], 2) ^ SHL2(q[6], 2) ^ ROL2(q[6], 28) ^ ROL2(q[6], 59)) + (SHR2(q[7], 1) ^ SHL2(q[7], 3) ^ ROL2(q[7], 4) ^ ROL2(q[7], 37));
	q[17] += (SHR2(q[7], 2) ^ SHL2(q[7], 2) ^ ROL2(q[7], 28) ^ ROL2(q[7], 59)) + (SHR2(q[8], 1) ^ SHL2(q[8], 3) ^ ROL2(q[8], 4) ^ ROL2(q[8], 37));

	tmp = vectorize((mxh[0]) - (mxh[3]) + (mxh[6]) - (mxh[7]) + (hash2[14]));
	q[9] = hash[10] + (SHR2(tmp, 1) ^ tmp);

	tmp = vectorize((512 ^ hash2[15]) + hash2[8] - (mxh[1]) - (mxh[4]) - (mxh[7]));
	q[10] = hash[11] + (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37));

	q[16] += (SHR2(q[8], 1) ^ SHL2(q[8], 2) ^ ROL2(q[8], 13) ^ ROL2(q[8], 43)) + (SHR2(q[9], 2) ^ SHL2(q[9], 1) ^ ROL2(q[9], 19) ^ ROL2(q[9], 53));
	q[17] += (SHR2(q[9], 1) ^ SHL2(q[9], 2) ^ ROL2(q[9], 13) ^ ROL2(q[9], 43)) + (SHR2(q[10], 2) ^ SHL2(q[10], 1) ^ ROL2(q[10], 19) ^ ROL2(q[10], 53));

	tmp = vectorize(hash2[9] + hash2[8] - (mxh[0]) - (mxh[2]) - (mxh[5]));
	q[11] = hash[12] + (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43));

	tmp = vectorize((mxh[1]) + (mxh[3]) - (mxh[6]) + hash2[10] - hash2[9]);
	q[12] = hash[13] + (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53));

	q[16] += (SHR2(q[10], 2) ^ SHL2(q[10], 2) ^ ROL2(q[10], 28) ^ ROL2(q[10], 59)) + (SHR2(q[11], 1) ^ SHL2(q[11], 3) ^ ROL2(q[11], 4) ^ ROL2(q[11], 37));
	q[17] += (SHR2(q[11], 2) ^ SHL2(q[11], 2) ^ ROL2(q[11], 28) ^ ROL2(q[11], 59)) + (SHR2(q[12], 1) ^ SHL2(q[12], 3) ^ ROL2(q[12], 4) ^ ROL2(q[12], 37));

	tmp = vectorize((mxh[2]) + (mxh[4]) + (mxh[7]) + hash2[10] + hash2[11]);
	q[13] = hash[14] + (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59));

	tmp = vectorize((mxh[3]) - (mxh[5]) + hash2[8] - hash2[11] - hash2[12]);
	q[14] = hash[15] + (SHR2(tmp, 1) ^ tmp);

	q[16] += (SHR2(q[12], 1) ^ SHL2(q[12], 2) ^ ROL2(q[12], 13) ^ ROL2(q[12], 43)) + (SHR2(q[13], 2) ^ SHL2(q[13], 1) ^ ROL2(q[13], 19) ^ ROL2(q[13], 53));
	q[17] += (SHR2(q[13], 1) ^ SHL2(q[13], 2) ^ ROL2(q[13], 13) ^ ROL2(q[13], 43)) + (SHR2(q[14], 2) ^ SHL2(q[14], 1) ^ ROL2(q[14], 19) ^ ROL2(q[14], 53));

	tmp = vectorize(hash2[12] - hash2[9] + hash2[13] - (mxh[4]) - (mxh[6]));
	q[15] = hash[0] + (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37));

	q[16] += (SHR2(q[14], 2) ^ SHL2(q[14], 2) ^ ROL2(q[14], 28) ^ ROL2(q[14], 59)) + (SHR2(q[15], 1) ^ SHL2(q[15], 3) ^ ROL2(q[15], 4) ^ ROL2(q[15], 37)) +
		((precalcf[0] + ROTL64(msg[0], 1) + ROTL64(msg[3], 4)) ^ hash[7]);

	q[17] +=
		(SHR2(q[15], 2) ^ SHL2(q[15], 2) ^ ROL2(q[15], 28) ^ ROL2(q[15], 59)) + (SHR2(q[16], 1) ^ SHL2(q[16], 3) ^ ROL2(q[16], 4) ^ ROL2(q[16], 37)) +
		((precalcf[1] + ROTL64(msg[1], 2) + ROTL64(msg[4], 5)) ^ hash[8]);

	uint64_t add1 = devectorize(q[2] + q[4] + q[6] + q[8] + q[10] + q[12] + q[14]);
	uint64_t add2 = devectorize(q[3] + q[5] + q[7] + q[9] + q[11] + q[13] + q[15]);

	uint2 XL64 = q[16] ^ q[17];

	q[18] = vectorize(CONST_EXP3d(2) + add1 + devectorize((precalcf[2] + ROTL64(msg[2], 3) + ROTL64(msg[5], 6)) ^ hash[9]));
	q[19] = vectorize(CONST_EXP3d(3) + add2 + devectorize((precalcf[3] + ROTL64(msg[3], 4) + ROTL64(msg[6], 7)) ^ hash[10]));

	add1 += devectorize(q[16] - q[2]);
	add2 += devectorize(q[17] - q[3]);

	XL64 = xor3x(XL64, q[18], q[19]);

	q[20] = vectorize(CONST_EXP3d(4) + add1 + devectorize((precalcf[4] + ROTL64(msg[4], 5) + ROTL64(msg[7], 8)) ^ hash[11]));
	q[21] = vectorize(CONST_EXP3d(5) + add2 + devectorize((precalcf[5] + ROTL64(msg[5], 6)) ^ hash[5 + 7]));

	add1 += devectorize(q[18] - q[4]);
	add2 += devectorize(q[19] - q[5]);

	XL64 = xor3x(XL64, q[20], q[21]);

	q[22] = vectorize(CONST_EXP3d(6) + add1 + devectorize((vectorize((22)*(0x0555555555555555ull)) + ROTL64(msg[6], 7) - ROTL64(msg[0], 1)) ^ hash[13]));
	q[23] = vectorize(CONST_EXP3d(7) + add2 + devectorize((vectorize((23)*(0x0555555555555555ull)) + ROTL64(msg[7], 8) - ROTL64(msg[1], 2)) ^ hash[14]));

	add1 += devectorize(q[20] - q[6]);
	add2 += devectorize(q[21] - q[7]);

	XL64 = xor3x(XL64, q[22], q[23]);

	q[24] = vectorize(CONST_EXP3d(8) + add1 + devectorize((vectorize((24)*(0x0555555555555555ull) + 0x10000) - ROTL64(msg[2], 3)) ^ hash[15]));
	q[25] = vectorize(CONST_EXP3d(9) + add2 + devectorize((vectorize((25)*(0x0555555555555555ull)) - ROTL64(msg[3], 4)) ^ hash[0]));

	add1 += devectorize(q[22] - q[8]);
	add2 += devectorize(q[23] - q[9]);

	uint2 XH64 = xor3x(XL64, q[24], q[25]);

	q[26] = vectorize(CONST_EXP3d(10) + add1 + devectorize((vectorize((26)*(0x0555555555555555ull)) - ROTL64(msg[4], 5)) ^ hash[1]));
	q[27] = vectorize(CONST_EXP3d(11) + add2 + devectorize((vectorize((27)*(0x0555555555555555ull)) - ROTL64(msg[5], 6)) ^ hash[2]));

	add1 += devectorize(q[24] - q[10]);
	add2 += devectorize(q[25] - q[11]);

	XH64 = xor3x(XH64, q[26], q[27]);

	q[28] = vectorize(CONST_EXP3d(12) + add1 + devectorize((vectorize(0x955555555755554C) - ROTL64(msg[6], 7)) ^ hash[3]));
	q[29] = vectorize(CONST_EXP3d(13) + add2 + devectorize((precalcf[6] + ROTL64(msg[0], 1) - ROTL64(msg[7], 8)) ^ hash[4]));

	add1 += devectorize(q[26] - q[12]);
	add2 += devectorize(q[27] - q[13]);

	XH64 = xor3x(XH64, q[28], q[29]);

	q[30] = vectorize(CONST_EXP3d(14) + add1 + devectorize((precalcf[7] + ROTL64(msg[1], 2)) ^ hash[5]));
	q[31] = vectorize(CONST_EXP3d(15) + add2 + devectorize((precalcf[8] + ROTL64(msg[2], 3)) ^ hash[6]));

	XH64 = xor3x(XH64, q[30], q[31]);

	h[0] = (SHL2(XH64, 5) ^ SHR2(q[16], 5) ^ vectorize(msg[0])) + (XL64 ^ q[24] ^ q[0]);
	h[1] = (SHR2(XH64, 7) ^ SHL8(q[17]) ^ vectorize(msg[1])) + (XL64 ^ q[25] ^ q[1]);
	h[2] = (SHR2(XH64, 5) ^ SHL2(q[18], 5) ^ vectorize(msg[2])) + (XL64 ^ q[26] ^ q[2]);
	h[3] = (SHR2(XH64, 1) ^ SHL2(q[19], 5) ^ vectorize(msg[3])) + (XL64 ^ q[27] ^ q[3]);
	h[4] = (SHR2(XH64, 3) ^ q[20] ^ vectorize(msg[4])) + (XL64 ^ q[28] ^ q[4]);
	h[5] = (SHL2(XH64, 6) ^ SHR2(q[21], 6) ^ vectorize(msg[5])) + (XL64 ^ q[29] ^ q[5]);
	h[6] = (SHR2(XH64, 4) ^ SHL2(q[22], 6) ^ vectorize(msg[6])) + (XL64 ^ q[30] ^ q[6]);
	h[7] = (SHR2(XH64, 11) ^ SHL2(q[23], 2) ^ vectorize(msg[7])) + (XL64 ^ q[31] ^ q[7]);

	h[8] = (ROL2(h[4], 9)) + (XH64 ^ q[24] ^ 0x80) + (SHL8(XL64) ^ q[23] ^ q[8]);
	h[9] = (ROL2(h[5], 10)) + (XH64 ^ q[25]) + (SHR2(XL64, 6) ^ q[16] ^ q[9]);
	h[10] = (ROL2(h[6], 11)) + (XH64 ^ q[26]) + (SHL2(XL64, 6) ^ q[17] ^ q[10]);
	h[11] = (ROL2(h[7], 12)) + (XH64 ^ q[27]) + (SHL2(XL64, 4) ^ q[18] ^ q[11]);
	h[12] = (ROL2(h[0], 13)) + (XH64 ^ q[28]) + (SHR2(XL64, 3) ^ q[19] ^ q[12]);
	h[13] = (ROL2(h[1], 14)) + (XH64 ^ q[29]) + (SHR2(XL64, 4) ^ q[20] ^ q[13]);
	h[14] = (ROL2(h[2], 15)) + (XH64 ^ q[30]) + (SHR2(XL64, 7) ^ q[21] ^ q[14]);
	h[15] = (ROL16(h[3])) + (XH64 ^ q[31] ^ 512) + (SHR2(XL64, 2) ^ q[22] ^ q[15]);
}

__global__ __launch_bounds__(256, 2)
void quark_bmw512_gpu_hash_64(uint32_t threads, uint64_t *const __restrict__ g_hash, const uint32_t *const __restrict__ g_nonceVector)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads){

		const uint32_t hashPosition = (g_nonceVector == NULL) ? thread : g_nonceVector[thread];

		uint64_t *inpHash = &g_hash[8 * hashPosition];

		uint64_t __align__(16) msg[16];
		uint2    __align__(16) h[16];

		uint2x4* phash = (uint2x4*)inpHash;
		uint2x4* outpt = (uint2x4*)msg;
		outpt[0] = __ldg4(&phash[0]);
		outpt[1] = __ldg4(&phash[1]);

		uint2 q[32];

		bmw512_round1(q, h, msg);

		const uint2 __align__(16) cmsg[16] = {
			0xaaaaaaa0, 0xaaaaaaaa, 0xaaaaaaa1, 0xaaaaaaaa, 0xaaaaaaa2, 0xaaaaaaaa, 0xaaaaaaa3, 0xaaaaaaaa,
			0xaaaaaaa4, 0xaaaaaaaa, 0xaaaaaaa5, 0xaaaaaaaa, 0xaaaaaaa6, 0xaaaaaaaa, 0xaaaaaaa7, 0xaaaaaaaa,
			0xaaaaaaa8, 0xaaaaaaaa, 0xaaaaaaa9, 0xaaaaaaaa, 0xaaaaaaaa, 0xaaaaaaaa, 0xaaaaaaab, 0xaaaaaaaa,
			0xaaaaaaac, 0xaaaaaaaa, 0xaaaaaaad, 0xaaaaaaaa, 0xaaaaaaae, 0xaaaaaaaa, 0xaaaaaaaf, 0xaaaaaaaa
		};

#pragma unroll 16
		for (int i = 0; i < 16; i++)
			msg[i] = devectorize(cmsg[i] ^ h[i]);

		const uint2 __align__(16) precalc[16] = {
			{ 0x55555550, 0x55555555 }, { 0xAAAAAAA5, 0x5AAAAAAA }, { 0xFFFFFFFA, 0x5FFFFFFF }, { 0x5555554F, 0x65555555 },
			{ 0xAAAAAAA4, 0x6AAAAAAA }, { 0xFFFFFFF9, 0x6FFFFFFF }, { 0x5555554E, 0x75555555 }, { 0xAAAAAAA3, 0x7AAAAAAA },
			{ 0xFFFFFFF8, 0x7FFFFFFF }, { 0x5555554D, 0x85555555 }, { 0xAAAAAAA2, 0x8AAAAAAA }, { 0xFFFFFFF7, 0x8FFFFFFF },
			{ 0x5555554C, 0x95555555 }, { 0xAAAAAAA1, 0x9AAAAAAA }, { 0xFFFFFFF6, 0x9FFFFFFF }, { 0x5555554B, 0xA5555555 }
		};

		const uint64_t p2 = msg[15] - msg[12];
		const uint64_t p3 = msg[14] - msg[7];
		const uint64_t p4 = msg[6] + msg[9];
		const uint64_t p5 = msg[8] - msg[5];
		const uint64_t p6 = msg[1] - msg[14];
		const uint64_t p7 = msg[8] - msg[1];
		const uint64_t p8 = msg[3] + msg[10];


		uint2 tmp = vectorize((msg[5]) + (msg[10]) + (msg[13]) + p3);
		q[0] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37)) + cmsg[1];

		tmp = vectorize((msg[6]) - (msg[8]) + (msg[11]) + (msg[14]) - (msg[15]));
		q[1] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43)) + cmsg[2];

		tmp = vectorize((msg[0]) + (msg[7]) + (msg[9]) + p2);
		q[2] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53)) + cmsg[3];

		tmp = vectorize((msg[0]) + p7 - (msg[10]) + (msg[13]));
		q[3] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59)) + cmsg[4];

		tmp = vectorize((msg[2]) + (msg[9]) - (msg[11]) + p6);
		q[4] = (SHR2(tmp, 1) ^ tmp) + cmsg[5];

		tmp = vectorize(p8 + p2 - (msg[2]));
		q[5] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37)) + cmsg[6];

		tmp = vectorize((msg[4]) - (msg[0]) - (msg[3]) - (msg[11]) + (msg[13]));
		q[6] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43)) + cmsg[7];

		tmp = vectorize(p6 - (msg[4]) - (msg[5]) - (msg[12]));
		q[7] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53)) + cmsg[8];

		tmp = vectorize((msg[2]) - (msg[5]) - (msg[6]) + (msg[13]) - (msg[15]));
		q[8] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59)) + cmsg[9];

		tmp = vectorize((msg[0]) - (msg[3]) + (msg[6]) + p3);
		q[9] = (SHR2(tmp, 1) ^ tmp) + cmsg[10];

		tmp = vectorize(p7 - (msg[4]) - (msg[7]) + (msg[15]));
		q[10] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37)) + cmsg[11];

		tmp = vectorize(p5 - (msg[0]) - (msg[2]) + (msg[9]));
		q[11] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43)) + cmsg[12];

		tmp = vectorize(p8 + msg[1] - p4);
		q[12] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53)) + cmsg[13];

		tmp = vectorize((msg[2]) + (msg[4]) + (msg[7]) + (msg[10]) + (msg[11]));
		q[13] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59)) + cmsg[14];

		tmp = vectorize((msg[3]) + p5 - (msg[11]) - (msg[12]));
		q[14] = (SHR2(tmp, 1) ^ tmp) + cmsg[15];

		tmp = vectorize((msg[12]) - (msg[4]) - p4 + (msg[13]));
		q[15] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37)) + cmsg[0];

		q[16] =
			vectorize(devectorize(SHR2(q[0], 1) ^ SHL2(q[0], 2) ^ ROL2(q[0], 13) ^ ROL2(q[0], 43)) + devectorize(SHR2(q[1], 2) ^ SHL2(q[1], 1) ^ ROL2(q[1], 19) ^ ROL2(q[1], 53)) +
			devectorize(SHR2(q[2], 2) ^ SHL2(q[2], 2) ^ ROL2(q[2], 28) ^ ROL2(q[2], 59)) + devectorize(SHR2(q[3], 1) ^ SHL2(q[3], 3) ^ ROL2(q[3], 4) ^ ROL2(q[3], 37)) +
			devectorize(SHR2(q[4], 1) ^ SHL2(q[4], 2) ^ ROL2(q[4], 13) ^ ROL2(q[4], 43)) + devectorize(SHR2(q[5], 2) ^ SHL2(q[5], 1) ^ ROL2(q[5], 19) ^ ROL2(q[5], 53)) +
			devectorize(SHR2(q[6], 2) ^ SHL2(q[6], 2) ^ ROL2(q[6], 28) ^ ROL2(q[6], 59)) + devectorize(SHR2(q[7], 1) ^ SHL2(q[7], 3) ^ ROL2(q[7], 4) ^ ROL2(q[7], 37)) +
			devectorize(SHR2(q[8], 1) ^ SHL2(q[8], 2) ^ ROL2(q[8], 13) ^ ROL2(q[8], 43)) + devectorize(SHR2(q[9], 2) ^ SHL2(q[9], 1) ^ ROL2(q[9], 19) ^ ROL2(q[9], 53)) +
			devectorize(SHR2(q[10], 2) ^ SHL2(q[10], 2) ^ ROL2(q[10], 28) ^ ROL2(q[10], 59)) + devectorize(SHR2(q[11], 1) ^ SHL2(q[11], 3) ^ ROL2(q[11], 4) ^ ROL2(q[11], 37)) +
			devectorize(SHR2(q[12], 1) ^ SHL2(q[12], 2) ^ ROL2(q[12], 13) ^ ROL2(q[12], 43)) + devectorize(SHR2(q[13], 2) ^ SHL2(q[13], 1) ^ ROL2(q[13], 19) ^ ROL2(q[13], 53)) +
			devectorize(SHR2(q[14], 2) ^ SHL2(q[14], 2) ^ ROL2(q[14], 28) ^ ROL2(q[14], 59)) + devectorize(SHR2(q[15], 1) ^ SHL2(q[15], 3) ^ ROL2(q[15], 4) ^ ROL2(q[15], 37)) +
			devectorize((precalc[0] + ROL2(h[0], 1) + ROL2(h[3], 4) - ROL2(h[10], 11)) ^ cmsg[7]));
		q[17] =
			vectorize(devectorize(SHR2(q[1], 1) ^ SHL2(q[1], 2) ^ ROL2(q[1], 13) ^ ROL2(q[1], 43)) + devectorize(SHR2(q[2], 2) ^ SHL2(q[2], 1) ^ ROL2(q[2], 19) ^ ROL2(q[2], 53)) +
			devectorize(SHR2(q[3], 2) ^ SHL2(q[3], 2) ^ ROL2(q[3], 28) ^ ROL2(q[3], 59)) + devectorize(SHR2(q[4], 1) ^ SHL2(q[4], 3) ^ ROL2(q[4], 4) ^ ROL2(q[4], 37)) +
			devectorize(SHR2(q[5], 1) ^ SHL2(q[5], 2) ^ ROL2(q[5], 13) ^ ROL2(q[5], 43)) + devectorize(SHR2(q[6], 2) ^ SHL2(q[6], 1) ^ ROL2(q[6], 19) ^ ROL2(q[6], 53)) +
			devectorize(SHR2(q[7], 2) ^ SHL2(q[7], 2) ^ ROL2(q[7], 28) ^ ROL2(q[7], 59)) + devectorize(SHR2(q[8], 1) ^ SHL2(q[8], 3) ^ ROL2(q[8], 4) ^ ROL2(q[8], 37)) +
			devectorize(SHR2(q[9], 1) ^ SHL2(q[9], 2) ^ ROL2(q[9], 13) ^ ROL2(q[9], 43)) + devectorize(SHR2(q[10], 2) ^ SHL2(q[10], 1) ^ ROL2(q[10], 19) ^ ROL2(q[10], 53)) +
			devectorize(SHR2(q[11], 2) ^ SHL2(q[11], 2) ^ ROL2(q[11], 28) ^ ROL2(q[11], 59)) + devectorize(SHR2(q[12], 1) ^ SHL2(q[12], 3) ^ ROL2(q[12], 4) ^ ROL2(q[12], 37)) +
			devectorize(SHR2(q[13], 1) ^ SHL2(q[13], 2) ^ ROL2(q[13], 13) ^ ROL2(q[13], 43)) + devectorize(SHR2(q[14], 2) ^ SHL2(q[14], 1) ^ ROL2(q[14], 19) ^ ROL2(q[14], 53)) +
			devectorize(SHR2(q[15], 2) ^ SHL2(q[15], 2) ^ ROL2(q[15], 28) ^ ROL2(q[15], 59)) + devectorize(SHR2(q[16], 1) ^ SHL2(q[16], 3) ^ ROL2(q[16], 4) ^ ROL2(q[16], 37)) +
			devectorize((precalc[1] + ROL2(h[1], 2) + ROL2(h[4], 5) - ROL2(h[11], 12)) ^ cmsg[8]));

		uint64_t add1 = devectorize(q[2] + q[4] + q[6] + q[8] + q[10] + q[12] + q[14]);
		uint64_t add2 = devectorize(q[3] + q[5] + q[7] + q[9] + q[11] + q[13] + q[15]);

		uint2 XL64 = q[16] ^ q[17];

		q[18] = vectorize(add1 + CONST_EXP3d(2) + devectorize((precalc[2] + ROL2(h[2], 3) + ROL2(h[5], 6) - ROL2(h[12], 13)) ^ cmsg[9]));
		q[19] = vectorize(add2 + CONST_EXP3d(3) + devectorize((precalc[3] + ROL2(h[3], 4) + ROL2(h[6], 7) - ROL2(h[13], 14)) ^ cmsg[10]));

		add1 = add1 - devectorize(q[2] - q[16]);
		add2 = add2 - devectorize(q[3] - q[17]);

		XL64 = xor3x(XL64, q[18], q[19]);

		q[20] = vectorize(add1 + CONST_EXP3d(4) + devectorize((precalc[4] + ROL2(h[4], 5) + ROL8(h[7]) - ROL2(h[14], 15)) ^ cmsg[11]));
		q[21] = vectorize(add2 + CONST_EXP3d(5) + devectorize((precalc[5] + ROL2(h[5], 6) + ROL2(h[8], 9) - ROL16(h[15])) ^ cmsg[12]));

		add1 = add1 - devectorize(q[4] - q[18]);
		add2 = add2 - devectorize(q[5] - q[19]);

		XL64 = xor3x(XL64, q[20], q[21]);

		q[22] = vectorize(add1 + CONST_EXP3d(6) + devectorize((precalc[6] + ROL2(h[6], 7) + ROL2(h[9], 10) - ROL2(h[0], 1)) ^ cmsg[13]));
		q[23] = vectorize(add2 + CONST_EXP3d(7) + devectorize((precalc[7] + ROL8(h[7]) + ROL2(h[10], 11) - ROL2(h[1], 2)) ^ cmsg[14]));

		add1 -= devectorize(q[6] - q[20]);
		add2 -= devectorize(q[7] - q[21]);

		XL64 = xor3x(XL64, q[22], q[23]);

		q[24] = vectorize(add1 + CONST_EXP3d(8) + devectorize((precalc[8] + ROL2(h[8], 9) + ROL2(h[11], 12) - ROL2(h[2], 3)) ^ cmsg[15]));
		q[25] = vectorize(add2 + CONST_EXP3d(9) + devectorize((precalc[9] + ROL2(h[9], 10) + ROL2(h[12], 13) - ROL2(h[3], 4)) ^ cmsg[0]));

		add1 -= devectorize(q[8] - q[22]);
		add2 -= devectorize(q[9] - q[23]);

		uint2 XH64 = xor3x(XL64, q[24], q[25]);

		q[26] = vectorize(add1 + CONST_EXP3d(10) + devectorize((precalc[10] + ROL2(h[10], 11) + ROL2(h[13], 14) - ROL2(h[4], 5)) ^ cmsg[1]));
		q[27] = vectorize(add2 + CONST_EXP3d(11) + devectorize((precalc[11] + ROL2(h[11], 12) + ROL2(h[14], 15) - ROL2(h[5], 6)) ^ cmsg[2]));

		add1 -= devectorize(q[10] - q[24]);
		add2 -= devectorize(q[11] - q[25]);

		XH64 = xor3x(XH64, q[26], q[27]);

		q[28] = vectorize(add1 + CONST_EXP3d(12) + devectorize((precalc[12] + ROL2(h[12], 13) + ROL16(h[15]) - ROL2(h[6], 7)) ^ cmsg[3]));
		q[29] = vectorize(add2 + CONST_EXP3d(13) + devectorize((precalc[13] + ROL2(h[13], 14) + ROL2(h[0], 1) - ROL8(h[7])) ^ cmsg[4]));

		add1 -= devectorize(q[12] - q[26]);
		add2 -= devectorize(q[13] - q[27]);

		XH64 = xor3x(XH64, q[28], q[29]);

		q[30] = vectorize(add1 + CONST_EXP3d(14) + devectorize((precalc[14] + ROL2(h[14], 15) + ROL2(h[1], 2) - ROL2(h[8], 9)) ^ cmsg[5]));
		q[31] = vectorize(add2 + CONST_EXP3d(15) + devectorize((precalc[15] + ROL16(h[15]) + ROL2(h[2], 3) - ROL2(h[9], 10)) ^ cmsg[6]));

		XH64 = xor3x(XH64, q[30], q[31]);

		msg[0] = devectorize((SHL2(XH64, 5) ^ SHR2(q[16], 5) ^ h[0]) + (XL64 ^ q[24] ^ q[0]));
		msg[1] = devectorize((SHR2(XH64, 7) ^ SHL8(q[17]) ^ h[1]) + (XL64 ^ q[25] ^ q[1]));
		msg[2] = devectorize((SHR2(XH64, 5) ^ SHL2(q[18], 5) ^ h[2]) + (XL64 ^ q[26] ^ q[2]));
		msg[3] = devectorize((SHR2(XH64, 1) ^ SHL2(q[19], 5) ^ h[3]) + (XL64 ^ q[27] ^ q[3]));
		msg[4] = devectorize((SHR2(XH64, 3) ^ q[20] ^ h[4]) + (XL64 ^ q[28] ^ q[4]));
		msg[5] = devectorize((SHL2(XH64, 6) ^ SHR2(q[21], 6) ^ h[5]) + (XL64 ^ q[29] ^ q[5]));
		msg[6] = devectorize((SHR2(XH64, 4) ^ SHL2(q[22], 6) ^ h[6]) + (XL64 ^ q[30] ^ q[6]));
		msg[7] = devectorize((SHR2(XH64, 11) ^ SHL2(q[23], 2) ^ h[7]) + (XL64 ^ q[31] ^ q[7]));
		msg[8] = devectorize((XH64 ^ q[24] ^ h[8]) + (SHL8(XL64) ^ q[23] ^ q[8]) + ROTL64(msg[4], 9));

		msg[9] = devectorize((XH64 ^ q[25] ^ h[9]) + (SHR2(XL64, 6) ^ q[16] ^ q[9]) + ROTL64(msg[5], 10));
		msg[10] = devectorize((XH64 ^ q[26] ^ h[10]) + (SHL2(XL64, 6) ^ q[17] ^ q[10]) + ROTL64(msg[6], 11));
		msg[11] = devectorize((XH64 ^ q[27] ^ h[11]) + (SHL2(XL64, 4) ^ q[18] ^ q[11]) + ROTL64(msg[7], 12));

#if __CUDA_ARCH__ > 500
		* (uint2x4*)&inpHash[0] = *(uint2x4*)&msg[8];
#endif

		msg[12] = devectorize((XH64 ^ q[28] ^ h[12]) + (SHR2(XL64, 3) ^ q[19] ^ q[12]) + ROTL64(msg[0], 13));
		msg[13] = devectorize((XH64 ^ q[29] ^ h[13]) + (SHR2(XL64, 4) ^ q[20] ^ q[13]) + ROTL64(msg[1], 14));
		msg[14] = devectorize((XH64 ^ q[30] ^ h[14]) + (SHR2(XL64, 7) ^ q[21] ^ q[14]) + ROTL64(msg[2], 15));
		msg[15] = devectorize((XH64 ^ q[31] ^ h[15]) + (SHR2(XL64, 2) ^ q[22] ^ q[15]) + ROTL64(msg[3], 16));

#if __CUDA_ARCH__ > 500
		* (uint2x4*)&inpHash[4] = *(uint2x4*)&msg[12];
#else
		*(uint2x4*)&inpHash[0] = *(uint2x4*)&msg[8];
		*(uint2x4*)&inpHash[4] = *(uint2x4*)&msg[12];
#endif
	}
}
__global__ __launch_bounds__(256, 2)
void quark_bmw512_gpu_hash_64_final(uint32_t threads, uint64_t *const __restrict__ g_hash, const uint32_t *const __restrict__ g_nonceVector, uint32_t* resNonce, const uint64_t target)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads){

		const uint32_t hashPosition = (g_nonceVector == NULL) ? thread : g_nonceVector[thread];

		uint64_t *inpHash = &g_hash[8 * hashPosition];

		uint64_t __align__(16) msg[16];
		uint2    __align__(16) h[16];

		uint2x4* phash = (uint2x4*)inpHash;
		uint2x4* outpt = (uint2x4*)msg;
		outpt[0] = __ldg4(&phash[0]);
		outpt[1] = __ldg4(&phash[1]);

		uint2 q[32];

		bmw512_round1(q, h, msg);

		const uint2 __align__(16) cmsg[16] = {
			0xaaaaaaa0, 0xaaaaaaaa, 0xaaaaaaa1, 0xaaaaaaaa, 0xaaaaaaa2, 0xaaaaaaaa, 0xaaaaaaa3, 0xaaaaaaaa,
			0xaaaaaaa4, 0xaaaaaaaa, 0xaaaaaaa5, 0xaaaaaaaa, 0xaaaaaaa6, 0xaaaaaaaa, 0xaaaaaaa7, 0xaaaaaaaa,
			0xaaaaaaa8, 0xaaaaaaaa, 0xaaaaaaa9, 0xaaaaaaaa, 0xaaaaaaaa, 0xaaaaaaaa, 0xaaaaaaab, 0xaaaaaaaa,
			0xaaaaaaac, 0xaaaaaaaa, 0xaaaaaaad, 0xaaaaaaaa, 0xaaaaaaae, 0xaaaaaaaa, 0xaaaaaaaf, 0xaaaaaaaa
		};

#pragma unroll 16
		for (int i = 0; i < 16; i++)
			msg[i] = devectorize(cmsg[i] ^ h[i]);

		const uint2 __align__(16) precalc[16] = {
			{ 0x55555550, 0x55555555 }, { 0xAAAAAAA5, 0x5AAAAAAA }, { 0xFFFFFFFA, 0x5FFFFFFF }, { 0x5555554F, 0x65555555 },
			{ 0xAAAAAAA4, 0x6AAAAAAA }, { 0xFFFFFFF9, 0x6FFFFFFF }, { 0x5555554E, 0x75555555 }, { 0xAAAAAAA3, 0x7AAAAAAA },
			{ 0xFFFFFFF8, 0x7FFFFFFF }, { 0x5555554D, 0x85555555 }, { 0xAAAAAAA2, 0x8AAAAAAA }, { 0xFFFFFFF7, 0x8FFFFFFF },
			{ 0x5555554C, 0x95555555 }, { 0xAAAAAAA1, 0x9AAAAAAA }, { 0xFFFFFFF6, 0x9FFFFFFF }, { 0x5555554B, 0xA5555555 }
		};

		const uint64_t p2 = msg[15] - msg[12];
		const uint64_t p3 = msg[14] - msg[7];
		const uint64_t p4 = msg[6] + msg[9];
		const uint64_t p5 = msg[8] - msg[5];
		const uint64_t p6 = msg[1] - msg[14];
		const uint64_t p7 = msg[8] - msg[1];
		const uint64_t p8 = msg[3] + msg[10];


		uint2 tmp = vectorize((msg[5]) + (msg[10]) + (msg[13]) + p3);
		q[0] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37)) + cmsg[1];

		tmp = vectorize((msg[6]) - (msg[8]) + (msg[11]) + (msg[14]) - (msg[15]));
		q[1] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43)) + cmsg[2];

		tmp = vectorize((msg[0]) + (msg[7]) + (msg[9]) + p2);
		q[2] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53)) + cmsg[3];

		tmp = vectorize((msg[0]) + p7 - (msg[10]) + (msg[13]));
		q[3] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59)) + cmsg[4];

		tmp = vectorize((msg[2]) + (msg[9]) - (msg[11]) + p6);
		q[4] = (SHR2(tmp, 1) ^ tmp) + cmsg[5];

		tmp = vectorize(p8 + p2 - (msg[2]));
		q[5] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37)) + cmsg[6];

		tmp = vectorize((msg[4]) - (msg[0]) - (msg[3]) - (msg[11]) + (msg[13]));
		q[6] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43)) + cmsg[7];

		tmp = vectorize(p6 - (msg[4]) - (msg[5]) - (msg[12]));
		q[7] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53)) + cmsg[8];

		tmp = vectorize((msg[2]) - (msg[5]) - (msg[6]) + (msg[13]) - (msg[15]));
		q[8] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59)) + cmsg[9];

		tmp = vectorize((msg[0]) - (msg[3]) + (msg[6]) + p3);
		q[9] = (SHR2(tmp, 1) ^ tmp) + cmsg[10];

		tmp = vectorize(p7 - (msg[4]) - (msg[7]) + (msg[15]));
		q[10] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37)) + cmsg[11];

		tmp = vectorize(p5 - (msg[0]) - (msg[2]) + (msg[9]));
		q[11] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43)) + cmsg[12];

		tmp = vectorize(p8 + msg[1] - p4);
		q[12] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53)) + cmsg[13];

		tmp = vectorize((msg[2]) + (msg[4]) + (msg[7]) + (msg[10]) + (msg[11]));
		q[13] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59)) + cmsg[14];

		tmp = vectorize((msg[3]) + p5 - (msg[11]) - (msg[12]));
		q[14] = (SHR2(tmp, 1) ^ tmp) + cmsg[15];

		tmp = vectorize((msg[12]) - (msg[4]) - p4 + (msg[13]));
		q[15] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37)) + cmsg[0];

		q[16] =
			vectorize(devectorize(SHR2(q[0], 1) ^ SHL2(q[0], 2) ^ ROL2(q[0], 13) ^ ROL2(q[0], 43)) + devectorize(SHR2(q[1], 2) ^ SHL2(q[1], 1) ^ ROL2(q[1], 19) ^ ROL2(q[1], 53)) +
			devectorize(SHR2(q[2], 2) ^ SHL2(q[2], 2) ^ ROL2(q[2], 28) ^ ROL2(q[2], 59)) + devectorize(SHR2(q[3], 1) ^ SHL2(q[3], 3) ^ ROL2(q[3], 4) ^ ROL2(q[3], 37)) +
			devectorize(SHR2(q[4], 1) ^ SHL2(q[4], 2) ^ ROL2(q[4], 13) ^ ROL2(q[4], 43)) + devectorize(SHR2(q[5], 2) ^ SHL2(q[5], 1) ^ ROL2(q[5], 19) ^ ROL2(q[5], 53)) +
			devectorize(SHR2(q[6], 2) ^ SHL2(q[6], 2) ^ ROL2(q[6], 28) ^ ROL2(q[6], 59)) + devectorize(SHR2(q[7], 1) ^ SHL2(q[7], 3) ^ ROL2(q[7], 4) ^ ROL2(q[7], 37)) +
			devectorize(SHR2(q[8], 1) ^ SHL2(q[8], 2) ^ ROL2(q[8], 13) ^ ROL2(q[8], 43)) + devectorize(SHR2(q[9], 2) ^ SHL2(q[9], 1) ^ ROL2(q[9], 19) ^ ROL2(q[9], 53)) +
			devectorize(SHR2(q[10], 2) ^ SHL2(q[10], 2) ^ ROL2(q[10], 28) ^ ROL2(q[10], 59)) + devectorize(SHR2(q[11], 1) ^ SHL2(q[11], 3) ^ ROL2(q[11], 4) ^ ROL2(q[11], 37)) +
			devectorize(SHR2(q[12], 1) ^ SHL2(q[12], 2) ^ ROL2(q[12], 13) ^ ROL2(q[12], 43)) + devectorize(SHR2(q[13], 2) ^ SHL2(q[13], 1) ^ ROL2(q[13], 19) ^ ROL2(q[13], 53)) +
			devectorize(SHR2(q[14], 2) ^ SHL2(q[14], 2) ^ ROL2(q[14], 28) ^ ROL2(q[14], 59)) + devectorize(SHR2(q[15], 1) ^ SHL2(q[15], 3) ^ ROL2(q[15], 4) ^ ROL2(q[15], 37)) +
			devectorize((precalc[0] + ROL2(h[0], 1) + ROL2(h[3], 4) - ROL2(h[10], 11)) ^ cmsg[7]));
		q[17] =
			vectorize(devectorize(SHR2(q[1], 1) ^ SHL2(q[1], 2) ^ ROL2(q[1], 13) ^ ROL2(q[1], 43)) + devectorize(SHR2(q[2], 2) ^ SHL2(q[2], 1) ^ ROL2(q[2], 19) ^ ROL2(q[2], 53)) +
			devectorize(SHR2(q[3], 2) ^ SHL2(q[3], 2) ^ ROL2(q[3], 28) ^ ROL2(q[3], 59)) + devectorize(SHR2(q[4], 1) ^ SHL2(q[4], 3) ^ ROL2(q[4], 4) ^ ROL2(q[4], 37)) +
			devectorize(SHR2(q[5], 1) ^ SHL2(q[5], 2) ^ ROL2(q[5], 13) ^ ROL2(q[5], 43)) + devectorize(SHR2(q[6], 2) ^ SHL2(q[6], 1) ^ ROL2(q[6], 19) ^ ROL2(q[6], 53)) +
			devectorize(SHR2(q[7], 2) ^ SHL2(q[7], 2) ^ ROL2(q[7], 28) ^ ROL2(q[7], 59)) + devectorize(SHR2(q[8], 1) ^ SHL2(q[8], 3) ^ ROL2(q[8], 4) ^ ROL2(q[8], 37)) +
			devectorize(SHR2(q[9], 1) ^ SHL2(q[9], 2) ^ ROL2(q[9], 13) ^ ROL2(q[9], 43)) + devectorize(SHR2(q[10], 2) ^ SHL2(q[10], 1) ^ ROL2(q[10], 19) ^ ROL2(q[10], 53)) +
			devectorize(SHR2(q[11], 2) ^ SHL2(q[11], 2) ^ ROL2(q[11], 28) ^ ROL2(q[11], 59)) + devectorize(SHR2(q[12], 1) ^ SHL2(q[12], 3) ^ ROL2(q[12], 4) ^ ROL2(q[12], 37)) +
			devectorize(SHR2(q[13], 1) ^ SHL2(q[13], 2) ^ ROL2(q[13], 13) ^ ROL2(q[13], 43)) + devectorize(SHR2(q[14], 2) ^ SHL2(q[14], 1) ^ ROL2(q[14], 19) ^ ROL2(q[14], 53)) +
			devectorize(SHR2(q[15], 2) ^ SHL2(q[15], 2) ^ ROL2(q[15], 28) ^ ROL2(q[15], 59)) + devectorize(SHR2(q[16], 1) ^ SHL2(q[16], 3) ^ ROL2(q[16], 4) ^ ROL2(q[16], 37)) +
			devectorize((precalc[1] + ROL2(h[1], 2) + ROL2(h[4], 5) - ROL2(h[11], 12)) ^ cmsg[8]));

		uint64_t add1 = devectorize(q[2] + q[4] + q[6] + q[8] + q[10] + q[12] + q[14]);
		uint64_t add2 = devectorize(q[3] + q[5] + q[7] + q[9] + q[11] + q[13] + q[15]);

		uint2 XL64 = q[16] ^ q[17];

		q[18] = vectorize(add1 + CONST_EXP3d(2) + devectorize((precalc[2] + ROL2(h[2], 3) + ROL2(h[5], 6) - ROL2(h[12], 13)) ^ cmsg[9]));
		q[19] = vectorize(add2 + CONST_EXP3d(3) + devectorize((precalc[3] + ROL2(h[3], 4) + ROL2(h[6], 7) - ROL2(h[13], 14)) ^ cmsg[10]));

		add1 = add1 - devectorize(q[2] - q[16]);
		add2 = add2 - devectorize(q[3] - q[17]);

		XL64 = xor3x(XL64, q[18], q[19]);

		q[20] = vectorize(add1 + CONST_EXP3d(4) + devectorize((precalc[4] + ROL2(h[4], 5) + ROL8(h[7]) - ROL2(h[14], 15)) ^ cmsg[11]));
		q[21] = vectorize(add2 + CONST_EXP3d(5) + devectorize((precalc[5] + ROL2(h[5], 6) + ROL2(h[8], 9) - ROL16(h[15])) ^ cmsg[12]));

		add1 = add1 - devectorize(q[4] - q[18]);
		add2 = add2 - devectorize(q[5] - q[19]);

		XL64 = xor3x(XL64, q[20], q[21]);

		q[22] = vectorize(add1 + CONST_EXP3d(6) + devectorize((precalc[6] + ROL2(h[6], 7) + ROL2(h[9], 10) - ROL2(h[0], 1)) ^ cmsg[13]));
		q[23] = vectorize(add2 + CONST_EXP3d(7) + devectorize((precalc[7] + ROL8(h[7]) + ROL2(h[10], 11) - ROL2(h[1], 2)) ^ cmsg[14]));

		add1 -= devectorize(q[6] - q[20]);
		add2 -= devectorize(q[7] - q[21]);

		XL64 = xor3x(XL64, q[22], q[23]);

		q[24] = vectorize(add1 + CONST_EXP3d(8) + devectorize((precalc[8] + ROL2(h[8], 9) + ROL2(h[11], 12) - ROL2(h[2], 3)) ^ cmsg[15]));
		q[25] = vectorize(add2 + CONST_EXP3d(9) + devectorize((precalc[9] + ROL2(h[9], 10) + ROL2(h[12], 13) - ROL2(h[3], 4)) ^ cmsg[0]));

		add1 -= devectorize(q[8] - q[22]);
		add2 -= devectorize(q[9] - q[23]);

		uint2 XH64 = xor3x(XL64, q[24], q[25]);

		q[26] = vectorize(add1 + CONST_EXP3d(10) + devectorize((precalc[10] + ROL2(h[10], 11) + ROL2(h[13], 14) - ROL2(h[4], 5)) ^ cmsg[1]));
		q[27] = vectorize(add2 + CONST_EXP3d(11) + devectorize((precalc[11] + ROL2(h[11], 12) + ROL2(h[14], 15) - ROL2(h[5], 6)) ^ cmsg[2]));

		add1 -= devectorize(q[10] - q[24]);
		add2 -= devectorize(q[11] - q[25]);

		XH64 = xor3x(XH64, q[26], q[27]);

		q[28] = vectorize(add1 + CONST_EXP3d(12) + devectorize((precalc[12] + ROL2(h[12], 13) + ROL16(h[15]) - ROL2(h[6], 7)) ^ cmsg[3]));
		q[29] = vectorize(add2 + CONST_EXP3d(13) + devectorize((precalc[13] + ROL2(h[13], 14) + ROL2(h[0], 1) - ROL8(h[7])) ^ cmsg[4]));

		add1 -= devectorize(q[12] - q[26]);
		add2 -= devectorize(q[13] - q[27]);

		XH64 = xor3x(XH64, q[28], q[29]);

		q[30] = vectorize(add1 + CONST_EXP3d(14) + devectorize((precalc[14] + ROL2(h[14], 15) + ROL2(h[1], 2) - ROL2(h[8], 9)) ^ cmsg[5]));
		q[31] = vectorize(add2 + CONST_EXP3d(15) + devectorize((precalc[15] + ROL16(h[15]) + ROL2(h[2], 3) - ROL2(h[9], 10)) ^ cmsg[6]));

		XH64 = xor3x(XH64, q[30], q[31]);

		//	msg[0] = devectorize((SHL2(XH64, 5) ^ SHR2(q[16], 5) ^ h[0]) + (XL64 ^ q[24] ^ q[0]));
		//	msg[1] = devectorize((SHR2(XH64, 7) ^ SHL8(q[17]) ^ h[1]) + (XL64 ^ q[25] ^ q[1]));
		//	msg[2] = devectorize((SHR2(XH64, 5) ^ SHL2(q[18], 5) ^ h[2]) + (XL64 ^ q[26] ^ q[2]));
		//	msg[3] = devectorize((SHR2(XH64, 1) ^ SHL2(q[19], 5) ^ h[3]) + (XL64 ^ q[27] ^ q[3]));
		//	msg[4] = devectorize((SHR2(XH64, 3) ^ q[20] ^ h[4]) + (XL64 ^ q[28] ^ q[4]));
		//	msg[5] = devectorize((SHL2(XH64, 6) ^ SHR2(q[21], 6) ^ h[5]) + (XL64 ^ q[29] ^ q[5]));
		//	msg[6] = devectorize((SHR2(XH64, 4) ^ SHL2(q[22], 6) ^ h[6]) + (XL64 ^ q[30] ^ q[6]));
		msg[7] = devectorize((SHR2(XH64, 11) ^ SHL2(q[23], 2) ^ h[7]) + (XL64 ^ q[31] ^ q[7]));
		//	msg[8] = devectorize((XH64 ^ q[24] ^ h[8]) + (SHL8(XL64) ^ q[23] ^ q[8]) + ROTL64(msg[4], 9));

		//	msg[9] = devectorize((XH64 ^ q[25] ^ h[9]) + (SHR2(XL64, 6) ^ q[16] ^ q[9]) + ROTL64(msg[5], 10));
		//	msg[10] = devectorize((XH64 ^ q[26] ^ h[10]) + (SHL2(XL64, 6) ^ q[17] ^ q[10]) + ROTL64(msg[6], 11));
		msg[11] = devectorize((XH64 ^ q[27] ^ h[11]) + (SHL2(XL64, 4) ^ q[18] ^ q[11]) + ROTL64(msg[7], 12));

		//#if __CUDA_ARCH__ > 500
		//		* (uint2x4*)&inpHash[0] = *(uint2x4*)&msg[8];
		//#endif

		//	msg[12] = devectorize((XH64 ^ q[28] ^ h[12]) + (SHR2(XL64, 3) ^ q[19] ^ q[12]) + ROTL64(msg[0], 13));
		//	msg[13] = devectorize((XH64 ^ q[29] ^ h[13]) + (SHR2(XL64, 4) ^ q[20] ^ q[13]) + ROTL64(msg[1], 14));
		//	msg[14] = devectorize((XH64 ^ q[30] ^ h[14]) + (SHR2(XL64, 7) ^ q[21] ^ q[14]) + ROTL64(msg[2], 15));
		//	msg[15] = devectorize((XH64 ^ q[31] ^ h[15]) + (SHR2(XL64, 2) ^ q[22] ^ q[15]) + ROTL64(msg[3], 16));

		if (msg[11] <= target)
		{
			uint32_t tmp = atomicExch(&resNonce[0], thread);
			if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}
		/*#if __CUDA_ARCH__ > 500
		* (uint2x4*)&inpHash[4] = *(uint2x4*)&msg[12];
		#else
		*(uint2x4*)&inpHash[0] = *(uint2x4*)&msg[8];
		*(uint2x4*)&inpHash[4] = *(uint2x4*)&msg[12];
		#endif
		*/
	}
}



__host__
void quark_bmw512_cpu_init(int thr_id, uint32_t threads)
{

}



__host__ void quark_bmw512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	quark_bmw512_gpu_hash_64 << <grid, block >> >(threads, (uint64_t*)d_hash, d_nonceVector);
}


__constant__ uint64_t c_PaddedMessage80[16]; // padded message (80 bytes + padding)


#undef SHL
#undef SHR
#undef CONST_EXP2

#define SHR(x, n) SHR2(x, n)
#define SHL(x, n) SHL2(x, n)
#define ROL(x, n) ROL2(x, n)


#define CONST_EXP2(i) \
	q[i+0] + ROL(q[i+1], 5)  + q[i+2] + ROL(q[i+3], 11) + \
	q[i+4] + ROL(q[i+5], 27) + q[i+6] + SWAPUINT2(q[i+7]) + \
	q[i+8] + ROL(q[i+9], 37) + q[i+10] + ROL(q[i+11], 43) + \
	q[i+12] + ROL(q[i+13], 53) + (SHR(q[i+14],1) ^ q[i+14]) + (SHR(q[i+15],2) ^ q[i+15])

__device__
void Compression512(uint2 *msg, uint2 *hash)
{
	// Compression ref. implementation
	uint2 q[32];
	uint2 tmp;

	tmp = (msg[5] ^ hash[5]) - (msg[7] ^ hash[7]) + (msg[10] ^ hash[10]) + (msg[13] ^ hash[13]) + (msg[14] ^ hash[14]);
	q[0] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp, 4) ^ ROL(tmp, 37)) + hash[1];
	tmp = (msg[6] ^ hash[6]) - (msg[8] ^ hash[8]) + (msg[11] ^ hash[11]) + (msg[14] ^ hash[14]) - (msg[15] ^ hash[15]);
	q[1] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROL(tmp, 13) ^ ROL(tmp, 43)) + hash[2];
	tmp = (msg[0] ^ hash[0]) + (msg[7] ^ hash[7]) + (msg[9] ^ hash[9]) - (msg[12] ^ hash[12]) + (msg[15] ^ hash[15]);
	q[2] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROL(tmp, 19) ^ ROL(tmp, 53)) + hash[3];
	tmp = (msg[0] ^ hash[0]) - (msg[1] ^ hash[1]) + (msg[8] ^ hash[8]) - (msg[10] ^ hash[10]) + (msg[13] ^ hash[13]);
	q[3] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROL(tmp, 28) ^ ROL(tmp, 59)) + hash[4];
	tmp = (msg[1] ^ hash[1]) + (msg[2] ^ hash[2]) + (msg[9] ^ hash[9]) - (msg[11] ^ hash[11]) - (msg[14] ^ hash[14]);
	q[4] = (SHR(tmp, 1) ^ tmp) + hash[5];
	tmp = (msg[3] ^ hash[3]) - (msg[2] ^ hash[2]) + (msg[10] ^ hash[10]) - (msg[12] ^ hash[12]) + (msg[15] ^ hash[15]);
	q[5] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp, 4) ^ ROL(tmp, 37)) + hash[6];
	tmp = (msg[4] ^ hash[4]) - (msg[0] ^ hash[0]) - (msg[3] ^ hash[3]) - (msg[11] ^ hash[11]) + (msg[13] ^ hash[13]);
	q[6] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROL(tmp, 13) ^ ROL(tmp, 43)) + hash[7];
	tmp = (msg[1] ^ hash[1]) - (msg[4] ^ hash[4]) - (msg[5] ^ hash[5]) - (msg[12] ^ hash[12]) - (msg[14] ^ hash[14]);
	q[7] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROL(tmp, 19) ^ ROL(tmp, 53)) + hash[8];
	tmp = (msg[2] ^ hash[2]) - (msg[5] ^ hash[5]) - (msg[6] ^ hash[6]) + (msg[13] ^ hash[13]) - (msg[15] ^ hash[15]);
	q[8] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROL(tmp, 28) ^ ROL(tmp, 59)) + hash[9];
	tmp = (msg[0] ^ hash[0]) - (msg[3] ^ hash[3]) + (msg[6] ^ hash[6]) - (msg[7] ^ hash[7]) + (msg[14] ^ hash[14]);
	q[9] = (SHR(tmp, 1) ^ tmp) + hash[10];
	tmp = (msg[8] ^ hash[8]) - (msg[1] ^ hash[1]) - (msg[4] ^ hash[4]) - (msg[7] ^ hash[7]) + (msg[15] ^ hash[15]);
	q[10] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp, 4) ^ ROL(tmp, 37)) + hash[11];
	tmp = (msg[8] ^ hash[8]) - (msg[0] ^ hash[0]) - (msg[2] ^ hash[2]) - (msg[5] ^ hash[5]) + (msg[9] ^ hash[9]);
	q[11] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROL(tmp, 13) ^ ROL(tmp, 43)) + hash[12];
	tmp = (msg[1] ^ hash[1]) + (msg[3] ^ hash[3]) - (msg[6] ^ hash[6]) - (msg[9] ^ hash[9]) + (msg[10] ^ hash[10]);
	q[12] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROL(tmp, 19) ^ ROL(tmp, 53)) + hash[13];
	tmp = (msg[2] ^ hash[2]) + (msg[4] ^ hash[4]) + (msg[7] ^ hash[7]) + (msg[10] ^ hash[10]) + (msg[11] ^ hash[11]);
	q[13] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROL(tmp, 28) ^ ROL(tmp, 59)) + hash[14];
	tmp = (msg[3] ^ hash[3]) - (msg[5] ^ hash[5]) + (msg[8] ^ hash[8]) - (msg[11] ^ hash[11]) - (msg[12] ^ hash[12]);
	q[14] = (SHR(tmp, 1) ^ tmp) + hash[15];
	tmp = (msg[12] ^ hash[12]) - (msg[4] ^ hash[4]) - (msg[6] ^ hash[6]) - (msg[9] ^ hash[9]) + (msg[13] ^ hash[13]);
	q[15] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp, 4) ^ ROL(tmp, 37)) + hash[0];

	q[0 + 16] =
		(SHR(q[0], 1) ^ SHL(q[0], 2) ^ ROL(q[0], 13) ^ ROL(q[0], 43)) +
		(SHR(q[0 + 1], 2) ^ SHL(q[0 + 1], 1) ^ ROL(q[0 + 1], 19) ^ ROL(q[0 + 1], 53)) +
		(SHR(q[0 + 2], 2) ^ SHL(q[0 + 2], 2) ^ ROL(q[0 + 2], 28) ^ ROL(q[0 + 2], 59)) +
		(SHR(q[0 + 3], 1) ^ SHL(q[0 + 3], 3) ^ ROL(q[0 + 3], 4) ^ ROL(q[0 + 3], 37)) +
		(SHR(q[0 + 4], 1) ^ SHL(q[0 + 4], 2) ^ ROL(q[0 + 4], 13) ^ ROL(q[0 + 4], 43)) +
		(SHR(q[0 + 5], 2) ^ SHL(q[0 + 5], 1) ^ ROL(q[0 + 5], 19) ^ ROL(q[0 + 5], 53)) +
		(SHR(q[0 + 6], 2) ^ SHL(q[0 + 6], 2) ^ ROL(q[0 + 6], 28) ^ ROL(q[0 + 6], 59)) +
		(SHR(q[0 + 7], 1) ^ SHL(q[0 + 7], 3) ^ ROL(q[0 + 7], 4) ^ ROL(q[0 + 7], 37)) +
		(SHR(q[0 + 8], 1) ^ SHL(q[0 + 8], 2) ^ ROL(q[0 + 8], 13) ^ ROL(q[0 + 8], 43)) +
		(SHR(q[0 + 9], 2) ^ SHL(q[0 + 9], 1) ^ ROL(q[0 + 9], 19) ^ ROL(q[0 + 9], 53)) +
		(SHR(q[0 + 10], 2) ^ SHL(q[0 + 10], 2) ^ ROL(q[0 + 10], 28) ^ ROL(q[0 + 10], 59)) +
		(SHR(q[0 + 11], 1) ^ SHL(q[0 + 11], 3) ^ ROL(q[0 + 11], 4) ^ ROL(q[0 + 11], 37)) +
		(SHR(q[0 + 12], 1) ^ SHL(q[0 + 12], 2) ^ ROL(q[0 + 12], 13) ^ ROL(q[0 + 12], 43)) +
		(SHR(q[0 + 13], 2) ^ SHL(q[0 + 13], 1) ^ ROL(q[0 + 13], 19) ^ ROL(q[0 + 13], 53)) +
		(SHR(q[0 + 14], 2) ^ SHL(q[0 + 14], 2) ^ ROL(q[0 + 14], 28) ^ ROL(q[0 + 14], 59)) +
		(SHR(q[0 + 15], 1) ^ SHL(q[0 + 15], 3) ^ ROL(q[0 + 15], 4) ^ ROL(q[0 + 15], 37)) +
		((make_uint2(0x55555550ul, 0x55555555) + ROL(msg[0], 0 + 1) +
		ROL(msg[0 + 3], 0 + 4) - ROL(msg[0 + 10], 0 + 11)) ^ hash[0 + 7]);

	q[1 + 16] =
		(SHR(q[1], 1) ^ SHL(q[1], 2) ^ ROL(q[1], 13) ^ ROL(q[1], 43)) +
		(SHR(q[1 + 1], 2) ^ SHL(q[1 + 1], 1) ^ ROL(q[1 + 1], 19) ^ ROL(q[1 + 1], 53)) +
		(SHR(q[1 + 2], 2) ^ SHL(q[1 + 2], 2) ^ ROL(q[1 + 2], 28) ^ ROL(q[1 + 2], 59)) +
		(SHR(q[1 + 3], 1) ^ SHL(q[1 + 3], 3) ^ ROL(q[1 + 3], 4) ^ ROL(q[1 + 3], 37)) +
		(SHR(q[1 + 4], 1) ^ SHL(q[1 + 4], 2) ^ ROL(q[1 + 4], 13) ^ ROL(q[1 + 4], 43)) +
		(SHR(q[1 + 5], 2) ^ SHL(q[1 + 5], 1) ^ ROL(q[1 + 5], 19) ^ ROL(q[1 + 5], 53)) +
		(SHR(q[1 + 6], 2) ^ SHL(q[1 + 6], 2) ^ ROL(q[1 + 6], 28) ^ ROL(q[1 + 6], 59)) +
		(SHR(q[1 + 7], 1) ^ SHL(q[1 + 7], 3) ^ ROL(q[1 + 7], 4) ^ ROL(q[1 + 7], 37)) +
		(SHR(q[1 + 8], 1) ^ SHL(q[1 + 8], 2) ^ ROL(q[1 + 8], 13) ^ ROL(q[1 + 8], 43)) +
		(SHR(q[1 + 9], 2) ^ SHL(q[1 + 9], 1) ^ ROL(q[1 + 9], 19) ^ ROL(q[1 + 9], 53)) +
		(SHR(q[1 + 10], 2) ^ SHL(q[1 + 10], 2) ^ ROL(q[1 + 10], 28) ^ ROL(q[1 + 10], 59)) +
		(SHR(q[1 + 11], 1) ^ SHL(q[1 + 11], 3) ^ ROL(q[1 + 11], 4) ^ ROL(q[1 + 11], 37)) +
		(SHR(q[1 + 12], 1) ^ SHL(q[1 + 12], 2) ^ ROL(q[1 + 12], 13) ^ ROL(q[1 + 12], 43)) +
		(SHR(q[1 + 13], 2) ^ SHL(q[1 + 13], 1) ^ ROL(q[1 + 13], 19) ^ ROL(q[1 + 13], 53)) +
		(SHR(q[1 + 14], 2) ^ SHL(q[1 + 14], 2) ^ ROL(q[1 + 14], 28) ^ ROL(q[1 + 14], 59)) +
		(SHR(q[1 + 15], 1) ^ SHL(q[1 + 15], 3) ^ ROL(q[1 + 15], 4) ^ ROL(q[1 + 15], 37)) +
		((make_uint2(0xAAAAAAA5, 0x5AAAAAAA) + ROL(msg[1], 1 + 1) +
		ROL(msg[1 + 3], 1 + 4) - ROL(msg[1 + 10], 1 + 11)) ^ hash[1 + 7]);

	q[2 + 16] = CONST_EXP2(2) +
		((make_uint2(0xFFFFFFFA, 0x5FFFFFFF) + ROL(msg[2], 2 + 1) +
		ROL(msg[2 + 3], 2 + 4) - ROL(msg[2 + 10], 2 + 11)) ^ hash[2 + 7]);
	q[3 + 16] = CONST_EXP2(3) +
		((make_uint2(0x5555554F, 0x65555555) + ROL(msg[3], 3 + 1) +
		ROL(msg[3 + 3], 3 + 4) - ROL(msg[3 + 10], 3 + 11)) ^ hash[3 + 7]);
	q[4 + 16] = CONST_EXP2(4) +
		((make_uint2(0xAAAAAAA4, 0x6AAAAAAA) + ROL(msg[4], 4 + 1) +
		ROL(msg[4 + 3], 4 + 4) - ROL(msg[4 + 10], 4 + 11)) ^ hash[4 + 7]);
	q[5 + 16] = CONST_EXP2(5) +
		((make_uint2(0xFFFFFFF9, 0x6FFFFFFF) + ROL(msg[5], 5 + 1) +
		ROL(msg[5 + 3], 5 + 4) - ROL(msg[5 + 10], 5 + 11)) ^ hash[5 + 7]);
	q[6 + 16] = CONST_EXP2(6) +
		((make_uint2(0x5555554E, 0x75555555) + ROL(msg[6], 6 + 1) +
		ROL(msg[6 + 3], 6 + 4) - ROL(msg[6 - 6], (6 - 6) + 1)) ^ hash[6 + 7]);
	q[7 + 16] = CONST_EXP2(7) +
		((make_uint2(0xAAAAAAA3, 0x7AAAAAAA) + ROL(msg[7], 7 + 1) +
		ROL(msg[7 + 3], 7 + 4) - ROL(msg[7 - 6], (7 - 6) + 1)) ^ hash[7 + 7]);
	q[8 + 16] = CONST_EXP2(8) +
		((make_uint2(0xFFFFFFF8, 0x7FFFFFFF) + ROL(msg[8], 8 + 1) +
		ROL(msg[8 + 3], 8 + 4) - ROL(msg[8 - 6], (8 - 6) + 1)) ^ hash[8 + 7]);
	q[9 + 16] = CONST_EXP2(9) +
		((make_uint2(0x5555554D, 0x85555555) + ROL(msg[9], 9 + 1) +
		ROL(msg[9 + 3], 9 + 4) - ROL(msg[9 - 6], (9 - 6) + 1)) ^ hash[9 - 9]);
	q[10 + 16] = CONST_EXP2(10) +
		((make_uint2(0xAAAAAAA2, 0x8AAAAAAA) + ROL(msg[10], 10 + 1) +
		ROL(msg[10 + 3], 10 + 4) - ROL(msg[10 - 6], (10 - 6) + 1)) ^ hash[10 - 9]);
	q[11 + 16] = CONST_EXP2(11) +
		((make_uint2(0xFFFFFFF7, 0x8FFFFFFF) + ROL(msg[11], 11 + 1) +
		ROL(msg[11 + 3], 11 + 4) - ROL(msg[11 - 6], (11 - 6) + 1)) ^ hash[11 - 9]);
	q[12 + 16] = CONST_EXP2(12) +
		((make_uint2(0x5555554C, 0x95555555) + ROL(msg[12], 12 + 1) +
		ROL(msg[12 + 3], 12 + 4) - ROL(msg[12 - 6], (12 - 6) + 1)) ^ hash[12 - 9]);
	q[13 + 16] = CONST_EXP2(13) +
		((make_uint2(0xAAAAAAA1, 0x9AAAAAAA) + ROL(msg[13], 13 + 1) +
		ROL(msg[13 - 13], (13 - 13) + 1) - ROL(msg[13 - 6], (13 - 6) + 1)) ^ hash[13 - 9]);
	q[14 + 16] = CONST_EXP2(14) +
		((make_uint2(0xFFFFFFF6, 0x9FFFFFFF) + ROL(msg[14], 14 + 1) +
		ROL(msg[14 - 13], (14 - 13) + 1) - ROL(msg[14 - 6], (14 - 6) + 1)) ^ hash[14 - 9]);
	q[15 + 16] = CONST_EXP2(15) +
		((make_uint2(0x5555554B, 0xA5555555) + ROL(msg[15], 15 + 1) +
		ROL(msg[15 - 13], (15 - 13) + 1) - ROL(msg[15 - 6], (15 - 6) + 1)) ^ hash[15 - 9]);

	uint2 XL64 = q[16] ^ q[17] ^ q[18] ^ q[19] ^ q[20] ^ q[21] ^ q[22] ^ q[23];
	uint2 XH64 = XL64^q[24] ^ q[25] ^ q[26] ^ q[27] ^ q[28] ^ q[29] ^ q[30] ^ q[31];

	hash[0] = (SHL(XH64, 5) ^ SHR(q[16], 5) ^ msg[0]) + (XL64 ^ q[24] ^ q[0]);
	hash[1] = (SHR(XH64, 7) ^ SHL(q[17], 8) ^ msg[1]) + (XL64 ^ q[25] ^ q[1]);
	hash[2] = (SHR(XH64, 5) ^ SHL(q[18], 5) ^ msg[2]) + (XL64 ^ q[26] ^ q[2]);
	hash[3] = (SHR(XH64, 1) ^ SHL(q[19], 5) ^ msg[3]) + (XL64 ^ q[27] ^ q[3]);
	hash[4] = (SHR(XH64, 3) ^ q[20] ^ msg[4]) + (XL64 ^ q[28] ^ q[4]);
	hash[5] = (SHL(XH64, 6) ^ SHR(q[21], 6) ^ msg[5]) + (XL64 ^ q[29] ^ q[5]);
	hash[6] = (SHR(XH64, 4) ^ SHL(q[22], 6) ^ msg[6]) + (XL64 ^ q[30] ^ q[6]);
	hash[7] = (SHR(XH64, 11) ^ SHL(q[23], 2) ^ msg[7]) + (XL64 ^ q[31] ^ q[7]);

	hash[8] = ROL(hash[4], 9) + (XH64 ^ q[24] ^ msg[8]) + (SHL(XL64, 8) ^ q[23] ^ q[8]);
	hash[9] = ROL(hash[5], 10) + (XH64 ^ q[25] ^ msg[9]) + (SHR(XL64, 6) ^ q[16] ^ q[9]);
	hash[10] = ROL(hash[6], 11) + (XH64 ^ q[26] ^ msg[10]) + (SHL(XL64, 6) ^ q[17] ^ q[10]);
	hash[11] = ROL(hash[7], 12) + (XH64 ^ q[27] ^ msg[11]) + (SHL(XL64, 4) ^ q[18] ^ q[11]);
	hash[12] = ROL(hash[0], 13) + (XH64 ^ q[28] ^ msg[12]) + (SHR(XL64, 3) ^ q[19] ^ q[12]);
	hash[13] = ROL(hash[1], 14) + (XH64 ^ q[29] ^ msg[13]) + (SHR(XL64, 4) ^ q[20] ^ q[13]);
	hash[14] = ROL(hash[2], 15) + (XH64 ^ q[30] ^ msg[14]) + (SHR(XL64, 7) ^ q[21] ^ q[14]);
	hash[15] = ROL(hash[3], 16) + (XH64 ^ q[31] ^ msg[15]) + (SHR(XL64, 2) ^ q[22] ^ q[15]);
}



__global__ __launch_bounds__(256, 2)
void quark_bmw512_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = startNounce + thread;

		// Init
		uint2 h[16] = {
			{ 0x84858687UL, 0x80818283UL },
			{ 0x8C8D8E8FUL, 0x88898A8BUL },
			{ 0x94959697UL, 0x90919293UL },
			{ 0x9C9D9E9FUL, 0x98999A9BUL },
			{ 0xA4A5A6A7UL, 0xA0A1A2A3UL },
			{ 0xACADAEAFUL, 0xA8A9AAABUL },
			{ 0xB4B5B6B7UL, 0xB0B1B2B3UL },
			{ 0xBCBDBEBFUL, 0xB8B9BABBUL },
			{ 0xC4C5C6C7UL, 0xC0C1C2C3UL },
			{ 0xCCCDCECFUL, 0xC8C9CACBUL },
			{ 0xD4D5D6D7UL, 0xD0D1D2D3UL },
			{ 0xDCDDDEDFUL, 0xD8D9DADBUL },
			{ 0xE4E5E6E7UL, 0xE0E1E2E3UL },
			{ 0xECEDEEEFUL, 0xE8E9EAEBUL },
			{ 0xF4F5F6F7UL, 0xF0F1F2F3UL },
			{ 0xFCFDFEFFUL, 0xF8F9FAFBUL }
		};
		// Nachricht kopieren (Achtung, die Nachricht hat 64 Byte,
		// BMW arbeitet mit 128 Byte!!!
		uint2 message[16];
#pragma unroll 16
		for (int i = 0; i<16; i++)
			message[i] = vectorize(c_PaddedMessage80[i]);

		// die Nounce durch die thread-spezifische ersetzen
		message[9].y = cuda_swab32(nounce);	//REPLACE_HIDWORD(message[9], cuda_swab32(nounce));

		// Compression 1
		Compression512(message, h);
#pragma unroll 16
		for (int i = 0; i<16; i++)
			message[i] = make_uint2(0xaaaaaaa0 + i, 0xaaaaaaaa);


		Compression512(h, message);

		// fertig
		uint64_t *outpHash = &g_hash[thread * 8];

#pragma unroll 8
		for (int i = 0; i<8; i++)
			outpHash[i] = devectorize(message[i + 8]);
	}
}


__constant__ uint64_t BMW512_IV[] = {
	(0x8081828384858687), (0x88898A8B8C8D8E8F),
	(0x9091929394959697), (0x98999A9B9C9D9E9F),
	(0xA0A1A2A3A4A5A6A7), (0xA8A9AAABACADAEAF),
	(0xB0B1B2B3B4B5B6B7), (0xB8B9BABBBCBDBEBF),
	(0xC0C1C2C3C4C5C6C7), (0xC8C9CACBCCCDCECF),
	(0xD0D1D2D3D4D5D6D7), (0xD8D9DADBDCDDDEDF),
	(0xE0E1E2E3E4E5E6E7), (0xE8E9EAEBECEDEEEF),
	(0xF0F1F2F3F4F5F6F7), (0xF8F9FAFBFCFDFEFF)
};



__constant__ uint64_t BMW512_FINAL[16] =
{
	0xAAAAAAAAAAAAAAA0UL, 0xAAAAAAAAAAAAAAA1UL, 0xAAAAAAAAAAAAAAA2UL, 0xAAAAAAAAAAAAAAA3UL,
	0xAAAAAAAAAAAAAAA4UL, 0xAAAAAAAAAAAAAAA5UL, 0xAAAAAAAAAAAAAAA6UL, 0xAAAAAAAAAAAAAAA7UL,
	0xAAAAAAAAAAAAAAA8UL, 0xAAAAAAAAAAAAAAA9UL, 0xAAAAAAAAAAAAAAAAUL, 0xAAAAAAAAAAAAAAABUL,
	0xAAAAAAAAAAAAAAACUL, 0xAAAAAAAAAAAAAAADUL, 0xAAAAAAAAAAAAAAAEUL, 0xAAAAAAAAAAAAAAAFUL
};


__host__
void quark_bmw512_cpu_setBlock_80(void *pdata)
{
	unsigned char PaddedMessage[128];
	memcpy(PaddedMessage, pdata, 80);
	memset(PaddedMessage + 80, 0, 48);
	uint64_t *message = (uint64_t*)PaddedMessage;
	message[10] = SPH_C64(0x80);
	message[15] = SPH_C64(640);
	cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, 16 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}



__host__
void quark_bmw512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = 256;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	quark_bmw512_gpu_hash_80 << <grid, block >> >(threads, startNounce, (uint64_t*)d_hash);
}

__host__ void quark_bmw512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash, uint32_t *resNonce, const uint64_t target)
{
	const uint32_t threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	quark_bmw512_gpu_hash_64_final << <grid, block >> >(threads, (uint64_t*)d_hash, d_nonceVector, resNonce, target);
}
