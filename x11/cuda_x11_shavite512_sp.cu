/*
	Based on Tanguy Pruvot's repo
	Provos Alexis - 2016
	optimized by sp - 2018/2019
*/
#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"

#define INTENSIVE_GMF
#include "cuda_x11_aes_sp.cuh"
__constant__ uint32_t c_PaddedMessage80[20]; // padded message (80 bytes + padding)

__device__ __forceinline__ void aes_round_s(const uint32_t sharedMemory[256][32], const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t x3, const uint32_t k0, uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3)
{
	const uint32_t index = threadIdx.x & 0x1f;

	y0 = sharedMemory[__byte_perm(x0, 0, 0x4440)][index];
	y3 = ROL8(sharedMemory[__byte_perm(x0, 0, 0x4441)][index]);
	y2 = ROL16(sharedMemory[__byte_perm(x0, 0, 0x4442)][index]);
	y1 = ROR8(sharedMemory[__byte_perm(x0, 0, 0x4443)][index]);

	y1 ^= sharedMemory[__byte_perm(x1, 0, 0x4440)][index];
	y0 ^= ROL8(sharedMemory[__byte_perm(x1, 0, 0x4441)][index]);
	y3 ^= ROL16(sharedMemory[__byte_perm(x1, 0, 0x4442)][index]);
	y2 ^= ROR8(sharedMemory[__byte_perm(x1, 0, 0x4443)][index]);

	y0 ^= k0;

	y2 ^= sharedMemory[__byte_perm(x2, 0, 0x4440)][index];
	y1 ^= ROL8(sharedMemory[__byte_perm(x2, 0, 0x4441)][index]);
	y0 ^= ROL16(sharedMemory[__byte_perm(x2, 0, 0x4442)][index]);
	y3 ^= ROR8(sharedMemory[__byte_perm(x2, 0, 0x4443)][index]);

	y3 ^= sharedMemory[__byte_perm(x3, 0, 0x4440)][index];
	y2 ^= ROL8(sharedMemory[__byte_perm(x3, 0, 0x4441)][index]);
	y1 ^= ROL16(sharedMemory[__byte_perm(x3, 0, 0x4442)][index]);
	y0 ^= ROR8(sharedMemory[__byte_perm(x3, 0, 0x4443)][index]);
}

__device__ __forceinline__ void aes_round_s(const uint32_t sharedMemory[256][32], const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t x3, uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3)
{
	const uint32_t index = threadIdx.x & 0x1f;

	y0 = sharedMemory[__byte_perm(x0, 0, 0x4440)][index];
	y3 = ROL8(sharedMemory[__byte_perm(x0, 0, 0x4441)][index]);
	y2 = ROL16(sharedMemory[__byte_perm(x0, 0, 0x4442)][index]);
	y1 = ROR8(sharedMemory[__byte_perm(x0, 0, 0x4443)][index]);

	y1 ^= sharedMemory[__byte_perm(x1, 0, 0x4440)][index];
	y0 ^= ROL8(sharedMemory[__byte_perm(x1, 0, 0x4441)][index]);
	y3 ^= ROL16(sharedMemory[__byte_perm(x1, 0, 0x4442)][index]);
	y2 ^= ROR8(sharedMemory[__byte_perm(x1, 0, 0x4443)][index]);

	y2 ^= sharedMemory[__byte_perm(x2, 0, 0x4440)][index];
	y1 ^= ROL8(sharedMemory[__byte_perm(x2, 0, 0x4441)][index]);
	y0 ^= ROL16(sharedMemory[__byte_perm(x2, 0, 0x4442)][index]);
	y3 ^= ROR8(sharedMemory[__byte_perm(x2, 0, 0x4443)][index]);

	y3 ^= sharedMemory[__byte_perm(x3, 0, 0x4440)][index];
	y2 ^= ROL8(sharedMemory[__byte_perm(x3, 0, 0x4441)][index]);
	y1 ^= ROL16(sharedMemory[__byte_perm(x3, 0, 0x4442)][index]);
	y0 ^= ROR8(sharedMemory[__byte_perm(x3, 0, 0x4443)][index]);
}


__device__ __forceinline__ void AES_ROUND_NOKEY_s(const uint32_t sharedMemory[256][32], uint4* x){

	uint32_t y0, y1, y2, y3;
	aes_round_s(sharedMemory, x->x, x->y, x->z, x->w, y0, y1, y2, y3);

	x->x = y0;
	x->y = y1;
	x->z = y2;
	x->w = y3;
}

__device__ __forceinline__ void KEY_EXPAND_ELT_s(const uint32_t sharedMemory[256][32], uint32_t *k)
{

	uint32_t y0, y1, y2, y3;
	aes_round_s(sharedMemory, k[0], k[1], k[2], k[3], y0, y1, y2, y3);

	k[0] = y1;
	k[1] = y2;
	k[2] = y3;
	k[3] = y0;
}


__device__ __forceinline__
void aes_gpu_init256_s(uint32_t sharedMemory[256][32])
{
	uint32_t temp = d_AES0[threadIdx.x];

	sharedMemory[threadIdx.x][0] = temp;
	sharedMemory[threadIdx.x][1] = temp;
	sharedMemory[threadIdx.x][2] = temp;
	sharedMemory[threadIdx.x][3] = temp;
	sharedMemory[threadIdx.x][4] = temp;
	sharedMemory[threadIdx.x][5] = temp;
	sharedMemory[threadIdx.x][6] = temp;
	sharedMemory[threadIdx.x][7] = temp;
	sharedMemory[threadIdx.x][8] = temp;
	sharedMemory[threadIdx.x][9] = temp;
	sharedMemory[threadIdx.x][10] = temp;
	sharedMemory[threadIdx.x][11] = temp;
	sharedMemory[threadIdx.x][12] = temp;
	sharedMemory[threadIdx.x][13] = temp;
	sharedMemory[threadIdx.x][14] = temp;
	sharedMemory[threadIdx.x][15] = temp;
	sharedMemory[threadIdx.x][16] = temp;
	sharedMemory[threadIdx.x][17] = temp;
	sharedMemory[threadIdx.x][18] = temp;
	sharedMemory[threadIdx.x][19] = temp;
	sharedMemory[threadIdx.x][20] = temp;
	sharedMemory[threadIdx.x][21] = temp;
	sharedMemory[threadIdx.x][22] = temp;
	sharedMemory[threadIdx.x][23] = temp;
	sharedMemory[threadIdx.x][24] = temp;
	sharedMemory[threadIdx.x][25] = temp;
	sharedMemory[threadIdx.x][26] = temp;
	sharedMemory[threadIdx.x][27] = temp;
	sharedMemory[threadIdx.x][28] = temp;
	sharedMemory[threadIdx.x][29] = temp;
	sharedMemory[threadIdx.x][30] = temp;
	sharedMemory[threadIdx.x][31] = temp;
}


__device__ __forceinline__ void round_3_7_11_s(const uint32_t sharedMemory[256][32], uint32_t* r, uint4 *p, uint4 &x){
	KEY_EXPAND_ELT_s(sharedMemory, &r[ 0]);
	*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
	x = p[ 2] ^ *(uint4*)&r[ 0];
	KEY_EXPAND_ELT_s(sharedMemory, &r[ 4]);
	r[4] ^= r[0];
	r[5] ^= r[1];
	r[6] ^= r[2];
	r[7] ^= r[3];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x.x ^= r[4];
	x.y ^= r[5];
	x.z ^= r[6];
	x.w ^= r[7];
	KEY_EXPAND_ELT_s(sharedMemory, &r[ 8]);
	r[8] ^= r[4];
	r[9] ^= r[5];
	r[10]^= r[6];
	r[11]^= r[7];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x.x ^= r[8];
	x.y ^= r[9];
	x.z ^= r[10];
	x.w ^= r[11];
	KEY_EXPAND_ELT_s(sharedMemory, &r[12]);
	r[12] ^= r[8];
	r[13] ^= r[9];
	r[14]^= r[10];
	r[15]^= r[11];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x.x ^= r[12];
	x.y ^= r[13];
	x.z ^= r[14];
	x.w ^= r[15];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[ 1].x ^= x.x;
	p[ 1].y ^= x.y;
	p[ 1].z ^= x.z;
	p[ 1].w ^= x.w;
	KEY_EXPAND_ELT_s(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	x = p[ 0] ^ *(uint4*)&r[16];
	KEY_EXPAND_ELT_s(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x ^= *(uint4*)&r[20];
	KEY_EXPAND_ELT_s(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x ^= *(uint4*)&r[24];
	KEY_EXPAND_ELT_s(sharedMemory,&r[28]);
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[ 3] ^= x;
}

__device__ __forceinline__
void round_4_8_12_s(const uint32_t sharedMemory[256][32], uint32_t* r, uint4 *p, uint4 &x){
	*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
	x = p[ 1] ^ *(uint4*)&r[ 0];
	AES_ROUND_NOKEY_s(sharedMemory, &x);

	r[ 4] ^= r[29];	r[ 5] ^= r[30];
	r[ 6] ^= r[31];	r[ 7] ^= r[ 0];

	x ^= *(uint4*)&r[ 4];
	*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x ^= *(uint4*)&r[ 8];
	*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[ 0] ^= x;
	*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
	x = p[ 3] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	*(uint4*)&r[20] ^= *(uint4*)&r[13];
	x ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	*(uint4*)&r[24] ^= *(uint4*)&r[17];
	x ^= *(uint4*)&r[24];
	*(uint4*)&r[28] ^= *(uint4*)&r[21];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[ 2] ^= x;
}

// GPU Hash
__global__ __launch_bounds__(448, 2) /* 64 registers with 128,8 - 72 regs with 128,7 */
void x11_shavite512_gpu_hash_64_sp(const uint32_t threads, uint64_t *g_hash)
{
	__shared__ uint32_t sharedMemory[256][32];

	if(threadIdx.x<256) aes_gpu_init256_s(sharedMemory);

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint4 p[ 4];
	uint4 x;
	uint32_t r[32];

	// kopiere init-state
	const uint32_t state[16] = {
		0x72FCCDD8, 0x79CA4727, 0x128A077B, 0x40D55AEC,	0xD1901A06, 0x430AE307, 0xB29F5CD1, 0xDF07FBFC,
		0x8E45D73D, 0x681AB538, 0xBDE86578, 0xDD577E47,	0xE275EADE, 0x502D9FCD, 0xB9357178, 0x022A4B9A
	};
	if (thread < threads)
	{
		uint64_t *Hash = &g_hash[thread<<3];

		// fülle die Nachricht mit 64-byte (vorheriger Hash)
		*(uint2x4*)&r[ 0] = __ldg4((uint2x4*)&Hash[ 0]);
		__syncthreads();
		
		*(uint2x4*)&p[ 0] = *(uint2x4*)&state[ 0];
		*(uint2x4*)&p[ 2] = *(uint2x4*)&state[ 8];
		r[16] = 0x80; r[17] = 0; r[18] = 0; r[19] = 0;
		r[20] = 0; r[21] = 0; r[22] = 0; r[23] = 0;
		r[24] = 0; r[25] = 0; r[26] = 0; r[27] = 0x02000000;
		r[28] = 0; r[29] = 0; r[30] = 0; r[31] = 0x02000000;
		/* round 0 */
		x = p[ 1] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);


		*(uint2x4*)&r[8] = __ldg4((uint2x4*)&Hash[4]);
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 0] ^= x;

		x = p[ 3];
		x.x ^= 0x80;
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		x.w ^= 0x02000000;
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		x.w ^= 0x02000000;
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 2]^= x;


		// 1
		KEY_EXPAND_ELT_s(sharedMemory, &r[ 0]);
		*(uint4*)&r[ 0]^=*(uint4*)&r[28];
		r[ 0] ^= 0x200;
		r[3] = ~r[3];

		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT_s(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 1] ^= x;
		*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
		x = p[ 3] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);

		r[ 4] ^= r[29]; r[ 5] ^= r[30];
		r[ 6] ^= r[31]; r[ 7] ^= r[ 0];

		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
		x = p[ 1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
//		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		r[24] ^= r[17];
		r[25] ^= r[18];
		r[26] ^= r[19];
		r[27] ^= r[20];


		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_s(sharedMemory, &x);

		p[ 0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11_s(sharedMemory,r,p,x);
		
		
		/* round 4, 8, 12 */
		round_4_8_12_s(sharedMemory,r,p,x);

		// 2
		KEY_EXPAND_ELT_s(sharedMemory,&r[ 0]);
		*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		r[ 7] ^= (~0x200);
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT_s(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory,&r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 1] ^= x;
	
		*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
		x = p[ 3] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		r[ 4] ^= r[29];
		r[ 5] ^= r[30];
		r[ 6] ^= r[31];
		r[ 7] ^= r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
		x = p[ 1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11_s(sharedMemory,r,p,x);

		/* round 4, 8, 12 */
		round_4_8_12_s(sharedMemory,r,p,x);

		// 3
		KEY_EXPAND_ELT_s(sharedMemory,&r[ 0]);
		*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT_s(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x^=*(uint4*)&r[20];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[24]);
		*(uint4*)&r[24]^=*(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory,&r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		r[30] ^= 0x200;
		r[31] = ~r[31];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 1] ^= x;

		*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
		x = p[ 3] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		r[ 4] ^= r[29];
		r[ 5] ^= r[30];
		r[ 6] ^= r[31];
		r[ 7] ^= r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
		x = p[ 1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11_s(sharedMemory,r,p,x);
		
		/* round 4, 8, 12 */
		round_4_8_12_s(sharedMemory,r,p,x);

		/* round 13 */
		KEY_EXPAND_ELT_s(sharedMemory,&r[ 0]);
		*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT_s(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		r[25] ^= 0x200;
		r[27] = ~r[27];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory,&r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 1] ^= x;

		*(uint2x4*)&Hash[ 0] = *(uint2x4*)&state[ 0] ^ *(uint2x4*)&p[ 2];
		*(uint2x4*)&Hash[ 4] = *(uint2x4*)&state[ 8] ^ *(uint2x4*)&p[ 0];
	}
}

__device__ __forceinline__ void round_3_7_11(const uint32_t sharedMemory[8*1024], uint32_t* r, uint4 *p, uint4 &x){
	KEY_EXPAND_ELT_32(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	x = p[2] ^ *(uint4*)&r[0];
	KEY_EXPAND_ELT_32(sharedMemory, &r[4]);
	r[4] ^= r[0];
	r[5] ^= r[1];
	r[6] ^= r[2];
	r[7] ^= r[3];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	x.x ^= r[4];
	x.y ^= r[5];
	x.z ^= r[6];
	x.w ^= r[7];
	KEY_EXPAND_ELT_32(sharedMemory, &r[8]);
	r[8] ^= r[4];
	r[9] ^= r[5];
	r[10] ^= r[6];
	r[11] ^= r[7];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	x.x ^= r[8];
	x.y ^= r[9];
	x.z ^= r[10];
	x.w ^= r[11];
	KEY_EXPAND_ELT_32(sharedMemory, &r[12]);
	r[12] ^= r[8];
	r[13] ^= r[9];
	r[14] ^= r[10];
	r[15] ^= r[11];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	x.x ^= r[12];
	x.y ^= r[13];
	x.z ^= r[14];
	x.w ^= r[15];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	p[1].x ^= x.x;
	p[1].y ^= x.y;
	p[1].z ^= x.z;
	p[1].w ^= x.w;
	KEY_EXPAND_ELT_32(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	x = p[0] ^ *(uint4*)&r[16];
	KEY_EXPAND_ELT_32(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	x ^= *(uint4*)&r[20];
	KEY_EXPAND_ELT_32(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	x ^= *(uint4*)&r[24];
	KEY_EXPAND_ELT_32(sharedMemory, &r[28]);
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	p[3] ^= x;
}

__device__ __forceinline__ void round_4_8_12(const uint32_t sharedMemory[8*1024], uint32_t* r, uint4 *p, uint4 &x){
	*(uint4*)&r[0] ^= *(uint4*)&r[25];
	x = p[1] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY_32(sharedMemory, &x);

	r[4] ^= r[29];	r[5] ^= r[30];
	r[6] ^= r[31];	r[7] ^= r[0];

	x ^= *(uint4*)&r[4];
	*(uint4*)&r[8] ^= *(uint4*)&r[1];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	x ^= *(uint4*)&r[8];
	*(uint4*)&r[12] ^= *(uint4*)&r[5];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	p[0] ^= x;
	*(uint4*)&r[16] ^= *(uint4*)&r[9];
	x = p[3] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	*(uint4*)&r[20] ^= *(uint4*)&r[13];
	x ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	*(uint4*)&r[24] ^= *(uint4*)&r[17];
	x ^= *(uint4*)&r[24];
	*(uint4*)&r[28] ^= *(uint4*)&r[21];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	p[2] ^= x;
}


__global__ __launch_bounds__(384, 2)
void x11_shavite512_gpu_hash_64_sp_final(const uint32_t threads, uint64_t *g_hash, uint32_t* resNonce, const uint64_t target)
{
	__shared__ uint32_t sharedMemory[8 * 1024];

	if (threadIdx.x<256) aes_gpu_init256_32(sharedMemory);

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint4 p[4];
	uint4 x;
	uint32_t r[32];

	// kopiere init-state
	const uint32_t state[16] = {
		0x72FCCDD8, 0x79CA4727, 0x128A077B, 0x40D55AEC, 0xD1901A06, 0x430AE307, 0xB29F5CD1, 0xDF07FBFC,
		0x8E45D73D, 0x681AB538, 0xBDE86578, 0xDD577E47, 0xE275EADE, 0x502D9FCD, 0xB9357178, 0x022A4B9A
	};
	if (thread < threads)
	{
		uint2 *Hash = (uint2 *)&g_hash[thread << 3];

		// fülle die Nachricht mit 64-byte (vorheriger Hash)
		*(uint2x4*)&r[0] = __ldg4((uint2x4*)&Hash[0]);
		*(uint2x4*)&r[8] = __ldg4((uint2x4*)&Hash[4]);
		__syncthreads();

		*(uint2x4*)&p[0] = *(uint2x4*)&state[0];
		*(uint2x4*)&p[2] = *(uint2x4*)&state[8];
		r[16] = 0x80; r[17] = 0; r[18] = 0; r[19] = 0;
		r[20] = 0; r[21] = 0; r[22] = 0; r[23] = 0;
		r[24] = 0; r[25] = 0; r[26] = 0; r[27] = 0x02000000;
		r[28] = 0; r[29] = 0; r[30] = 0; r[31] = 0x02000000;
		/* round 0 */
		x = p[1] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[0] ^= x;
		x = p[3];
		x.x ^= 0x80;

		AES_ROUND_NOKEY_32(sharedMemory, &x);

		AES_ROUND_NOKEY_32(sharedMemory, &x);

		x.w ^= 0x02000000;
		AES_ROUND_NOKEY_32(sharedMemory, &x);

		x.w ^= 0x02000000;
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[2] ^= x;
		// 1
		KEY_EXPAND_ELT_32(sharedMemory, &r[0]);
		*(uint4*)&r[0] ^= *(uint4*)&r[28];
		r[0] ^= 0x200;
		r[3] = ~r[3];
		x = p[0] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[4]);
		*(uint4*)&r[4] ^= *(uint4*)&r[0];
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[8]);
		*(uint4*)&r[8] ^= *(uint4*)&r[4];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[3] ^= x;
		KEY_EXPAND_ELT_32(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[1] ^= x;
		*(uint4*)&r[0] ^= *(uint4*)&r[25];
		x = p[3] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_32(sharedMemory, &x);

		r[4] ^= r[29]; r[5] ^= r[30];
		r[6] ^= r[31]; r[7] ^= r[0];

		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[8] ^= *(uint4*)&r[1];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[9];
		x = p[1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_32(sharedMemory, &x);

		p[0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory, r, p, x);


		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory, r, p, x);

		// 2
		KEY_EXPAND_ELT_32(sharedMemory, &r[0]);
		*(uint4*)&r[0] ^= *(uint4*)&r[28];
		x = p[0] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[4]);
		*(uint4*)&r[4] ^= *(uint4*)&r[0];
		r[7] ^= (~0x200);
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[8]);
		*(uint4*)&r[8] ^= *(uint4*)&r[4];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[3] ^= x;
		KEY_EXPAND_ELT_32(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[1] ^= x;

		*(uint4*)&r[0] ^= *(uint4*)&r[25];
		x = p[3] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		r[4] ^= r[29];
		r[5] ^= r[30];
		r[6] ^= r[31];
		r[7] ^= r[0];
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[8] ^= *(uint4*)&r[1];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[9];
		x = p[1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory, r, p, x);

		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory, r, p, x);

		// 3
		KEY_EXPAND_ELT_32(sharedMemory, &r[0]);
		*(uint4*)&r[0] ^= *(uint4*)&r[28];
		x = p[0] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[4]);
		*(uint4*)&r[4] ^= *(uint4*)&r[0];
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[8]);
		*(uint4*)&r[8] ^= *(uint4*)&r[4];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[3] ^= x;
		KEY_EXPAND_ELT_32(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		r[30] ^= 0x200;
		r[31] = ~r[31];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[1] ^= x;

		*(uint4*)&r[0] ^= *(uint4*)&r[25];
		x = p[3] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		r[4] ^= r[29];
		r[5] ^= r[30];
		r[6] ^= r[31];
		r[7] ^= r[0];
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[8] ^= *(uint4*)&r[1];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[9];
		x = p[1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory, r, p, x);

		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory, r, p, x);

		/* round 13 */
		KEY_EXPAND_ELT_32(sharedMemory, &r[0]);
		*(uint4*)&r[0] ^= *(uint4*)&r[28];
		x = p[0] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[4]);
		*(uint4*)&r[4] ^= *(uint4*)&r[0];
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[8]);
		*(uint4*)&r[8] ^= *(uint4*)&r[4];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[3] ^= x;
		/*		KEY_EXPAND_ELT_32(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		r[25] ^= 0x200;
		r[27] ^= 0xFFFFFFFF;
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[1] ^= x;
		*/
		//Hash[3] = 
		uint64_t test = (((uint64_t *)state)[3] ^ devectorize(make_uint2(p[3].z, p[3].w)));

		if (test <= target)
		{
			const uint32_t tmp = atomicExch(&resNonce[0], thread);
			if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}
	}
}


__host__
void x11_shavite512_cpu_hash_64_sp(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	dim3 grid((threads + 256 - 1) / 256);
	dim3 block(256);

	x11_shavite512_gpu_hash_64_sp<<<grid, block>>>(threads, (uint64_t*)d_hash);
}

__host__
void x11_shavite512_cpu_hash_64_sp_final(int thr_id, uint32_t threads, uint32_t *d_hash, const uint64_t target, uint32_t* resNonce)
{
	dim3 grid((threads + 384 - 1) / 384);
	dim3 block(384);

	x11_shavite512_gpu_hash_64_sp_final << <grid, block >> >(threads, (uint64_t*)d_hash,resNonce,target);
}