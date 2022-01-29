/*
	Based on Tanguy Pruvot's repo
	Provos Alexis - 2016
	Optimized for nvidia pascal/volta by sp (2018/2019)
*/

#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"

#define AESx(x) (x ##UL) /* SPH_C32(x) */

__constant__  static uint32_t __align__(16) d_AES0[256] = {
	0xA56363C6, 0x847C7CF8, 0x997777EE, 0x8D7B7BF6, 0x0DF2F2FF, 0xBD6B6BD6, 0xB16F6FDE, 0x54C5C591, 0x50303060, 0x03010102, 0xA96767CE, 0x7D2B2B56, 0x19FEFEE7, 0x62D7D7B5, 0xE6ABAB4D, 0x9A7676EC,
	0x45CACA8F, 0x9D82821F, 0x40C9C989, 0x877D7DFA, 0x15FAFAEF, 0xEB5959B2, 0xC947478E, 0x0BF0F0FB, 0xECADAD41, 0x67D4D4B3, 0xFDA2A25F, 0xEAAFAF45, 0xBF9C9C23, 0xF7A4A453, 0x967272E4, 0x5BC0C09B,
	0xC2B7B775, 0x1CFDFDE1, 0xAE93933D, 0x6A26264C, 0x5A36366C, 0x413F3F7E, 0x02F7F7F5, 0x4FCCCC83, 0x5C343468, 0xF4A5A551, 0x34E5E5D1, 0x08F1F1F9, 0x937171E2, 0x73D8D8AB, 0x53313162, 0x3F15152A,
	0x0C040408, 0x52C7C795, 0x65232346, 0x5EC3C39D, 0x28181830, 0xA1969637, 0x0F05050A, 0xB59A9A2F, 0x0907070E, 0x36121224, 0x9B80801B, 0x3DE2E2DF, 0x26EBEBCD, 0x6927274E, 0xCDB2B27F, 0x9F7575EA,
	0x1B090912, 0x9E83831D, 0x742C2C58, 0x2E1A1A34, 0x2D1B1B36, 0xB26E6EDC, 0xEE5A5AB4, 0xFBA0A05B, 0xF65252A4, 0x4D3B3B76, 0x61D6D6B7, 0xCEB3B37D, 0x7B292952, 0x3EE3E3DD, 0x712F2F5E, 0x97848413,
	0xF55353A6, 0x68D1D1B9, 0x00000000, 0x2CEDEDC1, 0x60202040, 0x1FFCFCE3, 0xC8B1B179, 0xED5B5BB6, 0xBE6A6AD4, 0x46CBCB8D, 0xD9BEBE67, 0x4B393972, 0xDE4A4A94, 0xD44C4C98, 0xE85858B0, 0x4ACFCF85,
	0x6BD0D0BB, 0x2AEFEFC5, 0xE5AAAA4F, 0x16FBFBED, 0xC5434386, 0xD74D4D9A, 0x55333366, 0x94858511, 0xCF45458A, 0x10F9F9E9, 0x06020204, 0x817F7FFE, 0xF05050A0, 0x443C3C78, 0xBA9F9F25, 0xE3A8A84B,
	0xF35151A2, 0xFEA3A35D, 0xC0404080, 0x8A8F8F05, 0xAD92923F, 0xBC9D9D21, 0x48383870, 0x04F5F5F1, 0xDFBCBC63, 0xC1B6B677, 0x75DADAAF, 0x63212142, 0x30101020, 0x1AFFFFE5, 0x0EF3F3FD, 0x6DD2D2BF,
	0x4CCDCD81, 0x140C0C18, 0x35131326, 0x2FECECC3, 0xE15F5FBE, 0xA2979735, 0xCC444488, 0x3917172E, 0x57C4C493, 0xF2A7A755, 0x827E7EFC, 0x473D3D7A, 0xAC6464C8, 0xE75D5DBA, 0x2B191932, 0x957373E6,
	0xA06060C0, 0x98818119, 0xD14F4F9E, 0x7FDCDCA3, 0x66222244, 0x7E2A2A54, 0xAB90903B, 0x8388880B, 0xCA46468C, 0x29EEEEC7, 0xD3B8B86B, 0x3C141428, 0x79DEDEA7, 0xE25E5EBC, 0x1D0B0B16, 0x76DBDBAD,
	0x3BE0E0DB, 0x56323264, 0x4E3A3A74, 0x1E0A0A14, 0xDB494992, 0x0A06060C, 0x6C242448, 0xE45C5CB8, 0x5DC2C29F, 0x6ED3D3BD, 0xEFACAC43, 0xA66262C4, 0xA8919139, 0xA4959531, 0x37E4E4D3, 0x8B7979F2,
	0x32E7E7D5, 0x43C8C88B, 0x5937376E, 0xB76D6DDA, 0x8C8D8D01, 0x64D5D5B1, 0xD24E4E9C, 0xE0A9A949, 0xB46C6CD8, 0xFA5656AC, 0x07F4F4F3, 0x25EAEACF, 0xAF6565CA, 0x8E7A7AF4, 0xE9AEAE47, 0x18080810,
	0xD5BABA6F, 0x887878F0, 0x6F25254A, 0x722E2E5C, 0x241C1C38, 0xF1A6A657, 0xC7B4B473, 0x51C6C697, 0x23E8E8CB, 0x7CDDDDA1, 0x9C7474E8, 0x211F1F3E, 0xDD4B4B96, 0xDCBDBD61, 0x868B8B0D, 0x858A8A0F,
	0x907070E0, 0x423E3E7C, 0xC4B5B571, 0xAA6666CC, 0xD8484890, 0x05030306, 0x01F6F6F7, 0x120E0E1C, 0xA36161C2, 0x5F35356A, 0xF95757AE, 0xD0B9B969, 0x91868617, 0x58C1C199, 0x271D1D3A, 0xB99E9E27,
	0x38E1E1D9, 0x13F8F8EB, 0xB398982B, 0x33111122, 0xBB6969D2, 0x70D9D9A9, 0x898E8E07, 0xA7949433, 0xB69B9B2D, 0x221E1E3C, 0x92878715, 0x20E9E9C9, 0x49CECE87, 0xFF5555AA, 0x78282850, 0x7ADFDFA5,
	0x8F8C8C03, 0xF8A1A159, 0x80898909, 0x170D0D1A, 0xDABFBF65, 0x31E6E6D7, 0xC6424284, 0xB86868D0, 0xC3414182, 0xB0999929, 0x772D2D5A, 0x110F0F1E, 0xCBB0B07B, 0xFC5454A8, 0xD6BBBB6D, 0x3A16162C
};



__device__ __forceinline__ void aes_round(const uint32_t sharedMemory[256][32], const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t x3, const uint32_t k0, uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3)
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

__device__ __forceinline__ void aes_round(const uint32_t sharedMemory[256][32], const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t x3, uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3)
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


__device__ __forceinline__ void AES_ROUND_NOKEY(const uint32_t sharedMemory[256][32], uint4* x){

	uint32_t y0, y1, y2, y3;
	aes_round(sharedMemory, x->x, x->y, x->z, x->w, y0, y1, y2, y3);

	x->x = y0;
	x->y = y1;
	x->z = y2;
	x->w = y3;
}

__device__ __forceinline__ void KEY_EXPAND_ELT(const uint32_t sharedMemory[256][32], uint32_t *k)
{

	uint32_t y0, y1, y2, y3;
	aes_round(sharedMemory, k[0], k[1], k[2], k[3], y0, y1, y2, y3);

	k[0] = y1;
	k[1] = y2;
	k[2] = y3;
	k[3] = y0;
}


__device__ __forceinline__
void aes_gpu_init256(uint32_t sharedMemory[256][32])
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

	/*	sharedMemory[(threadIdx.x << 1) + 0][0] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][1] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][2] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][3] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][4] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][5] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][6] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][7] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][8] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][9] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][10] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][11] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][12] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][13] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][14] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][15] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][16] = temp.x;

	sharedMemory[(threadIdx.x << 1) + 1][0] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][1] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][2] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][3] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][4] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][5] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][6] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][7] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][8] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][9] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][10] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][11] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][12] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][13] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][14] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][15] = temp.y;
	*/

	/*	sharedMemory[1][(threadIdx.x << 1) + 0] = ROL8(temp.x);
	sharedMemory[1][(threadIdx.x << 1) + 1] = ROL8(temp.y);
	sharedMemory[2][(threadIdx.x << 1) + 0] = ROL16(temp.x);
	sharedMemory[2][(threadIdx.x << 1) + 1] = ROL16(temp.y);
	sharedMemory[3][(threadIdx.x << 1) + 0] = ROR8(temp.x);
	sharedMemory[3][(threadIdx.x << 1) + 1] = ROR8(temp.y);
	*/
}

__device__ __forceinline__
void aes_gpu_init128(uint32_t sharedMemory[256][32])
{
	uint32_t temp = d_AES0[threadIdx.x<<1];
	uint32_t temp2 = d_AES0[(threadIdx.x << 1) +1];

	sharedMemory[threadIdx.x << 1][0] = temp;
	sharedMemory[threadIdx.x << 1][1] = temp;
	sharedMemory[threadIdx.x << 1][2] = temp;
	sharedMemory[threadIdx.x << 1][3] = temp;
	sharedMemory[threadIdx.x << 1][4] = temp;
	sharedMemory[threadIdx.x << 1][5] = temp;
	sharedMemory[threadIdx.x << 1][6] = temp;
	sharedMemory[threadIdx.x << 1][7] = temp;
	sharedMemory[threadIdx.x << 1][8] = temp;
	sharedMemory[threadIdx.x << 1][9] = temp;
	sharedMemory[threadIdx.x << 1][10] = temp;
	sharedMemory[threadIdx.x << 1][11] = temp;
	sharedMemory[threadIdx.x << 1][12] = temp;
	sharedMemory[threadIdx.x << 1][13] = temp;
	sharedMemory[threadIdx.x << 1][14] = temp;
	sharedMemory[threadIdx.x << 1][15] = temp;
	sharedMemory[threadIdx.x << 1][16] = temp;
	sharedMemory[threadIdx.x << 1][17] = temp;
	sharedMemory[threadIdx.x << 1][18] = temp;
	sharedMemory[threadIdx.x << 1][19] = temp;
	sharedMemory[threadIdx.x << 1][20] = temp;
	sharedMemory[threadIdx.x << 1][21] = temp;
	sharedMemory[threadIdx.x << 1][22] = temp;
	sharedMemory[threadIdx.x << 1][23] = temp;
	sharedMemory[threadIdx.x << 1][24] = temp;
	sharedMemory[threadIdx.x << 1][25] = temp;
	sharedMemory[threadIdx.x << 1][26] = temp;
	sharedMemory[threadIdx.x << 1][27] = temp;
	sharedMemory[threadIdx.x << 1][28] = temp;
	sharedMemory[threadIdx.x << 1][29] = temp;
	sharedMemory[threadIdx.x << 1][30] = temp;
	sharedMemory[threadIdx.x << 1][31] = temp;

	sharedMemory[(threadIdx.x << 1) + 1][0] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][1] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][2] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][3] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][4] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][5] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][6] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][7] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][8] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][9] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][10] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][11] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][12] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][13] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][14] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][15] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][16] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][17] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][18] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][19] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][20] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][21] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][22] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][23] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][24] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][25] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][26] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][27] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][28] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][29] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][30] = temp2;
	sharedMemory[(threadIdx.x << 1) + 1][31] = temp2;

	/*	sharedMemory[(threadIdx.x << 1) + 0][0] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][1] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][2] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][3] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][4] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][5] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][6] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][7] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][8] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][9] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][10] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][11] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][12] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][13] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][14] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][15] = temp.x;
	sharedMemory[(threadIdx.x << 1) + 0][16] = temp.x;

	sharedMemory[(threadIdx.x << 1) + 1][0] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][1] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][2] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][3] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][4] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][5] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][6] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][7] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][8] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][9] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][10] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][11] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][12] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][13] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][14] = temp.y;
	sharedMemory[(threadIdx.x << 1) + 1][15] = temp.y;
	*/
}


__device__ __forceinline__ void round_3_7_11(const uint32_t sharedMemory[256][32], uint32_t* r, uint4 *p, uint4 &x){
	KEY_EXPAND_ELT(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	x = p[2] ^ *(uint4*)&r[0];
	KEY_EXPAND_ELT(sharedMemory, &r[4]);
	r[4] ^= r[0];
	r[5] ^= r[1];
	r[6] ^= r[2];
	r[7] ^= r[3];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x.x ^= r[4];
	x.y ^= r[5];
	x.z ^= r[6];
	x.w ^= r[7];
	KEY_EXPAND_ELT(sharedMemory, &r[8]);
	r[8] ^= r[4];
	r[9] ^= r[5];
	r[10] ^= r[6];
	r[11] ^= r[7];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x.x ^= r[8];
	x.y ^= r[9];
	x.z ^= r[10];
	x.w ^= r[11];
	KEY_EXPAND_ELT(sharedMemory, &r[12]);
	r[12] ^= r[8];
	r[13] ^= r[9];
	r[14] ^= r[10];
	r[15] ^= r[11];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x.x ^= r[12];
	x.y ^= r[13];
	x.z ^= r[14];
	x.w ^= r[15];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[1].x ^= x.x;
	p[1].y ^= x.y;
	p[1].z ^= x.z;
	p[1].w ^= x.w;
	KEY_EXPAND_ELT(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	x = p[0] ^ *(uint4*)&r[16];
	KEY_EXPAND_ELT(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[20];
	KEY_EXPAND_ELT(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[24];
	KEY_EXPAND_ELT(sharedMemory, &r[28]);
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[3] ^= x;
}

__device__ __forceinline__
void round_4_8_12(const uint32_t sharedMemory[256][32], uint32_t* r, uint4 *p, uint4 &x){
	*(uint4*)&r[0] ^= *(uint4*)&r[25];
	x = p[1] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &x);

	r[4] ^= r[29];	r[5] ^= r[30];
	r[6] ^= r[31];	r[7] ^= r[0];

	x ^= *(uint4*)&r[4];
	*(uint4*)&r[8] ^= *(uint4*)&r[1];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[8];
	*(uint4*)&r[12] ^= *(uint4*)&r[5];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[0] ^= x;
	*(uint4*)&r[16] ^= *(uint4*)&r[9];
	x = p[3] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[20] ^= *(uint4*)&r[13];
	x ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[24] ^= *(uint4*)&r[17];
	x ^= *(uint4*)&r[24];
	*(uint4*)&r[28] ^= *(uint4*)&r[21];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[2] ^= x;
}

//--END OF SHAVITE MACROS------------------------------------


#define TPB 1024

__device__ __forceinline__
static void rrounds(uint32_t *x)
{
	#pragma unroll 1
	for (int r = 0; r < 16; r++) 
	{
		/* "add x_0jklm into x_1jklmn modulo 2^32 rotate x_0jklm upwards by 7 bits" */
		x[16] = x[16] + x[ 0]; x[ 0] = ROTL32(x[ 0], 7);x[17] = x[17] + x[ 1];x[ 1] = ROTL32(x[ 1], 7);
		x[18] = x[18] + x[ 2]; x[ 2] = ROTL32(x[ 2], 7);x[19] = x[19] + x[ 3];x[ 3] = ROTL32(x[ 3], 7);
		x[20] = x[20] + x[ 4]; x[ 4] = ROTL32(x[ 4], 7);x[21] = x[21] + x[ 5];x[ 5] = ROTL32(x[ 5], 7);
		x[22] = x[22] + x[ 6]; x[ 6] = ROTL32(x[ 6], 7);x[23] = x[23] + x[ 7];x[ 7] = ROTL32(x[ 7], 7);
		x[24] = x[24] + x[ 8]; x[ 8] = ROTL32(x[ 8], 7);x[25] = x[25] + x[ 9];x[ 9] = ROTL32(x[ 9], 7);
		x[26] = x[26] + x[10]; x[10] = ROTL32(x[10], 7);x[27] = x[27] + x[11];x[11] = ROTL32(x[11], 7);
		x[28] = x[28] + x[12]; x[12] = ROTL32(x[12], 7);x[29] = x[29] + x[13];x[13] = ROTL32(x[13], 7);
		x[30] = x[30] + x[14]; x[14] = ROTL32(x[14], 7);x[31] = x[31] + x[15];x[15] = ROTL32(x[15], 7);
		/* "swap x_00klm with x_01klm" */
		xchg(x[0], x[8]); x[0] ^= x[16]; x[8] ^= x[24]; xchg(x[1], x[9]); x[1] ^= x[17]; x[9] ^= x[25];
		xchg(x[2], x[10]); x[2] ^= x[18]; x[10] ^= x[26]; xchg(x[3], x[11]); x[3] ^= x[19]; x[11] ^= x[27];
		xchg(x[4], x[12]); x[4] ^= x[20]; x[12] ^= x[28]; xchg(x[5], x[13]); x[5] ^= x[21]; x[13] ^= x[29];
		xchg(x[6], x[14]); x[6] ^= x[22]; x[14] ^= x[30]; xchg(x[7], x[15]); x[7] ^= x[23]; x[15] ^= x[31];
		/* "swap x_1jk0m with x_1jk1m" */
		xchg(x[16], x[18]); xchg(x[17], x[19]); xchg(x[20], x[22]); xchg(x[21], x[23]); xchg(x[24], x[26]); xchg(x[25], x[27]); xchg(x[28], x[30]); xchg(x[29], x[31]);
		/* "add x_0jklm into x_1jklm modulo 2^32 rotate x_0jklm upwards by 11 bits" */
		x[16] = x[16] + x[ 0]; x[ 0] = ROTL32(x[ 0],11);x[17] = x[17] + x[ 1];x[ 1] = ROTL32(x[ 1],11);
		x[18] = x[18] + x[ 2]; x[ 2] = ROTL32(x[ 2],11);x[19] = x[19] + x[ 3];x[ 3] = ROTL32(x[ 3],11);
		x[20] = x[20] + x[ 4]; x[ 4] = ROTL32(x[ 4],11);x[21] = x[21] + x[ 5];x[ 5] = ROTL32(x[ 5],11);
		x[22] = x[22] + x[ 6]; x[ 6] = ROTL32(x[ 6],11);x[23] = x[23] + x[ 7];x[ 7] = ROTL32(x[ 7],11);
		x[24] = x[24] + x[ 8]; x[ 8] = ROTL32(x[ 8],11);x[25] = x[25] + x[ 9];x[ 9] = ROTL32(x[ 9],11);
		x[26] = x[26] + x[10]; x[10] = ROTL32(x[10],11);x[27] = x[27] + x[11];x[11] = ROTL32(x[11],11);
		x[28] = x[28] + x[12]; x[12] = ROTL32(x[12],11);x[29] = x[29] + x[13];x[13] = ROTL32(x[13],11);
		x[30] = x[30] + x[14]; x[14] = ROTL32(x[14],11);x[31] = x[31] + x[15];x[15] = ROTL32(x[15],11);
		/* "swap x_0j0lm with x_0j1lm" */
		xchg(x[0], x[4]); x[0] ^= x[16]; x[4] ^= x[20]; xchg(x[1], x[5]); x[1] ^= x[17]; x[5] ^= x[21];
		xchg(x[2], x[6]); x[2] ^= x[18]; x[6] ^= x[22]; xchg(x[3], x[7]); x[3] ^= x[19]; x[7] ^= x[23];
		xchg(x[8], x[12]); x[8] ^= x[24]; x[12] ^= x[28]; xchg(x[9], x[13]); x[9] ^= x[25]; x[13] ^= x[29];
		xchg(x[10], x[14]); x[10] ^= x[26]; x[14] ^= x[30]; xchg(x[11], x[15]); x[11] ^= x[27]; x[15] ^= x[31];
		/* "swap x_1jkl0 with x_1jkl1" */
		xchg(x[16], x[17]); xchg(x[18], x[19]); xchg(x[20], x[21]); xchg(x[22], x[23]); xchg(x[24], x[25]); xchg(x[26], x[27]); xchg(x[28], x[29]); xchg(x[30], x[31]);
	}
}

/***************************************************/
// GPU Hash Function
__global__
void x11_cubehash512_gpu_hash_64(uint32_t threads, uint64_t *g_hash){

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){

		uint32_t *Hash = (uint32_t*)&g_hash[8 * thread];

		uint32_t x[32] = {
			0x2AEA2A61, 0x50F494D4, 0x2D538B8B, 0x4167D83E,
			0x3FEE2313, 0xC701CF8C, 0xCC39968E, 0x50AC5695,
			0x4D42C787, 0xA647A8B3, 0x97CF0BEF, 0x825B4537,
			0xEEF864D2, 0xF22090C4, 0xD0E5CD33, 0xA23911AE,
			0xFCD398D9, 0x148FE485, 0x1B017BEF, 0xB6444532,
			0x6A536159, 0x2FF5781C, 0x91FA7934, 0x0DBADEA9,
			0xD65C8A2B, 0xA5A70E75, 0xB1C62456, 0xBC796576,
			0x1921C8F7, 0xE7989AF1, 0x7795D246, 0xD43E3B44
		};
	
		// erste Hälfte des Hashes (32 bytes)
		//Update32(x, (const BitSequence*)Hash);
		*(uint2x4*)&x[ 0] ^= __ldg4((uint2x4*)&Hash[0]);

		rrounds(x);

		// zweite Hälfte des Hashes (32 bytes)
	//        Update32(x, (const BitSequence*)(Hash+8));
		*(uint2x4*)&x[ 0] ^= __ldg4((uint2x4*)&Hash[8]);
		
		rrounds(x);

		// Padding Block
		x[ 0] ^= 0x80;
		rrounds(x);
	
	//	Final(x, (BitSequence*)Hash);
		x[31] ^= 1;

		/* "the state is then transformed invertibly through 10r identical rounds" */
		#pragma unroll 10
		for (int i = 0;i < 10;++i)
			rrounds(x);

		/* "output the first h/8 bytes of the state" */
		*(uint2x4*)&Hash[ 0] = *(uint2x4*)&x[ 0];
		*(uint2x4*)&Hash[ 8] = *(uint2x4*)&x[ 8];
	}
}

__global__
__launch_bounds__(448,2)
void x11_cubehashShavite512_gpu_hash_64(uint32_t threads, uint32_t *g_hash){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	__shared__ uint32_t sharedMemory[256][32];

	if (threadIdx.x<256) aes_gpu_init256(sharedMemory);

	if (thread < threads)
	{

	uint32_t *const hash = &g_hash[thread * 16U];

	//Cubehash

	uint32_t x[32] = {
		0x2AEA2A61, 0x50F494D4, 0x2D538B8B, 0x4167D83E, 0x3FEE2313, 0xC701CF8C, 0xCC39968E, 0x50AC5695,
		0x4D42C787, 0xA647A8B3, 0x97CF0BEF, 0x825B4537, 0xEEF864D2, 0xF22090C4, 0xD0E5CD33, 0xA23911AE,
		0xFCD398D9, 0x148FE485, 0x1B017BEF, 0xB6444532, 0x6A536159, 0x2FF5781C, 0x91FA7934, 0x0DBADEA9,
		0xD65C8A2B, 0xA5A70E75, 0xB1C62456, 0xBC796576, 0x1921C8F7, 0xE7989AF1, 0x7795D246, 0xD43E3B44
	};
	uint32_t Hash[16];
	*(uint2x4*)&Hash[0] = __ldg4((uint2x4*)&hash[0]);
	*(uint2x4*)&Hash[8] = __ldg4((uint2x4*)&hash[8]);

	*(uint2x4*)&x[0] ^= *(uint2x4*)&Hash[0];

	rrounds(x);

	*(uint2x4*)&x[0] ^= *(uint2x4*)&Hash[8];

	rrounds(x);
	x[0] ^= 0x80;

	rrounds(x);
	x[31] ^= 1;

//	#pragma unroll 10
	for (int i = 0; i < 9; ++i)
		rrounds(x);


	rrounds(x);

	uint4 y;
	uint32_t r[32];
	uint4 msg[4];
	// kopiere init-state
	uint4 p[4];
	const uint32_t state[16] = {
		0x72FCCDD8, 0x79CA4727, 0x128A077B, 0x40D55AEC, 0xD1901A06, 0x430AE307, 0xB29F5CD1, 0xDF07FBFC,
		0x8E45D73D, 0x681AB538, 0xBDE86578, 0xDD577E47, 0xE275EADE, 0x502D9FCD, 0xB9357178, 0x022A4B9A
	};
	*(uint2x4*)&p[0] = *(uint2x4*)&state[0];
	*(uint2x4*)&p[2] = *(uint2x4*)&state[8];

#pragma unroll 4
	for (int i = 0; i < 4; i++){
		*(uint4*)&msg[i] = *(uint4*)&x[i << 2];
		*(uint4*)&r[i << 2] = *(uint4*)&x[i << 2];
	}
	r[16] = 0x80; r[17] = 0; r[18] = 0; r[19] = 0;
	r[20] = 0; r[21] = 0; r[22] = 0; r[23] = 0;
	r[24] = 0; r[25] = 0; r[26] = 0; r[27] = 0x02000000;
	r[28] = 0; r[29] = 0; r[30] = 0; r[31] = 0x02000000;
	y = p[1] ^ msg[0];
	__syncthreads();

	AES_ROUND_NOKEY(sharedMemory, &y);
	y ^= msg[1];
	AES_ROUND_NOKEY(sharedMemory, &y);
	y ^= msg[2];
	AES_ROUND_NOKEY(sharedMemory, &y);
	y ^= msg[3];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[0] ^= y;
	y = p[3];
	y.x ^= 0x80;
	AES_ROUND_NOKEY(sharedMemory, &y);
	AES_ROUND_NOKEY(sharedMemory, &y);
	y.w ^= 0x02000000;
	AES_ROUND_NOKEY(sharedMemory, &y);
	y.w ^= 0x02000000;
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[2] ^= y;

	// 1
	KEY_EXPAND_ELT(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	r[0] ^= 0x200;
	r[3] ^= 0xFFFFFFFF;
	y = p[0] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[4]);
	*(uint4*)&r[4] ^= *(uint4*)&r[0];
	y ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[8]);
	*(uint4*)&r[8] ^= *(uint4*)&r[4];
	y ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[12]);
	*(uint4*)&r[12] ^= *(uint4*)&r[8];
	y ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[3] ^= y;
	KEY_EXPAND_ELT(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	y = p[2] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	y ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	y ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[28]);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	y ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[1] ^= y;
	*(uint4*)&r[0] ^= *(uint4*)&r[25];
	y = p[3] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &y);

	r[4] ^= r[29]; r[5] ^= r[30];
	r[6] ^= r[31]; r[7] ^= r[0];

	y ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[8] ^= *(uint4*)&r[1];
	y ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[12] ^= *(uint4*)&r[5];
	y ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[2] ^= y;
	*(uint4*)&r[16] ^= *(uint4*)&r[9];
	y = p[1] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[20] ^= *(uint4*)&r[13];
	y ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[24] ^= *(uint4*)&r[17];
	y ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[28] ^= *(uint4*)&r[21];
	y ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &y);

	p[0] ^= y;

	/* round 3, 7, 11 */
	round_3_7_11(sharedMemory, r, p, y);


	/* round 4, 8, 12 */
	round_4_8_12(sharedMemory, r, p, y);

	// 2
	KEY_EXPAND_ELT(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	y = p[0] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[4]);
	*(uint4*)&r[4] ^= *(uint4*)&r[0];
	r[7] ^= (~0x200);
	y ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[8]);
	*(uint4*)&r[8] ^= *(uint4*)&r[4];
	y ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[12]);
	*(uint4*)&r[12] ^= *(uint4*)&r[8];
	y ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[3] ^= y;
	KEY_EXPAND_ELT(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	y = p[2] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	y ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	y ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[28]);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	y ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[1] ^= y;

	*(uint4*)&r[0] ^= *(uint4*)&r[25];
	y = p[3] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &y);
	r[4] ^= r[29];
	r[5] ^= r[30];
	r[6] ^= r[31];
	r[7] ^= r[0];
	y ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[8] ^= *(uint4*)&r[1];
	y ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[12] ^= *(uint4*)&r[5];
	y ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[2] ^= y;
	*(uint4*)&r[16] ^= *(uint4*)&r[9];
	y = p[1] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[20] ^= *(uint4*)&r[13];
	y ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[24] ^= *(uint4*)&r[17];
	y ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[28] ^= *(uint4*)&r[21];
	y ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[0] ^= y;

	/* round 3, 7, 11 */
	round_3_7_11(sharedMemory, r, p, y);

	/* round 4, 8, 12 */
	round_4_8_12(sharedMemory, r, p, y);

	// 3
	KEY_EXPAND_ELT(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	y = p[0] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[4]);
	*(uint4*)&r[4] ^= *(uint4*)&r[0];
	y ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[8]);
	*(uint4*)&r[8] ^= *(uint4*)&r[4];
	y ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[12]);
	*(uint4*)&r[12] ^= *(uint4*)&r[8];
	y ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[3] ^= y;
	KEY_EXPAND_ELT(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	y = p[2] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	y ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	y ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[28]);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	r[30] ^= 0x200;
	r[31] ^= 0xFFFFFFFF;
	y ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[1] ^= y;

	*(uint4*)&r[0] ^= *(uint4*)&r[25];
	y = p[3] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &y);
	r[4] ^= r[29];
	r[5] ^= r[30];
	r[6] ^= r[31];
	r[7] ^= r[0];
	y ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[8] ^= *(uint4*)&r[1];
	y ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[12] ^= *(uint4*)&r[5];
	y ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[2] ^= y;
	*(uint4*)&r[16] ^= *(uint4*)&r[9];
	y = p[1] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[20] ^= *(uint4*)&r[13];
	y ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[24] ^= *(uint4*)&r[17];
	y ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[28] ^= *(uint4*)&r[21];
	y ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[0] ^= y;

	/* round 3, 7, 11 */
	round_3_7_11(sharedMemory, r, p, y);

	/* round 4, 8, 12 */
	round_4_8_12(sharedMemory, r, p, y);

	/* round 13 */
	KEY_EXPAND_ELT(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	y = p[0] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[4]);
	*(uint4*)&r[4] ^= *(uint4*)&r[0];
	y ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[8]);
	*(uint4*)&r[8] ^= *(uint4*)&r[4];
	y ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[12]);
	*(uint4*)&r[12] ^= *(uint4*)&r[8];
	y ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[3] ^= y;
	KEY_EXPAND_ELT(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	y = p[2] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	y ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	r[25] ^= 0x200;
	r[27] ^= 0xFFFFFFFF;
	y ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[28]);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	y ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[1] ^= y;

	*(uint2x4*)&hash[0] = *(uint2x4*)&state[0] ^ *(uint2x4*)&p[2];
	*(uint2x4*)&hash[8] = *(uint2x4*)&state[8] ^ *(uint2x4*)&p[0];
	}
}


__host__
void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash){

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + TPB-1)/TPB);
    dim3 block(TPB);

    x11_cubehash512_gpu_hash_64<<<grid, block>>>(threads, (uint64_t*)d_hash);
}

__host__
void x11_cubehash_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	dim3 grid((threads + 256 - 1) / 256);
	dim3 block(256);

	x11_cubehashShavite512_gpu_hash_64 << <grid, block >> > (threads, d_hash);
}
