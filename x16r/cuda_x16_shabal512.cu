/*
* Shabal-512 for X16R
* tpruvot 2018, based on alexis x14 and xevan kernlx code
* Optimized by sp june-2018 (31% faster)
*/

#include <cuda_helper_alexis.h>
#include <cuda_vectors_alexis.h>
//#include "x16r/cuda_vector_uint2x4.h"

typedef uint32_t sph_u32;

#define C32(x) (x)
#define T32(x) (x)

#define INPUT_BLOCK_ADD do { \
		B0 = T32(B0 + M0); \
		B1 = T32(B1 + M1); \
		B2 = T32(B2 + M2); \
		B3 = T32(B3 + M3); \
		B4 = T32(B4 + M4); \
		B5 = T32(B5 + M5); \
		B6 = T32(B6 + M6); \
		B7 = T32(B7 + M7); \
		B8 = T32(B8 + M8); \
		B9 = T32(B9 + M9); \
		BA = T32(BA + MA); \
		BB = T32(BB + MB); \
		BC = T32(BC + MC); \
		BD = T32(BD + MD); \
		BE = T32(BE + ME); \
		BF = T32(BF + MF); \
		} while (0)

#define INPUT_BLOCK_SUB do { \
		C0 = T32(C0 - M0); \
		C1 = T32(C1 - M1); \
		C2 = T32(C2 - M2); \
		C3 = T32(C3 - M3); \
		C4 = T32(C4 - M4); \
		C5 = T32(C5 - M5); \
		C6 = T32(C6 - M6); \
		C7 = T32(C7 - M7); \
		C8 = T32(C8 - M8); \
		C9 = T32(C9 - M9); \
		CA = T32(CA - MA); \
		CB = T32(CB - MB); \
		CC = T32(CC - MC); \
		CD = T32(CD - MD); \
		CE = T32(CE - ME); \
		CF = T32(CF - MF); \
		} while (0)

#define XOR_W   do { \
		A00 ^= Wlow; \
		A01 ^= Whigh; \
		} while (0)

#define SWAP(v1, v2) do { \
		sph_u32 tmp = (v1); \
		(v1) = (v2); \
		(v2) = tmp; \
		} while (0)

#define SWAP_BC do { \
		SWAP(B0, C0); \
		SWAP(B1, C1); \
		SWAP(B2, C2); \
		SWAP(B3, C3); \
		SWAP(B4, C4); \
		SWAP(B5, C5); \
		SWAP(B6, C6); \
		SWAP(B7, C7); \
		SWAP(B8, C8); \
		SWAP(B9, C9); \
		SWAP(BA, CA); \
		SWAP(BB, CB); \
		SWAP(BC, CC); \
		SWAP(BD, CD); \
		SWAP(BE, CE); \
		SWAP(BF, CF); \
		} while (0)

#define PERM_ELT(xa0, xa1, xb0, xb1, xb2, xb3, xc, xm) do { \
		t=((xa1 << 15) | (xa1 >> 17));\
		t=(xa0 ^  ((t<<2)+t) ^ xc);\
		xa0 = T32( t+t+t) ^ xb1 ^ (xb2 & ~xb3) ^ xm; \
		xb0 = T32(~(((xb0 << 1) | (xb0 >> 31)) ^ xa0)); \
		} while (0)
#define PERM_STEP_0 do { \
		PERM_ELT(A00, A0B, B0, BD, B9, B6, C8, M0); \
		PERM_ELT(A01, A00, B1, BE, BA, B7, C7, M1); \
		PERM_ELT(A02, A01, B2, BF, BB, B8, C6, M2); \
		PERM_ELT(A03, A02, B3, B0, BC, B9, C5, M3); \
		PERM_ELT(A04, A03, B4, B1, BD, BA, C4, M4); \
		PERM_ELT(A05, A04, B5, B2, BE, BB, C3, M5); \
		PERM_ELT(A06, A05, B6, B3, BF, BC, C2, M6); \
		PERM_ELT(A07, A06, B7, B4, B0, BD, C1, M7); \
		PERM_ELT(A08, A07, B8, B5, B1, BE, C0, M8); \
		PERM_ELT(A09, A08, B9, B6, B2, BF, CF, M9); \
		PERM_ELT(A0A, A09, BA, B7, B3, B0, CE, MA); \
		PERM_ELT(A0B, A0A, BB, B8, B4, B1, CD, MB); \
		PERM_ELT(A00, A0B, BC, B9, B5, B2, CC, MC); \
		PERM_ELT(A01, A00, BD, BA, B6, B3, CB, MD); \
		PERM_ELT(A02, A01, BE, BB, B7, B4, CA, ME); \
		PERM_ELT(A03, A02, BF, BC, B8, B5, C9, MF); \
		} while (0)

#define PERM_STEP_1 do { \
		PERM_ELT(A04, A03, B0, BD, B9, B6, C8, M0); \
		PERM_ELT(A05, A04, B1, BE, BA, B7, C7, M1); \
		PERM_ELT(A06, A05, B2, BF, BB, B8, C6, M2); \
		PERM_ELT(A07, A06, B3, B0, BC, B9, C5, M3); \
		PERM_ELT(A08, A07, B4, B1, BD, BA, C4, M4); \
		PERM_ELT(A09, A08, B5, B2, BE, BB, C3, M5); \
		PERM_ELT(A0A, A09, B6, B3, BF, BC, C2, M6); \
		PERM_ELT(A0B, A0A, B7, B4, B0, BD, C1, M7); \
		PERM_ELT(A00, A0B, B8, B5, B1, BE, C0, M8); \
		PERM_ELT(A01, A00, B9, B6, B2, BF, CF, M9); \
		PERM_ELT(A02, A01, BA, B7, B3, B0, CE, MA); \
		PERM_ELT(A03, A02, BB, B8, B4, B1, CD, MB); \
		PERM_ELT(A04, A03, BC, B9, B5, B2, CC, MC); \
		PERM_ELT(A05, A04, BD, BA, B6, B3, CB, MD); \
		PERM_ELT(A06, A05, BE, BB, B7, B4, CA, ME); \
		PERM_ELT(A07, A06, BF, BC, B8, B5, C9, MF); \
		} while (0)

#define PERM_STEP_2 do { \
		PERM_ELT(A08, A07, B0, BD, B9, B6, C8, M0); \
		PERM_ELT(A09, A08, B1, BE, BA, B7, C7, M1); \
		PERM_ELT(A0A, A09, B2, BF, BB, B8, C6, M2); \
		PERM_ELT(A0B, A0A, B3, B0, BC, B9, C5, M3); \
		PERM_ELT(A00, A0B, B4, B1, BD, BA, C4, M4); \
		PERM_ELT(A01, A00, B5, B2, BE, BB, C3, M5); \
		PERM_ELT(A02, A01, B6, B3, BF, BC, C2, M6); \
		PERM_ELT(A03, A02, B7, B4, B0, BD, C1, M7); \
		PERM_ELT(A04, A03, B8, B5, B1, BE, C0, M8); \
		PERM_ELT(A05, A04, B9, B6, B2, BF, CF, M9); \
		PERM_ELT(A06, A05, BA, B7, B3, B0, CE, MA); \
		PERM_ELT(A07, A06, BB, B8, B4, B1, CD, MB); \
		PERM_ELT(A08, A07, BC, B9, B5, B2, CC, MC); \
		PERM_ELT(A09, A08, BD, BA, B6, B3, CB, MD); \
		PERM_ELT(A0A, A09, BE, BB, B7, B4, CA, ME); \
		PERM_ELT(A0B, A0A, BF, BC, B8, B5, C9, MF); \
		} while (0)

#define APPLY_P do { \
		B0 = T32(B0 << 17) | (B0 >> 15); \
		B1 = T32(B1 << 17) | (B1 >> 15); \
		B2 = T32(B2 << 17) | (B2 >> 15); \
		B3 = T32(B3 << 17) | (B3 >> 15); \
		B4 = T32(B4 << 17) | (B4 >> 15); \
		B5 = T32(B5 << 17) | (B5 >> 15); \
		B6 = T32(B6 << 17) | (B6 >> 15); \
		B7 = T32(B7 << 17) | (B7 >> 15); \
		B8 = T32(B8 << 17) | (B8 >> 15); \
		B9 = T32(B9 << 17) | (B9 >> 15); \
		BA = T32(BA << 17) | (BA >> 15); \
		BB = T32(BB << 17) | (BB >> 15); \
		BC = T32(BC << 17) | (BC >> 15); \
		BD = T32(BD << 17) | (BD >> 15); \
		BE = T32(BE << 17) | (BE >> 15); \
		BF = T32(BF << 17) | (BF >> 15); \
		PERM_STEP_0; \
		PERM_STEP_1; \
		PERM_STEP_2; \
		A0B = T32(A0B + C6); \
		A0A = T32(A0A + C5); \
		A09 = T32(A09 + C4); \
		A08 = T32(A08 + C3); \
		A07 = T32(A07 + C2); \
		A06 = T32(A06 + C1); \
		A05 = T32(A05 + C0); \
		A04 = T32(A04 + CF); \
		A03 = T32(A03 + CE); \
		A02 = T32(A02 + CD); \
		A01 = T32(A01 + CC); \
		A00 = T32(A00 + CB); \
		A0B = T32(A0B + CA); \
		A0A = T32(A0A + C9); \
		A09 = T32(A09 + C8); \
		A08 = T32(A08 + C7); \
		A07 = T32(A07 + C6); \
		A06 = T32(A06 + C5); \
		A05 = T32(A05 + C4); \
		A04 = T32(A04 + C3); \
		A03 = T32(A03 + C2); \
		A02 = T32(A02 + C1); \
		A01 = T32(A01 + C0); \
		A00 = T32(A00 + CF); \
		A0B = T32(A0B + CE); \
		A0A = T32(A0A + CD); \
		A09 = T32(A09 + CC); \
		A08 = T32(A08 + CB); \
		A07 = T32(A07 + CA); \
		A06 = T32(A06 + C9); \
		A05 = T32(A05 + C8); \
		A04 = T32(A04 + C7); \
		A03 = T32(A03 + C6); \
		A02 = T32(A02 + C5); \
		A01 = T32(A01 + C4); \
		A00 = T32(A00 + C3); \
	} while (0)
//-----------cuda----------------

#define INPUT_BLOCK_ADD_CUDA do { \
		B0 = (B0 + M0); \
		B1 = (B1 + M1); \
		B2 = (B2 + M2); \
		B3 = (B3 + M3); \
		B4 = (B4 + M4); \
		B5 = (B5 + M5); \
		B6 = (B6 + M6); \
		B7 = (B7 + M7); \
		B8 = (B8 + M8); \
		B9 = (B9 + M9); \
		BA = (BA + MA); \
		BB = (BB + MB); \
		BC = (BC + MC); \
		BD = (BD + MD); \
		BE = (BE + ME); \
		BF = (BF + MF); \
				} while (0)

#define INPUT_BLOCK_SUB_CuDA do { \
		C0 = (C0 - M0); \
		C1 = (C1 - M1); \
		C2 = (C2 - M2); \
		C3 = (C3 - M3); \
		C4 = (C4 - M4); \
		C5 = (C5 - M5); \
		C6 = (C6 - M6); \
		C7 = (C7 - M7); \
		C8 = (C8 - M8); \
		C9 = (C9 - M9); \
		CA = (CA - MA); \
		CB = (CB - MB); \
		CC = (CC - MC); \
		CD = (CD - MD); \
		CE = (CE - ME); \
		CF = (CF - MF); \
				} while (0)

#define XOR_W_CUDA   do { \
		A00 ^= Wlow; \
		A01 ^= Whigh; \
				} while (0)

#define SWAP_CUDA(v1, v2) do { \
		sph_u32 tmp = (v1); \
		(v1) = (v2); \
		(v2) = tmp; \
				} while (0)

#define SWAP_BC_CUDA do { \
		SWAP(B0, C0); \
		SWAP(B1, C1); \
		SWAP(B2, C2); \
		SWAP(B3, C3); \
		SWAP(B4, C4); \
		SWAP(B5, C5); \
		SWAP(B6, C6); \
		SWAP(B7, C7); \
		SWAP(B8, C8); \
		SWAP(B9, C9); \
		SWAP(BA, CA); \
		SWAP(BB, CB); \
		SWAP(BC, CC); \
		SWAP(BD, CD); \
		SWAP(BE, CE); \
		SWAP(BF, CF); \
				} while (0)

__device__ __forceinline__ uint32_t Xxor3x(const uint32_t a, const uint32_t b, const uint32_t c)
{
	uint32_t result;
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
	asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result) : "r"(a), "r"(b), "r"(c)); //0x96 = 0xF0 ^ 0xCC ^ 0xAA
#else
	result = a^b^c;
#endif
	return result;
}

__device__ __forceinline__ void PERM_ELT_CUDA(uint32_t &xa0, const uint32_t xa1, uint32_t &xb0, const uint32_t xb1, const uint32_t xb2, const uint32_t xb3, const uint32_t xc, const uint32_t xm){

	uint32_t tmp;
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
	asm("lop3.b32 %0, %1, %2, %3, 0x9A;" : "=r"(tmp) : "r"(xb2), "r"(xb3), "r"(xm));  // 0x9A = (F0 &(~CC)) ^ (AA)
#else
	tmp = (xb2 & ~xb3) ^ xm;
#endif

	xa0 = Xxor3x( ( Xxor3x( xa0, xc, ( ROTL32( xa1, 15 ) * 5U ) ) * 3U ), xb1, tmp );  
	xb0 = Xxor3x( 0xFFFFFFFF, xa0, ROTL32( xb0, 1 ) );
}


#define PERM_STEP_0_CUDA do { \
		PERM_ELT_CUDA(A00, A0B, B0, BD, B9, B6, C8, M0); \
		PERM_ELT_CUDA(A01, A00, B1, BE, BA, B7, C7, M1); \
		PERM_ELT_CUDA(A02, A01, B2, BF, BB, B8, C6, M2); \
		PERM_ELT_CUDA(A03, A02, B3, B0, BC, B9, C5, M3); \
		PERM_ELT_CUDA(A04, A03, B4, B1, BD, BA, C4, M4); \
		PERM_ELT_CUDA(A05, A04, B5, B2, BE, BB, C3, M5); \
		PERM_ELT_CUDA(A06, A05, B6, B3, BF, BC, C2, M6); \
		PERM_ELT_CUDA(A07, A06, B7, B4, B0, BD, C1, M7); \
		PERM_ELT_CUDA(A08, A07, B8, B5, B1, BE, C0, M8); \
		PERM_ELT_CUDA(A09, A08, B9, B6, B2, BF, CF, M9); \
		PERM_ELT_CUDA(A0A, A09, BA, B7, B3, B0, CE, MA); \
		PERM_ELT_CUDA(A0B, A0A, BB, B8, B4, B1, CD, MB); \
		PERM_ELT_CUDA(A00, A0B, BC, B9, B5, B2, CC, MC); \
		PERM_ELT_CUDA(A01, A00, BD, BA, B6, B3, CB, MD); \
		PERM_ELT_CUDA(A02, A01, BE, BB, B7, B4, CA, ME); \
		PERM_ELT_CUDA(A03, A02, BF, BC, B8, B5, C9, MF); \
				} while (0)

#define PERM_STEP_1_CUDA do { \
		PERM_ELT_CUDA(A04, A03, B0, BD, B9, B6, C8, M0); \
		PERM_ELT_CUDA(A05, A04, B1, BE, BA, B7, C7, M1); \
		PERM_ELT_CUDA(A06, A05, B2, BF, BB, B8, C6, M2); \
		PERM_ELT_CUDA(A07, A06, B3, B0, BC, B9, C5, M3); \
		PERM_ELT_CUDA(A08, A07, B4, B1, BD, BA, C4, M4); \
		PERM_ELT_CUDA(A09, A08, B5, B2, BE, BB, C3, M5); \
		PERM_ELT_CUDA(A0A, A09, B6, B3, BF, BC, C2, M6); \
		PERM_ELT_CUDA(A0B, A0A, B7, B4, B0, BD, C1, M7); \
		PERM_ELT_CUDA(A00, A0B, B8, B5, B1, BE, C0, M8); \
		PERM_ELT_CUDA(A01, A00, B9, B6, B2, BF, CF, M9); \
		PERM_ELT_CUDA(A02, A01, BA, B7, B3, B0, CE, MA); \
		PERM_ELT_CUDA(A03, A02, BB, B8, B4, B1, CD, MB); \
		PERM_ELT_CUDA(A04, A03, BC, B9, B5, B2, CC, MC); \
		PERM_ELT_CUDA(A05, A04, BD, BA, B6, B3, CB, MD); \
		PERM_ELT_CUDA(A06, A05, BE, BB, B7, B4, CA, ME); \
		PERM_ELT_CUDA(A07, A06, BF, BC, B8, B5, C9, MF); \
				} while (0)

#define PERM_STEP_2_CUDA do { \
		PERM_ELT_CUDA(A08, A07, B0, BD, B9, B6, C8, M0); \
		PERM_ELT_CUDA(A09, A08, B1, BE, BA, B7, C7, M1); \
		PERM_ELT_CUDA(A0A, A09, B2, BF, BB, B8, C6, M2); \
		PERM_ELT_CUDA(A0B, A0A, B3, B0, BC, B9, C5, M3); \
		PERM_ELT_CUDA(A00, A0B, B4, B1, BD, BA, C4, M4); \
		PERM_ELT_CUDA(A01, A00, B5, B2, BE, BB, C3, M5); \
		PERM_ELT_CUDA(A02, A01, B6, B3, BF, BC, C2, M6); \
		PERM_ELT_CUDA(A03, A02, B7, B4, B0, BD, C1, M7); \
		PERM_ELT_CUDA(A04, A03, B8, B5, B1, BE, C0, M8); \
		PERM_ELT_CUDA(A05, A04, B9, B6, B2, BF, CF, M9); \
		PERM_ELT_CUDA(A06, A05, BA, B7, B3, B0, CE, MA); \
		PERM_ELT_CUDA(A07, A06, BB, B8, B4, B1, CD, MB); \
		PERM_ELT_CUDA(A08, A07, BC, B9, B5, B2, CC, MC); \
		PERM_ELT_CUDA(A09, A08, BD, BA, B6, B3, CB, MD); \
		PERM_ELT_CUDA(A0A, A09, BE, BB, B7, B4, CA, ME); \
		PERM_ELT_CUDA(A0B, A0A, BF, BC, B8, B5, C9, MF); \
				} while (0)

#define APPLY_P_CUDA do { \
		B0 = T32(B0 << 17) | (B0 >> 15); \
		B1 = T32(B1 << 17) | (B1 >> 15); \
		B2 = T32(B2 << 17) | (B2 >> 15); \
		B3 = T32(B3 << 17) | (B3 >> 15); \
		B4 = T32(B4 << 17) | (B4 >> 15); \
		B5 = T32(B5 << 17) | (B5 >> 15); \
		B6 = T32(B6 << 17) | (B6 >> 15); \
		B7 = T32(B7 << 17) | (B7 >> 15); \
		B8 = T32(B8 << 17) | (B8 >> 15); \
		B9 = T32(B9 << 17) | (B9 >> 15); \
		BA = T32(BA << 17) | (BA >> 15); \
		BB = T32(BB << 17) | (BB >> 15); \
		BC = T32(BC << 17) | (BC >> 15); \
		BD = T32(BD << 17) | (BD >> 15); \
		BE = T32(BE << 17) | (BE >> 15); \
		BF = T32(BF << 17) | (BF >> 15); \
		PERM_STEP_0_CUDA; \
		PERM_STEP_1_CUDA; \
		PERM_STEP_2_CUDA; \
		A0B = T32(A0B + C6); \
		A0A = T32(A0A + C5); \
		A09 = T32(A09 + C4); \
		A08 = T32(A08 + C3); \
		A07 = T32(A07 + C2); \
		A06 = T32(A06 + C1); \
		A05 = T32(A05 + C0); \
		A04 = T32(A04 + CF); \
		A03 = T32(A03 + CE); \
		A02 = T32(A02 + CD); \
		A01 = T32(A01 + CC); \
		A00 = T32(A00 + CB); \
		A0B = T32(A0B + CA); \
		A0A = T32(A0A + C9); \
		A09 = T32(A09 + C8); \
		A08 = T32(A08 + C7); \
		A07 = T32(A07 + C6); \
		A06 = T32(A06 + C5); \
		A05 = T32(A05 + C4); \
		A04 = T32(A04 + C3); \
		A03 = T32(A03 + C2); \
		A02 = T32(A02 + C1); \
		A01 = T32(A01 + C0); \
		A00 = T32(A00 + CF); \
		A0B = T32(A0B + CE); \
		A0A = T32(A0A + CD); \
		A09 = T32(A09 + CC); \
		A08 = T32(A08 + CB); \
		A07 = T32(A07 + CA); \
		A06 = T32(A06 + C9); \
		A05 = T32(A05 + C8); \
		A04 = T32(A04 + C7); \
		A03 = T32(A03 + C6); \
		A02 = T32(A02 + C5); \
		A01 = T32(A01 + C4); \
		A00 = T32(A00 + C3); \
		} while (0)



#define INCR_W do { \
	if ((Wlow = T32(Wlow + 1)) == 0) \
		Whigh = T32(Whigh + 1); \
	} while (0)


__constant__ uint32_t __align__(16) a_pre[16]; // padded message (80 bytes + padding)
__constant__ uint32_t __align__(16) b_pre[16]; // padded message (80 bytes + padding)
__constant__ uint32_t __align__(16) c_pre[16]; // padded message (80 bytes + padding)


__host__
void x16_shabal512_setBlock_80(void *pdata)
{
	uint32_t A[16];
	uint32_t B[16];
	uint32_t C[16];

	uint32_t M[16];

	static const sph_u32 A_init_512[] = {
		C32(0x20728DFD), C32(0x46C0BD53), C32(0xE782B699), C32(0x55304632),
		C32(0x71B4EF90), C32(0x0EA9E82C), C32(0xDBB930F1), C32(0xFAD06B8B),
		C32(0xBE0CAE40), C32(0x8BD14410), C32(0x76D2ADAC), C32(0x28ACAB7F)
	};

	static const sph_u32 B_init_512[] = {
		C32(0xC1099CB7), C32(0x07B385F3), C32(0xE7442C26), C32(0xCC8AD640),
		C32(0xEB6F56C7), C32(0x1EA81AA9), C32(0x73B9D314), C32(0x1DE85D08),
		C32(0x48910A5A), C32(0x893B22DB), C32(0xC5A0DF44), C32(0xBBC4324E),
		C32(0x72D2F240), C32(0x75941D99), C32(0x6D8BDE82), C32(0xA1A7502B)
	};

	static const sph_u32 C_init_512[] = {
		C32(0xD9BF68D1), C32(0x58BAD750), C32(0x56028CB2), C32(0x8134F359),
		C32(0xB5D469D8), C32(0x941A8CC2), C32(0x418B2A6E), C32(0x04052780),
		C32(0x7F07D787), C32(0x5194358F), C32(0x3C60D665), C32(0xBE97D79A),
		C32(0x950C3434), C32(0xAED9A06D), C32(0x2537DC8D), C32(0x7CDB5969)
	};

	uint32_t t;

	// todo: try __ldc
	*(uint2x4*)&M[0] = *(uint2x4*)&((uint32_t *)pdata)[0];
	*(uint2x4*)&M[8] = *(uint2x4*)&((uint32_t *)pdata)[8];

	sph_u32 A00 = A_init_512[0], A01 = A_init_512[1], A02 = A_init_512[2], A03 = A_init_512[3];
	sph_u32 A04 = A_init_512[4], A05 = A_init_512[5], A06 = A_init_512[6], A07 = A_init_512[7];
	sph_u32 A08 = A_init_512[8], A09 = A_init_512[9], A0A = A_init_512[10], A0B = A_init_512[11];

	sph_u32 B0 = B_init_512[0], B1 = B_init_512[1], B2 = B_init_512[2], B3 = B_init_512[3];
	sph_u32 B4 = B_init_512[4], B5 = B_init_512[5], B6 = B_init_512[6], B7 = B_init_512[7];
	sph_u32 B8 = B_init_512[8], B9 = B_init_512[9], BA = B_init_512[10], BB = B_init_512[11];
	sph_u32 BC = B_init_512[12], BD = B_init_512[13], BE = B_init_512[14], BF = B_init_512[15];

	sph_u32 C0 = C_init_512[0], C1 = C_init_512[1], C2 = C_init_512[2], C3 = C_init_512[3];
	sph_u32 C4 = C_init_512[4], C5 = C_init_512[5], C6 = C_init_512[6], C7 = C_init_512[7];
	sph_u32 C8 = C_init_512[8], C9 = C_init_512[9], CA = C_init_512[10], CB = C_init_512[11];
	sph_u32 CC = C_init_512[12], CD = C_init_512[13], CE = C_init_512[14], CF = C_init_512[15];

	sph_u32 M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, MA, MB, MC, MD, ME, MF;
	sph_u32 Wlow = 1, Whigh = 0;

	M0 = M[0];
	M1 = M[1];
	M2 = M[2];
	M3 = M[3];
	M4 = M[4];
	M5 = M[5];
	M6 = M[6];
	M7 = M[7];
	M8 = M[8];
	M9 = M[9];
	MA = M[10];
	MB = M[11];
	MC = M[12];
	MD = M[13];
	ME = M[14];
	MF = M[15];

	INPUT_BLOCK_ADD;
	XOR_W;
	APPLY_P;
	INPUT_BLOCK_SUB;
	SWAP_BC;
	INCR_W;
	/*
	M0 = ((uint32_t *)pdata)[16];
	M1 = ((uint32_t *)pdata)[17];
	M2 = ((uint32_t *)pdata)[18];
	M4 = 0x80;
	M5 = M6 = M7 = M8 = M9 = MA = MB = MC = MD = ME = MF = 0;

	B0 = T32(B0 + M0);
	B1 = T32(B1 + M1);
	B2 = T32(B2 + M2);
	B4 = T32(B4 + M4);
	B5 = T32(B5 + M5);
	B6 = T32(B6 + M6);
	B7 = T32(B7 + M7);
	B8 = T32(B8 + M8);
	B9 = T32(B9 + M9);
	BA = T32(BA + MA);
	BB = T32(BB + MB);
	BC = T32(BC + MC);
	BD = T32(BD + MD);
	BE = T32(BE + ME);
	BF = T32(BF + MF);

	A00 ^= Wlow;
	A01 ^= Whigh;

	B0 = T32(B0 << 17) | (B0 >> 15); \
	B1 = T32(B1 << 17) | (B1 >> 15); \
	B2 = T32(B2 << 17) | (B2 >> 15); \
	B4 = T32(B4 << 17) | (B4 >> 15); \
	B5 = T32(B5 << 17) | (B5 >> 15); \
	B6 = T32(B6 << 17) | (B6 >> 15); \
	B7 = T32(B7 << 17) | (B7 >> 15); \
	B8 = T32(B8 << 17) | (B8 >> 15); \
	B9 = T32(B9 << 17) | (B9 >> 15); \
	BA = T32(BA << 17) | (BA >> 15); \
	BB = T32(BB << 17) | (BB >> 15); \
	BC = T32(BC << 17) | (BC >> 15); \
	BD = T32(BD << 17) | (BD >> 15); \
	BE = T32(BE << 17) | (BE >> 15); \
	BF = T32(BF << 17) | (BF >> 15); \
	*/


	M0 = ((uint32_t *)pdata)[16];
	M1 = ((uint32_t *)pdata)[17];
	M2 = ((uint32_t *)pdata)[18];
	M4 = 0x80;
	M5 = M6 = M7 = M8 = M9 = MA = MB = MC = MD = ME = MF = 0;
	//		INPUT_BLOCK_ADD;
	B0 = T32(B0 + M0);
	B1 = T32(B1 + M1);
	B2 = T32(B2 + M2);
	B4 = T32(B4 + M4);
	B5 = T32(B5 + M5);
	B6 = T32(B6 + M6);
	B7 = T32(B7 + M7);
	B8 = T32(B8 + M8);
	B9 = T32(B9 + M9);
	BA = T32(BA + MA);
	BB = T32(BB + MB);
	BC = T32(BC + MC);
	BD = T32(BD + MD);
	BE = T32(BE + ME);
	BF = T32(BF + MF);
	A00 ^= Wlow;
	A01 ^= Whigh;

	B0 = T32(B0 << 17) | (B0 >> 15);
	B1 = T32(B1 << 17) | (B1 >> 15);
	B2 = T32(B2 << 17) | (B2 >> 15);
	B4 = T32(B4 << 17) | (B4 >> 15);
	B5 = T32(B5 << 17) | (B5 >> 15);
	B6 = T32(B6 << 17) | (B6 >> 15);
	B7 = T32(B7 << 17) | (B7 >> 15);
	B8 = T32(B8 << 17) | (B8 >> 15);
	B9 = T32(B9 << 17) | (B9 >> 15);
	BA = T32(BA << 17) | (BA >> 15);
	BB = T32(BB << 17) | (BB >> 15);
	BC = T32(BC << 17) | (BC >> 15);
	BD = T32(BD << 17) | (BD >> 15);
	BE = T32(BE << 17) | (BE >> 15);
	BF = T32(BF << 17) | (BF >> 15);


	PERM_ELT(A00, A0B, B0, BD, B9, B6, C8, M0);
	PERM_ELT(A01, A00, B1, BE, BA, B7, C7, M1);
	PERM_ELT(A02, A01, B2, BF, BB, B8, C6, M2);

	A03 = T32((A03 ^ (((A02 << 15) | (A02 >> 17)) * 5U) ^ C5) * 3U)
		^ B0 ^ (BC & ~B9);


	A[0] = A00;
	A[1] = A01;
	A[2] = A02;
	A[3] = A03;
	A[4] = A04;
	A[5] = A05;
	A[6] = A06;
	A[7] = A07;
	A[8] = A08;
	A[9] = A09;
	A[10] = A0A;
	A[11] = A0B;

	B[0] = B0;
	B[1] = B1;
	B[2] = B2;
	B[3] = B3;
	B[4] = B4;
	B[5] = B5;
	B[6] = B6;
	B[7] = B7;
	B[8] = B8;
	B[9] = B9;
	B[10] = BA;
	B[11] = BB;
	B[12] = BC;
	B[13] = BD;
	B[14] = BE;
	B[15] = BF;

	C[0] = C0;
	C[1] = C1;
	C[2] = C2;
	C[3] = C3;
	C[4] = C4;
	C[5] = C5;
	C[6] = C6;
	C[7] = C7;
	C[8] = C8;
	C[9] = C9;
	C[10] = CA;
	C[11] = CB;
	C[12] = CC;
	C[13] = CD;
	C[14] = CE;
	C[15] = CF;

	A[12] = ((uint32_t *)pdata)[16];
	A[13] = ((uint32_t *)pdata)[17];
	A[14] = ((uint32_t *)pdata)[18];

	//	cudaMemcpyToSymbol(c_PaddedMessage80, pre, sizeof(pre), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(a_pre, A, sizeof(a_pre), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(b_pre, B, sizeof(b_pre), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_pre, C, sizeof(c_pre), 0, cudaMemcpyHostToDevice);

}


#define TPB_SHABAL 256

__global__ __launch_bounds__(TPB_SHABAL, 2)
void x16_shabal512_gpu_hash_80(uint32_t threads, const uint32_t startNonce, uint32_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint32_t B[] = {
		0xC1099CB7, 0x07B385F3, 0xE7442C26, 0xCC8AD640, 0xEB6F56C7, 0x1EA81AA9, 0x73B9D314, 0x1DE85D08,
		0x48910A5A, 0x893B22DB, 0xC5A0DF44, 0xBBC4324E, 0x72D2F240, 0x75941D99, 0x6D8BDE82, 0xA1A7502B
	};

	/*	static const sph_u32 A_init_512[] = {
	C32(0x20728DFD), C32(0x46C0BD53), C32(0xE782B699), C32(0x55304632),
	C32(0x71B4EF90), C32(0x0EA9E82C), C32(0xDBB930F1), C32(0xFAD06B8B),
	C32(0xBE0CAE40), C32(0x8BD14410), C32(0x76D2ADAC), C32(0x28ACAB7F)
	};

	static const sph_u32 B_init_512[] = {
	C32(0xC1099CB7), C32(0x07B385F3), C32(0xE7442C26), C32(0xCC8AD640),
	C32(0xEB6F56C7), C32(0x1EA81AA9), C32(0x73B9D314), C32(0x1DE85D08),
	C32(0x48910A5A), C32(0x893B22DB), C32(0xC5A0DF44), C32(0xBBC4324E),
	C32(0x72D2F240), C32(0x75941D99), C32(0x6D8BDE82), C32(0xA1A7502B)
	};

	static const sph_u32 C_init_512[] = {
	C32(0xD9BF68D1), C32(0x58BAD750), C32(0x56028CB2), C32(0x8134F359),
	C32(0xB5D469D8), C32(0x941A8CC2), C32(0x418B2A6E), C32(0x04052780),
	C32(0x7F07D787), C32(0x5194358F), C32(0x3C60D665), C32(0xBE97D79A),
	C32(0x950C3434), C32(0xAED9A06D), C32(0x2537DC8D), C32(0x7CDB5969)
	};
	*/

	//	uint32_t M[16];

	if (thread < threads)
	{
		// todo: try __ldc
		//	*(uint2x4*)&M[0] = *(uint2x4*)&c_PaddedMessage80[0];
		//	*(uint2x4*)&M[8] = *(uint2x4*)&c_PaddedMessage80[8];


		uint32_t t;

		sph_u32 A00, A01, A02, A03;
		sph_u32 A04, A05, A06, A07;
		sph_u32 A08, A09, A0A, A0B;

		sph_u32 B0, B1, B2, B3;
		sph_u32 B4, B5, B6, B7;
		sph_u32 B8, B9, BA, BB;
		sph_u32 BC, BD, BE, BF;

		sph_u32 C0, C1, C2, C3;
		sph_u32 C4, C5, C6, C7;
		sph_u32 C8, C9, CA, CB;
		sph_u32 CC, CD, CE, CF;

		sph_u32 M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, MA, MB, MC, MD, ME, MF;
		sph_u32 Wlow = 2, Whigh = 0;

		/*		M0 = M[0];
		M1 = M[1];
		M2 = M[2];
		M3 = M[3];
		M4 = M[4];
		M5 = M[5];
		M6 = M[6];
		M7 = M[7];
		M8 = M[8];
		M9 = M[9];
		MA = M[10];
		MB = M[11];
		MC = M[12];
		MD = M[13];
		ME = M[14];
		MF = M[15];
		*/
		A00 = a_pre[0];
		A01 = a_pre[1];
		A02 = a_pre[2];
		A03 = a_pre[3];
		A04 = a_pre[4];
		A05 = a_pre[5];
		A06 = a_pre[6];
		A07 = a_pre[7];
		A08 = a_pre[8];
		A09 = a_pre[9];
		A0A = a_pre[10];
		A0B = a_pre[11];
		M0 = a_pre[12];
		M1 = a_pre[13];
		M2 = a_pre[14];

		B0 = b_pre[0];
		B1 = b_pre[1];
		B2 = b_pre[2];
		B3 = b_pre[3];
		B4 = b_pre[4];
		B5 = b_pre[5];
		B6 = b_pre[6];
		B7 = b_pre[7];
		B8 = b_pre[8];
		B9 = b_pre[9];
		BA = b_pre[10];
		BB = b_pre[11];
		BC = b_pre[12];
		BD = b_pre[13];
		BE = b_pre[14];
		BF = b_pre[15];

		C0 = c_pre[0];
		C1 = c_pre[1];
		C2 = c_pre[2];
		C3 = c_pre[3];
		C4 = c_pre[4];
		C5 = c_pre[5];
		C6 = c_pre[6];
		C7 = c_pre[7];
		C8 = c_pre[8];
		C9 = c_pre[9];
		CA = c_pre[10];
		CB = c_pre[11];
		CC = c_pre[12];
		CD = c_pre[13];
		CE = c_pre[14];
		CF = c_pre[15];


		M4 = 0x80;
		M5 = M6 = M7 = M8 = M9 = MA = MB = MC = MD = ME = MF = 0;

		//		XOR_W;
		//		APPLY_P;
		//		PERM_STEP_0; 

		M3 = cuda_swab32(startNonce + thread);
		B3 = T32(B3 + M3);
		B3 = T32(B3 << 17) | (B3 >> 15);

		//			PERM_ELT(A03, A02, B3, B0, BC, B9, C5, M3);
		A03 ^= M3;
		B3 = (~(((B3 << 1) | (B3 >> 31)) ^ A03));

		PERM_ELT_CUDA(A04, A03, B4, B1, BD, BA, C4, M4);
		PERM_ELT_CUDA(A05, A04, B5, B2, BE, BB, C3, M5);
		PERM_ELT_CUDA(A06, A05, B6, B3, BF, BC, C2, M6);
		PERM_ELT_CUDA(A07, A06, B7, B4, B0, BD, C1, M7);
		PERM_ELT_CUDA(A08, A07, B8, B5, B1, BE, C0, M8);
		PERM_ELT_CUDA(A09, A08, B9, B6, B2, BF, CF, M9);
		PERM_ELT_CUDA(A0A, A09, BA, B7, B3, B0, CE, MA);
		PERM_ELT_CUDA(A0B, A0A, BB, B8, B4, B1, CD, MB);
		PERM_ELT_CUDA(A00, A0B, BC, B9, B5, B2, CC, MC);
		PERM_ELT_CUDA(A01, A00, BD, BA, B6, B3, CB, MD);
		PERM_ELT_CUDA(A02, A01, BE, BB, B7, B4, CA, ME);
		PERM_ELT_CUDA(A03, A02, BF, BC, B8, B5, C9, MF);


		PERM_STEP_1_CUDA;
		PERM_STEP_2_CUDA;
		A0B = T32(A0B + C6);
		A0A = T32(A0A + C5);
		A09 = T32(A09 + C4);
		A08 = T32(A08 + C3);
		A07 = T32(A07 + C2);
		A06 = T32(A06 + C1);
		A05 = T32(A05 + C0);
		A04 = T32(A04 + CF);
		A03 = T32(A03 + CE);
		A02 = T32(A02 + CD);
		A01 = T32(A01 + CC);
		A00 = T32(A00 + CB);
		A0B = T32(A0B + CA);
		A0A = T32(A0A + C9);
		A09 = T32(A09 + C8);
		A08 = T32(A08 + C7);
		A07 = T32(A07 + C6);
		A06 = T32(A06 + C5);
		A05 = T32(A05 + C4);
		A04 = T32(A04 + C3);
		A03 = T32(A03 + C2);
		A02 = T32(A02 + C1);
		A01 = T32(A01 + C0);
		A00 = T32(A00 + CF);
		A0B = T32(A0B + CE);
		A0A = T32(A0A + CD);
		A09 = T32(A09 + CC);
		A08 = T32(A08 + CB);
		A07 = T32(A07 + CA);
		A06 = T32(A06 + C9);
		A05 = T32(A05 + C8);
		A04 = T32(A04 + C7);
		A03 = T32(A03 + C6);
		A02 = T32(A02 + C5);
		A01 = T32(A01 + C4);
		A00 = T32(A00 + C3);

		for (unsigned i = 0; i < 3; i++) {
			SWAP_BC_CUDA;
			XOR_W_CUDA;
			APPLY_P_CUDA;
		}

		B[0] = B0;
		B[1] = B1;
		B[2] = B2;
		B[3] = B3;
		B[4] = B4;
		B[5] = B5;
		B[6] = B6;
		B[7] = B7;
		B[8] = B8;
		B[9] = B9;
		B[10] = BA;
		B[11] = BB;
		B[12] = BC;
		B[13] = BD;
		B[14] = BE;
		B[15] = BF;

		// output
		uint64_t hashPosition = thread;
		uint32_t *Hash = &g_hash[hashPosition << 4];
		*(uint2x4*)&Hash[0] = *(uint2x4*)&B[0];
		*(uint2x4*)&Hash[8] = *(uint2x4*)&B[8];
	}
}
__host__
void x16_shabal512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = TPB_SHABAL;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	x16_shabal512_gpu_hash_80 << <grid, block >> >(threads, startNonce, d_hash);
}
