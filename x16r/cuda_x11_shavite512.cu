/*
	Based on Tanguy Pruvot's repo
	Provos Alexis - 2016
	optimized by sp - 2018
*/
#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"

#define INTENSIVE_GMF
//#include "cuda_x11_aes.cuh"
__constant__ uint32_t c_PaddedMessage80[20]; // padded message (80 bytes + padding)


#define TPB 128

#define AESx(x) (x ##UL) /* SPH_C32(x) */

__constant__ __align__(64) uint32_t d_AES1[256] = {
	AESx(0x6363C6A5), AESx(0x7C7CF884), AESx(0x7777EE99), AESx(0x7B7BF68D),
	AESx(0xF2F2FF0D), AESx(0x6B6BD6BD), AESx(0x6F6FDEB1), AESx(0xC5C59154),
	AESx(0x30306050), AESx(0x01010203), AESx(0x6767CEA9), AESx(0x2B2B567D),
	AESx(0xFEFEE719), AESx(0xD7D7B562), AESx(0xABAB4DE6), AESx(0x7676EC9A),
	AESx(0xCACA8F45), AESx(0x82821F9D), AESx(0xC9C98940), AESx(0x7D7DFA87),
	AESx(0xFAFAEF15), AESx(0x5959B2EB), AESx(0x47478EC9), AESx(0xF0F0FB0B),
	AESx(0xADAD41EC), AESx(0xD4D4B367), AESx(0xA2A25FFD), AESx(0xAFAF45EA),
	AESx(0x9C9C23BF), AESx(0xA4A453F7), AESx(0x7272E496), AESx(0xC0C09B5B),
	AESx(0xB7B775C2), AESx(0xFDFDE11C), AESx(0x93933DAE), AESx(0x26264C6A),
	AESx(0x36366C5A), AESx(0x3F3F7E41), AESx(0xF7F7F502), AESx(0xCCCC834F),
	AESx(0x3434685C), AESx(0xA5A551F4), AESx(0xE5E5D134), AESx(0xF1F1F908),
	AESx(0x7171E293), AESx(0xD8D8AB73), AESx(0x31316253), AESx(0x15152A3F),
	AESx(0x0404080C), AESx(0xC7C79552), AESx(0x23234665), AESx(0xC3C39D5E),
	AESx(0x18183028), AESx(0x969637A1), AESx(0x05050A0F), AESx(0x9A9A2FB5),
	AESx(0x07070E09), AESx(0x12122436), AESx(0x80801B9B), AESx(0xE2E2DF3D),
	AESx(0xEBEBCD26), AESx(0x27274E69), AESx(0xB2B27FCD), AESx(0x7575EA9F),
	AESx(0x0909121B), AESx(0x83831D9E), AESx(0x2C2C5874), AESx(0x1A1A342E),
	AESx(0x1B1B362D), AESx(0x6E6EDCB2), AESx(0x5A5AB4EE), AESx(0xA0A05BFB),
	AESx(0x5252A4F6), AESx(0x3B3B764D), AESx(0xD6D6B761), AESx(0xB3B37DCE),
	AESx(0x2929527B), AESx(0xE3E3DD3E), AESx(0x2F2F5E71), AESx(0x84841397),
	AESx(0x5353A6F5), AESx(0xD1D1B968), AESx(0x00000000), AESx(0xEDEDC12C),
	AESx(0x20204060), AESx(0xFCFCE31F), AESx(0xB1B179C8), AESx(0x5B5BB6ED),
	AESx(0x6A6AD4BE), AESx(0xCBCB8D46), AESx(0xBEBE67D9), AESx(0x3939724B),
	AESx(0x4A4A94DE), AESx(0x4C4C98D4), AESx(0x5858B0E8), AESx(0xCFCF854A),
	AESx(0xD0D0BB6B), AESx(0xEFEFC52A), AESx(0xAAAA4FE5), AESx(0xFBFBED16),
	AESx(0x434386C5), AESx(0x4D4D9AD7), AESx(0x33336655), AESx(0x85851194),
	AESx(0x45458ACF), AESx(0xF9F9E910), AESx(0x02020406), AESx(0x7F7FFE81),
	AESx(0x5050A0F0), AESx(0x3C3C7844), AESx(0x9F9F25BA), AESx(0xA8A84BE3),
	AESx(0x5151A2F3), AESx(0xA3A35DFE), AESx(0x404080C0), AESx(0x8F8F058A),
	AESx(0x92923FAD), AESx(0x9D9D21BC), AESx(0x38387048), AESx(0xF5F5F104),
	AESx(0xBCBC63DF), AESx(0xB6B677C1), AESx(0xDADAAF75), AESx(0x21214263),
	AESx(0x10102030), AESx(0xFFFFE51A), AESx(0xF3F3FD0E), AESx(0xD2D2BF6D),
	AESx(0xCDCD814C), AESx(0x0C0C1814), AESx(0x13132635), AESx(0xECECC32F),
	AESx(0x5F5FBEE1), AESx(0x979735A2), AESx(0x444488CC), AESx(0x17172E39),
	AESx(0xC4C49357), AESx(0xA7A755F2), AESx(0x7E7EFC82), AESx(0x3D3D7A47),
	AESx(0x6464C8AC), AESx(0x5D5DBAE7), AESx(0x1919322B), AESx(0x7373E695),
	AESx(0x6060C0A0), AESx(0x81811998), AESx(0x4F4F9ED1), AESx(0xDCDCA37F),
	AESx(0x22224466), AESx(0x2A2A547E), AESx(0x90903BAB), AESx(0x88880B83),
	AESx(0x46468CCA), AESx(0xEEEEC729), AESx(0xB8B86BD3), AESx(0x1414283C),
	AESx(0xDEDEA779), AESx(0x5E5EBCE2), AESx(0x0B0B161D), AESx(0xDBDBAD76),
	AESx(0xE0E0DB3B), AESx(0x32326456), AESx(0x3A3A744E), AESx(0x0A0A141E),
	AESx(0x494992DB), AESx(0x06060C0A), AESx(0x2424486C), AESx(0x5C5CB8E4),
	AESx(0xC2C29F5D), AESx(0xD3D3BD6E), AESx(0xACAC43EF), AESx(0x6262C4A6),
	AESx(0x919139A8), AESx(0x959531A4), AESx(0xE4E4D337), AESx(0x7979F28B),
	AESx(0xE7E7D532), AESx(0xC8C88B43), AESx(0x37376E59), AESx(0x6D6DDAB7),
	AESx(0x8D8D018C), AESx(0xD5D5B164), AESx(0x4E4E9CD2), AESx(0xA9A949E0),
	AESx(0x6C6CD8B4), AESx(0x5656ACFA), AESx(0xF4F4F307), AESx(0xEAEACF25),
	AESx(0x6565CAAF), AESx(0x7A7AF48E), AESx(0xAEAE47E9), AESx(0x08081018),
	AESx(0xBABA6FD5), AESx(0x7878F088), AESx(0x25254A6F), AESx(0x2E2E5C72),
	AESx(0x1C1C3824), AESx(0xA6A657F1), AESx(0xB4B473C7), AESx(0xC6C69751),
	AESx(0xE8E8CB23), AESx(0xDDDDA17C), AESx(0x7474E89C), AESx(0x1F1F3E21),
	AESx(0x4B4B96DD), AESx(0xBDBD61DC), AESx(0x8B8B0D86), AESx(0x8A8A0F85),
	AESx(0x7070E090), AESx(0x3E3E7C42), AESx(0xB5B571C4), AESx(0x6666CCAA),
	AESx(0x484890D8), AESx(0x03030605), AESx(0xF6F6F701), AESx(0x0E0E1C12),
	AESx(0x6161C2A3), AESx(0x35356A5F), AESx(0x5757AEF9), AESx(0xB9B969D0),
	AESx(0x86861791), AESx(0xC1C19958), AESx(0x1D1D3A27), AESx(0x9E9E27B9),
	AESx(0xE1E1D938), AESx(0xF8F8EB13), AESx(0x98982BB3), AESx(0x11112233),
	AESx(0x6969D2BB), AESx(0xD9D9A970), AESx(0x8E8E0789), AESx(0x949433A7),
	AESx(0x9B9B2DB6), AESx(0x1E1E3C22), AESx(0x87871592), AESx(0xE9E9C920),
	AESx(0xCECE8749), AESx(0x5555AAFF), AESx(0x28285078), AESx(0xDFDFA57A),
	AESx(0x8C8C038F), AESx(0xA1A159F8), AESx(0x89890980), AESx(0x0D0D1A17),
	AESx(0xBFBF65DA), AESx(0xE6E6D731), AESx(0x424284C6), AESx(0x6868D0B8),
	AESx(0x414182C3), AESx(0x999929B0), AESx(0x2D2D5A77), AESx(0x0F0F1E11),
	AESx(0xB0B07BCB), AESx(0x5454A8FC), AESx(0xBBBB6DD6), AESx(0x16162C3A)
};

__device__  static uint32_t __align__(16) d_AES0[256] = {
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

__device__ static uint32_t __align__(16) d_AES3[256] = {
	0xC6A56363, 0xF8847C7C, 0xEE997777, 0xF68D7B7B, 0xFF0DF2F2, 0xD6BD6B6B, 0xDEB16F6F, 0x9154C5C5, 0x60503030, 0x02030101, 0xCEA96767, 0x567D2B2B, 0xE719FEFE, 0xB562D7D7, 0x4DE6ABAB, 0xEC9A7676,
	0x8F45CACA, 0x1F9D8282, 0x8940C9C9, 0xFA877D7D, 0xEF15FAFA, 0xB2EB5959, 0x8EC94747, 0xFB0BF0F0, 0x41ECADAD, 0xB367D4D4, 0x5FFDA2A2, 0x45EAAFAF, 0x23BF9C9C, 0x53F7A4A4, 0xE4967272, 0x9B5BC0C0,
	0x75C2B7B7, 0xE11CFDFD, 0x3DAE9393, 0x4C6A2626, 0x6C5A3636, 0x7E413F3F, 0xF502F7F7, 0x834FCCCC, 0x685C3434, 0x51F4A5A5, 0xD134E5E5, 0xF908F1F1, 0xE2937171, 0xAB73D8D8, 0x62533131, 0x2A3F1515,
	0x080C0404, 0x9552C7C7, 0x46652323, 0x9D5EC3C3, 0x30281818, 0x37A19696, 0x0A0F0505, 0x2FB59A9A, 0x0E090707, 0x24361212, 0x1B9B8080, 0xDF3DE2E2, 0xCD26EBEB, 0x4E692727, 0x7FCDB2B2, 0xEA9F7575,
	0x121B0909, 0x1D9E8383, 0x58742C2C, 0x342E1A1A, 0x362D1B1B, 0xDCB26E6E, 0xB4EE5A5A, 0x5BFBA0A0, 0xA4F65252, 0x764D3B3B, 0xB761D6D6, 0x7DCEB3B3, 0x527B2929, 0xDD3EE3E3, 0x5E712F2F, 0x13978484,
	0xA6F55353, 0xB968D1D1, 0x00000000, 0xC12CEDED, 0x40602020, 0xE31FFCFC, 0x79C8B1B1, 0xB6ED5B5B, 0xD4BE6A6A, 0x8D46CBCB, 0x67D9BEBE, 0x724B3939, 0x94DE4A4A, 0x98D44C4C, 0xB0E85858, 0x854ACFCF,
	0xBB6BD0D0, 0xC52AEFEF, 0x4FE5AAAA, 0xED16FBFB, 0x86C54343, 0x9AD74D4D, 0x66553333, 0x11948585, 0x8ACF4545, 0xE910F9F9, 0x04060202, 0xFE817F7F, 0xA0F05050, 0x78443C3C, 0x25BA9F9F, 0x4BE3A8A8,
	0xA2F35151, 0x5DFEA3A3, 0x80C04040, 0x058A8F8F, 0x3FAD9292, 0x21BC9D9D, 0x70483838, 0xF104F5F5, 0x63DFBCBC, 0x77C1B6B6, 0xAF75DADA, 0x42632121, 0x20301010, 0xE51AFFFF, 0xFD0EF3F3, 0xBF6DD2D2,
	0x814CCDCD, 0x18140C0C, 0x26351313, 0xC32FECEC, 0xBEE15F5F, 0x35A29797, 0x88CC4444, 0x2E391717, 0x9357C4C4, 0x55F2A7A7, 0xFC827E7E, 0x7A473D3D, 0xC8AC6464, 0xBAE75D5D, 0x322B1919, 0xE6957373,
	0xC0A06060, 0x19988181, 0x9ED14F4F, 0xA37FDCDC, 0x44662222, 0x547E2A2A, 0x3BAB9090, 0x0B838888, 0x8CCA4646, 0xC729EEEE, 0x6BD3B8B8, 0x283C1414, 0xA779DEDE, 0xBCE25E5E, 0x161D0B0B, 0xAD76DBDB,
	0xDB3BE0E0, 0x64563232, 0x744E3A3A, 0x141E0A0A, 0x92DB4949, 0x0C0A0606, 0x486C2424, 0xB8E45C5C, 0x9F5DC2C2, 0xBD6ED3D3, 0x43EFACAC, 0xC4A66262, 0x39A89191, 0x31A49595, 0xD337E4E4, 0xF28B7979,
	0xD532E7E7, 0x8B43C8C8, 0x6E593737, 0xDAB76D6D, 0x018C8D8D, 0xB164D5D5, 0x9CD24E4E, 0x49E0A9A9, 0xD8B46C6C, 0xACFA5656, 0xF307F4F4, 0xCF25EAEA, 0xCAAF6565, 0xF48E7A7A, 0x47E9AEAE, 0x10180808,
	0x6FD5BABA, 0xF0887878, 0x4A6F2525, 0x5C722E2E, 0x38241C1C, 0x57F1A6A6, 0x73C7B4B4, 0x9751C6C6, 0xCB23E8E8, 0xA17CDDDD, 0xE89C7474, 0x3E211F1F, 0x96DD4B4B, 0x61DCBDBD, 0x0D868B8B, 0x0F858A8A,
	0xE0907070, 0x7C423E3E, 0x71C4B5B5, 0xCCAA6666, 0x90D84848, 0x06050303, 0xF701F6F6, 0x1C120E0E, 0xC2A36161, 0x6A5F3535, 0xAEF95757, 0x69D0B9B9, 0x17918686, 0x9958C1C1, 0x3A271D1D, 0x27B99E9E,
	0xD938E1E1, 0xEB13F8F8, 0x2BB39898, 0x22331111, 0xD2BB6969, 0xA970D9D9, 0x07898E8E, 0x33A79494, 0x2DB69B9B, 0x3C221E1E, 0x15928787, 0xC920E9E9, 0x8749CECE, 0xAAFF5555, 0x50782828, 0xA57ADFDF,
	0x038F8C8C, 0x59F8A1A1, 0x09808989, 0x1A170D0D, 0x65DABFBF, 0xD731E6E6, 0x84C64242, 0xD0B86868, 0x82C34141, 0x29B09999, 0x5A772D2D, 0x1E110F0F, 0x7BCBB0B0, 0xA8FC5454, 0x6DD6BBBB, 0x2C3A1616
};


__device__ __forceinline__
static void aes_round_s(const uint32_t sharedMemory[4][256], const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t x3, const uint32_t k0, uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3){

	y0 = __ldg(&d_AES0[__byte_perm(x0, 0, 0x4440)]);
	y3 = sharedMemory[1][__byte_perm(x0, 0, 0x4441)];
	y2 = sharedMemory[2][__byte_perm(x0, 0, 0x4442)];
	y1 = sharedMemory[3][__byte_perm(x0, 0, 0x4443)];

	y1 ^= __ldg(&d_AES0[__byte_perm(x1, 0, 0x4440)]);
	y0 ^= sharedMemory[1][__byte_perm(x1, 0, 0x4441)];
	y3 ^= sharedMemory[2][__byte_perm(x1, 0, 0x4442)];
	y2 ^= sharedMemory[3][__byte_perm(x1, 0, 0x4443)];

	y0 ^= k0;

	y2 ^= __ldg(&d_AES0[__byte_perm(x2, 0, 0x4440)]);
	y1 ^= sharedMemory[1][__byte_perm(x2, 0, 0x4441)];
	y0 ^= sharedMemory[2][__byte_perm(x2, 0, 0x4442)];
	y3 ^= __ldg(&d_AES3[__byte_perm(x2, 0, 0x4443)]);

	y3 ^= __ldg(&d_AES3[__byte_perm(x3, 0, 0x4440)]);
	y2 ^= sharedMemory[1][__byte_perm(x3, 0, 0x4441)];
	y1 ^= sharedMemory[2][__byte_perm(x3, 0, 0x4442)];
	y0 ^= sharedMemory[3][__byte_perm(x3, 0, 0x4443)];
}

__device__ __forceinline__
static void aes_round_s(const uint32_t sharedMemory[4][256], const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t x3, uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3){

	y0 = __ldg(&d_AES0[__byte_perm(x0, 0, 0x4440)]);
	y3 = sharedMemory[1][__byte_perm(x0, 0, 0x4441)];
	y2 = sharedMemory[2][__byte_perm(x0, 0, 0x4442)];
	y1 = sharedMemory[3][__byte_perm(x0, 0, 0x4443)];

	y1 ^= __ldg(&d_AES0[__byte_perm(x1, 0, 0x4440)]);
	y0 ^= sharedMemory[1][__byte_perm(x1, 0, 0x4441)];
	y3 ^= sharedMemory[2][__byte_perm(x1, 0, 0x4442)];
	y2 ^= __ldg(&d_AES3[__byte_perm(x1, 0, 0x4443)]);

	y2 ^= __ldg(&d_AES0[__byte_perm(x2, 0, 0x4440)]);
	y1 ^= sharedMemory[1][__byte_perm(x2, 0, 0x4441)];
	y0 ^= sharedMemory[2][__byte_perm(x2, 0, 0x4442)];
	y3 ^= sharedMemory[3][__byte_perm(x2, 0, 0x4443)];

	y3 ^= __ldg(&d_AES0[__byte_perm(x3, 0, 0x4440)]);
	y2 ^= sharedMemory[1][__byte_perm(x3, 0, 0x4441)];
	y1 ^= sharedMemory[2][__byte_perm(x3, 0, 0x4442)];
	y0 ^= sharedMemory[3][__byte_perm(x3, 0, 0x4443)];
}


__device__ __forceinline__
static void AES_ROUND_NOKEY_s(const uint32_t sharedMemory[4][256], uint4* x){

	uint32_t y0, y1, y2, y3;
	aes_round_s(sharedMemory, x->x, x->y, x->z, x->w, y0, y1, y2, y3);

	x->x = y0;
	x->y = y1;
	x->z = y2;
	x->w = y3;
}

__device__ __forceinline__
static void KEY_EXPAND_ELT_s(const uint32_t sharedMemory[4][256], uint32_t *k)
{

	uint32_t y0, y1, y2, y3;
	aes_round_s(sharedMemory, k[0], k[1], k[2], k[3], y0, y1, y2, y3);

	k[0] = y1;
	k[1] = y2;
	k[2] = y3;
	k[3] = y0;
}


__device__ __forceinline__
void aes_gpu_init128_s(uint32_t sharedMemory[4][256])
{
	/* each thread startup will fill 2 uint32 */
	uint2 temp = __ldg(&((uint2*)&d_AES1)[threadIdx.x]);

//	sharedMemory[0][(threadIdx.x << 1) + 0] = temp.x;
//	sharedMemory[0][(threadIdx.x << 1) + 1] = temp.y;
	sharedMemory[1][(threadIdx.x << 1) + 0] = (temp.x);
	sharedMemory[1][(threadIdx.x << 1) + 1] = (temp.y);
	sharedMemory[2][(threadIdx.x << 1) + 0] = ROL8(temp.x);
	sharedMemory[2][(threadIdx.x << 1) + 1] = ROL8(temp.y);
	sharedMemory[3][(threadIdx.x << 1) + 0] = ROL16(temp.x);
	sharedMemory[3][(threadIdx.x << 1) + 1] = ROL16(temp.y);
}


__device__ __forceinline__
static void round_3_7_11(const uint32_t sharedMemory[4][256], uint32_t* r, uint4 *p, uint4 &x){
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
static void round_4_8_12(const uint32_t sharedMemory[4][256], uint32_t* r, uint4 *p, uint4 &x){
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
__global__ __launch_bounds__(TPB,5) /* 64 registers with 128,8 - 72 regs with 128,7 */
void x11_shavite512_gpu_hash_64(const uint32_t threads, uint64_t *g_hash)
{
	__shared__ uint32_t sharedMemory[4][256];

	aes_gpu_init128_s(sharedMemory);

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint4 p[ 4];
	uint4 x;
	uint32_t r[32];

	// kopiere init-state
	const uint32_t state[16] = {
		0x72FCCDD8, 0x79CA4727, 0x128A077B, 0x40D55AEC,	0xD1901A06, 0x430AE307, 0xB29F5CD1, 0xDF07FBFC,
		0x8E45D73D, 0x681AB538, 0xBDE86578, 0xDD577E47,	0xE275EADE, 0x502D9FCD, 0xB9357178, 0x022A4B9A
	};
//	if (thread < threads)
//	{
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
		round_3_7_11(sharedMemory,r,p,x);
		
		
		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory,r,p,x);

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
		round_3_7_11(sharedMemory,r,p,x);

		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory,r,p,x);

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
		round_3_7_11(sharedMemory,r,p,x);
		
		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory,r,p,x);

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
//	}
}

__global__ __launch_bounds__(128, 5) /* 64 registers with 128,8 - 72 regs with 128,7 */
void x11_shavite512_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	__shared__ uint32_t sharedMemory[4][256];

	aes_gpu_init128_s(sharedMemory);

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint4 p[4];
	uint4 x;
	uint32_t r[32];

	// kopiere init-state
	const uint32_t state[16] = {
		0x72FCCDD8, 0x79CA4727, 0x128A077B, 0x40D55AEC, 0xD1901A06, 0x430AE307, 0xB29F5CD1, 0xDF07FBFC,
		0x8E45D73D, 0x681AB538, 0xBDE86578, 0xDD577E47, 0xE275EADE, 0x502D9FCD, 0xB9357178, 0x022A4B9A
	};


	//	if (thread < threads)
	//	{
	uint64_t *Hash = &g_hash[thread << 3];

	// fülle die Nachricht mit 64-byte (vorheriger Hash)
//	*(uint2x4*)&r[0] = __ldg4((uint2x4*)&Hash[0]);
//	*(uint2x4*)&r[8] = __ldg4((uint2x4*)&Hash[4]);
	const uint32_t nounce = startNounce + thread;

#pragma unroll 32
	for (int i = 0; i<20; i++) {
		r[i] = c_PaddedMessage80[i];
	}


	__syncthreads();

	*(uint2x4*)&p[0] = *(uint2x4*)&state[0];
	*(uint2x4*)&p[2] = *(uint2x4*)&state[8];

#pragma unroll 16
	for (int i = 20; i<32; i++)
	{
		r[i] = 0;
	}

	r[19] = cuda_swab32(nounce);
	r[20] = 0x80;
	r[27] = 0x2800000;
	r[31] = 0x2000000;

	/* round 0 */
	x = p[1] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[0] ^= x;


	x = p[3] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY_s(sharedMemory, &x);

	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x ^= *(uint4*)&r[24];



	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x ^= *(uint4*)&r[28];


	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[2] ^= x;
	// 1
	KEY_EXPAND_ELT_s(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	r[0] ^= 0x280;
	r[3] = ~r[3];

	x = p[0] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	KEY_EXPAND_ELT_s(sharedMemory, &r[4]);
	*(uint4*)&r[4] ^= *(uint4*)&r[0];
	x ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	KEY_EXPAND_ELT_s(sharedMemory, &r[8]);
	*(uint4*)&r[8] ^= *(uint4*)&r[4];
	x ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	KEY_EXPAND_ELT_s(sharedMemory, &r[12]);
	*(uint4*)&r[12] ^= *(uint4*)&r[8];
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[3] ^= x;
	KEY_EXPAND_ELT_s(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	x = p[2] ^ *(uint4*)&r[16];
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
	p[1] ^= x;
	*(uint4*)&r[0] ^= *(uint4*)&r[25];
	x = p[3] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY_s(sharedMemory, &x);

	r[4] ^= r[29]; r[5] ^= r[30];
	r[6] ^= r[31]; r[7] ^= r[0];

	x ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	*(uint4*)&r[8] ^= *(uint4*)&r[1];
	x ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	*(uint4*)&r[12] ^= *(uint4*)&r[5];
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[2] ^= x;
	*(uint4*)&r[16] ^= *(uint4*)&r[9];
	x = p[1] ^ *(uint4*)&r[16];
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

	p[0] ^= x;

	/* round 3, 7, 11 */
	round_3_7_11(sharedMemory, r, p, x);


	/* round 4, 8, 12 */
	round_4_8_12(sharedMemory, r, p, x);

	// 2
	KEY_EXPAND_ELT_s(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	x = p[0] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	KEY_EXPAND_ELT_s(sharedMemory, &r[4]);
	*(uint4*)&r[4] ^= *(uint4*)&r[0];
	r[7] ^= (~0x280);
	x ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	KEY_EXPAND_ELT_s(sharedMemory, &r[8]);
	*(uint4*)&r[8] ^= *(uint4*)&r[4];
	x ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	KEY_EXPAND_ELT_s(sharedMemory, &r[12]);
	*(uint4*)&r[12] ^= *(uint4*)&r[8];
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[3] ^= x;
	KEY_EXPAND_ELT_s(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	x = p[2] ^ *(uint4*)&r[16];
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
	p[1] ^= x;

	*(uint4*)&r[0] ^= *(uint4*)&r[25];
	x = p[3] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	r[4] ^= r[29];
	r[5] ^= r[30];
	r[6] ^= r[31];
	r[7] ^= r[0];
	x ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	*(uint4*)&r[8] ^= *(uint4*)&r[1];
	x ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	*(uint4*)&r[12] ^= *(uint4*)&r[5];
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[2] ^= x;
	*(uint4*)&r[16] ^= *(uint4*)&r[9];
	x = p[1] ^ *(uint4*)&r[16];
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
	p[0] ^= x;

	/* round 3, 7, 11 */
	round_3_7_11(sharedMemory, r, p, x);

	/* round 4, 8, 12 */
	round_4_8_12(sharedMemory, r, p, x);

	// 3
	KEY_EXPAND_ELT_s(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	x = p[0] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	KEY_EXPAND_ELT_s(sharedMemory, &r[4]);
	*(uint4*)&r[4] ^= *(uint4*)&r[0];
	x ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	KEY_EXPAND_ELT_s(sharedMemory, &r[8]);
	*(uint4*)&r[8] ^= *(uint4*)&r[4];
	x ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	KEY_EXPAND_ELT_s(sharedMemory, &r[12]);
	*(uint4*)&r[12] ^= *(uint4*)&r[8];
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[3] ^= x;
	KEY_EXPAND_ELT_s(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	x = p[2] ^ *(uint4*)&r[16];
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
	r[30] ^= 0x280;
	r[31] = ~r[31];
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[1] ^= x;

	*(uint4*)&r[0] ^= *(uint4*)&r[25];
	x = p[3] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	r[4] ^= r[29];
	r[5] ^= r[30];
	r[6] ^= r[31];
	r[7] ^= r[0];
	x ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	*(uint4*)&r[8] ^= *(uint4*)&r[1];
	x ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	*(uint4*)&r[12] ^= *(uint4*)&r[5];
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[2] ^= x;
	*(uint4*)&r[16] ^= *(uint4*)&r[9];
	x = p[1] ^ *(uint4*)&r[16];
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
	p[0] ^= x;

	/* round 3, 7, 11 */
	round_3_7_11(sharedMemory, r, p, x);

	/* round 4, 8, 12 */
	round_4_8_12(sharedMemory, r, p, x);

	/* round 13 */
	KEY_EXPAND_ELT_s(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	x = p[0] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	KEY_EXPAND_ELT_s(sharedMemory, &r[4]);
	*(uint4*)&r[4] ^= *(uint4*)&r[0];
	x ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	KEY_EXPAND_ELT_s(sharedMemory, &r[8]);
	*(uint4*)&r[8] ^= *(uint4*)&r[4];
	x ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	KEY_EXPAND_ELT_s(sharedMemory, &r[12]);
	*(uint4*)&r[12] ^= *(uint4*)&r[8];
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[3] ^= x;
	KEY_EXPAND_ELT_s(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	x = p[2] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	KEY_EXPAND_ELT_s(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	x ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	KEY_EXPAND_ELT_s(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	r[25] ^= 0x280;
	r[27] = ~r[27];
	x ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	KEY_EXPAND_ELT_s(sharedMemory, &r[28]);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[1] ^= x;

	*(uint2x4*)&Hash[0] = *(uint2x4*)&state[0] ^ *(uint2x4*)&p[2];
	*(uint2x4*)&Hash[4] = *(uint2x4*)&state[8] ^ *(uint2x4*)&p[0];
	//	}
}


__global__ __launch_bounds__(TPB, 5) /* 64 registers with 128,8 - 72 regs with 128,7 */
void x11_shavite512_gpu_hash_64_final(const uint32_t threads, uint64_t *g_hash, uint32_t* resNonce, const uint64_t target)
{
	__shared__ uint32_t sharedMemory[4][256];

	aes_gpu_init128_s(sharedMemory);

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
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[0] ^= x;
		x = p[3];
		x.x ^= 0x80;

		AES_ROUND_NOKEY_s(sharedMemory, &x);

		AES_ROUND_NOKEY_s(sharedMemory, &x);

		x.w ^= 0x02000000;
		AES_ROUND_NOKEY_s(sharedMemory, &x);

		x.w ^= 0x02000000;
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[2] ^= x;
		// 1
		KEY_EXPAND_ELT_s(sharedMemory, &r[0]);
		*(uint4*)&r[0] ^= *(uint4*)&r[28];
		r[0] ^= 0x200;
		r[3] = ~r[3];
		x = p[0] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[4]);
		*(uint4*)&r[4] ^= *(uint4*)&r[0];
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[8]);
		*(uint4*)&r[8] ^= *(uint4*)&r[4];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[3] ^= x;
		KEY_EXPAND_ELT_s(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[2] ^ *(uint4*)&r[16];
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
		p[1] ^= x;
		*(uint4*)&r[0] ^= *(uint4*)&r[25];
		x = p[3] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);

		r[4] ^= r[29]; r[5] ^= r[30];
		r[6] ^= r[31]; r[7] ^= r[0];

		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[8] ^= *(uint4*)&r[1];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[9];
		x = p[1] ^ *(uint4*)&r[16];
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

		p[0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory, r, p, x);


		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory, r, p, x);

		// 2
		KEY_EXPAND_ELT_s(sharedMemory, &r[0]);
		*(uint4*)&r[0] ^= *(uint4*)&r[28];
		x = p[0] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[4]);
		*(uint4*)&r[4] ^= *(uint4*)&r[0];
		r[7] ^= (~0x200);
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[8]);
		*(uint4*)&r[8] ^= *(uint4*)&r[4];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[3] ^= x;
		KEY_EXPAND_ELT_s(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[2] ^ *(uint4*)&r[16];
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
		p[1] ^= x;

		*(uint4*)&r[0] ^= *(uint4*)&r[25];
		x = p[3] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		r[4] ^= r[29];
		r[5] ^= r[30];
		r[6] ^= r[31];
		r[7] ^= r[0];
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[8] ^= *(uint4*)&r[1];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[9];
		x = p[1] ^ *(uint4*)&r[16];
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
		p[0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory, r, p, x);

		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory, r, p, x);

		// 3
		KEY_EXPAND_ELT_s(sharedMemory, &r[0]);
		*(uint4*)&r[0] ^= *(uint4*)&r[28];
		x = p[0] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[4]);
		*(uint4*)&r[4] ^= *(uint4*)&r[0];
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[8]);
		*(uint4*)&r[8] ^= *(uint4*)&r[4];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[3] ^= x;
		KEY_EXPAND_ELT_s(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[2] ^ *(uint4*)&r[16];
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
		r[30] ^= 0x200;
		r[31] = ~r[31];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[1] ^= x;

		*(uint4*)&r[0] ^= *(uint4*)&r[25];
		x = p[3] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		r[4] ^= r[29];
		r[5] ^= r[30];
		r[6] ^= r[31];
		r[7] ^= r[0];
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[8] ^= *(uint4*)&r[1];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[9];
		x = p[1] ^ *(uint4*)&r[16];
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
		p[0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory, r, p, x);

		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory, r, p, x);

		/* round 13 */
		KEY_EXPAND_ELT_s(sharedMemory, &r[0]);
		*(uint4*)&r[0] ^= *(uint4*)&r[28];
		x = p[0] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[4]);
		*(uint4*)&r[4] ^= *(uint4*)&r[0];
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[8]);
		*(uint4*)&r[8] ^= *(uint4*)&r[4];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[3] ^= x;
/*		KEY_EXPAND_ELT_s(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		r[25] ^= 0x200;
		r[27] ^= 0xFFFFFFFF;
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[1] ^= x;
*/
		//Hash[3] = 
		uint64_t test=(((uint64_t *)state)[3] ^ devectorize(make_uint2(p[3].z, p[3].w)));

		if (test <= target)
		{
			const uint32_t tmp = atomicExch(&resNonce[0], thread);
			if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}
	}
}


__host__
void x11_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	dim3 grid((threads + TPB-1)/TPB);
	dim3 block(TPB);

	// note: 128 threads minimum are required to init the shared memory array
	x11_shavite512_gpu_hash_64<<<grid, block>>>(threads, (uint64_t*)d_hash);
}

__host__
void x11_shavite512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, const uint64_t target, uint32_t* resNonce)
{
	dim3 grid((threads + TPB - 1) / TPB);
	dim3 block(TPB);

	// note: 128 threads minimum are required to init the shared memory array
	x11_shavite512_gpu_hash_64_final << <grid, block >> >(threads, (uint64_t*)d_hash,resNonce,target);
}

__host__
void x11_shavite512_setBlock_80(void *pdata)
{
	// Message with Padding
	// The nonce is at Byte 76.
	unsigned char PaddedMessage[128];
	memcpy(PaddedMessage, pdata, 80);
//	memset(PaddedMessage + 80, 0, 48);

	cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, 20 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

__host__
void x11_shavite512_cpu_hash_80(int thr_id, uint32_t threads,uint32_t startnonce, uint32_t *d_outputHash)
{
	const uint32_t threadsperblock = TPB;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	x11_shavite512_gpu_hash_80 << <grid, block >> >(threads, startnonce, (uint64_t *)d_outputHash);
}
