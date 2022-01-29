/**
 * X16R algorithm (X16 with Randomized chain order)
 *
 * tpruvot 2018 - GPL code
 * Sp may 2018
 */

#include <stdio.h>
#include <memory.h>
#include <unistd.h>

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"

#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"

#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"
#include "sph/sph_shabal.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_sha2.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_x16r.h"

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t h_resNonce[MAX_GPUS][4];

extern void quark_bmw512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash, uint32_t *resNonce, const uint64_t target);
extern void x11_luffa512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint64_t target, uint32_t *d_resNonce);
extern void x11_echo512_cpu_hash_64_final_sp(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target);
extern void x14_shabal512_cpu_hash_64_final_sp(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target);
extern void x17_sha512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *resNonce, const uint64_t target);
extern void x11_shavite512_cpu_hash_64_sp_final(int thr_id, uint32_t threads, uint32_t *d_hash, const uint64_t target, uint32_t* resNonce);
extern void quark_groestl512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target);
extern void quark_skein512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint64_t target, uint32_t *d_resNonce);
extern void x11_simd512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t startnonce, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target);
extern void x16_simd_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x11_cubehash_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void quark_blake512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_outputHash, uint32_t *resNonce, const uint64_t target);
extern void x13_fugue512_cpu_hash_64_final_sp(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target);
extern void x16_simd_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x16_simd_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x16_simd_whirlpool512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x13_hamsi512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target);
extern void x16_simd_jh512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x15_whirlpool_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target);


extern void x13_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);


static uint32_t *d_hash[MAX_GPUS];

enum Algo {
	BLAKE = 0,
	BMW,
	GROESTL,
	JH,
	KECCAK,
	SKEIN,
	LUFFA,
	CUBEHASH,
	SHAVITE,
	SIMD,
	ECHO,
	HAMSI,
	FUGUE,
	SHABAL,
	WHIRLPOOL,
	SHA512,
	HASH_FUNC_COUNT
};

static const char* algo_strings[] = {
	"blake",
	"bmw512",
	"groestl",
	"jh512",
	"keccak",
	"skein",
	"luffa",
	"cube",
	"shavite",
	"simd",
	"echo",
	"hamsi",
	"fugue",
	"shabal",
	"whirlpool",
	"sha512",
	NULL
};

static __thread uint32_t s_ntime = UINT32_MAX;
static __thread bool s_implemented = false;
static __thread char hashOrder[HASH_FUNC_COUNT + 1] = { 0 };

static void getAlgoString(const uint32_t* prevblock, char *output)
{
	char *sptr = output;
	uint8_t* data = (uint8_t*)prevblock;

	for (uint8_t j = 0; j < HASH_FUNC_COUNT; j++) {
		uint8_t b = (15 - j) >> 1; // 16 ascii hex chars, reversed
		uint8_t algoDigit = (j & 1) ? data[b] & 0xF : data[b] >> 4;
		if (algoDigit >= 10)
			sprintf(sptr, "%c", 'A' + (algoDigit - 10));
		else
			sprintf(sptr, "%u", (uint32_t) algoDigit);
		sptr++;
	}
	*sptr = '\0';
}
static uint8_t get_first_algo(const uint32_t* prevblock) 
{
	uint8_t* data = (uint8_t*)prevblock;
	return data[7] >> 4;
}

// X16R CPU Hash (Validation)
extern "C" void x16r_hash(void *output, const void *input)
{
	unsigned char _ALIGN(64) hash[128];

	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_skein512_context ctx_skein;
	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;
	sph_hamsi512_context ctx_hamsi;
	sph_fugue512_context ctx_fugue;
	sph_shabal512_context ctx_shabal;
	sph_whirlpool_context ctx_whirlpool;
	sph_sha512_context ctx_sha512;

	void *in = (void*) input;
	int size = 80;

	uint32_t *in32 = (uint32_t*) input;
	getAlgoString(&in32[1], hashOrder);

	char elem = hashOrder[0];
	uint8_t algo = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

	for (int i = 0; i < 16; i++)
	{
		switch (algo) {
		case BLAKE:
			sph_blake512_init(&ctx_blake);
			sph_blake512(&ctx_blake, in, size);
			sph_blake512_close(&ctx_blake, hash);
			break;
		case BMW:
			sph_bmw512_init(&ctx_bmw);
			sph_bmw512(&ctx_bmw, in, size);
			sph_bmw512_close(&ctx_bmw, hash);
			break;
		case GROESTL:
			sph_groestl512_init(&ctx_groestl);
			sph_groestl512(&ctx_groestl, in, size);
			sph_groestl512_close(&ctx_groestl, hash);
			break;
		case SKEIN:
			sph_skein512_init(&ctx_skein);
			sph_skein512(&ctx_skein, in, size);
			sph_skein512_close(&ctx_skein, hash);
			break;
		case JH:
			sph_jh512_init(&ctx_jh);
			sph_jh512(&ctx_jh, in, size);
			sph_jh512_close(&ctx_jh, hash);
			break;
		case KECCAK:
			sph_keccak512_init(&ctx_keccak);
			sph_keccak512(&ctx_keccak, in, size);
			sph_keccak512_close(&ctx_keccak, hash);
			break;
		case LUFFA:
			sph_luffa512_init(&ctx_luffa);
			sph_luffa512(&ctx_luffa, in, size);
			sph_luffa512_close(&ctx_luffa, hash);
			break;
		case CUBEHASH:
			sph_cubehash512_init(&ctx_cubehash);
			sph_cubehash512(&ctx_cubehash, in, size);
			sph_cubehash512_close(&ctx_cubehash, hash);
			break;
		case SHAVITE:
			sph_shavite512_init(&ctx_shavite);
			sph_shavite512(&ctx_shavite, in, size);
			sph_shavite512_close(&ctx_shavite, hash);
			break;
		case SIMD:
			sph_simd512_init(&ctx_simd);
			sph_simd512(&ctx_simd, in, size);
			sph_simd512_close(&ctx_simd, hash);
			break;
		case ECHO:
			sph_echo512_init(&ctx_echo);
			sph_echo512(&ctx_echo, in, size);
			sph_echo512_close(&ctx_echo, hash);
			break;
		case HAMSI:
			sph_hamsi512_init(&ctx_hamsi);
			sph_hamsi512(&ctx_hamsi, in, size);
			sph_hamsi512_close(&ctx_hamsi, hash);
			break;
		case FUGUE:
			sph_fugue512_init(&ctx_fugue);
			sph_fugue512(&ctx_fugue, in, size);
			sph_fugue512_close(&ctx_fugue, hash);
			break;
		case SHABAL:
			sph_shabal512_init(&ctx_shabal);
			sph_shabal512(&ctx_shabal, in, size);
			sph_shabal512_close(&ctx_shabal, hash);
			break;
		case WHIRLPOOL:
			sph_whirlpool_init(&ctx_whirlpool);
			sph_whirlpool(&ctx_whirlpool, in, size);
			sph_whirlpool_close(&ctx_whirlpool, hash);
			break;
		case SHA512:
			sph_sha512_init(&ctx_sha512);
			sph_sha512(&ctx_sha512,(const void*) in, size);
			sph_sha512_close(&ctx_sha512,(void*) hash);
			break;
		}
		in = (void*) hash;
		size = 64;

		if (i!=15)
		{
			elem = hashOrder[i+1];
			algo = elem >= 'A' ? elem - 'A' + 10 : elem - '0';
		}

	}
	memcpy(output, hash, 32);
}

void whirlpool_midstate(void *state, const void *input)
{
	sph_whirlpool_context ctx;

	sph_whirlpool_init(&ctx);
	sph_whirlpool(&ctx, input, 64);

	memcpy(state, ctx.state, 64);
}

static bool init[MAX_GPUS] = { 0 };

//#define _DEBUG
#define _DEBUG_PREFIX "x16r-"
#include "cuda_debug.cuh"

//static int algo80_tests[HASH_FUNC_COUNT] = { 0 };
//static int algo64_tests[HASH_FUNC_COUNT] = { 0 };
static int algo80_fails[HASH_FUNC_COUNT] = { 0 };

extern "C" int scanhash_x16r(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];
/*	int intensity = (device_sm[dev_id] > 500 ) ? 20 : 18;
	if (strstr(device_name[dev_id], "GTX 1080")) intensity = 21;
	if (strstr(device_name[dev_id], "GTX 1060")) intensity = 19;
	if (strstr(device_name[dev_id], "GTX 1050")) intensity = 18;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
*/

	if (opt_benchmark) {
		((uint32_t*)ptarget)[7] = 0x0ff;
//		((uint32_t*)pdata)[1] = 0xF9E9EA98;
//		((uint32_t*)pdata)[2] = 0x76543210;
		((uint32_t*)pdata)[1] = 0x88888888;
		((uint32_t*)pdata)[2] = 0x88888888;

		//		94E3A654 CBD9B14B

		//		((uint32_t*)pdata)[1] = 0x01234567;
		//		((uint32_t*)pdata)[2] = 0x22222222;
		//		((uint32_t*)pdata)[1] = 0x01234567;
		//		((uint32_t*)pdata)[2] = 0x89ABCDEF;
		//((uint8_t*)pdata)[8] = 0x90; // hashOrder[0] = '9'; for simd 80 + blake512 64
		//((uint8_t*)pdata)[8] = 0xA0; // hashOrder[0] = 'A'; for echo 80 + blake512 64
		//((uint8_t*)pdata)[8] = 0xB0; // hashOrder[0] = 'B'; for hamsi 80 + blake512 64
		//((uint8_t*)pdata)[8] = 0xC0; // hashOrder[0] = 'C'; for fugue 80 + blake512 64
		//((uint8_t*)pdata)[8] = 0xE0; // hashOrder[0] = 'E'; for whirlpool 80 + blake512 64
	}

	uint32_t _ALIGN(64) endiandata[20];

	for (int k = 0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	uint32_t ntime = swab32(pdata[17]);
	if (s_ntime != ntime)
	{
		getAlgoString(&endiandata[1], hashOrder);
		s_ntime = ntime;
		if (thr_id==0)
		{
			applog(LOG_INFO, "hash order %s (%08x)", hashOrder, ntime);
			cudaDeviceSynchronize();
		}
	}

	uint32_t default_throughput=1<<19;
	bool splitsimd = true;
	bool merge = false;
	if ((strstr(device_name[dev_id], "1060")) || (strstr(device_name[dev_id], "P106")))
	{
		default_throughput = (1 << 21);
		splitsimd = false;
	}
	else if ((strstr(device_name[dev_id], "970") || (strstr(device_name[dev_id], "980"))))
	{
		default_throughput = (1 << 21);
		splitsimd = false;
	}
	else if ((strstr(device_name[dev_id], "1050")))
	{
		default_throughput = 1 << 20;
		splitsimd = false;
	}
	else if ((strstr(device_name[dev_id], "950")))
	{
		default_throughput = 1 << 20;
		splitsimd = false;
	}
	else if ((strstr(device_name[dev_id], "960")))
	{
		default_throughput = 1 << 20;
		splitsimd = false;
	}
	else if ((strstr(device_name[dev_id], "750")))
	{
		default_throughput = 1 << 20;
		splitsimd = false;
	}
	else if ((strstr(device_name[dev_id], "166")) || (strstr(device_name[dev_id], "20")))
	{
		default_throughput = (1 << 24); //53686272; //1 << 20
		merge = true;
	}
	else if (strstr(device_name[dev_id], "1070") || (strstr(device_name[dev_id], "P104")))
	{
		default_throughput = (1 << 24); //53686272; //1 << 20
		merge = true;
	}
	else if ((strstr(device_name[dev_id], "1080 Ti")) || (strstr(device_name[dev_id], "1080")) || (strstr(device_name[dev_id], "P102")))
	{
		default_throughput = (1 << 24); //53686272; //1 << 20
		merge = true;
	}
	uint32_t throughput = cuda_default_throughput(thr_id, default_throughput);

	if (throughput > (1<<22)) splitsimd = true;

	if (init[thr_id])
	{
		throughput = min(throughput, max_nonce - first_nonce);
	}
	throughput &= 0xFFFFFF00; //multiples of 128 due to cubehash_shavite & simd_echo kernels

	if (!init[thr_id])
	{

		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) 
		{
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}

		gpulog(LOG_INFO, thr_id, "%s Intensity: %g ", device_name[dev_id], throughput2intensity(throughput));


		quark_blake512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		if (splitsimd)
		{
			x11_simd512_cpu_init(thr_id, (throughput >> 4));
		}
		else
		{
			x11_simd512_cpu_init(thr_id, (throughput));
		}
		x13_hamsi512_cpu_init(thr_id, throughput);
		x16_fugue512_cpu_init(thr_id, throughput);
		x15_whirlpool_cpu_init(thr_id, throughput, 0);
		x16_whirlpool512_init(thr_id, throughput);

		x11_luffa512_cpu_init(thr_id, throughput); // 64
		x16_echo512_cuda_init(thr_id, throughput);
		x13_fugue512_cpu_init(thr_id, throughput);
		x16_fugue512_cpu_init(thr_id, throughput);
		x14_shabal512_cpu_init(thr_id, throughput);


		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), 0);
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], 2 * sizeof(uint32_t)));

		cuda_check_cpu_init(thr_id, throughput);
		sleep(2);
		init[thr_id] = true;
	}
	

	cuda_check_cpu_setTarget(ptarget);

	char elem = hashOrder[0];
	const uint8_t algo80 = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

	switch (algo80) 
	{
		case BLAKE:
			quark_blake512_cpu_setBlock_80(thr_id, endiandata);
			break;
		case BMW:
			quark_bmw512_cpu_setBlock_80(endiandata);
			break;
		case GROESTL:
			groestl512_setBlock_80(thr_id, endiandata);
			break;
		case JH:
			jh512_setBlock_80(thr_id, endiandata);
			break;
		case KECCAK:
			keccak512_setBlock_80(thr_id, endiandata);
			break;
		case SKEIN:
			skein512_cpu_setBlock_80((void*)endiandata);
			break;
		case LUFFA:
			qubit_luffa512_cpu_setBlock_80_alexis((void*)endiandata);
			break;
		case CUBEHASH:
			cubehash512_setBlock_80(thr_id, endiandata);
			break;
		case SHAVITE:
			x16_shavite512_setBlock_80((void*)endiandata);
			break;
		case SIMD:
			x16_simd512_setBlock_80((void*)endiandata);
			break;
		case ECHO:
			x16_echo512_setBlock_80((void*)endiandata);
			break;
		case HAMSI:
			x16_hamsi512_setBlock_80((uint64_t*)endiandata);
			break;
		case FUGUE:
			x16_fugue512_setBlock_80((void*)pdata);
			break;
		case SHABAL:
			x16_shabal512_setBlock_80((void*)endiandata);
			break;
		case WHIRLPOOL:
			x16_whirlpool512_setBlock_80((void*)endiandata);
			break;
		case SHA512:
			x16_sha512_setBlock_80(endiandata);
			break;
		default: {
			if (!thr_id)
				applog(LOG_WARNING, "kernel %s %c unimplemented, order %s", algo_strings[algo80], elem, hashOrder);
			s_implemented = false;
			sleep(5);
			return -1;
		}
	}


	int warn = 0;

	do {
		int order = 0;
		CUDA_SAFE_CALL(cudaMemset(d_resNonce[thr_id], 0xFFFFFFFF, 2 * sizeof(uint32_t)));

		uint32_t start = pdata[19];
		//uint32_t foundNonce;
		bool addstart = false;



		// Hash with CUDA

		switch (algo80) 
		{
			case BLAKE:
				quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				break;
			case BMW:
				quark_bmw512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
				break;
			case GROESTL:
				groestl512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				break;
			case JH:
				jh512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				break;
			case KECCAK:
				keccak512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				break;
			case SKEIN:
				skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], 1); order++;
				break;
			case LUFFA:
				qubit_luffa512_cpu_hash_80_alexis(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				break;
			case CUBEHASH:
				cubehash512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				break;
			case SHAVITE:
				x16_shavite512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
				break;
			case SIMD:
				x16_simd512_cuda_hash_80_sp(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				break;
			case ECHO:
				x16_echo512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				break;
			case HAMSI:
				x16_hamsi512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				break;
			case FUGUE:
				x16_fugue512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				break;
			case SHABAL:
				x16_shabal512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				break;
			case WHIRLPOOL:
				x16_whirlpool512_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				break;
			case SHA512:
				x16_sha512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				break;
		}
		uint8_t algo;

	
		for (int i = 1; i < 16; i++)
		{
			elem = hashOrder[i];
			uint8_t algo64 = elem >= 'A' ? elem - 'A' + 10 : elem - '0';
			uint8_t nextalgo = -1;
			
			if (i < 15)
			{
				const char elem2 = hashOrder[i + 1];
				nextalgo = elem2 >= 'A' ? elem2 - 'A' + 10 : elem2 - '0';
			}
			switch (algo64) {
			case BLAKE:
				if (i == 15)
				{
					quark_blake512_cpu_hash_64_final(thr_id, throughput, NULL, d_hash[thr_id], d_resNonce[thr_id], ((uint64_t *)ptarget)[3]);
					CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
					work->nonces[0] = h_resNonce[thr_id][0];
					addstart = true;
				}
				else
				{

					quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				}

				break;
			case BMW:
				if (i == 15)
				{
					quark_bmw512_cpu_hash_64_final(thr_id, throughput, NULL, d_hash[thr_id], d_resNonce[thr_id],((uint64_t *)ptarget)[3]);
					CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
					work->nonces[0] = h_resNonce[thr_id][0];
					addstart = true;
				}
				else
				{
					quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				}

				break;
			case GROESTL:
				if (i == 15)
				{
					quark_groestl512_cpu_hash_64_final(thr_id, throughput,d_hash[thr_id], d_resNonce[thr_id], ((uint64_t *)ptarget)[3]);
					CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
					work->nonces[0] = h_resNonce[thr_id][0];
					addstart = true;
				}
				else
				{
					quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				}
				break;
			case JH:
				quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				break;
			case KECCAK:
				//quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				quark_keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]); order++;
				break;
			case SKEIN:
				if (i == 15)
				{
					quark_skein512_cpu_hash_64_final(thr_id, throughput, d_hash[thr_id], ((uint64_t *)ptarget)[3], d_resNonce[thr_id]);
					CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
					work->nonces[0] = h_resNonce[thr_id][0];
					addstart = true;
				}
				else
				{
					quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				}
				break;
			case LUFFA:
				if (i == 15)
				{
					x11_luffa512_cpu_hash_64_final(thr_id, throughput, d_hash[thr_id], ((uint64_t *)ptarget)[3], d_resNonce[thr_id]);
					CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
					work->nonces[0] = h_resNonce[thr_id][0];
					addstart = true;
				}
				else
				{
					x11_luffa512_cpu_hash_64_alexis(thr_id, throughput, d_hash[thr_id]); order++;
				}
				break;
			case CUBEHASH:
				if (nextalgo == SHAVITE)
				{
					x11_cubehash_shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
					i = i + 1;
				}
				else
				{
					x11_cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
				}
				break;
			case SHAVITE:
				if (i == 15)
				{
					x11_shavite512_cpu_hash_64_sp_final(thr_id, throughput, d_hash[thr_id], ((uint64_t *)ptarget)[3], d_resNonce[thr_id]);
					CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
					work->nonces[0] = h_resNonce[thr_id][0];
					addstart = true;
				}
				else
				{
					x11_shavite512_cpu_hash_64_sp(thr_id, throughput, d_hash[thr_id]); order++;
				}
				break;
			case SIMD:
				if (!splitsimd)
				{
					if(i== 15)
					{
						x11_simd512_cpu_hash_64_final(thr_id, throughput, 0, d_hash[thr_id], d_resNonce[thr_id], ((uint64_t *)ptarget)[3]);
						CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
						work->nonces[0] = h_resNonce[thr_id][0];
						addstart = true;
					} else if (nextalgo == ECHO)
					{
						x16_simd_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
						i = i + 1;
					}/*
					else if (nextalgo == WHIRLPOOL)
					{
						x16_simd_whirlpool512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
						i = i + 1;
					}*/
					else if (nextalgo == HAMSI)
					{
						x16_simd_hamsi512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
						i = i + 1;
					}/*
					else if (nextalgo == FUGUE)
					{
						x16_simd_fugue512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
						i = i + 1;
					}*/
					else
					{
						x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order);
					}
				}
				else
				{
					if (i == 15)
					{
						for (int j = 0; j < 256; j += 16)
						{
							x11_simd512_cpu_hash_64_final(thr_id, throughput >> 4, (j >> 4)*(throughput >> 4), d_hash[thr_id] + (((throughput / 4)*j) / (sizeof(int))), d_resNonce[thr_id], ((uint64_t *)ptarget)[3]);
						}
						CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
						work->nonces[0] = h_resNonce[thr_id][0];
						addstart = true;
					}
					if (nextalgo == ECHO)
					{
						for (int j = 0; j < 256; j += 16)
						{
							x16_simd_echo512_cpu_hash_64(thr_id, throughput >> 4, d_hash[thr_id] + (((throughput / 4)*j) / (sizeof(int))));
						}
						i = i + 1;
					}
					/*else if (nextalgo == WHIRLPOOL)
					{
						for (int j = 0; j < 256; j += 16)
						{
							x16_simd_whirlpool512_cpu_hash_64(thr_id, throughput >> 4, d_hash[thr_id] + (((throughput / 4)*j) / (sizeof(int))));
						}
						i = i + 1;
					}*/
					else if (nextalgo == HAMSI)
					{
						for (int j = 0; j < 256; j += 16)
						{
							x16_simd_hamsi512_cpu_hash_64(thr_id, throughput >> 4, d_hash[thr_id] + (((throughput / 4)*j) / (sizeof(int))));
						}
						i = i + 1;
					}/* Slower with cuda 9.2 / latest driver. 40% slower
					else if (nextalgo == FUGUE)
					{
						for (int j = 0; j < 256; j += 16)
						{
							x16_simd_fugue512_cpu_hash_64(thr_id, throughput >> 4, d_hash[thr_id] + (((throughput / 4)*j) / (sizeof(int))));
						}
						
						i = i + 1;
					}*/
					else
					{
						for (int j = 0; j < 256; j += 16)
						{
							x11_simd512_cpu_hash_64(thr_id, throughput >> 4, pdata[19], NULL, d_hash[thr_id] + (((throughput / 4)*j) / (sizeof(int))),order);
						}
					}
				}
				break;
			case ECHO:
				if (i == 15)
				{
					x11_echo512_cpu_hash_64_final_sp(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id], ((uint64_t *)ptarget)[3]);
					CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
					work->nonces[0] = h_resNonce[thr_id][0];
					addstart = true;
				}
				else
				{
					x11_echo512_cpu_hash_64_sp(thr_id, throughput, d_hash[thr_id]); order++;
				}
				break;
			case HAMSI:
				if (i == 15)
				{
					x13_hamsi512_cpu_hash_64_final(thr_id,throughput, d_hash[thr_id], d_resNonce[thr_id], ((uint64_t *)ptarget)[3]);
					CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
					work->nonces[0] = h_resNonce[thr_id][0];
					addstart = true;
				}
				else
				{
					x13_hamsi512_cpu_hash_64(thr_id,throughput, d_hash[thr_id]);
				}

				break;
			case FUGUE:
				if (i == 15)
				{
					x13_fugue512_cpu_hash_64_final_sp(thr_id, throughput,d_hash[thr_id], d_resNonce[thr_id], ((uint64_t *)ptarget)[3]);
					CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
					work->nonces[0] = h_resNonce[thr_id][0];
					addstart = true;
				}
				else
				{
					x13_fugue512_cpu_hash_64_sp(thr_id, throughput, d_hash[thr_id]);
				}
				break;
			case SHABAL:
				if (i == 15)
				{
					x14_shabal512_cpu_hash_64_final_sp(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id], ((uint64_t *)ptarget)[3]);
					CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
					work->nonces[0] = h_resNonce[thr_id][0];
					addstart = true;
				}
				else
				{
					x14_shabal512_cpu_hash_64_sp(thr_id, throughput, d_hash[thr_id]); order++;
				}
				break;
			case WHIRLPOOL:
				if (i == 15)
				{
					x15_whirlpool_cpu_hash_64_final(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id], ((uint64_t *)ptarget)[3]);
					CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
					work->nonces[0] = h_resNonce[thr_id][0];
					addstart = true;
				}
				else
				{
					x15_whirlpool_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
				}
				break;
			case SHA512:
				if (i == 15)
				{
					x17_sha512_cpu_hash_64_final(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id], ((uint64_t *)ptarget)[3]);
					CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
					work->nonces[0] = h_resNonce[thr_id][0];
					addstart = true;
				}
				else
				{
					x17_sha512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
				}
				break;
			}
		}

		if (!addstart)
		{
			work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		}

		*hashes_done = pdata[19] - first_nonce + throughput;
		if (work->nonces[0] != UINT32_MAX)
		{
			if (opt_benchmark) gpulog(LOG_BLUE, dev_id, "found");

			if (addstart) work->nonces[0] += pdata[19];

			/*if (work_restart[thr_id].restart)
			{
				pdata[19] += throughput;
				cudaDeviceSynchronize();
				gpulog(LOG_WARNING, thr_id, "restart");
				goto out;
			}*/

			uint32_t _ALIGN(64) vhash64[8];
			//			const uint32_t Htarg = ptarget[7];
			be32enc(&endiandata[19], work->nonces[0]);
			x16r_hash(vhash64, endiandata);

			if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget))
			{
				int res = 1;
				// check if there was some other ones...
				//uint32_t secNonce = UINT32_MAX;

				if (addstart && (h_resNonce[thr_id][1] != UINT32_MAX))
				{
					work->nonces[1] = h_resNonce[thr_id][1] + pdata[19];
				}
				else
				{
					work->nonces[1] = UINT32_MAX;
				}
				if (!addstart)
				{
					work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
					if (work->nonces[1] == 0) work->nonces[1] = UINT32_MAX;
				}

				/*
				uint32_t secNonce2 = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash2[thr_id], 1);

				if (secNonce != secNonce2)
				{
				gpulog(LOG_WARNING, thr_id, "result %08x, %08x ", secNonce, secNonce2);

				if (secNonce == 0xffffffff || secNonce2 == 0xffffffff)
				gpulog(LOG_WARNING, thr_id, "second broken");
				}
				*/

				work_set_target_ratio(work, vhash64);
				*hashes_done = pdata[19] - first_nonce + throughput;
				pdata[19] = work->nonces[0];
				if (work->nonces[1] != UINT32_MAX)
				{
					//					gpulog(LOG_BLUE, dev_id, "found2");

					//					if(!opt_quiet)
					//						gpulog(LOG_BLUE,dev_id,"Found 2nd nonce: %08x", secNonce);
					be32enc(&endiandata[19], work->nonces[1]);
					pdata[21] = work->nonces[1];
					x16r_hash(vhash64, endiandata);
					if (vhash64[7] > ptarget[7] && !fulltest(vhash64, ptarget)) gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[1]);
					/*if (bn_hash_target_ratio(vhash64, ptarget) > work->shareratio[0])
					{
						work_set_target_ratio(work, vhash64);
						xchg(pdata[19], pdata[21]);
					}
					*/
					res++;
				}
				cudaDeviceSynchronize();
				return res;
		}
			else
			{
				if (vhash64[7] != ptarget[7]) gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
			}
			}
		pdata[19] += throughput;
		} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > (uint64_t)throughput + pdata[19]));
		cudaDeviceSynchronize();
	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_x16r(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);

	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	x11_simd512_cpu_free(thr_id);
	x13_fugue512_cpu_free(thr_id);
	x16_fugue512_cpu_free(thr_id); // to merge with x13_fugue512 ?
	x15_whirlpool_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
