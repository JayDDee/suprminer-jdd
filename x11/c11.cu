extern "C"
{
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
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_x11.h"

void tribus_echo512_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target);
extern void x16_simd_echo512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t startnonce, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target);
extern void x11_cubehash_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x11_luffa512_cpu_hash_64_alexis(int thr_id, uint32_t threads, uint32_t *d_hash);

#include <stdio.h>
#include <memory.h>

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];

// Flax/Chaincoin C11 CPU Hash
extern "C" void c11hash(void *output, const void *input)
{
	unsigned char hash[128] = { 0 };

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

	sph_blake512_init(&ctx_blake);
	sph_blake512 (&ctx_blake, input, 80);
	sph_blake512_close(&ctx_blake, (void*) hash);

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512 (&ctx_bmw, (const void*) hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*) hash);

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512 (&ctx_groestl, (const void*) hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*) hash);

	sph_jh512_init(&ctx_jh);
	sph_jh512 (&ctx_jh, (const void*) hash, 64);
	sph_jh512_close(&ctx_jh, (void*) hash);

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512 (&ctx_keccak, (const void*) hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*) hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512 (&ctx_skein, (const void*) hash, 64);
	sph_skein512_close(&ctx_skein, (void*) hash);

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512 (&ctx_luffa, (const void*) hash, 64);
	sph_luffa512_close (&ctx_luffa, (void*) hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512 (&ctx_cubehash, (const void*) hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*) hash);

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512 (&ctx_shavite, (const void*) hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*) hash);

	sph_simd512_init(&ctx_simd);
	sph_simd512 (&ctx_simd, (const void*) hash, 64);
	sph_simd512_close(&ctx_simd, (void*) hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512 (&ctx_echo, (const void*) hash, 64);
	sph_echo512_close(&ctx_echo, (void*) hash);

	memcpy(output, hash, 32);
}

#ifdef _DEBUG
#define TRACE(algo) { \
	if (max_nonce == 1 && pdata[19] <= 1) { \
		uint32_t* debugbuf = NULL; \
		cudaMallocHost(&debugbuf, 8*sizeof(uint32_t)); \
		cudaMemcpy(debugbuf, d_hash[thr_id], 8*sizeof(uint32_t), cudaMemcpyDeviceToHost); \
		printf("X11 %s %08x %08x %08x %08x...\n", algo, swab32(debugbuf[0]), swab32(debugbuf[1]), \
			swab32(debugbuf[2]), swab32(debugbuf[3])); \
		cudaFreeHost(debugbuf); \
	} \
}
#else
#define TRACE(algo) {}
#endif

static bool init[MAX_GPUS] = { 0 };
//static bool use_compat_kernels[MAX_GPUS] = { 0 };

extern "C" int scanhash_c11(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];


	uint32_t default_throughput = 1 << 19;
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
	else if ((strstr(device_name[dev_id], "1070")) || (strstr(device_name[dev_id], "P104")))
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

	if (throughput > (1 << 22)) splitsimd = true;

	if (init[thr_id])
	{
		throughput = min(throughput, max_nonce - first_nonce);
	}
	throughput &= 0xFFFFFF00; //multiples of 128 due to cubehash_shavite & simd_echo kernels



	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0xff;

	if (!init[thr_id])
	{
		int dev_id = device_map[thr_id];
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);
	//	use_compat_kernels[thr_id] = (cuda_arch[dev_id] < 500);

		quark_blake512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		//quark_keccak512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		x11_luffaCubehash512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
//		if (use_compat_kernels[thr_id])
//			x11_echo512_cpu_init(thr_id, throughput);
//		if (x11_simd512_cpu_init(thr_id, throughput) != 0) {
//			return 0;
//		}
		if (splitsimd)
			x11_simd512_cpu_init(thr_id, throughput >> 4);
		else
			x11_simd512_cpu_init(thr_id, throughput);


		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], 64 * throughput), 0);
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], 2 * sizeof(uint32_t)));

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);
	cudaDeviceSynchronize();
	quark_blake512_cpu_setBlock_80(thr_id, endiandata);
//	if (use_compat_kernels[thr_id])
//		cuda_check_cpu_setTarget(ptarget);
//	else
	cudaMemset(d_resNonce[thr_id], 0xFF, 2 * sizeof(uint32_t));

	do {
		int order = 0;

		// Hash with CUDA
		quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
		quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		//quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]); order++;
		quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_luffa512_cpu_hash_64_alexis(thr_id, throughput, d_hash[thr_id]); order++;
		x11_cubehash_shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;


		if (splitsimd)
		{
			for (int j = 0; j < 256; j += 16)
			{
				x16_simd_echo512_cpu_hash_64_final(thr_id, throughput >> 4, (j >> 4)*(throughput >> 4), d_hash[thr_id] + (((throughput / 4)*j) / (sizeof(int))), d_resNonce[thr_id], ((uint64_t *)ptarget)[3]);
			}
		}
		else
		{
			x16_simd_echo512_cpu_hash_64_final(thr_id, throughput, 0, d_hash[thr_id], d_resNonce[thr_id], ((uint64_t *)ptarget)[3]);
		}


		cudaMemcpy(&work->nonces[0], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (work->nonces[0] != UINT32_MAX)
		{
			if (opt_benchmark) gpulog(LOG_BLUE, dev_id, "found");

			uint32_t _ALIGN(64) vhash[8];
			const uint32_t Htarg = ptarget[7];
			const uint32_t startNounce = pdata[19];
			work->nonces[0] += startNounce;
			be32enc(&endiandata[19], work->nonces[0]);
			c11hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != UINT32_MAX) {
					work->nonces[1] += startNounce;
					be32enc(&endiandata[19], work->nonces[1]);
					c11hash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				cudaDeviceSynchronize();
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				cudaMemset(d_resNonce[thr_id], 0xFF, 2 * sizeof(uint32_t));
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);
	cudaDeviceSynchronize();

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_c11(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	x11_simd512_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
