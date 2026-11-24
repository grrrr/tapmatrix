/* tapmatrix~: what differs between the platforms besides the DSP kernels.
 *   CYCLES()        a free-running counter for "stats" (only differences are used)
 *   CYCLES_PER_US   its ticks per microsecond
 *   fastmem(n), fastfree(p, n)   scratch memory touched for every tap and block
 */
#ifndef TAPMATRIX_PLATFORM_H
#define TAPMATRIX_PLATFORM_H

#include <stdint.h>
#include <stddef.h>

#if defined(ESP_PLATFORM)
/* ESP32: the CPU cycle counter; scratch in internal RAM, not PSRAM (getbytes) */
#include "sdkconfig.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_memory_utils.h"
#define CYCLES() esp_cpu_get_cycle_count()
#define CYCLES_PER_US ((float)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ)
static void *fastmem(size_t n) { void *p = heap_caps_calloc(1, n, MALLOC_CAP_INTERNAL); return p ? p : getbytes(n); }
static void fastfree(void *p, size_t n) { if (esp_ptr_internal(p)) heap_caps_free(p); else freebytes(p, n); }

#elif defined(_WIN32)
/* Windows: the performance counter, in nanoseconds (MinGW has no clock_gettime
   without winpthread) */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static inline uint32_t ns_now(void)    /* wraps after 4.3 s: only differences are used */
{
    static double ns_per_tick = 0;
    LARGE_INTEGER t;
    if (ns_per_tick == 0) {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        ns_per_tick = 1e9 / (double)f.QuadPart;
    }
    QueryPerformanceCounter(&t);
    return (uint32_t)(uint64_t)((double)t.QuadPart * ns_per_tick);
}
#define CYCLES() ns_now()
#define CYCLES_PER_US 1000.f              /* here: nanoseconds */
#define fastmem getbytes
#define fastfree freebytes

#else
/* POSIX: the monotonic clock, in nanoseconds */
#include <time.h>
static inline uint32_t ns_now(void)    /* wraps after 4.3 s: only differences are used */
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000000000u + ts.tv_nsec);
}
#define CYCLES() ns_now()
#define CYCLES_PER_US 1000.f              /* here: nanoseconds */
#define fastmem getbytes
#define fastfree freebytes
#endif

#endif
