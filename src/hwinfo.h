#ifndef BASI_HWINFO_H
#define BASI_HWINFO_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    bool     has_gpu;
    char     gpu_name[256];   /* matches VK_MAX_PHYSICAL_DEVICE_NAME_SIZE */
    uint32_t vendor_id;        /* PCI vendor: 0x8086 Intel, 0x1002 AMD, 0x10DE NVIDIA */
    uint64_t vram_total_mb;    /* DEVICE_LOCAL heap size for the chosen GPU */

    /* Live VRAM accounting from VK_EXT_memory_budget (the only mechanism that
     * works for the Intel Arc dGPU — its i915/xe driver does not expose
     * mem_info_vram_* in sysfs the way amdgpu does). Valid only when
     * vram_budget_known is true; otherwise callers must fall back to a
     * fraction of vram_total_mb. */
    bool     vram_budget_known;
    uint64_t vram_budget_mb;   /* OS-granted budget this process may allocate */
    uint64_t vram_used_mb;     /* bytes already resident on the heap (any process) */
    uint64_t vram_avail_mb;    /* free right now = budget - used, clamped >= 0 */

    uint64_t ram_total_mb;     /* System RAM */
} HwInfo;

/* One-shot hardware probe via Vulkan + sysconf. Picks the first DISCRETE_GPU,
 * falling back to the first non-CPU device. Returns has_gpu=false if Vulkan
 * cannot enumerate any usable device. When the chosen device supports
 * VK_EXT_memory_budget, live VRAM usage/availability is filled in. */
HwInfo hw_probe(void);

/* Vendor short label ("Intel", "AMD", "NVIDIA", or "GPU"). */
const char *hw_vendor_label(uint32_t vendor_id);

#endif /* BASI_HWINFO_H */
