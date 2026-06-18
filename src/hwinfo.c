#include "hwinfo.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

const char *hw_vendor_label(uint32_t vendor_id) {
    switch (vendor_id) {
    case 0x8086: return "Intel";
    case 0x1002: return "AMD";
    case 0x10DE: return "NVIDIA";
    default:     return "GPU";
    }
}

HwInfo hw_probe(void) {
    HwInfo info = {0};

    long pages     = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        info.ram_total_mb = (uint64_t)pages * (uint64_t)page_size / (1024ULL * 1024ULL);
    }

    VkApplicationInfo app = {
        .sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "basi-cli",
        /* 1.1 makes vkGetPhysicalDeviceMemoryProperties2 a core entry point,
         * which we need to chain the VK_EXT_memory_budget query for live usage. */
        .apiVersion       = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo ci = {
        .sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ci, NULL, &inst) != VK_SUCCESS) return info;

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    if (n == 0) { vkDestroyInstance(inst, NULL); return info; }
    if (n > 16) n = 16;

    VkPhysicalDevice devs[16];
    vkEnumeratePhysicalDevices(inst, &n, devs);

    int chosen = -1;
    VkPhysicalDeviceProperties chosen_props = {0};
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            chosen = (int)i;
            chosen_props = p;
            break;
        }
        if (chosen < 0 && p.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
            chosen = (int)i;
            chosen_props = p;
        }
    }

    if (chosen < 0) { vkDestroyInstance(inst, NULL); return info; }

    info.has_gpu   = true;
    info.vendor_id = chosen_props.vendorID;
    snprintf(info.gpu_name, sizeof(info.gpu_name), "%s", chosen_props.deviceName);

    /* Does the chosen device advertise VK_EXT_memory_budget? If so the driver
     * will report live per-heap budget/usage, which is the real free-VRAM
     * figure (it already excludes what the compositor and other apps hold). */
    bool has_budget = false;
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(devs[chosen], NULL, &ext_count, NULL);
    if (ext_count > 0) {
        VkExtensionProperties *exts = malloc(ext_count * sizeof(*exts));
        if (exts) {
            vkEnumerateDeviceExtensionProperties(devs[chosen], NULL, &ext_count, exts);
            for (uint32_t i = 0; i < ext_count; i++) {
                if (strcmp(exts[i].extensionName,
                           VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
                    has_budget = true;
                    break;
                }
            }
            free(exts);
        }
    }

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
    };
    VkPhysicalDeviceMemoryProperties2 mem2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
        .pNext = has_budget ? &budget : NULL,
    };
    vkGetPhysicalDeviceMemoryProperties2(devs[chosen], &mem2);
    VkPhysicalDeviceMemoryProperties mem = mem2.memoryProperties;

    /* The GPU's VRAM is its largest DEVICE_LOCAL heap. Track the index so the
     * per-heap budget/usage arrays can be read for that same heap. */
    uint64_t max_local_bytes = 0;
    int local_heap = -1;
    for (uint32_t h = 0; h < mem.memoryHeapCount; h++) {
        if (mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            if (mem.memoryHeaps[h].size > max_local_bytes) {
                max_local_bytes = mem.memoryHeaps[h].size;
                local_heap = (int)h;
            }
        }
    }
    info.vram_total_mb = max_local_bytes / (1024ULL * 1024ULL);

    if (has_budget && local_heap >= 0) {
        uint64_t b = budget.heapBudget[local_heap];
        uint64_t u = budget.heapUsage[local_heap];
        info.vram_budget_known = true;
        info.vram_budget_mb = b / (1024ULL * 1024ULL);
        info.vram_used_mb   = u / (1024ULL * 1024ULL);
        info.vram_avail_mb  = (b > u ? b - u : 0) / (1024ULL * 1024ULL);
    }

    vkDestroyInstance(inst, NULL);
    return info;
}
