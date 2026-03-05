#if TENSORFLOW_USE_NUMA
#include <unistd.h>

#include <algorithm>

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "hwloc.h"  // NOLINT(build/include_subdir)
#else
#include "xla/tsl/platform/mem.h"
#endif

#include "xla/tsl/platform/numa.h"

namespace tsl::port {

#ifdef TENSORFLOW_USE_NUMA
namespace {
static hwloc_topology_t hwloc_topology_handle;

bool HaveHWLocTopology() {
  // One time initialization using absl::call_once
  static absl::once_flag init_flag;
  static bool init_success = false;
  absl::call_once(init_flag, []() {
    if (hwloc_topology_init(&hwloc_topology_handle)) {
      LOG(ERROR) << "Call to hwloc_topology_init() failed";
      init_success = false;
      return;
    }
    if (hwloc_topology_load(hwloc_topology_handle)) {
      LOG(ERROR) << "Call to hwloc_topology_load() failed";
      init_success = false;
      return;
    }
    init_success = true;
  });
  return init_success;
}

// Return the first hwloc object of the given type whose os_index
// matches 'index'.
hwloc_obj_t GetHWLocTypeIndex(hwloc_obj_type_t tp, int index) {
  hwloc_obj_t obj = nullptr;
  if (index >= 0) {
    while ((obj = hwloc_get_next_obj_by_type(hwloc_topology_handle, tp, obj)) !=
           nullptr) {
      if (obj->os_index == index) break;
    }
  }
  return obj;
}
}  // namespace
#endif  // TENSORFLOW_USE_NUMA

bool NUMAEnabled() { return NUMANumNodes() > 1; }

int NUMANumNodes() {
#ifdef TENSORFLOW_USE_NUMA
  if (HaveHWLocTopology()) {
    int num_numanodes =
        hwloc_get_nbobjs_by_type(hwloc_topology_handle, HWLOC_OBJ_NUMANODE);
    return std::max(1, num_numanodes);
  } else {
    return 1;
  }
#else
  return 1;
#endif  // TENSORFLOW_USE_NUMA
}

void NUMASetThreadNodeAffinity(int node) {
#ifdef TENSORFLOW_USE_NUMA
  if (HaveHWLocTopology()) {
    // Find the corresponding NUMA node topology object.
    hwloc_obj_t obj = GetHWLocTypeIndex(HWLOC_OBJ_NUMANODE, node);
    if (obj) {
      hwloc_set_cpubind(hwloc_topology_handle, obj->cpuset,
                        HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT);
    } else {
      LOG(ERROR) << "Could not find hwloc NUMA node " << node;
    }
  }
#endif  // TENSORFLOW_USE_NUMA
}

int NUMAGetThreadNodeAffinity() {
  int node_index = kNUMANoAffinity;
#ifdef TENSORFLOW_USE_NUMA
  if (HaveHWLocTopology()) {
    hwloc_cpuset_t thread_cpuset = hwloc_bitmap_alloc();
    hwloc_get_cpubind(hwloc_topology_handle, thread_cpuset,
                      HWLOC_CPUBIND_THREAD);
    hwloc_obj_t obj = nullptr;
    // Return the first NUMA node whose cpuset is a (non-proper) superset of
    // that of the current thread.
    while ((obj = hwloc_get_next_obj_by_type(
                hwloc_topology_handle, HWLOC_OBJ_NUMANODE, obj)) != nullptr) {
      if (hwloc_bitmap_isincluded(thread_cpuset, obj->cpuset)) {
        node_index = obj->os_index;
        break;
      }
    }
    hwloc_bitmap_free(thread_cpuset);
  }
#endif  // TENSORFLOW_USE_NUMA
  return node_index;
}

void* NUMAMalloc(int node, size_t size, int minimum_alignment) {
#ifdef TENSORFLOW_USE_NUMA
  if (HaveHWLocTopology()) {
    hwloc_obj_t numa_node = GetHWLocTypeIndex(HWLOC_OBJ_NUMANODE, node);
    if (numa_node) {
      size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
      if (static_cast<size_t>(minimum_alignment) > page_size) {
        LOG(WARNING)
            << "Requested minimum_alignment (" << minimum_alignment
            << ") is greater than system page size (" << page_size
            << "). hwloc_alloc_membind only guarantees page alignment, "
            << "so the requested alignment may not be honored.";
      }
      hwloc_alloc_membind(hwloc_topology_handle, size, numa_node->nodeset,
                          HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
    } else {
      LOG(ERROR) << "Failed to find hwloc NUMA node " << node;
    }
  }
#endif  // TENSORFLOW_USE_NUMA
  return AlignedMalloc(size, minimum_alignment);
}

void NUMAFree(void* ptr, size_t size) {
#ifdef TENSORFLOW_USE_NUMA
  if (HaveHWLocTopology()) {
    hwloc_free(hwloc_topology_handle, ptr, size);
    return;
  }
#endif  // TENSORFLOW_USE_NUMA
  Free(ptr);
}

int NUMAGetMemAffinity(const void* addr) {
  int node = kNUMANoAffinity;
#ifdef TENSORFLOW_USE_NUMA
  if (HaveHWLocTopology() && addr) {
    hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();
    if (!hwloc_get_area_memlocation(hwloc_topology_handle, addr, 4, nodeset,
                                    HWLOC_MEMBIND_BYNODESET)) {
      hwloc_obj_t obj = nullptr;
      while ((obj = hwloc_get_next_obj_by_type(
                  hwloc_topology_handle, HWLOC_OBJ_NUMANODE, obj)) != nullptr) {
        if (hwloc_bitmap_isincluded(nodeset, obj->nodeset)) {
          node = obj->os_index;
          break;
        }
      }
    } else {
      LOG(ERROR) << "Failed call to hwloc_get_area_memlocation.";
    }
    hwloc_bitmap_free(nodeset);
  }
#endif  // TENSORFLOW_USE_NUMA
  return node;
}

}  // namespace tsl::port
