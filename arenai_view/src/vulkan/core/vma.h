//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_VMA_H
#define ARENAI_VK_VMA_H

#include "./vk.h"

// VulkanMemoryAllocator (header-only); the single VMA_IMPLEMENTATION lives in
// vma_impl.cpp.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100 4127 4189 4324 4505)
#endif

#include <vk_mem_alloc.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif// ARENAI_VK_VMA_H
