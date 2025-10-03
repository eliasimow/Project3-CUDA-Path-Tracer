/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <libntc/ntc.h>

#if NTC_WITH_VULKAN
#include <vulkan/vulkan.hpp>
#include "../external/vk_cooperative_vector.h"
#endif
#ifdef _WIN32
#include <d3d12.h>
#endif

namespace ntc
{

// This class encapsulates the graphics API resources, such as device and function pointers.
// It's only used as a per-Context singleton, and its purpose is to avoid including the Vulkan/D3D12 headers
// into Context.h which is used in many places.
class GraphicsResources
{
public:
    GraphicsResources(ContextParameters const& params);
    ~GraphicsResources();

    [[nodiscard]] IAllocator* GetAllocator() const { return m_allocator; }

    GraphicsAPI GetGraphicsApi() const { return m_graphicsApi; }

    bool IsDP4aSupported() const { return m_dp4aSupported; }

    bool IsFloat16Supported() const { return m_float16Supported; }
    
    bool IsCoopVecInt8Supported() const { return m_coopVecInt8Supported; }

    bool IsCoopVecFP8Supported() const { return m_coopVecFP8Supported; }

#if NTC_WITH_VULKAN
    VkDevice GetVulkanDevice() const { return m_vulkanDevice; }

    VkPhysicalDevice GetVulkanPhysicalDevice() const { return m_vulkanPhysicalDevice; }

    PFN_vkGetPhysicalDeviceCooperativeVectorPropertiesNV GetGetPhysicalDeviceCooperativeVectorPropertiesNV() const
    { return m_vkGetPhysicalDeviceCooperativeVectorPropertiesNV; }

    PFN_vkConvertCooperativeVectorMatrixLayoutNV GetConvertCooperativeVectorMatrixLayoutNV() const
    { return m_vkConvertCooperativeVectorMatrixLayoutNV; }
#endif
#if NTC_WITH_DX12
    ID3D12Device* GetD3D12Device() const { return m_d3d12Device; }
#endif

private:
    IAllocator* m_allocator;
    GraphicsAPI m_graphicsApi;
    bool m_dp4aSupported = false;
    bool m_float16Supported = false;
    bool m_coopVecInt8Supported = false;
    bool m_coopVecFP8Supported = false;


#if NTC_WITH_VULKAN
#if VK_HEADER_VERSION >= 301
    typedef vk::detail::DynamicLoader VulkanDynamicLoader;
#else
    typedef vk::DynamicLoader VulkanDynamicLoader;
#endif

    VulkanDynamicLoader* m_vulkanLoader = nullptr;
    VkInstance m_vulkanInstance = nullptr;
    VkPhysicalDevice m_vulkanPhysicalDevice = nullptr;
    VkDevice m_vulkanDevice = nullptr;
    PFN_vkGetPhysicalDeviceCooperativeVectorPropertiesNV m_vkGetPhysicalDeviceCooperativeVectorPropertiesNV = nullptr;
    PFN_vkConvertCooperativeVectorMatrixLayoutNV m_vkConvertCooperativeVectorMatrixLayoutNV = nullptr;
#endif

#if NTC_WITH_DX12
    ID3D12Device* m_d3d12Device = nullptr;
    bool m_nvapiInitialized = false;
#endif
};

}
