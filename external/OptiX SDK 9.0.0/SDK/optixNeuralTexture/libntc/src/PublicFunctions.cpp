/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "Context.h"
#include "Errors.h"
#include "FeatureGridMath.h"
#include "KnownLatentShapes.h"
#include "MlpDesc.h"
#include "TextureSetMetadata.h"
#include <cmath>
#include <limits>

// This file contains implementations for the public functions declared in ntc.h
// that do not belong to any class.

namespace ntc
{

uint32_t GetInterfaceVersion()
{
    return InterfaceVersion;
}

class DefaultAllocator : public IAllocator
{
public:
    void* Allocate(size_t size) override
    {
        return malloc(size);
    }

    void Deallocate(void* ptr, size_t size) override
    {
        free(ptr);
    }
};

static DefaultAllocator g_DefaultAllocator;

Status CreateContext(IContext** pOutContext, ContextParameters const& params)
{
    ClearErrorMessage();

    if (params.interfaceVersion != InterfaceVersion)
    {
        SetErrorMessage("Invalid interface version: provided 0x%08x, expected 0x%08x.",
            params.interfaceVersion, InterfaceVersion);
        return Status::InterfaceMismatch;
    }

    if (!pOutContext)
    {
        SetErrorMessage("pOutContext is NULL.");
        return Status::InvalidArgument;
    }

    ContextParameters paramsCopy = params;

    bool cudaAvailable = false;

#if NTC_WITH_CUDA
    if (params.cudaDevice >= 0)
    {
        cudaDeviceProp prop{};
        cudaError_t err = cudaGetDeviceProperties(&prop, params.cudaDevice);
        
        if (err != cudaSuccess)
        {
            SetErrorMessage("Cannot get information about CUDA device %d, error code = %s.",
                params.cudaDevice, cudaGetErrorName(err));
        }

        if (prop.major < NTC_CUDA_MIN_COMPUTE_MAJOR ||
            prop.major == NTC_CUDA_MIN_COMPUTE_MAJOR && prop.minor < NTC_CUDA_MIN_COMPUTE_MINOR)
        {
            SetErrorMessage("The CUDA device compute capability (%d.%d) is less than the minimum required "
                "by LibNTC (%d.%d).", prop.major, prop.minor, NTC_CUDA_MIN_COMPUTE_MAJOR, NTC_CUDA_MIN_COMPUTE_MINOR);
        }
        else
        {
            err = cudaInitDevice(params.cudaDevice, 0, 0);
            if (err == cudaSuccess)
            {
                cudaAvailable = true;
            }
            else
            {
                SetCudaErrorMessage("cudaInitDevice", err);
            }
        }
    }
#endif

    if (!cudaAvailable)
        paramsCopy.cudaDevice = -1;

    // Validate the graphicsAPI value and the device objects
    switch (params.graphicsApi)
    {
        case GraphicsAPI::None:
            break;
            
        case GraphicsAPI::D3D12:
#if NTC_WITH_DX12
            if (!params.d3d12Device)
            {
                SetErrorMessage("d3d12Device is NULL.");
                return Status::InvalidArgument;
            }
            break;
#else
            SetErrorMessage("This version of LibNTC doesn't support D3D12.");
            return Status::NotImplemented;
#endif

        case GraphicsAPI::Vulkan:
#if NTC_WITH_VULKAN
            if (!params.vkDevice || !params.vkInstance || !params.vkPhysicalDevice)
            {
                SetErrorMessage("One of (vkDevice, vkInstance, vkPhysicalDevice) is NULL.");
                return Status::InvalidArgument;
            }
#else
            SetErrorMessage("This version of LibNTC doesn't support Vulkan.");
            return Status::NotImplemented;
#endif
            break;

        default:
            SetErrorMessage("Invalid graphicsApi (%d).", int(params.graphicsApi));
            return Status::InvalidArgument;
    }

    if (!paramsCopy.pAllocator)
        paramsCopy.pAllocator = &g_DefaultAllocator;

    Context* context = new(paramsCopy.pAllocator->Allocate(sizeof(Context))) Context(paramsCopy);
    *pOutContext = context;
    
    return cudaAvailable ? Status::Ok : Status::CudaUnavailable;
}

void DestroyContext(IContext* context)
{
    if (!context)
        return;

    auto allocator = ((Context*)context)->GetAllocator();
    context->~IContext();
    allocator->Deallocate(context, sizeof(Context));
}

const char* GetLastErrorMessage()
{
    return GetErrorBuffer();
}

const char* StatusToString(Status status)
{
    switch(status)
    {
    case Status::Ok:
        return "Ok";
    case Status::InterfaceMismatch:
        return "InterfaceMismatch";
    case Status::Incomplete:
        return "Incomplete";
    case Status::Unsupported:
        return "Unsupported";
    case Status::InvalidArgument:
        return "InvalidArgument";
    case Status::InvalidState:
        return "InvalidState";
    case Status::UnknownError:
        return "UnknownError";
    case Status::FileUnavailable:
        return "FileUnavailable";
    case Status::FileUnrecognized:
        return "FileUnrecognized";
    case Status::FileIncompatible:
        return "FileIncompatible";
    case Status::IOError:
        return "IOError";
    case Status::OutOfRange:
        return "OutOfRange";
    case Status::CudaUnavailable:
        return "CudaUnavailable";
    case Status::CudaError:
        return "CudaError";
    case Status::NotImplemented:
        return "NotImplemented";
    case Status::InternalError:
        return "InternalError";
    case Status::OutOfMemory:
        return "OutOfMemory";
    case Status::ShaderUnavailable:
        return "ShaderUnavailable";
    default:
        static char string[16];
        snprintf(string, sizeof(string), "%d", int(status));
        return string;
    }
}

const char* ChannelFormatToString(ChannelFormat format)
{
    switch (format)
    {
    case ChannelFormat::UNKNOWN:
        return "UNKNOWN";
    case ChannelFormat::UNORM8:
        return "UNORM8";
    case ChannelFormat::UNORM16:
        return "UNORM16";
    case ChannelFormat::FLOAT16:
        return "FLOAT16";
    case ChannelFormat::FLOAT32:
        return "FLOAT32";
    case ChannelFormat::UINT32:
        return "UINT32";
    default:
        static char string[16];
        snprintf(string, sizeof(string), "%d", int(format));
        return string;
    }
}

const char* BlockCompressedFormatToString(BlockCompressedFormat format)
{
    switch (format)
    {
    case BlockCompressedFormat::None:
        return "None";
    case BlockCompressedFormat::BC1:
        return "BC1";
    case BlockCompressedFormat::BC2:
        return "BC2";
    case BlockCompressedFormat::BC3:
        return "BC3";
    case BlockCompressedFormat::BC4:
        return "BC4";
    case BlockCompressedFormat::BC5:
        return "BC5";
    case BlockCompressedFormat::BC6:
        return "BC6";
    case BlockCompressedFormat::BC7:
        return "BC7";
    default:
        static char string[16];
        snprintf(string, sizeof(string), "%d", int(format));
        return string;
    }
}

const char* ColorSpaceToString(ColorSpace colorSpace)
{
    switch (colorSpace)
    {
    case ColorSpace::Linear:
        return "Linear";
    case ColorSpace::sRGB:
        return "sRGB";
    case ColorSpace::HLG:
        return "HLG";
    default:
        static char string[16];
        snprintf(string, sizeof(string), "%d", int(colorSpace));
        return string;
    }
}

const char* NetworkVersionToString(int networkVersion)
{
    switch(networkVersion)
    {
    case NTC_NETWORK_UNKNOWN:
        return "NTC_NETWORK_UNKNOWN";
    case NTC_NETWORK_SMALL:
        return "NTC_NETWORK_SMALL";
    case NTC_NETWORK_MEDIUM:
        return "NTC_NETWORK_MEDIUM";
    case NTC_NETWORK_LARGE:
        return "NTC_NETWORK_LARGE";
    case NTC_NETWORK_XLARGE:
        return "NTC_NETWORK_XLARGE";
    default:
        static char string[16];
        snprintf(string, sizeof(string), "%d", int(networkVersion));
        return string;
    }
}

const char* InferenceWeightTypeToString(InferenceWeightType weightType)
{
    switch(weightType)
    {
    case InferenceWeightType::GenericInt8:
        return "GenericInt8";
    case InferenceWeightType::GenericFP8:
        return "GenericFP8";
    case InferenceWeightType::CoopVecInt8:
        return "CoopVecInt8";
    case InferenceWeightType::CoopVecFP8:
        return "CoopVecFP8";
    default:
        static char string[16];
        snprintf(string, sizeof(string), "%d", int(weightType));
        return string;
    }
}

size_t GetBytesPerPixelComponent(ChannelFormat format)
{
    switch (format)
    {
    case ChannelFormat::UNORM8:
        return 1;
    case ChannelFormat::UNORM16:
    case ChannelFormat::FLOAT16:
        return 2;
    case ChannelFormat::FLOAT32:
    case ChannelFormat::UINT32:
        return 4;
    default:
        return 0;
    }
}

float LossToPSNR(float loss)
{
    // Math quirk: when loss is 0, log10f(0) returns inf, which turns into -inf. 
    // Returning positive inf for zero loss makes more sense.
    // Also, this function assumes that max value of the signal is 1.0, which is not true for HDR...
    return (loss == 0.f) ? std::numeric_limits<float>::infinity() : -10.f * log10f(loss);
}

Status EstimateCompressedTextureSetSize(TextureSetDesc const& textureSetDesc,
    LatentShape const& latentShape, size_t& outSize)
{
    Status status = TextureSetMetadata::ValidateTextureSetDesc(textureSetDesc);
    if (status != Status::Ok)
        return status;

    status = TextureSetMetadata::ValidateLatentShape(latentShape, NTC_NETWORK_UNKNOWN);
    if (status != Status::Ok)
        return status;

    size_t latentSize = FeatureGridMath::CalculateQuantizedLatentsSize(
        textureSetDesc.width,
        textureSetDesc.height,
        textureSetDesc.mips,
        latentShape.gridSizeScale,
        latentShape.highResFeatures,
        latentShape.lowResFeatures,
        latentShape.highResQuantBits,
        latentShape.lowResQuantBits);

    MlpDesc const* mlpDesc = MlpDesc::PickOptimalConfig(latentShape.highResFeatures, latentShape.lowResFeatures);
    if (!mlpDesc)
        return Status::InvalidArgument;

    // Int8 per weight, 2x float per output (scale and bias)
    size_t const mlpSize = mlpDesc->GetWeightCount() + mlpDesc->GetLayerOutputCount() * 2 * sizeof(float);

    outSize = latentSize + mlpSize;

    return Status::Ok;
}

// The table of statistically best latent shapes for a set of BPP values.
// Larger network versions support more latent shapes for each BPP configuraion, so a better choice can be made.
// Generated using 'scripts/find_optimal_configs.py'
KnownLatentShape const g_KnownLatentShapes[KnownLatentShapeCount] = {
    //  bpp         small network        medium network       large network        xlarge network
    //         scale hrf lrf hrq lrq
    {  0.500f, { { 4,  4,  4, 1, 4 }, { 4,  4,  4, 1, 4 }, { 4,  4,  4, 1, 4 }, { 4,  4,  4, 1, 4 } } },
    {  0.625f, { { 4,  4, 12, 1, 2 }, { 4,  4, 12, 1, 2 }, { 4,  4, 12, 1, 2 }, { 4,  4, 12, 1, 2 } } },
    {  0.750f, { { 4,  4,  8, 1, 4 }, { 4,  4,  8, 1, 4 }, { 4,  4,  8, 1, 4 }, { 4,  4,  8, 1, 4 } } },
    {  0.875f, { { 4,  4, 12, 2, 2 }, { 4,  4, 12, 2, 2 }, { 4,  4, 12, 2, 2 }, { 4,  4, 12, 2, 2 } } },
    {  1.000f, { { 4,  4,  8, 2, 4 }, { 4,  4,  8, 2, 4 }, { 4,  4,  8, 2, 4 }, { 4,  4,  8, 2, 4 } } },
    {  1.250f, { { 4,  4, 12, 2, 4 }, { 4,  4, 12, 2, 4 }, { 4,  4, 12, 2, 4 }, { 4,  4, 12, 2, 4 } } },
    {  1.500f, { { 4,  4, 16, 2, 4 }, { 4,  4, 16, 2, 4 }, { 4,  4, 16, 2, 4 }, { 4,  4, 16, 2, 4 } } },
    {  1.750f, { { 4,  4, 12, 4, 4 }, { 4,  8, 12, 2, 4 }, { 4,  8, 12, 2, 4 }, { 4,  8, 12, 2, 4 } } },
    {  2.000f, { { 4,  4, 16, 4, 4 }, { 4,  8, 16, 2, 4 }, { 4,  8, 16, 2, 4 }, { 4,  8, 16, 2, 4 } } },
    {  2.250f, { { 4,  4, 16, 1, 8 }, { 4,  8,  4, 4, 4 }, { 4, 12, 12, 2, 4 }, { 4, 12, 12, 2, 4 } } },
    {  2.500f, { { 4,  4, 12, 4, 8 }, { 4,  8,  8, 4, 4 }, { 4, 12, 16, 2, 4 }, { 4, 12, 16, 2, 4 } } },
    {  3.000f, { { 2,  4,  8, 1, 4 }, { 4,  8, 16, 4, 4 }, { 4,  8, 16, 4, 4 }, { 4, 16, 16, 2, 4 } } },
    {  3.500f, { { 2,  4, 12, 2, 2 }, { 4,  8, 12, 4, 8 }, { 4, 12,  8, 4, 4 }, { 4, 12,  8, 4, 4 } } },
    {  4.000f, { { 2,  4,  8, 2, 4 }, { 2,  4,  8, 2, 4 }, { 4, 12, 16, 4, 4 }, { 4, 12, 16, 4, 4 } } },
    {  4.500f, { { 2,  4,  4, 4, 2 }, { 2,  4,  4, 4, 2 }, { 4, 12, 12, 4, 8 }, { 4, 16,  8, 4, 4 } } },
    {  5.000f, { { 2,  4, 12, 2, 4 }, { 2,  4, 12, 2, 4 }, { 2,  4, 12, 2, 4 }, { 4, 16, 16, 4, 4 } } },
    {  6.000f, { { 2,  4, 16, 2, 4 }, { 2,  8,  8, 2, 4 }, { 2,  8,  8, 2, 4 }, { 2,  8,  8, 2, 4 } } },
    {  7.000f, { { 2,  4, 12, 4, 4 }, { 2,  4, 12, 4, 4 }, { 2,  4, 12, 4, 4 }, { 2,  4, 12, 4, 4 } } },
    {  8.000f, { { 2,  4, 16, 4, 4 }, { 2,  8, 16, 2, 4 }, { 2,  8, 16, 2, 4 }, { 2,  8, 16, 2, 4 } } },
    {  9.000f, { { 2,  4, 16, 1, 8 }, { 2,  8,  4, 4, 4 }, { 2, 12, 12, 2, 4 }, { 2, 12, 12, 2, 4 } } },
    { 10.000f, { { 2,  4, 12, 4, 8 }, { 2,  8,  8, 4, 4 }, { 2, 12, 16, 2, 4 }, { 2, 12, 16, 2, 4 } } },
    { 12.000f, { { 2,  4, 16, 4, 8 }, { 2,  8, 16, 4, 4 }, { 2,  8, 16, 4, 4 }, { 2,  8, 16, 4, 4 } } },
    { 14.000f, { { 2,  4, 12, 8, 8 }, { 2,  8, 12, 4, 8 }, { 2, 12,  4, 4, 8 }, { 2, 12,  4, 4, 8 } } },
    { 16.000f, { { 2,  4, 16, 8, 8 }, { 2,  8, 16, 4, 8 }, { 2, 12, 16, 4, 4 }, { 2, 12, 16, 4, 4 } } },
    { 18.000f, { { 0,  0,  0, 0, 0 }, { 2,  8,  8, 8, 4 }, { 2, 12, 12, 4, 8 }, { 2, 16,  4, 4, 8 } } },
    { 20.000f, { { 0,  0,  0, 0, 0 }, { 2,  8, 16, 8, 4 }, { 2, 12, 16, 4, 8 }, { 2, 16,  8, 4, 8 } } },
};

int GetKnownLatentShapeCount(int networkVersion)
{
    if (networkVersion == NTC_NETWORK_UNKNOWN)
        networkVersion = NTC_NETWORK_XLARGE;

    if (networkVersion < NTC_NETWORK_SMALL || networkVersion > NTC_NETWORK_XLARGE)
        return 0;

    // See g_KnownLatentShapes - the small network doesn't support the 2 highest bpp values
    if (networkVersion == NTC_NETWORK_SMALL)
        return KnownLatentShapeCount - 2;

    return KnownLatentShapeCount;
}

Status EnumerateKnownLatentShapes(int index, int networkVersion, float& outBitsPerPixel, LatentShape& outShape)
{
    ClearErrorMessage();

    if (index < 0 || index >= GetKnownLatentShapeCount(networkVersion))
    {
        SetErrorMessage("Invalid index (%d), must be 0-%d.", index, GetKnownLatentShapeCount(networkVersion) - 1);
        return Status::OutOfRange;
    }

    outBitsPerPixel = g_KnownLatentShapes[index].bitsPerPixel;
    outShape = g_KnownLatentShapes[index].shapes[networkVersion - NTC_NETWORK_SMALL];
    return Status::Ok;
}

Status PickLatentShape(float requestedBitsPerPixel, int networkVersion, float& outBitsPerPixel, LatentShape& outShape)
{
    ClearErrorMessage();

    if (requestedBitsPerPixel <= 0.f)
    {
        SetErrorMessage("requestedBitsPerPixel (%f) must be positive.", requestedBitsPerPixel);
        return Status::OutOfRange;
    }
     
    if (networkVersion == NTC_NETWORK_UNKNOWN)
        networkVersion = NTC_NETWORK_XLARGE;

    int const presetCount = GetKnownLatentShapeCount(networkVersion);
    if (presetCount == 0)
    {
        SetErrorMessage("Invalid networkVersion (%d).", networkVersion);
        return Status::OutOfRange;
    }

    bool found = false;
    float currentDiff = std::numeric_limits<float>::max();
    for (int index = 0; index < presetCount; ++index)
    {
        KnownLatentShape const& shape = g_KnownLatentShapes[index];
        float const diff = fabsf(shape.bitsPerPixel - requestedBitsPerPixel);
        if (diff < requestedBitsPerPixel * 0.25f)
        {
            if (diff < currentDiff)
            {
                found = true;
                currentDiff = diff;
                outBitsPerPixel = shape.bitsPerPixel;
                outShape = shape.shapes[networkVersion - NTC_NETWORK_SMALL];
            }
        }
        else if (found)
        {
            // The known shape list is sorted, and we're past the matching range,
            // so there won't be anything interesting anymore.
            return Status::Ok;
        }
    }

    if (found)
        return Status::Ok;


    SetErrorMessage("The provided requestedBitsPerPixel (%f) is out of the supported range.", requestedBitsPerPixel);
    return Status::OutOfRange;
}

float GetLatentShapeBitsPerPixel(LatentShape const& shape)
{
    return (float(shape.highResFeatures * shape.highResQuantBits) + float(shape.lowResFeatures * shape.lowResQuantBits) * 0.25f)
        / float(shape.gridSizeScale * shape.gridSizeScale);
}

double DecodeImageDifferenceResult(uint64_t value)
{
    // Convert from 16.48 fixed point to double
    return double(value) * 0x1p-48;
}

}