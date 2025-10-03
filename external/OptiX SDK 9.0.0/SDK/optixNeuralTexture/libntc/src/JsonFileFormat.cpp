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

#include "JsonFileFormat.h"
#include "JsonSerialization.h"

#define RUN_TEST 0

#if RUN_TEST
#include <chrono>
#endif

namespace ntc::json
{

static auto const CompressionEnum = MakeEnumSchema({
    EnumValue(Compression::None, "None")
});

static auto const BufferViewSchema = MakeObjectSchema("BufferView", {
    Field::UInt64("offset",                   &BufferView::offset),
    Field::UInt64("storedSize",               &BufferView::storedSize),
    Field::OptionalEnum("compression",        &BufferView::compression, CompressionEnum),
    Field::OptionalUInt64("uncompressedSize", &BufferView::uncompressedSize),
});

static ArrayOfObjectHandler<BufferView> const ArrayOfBufferViewHandler(BufferViewSchema);

static auto const LatentShapeSchema = MakeObjectSchema("LatentShape", {
    Field::UInt("highResFeatures",  &LatentShape::highResFeatures),
    Field::UInt("highResQuantBits", &LatentShape::highResQuantBits),
    Field::UInt("lowResFeatures",   &LatentShape::lowResFeatures),
    Field::UInt("lowResQuantBits",  &LatentShape::lowResQuantBits),
});

static OptionalObjectHandler<LatentShape> const OptionalLatentShapeHandler(LatentShapeSchema);

static auto const MatrixLayoutEnum = MakeEnumSchema({
    EnumValue(MatrixLayout::RowMajor,       "RowMajor"),
    EnumValue(MatrixLayout::ColumnMajor,    "ColumnMajor")
});

static auto const ActivationTypeEnum = MakeEnumSchema({
    EnumValue(ActivationType::HGELUClamp, "HGELUClamp")
});

static auto const MlpDataTypeEnum = MakeEnumSchema({
    EnumValue(MlpDataType::Int8,      "Int8"),
    EnumValue(MlpDataType::FloatE4M3, "FloatE4M3"),
    EnumValue(MlpDataType::FloatE5M2, "FloatE5M2"),
    EnumValue(MlpDataType::Float16,   "Float16"),
    EnumValue(MlpDataType::Float32,   "Float32"),
});

static auto const MLPLayerSchema = MakeObjectSchema("MLPLayer", {
    Field::UInt("inputChannels",        &MLPLayer::inputChannels),
    Field::UInt("outputChannels",       &MLPLayer::outputChannels),
    Field::UInt("weightView",           &MLPLayer::weightView),
    Field::OptionalUInt("scaleView",    &MLPLayer::scaleView),
    Field::UInt("biasView",             &MLPLayer::biasView),
    Field::OptionalEnum("weightType",   &MLPLayer::weightType, MlpDataTypeEnum),
    Field::OptionalEnum("scaleBiasType",&MLPLayer::scaleBiasType, MlpDataTypeEnum),
});

static ArrayOfObjectHandler<MLPLayer> const ArrayOfMLPLayerHandler(MLPLayerSchema);

static auto const MLPSchema = MakeObjectSchema("MLP", {
    Field::ArrayOfObject("layers",       &MLP::layers, ArrayOfMLPLayerHandler),
    Field::OptionalEnum("activation",    &MLP::activation, ActivationTypeEnum),
    Field::OptionalEnum("weightLayout",  &MLP::weightLayout, MatrixLayoutEnum),
    Field::Enum("weightType",            &MLP::weightType, MlpDataTypeEnum),
    Field::Enum("scaleBiasType",         &MLP::scaleBiasType, MlpDataTypeEnum),
});

static OptionalObjectHandler<MLP> const OptionalMLPHandler(MLPSchema);
static ArrayOfObjectHandler<MLP> const ArrayOfMLPHandler(MLPSchema);

static auto const ChannelFormatEnum = MakeEnumSchema({
    EnumValue(ChannelFormat::UNKNOWN,   "UNKNOWN"),
    EnumValue(ChannelFormat::UNORM8,    "UNORM8"),
    EnumValue(ChannelFormat::UNORM16,   "UNORM16"),
    EnumValue(ChannelFormat::FLOAT16,   "FLOAT16"),
    EnumValue(ChannelFormat::FLOAT32,   "FLOAT32"),
    EnumValue(ChannelFormat::UINT32,    "UINT32"),
});

static auto const ColorSpaceEnum = MakeEnumSchema({
    EnumValue(ColorSpace::Linear,   "Linear"),
    EnumValue(ColorSpace::sRGB,     "sRGB"),
    EnumValue(ColorSpace::HLG,      "HLG"),
});

static auto const BlockCompressedFormatEnum = MakeEnumSchema({
    EnumValue(BlockCompressedFormat::None,  "None"),
    EnumValue(BlockCompressedFormat::BC1,   "BC1"),
    EnumValue(BlockCompressedFormat::BC2,   "BC2"),
    EnumValue(BlockCompressedFormat::BC3,   "BC3"),
    EnumValue(BlockCompressedFormat::BC4,   "BC4"),
    EnumValue(BlockCompressedFormat::BC5,   "BC5"),
    EnumValue(BlockCompressedFormat::BC6,   "BC6"),
    EnumValue(BlockCompressedFormat::BC7,   "BC7"),
});

static auto const TextureSchema = MakeObjectSchema("Texture", {
    Field::String("name",                             &Texture::name),
    Field::UInt("firstChannel",                       &Texture::firstChannel),
    Field::UInt("numChannels",                        &Texture::numChannels),
    Field::OptionalEnum("channelFormat",              &Texture::channelFormat, ChannelFormatEnum),
    Field::OptionalEnum("rgbColorSpace",              &Texture::rgbColorSpace, ColorSpaceEnum),
    Field::OptionalEnum("alphaColorSpace",            &Texture::alphaColorSpace, ColorSpaceEnum),
    Field::OptionalEnum("bcFormat",                   &Texture::bcFormat, BlockCompressedFormatEnum),
    Field::OptionalUInt("bcQuality",                  &Texture::bcQuality),
    Field::OptionalUInt("bcAccelerationDataView",     &Texture::bcAccelerationDataView),
});

static ArrayOfObjectHandler<Texture> const ArrayOfTextureHandler(TextureSchema);

static auto const ChannelSchema = MakeObjectSchema("Channel", {
    Field::Float("scale",             &Channel::scale),
    Field::Float("bias",              &Channel::bias),
    Field::OptionalEnum("colorSpace", &Channel::colorSpace, ColorSpaceEnum),
});

static ArrayOfObjectHandler<Channel> const ArrayOfChannelHandler(ChannelSchema);

static auto const LatentImageSchema = MakeObjectSchema("LatentImage", {
    Field::UInt("highResWidth",          &LatentImage::highResWidth),
    Field::UInt("highResHeight",         &LatentImage::highResHeight),
    Field::UInt("lowResWidth",           &LatentImage::lowResWidth),
    Field::UInt("lowResHeight",          &LatentImage::lowResHeight),
    Field::UInt("highResBitsPerPixel",   &LatentImage::highResBitsPerPixel),
    Field::UInt("lowResBitsPerPixel",    &LatentImage::lowResBitsPerPixel),
    Field::UInt("highResView",           &LatentImage::highResView),
    Field::UInt("lowResView",            &LatentImage::lowResView),
});

static ArrayOfObjectHandler<LatentImage> const ArrayOfLatentImageHandler(LatentImageSchema);

static auto const ColorImageDataSchema = MakeObjectSchema("ColorImageData", {
    Field::UInt("view",                       &ColorImageData::view),
    Field::OptionalEnum("uncompressedFormat", &ColorImageData::uncompressedFormat, ChannelFormatEnum),
    Field::OptionalEnum("bcFormat",           &ColorImageData::bcFormat, BlockCompressedFormatEnum),
    Field::OptionalUInt("rowPitch",           &ColorImageData::rowPitch),
    Field::OptionalUInt("pixelStride",        &ColorImageData::pixelStride),
    Field::OptionalUInt("numChannels",        &ColorImageData::numChannels),
});

static OptionalObjectHandler<ColorImageData> const OptionalColorImageDataHandler(ColorImageDataSchema);
static ArrayOfObjectHandler<ColorImageData> const ArrayOfColorImageDataHandler(ColorImageDataSchema);

static auto const ColorMipSchema = MakeObjectSchema("ColorMip", {
    Field::OptionalUInt("width",                &ColorMip::width),
    Field::OptionalUInt("height",               &ColorMip::height),
    Field::OptionalUInt("latentMip",            &ColorMip::latentMip),
    Field::OptionalFloat("positionLod",         &ColorMip::positionLod),
    Field::OptionalFloat("positionScale",       &ColorMip::positionScale),
    Field::OptionalObject("combinedColorData",  &ColorMip::combinedColorData, OptionalColorImageDataHandler),
    Field::ArrayOfObject("perTextureColorData", &ColorMip::perTextureColorData, ArrayOfColorImageDataHandler),
});

static ArrayOfObjectHandler<ColorMip> const ArrayOfColorMipHandler(ColorMipSchema);

static auto const DocumentSchema = MakeObjectSchema("Document", {
    Field::UInt("schemaVersion",         &Document::schemaVersion),
    Field::UInt("width",                 &Document::width),
    Field::UInt("height",                &Document::height),
    Field::UInt("numChannels",           &Document::numChannels),
    Field::OptionalUInt("numColorMips",  &Document::numColorMips),
    Field::OptionalObject("latentShape", &Document::latentShape, OptionalLatentShapeHandler),
    Field::OptionalObject("mlp",         &Document::mlp, OptionalMLPHandler),
    Field::ArrayOfObject("mlpVersions",  &Document::mlpVersions, ArrayOfMLPHandler),
    Field::ArrayOfObject("textures",     &Document::textures, ArrayOfTextureHandler),
    Field::ArrayOfObject("channels",     &Document::channels, ArrayOfChannelHandler),
    Field::ArrayOfObject("latents",      &Document::latents, ArrayOfLatentImageHandler),
    Field::ArrayOfObject("colorMips",    &Document::colorMips, ArrayOfColorMipHandler),
    Field::ArrayOfObject("views",        &Document::views, ArrayOfBufferViewHandler),
});


bool SerializeDocument(Document const& document, String& outString, String& outErrorMessage)
{
    return SerializeAbstractDocument(&document, DocumentSchema, document.allocator,
        outString, outErrorMessage, JsonChunkSizeLimit);
}

bool ParseDocument(Document& outDocument, char* input, String& outErrorMessage)
{
    return ParseAbstractDocument(&outDocument, DocumentSchema, outDocument.allocator, input, outErrorMessage);
}

#if RUN_TEST

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

static bool Test()
{
    using namespace std::chrono;

    DefaultAllocator allocator;
    Document document(&allocator);
    document.width = 1024;
    document.height = 512;
    document.numChannels = 12;
    document.latentShape = LatentShape();
    document.latentShape->highResFeatures = 16;
    document.latentShape->lowResFeatures = 12;
    document.latentShape->highResQuantBits = 4;
    document.latentShape->lowResQuantBits = 2;
    
    BufferView view;
    view.offset = 12;
    view.storedSize = 256;
    document.views.push_back(view);
    view.offset = 268;
    view.storedSize = 384;
    document.views.push_back(view);

    MLP mlp(&allocator);
    mlp.activation = ActivationType::HGELUClamp;
    mlp.weightType = MlpDataType::Int8;
    mlp.scaleBiasType = MlpDataType::Float32;

    MLPLayer layer(&allocator);
    layer.inputChannels = 64;
    layer.outputChannels = 64;
    layer.weightView = 0;
    layer.scaleView = 1;
    layer.biasView = 2;
    mlp.layers.push_back(layer);
    document.mlp = mlp;

    Texture texture(&allocator);
    texture.name = String("Diffuse", &allocator);
    texture.firstChannel = 0;
    texture.numChannels = 3;
    document.textures.push_back(texture);

    LatentImage limg(&allocator);
    limg.highResWidth = 256;
    limg.lowResWidth = 128;
    limg.highResHeight = 128;
    limg.lowResHeight = 64;
    limg.highResBitsPerPixel = 64;
    limg.lowResBitsPerPixel = 24;
    limg.highResView = 3;
    limg.lowResView = 4;
    document.latents.push_back(limg);

    ColorImageData cimg(&allocator);
    cimg.view = 5;
    cimg.uncompressedFormat = ChannelFormat::UNORM8;
    cimg.numChannels = 3;
    
    ColorMip cmip(&allocator);
    cmip.latentMip = 0;
    cmip.perTextureColorData.push_back(cimg);
    document.colorMips.push_back(cmip);

    String str(&allocator);
    String error(&allocator);
    if (!SerializeDocument(document, str, error))
    {
        printf("Serialize failed: %s!\n", error.c_str());
        return false;
    }
    printf("%s\n", str.c_str());

    Document newDocument(&allocator);
    auto t1 = high_resolution_clock::now();

    String errorMessage(&allocator);
    //char const* customJson = "{\"width\": 512, \"something\": 127, \"numChannels\": 12, \"unknownObj\": {\"a\": null}, \"height\": 12, \"unknownArray\": [1,2,null,\"c\"]}";
    //bool success = ParseDocument(newDocument, const_cast<char*>(customJson), errorMessage);
    bool success = ParseDocument(newDocument, const_cast<char*>(str.c_str()), errorMessage);
    if (!success)
        printf("%s\n", errorMessage.c_str());

    auto t2 = high_resolution_clock::now();
    uint32_t usec = uint32_t(duration_cast<microseconds>(t2 - t1).count());

    printf("Parsed successfully: %d, time: %d us\n", int(success), usec);

    return success;
}

static bool g_TestPassed = Test();

#endif

}