# NTC Texture Set File Format

## Design Goals

1. **Backward compatibility**. Files created with earlier versions of the library should be compatible with later vesrsions.

2. **Extensibility**. It should be possible to add extra data fields to the file and maintain compatibility with implementations that are not aware of such fields.

3. **Quick indexing**. Reading metadata from the file should not require reading the entire file or seeking through it to locate chunks or such.

4. **Compact and quickly parseable headers**. While the header information should be extensible etc., parsing it should not take a lot of machine time, nor should it take a lot of disk space.

5. **Optional data compression**. It should be possible to compress portions of the data, such as latents or even headers, with common compression algorithms - specifically GDeflate for GPU decompression.


## Container Format

The NTC container format is loosely modeled after [GLB (Binary GLTF)](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#glb-file-format-specification), as a modern format most closely matching our goals. We could also implement formats similar to GLTF Separate or Embedded, but that doesn't seem to be necessary.

The NTC container stores binary data in little-endian format.

The container consists of a file header and two chunks: JSON descriptor and binary data. Chunks have no separate headers as it doesn't seem necessary when there are only two of them.

File header:

- Signature, 4 bytes, "NTEX"
- Container version, 4 bytes. Current version is 0x100 because the previous binary-only NTC format had the same signature and its versions went up to 19.
- JSON chunk offset, 8 bytes.
- JSON chunk size, 8 bytes.
- Data chunk offset, 8 bytes.
- Data chunk size, 8 bytes.

Chunk offsets and sizes must be multiples of 4 bytes for efficient access, particularly in the case when an implementation loads an entire NTC container into GPU memory for decompression. If the chunk data is not multiple of 4 bytes, it is padded with zeros until it is.

The JSON descriptor will refer to portions of the data chunk using offsets inside the data chunk.

## JSON Descriptor Schema

### Document object

- `schemaVersion`: int, required, current version value is 1 and should only be incremented when breaking changes are made
- `width`: int, required, width of the texture set in pixels, `1..16384`.
- `height`: int, required, height of the texture set in pixels, `1..16384`.
- `numColorMips`: int, optional(1), number of mip levels in the decompressed, color version of the texture set, `1..log2(max(width, height))+1`.
- `numChannels`: int, required, number of color channels in the texture set, `1..16`.
- `latentShape`: object `LatentShape`, optional.
- `mlp`: object `MLP`, optional. This is a legacy field that should be concatenated with `mlpVersions` when reading for backward compatibility.
- `mlpVersions`: unordered array of object `MLP`, optional.
- `textures`: unordered array of object `Texture`, optional.
- `channels`: ordered array of object `Channel`, optional, if present then count must match `numChannels`.
- `latents`: ordered array of object `LatentImage`, optional.
- `colorMips`: ordered array of object `ColorMip`, required.
- `views`: ordered array of object `BufferView`, required.

### `LatentShape` object

- `highResQuantBits`: int, required, number of quantization bits for the high-res grid, `1, 2, 4, 8`
- `lowResQuantBits`: int, required, number of quantization bits for the low-res grid, `1, 2, 4, 8`
- `highResFeatures`: int, required, number of features in the high-res grid, `4, 8, 12, 16`
- `lowResFeatures`: int, required, number of features in the low-res grid, `4, 8, 12, 16`

Note that `LatentShape` doesn't specify the `gridSizeScale` parameter, as it is redundant (can be derived from the color and latent image sizes) and not really necessary for decompression.

### `MLP` object

- `layers`: ordered array of object `MLPLayer`, required to be non-empty.
- `activation`: `ActivationType`, optional(`HGELUClamp`), curerntly only `HGELUClamp` is supported, applied to all layers except the last one
- `weightLayout`: `MatrixLayout`, optional(`ColumnMajor`), currently only `ColumnMajor` is supported
- `weightType`: `MlpDataType`, required, currently `Int8` and `FloatE4M3` are supported - note that layers may override the type
- `scaleBiasType`: `MlpDataType`, required, currently `Float16` and `Float32` are supported - note that layers may override the type

### `MLPLayer` object

- `inputChannels`: int, required
- `outputChannels`: int, required
- `weightView`: int, required, index of the `BufferView` with the weight data
- `scaleView`: int, optional, index of the `BufferView` with the scale data - when there is no view, all scales are assumed to be 1.0
- `biasView`: int, required, index of the `BufferView` with the bias data
- `weightType`: `MlpDataType`, optional, overrides the type specified in the `MLP` for this layer
- `scaleBiasType`: `MlpDataType`, optional, overrides the type specified in the `MLP` for this layer

### `Texture` object

- `name`: string, required non-empty, user-provided name for this texture within the texture set
- `firstChannel`: int, required, index of the first channel within the texture set that contains this texture's data, `0..document.numChannels-1`
- `numChannels`: int, required, number of channels in this texture, `1..4`
- `channelFormat`: `ChannelFormat`, optional(`UNORM8`), preferred data type for storing decompressed colors for this texture
- `rgbColorSpace`: `ColorSpace`, optional(`Linear`), color space that will be used for storing RGB components of decompressed colors
- `alpaColorSpace`: `ColorSpace`, optional(`Linear`), color space that will be used for storing the Alpha component of decompressed colors, if applicable
- `bcFormat`: `BlockCompressedFormat`, optional(`None`), preferred BCn compression format for transcoding this texture
- `bcQuality`: preferred quality parameter for transcoding into BCn
- `bcAccelerationDataView`: int, optional, index of the `BufferView` object with the BCn transcoding acceleration data

### `Channel` object

- `scale`: float, required, scaling that needs to be applied to inference output to get color
- `bias`: float, required, bias that needs to be applied to inference output after scaling to get color
- `colorSpace`: `ColorSpace`, optional(`Linear`), color space in which the data is encoded after applying scale and bias

### `LatentImage` object

- `highResWidth`: int, required
- `highResHeight`: int, required
- `lowResWidth`: int, required
- `lowResHeight`: int, required
- `highResBitsPerPixel`: int, required, packing stride of high-res latents
- `lowResBitsPerPixel`: int, required, packing stride of low-res latents
- `lowResView`: int, required, index of the `BufferView` object with the high-res latent data
- `highResView`: int, required, index of the `BufferView` object with the low-res latent data

### `ColorMip` object

- `width`: int, optional, normally derived from the texture set width and mip level index: `max(1, textureSetWidth >> mipLevel)`
- `height`: int, optional, normally derived from the texture set height and mip level index: `max(1, textureSetHeight >> mipLevel)`
- `latentMip`: int, optional, index of the latent MIP level that should be used to decompress this color MIP level
- `positionLod`: float, optional, the LOD constant used in positional encoding for this MIP level. Required when `latentMip` is set.
- `positionScale`: float, optional, the position scaling factor used in positional encoding for this MIP level. Required when `latentMip` is set.
- `combinedColorData`: object `ColorImageData`, optional
- `perTextureColorData`: ordered array of object `ColorImageData`, optional, if present then count must be equal to count of `Texture` objects

Note that `ColorMip` supports three storage modes that are NOT mutually exclusive:

- Mode that refers to a latent MIP level and implies NTC decompression
- Mode that includes combined raw color data for all channels
- Mode that includes per-texture raw color or BCn data for each texture

### `ColorImageData` object

- `view`: int, required, index of the `BufferView` object with the color data
- `uncompressedFormat`: `ChannelFormat`, optional(`UNORM8`), storage format for the color data if it's not compressed
- `bcFormat`: `BlockCompressedFormat`, optional(`None`), storage format for the color data if it's block compressed
- `rowPitch`: int, optional, stride in bytes between consecutive rows of color data
- `pixelStride`: int, optional, stride in bytes between consecutive pixels
- `numChannels`: int, number of channels, unless a block compressed format is used

### `BufferView` object

- `offset`: int, required, offset of data within the NTC container data chunk
- `storedSize`: int, required, size of data stored in the NTC container
- `compression`: `Compression`, optional(`None`), specifies the compression algorithm used for this buffer. No algorithms are currently supported
- `uncompressedSize`: int, optional, size of data after decompression, if applicable
