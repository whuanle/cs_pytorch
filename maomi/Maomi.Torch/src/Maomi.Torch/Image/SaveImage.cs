using SkiaSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace Maomi.Torch;

public static partial class MM
{
    /// <summary>
    /// Save the tensor data as a .png image file。<br />
    /// 将张量数据保存为 .png 图片文件.
    /// </summary>
    /// <param name="imageTensor"></param>
    /// <param name="filePath"></param>
    public static void SavePng(this Tensor imageTensor, string filePath)
    {
        SaveImage(imageTensor, filePath, SKEncodedImageFormat.Png);
    }

    /// <summary>
    /// Save the tensor data as a .jpeg image file。<br />
    /// 将张量数据保存为 .jpeg 图片文件.
    /// </summary>
    /// <param name="imageTensor"></param>
    /// <param name="filePath"></param>
    public static void SaveJpeg(this Tensor imageTensor, string filePath)
    {
        SaveImage(imageTensor, filePath, SKEncodedImageFormat.Jpeg);
    }

    /// <summary>
    /// 将张量数据保存为图片文件.
    /// </summary>
    /// <param name="imageTensor"></param>
    /// <param name="filePath">图片路径.</param>
    /// <param name="imageFormat">图像编码器,<see href="https://github.com/SixLabors/ImageSharp/tree/main/src/ImageSharp/Formats"/>.</param>
    public static void SaveImage(this Tensor imageTensor, string filePath, SKEncodedImageFormat imageFormat)
    {
        var shapeSize = imageTensor.shape;

        // N Batch size, number of C channels, H height, and W width.
        // N 批大小、C 通道数、H 高度、W 宽度.
        var (N, C, H, W) = (0L, 0L, 0L, 0L);

        if (shapeSize.Length == 3)
        {
            (C, H, W) = (shapeSize[0], shapeSize[1], shapeSize[2]);
        }
        else if (shapeSize.Length == 4)
        {
            (N, C, H, W) = (shapeSize[0], shapeSize[1], shapeSize[2], shapeSize[3]);
        }
        else
        {
            // 张量数据维度不正确，应为 3 或 4 维.
            throw new ArgumentException("The tensor data dimension is incorrect and should be 3 or 4 dimensional.");
        }

        switch (imageTensor.dtype)
        {
            case torch.ScalarType.Byte:
                ConvertTensorByte(imageTensor, filePath, imageFormat, (int)H, (int)W, (int)C); break;
            case torch.ScalarType.Float32:
                ConvertTensorFloat32(imageTensor, filePath, imageFormat, (int)H, (int)W, (int)C); break;
            case torch.ScalarType.Float64:
                ConvertTensorFloat64(imageTensor, filePath, imageFormat, (int)H, (int)W, (int)C); break;
            default: throw new NotSupportedException($"Unsupported data types {imageTensor.dtype}");
        }
    }

    private static void ConvertTensorByte(Tensor imageTensor, string filePath, SKEncodedImageFormat imageFormat, int height, int width, int channels)
    {
        var flattenedData = imageTensor.data<byte>();
        byte[,,] imageData = new byte[channels, height, width];
        Buffer.BlockCopy(flattenedData.ToArray(), 0, imageData, 0, (int)flattenedData.Count * sizeof(byte));

        using (var bitmap = new SKBitmap(width, height, SKColorType.Bgra8888, SKAlphaType.Unpremul))
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    byte r = (byte)(imageData[0, y, x] * 255.0f);
                    byte g = (byte)(imageData[1, y, x] * 255.0f);
                    byte b = (byte)(imageData[2, y, x] * 255.0f);

                    var color = new SKColor(r, g, b);
                    bitmap.SetPixel(x, y, color);
                }
            }

            using (var image = SKImage.FromBitmap(bitmap))
            using (var data = image.Encode(imageFormat, 100))
            {
                using (var stream = File.OpenWrite(filePath))
                {
                    data.SaveTo(stream);
                }
            }
        }
    }

    private static void ConvertTensorFloat32(Tensor imageTensor, string filePath, SKEncodedImageFormat imageFormat, int height, int width, int channels)
    {
        var flattenedData = imageTensor.data<float>();
        float[,,] imageData = new float[channels, height, width];
        Buffer.BlockCopy(flattenedData.ToArray(), 0, imageData, 0, (int)flattenedData.Count * sizeof(float));

        using (var bitmap = new SKBitmap(width, height, SKColorType.Bgra8888, SKAlphaType.Unpremul))
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    byte r = (byte)(imageData[0, y, x] * 255.0f);
                    byte g = (byte)(imageData[1, y, x] * 255.0f);
                    byte b = (byte)(imageData[2, y, x] * 255.0f);

                    var color = new SKColor(r, g, b);
                    bitmap.SetPixel(x, y, color);
                }
            }

            using (var image = SKImage.FromBitmap(bitmap))
            using (var data = image.Encode(imageFormat, 100))
            {
                using (var stream = File.OpenWrite(filePath))
                {
                    data.SaveTo(stream);
                }
            }
        }
    }

    private static void ConvertTensorFloat64(Tensor imageTensor, string filePath, SKEncodedImageFormat imageFormat, int height, int width, int channels)
    {
        var flattenedData = imageTensor.data<double>();
        double[,,] imageData = new double[channels, height, width];
        Buffer.BlockCopy(flattenedData.ToArray(), 0, imageData, 0, (int)flattenedData.Count * sizeof(double));

        using (var bitmap = new SKBitmap(width, height, SKColorType.Bgra8888, SKAlphaType.Unpremul))
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    byte r = (byte)(imageData[0, y, x] * 255.0f);
                    byte g = (byte)(imageData[1, y, x] * 255.0f);
                    byte b = (byte)(imageData[2, y, x] * 255.0f);

                    var color = new SKColor(r, g, b);
                    bitmap.SetPixel(x, y, color);
                }
            }

            using (var image = SKImage.FromBitmap(bitmap))
            using (var data = image.Encode(imageFormat, 100))
            {
                using (var stream = File.OpenWrite(filePath))
                {
                    data.SaveTo(stream);
                }
            }
        }
    }
}
