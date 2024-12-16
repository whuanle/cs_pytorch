using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using TorchSharp;
using static System.Net.WebRequestMethods;
using static TorchSharp.torch;

namespace Maomi.Torch;

/// <summary>
/// Tensor 图片数据集处理.
/// </summary>
public static partial class MM
{
    /// <summary>
    /// 将张量数据转换为 <see cref="SixLabors.ImageSharp.Image"/> 对象.
    /// </summary>
    /// <param name="imageTensor"></param>
    /// <returns></returns>
    public static SixLabors.ImageSharp.Image ToImage(this Tensor imageTensor)
    {
        // 将张量数据转换为 byte 数组
        var imageSize = imageTensor.shape[1];
        var byteArray = new byte[imageSize * imageSize];
        var imageData = imageTensor.data<float>();

        var image = new Image<L8>((int)imageSize, (int)imageSize);
        for (int y = 0; y < imageSize; y++)
        {
            for (int x = 0; x < imageSize; x++)
            {
                var pixelValue = (byte)(imageData[y * imageSize + x] * 255);
                image[x, y] = new L8(pixelValue);
            }
        }

        return image;
    }

    public static Tensor LoadImage(string filePath)
    {
        // 读取图片
        using var image = Image.Load<L8>(filePath);

        // 获取图像的尺寸
        int width = image.Width;
        int height = image.Height;

        // 创建一个浮点型数据数组来存储像素值
        var imageData = new float[width * height];

        // 将图像数据转换为浮点型，并填充到数组中
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var pixel = image[x, y];
                imageData[y * width + x] = pixel.PackedValue / 255f;  // 将数据标准化为[0, 1]
            }
        }

        // 将数据转换为 TorchSharp 的 Tensor
        Tensor imageTensor = torch.tensor(imageData, new long[] { 1, height, width });

        return imageTensor;
    }
    public static Tensor LoadImageRgba32(string filePath)
    {
        // 读取图片
        using var image = Image.Load<Rgba32>(filePath);

        // 获取图像的尺寸
        int width = image.Width;
        int height = image.Height;

        // 创建一个浮点型数据数组来存储像素值
        var imageData = new float[width * height];

        // 将图像数据转换为浮点型，并填充到数组中
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var pixel = image[x, y];
                imageData[y * width + x] = pixel.PackedValue / 255f;  // 将数据标准化为[0, 1]
            }
        }

        // 将数据转换为 TorchSharp 的 Tensor
        Tensor imageTensor = torch.tensor(imageData, new long[] { 1, height, width });

        return imageTensor;
    }
    public static Tensor LoadImage<TPixel>(string filePath)
        where TPixel : unmanaged, IPixel<TPixel>
    {
        // 读取图片
        using var image = Image.Load<Rgb24>(filePath);

        // 获取图像的尺寸
        int width = image.Width;
        int height = image.Height;

        // 创建一个浮点型数据数组来存储像素值（每个像素包含3个通道：R, G, B）
        var imageData = new float[width * height * 3];

        // 将图像数据转换为浮点型，并填充到数组中
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var pixel = image[x, y];
                int index = (y * width + x) * 3;
                imageData[index] = pixel.R / 255f;     // 将红色通道标准化为[0, 1]
                imageData[index + 1] = pixel.G / 255f; // 将绿色通道标准化为[0, 1]
                imageData[index + 2] = pixel.B / 255f; // 将蓝色通道标准化为[0, 1]
            }
        }

        // 将数据转换为 TorchSharp 的 Tensor
        Tensor imageTensor = torch.tensor(imageData, new long[] { 1, height, width });

        return imageTensor;
    }

    public static void SavePng(this Tensor imageTensor, string filePath)
    {
        // 将张量数据转换为 byte 数组
        var imageSize = imageTensor.shape[1];
        var byteArray = new byte[imageSize * imageSize];
        var imageData = imageTensor.data<float>();

        var image = new Image<L8>((int)imageSize, (int)imageSize);
        for (int y = 0; y < imageSize; y++)
        {
            for (int x = 0; x < imageSize; x++)
            {
                var pixelValue = (byte)(imageData[y * imageSize + x] * 255);
                image[x, y] = new L8(pixelValue);
            }
        }

        using var stream = System.IO.File.Create(filePath);

        image.Save(stream, new PngEncoder());
    }

    public static void SaveJpeg(this Tensor imageTensor, string filePath)
    {
        // 将张量数据转换为 byte 数组
        var imageSize = imageTensor.shape[1];
        var byteArray = new byte[imageSize * imageSize];
        var imageData = imageTensor.data<float>();

        var image = new Image<L8>((int)imageSize, (int)imageSize);
        for (int y = 0; y < imageSize; y++)
        {
            for (int x = 0; x < imageSize; x++)
            {
                var pixelValue = (byte)(imageData[y * imageSize + x] * 255);
                image[x, y] = new L8(pixelValue);
            }
        }

        using var stream = System.IO.File.Create(filePath);
        image.Save(stream, new JpegEncoder());
    }

    /// <summary>
    /// 将张量数据保存为图片文件.
    /// </summary>
    /// <param name="imageTensor"></param>
    /// <param name="filePath">图片路径.</param>
    /// <param name="imageEncoder">图像编码器,<see href="https://github.com/SixLabors/ImageSharp/tree/main/src/ImageSharp/Formats"/>.</param>
    public static void SaveImage(this Tensor imageTensor, string filePath, ImageEncoder imageEncoder)
    {
        // 将张量数据转换为 byte 数组
        var imageSize = imageTensor.shape[1];
        var byteArray = new byte[imageSize * imageSize];
        var imageData = imageTensor.data<float>();

        var image = new Image<L8>((int)imageSize, (int)imageSize);
        for (int y = 0; y < imageSize; y++)
        {
            for (int x = 0; x < imageSize; x++)
            {
                var pixelValue = (byte)(imageData[y * imageSize + x] * 255);
                image[x, y] = new L8(pixelValue);
            }
        }

        using var stream = System.IO.File.Create(filePath);
        image.Save(stream, new JpegEncoder());
    }
}
