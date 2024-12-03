using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using static System.Net.WebRequestMethods;
using static TorchSharp.torch;

namespace Maomi.Torch;

/// <summary>
/// Tensor 图片数据集处理.
/// </summary>
public static class TensorImageExtensions
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

        image.Save(filePath, new PngEncoder());
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

        image.Save(filePath, new JpegEncoder());
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

        image.Save(filePath, new JpegEncoder());
    }
}
