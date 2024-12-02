using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace Maomi.Torch;

public static class TensorImageExtensions
{
    public static Image ToImage(this Tensor imageTensor)
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

    public static void SavePng(Tensor imageTensor, string filePath)
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

        image.Save("fashion_mnist_image.png", new PngEncoder());
    }
}
