using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SkiaSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace Maomi.Torch;

/// <summary>
/// Tensor 图片数据集处理.
/// </summary>
public static partial class MM
{
    public static Tensor LoadImage(string filePath)
    {
        SKBitmap bitmap = SKBitmap.Decode(filePath);
        int width = bitmap.Width;
        int height = bitmap.Height;

        // 创建一个浮点型数据数组来存储像素值（每个像素包含3个通道：R, G, B）
        var imageData = new float[3 * height * width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                SKColor pixel = bitmap.GetPixel(x, y);

                int index = (y * width + x);
                imageData[index] = pixel.Red / 255f;      // 将红色通道标准化为[0, 1]
                imageData[height * width + index] = pixel.Green / 255f;  // 将绿色通道标准化为[0, 1]
                imageData[2 * height * width + index] = pixel.Blue / 255f; // 将蓝色通道标准化为[0, 1]
            }
        }

        // 返回形状为 [1, 3, height, width] 的张量
        Tensor imageTensor = torch.tensor(imageData, new long[] { 1, 3, height, width });
        return imageTensor;
    }

    public static Tensor LoadImageA(string filePath)
    {
        SKBitmap bitmap = SKBitmap.Decode(filePath);
        int width = bitmap.Width;
        int height = bitmap.Height;

        // 创建一个浮点型数据数组来存储像素值（每个像素包含3个通道：R, G, B）
        var imageData = new float[3 * height * width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                SKColor pixel = bitmap.GetPixel(x, y);

                int index = (y * width + x);
                imageData[index] = pixel.Red / 255f;      // 将红色通道标准化为[0, 1]
                imageData[height * width + index] = pixel.Green / 255f;  // 将绿色通道标准化为[0, 1]
                imageData[2 * height * width + index] = pixel.Blue / 255f; // 将蓝色通道标准化为[0, 1]
            }
        }

        // 返回形状为 [1, 3, height, width] 的张量
        Tensor imageTensor = torch.tensor(imageData, new long[] { 1, 3, height, width });
        return imageTensor;
    }
}
