using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace Maomi.Torch;
public static partial class MM
{
    public static void SavePng(this Tensor imageTensor, string filePath)
    {
        SaveImage(imageTensor, filePath, new PngEncoder());
    }

    public static void SaveJpeg(this Tensor imageTensor, string filePath)
    {
        SaveImage(imageTensor,filePath, new JpegEncoder());
    }

    /// <summary>
    /// 将张量数据保存为图片文件.
    /// </summary>
    /// <param name="imageTensor"></param>
    /// <param name="filePath">图片路径.</param>
    /// <param name="imageEncoder">图像编码器,<see href="https://github.com/SixLabors/ImageSharp/tree/main/src/ImageSharp/Formats"/>.</param>
    public static void SaveImage(this Tensor imageTensor, string filePath, ImageEncoder imageEncoder)
    {
        var shapeSize = imageTensor.shape;

        // N 批大小、C 通道数、H 高度、W 宽度
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
            throw new ArgumentException("张量数据维度不正确，应为 3 或 4 维");
        }

        // 将张量数据转换为 byte 数组
        var byteArray = new byte[W * H];
        var imageData = imageTensor.data<float>();

        var image = new Image<L8>((int)W, (int)H);
        for (int y = 0; y < H; y++)
        {
            for (int x = 0; x < W; x++)
            {
                var pixelValue = (byte)(imageData[y * W + x] * 255);
                image[x, y] = new L8(pixelValue);
            }
        }

        using var stream = System.IO.File.Create(filePath);
        image.Save(stream, imageEncoder);
    }
}
