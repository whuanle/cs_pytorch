using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace Maomi.Torch;

/// <summary>
/// 使用 System.Drawing.Imaging 命名空间中的类将张量数据转换为图像，只能在 Windows 下使用.
/// </summary>
public static partial class MMS
{
    /// <summary>
    /// 张量转换为位图.
    /// </summary>
    /// <param name="imageTensor"></param>
    /// <returns></returns>
    public static Bitmap ToBitmap(this Tensor imageTensor)
    {
        // 将张量数据转换为 byte 数组
        var imageSize = imageTensor.shape[1];
        var byteArray = new byte[imageSize * imageSize];
        var imageData = imageTensor.data<float>();

        for (int i = 0; i < imageData.Count; i++)
        {
            byteArray[i] = (byte)(imageData[i] * 255);
        }

        Bitmap bitmap = new Bitmap((int)imageSize, (int)imageSize, PixelFormat.Format8bppIndexed);

        var bitmapData = bitmap.LockBits(new Rectangle(0, 0, (int)imageSize, (int)imageSize), ImageLockMode.WriteOnly, bitmap.PixelFormat);
        Marshal.Copy(byteArray, 0, bitmapData.Scan0, byteArray.Length);
        bitmap.UnlockBits(bitmapData);

        // 设置调色板
        ColorPalette palette = bitmap.Palette;
        for (int i = 0; i < 256; i++)
        {
            palette.Entries[i] = Color.FromArgb(i, i, i);
        }
        bitmap.Palette = palette;

        return bitmap;
    }

    public static void DrawingSavePng(this Tensor imageTensor, string filePath)
    {
        // 将张量数据转换为 byte 数组
        var imageSize = imageTensor.shape[1];
        var byteArray = new byte[imageSize * imageSize];
        var imageData = imageTensor.data<float>();

        for (int i = 0; i < imageData.Count; i++)
        {
            byteArray[i] = (byte)(imageData[i] * 255);
        }

        // 创建图像并将其保存为 PNG 文件
        using Bitmap bitmap = new Bitmap((int)imageSize, (int)imageSize, PixelFormat.Format8bppIndexed);


        var bitmapData = bitmap.LockBits(new Rectangle(0, 0, (int)imageSize, (int)imageSize), ImageLockMode.WriteOnly, bitmap.PixelFormat);
        Marshal.Copy(byteArray, 0, bitmapData.Scan0, byteArray.Length);
        bitmap.UnlockBits(bitmapData);

        // 设置调色板
        ColorPalette palette = bitmap.Palette;
        for (int i = 0; i < 256; i++)
        {
            palette.Entries[i] = Color.FromArgb(i, i, i);
        }

        bitmap.Palette = palette;
        bitmap.Save(filePath, System.Drawing.Imaging.ImageFormat.Png);
    }

    public static void DrawingSaveJpeg(this Tensor imageTensor, string filePath)
    {
        // 将张量数据转换为 byte 数组
        var imageSize = imageTensor.shape[1];
        var byteArray = new byte[imageSize * imageSize];
        var imageData = imageTensor.data<float>();

        for (int i = 0; i < imageData.Count; i++)
        {
            byteArray[i] = (byte)(imageData[i] * 255);
        }

        // 创建图像并将其保存为 PNG 文件
        using Bitmap bitmap = new Bitmap((int)imageSize, (int)imageSize, PixelFormat.Format8bppIndexed);


        var bitmapData = bitmap.LockBits(new Rectangle(0, 0, (int)imageSize, (int)imageSize), ImageLockMode.WriteOnly, bitmap.PixelFormat);
        Marshal.Copy(byteArray, 0, bitmapData.Scan0, byteArray.Length);
        bitmap.UnlockBits(bitmapData);

        // 设置调色板
        ColorPalette palette = bitmap.Palette;
        for (int i = 0; i < 256; i++)
        {
            palette.Entries[i] = Color.FromArgb(i, i, i);
        }

        bitmap.Palette = palette;
        bitmap.Save(filePath, System.Drawing.Imaging.ImageFormat.Jpeg);
    }

    /// <summary>
    /// 将张量数据保存为图像文件.
    /// </summary>
    /// <param name="imageTensor"></param>
    /// <param name="filePath"></param>
    /// <param name="imageFormat">例如 <see cref="System.Drawing.Imaging.ImageFormat.Jpeg"/></param>
    public static void DrawingSaveImage(this Tensor imageTensor, string filePath, System.Drawing.Imaging.ImageFormat imageFormat)
    {
        // 将张量数据转换为 byte 数组
        var imageSize = imageTensor.shape[1];
        var byteArray = new byte[imageSize * imageSize];
        var imageData = imageTensor.data<float>();

        for (int i = 0; i < imageData.Count; i++)
        {
            byteArray[i] = (byte)(imageData[i] * 255);
        }

        // 创建图像并将其保存为 PNG 文件
        using Bitmap bitmap = new Bitmap((int)imageSize, (int)imageSize, PixelFormat.Format8bppIndexed);


        var bitmapData = bitmap.LockBits(new Rectangle(0, 0, (int)imageSize, (int)imageSize), ImageLockMode.WriteOnly, bitmap.PixelFormat);
        Marshal.Copy(byteArray, 0, bitmapData.Scan0, byteArray.Length);
        bitmap.UnlockBits(bitmapData);

        // 设置调色板
        ColorPalette palette = bitmap.Palette;
        for (int i = 0; i < 256; i++)
        {
            palette.Entries[i] = Color.FromArgb(i, i, i);
        }

        bitmap.Palette = palette;
        bitmap.Save(filePath, imageFormat);
    }
}