using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using static TorchSharp.torch;

namespace Maomi.Torch;

public static partial class MM
{
    /// <summary>
    /// The tensor data into <see cref= "SixLabors.ImageSharp.Image" /> object.<br />
    /// 将张量数据转换为 <see cref="SixLabors.ImageSharp.Image"/> 对象.
    /// </summary>
    /// <param name="imageTensor"></param>
    /// <returns></returns>
    public static SixLabors.ImageSharp.Image ToImageAbgr32(this Tensor imageTensor)
    {
        // 将张量数据转换为 byte 数组
        var imageSize = imageTensor.shape[1];
        var byteArray = new byte[imageSize * imageSize];
        var imageData = imageTensor.data<float>();

        var image = new Image<Abgr32>((int)imageSize, (int)imageSize);
        for (int y = 0; y < imageSize; y++)
        {
            for (int x = 0; x < imageSize; x++)
            {
                var pixelValue = (byte)(imageData[y * imageSize + x] * 255);
                image[x, y] = new Abgr32(pixelValue);
            }
        }

        return image;
    }

    /// <summary>
    /// The tensor data into <see cref= "SixLabors.ImageSharp.Image" /> object.<br />
    /// 将张量数据转换为 <see cref="SixLabors.ImageSharp.Image"/> 对象.
    /// </summary>
    /// <param name="imageTensor"></param>
    /// <returns></returns>
    public static SixLabors.ImageSharp.Image ToImageArgb32(this Tensor imageTensor)
    {
        // 将张量数据转换为 byte 数组
        var imageSize = imageTensor.shape[1];
        var byteArray = new byte[imageSize * imageSize];
        var imageData = imageTensor.data<float>();

        var image = new Image<Argb32>((int)imageSize, (int)imageSize);
        for (int y = 0; y < imageSize; y++)
        {
            for (int x = 0; x < imageSize; x++)
            {
                var pixelValue = (byte)(imageData[y * imageSize + x] * 255);
                image[x, y] = new Argb32(pixelValue);
            }
        }

        return image;
    }

    /// <summary>
    /// The tensor data into <see cref= "SixLabors.ImageSharp.Image" /> object.<br />
    /// 将张量数据转换为 <see cref="SixLabors.ImageSharp.Image"/> 对象.
    /// </summary>
    /// <param name="imageTensor"></param>
    /// <returns></returns>
    public static SixLabors.ImageSharp.Image ToImageL8(this Tensor imageTensor)
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

    /// <summary>
    /// The tensor data into <see cref= "SixLabors.ImageSharp.Image" /> object.<br />
    /// 将张量数据转换为 <see cref="SixLabors.ImageSharp.Image"/> 对象.
    /// </summary>
    /// <param name="imageTensor"></param>
    /// <returns></returns>
    public static SixLabors.ImageSharp.Image ToImageL16(this Tensor imageTensor)
    {
        // 将张量数据转换为 byte 数组
        var imageSize = imageTensor.shape[1];
        var byteArray = new byte[imageSize * imageSize];
        var imageData = imageTensor.data<float>();

        var image = new Image<L16>((int)imageSize, (int)imageSize);
        for (int y = 0; y < imageSize; y++)
        {
            for (int x = 0; x < imageSize; x++)
            {
                var pixelValue = (byte)(imageData[y * imageSize + x] * 255);
                image[x, y] = new L16(pixelValue);
            }
        }

        return image;
    }
}
