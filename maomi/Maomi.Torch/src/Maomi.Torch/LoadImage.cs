using SkiaSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace Maomi.Torch;

/// <summary>
/// Tensor 图片数据集处理.
/// </summary>
public static partial class MM
{

    public static Tensor LoadImage(string filePath, int channels, int height, int width)
    {
        using SKBitmap bitmap = SKBitmap.Decode(filePath);

        var imgSize = channels * height * width;
        int take = 1;

        using var dataTensor = torch.zeros(new long[] { take, imgSize }, ScalarType.Byte);
        using var inputTensor = torch.tensor(GetBytesWithoutAlpha(bitmap, channels));

        Tensor finalized = inputTensor;
        var nz = inputTensor.count_nonzero().item<long>();

        if (bitmap.Width != width || bitmap.Height != height)
        {
            var t = inputTensor.reshape(1, channels, bitmap.Height, bitmap.Width);
            finalized = torchvision.transforms.functional.resize(t, height, width).reshape(imgSize);
        }

        dataTensor.index_put_(finalized, TensorIndex.Single(0));
        var result = dataTensor.reshape(take, channels, height, width);
        return result;
    }

    /// <summary>
    /// 使用 3 通道加载图片.
    /// </summary>
    /// <param name="filePath"></param>
    /// <param name="height"></param>
    /// <param name="width"></param>
    /// <returns></returns>
    public static Tensor LoadImageByChannel3(string filePath, int height, int width)
    {
        var channels = 3;
        SKBitmap bitmap = SKBitmap.Decode(filePath);
        var imgSize = channels * height * width;

        // 创建一个浮点型数据数组来存储像素值（每个像素包含3个通道：R, G, B）
        var imageData = new byte[channels * bitmap.Height * bitmap.Width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                SKColor pixel = bitmap.GetPixel(x, y);

                int index = (y * width + x);
                imageData[index] = pixel.Red;            // 红色通道
                imageData[height * width + index] = pixel.Green;  // 绿色通道
                imageData[2 * height * width + index] = pixel.Blue; // 蓝色通道
            }
        }

        using Tensor inputTensor = torch.tensor(imageData, new long[] { 1, channels, bitmap.Height, bitmap.Width });

        using var dataTensor = torch.zeros(new long[] { 1, imgSize }, ScalarType.Byte);

        Tensor finalized = inputTensor;
        var nz = inputTensor.count_nonzero().item<long>();

        if (bitmap.Width != width || bitmap.Height != height)
        {
            var t = inputTensor.reshape(1, channels, bitmap.Height, bitmap.Width);
            finalized = torchvision.transforms.functional.resize(t, height, width).reshape(imgSize);
        }

        dataTensor.index_put_(finalized, TensorIndex.Single(0));
        var result = dataTensor.reshape(1, channels, height, width);
        return result;
    }

    public static List<Tensor> LoadImages(IList<string> images, int batchSize, int channels, int height, int width)
    {
        List<Tensor> tensors = new List<Tensor>();

        var imgSize = channels * height * width;
        bool shuffle = false;

        Random rnd = new Random();
        var indices = !shuffle ?
            Enumerable.Range(0, images.Count).ToArray() :
            Enumerable.Range(0, images.Count).OrderBy(c => rnd.Next()).ToArray();

        for (var i = 0; i < images.Count;)
        {
            var take = Math.Min(batchSize, Math.Max(0, images.Count - i));

            if (take < 1)
            {
                break;
            }

            var dataTensor = torch.zeros(new long[] { take, imgSize }, ScalarType.Byte);

            for (var j = 0; j < take; j++)
            {
                var idx = indices[i++];
                var lblStart = idx * (1 + imgSize);
                var imgStart = lblStart + 1;

                using (var stream = new SKManagedStream(File.OpenRead(images[idx])))
                using (var bitmap = SKBitmap.Decode(stream))
                {
                    using (var inputTensor = torch.tensor(GetBytesWithoutAlpha(bitmap, channels)))
                    {

                        Tensor finalized = inputTensor;

                        var nz = inputTensor.count_nonzero().item<long>();

                        if (bitmap.Width != width || bitmap.Height != height)
                        {
                            var t = inputTensor.reshape(1, channels, bitmap.Height, bitmap.Width);
                            finalized = torchvision.transforms.functional.resize(t, height, width).reshape(imgSize);
                        }

                        dataTensor.index_put_(finalized, TensorIndex.Single(j));
                    }
                }
            }

            tensors.Add(dataTensor.reshape(take, channels, height, width));
            dataTensor.Dispose();
        }

        return tensors;
    }

    private static byte[] GetBytesWithoutAlpha(SKBitmap bitmap, int channels = 3)
    {
        var height = bitmap.Height;
        var width = bitmap.Width;

        var inputBytes = bitmap.Bytes;

        if (bitmap.ColorType == SKColorType.Gray8)
        {
            return inputBytes;
        }

        if (bitmap.BytesPerPixel != 4 && bitmap.BytesPerPixel != 1)
        {
            throw new ArgumentException("Conversion only supports grayscale and ARGB");
        }

        var channelLength = height * width;

        var channelCount = channels;

        int inputBlue = 0, inputGreen = 0, inputRed = 0;
        int outputRed = 0, outputGreen = channelLength, outputBlue = channelLength * 2;

        switch (bitmap.ColorType)
        {
            case SKColorType.Bgra8888:
                inputBlue = 0;
                inputGreen = 1;
                inputRed = 2;
                break;

            default:
                throw new NotImplementedException($"Conversion from {bitmap.ColorType} to bytes");
        }
        var outBytes = new byte[channelCount * channelLength];

        for (int i = 0, j = 0; i < channelLength; i += 1, j += 4)
        {
            outBytes[outputRed + i] = inputBytes[inputRed + j];
            outBytes[outputGreen + i] = inputBytes[inputGreen + j];
            outBytes[outputBlue + i] = inputBytes[inputBlue + j];
        }

        return outBytes;
    }
}
