using Maomi.Torch;
using SkiaSharp;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torchvision;
using model = TorchSharp.torchvision.models;

var device = MM.GetOpTimalDevice();
torch.set_default_device(device);

var preprocess = transforms.Compose(
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ConvertImageDtype(torch.ScalarType.Float32),
    transforms.Normalize(means: new double[] { 0.485, 0.456, 0.406 }, stdevs: new double[] { 0.229, 0.224, 0.225 })
    );

var tensors = LoadImages(new string[] { "bobby.jpg" }, 4, 3, 256, 256);

var first = tensors[0];
var img = preprocess.call(first);

var resnet101 = model.resnet101(device: device);
resnet101.eval();

var batch_t = torch.unsqueeze(img, 0);

var @out =  resnet101.call(img);
@out.print();

static List<Tensor> LoadImages(IList<string> images, int batchSize, int channels, int height, int width)
{
    List<Tensor> tensors = new List<Tensor>();

    var imgSize = channels * height * width;
    bool shuffle = false;

    Random rnd = new Random();
    var indices = !shuffle ?
        Enumerable.Range(0, images.Count).ToArray() :
        Enumerable.Range(0, images.Count).OrderBy(c => rnd.Next()).ToArray();


    // Go through the data and create tensors
    for (var i = 0; i < images.Count;)
    {

        var take = Math.Min(batchSize, Math.Max(0, images.Count - i));

        if (take < 1)
        {
            break;
        }

        var dataTensor = torch.zeros(new long[] { take, imgSize }, ScalarType.Byte);

        // Take
        for (var j = 0; j < take; j++)
        {
            var idx = indices[i++];
            var lblStart = idx * (1 + imgSize);
            var imgStart = lblStart + 1;

            using (var stream = new SKManagedStream(File.OpenRead(images[idx])))
            using (var bitmap = SKBitmap.Decode(stream))
            {
                using (var inputTensor = torch.tensor(GetBytesWithoutAlpha(bitmap)))
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

static byte[] GetBytesWithoutAlpha(SKBitmap bitmap)
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

    var channelCount = 3;

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