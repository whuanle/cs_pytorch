#pragma warning disable CS1591 // 缺少对公共可见类型或成员的 XML 注释

using System.Security.Cryptography;
using TorchSharp.Modules;

namespace Maomi.Torch;
public static partial class MM
{
    private const string ReposityBase = "https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/";

    public static AlexNet LoadModel(this AlexNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadModelAsync(net, strict, skip, loadedParameters).Result;
    }

    public static async Task<AlexNet> LoadModelAsync(this AlexNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/alexnet.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "alexnet.dat");
        var tempFilePath = $"alexnet_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "alexnet.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as AlexNet)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "alexnet.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as AlexNet)!;
    }
    public static GoogleNet LoadModel(this GoogleNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadModelAsync(net, strict, skip, loadedParameters).Result;
    }
    public static async Task<GoogleNet> LoadModelAsync(this GoogleNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/googlenet.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "googlenet.dat");
        var tempFilePath = $"googlenet_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "googlenet.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as GoogleNet)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "googlenet.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as GoogleNet)!;
    }
    public static InceptionV3 LoadModel(this InceptionV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadModelAsync(net, strict, skip, loadedParameters).Result;
    }
    public static async Task<InceptionV3> LoadModelAsync(this InceptionV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/inception_v3.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "inception_v3.dat");
        var tempFilePath = $"inception_v3_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "inception_v3.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as InceptionV3)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "inception_v3.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as InceptionV3)!;
    }

    public static MobileNetV2 LoadModel(this MobileNetV2 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadModelAsync(net, strict, skip, loadedParameters).Result;
    }

    public static async Task<MobileNetV2> LoadModelAsync(this MobileNetV2 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/mobilenet_v2.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "mobilenet_v2.dat");
        var tempFilePath = $"mobilenet_v2_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "mobilenet_v2.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV2)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "mobilenet_v2.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV2)!;
    }

    public static MobileNetV3 LoadMobileNetV3Large(this MobileNetV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadMobileNetV3LargeAsync(net, strict, skip, loadedParameters).Result;
    }

    public static async Task<MobileNetV3> LoadMobileNetV3LargeAsync(this MobileNetV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/mobilenet_v3_large.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "mobilenet_v3_large.dat");
        var tempFilePath = $"mobilenet_v3_large_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "mobilenet_v3_large.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "mobilenet_v3_large.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
    }

    public static MobileNetV3 LoadMobileNetV3Small(this MobileNetV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadMobileNetV3SmallAsync(net, strict, skip, loadedParameters).Result;
    }

    public static async Task<MobileNetV3> LoadMobileNetV3SmallAsync(this MobileNetV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/mobilenet_v3_small.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "mobilenet_v3_small.dat");
        var tempFilePath = $"mobilenet_v3_small_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "mobilenet_v3_small.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "mobilenet_v3_small.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
    }

    public static ResNet LoadResnet101(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadResnet101Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<ResNet> LoadResnet101Async(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/resnet101.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "resnet101.dat");
        var tempFilePath = $"resnet101_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "resnet101.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "resnet101.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static ResNet LoadResnet152(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadResnet152Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<ResNet> LoadResnet152Async(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/resnet152.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "resnet152.dat");
        var tempFilePath = $"resnet152_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "resnet152.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "resnet152.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static ResNet LoadResnet18(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadResnet18Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<ResNet> LoadResnet18Async(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/resnet18.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "resnet18.dat");
        var tempFilePath = $"resnet18_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "resnet18.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "resnet18.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static ResNet LoadResnet34(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadResnet34Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<ResNet> LoadResnet34Async(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/resnet34.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "resnet34.dat");
        var tempFilePath = $"resnet34_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "resnet34.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "resnet34.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static ResNet LoadResnet50(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadResnet50Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<ResNet> LoadResnet50Async(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/resnet50.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "resnet50.dat");
        var tempFilePath = $"resnet50_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "resnet50.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "resnet50.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static VGG LoadVG11(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVG11Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<VGG> LoadVG11Async(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/vgg11.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg11.dat");
        var tempFilePath = $"vgg11_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "vgg11.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "vgg11.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
    }

    public static VGG LoadVG11BN(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVG11BNAsync(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<VGG> LoadVG11BNAsync(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/vgg11_bn.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg11_bn.dat");
        var tempFilePath = $"vgg11_bn_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "vgg11_bn.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "vgg11_bn.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
    }

    public static VGG LoadVG13(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVG13Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<VGG> LoadVG13Async(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/vgg13.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg13.dat");
        var tempFilePath = $"vgg13_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "vgg13.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "vgg13.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
    }

    public static VGG LoadVG13BN(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVG13BNAsync(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<VGG> LoadVG13BNAsync(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/vgg13_bn.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg13_bn.dat");
        var tempFilePath = $"vgg13_bn_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "vgg13_bn.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "vgg13_bn.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
    }

    public static VGG LoadVG16(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVG16Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<VGG> LoadVG16Async(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/vgg16.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg16.dat");
        var tempFilePath = $"vgg16_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "vgg16.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "vgg16.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
    }


    public static VGG LoadVG16BN(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVG16BNAsync(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<VGG> LoadVG16BNAsync(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/vgg16_bn.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg16_bn.dat");
        var tempFilePath = $"vgg16_bn_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "vgg16_bn.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "vgg16_bn.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
    }

    public static VGG LoadVG19(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVG19Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<VGG> LoadVG19Async(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/vgg19.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg19.dat");
        var tempFilePath = $"vgg19_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "vgg19.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "vgg19.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
    }


    public static VGG LoadVG19BN(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVG19BNAsync(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<VGG> LoadVG19BNAsync(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/vgg19_bn.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg19_bn.dat");
        var tempFilePath = $"vgg19_bn_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "vgg19_bn.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
            }

            File.Delete(modelPath);
        }

        using var modelStream = await httpClient.GetStreamAsync(ReposityBase + "vgg19_bn.dat");
        using (var fileStream = File.Create(Path.Combine(torchcsharpDir, tempFilePath)))
        {
            await modelStream.CopyToAsync(fileStream);
        }

        File.Move(tempFilePath, modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
    }

    private static string CalculateMD5(string fpath)
    {
        MD5 md5 = MD5.Create();
        byte[] hash;
        using (var stream = File.OpenRead(fpath))
        {
            hash = md5.ComputeHash(stream);
        }
        return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
    }

    private static async Task<string> GetModelMd5(HttpClient httpClient, string modelFileName)
    {
        var md5 = await httpClient.GetStringAsync(ReposityBase + modelFileName + ".md5");
        return md5;
    }

    private static string CheckPath()
    {
        string userDirectory = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var cacheDir = Path.Combine(userDirectory, ".cache");
        var torchcsharpDir = Path.Combine(userDirectory, ".cache", "torchcsharp");
        if (!Directory.Exists(cacheDir))
        {
            Directory.CreateDirectory(cacheDir);
        }
        if (!Directory.Exists(torchcsharpDir))
        {
            Directory.CreateDirectory(torchcsharpDir);
        }

        return torchcsharpDir;
    }
}
