using Maomi.Torch;
using Maomi.Torch;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;
using nn = TorchSharp.torch.nn;
using optim = TorchSharp.torch.optim;
using datasets = TorchSharp.torchvision.datasets;
using transforms = TorchSharp.torchvision.transforms;
using TorchSharp.Modules;
using static TorchSharp.torch.optim.lr_scheduler.impl;

Device defaultDevice = default;
if (torch.cuda.is_available())
{
    Console.WriteLine("当前设备支持 GPU");
    defaultDevice = torch.device("cuda", index: 0);
    // 使用 GPU 启动
    torch.set_default_device(defaultDevice);
}
else if (torch.mps_is_available())
{
    Console.WriteLine("当前设备支持 MPS");
    defaultDevice = torch.device("mps", index: 0);
    // 使用 MPS 启动
    torch.set_default_device(defaultDevice);
}
else
{
    defaultDevice = torch.device("cpu");
    // 使用 CPU 启动
    torch.set_default_device(defaultDevice);
}


var default_device = torch.get_default_device();
Console.WriteLine($"当前正在使用 {default_device}");

// 指定训练数据集
var training_data = datasets.FashionMNIST(
    root: "data",   // 数据集在那个目录下
    train: true,    // 加载该数据集，用于训练
    download: true, // 如果数据集不存在，是否下载
    target_transform: transforms.ConvertImageDtype(ScalarType.Float32) // 指定特征和标签转换，将标签转换为Float32
    );

// 指定测试数据集
var test_data = datasets.FashionMNIST(
    root: "data",   // 数据集在那个目录下
    train: false,    // 加载该数据集，用于训练
    download: true, // 如果数据集不存在，是否下载
    target_transform: transforms.ConvertImageDtype(ScalarType.Float32) // 指定特征和标签转换，将标签转换为Float32
    );

// 分批加载图像，打乱顺序
var train_loader = torch.utils.data.DataLoader(training_data, batchSize: 100, shuffle: true);
// 分批加载图像，不打乱顺序
var test_loader = torch.utils.data.DataLoader(test_data, batchSize: 100, shuffle: false);

for (int i = 0; i < training_data.Count; i++)
{
    var dic = training_data.GetTensor(i);
    var img = dic["data"];
    var label = dic["label"];
}
var tinymodel = new TinyModel();


public class TinyModel : nn.Module<Tensor, Tensor>
{
    // 传递给基类的参数是模型的名称
    public TinyModel() : base(nameof(TinyModel))
    {
        linear1 = nn.Linear(100, 200);
        activation = nn.ReLU();
        softmax = nn.Softmax(1);
        linear2 = nn.Linear(200, 10);

        // C# 版本需要调用这个函数，将模型的组件注册到模型中
        RegisterComponents();
    }

    Linear linear1;
    ReLU activation;
    Linear linear2;
    Softmax softmax;

    public override Tensor forward(Tensor input)
    {
        // 将输入一层层处理并传递给下一层
        var x = linear1.call(input);
        x = activation.call(x);
        x = linear2.call(x);
        x = softmax.call(x);
        return x;
    }
}