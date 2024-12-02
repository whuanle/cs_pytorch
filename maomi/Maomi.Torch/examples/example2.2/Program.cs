using Maomi.Plot;
using Maomi.Torch;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using datasets = TorchSharp.torchvision.datasets;
using transforms = TorchSharp.torchvision.transforms;


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

for (int i = 0; i < training_data.Count; i++)
{
    var dic = training_data.GetTensor(i);
    var img = dic["data"];
    var label = dic["label"];

    if (i > 1)
    {
        break;
    }

    PlotExtensions.Show(img.ToImage());
}
Console.ReadLine();
// 分批加载图像，打乱顺序
var train_loader = torch.utils.data.DataLoader(training_data, batchSize: 100, shuffle: true);
// 分批加载图像，不打乱顺序
var test_loader = torch.utils.data.DataLoader(test_data, batchSize: 100, shuffle: false);


foreach (var item in train_loader)
{
    var images = item["data"];
    var lables = item["label"];
}
