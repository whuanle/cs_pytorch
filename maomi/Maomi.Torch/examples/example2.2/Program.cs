using Maomi.Plot;
using System.Drawing;
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

    var h = TensorToBitmap(img);
    PlotExtensions.Show(h);
}

// 分批加载图像，打乱顺序
var train_loader = torch.utils.data.DataLoader(training_data, batchSize: 100, shuffle: true);
// 分批加载图像，不打乱顺序
var test_loader = torch.utils.data.DataLoader(test_data, batchSize: 100, shuffle: false);


foreach (var item in train_loader)
{
    var images = item["data"];
    var lables = item["label"];
}
Bitmap TensorToBitmap(Tensor tensor)
{
    var data = tensor.data<float>();
    var height = tensor.shape[1];
    var width = tensor.shape[2];

    var bitmap = new Bitmap((int)width, (int)height);
    var index = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            var r = (int)(data[index++] * 255);
            var g = (int)(data[index++] * 255);
            var b = (int)(data[index++] * 255);
            var color = Color.FromArgb(r, g, b);
            bitmap.SetPixel(x, y, color);
        }
    }

    return bitmap;
}
/*
 import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
 */