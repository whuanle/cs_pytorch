﻿using Maomi.Torch;
using ScottPlot;
using ScottPlot.PlotStyles;
using SkiaSharp;
using System.Threading.Channels;
using TorchSharp;
using static Tensorboard.TensorShapeProto.Types;
using static TorchSharp.torch;
using static TorchSharp.torchvision;
using model = TorchSharp.torchvision.models;

var modelPath = "C:\\Users\\ASUS\\.cache\\torch\\hub\\checkpoints\\resnet101-5d3b4d8f.pth";
var weights = torch.load(modelPath);

await Task.Delay(5000);

var device = MM.GetOpTimalDevice();
torch.set_default_device(device);

var preprocess = transforms.Compose(
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ConvertImageDtype(torch.ScalarType.Float32),
    transforms.Normalize(means: new double[] { 0.485, 0.456, 0.406 }, stdevs: new double[] { 0.229, 0.224, 0.225 })
    );

// 加载图形并缩放裁剪
var img = MM.LoadImageByChannel3("bobby.jpg", 256, 256);

// 使用转换函数处理图形
img = preprocess.call(img);

img = img.reshape(3, img.shape[2], img.shape[3]);
var batch_t = torch.unsqueeze(img, 0);

var resnet101 = model.resnet101(device: device);
//resnet101.load_state_dict(torch.load("C:\\Users\\ASUS\\.cache\\torch\\hub\\checkpoints\\resnet101-63fe2227.pth").state_dict());
resnet101.eval();


var @out = resnet101.call(batch_t);
@out.print();

List<string> labels = new();
using (StreamReader sr = new StreamReader("imagenet_classes.txt"))
{
    string? line;
    while ((line = sr.ReadLine()) != null)
    {
        labels.Add(line.Trim());
    }
}

var percentage = torch.nn.functional.softmax(@out, dim: 1)[0] * 100;

// 对识别结果和分数进行排序
var (_, indices) = torch.sort(@out, descending: true);

// 输出概率前五的物品名称
for (int i = 0; i < 5; i++)
{
    Console.WriteLine("result:" + labels[(int)indices[0][i]] + ",chance:" + percentage[(int)indices[0][i]].item<float>().ToString());
}