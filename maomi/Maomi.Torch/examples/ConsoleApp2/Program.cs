using Maomi.Torch;
using ScottPlot;
using SkiaSharp;
using TorchSharp;
using static Tensorboard.TensorShapeProto.Types;
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

// 加载图形并缩放裁剪
var img = MM.LoadImageByChannel3("bobby.jpg", 256, 256);

// 使用转换函数处理图形
img = preprocess.call(img);

var resnet101 = model.resnet101(device: device);
resnet101.eval();

var batch_t = torch.unsqueeze(img, 0);

var @out = resnet101.call(img);
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

(_, Tensor index) = torch.max(@out, 1);

var i = index.ToArray<long>()[0];

var percentage = torch.nn.functional.softmax(@out, dim : 1)[0] * 100;

Console.WriteLine("预测结果:" + labels[(int)i] + ",概率:" + percentage[(int)i].item<float>().ToString());
