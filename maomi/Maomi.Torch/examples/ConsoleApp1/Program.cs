using static TorchSharp.torchvision;
using model = TorchSharp.torchvision.models;
var resnet101 = model.resnet101();
foreach (var item in resnet101.parameters())
{
    Console.WriteLine();
}
