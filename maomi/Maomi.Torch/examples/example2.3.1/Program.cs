using TorchSharp;
using static TorchSharp.torch;

using TorchSharp.Modules;
using TorchSharp.Data;


using nn = TorchSharp.torch.nn;
using optim = TorchSharp.torch.optim;
using datasets = TorchSharp.torchvision.datasets;
using transforms = TorchSharp.torchvision.transforms;
using System.Drawing;
using Maomi.Torch;
using static TorchSharp.torch.nn;


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

// 输入层大小，按图片的宽高计算
var input_size = 28 * 28;

// 隐藏层大小，大小不固定，可以自己调整
var hidden_size = 512;

// 手动配置分类结果个数
var num_classes = 10;

var model = new MLP(input_size, hidden_size, num_classes, defaultDevice);
model.load("mnist_mlp_model.pkl");

// 把模型转为test模式
model.eval();

// 加载图片为张量
torch.Tensor image = Maomi.Torch.MM.LoadImage("0.jpg");
image = image.reshape(-1, 28 * 28);

var transform = transforms.ConvertImageDtype(ScalarType.Float32);

var img = transform.call(image).unsqueeze(0);

using (torch.no_grad())
{
    var oputput = model.call(image);
    var prediction = oputput.argmax(dim: 1, keepdim: true);
    Console.WriteLine("Predicted Digit: " + prediction.item<long>().ToString());
}

Console.ReadLine();