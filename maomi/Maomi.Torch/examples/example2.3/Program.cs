using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;

using TorchSharp.Modules;
using TorchSharp.Data;


using nn = TorchSharp.torch.nn;
using optim = TorchSharp.torch.optim;
using datasets = TorchSharp.torchvision.datasets;
using transforms = TorchSharp.torchvision.transforms;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;

//// 使用 GPU 启动
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

// 1. 加载数据集

// 从 MNIST 数据集下载数据或者加载已经下载的数据
using var train_data = datasets.MNIST("./mnist/data", train: true, download: true, target_transform: transforms.ConvertImageDtype(ScalarType.Float32));
using var test_data = datasets.MNIST("./mnist/data", train: false, download: true, target_transform: transforms.ConvertImageDtype(ScalarType.Float32));

Console.WriteLine("Train data size: " + train_data.Count);
Console.WriteLine("Test data size: " + test_data.Count);

var batch_size = 100;
// 分批加载图像，打乱顺序
var train_loader = torch.utils.data.DataLoader(train_data, batchSize: batch_size, shuffle: true, defaultDevice);

// 分批加载图像，不打乱顺序
var test_loader = torch.utils.data.DataLoader(test_data, batchSize: batch_size, shuffle: false, defaultDevice);

// 输入层大小，按图片的宽高计算
var input_size = 28 * 28;

// 隐藏层大小，大小不固定，可以自己调整
var hidden_size = 512;

// 手动配置分类结果个数
var num_classes = 10;

var model = new MLP(input_size, hidden_size, num_classes, defaultDevice);

// 创建损失函数
var criterion = nn.CrossEntropyLoss();

// 学习率
var learning_rate = 0.001;
// 优化器
var optimizer = optim.Adam(model.parameters(), lr: learning_rate);

// 训练的轮数
var num_epochs = 10;

foreach (var epoch in Enumerable.Range(0, num_epochs))
{
    int i = 0;
    foreach (var item in train_loader)
    {
        var images = item["data"];
        var lables = item["label"];

        images = images.reshape(-1, 28 * 28);
        var outputs = model.call(images);

        var loss = criterion.call(outputs, lables);

        optimizer.zero_grad();

        loss.backward();

        optimizer.step();

        i++;
        if ((i + 1) % 300 == 0)
        {
            Console.WriteLine($"Epoch [{(epoch + 1)}/{num_epochs}], Step [{(i + 1)}/{train_data.Count / batch_size}], Loss: {loss.ToSingle():F4}");
        }
    }
}


using (torch.no_grad())
{
    long correct = 0;
    long total = 0;

    foreach (var item in test_loader)
    {
        var images = item["data"];
        var labels = item["label"];

        images = images.reshape(-1, 28 * 28);
        var outputs = model.call(images);

        var (_, predicted) = torch.max(outputs, 1);
        total += labels.size(0);
        correct += (predicted == labels).sum().item<long>();
    }
    Console.WriteLine($"Accuracy of the network on the 10000 test images: {100 * correct / total} %");
}


model.save("mnist_mlp_model.pkl");

Console.ReadLine();
