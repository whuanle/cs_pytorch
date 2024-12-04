using TorchSharp;
using static TorchSharp.torch;

using TorchSharp.Modules;
using TorchSharp.Data;


using nn = TorchSharp.torch.nn;
using optim = TorchSharp.torch.optim;
using datasets = TorchSharp.torchvision.datasets;
using transforms = TorchSharp.torchvision.transforms;
using System.Drawing;

public class MLP : nn.Module<Tensor, Tensor>, IDisposable
{
    private readonly int _inputSize;
    private readonly int _hiddenSize;
    private readonly int _numClasses;

    private TorchSharp.Modules.Linear fc1;
    private TorchSharp.Modules.ReLU relu;
    private TorchSharp.Modules.Linear fc2;
    private TorchSharp.Modules.Linear fc3;

    /// <summary>
    /// 
    /// </summary>
    /// <param name="inputSize">输入层大小，图片的宽*高.</param>
    /// <param name="hiddenSize">隐藏层大小.</param>
    /// <param name="outputSize">输出层大小，例如有多少个分类.</param>
    /// <param name="device"></param>
    public MLP(int inputSize, int hiddenSize, int outputSize, Device device) : base(nameof(MLP))
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _numClasses = outputSize;

        // 定义激活函数和线性层
        relu = nn.ReLU();
        fc1 = nn.Linear(inputSize, hiddenSize, device: device);
        fc2 = nn.Linear(hiddenSize, hiddenSize, device: device);
        fc3 = nn.Linear(hiddenSize, outputSize, device: device);

        RegisterComponents();

    }

    public override torch.Tensor forward(torch.Tensor input)
    {
        // 一层一层传递
        // 第一层读取输入，然后传递给激活函数，
        // 第二层读取第一层的输出，然后传递给激活函数，
        // 第三层读取第二层的输出，然后生成输出结果
        var @out = fc1.call(input);
        @out = relu.call(@out);
        @out = fc2.call(@out);
        @out = relu.call(@out);
        @out = fc3.call(@out);
        return @out;
    }

    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);
        fc1.Dispose();
        relu.Dispose();
        fc2.Dispose();
        fc3.Dispose();
    }
}