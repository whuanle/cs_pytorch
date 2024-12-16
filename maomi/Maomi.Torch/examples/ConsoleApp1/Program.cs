using Maomi.Torch;
using TorchSharp;
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

var img = MM.LoadImage("bobby.jpg");
img = preprocess.call(img);

var resnet101 = model.resnet101(device: device);
resnet101.eval();

var batch_t = torch.unsqueeze(img, 0);

var @out =  resnet101.call(batch_t);
@out.print();