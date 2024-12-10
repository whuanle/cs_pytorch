这里是一些未整理的内容.





加载图片转换为黑白通道

```

// 加载图片为张量
torch.Tensor image = TensorImageExtensions.LoadImage("0.jpg");
// 定义变换：将图像变为灰度并转换为张量
var transform = transforms.Compose(new torchvision.ITransform[] {
   transforms.Grayscale(outputChannels:1),
   transforms.ConvertImageDtype(ScalarType.Float32)
});

var img = transform.call(image).unsqueeze(0);
img.SaveJpeg("./aaa.jpg");
image = image.reshape(-1, 28 * 28);
```

