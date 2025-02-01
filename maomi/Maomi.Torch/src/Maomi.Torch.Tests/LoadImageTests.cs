namespace Maomi.Torch.Tests;

public class LoadImageTests
{
    [Fact]
    public void LoadImage_FromFilePath_ReturnsTensor()
    {
        string imagePath = "test_image.jpg";
        var tensor = MM.LoadImage(imagePath);
        Assert.NotNull(tensor);
        Assert.Equal(new long[] { 1, 3, 1280, 720 }, tensor.shape);
    }

    [Fact]
    public void LoadImage_FromStream_ReturnsTensor()
    {
        using var stream = File.OpenRead("test_image.jpg");
        var tensor = MM.LoadImage(stream);
        Assert.NotNull(tensor);
        Assert.Equal(new long[] { 1, 3, 1280, 720 }, tensor.shape);
    }
}
