using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using static TorchSharp.torch;

namespace Maomi.Plot;

public static class PlotExtensions
{
    public static Form Show(this ScottPlot.Plot plot, int width = 400, int height = 300, ImageFormat imageFormat = ImageFormat.Png)
    {

        var tempDir = Path.Combine(Path.GetTempPath(), "ScottPlot");
        if (!Directory.Exists(tempDir))
        {
            Directory.CreateDirectory(tempDir);
        }

        var fileName = DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString() + ".png";

        var imgPath = Path.Combine(tempDir, fileName);

        plot.SavePng(imgPath, width, height);

        return Show(imgPath);

    }

    public static Form Show(string imgPath)
    {
        PictureBox pictureBox = new PictureBox();
        pictureBox.Image = System.Drawing.Image.FromFile(imgPath);
        return Show(pictureBox);
    }
    public static Form Show(Stream stream)
    {
        PictureBox pictureBox = new PictureBox();
        pictureBox.Image = System.Drawing.Image.FromStream(stream);
        return Show(pictureBox);
    }
    public static Form Show(SixLabors.ImageSharp.Image bitmap)
    {
        PictureBox pictureBox = new PictureBox();
        pictureBox.Image = Bitmap.FromStream(new MemoryStream());
        return Show(pictureBox);
    }
    private static Form Show(PictureBox pictureBox)
    {
        pictureBox.SizeMode = PictureBoxSizeMode.Zoom;
        pictureBox.Dock = DockStyle.Fill;

        // 创建并设置ContextMenuStrip
        ContextMenuStrip contextMenuStrip = new ContextMenuStrip();
        ToolStripMenuItem copyMenuItem = new ToolStripMenuItem("复制");
        contextMenuStrip.Items.Add(copyMenuItem);

        // 将ContextMenuStrip关联到PictureBox
        pictureBox.ContextMenuStrip = contextMenuStrip;

        // 处理复制菜单项的点击事件
        copyMenuItem.Click += (sender, e) =>
        {
            if (pictureBox.Image != null)
            {
                // 将图片复制到剪贴板
                Clipboard.SetImage(pictureBox.Image);
            }
            else
            {
                MessageBox.Show("没有图片可复制");
            }
        };

        // 创建一个新的窗口
        var form = new Form();
        form.Text = "图片显示";
        form.ClientSize = new Size(800, 600);
        // 将PictureBox添加到Form中
        form.Controls.Add(pictureBox);

        var t = new Thread(() =>
        {
            Application.Run(form);
        });

        t.SetApartmentState(ApartmentState.STA);
        t.Start();
        return form;
    }
}

public static class TensorImageExtensions
{
    public static Image ToBitmap(this Tensor imageTensor)
    {
        // 将张量数据转换为 byte 数组
        var imageSize = imageTensor.shape[1];
        var byteArray = new byte[imageSize * imageSize];
        var imageData = imageTensor.data<float>();

        for (int i = 0; i < imageData.Count; i++)
        {
            byteArray[i] = (byte)(imageData[i] * 255);
        }

        // 创建图像并将其保存为 PNG 文件
        Bitmap bitmap = new Bitmap((int)imageSize, (int)imageSize, PixelFormat.Format8bppIndexed);


        var bitmapData = bitmap.LockBits(new Rectangle(0, 0, (int)imageSize, (int)imageSize), ImageLockMode.WriteOnly, bitmap.PixelFormat);
        Marshal.Copy(byteArray, 0, bitmapData.Scan0, byteArray.Length);
        bitmap.UnlockBits(bitmapData);

        // 设置调色板
        ColorPalette palette = bitmap.Palette;
        for (int i = 0; i < 256; i++)
        {
            palette.Entries[i] = Color.FromArgb(i, i, i);
        }
        bitmap.Palette = palette;

        return bitmap;
    }
    public static void TensorToPng(Tensor imageTensor, string filePath)
    {
        // 将张量数据转换为 byte 数组
        var imageSize = imageTensor.shape[1];
        var byteArray = new byte[imageSize * imageSize];
        var imageData = imageTensor.data<float>();

        for (int i = 0; i < imageData.Count; i++)
        {
            byteArray[i] = (byte)(imageData[i] * 255);
        }

        // 创建图像并将其保存为 PNG 文件
        using Bitmap bitmap = new Bitmap((int)imageSize, (int)imageSize, PixelFormat.Format8bppIndexed);


        var bitmapData = bitmap.LockBits(new Rectangle(0, 0, (int)imageSize, (int)imageSize), ImageLockMode.WriteOnly, bitmap.PixelFormat);
        Marshal.Copy(byteArray, 0, bitmapData.Scan0, byteArray.Length);
        bitmap.UnlockBits(bitmapData);

        // 设置调色板
        ColorPalette palette = bitmap.Palette;
        for (int i = 0; i < 256; i++)
        {
            palette.Entries[i] = Color.FromArgb(i, i, i);
        }

        bitmap.Palette = palette;
        bitmap.Save("fashion_mnist_image.png", ImageFormat.Png);
    }
}