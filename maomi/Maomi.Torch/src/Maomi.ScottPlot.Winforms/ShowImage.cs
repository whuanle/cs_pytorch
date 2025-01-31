using SkiaSharp;
using System.Drawing;
using System.Windows.Forms;
using TorchSharp;

namespace Maomi.Torch;

public static partial class MMS
{
    /// <summary>
    /// Graph and display the data.<br />
    /// 绘制图形.
    /// </summary>
    /// <param name="plot"></param>
    /// <param name="width"></param>
    /// <param name="height"></param>
    /// <param name="imageFormat"></param>
    /// <returns></returns>
    public static Form Show(this ScottPlot.Plot plot, int width = 400, int height = 300, SKEncodedImageFormat imageFormat = SKEncodedImageFormat.Png)
    {

        var tempDir = Path.Combine(Path.GetTempPath(), "ScottPlot");
        if (!Directory.Exists(tempDir))
        {
            Directory.CreateDirectory(tempDir);
        }

        var fileName = DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString() + ".png";

        var imgPath = Path.Combine(tempDir, fileName);

        plot.SavePng(imgPath, width, height);

        return ShowImageToForm(imgPath);
    }

    /// <summary>
    /// Use drawers to display graphics.<br />
    /// 使用窗体显示图形.
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static Form ShowImageToForm(this torch.Tensor tensor)
    {
        var tempName = Path.GetTempFileName() + ".png";
        tensor.SavePng(tempName);
        return ShowImageToForm(tempName);
    }

    /// <summary>
    /// Use drawers to display graphics.<br />
    /// 使用窗体显示图形.
    /// </summary>
    /// <param name="imgPath"></param>
    /// <returns></returns>
    public static Form ShowImageToForm(string imgPath)
    {
        PictureBox pictureBox = new PictureBox();
        pictureBox.Image = System.Drawing.Image.FromFile(imgPath);
        return Show(pictureBox);
    }

    /// <summary>
    /// Use drawers to display graphics.<br />
    /// 使用窗体显示图形.
    /// </summary>
    /// <param name="stream"></param>
    /// <returns></returns>
    public static Form ShowImageToForm(Stream stream)
    {
        PictureBox pictureBox = new PictureBox();
        pictureBox.Image = System.Drawing.Image.FromStream(stream);
        return Show(pictureBox);
    }

    /// <summary>
    /// Use drawers to display graphics.<br />
    /// 使用窗体显示图形.
    /// </summary>
    /// <param name="bitmap"></param>
    /// <returns></returns>
    public static Form ShowImageToForm(Bitmap bitmap)
    {
        PictureBox pictureBox = new PictureBox();
        pictureBox.Image = bitmap;
        return Show(pictureBox);
    }


    private static Form Show(PictureBox pictureBox)
    {
        pictureBox.SizeMode = PictureBoxSizeMode.Zoom;
        pictureBox.Dock = DockStyle.Fill;

        ContextMenuStrip contextMenuStrip = new ContextMenuStrip();
        ToolStripMenuItem copyMenuItem = new ToolStripMenuItem("Copy");
        contextMenuStrip.Items.Add(copyMenuItem);

        pictureBox.ContextMenuStrip = contextMenuStrip;

        copyMenuItem.Click += (sender, e) =>
        {
            if (pictureBox.Image != null)
            {
                Clipboard.SetImage(pictureBox.Image);
            }
            else
            {
                MessageBox.Show("No pictures to copy");
            }
        };

        var form = new Form();
        form.Text = "Image";
        form.ClientSize = new System.Drawing.Size(800, 600);

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
