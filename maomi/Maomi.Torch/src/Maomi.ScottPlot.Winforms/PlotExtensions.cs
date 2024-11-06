using System.Drawing;
using System.Windows.Forms;

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

        PictureBox pictureBox = new PictureBox();
        pictureBox.Image = System.Drawing.Image.FromFile(imgPath);
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
                MessageBox.Show("图片已复制到剪贴板");
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