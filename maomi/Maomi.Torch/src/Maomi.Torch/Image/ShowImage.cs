using System.Diagnostics;
using static TorchSharp.torch;

namespace Maomi.Torch;

public static partial class MM
{
    /// <summary>
    /// Show the image in the system default image viewer.<br />
    /// </summary>
    /// <param name="tensor"></param>
    public static void ShowImage(this Tensor tensor)
    {
        var tempName = Path.GetTempFileName() + ".png";

        tensor.SavePng(tempName);
        Process.Start(new ProcessStartInfo(tempName) { UseShellExecute = true });
    }
}
