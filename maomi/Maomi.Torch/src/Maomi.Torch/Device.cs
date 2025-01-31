using TorchSharp;
using static TorchSharp.torch;

namespace Maomi.Torch;

public static partial class MM
{
    /// <summary>
    /// Get the best equipment.<br />
    /// 获取最优的设备.
    /// </summary>
    /// <returns></returns>
    public static torch.Device GetOpTimalDevice()
    {
        Device defaultDevice = default!;
        if (torch.cuda.is_available())
        {
            defaultDevice = torch.device("cuda", index: 0);
        }
        else if (torch.mps_is_available())
        {
            defaultDevice = torch.device("mps", index: 0);
        }
        else
        {
            defaultDevice = torch.device("cpu");
        }

        return defaultDevice;
    }
}