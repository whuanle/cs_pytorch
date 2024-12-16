using MathNet.Numerics;
using System.Numerics;
using System.Runtime.CompilerServices;
using TorchSharp;
using static TorchSharp.torch;

namespace Maomi.Torch;

public static partial class MM
{
    /// <summary>
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

/// <summary>
/// extensions.
/// </summary>
public static partial class MM
{
    /// <summary>
    /// 张量转换为数组.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="tensor"></param>
    /// <returns></returns>
    /// <exception cref="NotSupportedException"></exception>
    public static T[] ToArray<T>(this torch.Tensor tensor)
    {
        switch (tensor.dtype)
        {
            case torch.ScalarType.Byte:
                return Unsafe.As<T[]>(ToByteArray(tensor));
            case torch.ScalarType.Int8:
                return Unsafe.As<T[]>(ToInt8Array(tensor));
            case torch.ScalarType.Int16:
                return Unsafe.As<T[]>(ToInt16Array(tensor));
            case torch.ScalarType.Int32:
                return Unsafe.As<T[]>(ToInt32Array(tensor));
            case torch.ScalarType.Int64:
                return Unsafe.As<T[]>(ToInt64Array(tensor));
            case torch.ScalarType.Float16:
                return Unsafe.As<T[]>(ToFloat16Array(tensor));
            case torch.ScalarType.Float32:
                return Unsafe.As<T[]>(ToFloat32Array(tensor));
            case torch.ScalarType.Float64:
                return Unsafe.As<T[]>(ToFloat64Array(tensor));
            case torch.ScalarType.ComplexFloat32:
                return Unsafe.As<T[]>(ToComplex32Array(tensor));
            case torch.ScalarType.ComplexFloat64:
                return Unsafe.As<T[]>(ToComplex64Array(tensor));
            case torch.ScalarType.Bool:
                return Unsafe.As<T[]>(ToBoolArray(tensor));
            case torch.ScalarType.BFloat16:
                return Unsafe.As<T[]>(ToBfloat16Array(tensor));
        }

        throw new NotSupportedException($"Unsupported data type {tensor.dtype}");
    }

    public static byte[] ToByteArray(this torch.Tensor tensor)
    {
        byte[] array = new byte[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToByte();
        }

        return array;
    }

    public static sbyte[] ToInt8Array(this torch.Tensor tensor)
    {
        sbyte[] array = new sbyte[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToSByte();
        }

        return array;
    }

    public static Int16[] ToInt16Array(this torch.Tensor tensor)
    {
        Int16[] array = new Int16[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToInt16();
        }

        return array;
    }

    public static Int32[] ToInt32Array(this torch.Tensor tensor)
    {
        Int32[] array = new Int32[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToInt32();
        }

        return array;
    }

    public static Int64[] ToInt64Array(this torch.Tensor tensor)
    {
        Int64[] array = new Int64[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToInt64();
        }

        return array;
    }

    public static Half[] ToFloat16Array(this torch.Tensor tensor)
    {
        Half[] array = new Half[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToHalf();
        }

        return array;
    }

    public static float[] ToFloat32Array(this torch.Tensor tensor)
    {
        float[] array = new float[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToSingle();
        }

        return array;
    }

    public static double[] ToFloat64Array(this torch.Tensor tensor)
    {
        double[] array = new double[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToDouble();
        }

        return array;
    }

    public static Complex32[] ToComplex32Array(this torch.Tensor tensor)
    {
        Complex32[] array = new Complex32[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            var complex = tensor.ToComplex32();
            array[i] = new Complex32(complex.Real, complex.Imaginary);

        }

        return array;
    }

    public static Complex[] ToComplex64Array(this torch.Tensor tensor)
    {
        Complex[] array = new Complex[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            var complex = tensor.ToComplex64();
            array[i] = new Complex(complex.Real, complex.Imaginary);

        }

        return array;
    }

    public static bool[] ToBoolArray(this torch.Tensor tensor)
    {
        bool[] array = new bool[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToBoolean();
        }

        return array;
    }


    public static Scalar[] ToBfloat16Array(this torch.Tensor tensor)
    {
        Scalar[] array = new Scalar[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToBoolean();
        }

        return array;
    }   
}