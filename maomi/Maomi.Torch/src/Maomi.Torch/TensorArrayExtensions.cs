using MathNet.Numerics;
using System.Numerics;
using System.Runtime.CompilerServices;
using TorchSharp;

namespace Maomi.Torch;

/// <summary>
/// extensions.
/// </summary>
public static partial class MM
{
    /// <summary>
    /// Convert a tensor to an array.
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

    /// <summary>
    /// Convert a tensor to a byte[].<br />
    /// 将张量转换为 byte[].
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static byte[] ToByteArray(this torch.Tensor tensor)
    {
        return tensor.data<byte>().ToArray();
    }

    /// <summary>
    /// Convert a tensor to a sbyte[].<br />
    /// 将张量转换为 sbyte[].
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static sbyte[] ToInt8Array(this torch.Tensor tensor)
    {
        return tensor.data<sbyte>().ToArray();
    }

    /// <summary>
    /// Convert a tensor to a int16[].<br />
    /// 将张量转换为 int16[]
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static Int16[] ToInt16Array(this torch.Tensor tensor)
    {
        return tensor.data<Int16>().ToArray();
    }

    /// <summary>
    /// Convert a tensor to a int32[].<br />
    /// 将张量转换为 int32[].
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static Int32[] ToInt32Array(this torch.Tensor tensor)
    {
        return tensor.data<Int32>().ToArray();
    }

    /// <summary>
    /// Convert a tensor to a int64[].<br />
    /// 将张量转换为 int64[].
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static Int64[] ToInt64Array(this torch.Tensor tensor)
    {
        return tensor.data<Int64>().ToArray();
    }

    /// <summary>
    /// Convert a tensor to a float16[].<br />
    /// 将张量转换为 float16[].
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static Half[] ToFloat16Array(this torch.Tensor tensor)
    {
        return tensor.data<Half>().ToArray();
    }

    /// <summary>
    /// Convert a tensor to a float[].<br />
    /// 将张量转换为 float32[].
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static float[] ToFloat32Array(this torch.Tensor tensor)
    {
        return tensor.data<float>().ToArray();
    }

    /// <summary>
    /// Convert a tensor to a double[].<br />
    /// 将张量转换为 float64[].
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static double[] ToFloat64Array(this torch.Tensor tensor)
    {
        return tensor.data<double>().ToArray();
    }

    /// <summary>
    /// Convert a tensor to a Complex32[].<br />
    /// 将张量转换为 Complex32[].
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
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

    /// <summary>
    /// Convert a tensor to a Complex64[].<br />
    /// 将张量转换为 Complex64[].
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static Complex[] ToComplex64Array(this torch.Tensor tensor)
    {
        return tensor.data<Complex>().ToArray();
    }

    /// <summary>
    /// Convert a tensor to a bool[].<br />
    /// 将张量转换为 bool[].
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static bool[] ToBoolArray(this torch.Tensor tensor)
    {
        return tensor.data<bool>().ToArray();
    }

    /// <summary>
    /// Convert a tensor to a Scalar[].<br />
    /// 将张量转换为 Scalar[].
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
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