using MathNet.Numerics;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using TorchSharp;

namespace Maomi.Torch;

/// <summary>
/// extensions.
/// </summary>
public static class TensorPrintExtensions
{
    /// <summary>
    /// Uses Console.WriteLine to print a array expression on stdout.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="array"></param>
    public static void print<T>(this T[] array)
        where T : struct
    {
        Console.Write("[");
        for (int i = 0; i < array.Length; i++)
        {
            Console.Write(array[i]);
            if (i < array.Length - 1)
            {
                Console.Write(", ");
            }
        }
        Console.WriteLine("]");
    }

    /// <summary>
    /// Uses Console.WriteLine to print a tensor expression on stdout. This is intendedfor interactive notebook use, primarily.
    /// </summary>
    /// <param name="t">The input tensor.</param>
    /// <param name="fltFormat">The format string to use for floating point values. See: https://learn.microsoft.com/en-us/dotnet/standard/base-types/standard-numeric-format-strings</param>
    /// <param name="width">The width of each line of the output string.</param>
    /// <param name="newLine">The newline string to use, defaults to system default.</param>
    /// <param name="cultureInfo">The culture info to be used when formatting the numbers.</param>
    public static void print_numpy(this torch.Tensor t, string? fltFormat = "g5", int? width = 100, string? newLine = "", CultureInfo? cultureInfo = null)
    {
        t.print(fltFormat, width, newLine, cultureInfo, TensorStringStyle.Numpy);
    }

    /// <summary>
    /// Uses Console.WriteLine to print a tensor expression on stdout. This is intendedfor interactive notebook use, primarily.
    /// </summary>
    /// <param name="t">The input tensor.</param>
    /// <param name="fltFormat">The format string to use for floating point values. See: https://learn.microsoft.com/en-us/dotnet/standard/base-types/standard-numeric-format-strings</param>
    /// <param name="width">The width of each line of the output string.</param>
    /// <param name="newLine">The newline string to use, defaults to system default.</param>
    /// <param name="cultureInfo">The culture info to be used when formatting the numbers.</param>
    public static void print_csharp(this torch.Tensor t, string? fltFormat = "g5", int? width = 100, string? newLine = "", CultureInfo? cultureInfo = null)
    {
        t.print(fltFormat, width, newLine, cultureInfo, TensorStringStyle.CSharp);
    }

    /// <summary>
    /// Uses Console.WriteLine to print a tensor expression on stdout. This is intendedfor interactive notebook use, primarily.
    /// </summary>
    /// <param name="t">The input tensor.</param>
    /// <param name="fltFormat">The format string to use for floating point values. See: https://learn.microsoft.com/en-us/dotnet/standard/base-types/standard-numeric-format-strings</param>
    /// <param name="width">The width of each line of the output string.</param>
    /// <param name="newLine">The newline string to use, defaults to system default.</param>
    /// <param name="cultureInfo">The culture info to be used when formatting the numbers.</param>
    public static void print_metadata(this torch.Tensor t, string? fltFormat = "g5", int? width = 100, string? newLine = "", CultureInfo? cultureInfo = null)
    {
        t.print(fltFormat, width, newLine, cultureInfo, TensorStringStyle.Metadata);
    }

    /// <summary>
    /// Uses Console.WriteLine to print a tensor expression on stdout. This is intendedfor interactive notebook use, primarily.
    /// </summary>
    /// <param name="t">The input tensor.</param>
    /// <param name="fltFormat">The format string to use for floating point values. See: https://learn.microsoft.com/en-us/dotnet/standard/base-types/standard-numeric-format-strings</param>
    /// <param name="width">The width of each line of the output string.</param>
    /// <param name="newLine">The newline string to use, defaults to system default.</param>
    /// <param name="cultureInfo">The culture info to be used when formatting the numbers.</param>
    public static void print_julia(this torch.Tensor t, string? fltFormat = "g5", int? width = 100, string? newLine = "", CultureInfo? cultureInfo = null)
    {
        t.print(fltFormat, width, newLine, cultureInfo, TensorStringStyle.Julia);
    }
}



/// <summary>
/// extensions.
/// </summary>
public static class TensorArrayExtensions
{
    public static T[] to_array<T>(this torch.Tensor tensor)
    {
        switch (tensor.dtype)
        {
            case torch.ScalarType.Byte:
                return Unsafe.As<T[]>(to_byte_array(tensor));
            case torch.ScalarType.Int8:
                return Unsafe.As<T[]>(to_int8_array(tensor));
            case torch.ScalarType.Int16:
                return Unsafe.As<T[]>(to_int16_array(tensor));
            case torch.ScalarType.Int32:
                return Unsafe.As<T[]>(to_int32_array(tensor));
            case torch.ScalarType.Int64:
                return Unsafe.As<T[]>(to_int64_array(tensor));
            case torch.ScalarType.Float16:
                return Unsafe.As<T[]>(to_float16_array(tensor));
            case torch.ScalarType.Float32:
                return Unsafe.As<T[]>(to_float32_array(tensor));
            case torch.ScalarType.Float64:
                return Unsafe.As<T[]>(to_float64_array(tensor));
            case torch.ScalarType.ComplexFloat32:
                return Unsafe.As<T[]>(to_complex32_array(tensor));
            case torch.ScalarType.ComplexFloat64:
                return Unsafe.As<T[]>(to_complex64_array(tensor));
            case torch.ScalarType.Bool:
                return Unsafe.As<T[]>(to_bool_array(tensor));
            case torch.ScalarType.BFloat16:
                return Unsafe.As<T[]>(to_bfloat16_array(tensor));
        }

        throw new NotSupportedException($"Unsupported data type {tensor.dtype}");
    }

    public static byte[] to_byte_array(this torch.Tensor tensor)
    {
        byte[] array = new byte[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToByte();
        }

        return array;
    }

    public static sbyte[] to_int8_array(this torch.Tensor tensor)
    {
        sbyte[] array = new sbyte[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToSByte();
        }

        return array;
    }

    public static Int16[] to_int16_array(this torch.Tensor tensor)
    {
        Int16[] array = new Int16[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToInt16();
        }

        return array;
    }

    public static Int32[] to_int32_array(this torch.Tensor tensor)
    {
        Int32[] array = new Int32[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToInt32();
        }

        return array;
    }

    public static Int64[] to_int64_array(this torch.Tensor tensor)
    {
        Int64[] array = new Int64[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToInt64();
        }

        return array;
    }

    public static Half[] to_float16_array(this torch.Tensor tensor)
    {
        Half[] array = new Half[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToHalf();
        }

        return array;
    }

    public static float[] to_float32_array(this torch.Tensor tensor)
    {
        float[] array = new float[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToSingle();
        }

        return array;
    }

    public static double[] to_float64_array(this torch.Tensor tensor)
    {
        double[] array = new double[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToDouble();
        }

        return array;
    }

    public static Complex32[] to_complex32_array(this torch.Tensor tensor)
    {
        Complex32[] array = new Complex32[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            var complex = tensor.ToComplex32();
            array[i] = new Complex32(complex.Real, complex.Imaginary);

        }

        return array;
    }

    public static Complex[] to_complex64_array(this torch.Tensor tensor)
    {
        Complex[] array = new Complex[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            var complex = tensor.ToComplex64();
            array[i] = new Complex(complex.Real, complex.Imaginary);

        }

        return array;
    }

    public static bool[] to_bool_array(this torch.Tensor tensor)
    {
        bool[] array = new bool[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToBoolean();
        }

        return array;
    }


    public static Scalar[] to_bfloat16_array(this torch.Tensor tensor)
    {
        Scalar[] array = new Scalar[tensor.size()[0]];

        for (int i = 0; i < array.Length; i++)
        {
            array[i] = tensor[i].ToBoolean();
        }

        return array;
    }   
}