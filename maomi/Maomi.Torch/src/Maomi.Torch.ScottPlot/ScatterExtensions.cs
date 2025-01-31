using MathNet.Numerics;
using ScottPlot;
using ScottPlot.Plottables;
using System.Numerics;
using TorchSharp;

namespace Maomi.Torch;

/// <summary>
/// Draw image.
/// </summary>
public static partial class MMS
{
    /// <summary>
    /// Plot the tensor data as a scatter plot.
    /// 将张量数据绘制为散点图.
    /// </summary>
    /// <param name="adder"></param>
    /// <param name="xs"></param>
    /// <param name="ys"></param>
    /// <param name="color"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    /// <exception cref="ArgumentException"></exception>
    /// <exception cref="NotSupportedException"></exception>
    public static ScottPlot.Plottables.Scatter Scatter(this ScottPlot.PlottableAdder adder, torch.Tensor xs, torch.Tensor ys, ScottPlot.Color? color = null)
    {
        var xsSize = xs.size();
        var ysSize = ys.size();

        if (xsSize.Length > 1 || ysSize.Length > 1)
        {
            throw new Exception("Only one-dimensional arrays are supported.");
        }

        if (xsSize[0] != ys.size()[0])
        {
            throw new ArgumentException("X and Y data must have the same length");
        }

        switch (xs.dtype)
        {
            case torch.ScalarType.Byte:
                return ScatterByte(adder, xs, ys, color, xsSize, ysSize);
            case torch.ScalarType.Int8:
                return ScatterInt8(adder, xs, ys, color, xsSize, ysSize);
            case torch.ScalarType.Int16:
                return ScatterInt16(adder, xs, ys, color, xsSize, ysSize);
            case torch.ScalarType.Int32:
                return ScatterInt32(adder, xs, ys, color, xsSize, ysSize);
            case torch.ScalarType.Int64:
                return ScatterInt64(adder, xs, ys, color, xsSize, ysSize);
            case torch.ScalarType.Float16:
                return ScatterFloat16(adder, xs, ys, color, xsSize, ysSize);
            case torch.ScalarType.Float32:
                return ScatterFloat32(adder, xs, ys, color, xsSize, ysSize);
            case torch.ScalarType.Float64:
                return ScatterFloat64(adder, xs, ys, color, xsSize, ysSize);
            case torch.ScalarType.ComplexFloat32:
                return ScatterComplex32(adder, xs, ys, color, xsSize, ysSize);
            case torch.ScalarType.ComplexFloat64:
                return ScatterComplex64(adder, xs, ys, color, xsSize, ysSize);
            case torch.ScalarType.Bool:
                return ScatterBool(adder, xs, ys, color, xsSize, ysSize);
            case torch.ScalarType.BFloat16:
                return ScatterBFloat16(adder, xs, ys, color, xsSize, ysSize);
        }

        throw new NotSupportedException($"Unsupported data type {xs.dtype}");
    }

    private static Scatter ScatterByte(PlottableAdder adder, torch.Tensor xs, torch.Tensor ys, ScottPlot.Color? color, long[] xsSize, long[] ysSize)
    {
        byte[] xsFloat = new byte[xsSize[0]];
        byte[] ysFloat = new byte[ysSize[0]];

        for (int i = 0; i < xsSize[0]; i++)
        {
            xsFloat[i] = xs[i].ToByte();
            ysFloat[i] = ys[i].ToByte();
        }

        return adder.Scatter(xsFloat, ysFloat, color);
    }

    private static Scatter ScatterInt8(PlottableAdder adder, torch.Tensor xs, torch.Tensor ys, ScottPlot.Color? color, long[] xsSize, long[] ysSize)
    {
        sbyte[] xsFloat = new sbyte[xsSize[0]];
        sbyte[] ysFloat = new sbyte[ysSize[0]];

        for (int i = 0; i < xsSize[0]; i++)
        {
            xsFloat[i] = xs[i].ToSByte();
            ysFloat[i] = ys[i].ToSByte();
        }

        return adder.Scatter(xsFloat, ysFloat, color);
    }

    private static Scatter ScatterInt16(PlottableAdder adder, torch.Tensor xs, torch.Tensor ys, ScottPlot.Color? color, long[] xsSize, long[] ysSize)
    {
        Int16[] xsFloat = new Int16[xsSize[0]];
        Int16[] ysFloat = new Int16[ysSize[0]];

        for (int i = 0; i < xsSize[0]; i++)
        {
            xsFloat[i] = xs[i].ToInt16();
            ysFloat[i] = ys[i].ToInt16();
        }

        return adder.Scatter(xsFloat, ysFloat, color);
    }

    private static Scatter ScatterInt32(PlottableAdder adder, torch.Tensor xs, torch.Tensor ys, ScottPlot.Color? color, long[] xsSize, long[] ysSize)
    {
        Int32[] xsFloat = new Int32[xsSize[0]];
        Int32[] ysFloat = new Int32[ysSize[0]];

        for (int i = 0; i < xsSize[0]; i++)
        {
            xsFloat[i] = xs[i].ToInt32();
            ysFloat[i] = ys[i].ToInt32();
        }

        return adder.Scatter(xsFloat, ysFloat, color);
    }

    private static Scatter ScatterInt64(PlottableAdder adder, torch.Tensor xs, torch.Tensor ys, ScottPlot.Color? color, long[] xsSize, long[] ysSize)
    {
        Int64[] xsFloat = new Int64[xsSize[0]];
        Int64[] ysFloat = new Int64[ysSize[0]];

        for (int i = 0; i < xsSize[0]; i++)
        {
            xsFloat[i] = xs[i].ToInt64();
            ysFloat[i] = ys[i].ToInt64();
        }

        return adder.Scatter(xsFloat, ysFloat, color);
    }
    private static Scatter ScatterFloat16(PlottableAdder adder, torch.Tensor xs, torch.Tensor ys, ScottPlot.Color? color, long[] xsSize, long[] ysSize)
    {
        Half[] xsFloat = new Half[xsSize[0]];
        Half[] ysFloat = new Half[ysSize[0]];

        for (int i = 0; i < xsSize[0]; i++)
        {
            xsFloat[i] = xs[i].ToHalf();
            ysFloat[i] = ys[i].ToHalf();
        }

        return adder.Scatter(xsFloat, ysFloat, color);
    }

    private static Scatter ScatterFloat32(PlottableAdder adder, torch.Tensor xs, torch.Tensor ys, ScottPlot.Color? color, long[] xsSize, long[] ysSize)
    {
        float[] xsFloat = new float[xsSize[0]];
        float[] ysFloat = new float[ysSize[0]];

        for (int i = 0; i < xsSize[0]; i++)
        {
            xsFloat[i] = xs[i].ToSingle();
            ysFloat[i] = ys[i].ToSingle();
        }

        return adder.Scatter(xsFloat, ysFloat, color);
    }

    private static Scatter ScatterFloat64(PlottableAdder adder, torch.Tensor xs, torch.Tensor ys, ScottPlot.Color? color, long[] xsSize, long[] ysSize)
    {
        double[] xsFloat = new double[xsSize[0]];
        double[] ysFloat = new double[ysSize[0]];

        for (int i = 0; i < xsSize[0]; i++)
        {
            xsFloat[i] = xs[i].ToDouble();
            ysFloat[i] = ys[i].ToDouble();
        }

        return adder.Scatter(xsFloat, ysFloat, color);
    } 

    private static Scatter ScatterComplex32(PlottableAdder adder, torch.Tensor xs, torch.Tensor ys, ScottPlot.Color? color, long[] xsSize, long[] ysSize)
    {
        Complex32[] xsFloat = new Complex32[xsSize[0]];
        Complex32[] ysFloat = new Complex32[ysSize[0]];

        for (int i = 0; i < xsSize[0]; i++)
        {
            var cx = xs[i].ToComplex32();
            var cy = ys[i].ToComplex32();
            xsFloat[i] = new Complex32(cx.Real,cx.Imaginary);
            ysFloat[i] = new Complex32(cy.Real, cy.Imaginary);
        }

        return adder.Scatter(xsFloat, ysFloat, color);
    } 

    private static Scatter ScatterComplex64(PlottableAdder adder, torch.Tensor xs, torch.Tensor ys, ScottPlot.Color? color, long[] xsSize, long[] ysSize)
    {
        Complex[] xsFloat = new Complex[xsSize[0]];
        Complex[] ysFloat = new Complex[ysSize[0]];

        for (int i = 0; i < xsSize[0]; i++)
        {
            xsFloat[i] = xs[i].ToComplex64();
            ysFloat[i] = ys[i].ToComplex64();
        }

        return adder.Scatter(xsFloat, ysFloat, color);
    }

    private static Scatter ScatterBool(PlottableAdder adder, torch.Tensor xs, torch.Tensor ys, ScottPlot.Color? color, long[] xsSize, long[] ysSize)
    {
        bool[] xsFloat = new bool[xsSize[0]];
        bool[] ysFloat = new bool[ysSize[0]];

        for (int i = 0; i < xsSize[0]; i++)
        {
            xsFloat[i] = xs[i].ToBoolean();
            ysFloat[i] = ys[i].ToBoolean();
        }

        return adder.Scatter(xsFloat, ysFloat, color);
    }

    private static Scatter ScatterBFloat16(PlottableAdder adder, torch.Tensor xs, torch.Tensor ys, ScottPlot.Color? color, long[] xsSize, long[] ysSize)
    {
        Scalar[] xsFloat = new Scalar[xsSize[0]];
        Scalar[] ysFloat = new Scalar[ysSize[0]];

        for (int i = 0; i < xsSize[0]; i++)
        {
            xsFloat[i] = xs[i].ToScalar();
            ysFloat[i] = ys[i].ToScalar();
        }

        return adder.Scatter(xsFloat, ysFloat, color);
    }
}
