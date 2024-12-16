using System.Globalization;
using TorchSharp;

namespace Maomi.Torch;

/// <summary>
/// extensions.
/// </summary>
public static partial class MM
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