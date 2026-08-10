using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Inference;

/// <summary>Which device a session runs on. A string-backed enum, so it matches the wire values.</summary>
public enum Device
{
    Cpu,
    Gpu,
}

public static class DeviceNames
{
    public static string Wire(this Device device) => device == Device.Gpu ? "gpu" : "cpu";

    public static Device Parse(string? value) => value?.ToLowerInvariant() switch
    {
        "gpu" => Device.Gpu,
        "cpu" or null or "" => Device.Cpu,
        _ => throw new ArgumentException($"inference: unknown device \"{value}\""),
    };
}

/// <summary>
/// One ONNX Runtime session.
///
/// <para>
/// **The lock is per session and covers only <c>Run</c>.** It is not defensive tidiness: concurrent
/// <c>Run</c> calls on a single CUDA session were measured to destroy the process — Python saw
/// <c>cudaErrorIllegalAddress</c> after roughly 200 calls, and Go degraded from 6.6 s to over 600 s
/// for the same work, which for a service is indistinguishable from a hang. Holding it only around
/// <c>Run</c> is what preserves the speedup of the parallel groups, because those use five DIFFERENT
/// sessions and therefore five different locks.
/// </para>
///
/// <para>
/// CPU sessions deliberately do NOT lock. ONNX Runtime's CPU provider is re-entrant, and locking
/// there would serialise the quality group for no reason.
/// </para>
/// </summary>
public sealed class Session : IDisposable
{
    private readonly InferenceSession _session;
    private readonly object? _gate;
    private readonly Dictionary<string, NodeMetadata> _inputs;

    /// <summary>Output names in the order the model DECLARES them.</summary>
    public string[] OutputNames { get; }

    public Device Device { get; }

    public Session(string modelPath, Device device, int intraOpThreads)
    {
        Device = device;

        var options = new SessionOptions();
        try
        {
            if (device == Device.Gpu)
            {
                // Throws if the provider is unavailable, which the caller's [gpu, cpu] attempt loop
                // handles. What it CANNOT catch is a container started without --gpus, where the
                // provider segfaults instead — hence the device-node probe before we ever get here.
                options.AppendExecutionProvider_CUDA(0);
            }

            if (intraOpThreads > 0)
            {
                // Pinned for conformance runs. ONNX Runtime's CPU reductions split across threads,
                // so a different thread count legitimately shifts results by ~1e-6 — inside the
                // float tolerance, but enough to flip an argmax on near-equal values, which is an
                // exact-match failure with no float anywhere near it.
                options.IntraOpNumThreads = intraOpThreads;
            }

            _session = new InferenceSession(modelPath, options);
        }
        catch
        {
            options.Dispose();
            throw;
        }

        _inputs = _session.InputMetadata.ToDictionary(kv => kv.Key, kv => kv.Value);
        OutputNames = _session.OutputMetadata.Keys.ToArray();
        _gate = device == Device.Gpu ? new object() : null;
    }

    /// <summary>
    /// Runs the model. Inputs are positional; the caller supplies them in declaration order.
    ///
    /// <para>
    /// Each input is CAST to the dtype the model declares. The detectors want float32 and the OCR
    /// nets want uint8, and feeding the wrong one is not a graceful failure — ONNX Runtime either
    /// refuses or reinterprets the bytes.
    /// </para>
    /// </summary>
    public NdArray[] Run(IReadOnlyList<NdArray> inputs)
    {
        string[] names = _inputs.Keys.ToArray();
        if (inputs.Count != names.Length)
        {
            throw new ArgumentException(
                $"inference: model wants {names.Length} inputs, got {inputs.Count}");
        }

        var feeds = new List<NamedOnnxValue>(inputs.Count);
        var owned = new List<IDisposable>();
        try
        {
            for (int i = 0; i < inputs.Count; i++)
            {
                feeds.Add(ToNamedValue(names[i], inputs[i], _inputs[names[i]]));
            }

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
            if (_gate is not null)
            {
                // Acquire, run, release — and read the results AFTER releasing, outside the lock.
                lock (_gate)
                {
                    results = _session.Run(feeds);
                }
            }
            else
            {
                results = _session.Run(feeds);
            }

            using (results)
            {
                // Collected BY DECLARED NAME, not by iteration order. DocTypeAngles has two outputs
                // whose meanings are not interchangeable — embeddings and an angle — and an
                // unexpected reordering must be visible rather than quietly swapping two heads.
                var byName = results.ToDictionary(v => v.Name, v => v);
                var outputs = new NdArray[OutputNames.Length];
                for (int i = 0; i < OutputNames.Length; i++)
                {
                    outputs[i] = FromOnnxValue(byName[OutputNames[i]]);
                }
                return outputs;
            }
        }
        finally
        {
            foreach (IDisposable d in owned)
            {
                d.Dispose();
            }
        }
    }

    private static NamedOnnxValue ToNamedValue(string name, NdArray array, NodeMetadata meta)
    {
        int[] shape = array.Shape;
        if (meta.ElementType == typeof(float))
        {
            float[] values = array.Dtype switch
            {
                Dtype.Float32 => array.AsFloat32().ToArray(),
                // uint8 to float32 without scaling: normalisation is baked into the graphs, and
                // dividing here would silently double-normalise.
                Dtype.UInt8 => ToFloats(array.AsUInt8()),
                _ => throw new InvalidOperationException(
                    $"inference: {name} wants float32, cannot cast from {array.Dtype}"),
            };
            return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(values, shape));
        }
        if (meta.ElementType == typeof(byte))
        {
            byte[] values = array.Dtype == Dtype.UInt8
                ? array.AsUInt8().ToArray()
                : throw new InvalidOperationException(
                    $"inference: {name} wants uint8, cannot cast from {array.Dtype}");
            return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<byte>(values, shape));
        }
        throw new InvalidOperationException(
            $"inference: {name} has unsupported element type {meta.ElementType}");
    }

    private static float[] ToFloats(ReadOnlySpan<byte> bytes)
    {
        var values = new float[bytes.Length];
        for (int i = 0; i < bytes.Length; i++)
        {
            values[i] = bytes[i];
        }
        return values;
    }

    private static NdArray FromOnnxValue(DisposableNamedOnnxValue value)
    {
        if (value.ValueType != OnnxValueType.ONNX_TYPE_TENSOR)
        {
            throw new InvalidOperationException($"inference: {value.Name} is not a tensor");
        }

        if (value.ElementType == TensorElementType.Float)
        {
            var tensor = value.AsTensor<float>();
            return NdArray.FromFloat32(tensor.ToArray(), tensor.Dimensions.ToArray());
        }
        if (value.ElementType == TensorElementType.UInt8)
        {
            var tensor = value.AsTensor<byte>();
            return NdArray.FromUInt8(tensor.ToArray(), tensor.Dimensions.ToArray());
        }
        throw new InvalidOperationException(
            $"inference: {value.Name} has unsupported output type {value.ElementType}");
    }

    public void Dispose() => _session.Dispose();
}
