using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json.Serialization;

namespace RussianDocs.Service.Api;

/// <summary>The host block. JSON names match the frontend exactly.</summary>
public sealed record ServerStats
{
    [JsonPropertyName("cpu_pct")] public double CpuPct { get; init; }
    [JsonPropertyName("cpu_name")] public string CpuName { get; init; } = "";
    [JsonPropertyName("cpu_cores")] public int CpuCores { get; init; }
    [JsonPropertyName("cpu_threads")] public int CpuThreads { get; init; }
    [JsonPropertyName("ram_used_gb")] public double RamUsedGb { get; init; }
    [JsonPropertyName("ram_total_gb")] public double RamTotalGb { get; init; }
    [JsonPropertyName("disk_used_gb")] public double DiskUsedGb { get; init; }
    [JsonPropertyName("disk_total_gb")] public double DiskTotalGb { get; init; }
}

/// <summary>The accelerator block, or <c>null</c> when there is no GPU to describe.</summary>
public sealed record GpuStats
{
    [JsonPropertyName("name")] public string Name { get; init; } = "";
    [JsonPropertyName("utilization_pct")] public int UtilizationPct { get; init; }
    [JsonPropertyName("vram_used_gb")] public double VramUsedGb { get; init; }
    [JsonPropertyName("vram_total_gb")] public double VramTotalGb { get; init; }
    [JsonPropertyName("temperature_c")] public int TemperatureC { get; init; }
}

/// <summary>
/// Host CPU, memory, disk and GPU for the status page.
///
/// <para>
/// **The field names here are fixed by the SHARED FRONTEND, not chosen.** <c>web/</c> is reused
/// unchanged by every port, so <c>web/src/views/pages/status/Index.vue</c> is the contract: it reads
/// <c>server.cpu_pct</c>, <c>server.cpu_name</c>, <c>server.ram_used_gb</c> and the rest by name. An
/// earlier version of the Go port returned a thinner, more idiomatic block on the reasoning that
/// pulling in a dependency to render a CPU gauge was a poor trade — and the status page rendered
/// completely empty. The lesson is worth writing down: when a UI is shared, the UI owns the wire
/// format.
/// </para>
///
/// <para>
/// **Every probe is INDIVIDUALLY GUARDED and degrades to a zero value.** A service that cannot
/// describe its own host must still recognise documents, so nothing in here may throw at a caller.
/// </para>
///
/// <para>Port of the <c>_server_stats</c> / <c>_gpu_stats</c> helpers in <c>service/api/status.py</c>.</para>
/// </summary>
public static class SysInfo
{
    private static readonly Process Self = Process.GetCurrentProcess();
    private static TimeSpan _lastCpuTime = TimeSpan.Zero;
    private static long _lastCpuStamp;
    private static readonly object CpuGate = new();

    public static ServerStats ReadServer()
    {
        (double ramUsed, double ramTotal) = Memory();
        (double diskUsed, double diskTotal) = Disk();
        return new ServerStats
        {
            CpuName = CpuName(),
            // .NET exposes only logical processors portably. Reporting the same number for both is
            // honest — the alternative is a per-platform probe (WMI on Windows, /proc/cpuinfo on
            // Linux) for a figure the status page shows and nothing acts on.
            CpuCores = Environment.ProcessorCount,
            CpuThreads = Environment.ProcessorCount,
            CpuPct = CpuPercent(),
            RamUsedGb = ramUsed,
            RamTotalGb = ramTotal,
            DiskUsedGb = diskUsed,
            DiskTotalGb = diskTotal,
        };
    }

    /// <summary>
    /// Process CPU usage since the previous call, as a percentage of one core-second per wall
    /// second, scaled by processor count.
    ///
    /// <para>
    /// The FIRST call returns 0 by construction: there is no earlier sample to difference against, and
    /// inventing one from process start time would report an average over the whole lifetime, which is
    /// not what a live gauge means.
    /// </para>
    /// </summary>
    private static double CpuPercent()
    {
        lock (CpuGate)
        {
            try
            {
                Self.Refresh();
                TimeSpan cpu = Self.TotalProcessorTime;
                long stamp = Stopwatch.GetTimestamp();
                if (_lastCpuStamp == 0)
                {
                    (_lastCpuTime, _lastCpuStamp) = (cpu, stamp);
                    return 0;
                }
                double elapsed = Stopwatch.GetElapsedTime(_lastCpuStamp, stamp).TotalSeconds;
                double used = (cpu - _lastCpuTime).TotalSeconds;
                (_lastCpuTime, _lastCpuStamp) = (cpu, stamp);
                if (elapsed <= 0)
                {
                    return 0;
                }
                return Round1(Math.Clamp(used / elapsed / Environment.ProcessorCount * 100, 0, 100));
            }
            catch
            {
                return 0;
            }
        }
    }

    private static string CpuName()
    {
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) &&
                File.Exists("/proc/cpuinfo"))
            {
                foreach (string line in File.ReadLines("/proc/cpuinfo"))
                {
                    if (line.StartsWith("model name", StringComparison.Ordinal) &&
                        line.Split(':', 2) is [_, string value])
                    {
                        return TrimSpaces(value);
                    }
                }
            }
            // Both the registry and /proc/cpuinfo pad CPU names, sometimes in the middle, so the
            // value goes through TrimSpaces either way.
            return TrimSpaces(
                Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER")
                ?? RuntimeInformation.ProcessArchitecture.ToString());
        }
        catch
        {
            return "";
        }
    }

    private static (double Used, double Total) Memory()
    {
        try
        {
            GCMemoryInfo info = GC.GetGCMemoryInfo();
            double total = Gb(info.TotalAvailableMemoryBytes);
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) &&
                File.Exists("/proc/meminfo"))
            {
                double totalKb = 0, availableKb = 0;
                foreach (string line in File.ReadLines("/proc/meminfo"))
                {
                    if (line.StartsWith("MemTotal:", StringComparison.Ordinal))
                    {
                        totalKb = ParseMeminfoKb(line);
                    }
                    else if (line.StartsWith("MemAvailable:", StringComparison.Ordinal))
                    {
                        availableKb = ParseMeminfoKb(line);
                    }
                }
                if (totalKb > 0)
                {
                    return (Gb((long)((totalKb - availableKb) * 1024)),
                        Gb((long)(totalKb * 1024)));
                }
            }
            // Without a host probe the honest answer is this process's working set against the
            // limit the runtime knows about — a smaller number than the host's, and labelled as
            // used rather than pretended to be host-wide.
            Self.Refresh();
            return (Gb(Self.WorkingSet64), total);
        }
        catch
        {
            return (0, 0);
        }
    }

    private static double ParseMeminfoKb(string line)
    {
        string[] parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        return parts.Length >= 2 &&
               double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture,
                   out double value)
            ? value
            : 0;
    }

    private static (double Used, double Total) Disk()
    {
        try
        {
            var drive = new DriveInfo(Path.GetPathRoot(AppContext.BaseDirectory) ?? "/");
            return (Gb(drive.TotalSize - drive.AvailableFreeSpace), Gb(drive.TotalSize));
        }
        catch
        {
            return (0, 0);
        }
    }

    /// <summary>
    /// Queries the GPU through <c>nvidia-smi</c>, returning <c>null</c> when there is none.
    ///
    /// <para>
    /// **Why a subprocess rather than NVML.** NVML is the proper API and the Python service uses it
    /// through pynvml. Reaching it from .NET means P/Invoke into a library that may not exist, on two
    /// platforms, for information that is purely diagnostic and polled by one page. <c>nvidia-smi</c>
    /// ships with the driver, is present in the CUDA runtime images, and its CSV output is stable
    /// across driver generations.
    /// </para>
    ///
    /// <para>
    /// The cost is real and bounded: one process spawn per status request, with a hard timeout. If that
    /// ever becomes a problem the fix is a cached value with a TTL, not NVML.
    /// </para>
    ///
    /// <para>
    /// **Absence is NOT an error.** No GPU, no driver, or a CPU-only container all mean <c>null</c>,
    /// and the status page then shows the compute block alone — which is the part that answers whether
    /// the GPU is actually being used.
    /// </para>
    /// </summary>
    public static GpuStats? ReadGpu()
    {
        try
        {
            using var process = new Process
            {
                StartInfo = new ProcessStartInfo("nvidia-smi")
                {
                    Arguments =
                        "--query-gpu=name,utilization.gpu,memory.used,memory.total," +
                        "temperature.gpu --format=csv,noheader,nounits",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                },
            };
            if (!process.Start())
            {
                return null;
            }
            string output = process.StandardOutput.ReadToEnd();
            if (!process.WaitForExit(3000) || process.ExitCode != 0)
            {
                return null;
            }

            // The first line only: a multi-GPU host reports one row per device, and the pipeline pins
            // device 0.
            string line = output.Split('\n', 2)[0].Trim();
            if (line.Length == 0)
            {
                return null;
            }
            string[] parts = line.Split(',').Select(p => p.Trim()).ToArray();
            if (parts.Length < 5)
            {
                return null;
            }

            // memory.* is reported in MiB with `nounits`.
            return new GpuStats
            {
                Name = parts[0],
                UtilizationPct = Atoi(parts[1]),
                VramUsedGb = Round1(Atof(parts[2]) * 1024 * 1024 / 1e9),
                VramTotalGb = Round1(Atof(parts[3]) * 1024 * 1024 / 1e9),
                TemperatureC = Atoi(parts[4]),
            };
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Bytes to gigabytes at one decimal.
    ///
    /// <para>
    /// DECIMAL gigabytes (1e9), matching the Python service, so the two report the same number for the
    /// same machine. Not GiB — a status page saying 32.0 GB for a 32 GB stick is what an operator
    /// expects, whatever the pedantically correct unit is.
    /// </para>
    /// </summary>
    private static double Gb(long bytes) => Round1(bytes / 1e9);

    internal static double Round1(double v) => (int)(v * 10 + 0.5) / 10.0;

    private static int Atoi(string s) =>
        int.TryParse(s.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int v)
            ? v
            : 0;

    private static double Atof(string s) =>
        double.TryParse(s.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double v)
            ? v
            : 0;

    private static string TrimSpaces(string s) =>
        string.Join(" ", s.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
}
