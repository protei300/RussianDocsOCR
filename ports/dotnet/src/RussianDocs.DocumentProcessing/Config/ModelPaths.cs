using System.Text;

namespace RussianDocs.DocumentProcessing.Config;

/// <summary>
/// Locates the model artifacts and reads <c>models_path.yaml</c>.
///
/// <para>Port of <c>document_processing/config/__init__.py</c> and its ROOT resolution.</para>
/// </summary>
public static class ModelPaths
{
    /// <summary>
    /// A UTF-8 BOM, as a string.
    ///
    /// <para>
    /// Needed because <c>model.json</c> files must be read BOM-free (D-10): PowerShell's
    /// <c>Set-Content -Encoding utf8</c> adds one, and a JSON parser then fails on the very first
    /// character with a message about invalid syntax rather than about encoding.
    /// </para>
    /// </summary>
    private const string Utf8Bom = "﻿";

    /// <summary>
    /// The directory that CONTAINS <c>document_processing/models</c>.
    ///
    /// <para>
    /// <c>RDOCS_MODELS_ROOT</c> wins if set; otherwise the repository root is located by walking up
    /// from the executable and then from the working directory. Mirrors the reference's ROOT
    /// resolution and exists for the same reason: the CLI is invoked from several places and a
    /// cwd-relative path silently picks up the wrong models.
    /// </para>
    ///
    /// <para>
    /// Note the semantics of the variable, which the Go port's Docker image got wrong: it names a
    /// ROOT, not the models directory. Pointed at the models directory, the service starts, serves
    /// its frontend and fails every document.
    /// </para>
    /// </summary>
    public static string Root()
    {
        string? configured = Environment.GetEnvironmentVariable("RDOCS_MODELS_ROOT");
        if (!string.IsNullOrEmpty(configured))
        {
            return IsLibraryRoot(configured)
                ? configured
                : throw new DirectoryNotFoundException(
                    $"config: RDOCS_MODELS_ROOT=\"{configured}\" has no document_processing/models");
        }

        foreach (string start in new[] { AppContext.BaseDirectory, Directory.GetCurrentDirectory() })
        {
            var dir = new DirectoryInfo(start);
            while (dir is not null)
            {
                if (IsLibraryRoot(dir.FullName))
                {
                    return dir.FullName;
                }
                dir = dir.Parent;
            }
        }

        throw new DirectoryNotFoundException(
            "config: could not locate document_processing/models; set RDOCS_MODELS_ROOT to the " +
            "repository root");
    }

    private static bool IsLibraryRoot(string dir) =>
        Directory.Exists(Path.Combine(dir, "document_processing", "models"));

    /// <summary>
    /// Reads <c>models_path.yaml</c> into module name to relative path.
    ///
    /// <para>
    /// A hand-written parser rather than a YAML dependency, matching the Go port. The file is
    /// fourteen lines of <c>Key: value</c> with no nesting, lists, anchors or quoting, so a parser
    /// costs one dependency and buys nothing — and a dependency that can interpret this file
    /// differently in one language than another is precisely what a port cannot afford.
    /// </para>
    /// </summary>
    public static IReadOnlyDictionary<string, string> Load(string root)
    {
        string path = Path.Combine(root, "document_processing", "config", "models_path.yaml");
        var result = new Dictionary<string, string>(StringComparer.Ordinal);

        string[] lines = File.ReadAllLines(path, Encoding.UTF8);
        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i];
            if (i == 0 && line.StartsWith(Utf8Bom, StringComparison.Ordinal))
            {
                line = line[Utf8Bom.Length..];
            }
            line = line.Trim();
            if (line.Length == 0 || line.StartsWith('#'))
            {
                continue;
            }

            int colon = line.IndexOf(':');
            if (colon <= 0)
            {
                throw new InvalidDataException($"config: {path}:{i + 1}: cannot parse \"{line}\"");
            }

            string key = line[..colon].Trim();
            string value = line[(colon + 1)..].Trim();
            result[key] = NormaliseSeparators(value);
        }
        return result;
    }

    /// <summary>
    /// Resolves a module's model directory to an absolute path.
    /// </summary>
    public static string Resolve(string root, IReadOnlyDictionary<string, string> paths, string module) =>
        paths.TryGetValue(module, out string? relative)
            ? Path.Combine(root, "document_processing", relative)
            : throw new KeyNotFoundException($"config: models_path.yaml has no entry for \"{module}\"");

    /// <summary>
    /// Turns Windows separators into the platform's own.
    ///
    /// <para>
    /// **Every port must do this.** The committed YAML contains <c>models\Borders</c>, and
    /// <c>model.json</c> refers to <c>resources\centers.npz</c>; on Linux those are single filenames
    /// containing a backslash, so <c>DocTypeAngles</c> fails to construct — but only in a container,
    /// never on a Windows developer's machine, which is the worst possible distribution of a bug.
    /// </para>
    /// </summary>
    public static string NormaliseSeparators(string value) =>
        value.Replace('\\', Path.DirectorySeparatorChar)
             .Replace('/', Path.DirectorySeparatorChar);
}
