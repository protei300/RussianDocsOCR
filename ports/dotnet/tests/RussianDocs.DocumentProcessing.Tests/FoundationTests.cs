using RussianDocs.DocumentProcessing.Config;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Tests;

/// <summary>
/// M1's unit tests. Each one pins a trap that CONVENTIONS §6 records, so a regression fails here
/// rather than as an unexplained stage divergence three milestones later.
/// </summary>
[TestFixture]
public class PyNumTests
{
    /// <summary>
    /// The case that made <see cref="PyNum.FloorDiv"/> necessary.
    ///
    /// <para>
    /// A 2999x1777 image at img_size 1500: CPython's <c>//</c> gives 1499, plain
    /// <c>Math.Floor(2999 / ratio)</c> gives 1500. One pixel of canvas width, and every box
    /// downstream shifts. The assertion below is written against BOTH values on purpose — it
    /// documents the wrong answer as well as the right one.
    /// </para>
    /// </summary>
    [Test]
    public void FloorDiv_MatchesCPython_OnTheCaseThatDiffers()
    {
        const int w = 2999;
        double ratio = Math.Max(2999.0 / 1500, 1.0);

        Assert.That(PyNum.FloorDivInt(w, ratio), Is.EqualTo(1499),
            "CPython's float // gives 1499 here");
        Assert.That((int)Math.Floor(w / ratio), Is.EqualTo(1500),
            "and naive floor division gives 1500 — which is why FloorDiv exists");
    }

    [Test]
    public void FloorDiv_HandlesTheOrdinaryCases()
    {
        Assert.Multiple(() =>
        {
            Assert.That(PyNum.FloorDiv(7, 2), Is.EqualTo(3));
            Assert.That(PyNum.FloorDiv(-7, 2), Is.EqualTo(-4), "rounds toward negative infinity");
            Assert.That(PyNum.FloorDiv(7, -2), Is.EqualTo(-4));
            Assert.That(PyNum.FloorDiv(6, 3), Is.EqualTo(2));
            Assert.That(double.IsNaN(PyNum.FloorDiv(1, 0)), Is.True);
        });
    }

    /// <summary>
    /// .NET rounds half to even by default, matching <c>np.round</c>. Asserted rather than assumed,
    /// because Go does NOT and the wrapper exists to make the intent explicit.
    /// </summary>
    [Test]
    public void RoundHalfEven_MatchesNumpy()
    {
        Assert.Multiple(() =>
        {
            Assert.That(PyNum.RoundHalfEven(0.5), Is.EqualTo(0.0), "np.round(0.5) == 0");
            Assert.That(PyNum.RoundHalfEven(1.5), Is.EqualTo(2.0));
            Assert.That(PyNum.RoundHalfEven(2.5), Is.EqualTo(2.0));
            Assert.That(PyNum.RoundHalfEven(-0.5), Is.EqualTo(0.0));
        });
    }
}

[TestFixture]
public class CropTests
{
    /// <summary>
    /// The reason <see cref="Crop.ClampedCrop"/> is the only sanctioned crop path: a raw
    /// <c>new Mat(mat, rect)</c> throws where Python's slice quietly clamps, and detectors do return
    /// boxes a pixel or two outside the image.
    /// </summary>
    [Test]
    public void ClampedCrop_ClampsInsteadOfThrowing()
    {
        using Image src = Io.NewFilled(10, 20, 1, 2, 3);

        using Image beyond = Crop.ClampedCrop(src, 15, 5, 40, 30);
        Assert.Multiple(() =>
        {
            Assert.That(beyond.Width, Is.EqualTo(5), "x2 clamps to the width");
            Assert.That(beyond.Height, Is.EqualTo(5), "y2 clamps to the height");
        });

        using Image negative = Crop.ClampedCrop(src, -5, -5, 4, 4);
        Assert.Multiple(() =>
        {
            Assert.That(negative.Width, Is.EqualTo(4), "a negative start clamps to 0, not the end");
            Assert.That(negative.Height, Is.EqualTo(4));
        });
    }

    /// <summary>An empty or reversed range is a zero-sized image, exactly as the slice yields.</summary>
    [Test]
    public void ClampedCrop_EmptyRangeIsZeroSizedNotAnError()
    {
        using Image src = Io.NewFilled(10, 20, 0, 0, 0);

        using Image reversed = Crop.ClampedCrop(src, 8, 8, 3, 3);
        using Image outside = Crop.ClampedCrop(src, 100, 100, 120, 120);
        Assert.Multiple(() =>
        {
            Assert.That(reversed.Width, Is.EqualTo(0));
            Assert.That(reversed.Height, Is.EqualTo(0));
            Assert.That(outside.Width, Is.EqualTo(0));
        });
    }

    /// <summary>
    /// The crop must not alias its parent: pipeline intermediates are released as soon as the stage
    /// ends, and a submat sharing that buffer would dangle.
    /// </summary>
    [Test]
    public void ClampedCrop_SurvivesTheSourceBeingDisposed()
    {
        Image crop;
        using (Image src = Io.NewFilled(8, 8, 9, 9, 9))
        {
            crop = Crop.ClampedCrop(src, 1, 1, 5, 5);
        }
        using (crop)
        {
            Assert.That(crop.ToArray().AsUInt8()[0], Is.EqualTo(9),
                "reading after the parent is gone must still work");
        }
    }
}

[TestFixture]
public class NpyTests
{
    [Test]
    public void RoundTrip_PreservesShapeDtypeAndBytes()
    {
        var original = NdArray.FromFloat32([1.5f, -2.25f, 0f, 3.75f, 5f, 6f], 3, 2);
        string path = Path.Combine(TestContext.CurrentContext.WorkDirectory, "roundtrip.npy");

        Npy.Save(path, original);
        NdArray reloaded = Npy.Load(path);

        Assert.Multiple(() =>
        {
            Assert.That(reloaded.Shape, Is.EqualTo(new[] { 3, 2 }));
            Assert.That(reloaded.Dtype, Is.EqualTo(Dtype.Float32));
            Assert.That(reloaded.AsFloat32().ToArray(), Is.EqualTo(original.AsFloat32().ToArray()));
        });
    }

    /// <summary>
    /// A one-element shape must be written as <c>(3,)</c>. Without the comma it is the integer 3 in
    /// Python, and NumPy's own reader rejects it.
    /// </summary>
    [Test]
    public void OneDimensionalShape_KeepsTheTrailingComma()
    {
        var array = NdArray.FromUInt8([1, 2, 3], 3);
        string path = Path.Combine(TestContext.CurrentContext.WorkDirectory, "onedim.npy");
        Npy.Save(path, array);

        string header = System.Text.Encoding.ASCII.GetString(File.ReadAllBytes(path)[..128]);
        Assert.That(header, Does.Contain("'shape': (3,)"));
    }

    /// <summary>
    /// The reference's real <c>centers.npz</c> is read here rather than a synthetic fixture, because
    /// the <c>&lt;U64</c> label decode is the part a port gets wrong: NumPy pads fixed-width UTF-32
    /// with NULs, and naive byte slicing yields empty strings that then match nothing.
    /// </summary>
    [Test]
    public void Unicode_LabelsDecodeToRealDocumentTypes()
    {
        string root = ModelPaths.Root();
        string npz = Path.Combine(root, "document_processing", "models", "DocTypeAngles", "ONNX",
            "resources", "centers.npz");
        if (!File.Exists(npz))
        {
            Assert.Ignore($"centers.npz not present at {npz}");
        }

        using var zip = System.IO.Compression.ZipFile.OpenRead(npz);
        var entry = zip.GetEntry("labels.npy") ?? zip.Entries.First(e => e.Name.StartsWith("labels"));
        using var stream = entry.Open();
        using var buffer = new MemoryStream();
        stream.CopyTo(buffer);

        NdArray labels = Npy.Parse(buffer.ToArray(), entry.Name);
        string[] decoded = labels.AsUnicode();

        Assert.Multiple(() =>
        {
            Assert.That(decoded, Is.Not.Empty);
            Assert.That(decoded, Has.All.Not.Empty, "NUL-padded UTF-32 must not decode to blanks");
            Assert.That(decoded, Has.Some.Contains("INTPASSPORT"));
        });
    }
}

[TestFixture]
public class ConfigTests
{
    [Test]
    public void ModelPaths_ResolveEveryModuleToAnExistingDirectory()
    {
        string root = ModelPaths.Root();
        var paths = ModelPaths.Load(root);

        Assert.That(paths, Is.Not.Empty);
        foreach ((string module, string _) in paths)
        {
            string resolved = ModelPaths.Resolve(root, paths, module);
            Assert.That(Directory.Exists(resolved), Is.True, $"{module} -> {resolved}");
        }
    }

    /// <summary>
    /// Backslashes in the committed YAML must become the platform separator. Without this the
    /// library fails to construct on Linux only — never on a Windows developer's machine, which is
    /// the worst possible distribution of a bug.
    /// </summary>
    [Test]
    public void NormaliseSeparators_ConvertsWindowsPaths()
    {
        string normalised = ModelPaths.NormaliseSeparators(@"models\OCR\latin_accurate");
        Assert.That(normalised, Does.Not.Contain('\\').Or.EqualTo(normalised).And.Not.Contain('/')
            .Or.EqualTo(Path.Combine("models", "OCR", "latin_accurate")));
        Assert.That(normalised, Is.EqualTo(Path.Combine("models", "OCR", "latin_accurate")));
    }
}
