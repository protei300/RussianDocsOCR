using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Postprocess;

public sealed record SegmentResult(List<Point[]> Segments) : IResult;

/// <summary>
/// Turns proto masks plus per-box coefficients into contours.
///
/// <para>
/// Not reachable through <see cref="IPostprocessor.Apply"/>: it needs the detector's boxes as well as
/// the proto tensor, so the segmentation model calls <see cref="Segment"/> directly. Apply throws
/// rather than returning something plausible.
/// </para>
/// </summary>
public sealed class YoloSegmentor(double maskFilter) : IPostprocessor
{
    public IResult Apply(NdArray output, Context context) =>
        throw new InvalidOperationException(
            "postprocess: YOLOSegmentor needs Segment(), not Apply()");

    /// <summary>
    /// One contour per box.
    /// </summary>
    public List<Point[]> Segment(NdArray proto, List<Box> boxes, int[] extraPad, int origH, int origW)
    {
        if (boxes.Count == 0)
        {
            return [];
        }

        int[] shape = proto.Shape;
        if (shape.Length == 4 && shape[0] == 1)
        {
            shape = shape[1..];
        }
        if (shape.Length != 3)
        {
            throw new InvalidDataException(
                $"postprocess: proto masks expect [h,w,chn], got {NdArray.Describe(proto.Shape)}");
        }

        int imh = shape[0], imw = shape[1], chn = shape[2];
        ReadOnlySpan<float> protoData = proto.AsFloat32();

        var segments = new List<Point[]>(boxes.Count);
        foreach (Box box in boxes)
        {
            if (box.Seg.Length != chn)
            {
                throw new InvalidDataException(
                    $"postprocess: {box.Seg.Length} mask coefficients for {chn} proto channels");
            }

            // masks @ proto.transpose(-1,0,1).reshape(chn,-1), then sigmoid — written as a direct
            // loop over pixels. The transpose exists in the reference only to make numpy's matmul
            // line up, and materialising it here would be pure copying.
            //
            // **Accumulated in float32**, matching the reference's dtype. Widening to double "for
            // accuracy" moves the mask boundary and therefore the contour.
            var mask = new float[imh * imw];
            for (int y = 0; y < imh; y++)
            {
                for (int x = 0; x < imw; x++)
                {
                    int at = (y * imw + x) * chn;
                    float acc = 0;
                    for (int c = 0; c < chn; c++)
                    {
                        acc += (float)box.Seg[c] * protoData[at + c];
                    }
                    mask[y * imw + x] = Sigmoid(acc);
                }
            }

            // Undo the letterbox INSIDE the proto resolution, before upscaling. gain is old/new, and
            // the padding is halved because it was split across both sides.
            double gain = Math.Min((double)imh / origH, (double)imw / origW);
            double padX = (imw - origW * gain) / 2;
            double padY = (imh - origH * gain) / 2;
            int top = (int)padY, left = (int)padX;

            // **TRUNCATE THE DIFFERENCE, do not subtract the truncated padding.** The reference
            // writes `int(imh - pad[1])`, and `imh - int(pad[1])` is a different number whenever the
            // padding is fractional: for imh=160 and pad=20.5 the two give 139 and 140. One row at
            // proto resolution becomes about nine rows after the upscale — which is exactly how the
            // Go port caught it, with borders.canvas reporting 868 rows against the golden's 877 and
            // the width matching to the pixel.
            int bottom = (int)(imh - padY), right = (int)(imw - padX);
            if (top < 0 || left < 0 || bottom > imh || right > imw || bottom <= top || right <= left)
            {
                throw new InvalidDataException(
                    $"postprocess: degenerate mask crop [{top}:{bottom}, {left}:{right}] in {imh}x{imw}");
            }

            using FloatMask full = BuildFull(mask, imh, imw, top, bottom, left, right,
                origH, origW, extraPad);

            // Zero everything outside the instance's own box, so two adjacent documents cannot bleed
            // into each other's contour. The box is expressed in original pixels, so the extra
            // padding comes off first.
            full.ZeroOutsideBox(box.X1 - extraPad[0], box.Y1 - extraPad[1],
                box.X2 - extraPad[0], box.Y2 - extraPad[1]);

            using Image binary = full.Threshold(maskFilter);
            List<Point[]> contours = Contours.FindExternalContours(binary);
            if (contours.Count == 0)
            {
                continue;
            }

            // The LARGEST contour by area. A mask can produce specks; the document is the big one.
            Point[] largest = contours.OrderByDescending(Contours.ContourArea).First();
            segments.Add(largest);
        }
        return segments;
    }

    private static FloatMask BuildFull(float[] mask, int imh, int imw,
        int top, int bottom, int left, int right, int origH, int origW, int[] extraPad)
    {
        using FloatMask atProto = FloatMask.FromValues(mask, imh, imw);
        using FloatMask cropped = atProto.Crop(top, bottom, left, right);

        // Upscale to the PRE-letterbox size, then strip the extra padding — the reference's order,
        // and it matters because the extra padding is expressed in original pixels.
        using FloatMask upscaled = cropped.Resize(origW, origH);
        return upscaled.Crop(extraPad[1], origH - extraPad[1], extraPad[0], origW - extraPad[0]);
    }

    /// <summary>
    /// float32 sigmoid.
    ///
    /// <para>
    /// <c>MathF.Exp</c>, not <c>Math.Exp</c>: the double version would promote, accumulate in double
    /// precision and return a value the reference never computed. CONVENTIONS §6.17, and the JVM has
    /// the same hazard with <c>Math.exp</c>.
    /// </para>
    /// </summary>
    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}
