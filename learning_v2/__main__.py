import sys

from . import render_svg, run_image, _seg_summary


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m learning_v2 <image.png> [num_colors]")
        sys.exit(1)

    path = sys.argv[1]
    nc = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    if "--svg" in sys.argv:
        svg = render_svg(path, nc)
        out = path.rsplit(".", 1)[0] + ".svg"
        with open(out, "w", encoding="utf-8") as f:
            f.write(svg)
        print(f"Wrote {out} ({len(svg)} chars)")
    else:
        results = run_image(path, nc)
        print(f"\n{path}  ({len(results)} shapes)\n")
        for r in results:
            print("  %-14s %-8s gvar=%.4f peaks=%d conv=%d segs=%s"
                  % (str(r["color"]), r["family"], r["global_var"],
                     r["n_peaks"], len(r["lc"]["convergence"]),
                     _seg_summary(r["lc"])))


if __name__ == "__main__":
    main()
