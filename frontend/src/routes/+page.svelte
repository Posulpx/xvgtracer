<script>
  let file = $state(null);
  let numColors = $state(5);
  let simplify = $state(1.0);
  let angleThreshold = $state(30);
  let smoothSigma = $state(1.0);
  let svg = $state('');
  let layers = $state([]);
  let hidden = $state(new Set());
  let loading = $state(false);
  let status = $state('');

  // learning-system state
  let learned = $state(null);
  let learnedSvg = $state('');
  let learning = $state(false);

  function onFile(e) {
    file = e.target.files[0];
  }

  async function trace() {
    if (!file) return;
    loading = true;
    status = 'Tracing…';
    const fd = new FormData();
    fd.append('image', file);
    fd.append('num_colors', numColors);
    fd.append('simplify', simplify);
    fd.append('angle_threshold', angleThreshold);
    fd.append('smooth_sigma', smoothSigma);
    try {
      const res = await fetch('/api/trace', { method: 'POST', body: fd });
      svg = await res.text();
      // discover layers from <g data-color>
      const re = /data-color="(#[0-9a-fA-F]{6}|rgb\([^)]*\))"/g;
      const found = [];
      let m;
      while ((m = re.exec(svg))) found.push(m[1]);
      layers = found;
      hidden = new Set();
      status = `✅ ${found.length} layers`;
    } catch (err) {
      status = '❌ ' + err.message;
    } finally {
      loading = false;
    }
  }

  async function learn() {
    if (!file) return;
    learning = true;
    status = 'Learning shapes…';
    const fd = new FormData();
    fd.append('image', file);
    fd.append('num_colors', numColors);
    try {
      const res = await fetch('/api/learn', { method: 'POST', body: fd });
      const data = await res.json();
      learned = data.model;
      learnedSvg = data.svg;
      svg = data.svg;
      layers = learned.shapes.map(s => `rgb(${s.color.join(',')})`);
      hidden = new Set();
      status = `✅ Learned ${learned.shapes.length} primitives, ` +
               `${learned.composites.length} composites`;
    } catch (err) {
      status = '❌ ' + err.message;
    } finally {
      learning = false;
    }
  }

  function toggle(i) {
    const s = new Set(hidden);
    s.has(i) ? s.delete(i) : s.add(i);
    hidden = s;
  }

  // layer visibility applied by mutating injected svg via regex on class
  let visibleSvg = $derived((() => {
    if (!svg) return '';
    // hide groups whose index is in `hidden`
    let out = svg;
    let idx = 0;
    out = out.replace(/<g id="layer_\d+"[^>]*>/g, (tag) => {
      const i = idx++;
      return hidden.has(i) ? tag.replace('class="xvg-layer"', 'class="xvg-layer" style="display:none"') : tag;
    });
    return out;
  })());

  function download() {
    const blob = new Blob([visibleSvg], { type: 'image/svg+xml' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'xvgtracer_output.svg';
    a.click();
  }

  async function svgo() {
    // Lightweight client-side optimization: strip empty groups & round numbers.
    let out = svg
      .replace(/<g[^>]*>\s*<\/g>/g, '')
      .replace(/(-?\d+\.\d{2})\d+/g, '$1');
    svg = out;
    status = '✅ Optimized (basic SVGO-style pass)';
  }
</script>

<main>
  <h1>XVGTracer</h1>
  <p class="sub">Color-quantize + per-layer contour tracing → editable layered SVG</p>

  <div class="controls">
    <input type="file" accept="image/*" onchange={onFile} />
    <label>Colors <input type="number" min="2" max="16" bind:value={numColors} /></label>
    <label>Simplify <input type="number" min="0.5" max="5" step="0.1" bind:value={simplify} /></label>
    <label>Corner° <input type="number" min="5" max="120" step="5" bind:value={angleThreshold} /></label>
    <label>Smooth <input type="number" min="0" max="4" step="0.5" bind:value={smoothSigma} /></label>
    <button onclick={trace} disabled={!file || loading}>{loading ? '…' : 'Trace'}</button>
    <button onclick={learn} disabled={!file || learning}>{learning ? '…' : 'Learn'}</button>
    <button onclick={svgo} disabled={!svg}>SVGO</button>
    <button onclick={download} disabled={!svg}>Download</button>
  </div>

  <div class="status">{status}</div>

  <div class="layout">
    <div class="preview">
      {#if visibleSvg}
        {@html visibleSvg}
      {:else}
        <div class="placeholder">Upload an image and hit Trace</div>
      {/if}
    </div>

    {#if layers.length}
      <aside class="layers">
        <h3>Layers</h3>
        {#each layers as c, i}
          <label class="layer">
            <input type="checkbox" checked={!hidden.has(i)} onchange={() => toggle(i)} />
            <span class="swatch" style="background:{c}"></span>
            {c}
          </label>
        {/each}
        {#if learned}
          <h3>Learned shapes</h3>
          {#each learned.shapes as s, i}
            <div class="layer learned">
              <span class="tag {s.type}">{s.type}</span>
              {#if s.transform}<span class="tag transform">↻{s.transform.rotate}°</span>{/if}
              <span class="swatch" style="background:rgb({s.color.join(',')})"></span>
            </div>
          {/each}
          {#each learned.composites as c, i}
            <div class="layer learned">
              <span class="tag composite">{c.type}</span>
              <span class="muted">({c.children.map(ch => ch.type).join(' ⊕ ')})</span>
            </div>
          {/each}
        {/if}
      </aside>
    {/if}
  </div>
</main>

<style>
  :global(body) { font-family: system-ui, sans-serif; margin: 0; background: #0f1115; color: #e6e6e6; }
  main { max-width: 1100px; margin: 0 auto; padding: 2rem; }
  h1 { margin: 0; }
  .sub { color: #9aa; margin-top: 0.2rem; }
  .controls { display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; margin: 1.5rem 0; }
  .controls input[type="number"] { width: 60px; }
  button { background: #3b82f6; color: white; border: 0; padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  .status { color: #6ee7b7; min-height: 1.2rem; }
  .layout { display: grid; grid-template-columns: 1fr 200px; gap: 1rem; }
  .preview { border: 1px solid #222; border-radius: 8px; min-height: 400px; display: flex; align-items: center; justify-content: center; background: #fff; }
  .preview :global(svg) { max-width: 100%; max-height: 600px; }
  .placeholder { color: #999; }
  .layers { border: 1px solid #222; border-radius: 8px; padding: 0.75rem; }
  .layers h3 { margin-top: 0; }
  .layer { display: flex; align-items: center; gap: 0.5rem; margin: 0.4rem 0; font-size: 0.85rem; }
  .swatch { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #0006; }
  .layer.learned { display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap; }
  .tag { font-size: 0.72rem; padding: 0.1rem 0.4rem; border-radius: 4px; background: #1e293b; color: #cbd5e1; }
  .tag.composite { background: #7c3aed22; color: #c4b5fd; }
  .tag.transform { background: #f59e0b22; color: #fcd34d; }
  .muted { color: #94a3b8; font-size: 0.72rem; }
</style>
