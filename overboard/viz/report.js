/* OVERBOARD Studio - the one-page mission report.

   The report is a single, self-contained HTML document: the mission, the
   robot and the result on one page, printable and downloadable. It is
   built from the numbers the Result tab already shows, so it cannot
   disagree with the studio.

   The file is a plain script, so it works in the studio (which opens the
   report in a new tab) and on its own. */

(function (global) {
    'use strict';

    /* ---------------------------------------------------------------- html */

    function esc(value) {
        return String(value === null || value === undefined ? '' : value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function num(value, digits) {
        if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
        return Number(value).toFixed(digits === undefined ? 2 : digits);
    }

    function cm(value) {
        if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
        return `${(Number(value) * 100).toFixed(2)} cm`;
    }

    /* ----------------------------------------------------------------- css */

    const CSS = `
:root {
  --ink: #14181f;
  --muted: #5f6b7c;
  --line: #d9dfe8;
  --hair: #eef1f5;
  --ok: #15703f;
  --warn: #8a6100;
  --bad: #b3261e;
  --accent: #0b6fa4;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #eef1f5;
  color: var(--ink);
  font: 14px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}
.toolbar {
  position: sticky; top: 0; z-index: 2;
  display: flex; gap: 8px; justify-content: flex-end;
  max-width: 880px; margin: 0 auto; padding: 10px 0 0;
}
.toolbar button {
  font: inherit; font-size: 12px; padding: 6px 12px; cursor: pointer;
  color: var(--ink); background: #fff; border: 1px solid var(--line); border-radius: 5px;
}
.toolbar button:hover { border-color: var(--accent); color: var(--accent); }
.page {
  max-width: 880px; margin: 14px auto 40px; background: #fff;
  padding: 36px 40px 32px; border: 1px solid var(--line); border-radius: 6px;
  box-shadow: 0 1px 3px rgba(20, 24, 31, .07);
}
.head { display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;
  border-bottom: 2px solid var(--ink); padding-bottom: 14px; }
.brand { font-size: 11px; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; color: var(--accent); }
h1 { font-size: 25px; line-height: 1.2; margin: 3px 0 0; font-weight: 650; letter-spacing: -.01em; }
.meta { color: var(--muted); font-size: 12px; text-align: right; white-space: nowrap; }
.meta div + div { margin-top: 2px; }
.verdict { display: inline-flex; align-items: center; gap: 7px; margin-top: 12px;
  padding: 6px 13px; border-radius: 5px; font-weight: 600; font-size: 13px; }
.verdict.ok { background: #e4f3ea; color: var(--ok); }
.verdict.warn { background: #fcf2da; color: var(--warn); }
.verdict.bad { background: #fcebea; color: var(--bad); }
.verdict .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
section { margin-top: 17px; }
h2 { font-size: 10.5px; text-transform: uppercase; letter-spacing: .13em; color: var(--muted);
  margin: 0 0 8px; padding-bottom: 5px; border-bottom: 1px solid var(--line); }
h3 { font-size: 12px; margin: 13px 0 5px; font-weight: 600; }
p.note { color: var(--muted); font-size: 12px; margin: 7px 0 0; }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));
  gap: 9px 20px; margin: 0; }
.facts dt { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .07em; }
.facts dd { margin: 2px 0 0; font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 15px; }
.facts dd small { font-size: 11px; color: var(--muted); }
.figures { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; margin-top: 9px; }
figure { margin: 0; }
figcaption { font-size: 10px; text-transform: uppercase; letter-spacing: .07em;
  color: var(--muted); margin-bottom: 5px; }
svg { width: 100%; height: auto; display: block; background: #fbfcfe;
  border: 1px solid var(--line); border-radius: 4px; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { text-align: left; font-size: 10px; text-transform: uppercase; letter-spacing: .07em;
  color: var(--muted); font-weight: 600; border-bottom: 1px solid var(--line); padding: 4px 10px 4px 0; }
td { padding: 5px 10px 5px 0; border-bottom: 1px solid var(--hair); vertical-align: top; }
td.num, th.num { text-align: right; padding-right: 0; }
td.mono, th.mono { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; }
tr.bad td { color: var(--bad); }
tr.bad td.num { font-weight: 600; }
tr.ok td.verdict-cell { color: var(--ok); }
.tag { display: inline-block; font-size: 10px; letter-spacing: .04em; padding: 1px 6px;
  border-radius: 3px; border: 1px solid var(--line); color: var(--muted); margin-left: 4px; }
.tag.stop { border-color: #bcd9f0; color: #0b6fa4; background: #eef6fc; }
.tag.event { border-color: #e6cfe3; color: #8a3a76; background: #fbf1f8; }
.foot { margin-top: 20px; padding-top: 10px; border-top: 1px solid var(--line);
  color: var(--muted); font-size: 11px; display: flex; justify-content: space-between; gap: 16px; }
@page { size: A4 portrait; margin: 14mm 13mm; }
@media print {
  body { background: #fff; }
  .toolbar { display: none; }
  .page { max-width: none; margin: 0; padding: 0; border: none; border-radius: 0; box-shadow: none; }
  figure, table { break-inside: avoid; }
  h2, h3 { break-after: avoid; }
}
`;

    /* --------------------------------------------------------------- figures */

    /* The field, with the waypoints and the planned path. */
    function fieldDiagram(data) {
        const waypoints = (data.mission && data.mission.waypoints) || [];
        const samples = (data.result && data.result.samples) || [];
        const points = [];
        waypoints.forEach((w) => points.push([Number(w.x), Number(w.y)]));
        samples.forEach((s) => points.push([Number(s.x), Number(s.y)]));
        if (points.length < 2) return '';

        let minX = Infinity; let maxX = -Infinity; let minY = Infinity; let maxY = -Infinity;
        points.forEach(([x, y]) => {
            if (!Number.isFinite(x) || !Number.isFinite(y)) return;
            minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            minY = Math.min(minY, y); maxY = Math.max(maxY, y);
        });
        if (!Number.isFinite(minX)) return '';
        const pad = 0.08;
        const w = Math.max(maxX - minX, 0.2) + pad * 2;
        const h = Math.max(maxY - minY, 0.2) + pad * 2;
        // The world's y goes up; the page's goes down.
        const sx = (x) => x - minX + pad;
        const sy = (y) => maxY + pad - y;
        const f = (v) => v.toFixed(4);

        const parts = [];
        parts.push(`<svg viewBox="0 0 ${f(w)} ${f(h)}" role="img" `
            + `aria-label="The planned path on the field">`);
        // A light grid, so the scale is readable.
        const step = w > 2 ? 0.5 : (w > 0.8 ? 0.25 : 0.1);
        for (let x = Math.ceil(minX / step) * step; x <= maxX + pad; x += step) {
            parts.push(`<line x1="${f(sx(x))}" y1="0" x2="${f(sx(x))}" y2="${f(h)}" stroke="#e8edf3" stroke-width="0.004"/>`);
        }
        for (let y = Math.ceil(minY / step) * step; y <= maxY + pad; y += step) {
            parts.push(`<line x1="0" y1="${f(sy(y))}" x2="${f(w)}" y2="${f(sy(y))}" stroke="#e8edf3" stroke-width="0.004"/>`);
        }
        // The straight path between the waypoints.
        if (waypoints.length > 1) {
            const d = waypoints.map((p, i) => `${i ? 'L' : 'M'} ${f(sx(Number(p.x)))} ${f(sy(Number(p.y)))}`).join(' ');
            parts.push(`<path d="${d}" fill="none" stroke="#9aa7b8" stroke-width="0.012" stroke-dasharray="0.05 0.035"/>`);
        }
        // The planned trajectory.
        if (samples.length > 1) {
            const d = samples.map((s, i) => `${i ? 'L' : 'M'} ${f(sx(Number(s.x)))} ${f(sy(Number(s.y)))}`).join(' ');
            parts.push(`<path d="${d}" fill="none" stroke="#0b6fa4" stroke-width="0.022" stroke-linejoin="round" stroke-linecap="round"/>`);
        }
        // The waypoints.
        waypoints.forEach((p, i) => {
            const x = sx(Number(p.x)); const y = sy(Number(p.y));
            parts.push(`<circle cx="${f(x)}" cy="${f(y)}" r="0.032" fill="#f472b6" stroke="#ffffff" stroke-width="0.01"/>`);
            if (p.heading !== null && p.heading !== undefined) {
                const hx = Number(p.x) + Math.cos(Number(p.heading)) * 0.12;
                const hy = Number(p.y) + Math.sin(Number(p.heading)) * 0.12;
                parts.push(`<line x1="${f(x)}" y1="${f(y)}" x2="${f(sx(hx))}" y2="${f(sy(hy))}" stroke="#14181f" stroke-width="0.012"/>`);
            }
            parts.push(`<text x="${f(x)}" y="${f(y - 0.055)}" font-size="0.075" `
                + `text-anchor="middle" fill="#5f6b7c" font-family="monospace">W${i}</text>`);
        });
        parts.push('</svg>');
        return parts.join('');
    }

    /* The wheel speeds over the run. */
    function velocityDiagram(data) {
        const samples = (data.result && data.result.samples) || [];
        if (samples.length < 2) return '';
        const t0 = Number(samples[0].t);
        const t1 = Number(samples[samples.length - 1].t);
        const span = Math.max(t1 - t0, 0.1);
        let vmax = 0;
        samples.forEach((s) => {
            vmax = Math.max(vmax, Math.abs(Number(s.vl) || 0), Math.abs(Number(s.vr) || 0));
        });
        vmax = Math.max(vmax, 0.1);
        const W = 100; const H = 31;
        const px = (t) => (4 + ((Number(t) - t0) / span) * (W - 8)).toFixed(3);
        const py = (v) => (H - 3.5 - ((Number(v) + vmax) / (2 * vmax)) * (H - 7)).toFixed(3);
        const line = (key) => samples.map((s, i) => `${i ? 'L' : 'M'} ${px(s.t)} ${py(s[key])}`).join(' ');
        const parts = [`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="The wheel speeds over the run">`];
        parts.push(`<line x1="4" y1="${py(0)}" x2="${W - 4}" y2="${py(0)}" stroke="#c8d1dd" stroke-width="0.4"/>`);
        parts.push(`<path d="${line('vl')}" fill="none" stroke="#0b6fa4" stroke-width="1.1" stroke-linejoin="round"/>`);
        parts.push(`<path d="${line('vr')}" fill="none" stroke="#e08b2f" stroke-width="1.1" stroke-linejoin="round" stroke-dasharray="3 2"/>`);
        parts.push(`<text x="4" y="${(H - 0.5).toFixed(3)}" font-size="2.6" fill="#5f6b7c" font-family="monospace">${t0.toFixed(2)}s</text>`);
        parts.push(`<text x="${W - 4}" y="${(H - 0.5).toFixed(3)}" font-size="2.6" text-anchor="end" fill="#5f6b7c" font-family="monospace">${t1.toFixed(2)}s</text>`);
        parts.push(`<text x="4" y="3.4" font-size="2.6" fill="#5f6b7c" font-family="monospace">${vmax.toFixed(2)} m/s</text>`);
        parts.push('</svg>');
        return parts.join('');
    }

    /* --------------------------------------------------------------- tables */

    function waypointTable(waypoints) {
        const rows = waypoints.map((w, i) => {
            const tags = [];
            if (w.stop) tags.push('<span class="tag stop">stop</span>');
            if (w.event) tags.push(`<span class="tag event">${esc(w.event)}</span>`);
            const heading = (w.heading === null || w.heading === undefined)
                ? 'free'
                : `${num(Number(w.heading) * 180 / Math.PI, 1)}\u00b0`;
            return `<tr><td class="mono">W${i}</td>`
                + `<td class="num mono">${num(w.x, 3)}</td>`
                + `<td class="num mono">${num(w.y, 3)}</td>`
                + `<td class="num mono">${heading}</td>`
                + `<td>${tags.join(' ')}</td></tr>`;
        });
        return `<table><thead><tr><th>Waypoint</th><th class="num">X (m)</th>`
            + `<th class="num">Y (m)</th><th class="num">Heading</th><th></th></tr></thead>`
            + `<tbody>${rows.join('')}</tbody></table>`;
    }

    function candidateTable(candidates, selected) {
        if (!candidates || !candidates.length) return '';
        const rows = candidates.map((entry, i) => {
            const ok = entry.ok !== false;
            const accepted = Boolean(entry.accepted);
            const reasons = (entry.reasons || []).join('; ');
            const score = entry.score === null || entry.score === undefined
                ? '-' : num(entry.score, 1);
            const verdict = ok ? 'pass' : (accepted ? 'accepted' : esc(reasons || 'failed'));
            return `<tr class="${ok ? '' : (accepted ? 'warn' : 'bad')}">`
                + `<td class="mono">${i + 1}${i === selected ? ' \u2713' : ''}</td>`
                + `<td>${esc(entry.timing === 'per-segment' ? 'per segment' : 'equal duration')}`
                + `${entry.accuracy_weight ? `, weight ${esc(entry.accuracy_weight)}` : ''}</td>`
                + `<td class="num mono">${entry.duration_s === null || entry.duration_s === undefined ? '-' : `${num(entry.duration_s, 3)} s`}</td>`
                + `<td class="num mono">${entry.envelope_cm === null || entry.envelope_cm === undefined ? '-' : `${num(entry.envelope_cm, 2)} cm`}</td>`
                + `<td class="num mono">${entry.tracking_cm === null || entry.tracking_cm === undefined ? '-' : `${num(entry.tracking_cm, 2)} cm`}</td>`
                + `<td class="num mono">${score}</td>`
                + `<td class="verdict-cell">${verdict}</td></tr>`;
        });
        return `<table><thead><tr><th>#</th><th>Timing</th><th class="num">Duration</th>`
            + `<th class="num">Envelope</th><th class="num">Tracking</th>`
            + `<th class="num">Score</th><th>Verdict</th></tr></thead>`
            + `<tbody>${rows.join('')}</tbody></table>`;
    }

    function segmentTable(segments) {
        if (!segments || !segments.length) return '';
        const rows = segments.map((segment) => {
            const ok = segment.ok !== false;
            return `<tr class="${ok ? '' : 'bad'}">`
                + `<td class="mono">W${segment.from} \u2192 W${segment.to}</td>`
                + `<td class="num mono">${cm(segment.envelope_m)}</td>`
                + `<td class="num mono">${cm(segment.limit_m)}</td>`
                + `<td class="num mono">${cm(segment.drift_m)}</td></tr>`;
        });
        return `<table><thead><tr><th>Segment</th><th class="num">Envelope</th>`
            + `<th class="num">Limit</th><th class="num">Drift</th></tr></thead>`
            + `<tbody>${rows.join('')}</tbody></table>`;
    }

    /* ---------------------------------------------------------------- facts */

    function fact(label, value, note) {
        return `<div><dt>${esc(label)}</dt><dd>${value}`
            + (note ? ` <small>${esc(note)}</small>` : '') + '</dd></div>';
    }

    function robotFacts(robot) {
        const fields = (robot && robot.fields) || {};
        const out = [];
        if (fields.mass !== undefined) out.push(fact('Mass', `${num(fields.mass, 3)}`, 'kg'));
        if (fields.wheel_radius !== undefined) {
            out.push(fact('Wheel radius', `${num(fields.wheel_radius * 1000, 1)}`, 'mm'));
        }
        if (fields.track_width !== undefined) {
            out.push(fact('Track width', `${num(fields.track_width * 1000, 1)}`, 'mm'));
        }
        if (fields.cof !== undefined) out.push(fact('Friction', num(fields.cof, 2), 'cof'));
        if (fields.t_max_nm !== undefined) {
            out.push(fact('Stall torque', num(fields.t_max_nm, 3), 'N m'));
        }
        if (fields.v_max_rad_s !== undefined) {
            out.push(fact('No-load speed', num(fields.v_max_rad_s, 2), 'rad/s'));
        }
        return out.join('');
    }

    /* ----------------------------------------------------------------- page */

    function buildMissionReport(data) {
        data = data || {};
        const mission = data.mission || {};
        const robot = data.robot || {};
        const field = data.field || {};
        const result = data.result || {};
        const waypoints = mission.waypoints || [];
        const summary = result.summary || {};

        const when = result.when || new Date().toISOString();
        const dateText = new Date(when).toLocaleString(undefined, {
            year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit',
        });

        // The verdict: what the gate said, in one word. Acceptance is a
        // decision against a threshold; it is never called a pass.
        let verdictClass = 'warn';
        let verdictText = 'Not validated';
        let verdictNote = 'Validation was off, so the plan was not checked against the gates.';
        if (result.failed && result.outcome !== 'accepted') {
            verdictClass = 'bad';
            verdictText = 'Failed validation';
            verdictNote = result.reason || 'The plan did not pass the gates.';
        } else if (result.outcome === 'accepted') {
            verdictClass = 'warn';
            verdictText = 'Accepted plan';
            verdictNote = 'The plan failed validation and was accepted at the recorded score. '
                + 'It is not a passed plan.';
        } else if (result.validated) {
            verdictClass = 'ok';
            verdictText = 'Passed validation';
            verdictNote = 'The plan is inside every reachability and tracking gate.';
        }

        const resultFacts = [];
        if (result.score !== null && result.score !== undefined) {
            resultFacts.push(fact('Score', `${num(result.score, 1)}`,
                '/ 100 decision score - decision units, not a probability of hardware success'));
            resultFacts.push(fact('Score version', String(result.score_version || 1), ''));
            resultFacts.push(fact('Profile', esc(result.profile || 'balanced'), ''));
        }
        if (summary.duration_s !== undefined) resultFacts.push(fact('Duration', `${num(summary.duration_s, 3)}`, 's'));
        if (summary.path_length_m !== undefined) resultFacts.push(fact('Path', `${num(summary.path_length_m, 3)}`, 'm'));
        if (summary.peak_wheel_speed_m_s !== undefined) {
            resultFacts.push(fact('Peak wheel speed', num(summary.peak_wheel_speed_m_s, 2), 'm/s'));
        }
        if (summary.has_envelope && summary.max_envelope_m !== undefined) {
            resultFacts.push(fact('Max envelope', num(summary.max_envelope_m * 100, 2), 'cm'));
        }
        if (result.tracking_cm !== null && result.tracking_cm !== undefined) {
            resultFacts.push(fact('Tracking error', num(result.tracking_cm, 2), 'cm'));
        }
        /* The field-clearance channels of the physical side
           (--field-clearance): the util is the tube over the unclamped
           clearance (1.0 = touching), the penetration how far past it the
           tube reaches. */
        if (result.field_clearance_util !== null
                && result.field_clearance_util !== undefined) {
            resultFacts.push(fact('Field clearance util',
                `${num(result.field_clearance_util, 2)}x`,
                'of the field clearance (1.0 = touching)'));
        }
        if (result.penetration_m !== null && result.penetration_m !== undefined
                && Number(result.penetration_m) > 0) {
            resultFacts.push(fact('Field penetration',
                num(result.penetration_m * 100, 2), 'cm past the clearance'));
        }
        if (result.wall_s !== null && result.wall_s !== undefined) {
            resultFacts.push(fact('Solve (wall)', num(result.wall_s, 2), 's'));
        }

        const model = robot.model;
        const modelText = !robot.useModel
            ? 'The nominal model: the declared motor and the nominal gains.'
            : (model
                ? `Measured: straight gain ${num(model.straight && model.straight.gain, 4)}`
                  + (model.straight && model.straight.gain_ci95
                    ? ` \u00b1 ${num(model.straight.gain_ci95, 4)}` : '')
                  + ', time constant '
                  + `${num(model.actuator && model.actuator.tau_s, 4)} s`
                : 'Measured: the fit file is no longer on disk.');

        const options = data.options || {};
        const optionText = [
            `${options.samples} samples per segment`,
            `accuracy weight ${options.accuracy_weight}`,
            `${options.candidates} candidate${options.candidates === 1 ? '' : 's'}`,
            options.validate ? 'validated' : 'not validated',
            options.fast_polish ? 'fast polish' : null,
        ].filter(Boolean).join(' \u00b7 ');

        const body = [];
        body.push(`<div class="head"><div>`
            + `<div class="brand">OVERBOARD</div>`
            + `<h1>${esc(mission.name || 'Mission')}</h1>`
            + `</div><div class="meta">`
            + `<div>${esc(dateText)}</div>`
            + `<div>${esc(robot.name || 'the default robot')}</div>`
            + (field.season ? `<div>${esc(field.season)} field</div>` : '')
            + `</div></div>`);

        body.push(`<div class="verdict ${verdictClass}"><span class="dot"></span>`
            + `${esc(verdictText)}</div>`);

        /* The mission. */
        body.push('<section><h2>The mission</h2>');
        body.push(`<dl class="facts">`
            + fact('Waypoints', String(waypoints.length))
            + (result.path_length_m !== undefined
                ? fact('Path length', num(result.path_length_m, 3), 'm') : '')
            + (result.min_time_s !== undefined
                ? fact('At least', num(result.min_time_s, 2), 's at top speed') : '')
            + (result.events ? fact('Events', String(result.events)) : '')
            + `</dl>`);
        const fieldSvg = fieldDiagram(data);
        const missionTable = waypointTable(waypoints);
        if (fieldSvg || missionTable) {
            body.push('<div class="figures">');
            if (fieldSvg) body.push(`<figure><figcaption>The planned path</figcaption>${fieldSvg}</figure>`);
            if (missionTable) body.push(`<figure><figcaption>The waypoints</figcaption>${missionTable}</figure>`);
            body.push('</div>');
        }
        body.push('</section>');

        /* The robot. */
        body.push('<section><h2>The robot</h2>');
        body.push(`<dl class="facts">${robotFacts(robot)}</dl>`);
        body.push(`<p class="note">${esc(modelText)}</p>`);
        body.push('</section>');

        /* The result. */
        body.push('<section><h2>The result</h2>');
        if (resultFacts.length) body.push(`<dl class="facts">${resultFacts.join('')}</dl>`);
        if (!resultFacts.length) {
            body.push('<p class="note">No solve yet, so there is no result to report.</p>');
        }
        const candidates = candidateTable(result.candidates, result.selected);
        if (candidates) {
            body.push('<h3>The candidates</h3>');
            body.push(candidates);
        }
        const segments = segmentTable(result.segments);
        const velocity = velocityDiagram(data);
        if (segments || velocity) {
            body.push('<div class="figures">');
            if (segments) body.push(`<figure><figcaption>Segment by segment</figcaption>${segments}</figure>`);
            if (velocity) body.push(`<figure><figcaption>Wheel speeds</figcaption>${velocity}</figure>`);
            body.push('</div>');
        }
        body.push(`<p class="note">${esc(optionText)}</p>`);
        body.push('</section>');

        body.push(`<div class="foot"><div>OVERBOARD Studio</div>`
            + `<div>A snapshot of the studio at the time of the solve. `
            + `The numbers are the ones the studio showed.</div></div>`);

        return `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${esc(mission.name || 'Mission')} - OVERBOARD mission report</title>
<style>${CSS}</style></head>
<body>
<div class="toolbar">
  <button type="button" id="print">Print / save as PDF</button>
  <button type="button" id="save">Download this page</button>
</div>
<main class="page">${body.join('\n')}</main>
<script>
document.getElementById('print').addEventListener('click', function () { window.print(); });
document.getElementById('save').addEventListener('click', function () {
  var source = '<!DOCTYPE html>\\n' + document.documentElement.outerHTML;
  var blob = new Blob([source], { type: 'text/html' });
  var link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = ${JSON.stringify((mission.name || 'mission').replace(/[^a-z0-9_-]+/gi, '-'))} + '-report.html';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  setTimeout(function () { URL.revokeObjectURL(link.href); }, 1000);
});
</script>
</body></html>`;
    }

    global.buildMissionReport = buildMissionReport;
}(typeof window !== 'undefined' ? window : this));
