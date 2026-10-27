"""
The measurement sheet as a form.

The operator has a couple of dozen numbers to enter and they all come off a
tape measure, so the sheet is a form in a browser rather than a CSV in a text
editor. The form writes the same CSV the fitter reads, so the browser path
and the spreadsheet path produce the same file:

    python sysid_fit.py session.csv --write-form calibration_sheet.html

The form is one self-contained file: no server, no network and no build step,
so it works on the same laptop that pushes the trajectory to the hub.

It is a guided form: one card per test, saying what the robot does in that
test, what to measure, and the exact three numbers wanted; a diagram of the
measurement frame at the top; unit labels on every input; soft warnings for
a number that cannot be the robot's (never blocking the download); and a
count of what is still missing. The wording of the measurement convention is
shared with ``overboard.sysid.collect``'s step banners (``CONVENTION`` and
``TEST_GUIDE``), so the console and the form tell the operator the same
thing.

The studio has its own version of this form, on the server
(``gui_server.py``). This one is for the command line, where there is no
server to hold the rows.
"""

import html

# The columns the fitter reads. The form writes exactly these, in this order.
SHEET_FIELDS = ["test", "repeat", "stop", "x_mm", "y_mm", "turn_right_deg",
                "mass_kg"]
CSV_HEADER = ",".join(SHEET_FIELDS)
DEFAULT_CSV_NAME = "calibration_sheet.csv"

# The weigh-in rows: the robot's mass with the battery, on a bench scale,
# three times per the KI-010 discipline. They are bench measurements, not
# hub tests, so they are always the same three rows and are appended to
# every sheet and form.
WEIGH_IN_TEST = "weigh_in"
WEIGH_REPEATS = 3

# The measurement convention, in the words every surface uses: the form's
# top, the collect banners, and docs/CALIBRATION_PROCEDURE.md all say it the
# same way. The frame is the one the robot sat in at the start of the test.
CONVENTION = (
    "Every measurement is in the frame the robot sat in at the start of the "
    "test. x_mm is how far the robot ended forward from the start pose (998 "
    "means it went 998 mm forward). y_mm is how far it ended to the left of "
    "the runway (20 means it drifted 20 mm left). turn_right_deg is how far "
    "it turned to the right (90 means it ended pointing 90 degrees right of "
    "the runway). The turn is written to the right because that is how you "
    "read it off the robot; the tool flips the sign. Measure from the "
    "robot's centre, not its bumper."
)

# What each measured test is, in plain words: what the robot does, what to
# measure, and the exact numbers wanted. Shared with the collect banners.
TEST_GUIDE = {
    "odometry_straight": {
        "does": "The robot drives straight at 100 mm/s for 10 seconds, "
                "about one metre.",
        "measure": "Where the robot's centre ended, from the start mark. "
                   "The sideways drift matters as much as the distance: it "
                   "is how the fit sees the robot curving when it is told "
                   "to drive straight.",
        "numbers": "x_mm: how far it went forward (about 1000). y_mm: how "
                   "far it drifted left (0 if it stayed on the line). "
                   "turn_right_deg: how far it turned right (0 if it kept "
                   "its heading).",
    },
    "odometry_turn_cw": {
        "does": "The robot turns clockwise on the spot at 50 deg/s for 1.8 "
                "seconds, about a quarter turn.",
        "measure": "Where the robot's centre ended, and which way it points.",
        "numbers": "x_mm: where it ended forward (near 0 if it only turned). "
                   "y_mm: where it ended left. turn_right_deg: how far it "
                   "turned to the right (about 90).",
    },
    "multi_stop": {
        "does": "The robot drives down the runway and pauses at three stops, "
                "at about 500, 1000 and 1500 mm. The hub waits at each stop "
                "for you.",
        "measure": "The pose at each of the three stops. One row per stop, "
                   "all measured from the same start mark.",
        "numbers": "For each stop: x_mm how far forward it is, y_mm how far "
                   "left, turn_right_deg how far it turned right.",
    },
    "brake_from_300": {
        "does": "The robot drives and brakes hard from 300 mm/s. It may "
                "skid and end short - that gap is the brake slip, and it is "
                "the point of the test.",
        "measure": "Where the robot's centre ended after the stop.",
        "numbers": "x_mm: how far forward it ended (short of the drive-to "
                   "point is normal). y_mm: how far left. turn_right_deg: "
                   "how far it turned right.",
    },
    "brake_from_600": {
        "does": "The same as brake_from_300, braking from 600 mm/s.",
        "measure": "Where the robot's centre ended after the stop.",
        "numbers": "x_mm: how far forward it ended (short of the drive-to "
                   "point is normal). y_mm: how far left. turn_right_deg: "
                   "how far it turned right.",
    },
    WEIGH_IN_TEST: {
        "does": "Not a hub test. This one happens on the kitchen scale.",
        "measure": "The robot's mass with the battery in, three times. Pick "
                   "it up, put it down, read the scale each time - a repeat "
                   "that re-weighs the same static reading is not a repeat.",
        "numbers": "mass_kg: the mass in kilograms (about 0.85 for a small "
                   "robot).",
    },
}

# A test the guide does not know: the session still asked for a pose, so the
# generic instruction is the right one.
GUIDE_FALLBACK = {
    "does": "The hub ran this test and asked for a measurement.",
    "measure": "Where the robot's centre ended at each stop, measured from "
               "the start mark.",
    "numbers": "x_mm: how far forward. y_mm: how far left. "
               "turn_right_deg: how far it turned right.",
}


def guide_for(test):
    """What to say about one test: what the robot does and what to measure."""
    return TEST_GUIDE.get(test, GUIDE_FALLBACK)


def weigh_in_rows():
    """The blank weigh-in rows every sheet carries."""
    return [{"test": WEIGH_IN_TEST, "repeat": repeat, "stop": 0}
            for repeat in range(WEIGH_REPEATS)]


_DIAGRAM = (
    '<svg viewBox="0 0 460 210" role="img" aria-label="The measurement '
    'frame: x forward, y to the left, a turn to the right">'
    '<line x1="40" y1="120" x2="330" y2="120" stroke="currentColor" '
    'stroke-dasharray="4 4" opacity="0.4"/>'
    '<rect x="52" y="104" width="56" height="32" fill="none" '
    'stroke="currentColor" stroke-width="2"/>'
    '<line x1="108" y1="120" x2="126" y2="120" stroke="currentColor" '
    'stroke-width="2"/>'
    '<polygon points="126,114 140,120 126,126" fill="currentColor"/>'
    '<text x="40" y="98" font-size="12" fill="currentColor">start pose '
    '(the robot&apos;s centre)</text>'
    '<line x1="150" y1="160" x2="330" y2="160" stroke="currentColor" '
    'stroke-width="2"/>'
    '<polygon points="330,154 344,160 330,166" fill="currentColor"/>'
    '<text x="196" y="182" font-size="13" fill="currentColor">x_mm: '
    'forward</text>'
    '<line x1="150" y1="160" x2="150" y2="56" stroke="currentColor" '
    'stroke-width="2"/>'
    '<polygon points="144,56 150,42 156,56" fill="currentColor"/>'
    '<text x="160" y="50" font-size="13" fill="currentColor">y_mm: to the '
    'left</text>'
    '<path d="M 250 108 A 44 44 0 0 1 294 152" fill="none" '
    'stroke="currentColor" stroke-width="2"/>'
    '<polygon points="299,157 288,164 281,153" fill="currentColor"/>'
    '<text x="308" y="118" font-size="13" fill="currentColor">'
    'turn_right_deg:</text>'
    '<text x="308" y="136" font-size="13" fill="currentColor">to the '
    'right</text>'
    "</svg>"
)

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.45 system-ui, -apple-system, Segoe UI, sans-serif;
         margin: 0; padding: 24px; }
  h1 { font-size: 20px; margin: 0 0 6px; }
  h2 { font-size: 16px; margin: 0 0 6px; }
  .sub, .note, .how, .numbers { opacity: .82; margin: 0 0 10px;
                                max-width: 78ch; }
  .bar { display: flex; gap: 12px; align-items: center; margin: 18px 0;
         flex-wrap: wrap; }
  button { font: inherit; padding: 7px 14px; cursor: pointer; }
  .done { color: #1a7f37; font-weight: 600; }
  .card { border: 1px solid rgba(128,128,128,.35); border-radius: 8px;
          padding: 12px 14px; margin: 0 0 14px; max-width: 920px; }
  .card h2 .plain { font-weight: 400; opacity: .75; }
  .card .card-count { float: right; font-size: 13px; opacity: .75; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border-bottom: 1px solid rgba(128,128,128,.35); padding: 5px 8px;
           text-align: left; }
  th { font-weight: 600; }
  td.key { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
           white-space: nowrap; }
  input[type=number] { width: 92px; padding: 4px 6px; font: inherit;
                       box-sizing: border-box; }
  .unit { opacity: .7; margin-left: 4px; font-size: 13px; }
  .warn { color: #b45309; }
  tr.has-warning input { outline: 2px solid #b45309; }
  .progress { height: 10px; background: rgba(128,128,128,.25);
              border-radius: 5px; overflow: hidden; max-width: 420px;
              margin: 6px 0 10px; }
  .progress > div { height: 100%; width: 0; background: #1a7f37; }
  svg { max-width: 460px; height: auto; display: block; margin: 10px 0; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  @media print { .bar, input, .progress { display: none; }
                 th, td { border: none; } }
</style>
</head>
<body>
<h1>{{TITLE}}</h1>
<p class="sub">{{CONVENTION}}</p>
{{DIAGRAM}}
<div class="bar">
  <button type="button" id="download">Download CSV</button>
  <button type="button" id="copy">Copy CSV</button>
  <label class="note">Load a sheet you saved earlier:
    <input type="file" id="load" accept=".csv,text/csv"></label>
</div>
<div class="progress"><div id="progress-fill"></div></div>
<p id="count"></p>
<ul id="missing" class="note"></ul>
<ul id="warnings" class="warn"></ul>
{{CARDS}}
<p class="note">Save the CSV in the folder you run the command from, then fit the
model:
<br><code>{{FIT_COMMAND}}</code>
</p>
<script>
(function () {
  var HEADER = "{{CSV_HEADER}}";
  var CSV_NAME = "{{CSV_NAME}}";

  function q(sel) { return document.querySelector(sel); }
  function rows() {
    return Array.prototype.slice.call(
      document.querySelectorAll("tr[data-test]"));
  }
  function cell(value) {
    var text = String(value);
    return /[",\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  }
  function rowValues(tr) {
    var m = tr.querySelector(".m");
    return [
      tr.getAttribute("data-test"),
      tr.getAttribute("data-repeat"),
      tr.getAttribute("data-stop"),
      m ? "" : tr.querySelector(".x").value.trim(),
      m ? "" : tr.querySelector(".y").value.trim(),
      m ? "" : tr.querySelector(".t").value.trim(),
      m ? m.value.trim() : ""
    ];
  }
  function values() { return rows().map(rowValues); }
  function csv() {
    var lines = [HEADER];
    values().forEach(function (row) {
      lines.push(row.map(cell).join(","));
    });
    return lines.join("\n") + "\n";
  }
  // A number that cannot be the robot's gets a note, never a block: the
  // operator may know better, and the download always works.
  function rowWarning(row) {
    var out = [];
    function num(i) { return row[i] === "" ? null : Number(row[i]); }
    var turn = num(5);
    if (turn !== null && isFinite(turn) && (turn < 0 || turn > 360)) {
      out.push("turn_right_deg is usually 0 to 360 to the right; "
        + "check the sign and the units");
    }
    var x = num(3);
    if (x !== null && isFinite(x) && Math.abs(x) > 3000) {
      out.push("x_mm is over 3000 mm; check the units (millimetres)");
    }
    var y = num(4);
    if (y !== null && isFinite(y) && Math.abs(y) > 3000) {
      out.push("y_mm is over 3000 mm; check the units (millimetres)");
    }
    var mass = num(6);
    if (mass !== null && isFinite(mass) && (mass <= 0 || mass > 5)) {
      out.push("mass_kg is outside 0.05 to 5 kg; weigh the robot with the "
        + "battery in");
    }
    return out;
  }
  function done(row) {
    // A row with no mass needs its three measurements, and vice versa.
    return row[6] !== "" ||
      (row[3] !== "" && row[4] !== "" && row[5] !== "");
  }
  function label(row) {
    return row[0] + " repeat " + row[1] + " stop " + row[2];
  }
  function update() {
    var all = values();
    var missing = all.filter(function (row) { return !done(row); });
    var el = q("#count");
    if (missing.length === 0) {
      el.textContent = "all " + all.length + " measurements entered";
      el.className = "done";
    } else {
      el.textContent = missing.length + " of " + all.length +
        " measurements still missing";
      el.className = "";
    }
    q("#progress-fill").style.width = Math.round(
      100 * (all.length - missing.length) / Math.max(all.length, 1)) + "%";
    var list = q("#missing");
    list.innerHTML = "";
    missing.forEach(function (row) {
      var li = document.createElement("li");
      li.textContent = label(row);
      list.appendChild(li);
    });
    var warnings = q("#warnings");
    warnings.innerHTML = "";
    rows().forEach(function (tr) {
      var notes = rowWarning(rowValues(tr));
      tr.classList.toggle("has-warning", notes.length > 0);
      notes.forEach(function (note) {
        var li = document.createElement("li");
        li.textContent = label(rowValues(tr)) + ": " + note;
        warnings.appendChild(li);
      });
    });
    Array.prototype.forEach.call(
      document.querySelectorAll("section.card"), function (card) {
        var mine = Array.prototype.slice.call(
          card.querySelectorAll("tr[data-test]")).map(rowValues);
        var left = mine.filter(function (row) { return !done(row); }).length;
        card.querySelector(".card-count").textContent =
          (mine.length - left) + " of " + mine.length + " entered";
      });
  }
  q("#download").addEventListener("click", function () {
    var blob = new Blob([csv()], {type: "text/csv;charset=utf-8"});
    var link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = CSV_NAME;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });
  q("#copy").addEventListener("click", function () {
    navigator.clipboard.writeText(csv()).then(function () {
      q("#count").textContent = "the CSV is on the clipboard";
    });
  });
  q("#load").addEventListener("change", function (event) {
    var file = event.target.files[0];
    if (!file) { return; }
    file.text().then(function (text) {
      var lines = text.split(/\r?\n/).filter(function (line) {
        return line.trim() !== "";
      });
      if (lines.length < 2) { return; }
      var head = lines[0].split(",").map(function (name) {
        return name.trim();
      });
      var ix = head.indexOf("x_mm");
      var iy = head.indexOf("y_mm");
      var it = head.indexOf("turn_right_deg");
      var im = head.indexOf("mass_kg");
      if (ix < 0 || iy < 0 || it < 0) {
        q("#count").textContent =
          "that file has no x_mm, y_mm and turn_right_deg columns";
        return;
      }
      var index = {};
      rows().forEach(function (tr) {
        index[tr.getAttribute("data-test") + "|" +
              tr.getAttribute("data-repeat") + "|" +
              tr.getAttribute("data-stop")] = tr;
      });
      lines.slice(1).forEach(function (line) {
        var parts = line.split(",");
        var key = parts[0].trim() + "|" + (parts[1] || "").trim() + "|" +
          (parts[2] || "").trim();
        var tr = index[key];
        if (!tr) { return; }
        var m = tr.querySelector(".m");
        if (m) {
          if (im >= 0) { m.value = (parts[im] || "").trim(); }
          return;
        }
        tr.querySelector(".x").value = (parts[ix] || "").trim();
        tr.querySelector(".y").value = (parts[iy] || "").trim();
        tr.querySelector(".t").value = (parts[it] || "").trim();
      });
      update();
    });
  });
  rows().forEach(function (tr) {
    ["x", "y", "t", "m"].forEach(function (name) {
      var el = tr.querySelector("." + name);
      if (el) { el.addEventListener("input", update); }
    });
  });
  update();
})();
</script>
</body>
</html>
"""

_ROW = (
    '<tr data-test="{test}" data-repeat="{repeat}" data-stop="{stop}">'
    '<td class="key">{repeat}</td>'
    '<td class="key">{stop}</td>'
    '<td><input type="number" step="any" inputmode="decimal" class="x">'
    '<span class="unit">mm</span></td>'
    '<td><input type="number" step="any" inputmode="decimal" class="y">'
    '<span class="unit">mm</span></td>'
    '<td><input type="number" step="any" inputmode="decimal" class="t">'
    '<span class="unit">deg</span></td>'
    "</tr>"
)

_MASS_ROW = (
    '<tr data-test="{test}" data-repeat="{repeat}" data-stop="{stop}">'
    '<td class="key">{repeat}</td>'
    '<td class="key">{stop}</td>'
    '<td><input type="number" step="any" inputmode="decimal" class="m">'
    '<span class="unit">kg</span></td>'
    "</tr>"
)

_MEASURE_HEAD = (
    "<tr><th>repeat</th><th>stop</th>"
    "<th>x_mm (forward)</th><th>y_mm (left)</th>"
    "<th>turn_right_deg (right)</th></tr>"
)

_MASS_HEAD = "<tr><th>repeat</th><th>stop</th><th>mass_kg</th></tr>"


def row_html(row):
    """One table row for one measurement (or one weigh-in)."""
    template = _MASS_ROW if row["test"] == WEIGH_IN_TEST else _ROW
    return template.format(
        test=html.escape(str(row["test"])),
        repeat=html.escape(str(row["repeat"])),
        stop=html.escape(str(row["stop"])),
    )


def _card_html(test, rows):
    """One card: what the test is, and its measurement rows."""
    guide = guide_for(test)
    weigh = test == WEIGH_IN_TEST
    body = "\n".join(row_html(row) for row in rows)
    # quote=False: this copy lands in element text, where an apostrophe is
    # an apostrophe and only &, < and > need escaping.
    def esc(text):
        return html.escape(text, quote=False)
    return (
        '<section class="card">'
        f"<h2>{html.escape(str(test))} "
        f'<span class="card-count"></span></h2>'
        f'<p class="how"><b>What the robot does:</b> '
        f'{esc(guide["does"])}</p>'
        f'<p class="how"><b>What to measure:</b> '
        f'{esc(guide["measure"])}</p>'
        f'<p class="numbers"><b>The numbers wanted:</b> '
        f'{esc(guide["numbers"])}</p>'
        f"<table><thead>{_MASS_HEAD if weigh else _MEASURE_HEAD}</thead>"
        f"<tbody>{body}</tbody></table>"
        "</section>"
    )


def write_form(rows, path, csv_name=DEFAULT_CSV_NAME, title="Measurement sheet",
               fit_command=""):
    """Write the measurement sheet as a self-contained HTML form.

    ``rows`` are the rows of the sheet: a test name, a repeat and a stop for
    each measurement. The weigh-in rows are appended to them, so every form
    can land the mass too. The form writes a CSV with the same columns the
    fitter reads, so the file it downloads drops straight into ``--sheet``.

    Returns the path written.
    """
    groups = {}
    for row in list(rows) + weigh_in_rows():
        groups.setdefault(row["test"], []).append(row)
    cards = "\n".join(_card_html(test, grouped)
                      for test, grouped in groups.items())
    text = (_TEMPLATE
            .replace("{{TITLE}}", html.escape(title))
            .replace("{{CONVENTION}}", html.escape(CONVENTION, quote=False))
            .replace("{{DIAGRAM}}", _DIAGRAM)
            .replace("{{CARDS}}", cards)
            .replace("{{CSV_HEADER}}", CSV_HEADER)
            .replace("{{CSV_NAME}}", html.escape(csv_name))
            .replace("{{FIT_COMMAND}}", html.escape(fit_command)))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path
