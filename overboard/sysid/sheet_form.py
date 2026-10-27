"""
The measurement sheet as a form.

The operator has a couple of dozen numbers to enter and they all come off a
tape measure, so the sheet is a form in a browser rather than a CSV in a text
editor. The form writes the same CSV the fitter reads, so the browser path
and the spreadsheet path produce the same file:

    python sysid_fit.py session.csv --write-form calibration_sheet.html

The form is one self-contained file: no server, no network and no build step,
so it works on the same laptop that pushes the trajectory to the hub.

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


def weigh_in_rows():
    """The blank weigh-in rows every sheet carries."""
    return [{"test": WEIGH_IN_TEST, "repeat": repeat, "stop": 0}
            for repeat in range(WEIGH_REPEATS)]

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
  .sub { opacity: .78; margin: 0 0 18px; max-width: 78ch; }
  table { border-collapse: collapse; width: 100%; max-width: 920px; }
  th, td { border-bottom: 1px solid rgba(128,128,128,.35); padding: 5px 8px;
           text-align: left; }
  th { font-weight: 600; }
  td.key { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
           white-space: nowrap; }
  input[type=number] { width: 100%; max-width: 110px; padding: 4px 6px;
                       font: inherit; box-sizing: border-box; }
  .bar { display: flex; gap: 12px; align-items: center; margin: 18px 0;
         flex-wrap: wrap; }
  button { font: inherit; padding: 7px 14px; cursor: pointer; }
  .done { color: #1a7f37; font-weight: 600; }
  .note { max-width: 78ch; opacity: .8; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  @media print { .bar, input { display: none; } th, td { border: none; } }
</style>
</head>
<body>
<h1>{{TITLE}}</h1>
<p class="sub">
Every measurement is in the frame the robot sat in at the start of the test.
<strong>x_mm</strong> is how far it ended <em>forward</em> from the start pose
(998 means it went 998 mm forward). <strong>y_mm</strong> is how far it ended to
the <em>left</em> of the runway (20 means it drifted 20 mm left).
<strong>turn_right_deg</strong> is how far it turned to the <em>right</em>
(90 means it ended pointing 90 degrees right of the runway). Measure from the
robot's centre, not its bumper. The <strong>mass_kg</strong> rows are the
weigh-in: the robot's mass with the battery in, on a bench scale, three
times.
</p>
<div class="bar">
  <button type="button" id="download">Download CSV</button>
  <button type="button" id="copy">Copy CSV</button>
  <label class="note">Load a sheet you saved earlier:
    <input type="file" id="load" accept=".csv,text/csv"></label>
  <span id="count"></span>
</div>
<table>
<thead><tr><th>test</th><th>repeat</th><th>stop</th><th>x_mm</th><th>y_mm</th>
<th>turn_right_deg</th><th>mass_kg</th></tr></thead>
<tbody id="rows">
{{ROWS}}
</tbody>
</table>
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
      document.querySelectorAll("#rows tr"));
  }
  function cell(value) {
    var text = String(value);
    return /[",\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  }
  function values() {
    return rows().map(function (tr) {
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
    });
  }
  function csv() {
    var lines = [HEADER];
    values().forEach(function (row) {
      lines.push(row.map(cell).join(","));
    });
    return lines.join("\n") + "\n";
  }
  function update() {
    var all = values();
    var missing = all.filter(function (row) {
      // A row with no mass needs its three measurements, and vice versa.
      return row[6] === "" &&
        (row[3] === "" || row[4] === "" || row[5] === "");
    }).length;
    var el = q("#count");
    if (missing === 0) {
      el.textContent = "all " + all.length + " measurements entered";
      el.className = "done";
    } else {
      el.textContent = missing + " of " + all.length +
        " measurements still missing";
      el.className = "";
    }
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
    '<td class="key">{test}</td>'
    '<td class="key">{repeat}</td>'
    '<td class="key">{stop}</td>'
    '<td><input type="number" step="any" inputmode="decimal" class="x"></td>'
    '<td><input type="number" step="any" inputmode="decimal" class="y"></td>'
    '<td><input type="number" step="any" inputmode="decimal" class="t"></td>'
    "<td></td>"
    "</tr>"
)

_MASS_ROW = (
    '<tr data-test="{test}" data-repeat="{repeat}" data-stop="{stop}">'
    '<td class="key">{test}</td>'
    '<td class="key">{repeat}</td>'
    '<td class="key">{stop}</td>'
    "<td></td><td></td><td></td>"
    '<td><input type="number" step="any" inputmode="decimal" class="m"></td>'
    "</tr>"
)


def row_html(row):
    """One table row for one measurement (or one weigh-in)."""
    template = _MASS_ROW if row["test"] == WEIGH_IN_TEST else _ROW
    return template.format(
        test=html.escape(str(row["test"])),
        repeat=html.escape(str(row["repeat"])),
        stop=html.escape(str(row["stop"])),
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
    body = "\n".join(row_html(row) for row in list(rows) + weigh_in_rows())
    text = (_TEMPLATE
            .replace("{{TITLE}}", html.escape(title))
            .replace("{{ROWS}}", body)
            .replace("{{CSV_HEADER}}", CSV_HEADER)
            .replace("{{CSV_NAME}}", html.escape(csv_name))
            .replace("{{FIT_COMMAND}}", html.escape(fit_command)))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path
