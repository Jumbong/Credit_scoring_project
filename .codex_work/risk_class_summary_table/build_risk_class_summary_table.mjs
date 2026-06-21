import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = "/Users/juniorjumbong/Desktop/personal-website/00_tds/01_PD_credit_scoring";
const outputDir = path.join(projectRoot, "outputs", "model_selection");
const sourceCsv = path.join(outputDir, "score_risk_classes.csv");
const outputXlsx = path.join(outputDir, "score_risk_classes_summary_table.xlsx");
const outputPng = path.join(outputDir, "score_risk_classes_summary_table.png");

function parseSimpleCsv(csvText) {
  const lines = csvText.trim().split(/\r?\n/);
  const headers = lines[0].split(",");

  return lines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function toPercent(value) {
  if (value === "" || value === undefined || value === null || Number.isNaN(Number(value))) {
    return "-";
  }

  return Number(value);
}

const csvText = await fs.readFile(sourceCsv, "utf8");
const sourceRows = parseSimpleCsv(csvText);
const rows = sourceRows.map((row) => [
  Number(row.score_class),
  row.score_interval,
  row.risk_label,
  toPercent(row.population_share),
  toPercent(row.default_rate),
  toPercent(row.relative_gap_vs_previous),
]);

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Risk classes");
sheet.showGridLines = false;

const headers = [
  "#",
  "Score interval",
  "Risk label",
  "Population share",
  "Default rate",
  "Relative gap",
];
const table = [headers, ...rows];

sheet.getRange("A1:F7").values = table;

const headerRange = sheet.getRange("A1:F1");
headerRange.format = {
  fill: "#9D1712",
  font: { bold: true, color: "#FFFFFF", size: 12 },
  horizontalAlignment: "center",
  verticalAlignment: "middle",
};

const tableRange = sheet.getRange("A1:F7");
tableRange.format.borders = {
  preset: "all",
  style: "thin",
  color: "#FFFFFF",
};

sheet.getRange("A2:A7").format = {
  fill: "#7F0D09",
  font: { bold: true, color: "#FFFFFF", size: 12 },
  horizontalAlignment: "center",
  verticalAlignment: "middle",
};

for (let rowIndex = 2; rowIndex <= 7; rowIndex += 1) {
  const fill = rowIndex % 2 === 0 ? "#E9E9E9" : "#C7C7C7";
  const rowRange = sheet.getRange(`B${rowIndex}:F${rowIndex}`);
  rowRange.format = {
    fill,
    font: { color: "#222222", size: 11 },
    horizontalAlignment: "center",
    verticalAlignment: "middle",
  };
}

sheet.getRange("B2:C7").format.horizontalAlignment = "center";
sheet.getRange("D2:F7").format.numberFormat = "0.00%";
sheet.getRange("F2").values = [["-"]];
sheet.getRange("F2").format.numberFormat = "@";

sheet.getRange("A1:A7").format.columnWidthPx = 52;
sheet.getRange("B1:B7").format.columnWidthPx = 185;
sheet.getRange("C1:C7").format.columnWidthPx = 165;
sheet.getRange("D1:D7").format.columnWidthPx = 125;
sheet.getRange("E1:E7").format.columnWidthPx = 135;
sheet.getRange("F1:F7").format.columnWidthPx = 125;
sheet.getRange("A1:F1").format.rowHeightPx = 42;
sheet.getRange("A2:F7").format.rowHeightPx = 40;
sheet.freezePanes.freezeRows(1);

const inspect = await workbook.inspect({
  kind: "table",
  range: "Risk classes!A1:F7",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 6,
});
console.log(inspect.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 20 },
});
console.log(errors.ndjson);

const preview = await workbook.render({
  sheetName: "Risk classes",
  range: "A1:F7",
  scale: 2,
  format: "png",
});
await fs.writeFile(outputPng, new Uint8Array(await preview.arrayBuffer()));

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputXlsx);

console.log(outputXlsx);
console.log(outputPng);
