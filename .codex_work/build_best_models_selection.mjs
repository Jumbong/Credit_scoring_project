import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const sourcePath = "/Users/juniorjumbong/Desktop/personal-website/00_tds/01_PD_credit_scoring/outputs/summary_models.xlsx";
const outputDir = "/Users/juniorjumbong/Desktop/personal-website/00_tds/01_PD_credit_scoring/outputs/model_selection";
const outputPath = path.join(outputDir, "best_models_selection.xlsx");

const RED = "#B51E18";
const DARK_RED = "#8E1712";
const YELLOW = "#F2BE35";
const LIGHT_YELLOW = "#FFF4D8";
const LIGHT_RED = "#F8E4E1";
const LIGHT_GRAY = "#F3F3F3";
const TEXT = "#222222";
const GREEN = "#3A7D44";

function byHeader(headers, row) {
  const obj = {};
  headers.forEach((header, index) => {
    obj[header] = row[index];
  });
  return obj;
}

function isOkModel(row) {
  return (
    row.global_signif_flag === "OK"
    && row.variables_signif_flag === "OK"
    && row.modalities_signif_flag === "OK"
    && row.vif_flag === "OK"
  );
}

function numeric(value) {
  if (value === null || value === undefined || value === "") {
    return Number.NEGATIVE_INFINITY;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
}

function simplifyFormula(rawFormula) {
  if (!rawFormula) {
    return "";
  }

  const targetMatch = String(rawFormula).match(/Q\("([^"]+)"\)\s*~/);
  const target = targetMatch ? targetMatch[1] : "def";
  const variables = [...String(rawFormula).matchAll(/C\(Q\("([^"]+)"\),\s*Treatment\(reference=.*?\)\)/g)]
    .map((match) => match[1]);

  if (variables.length === 0) {
    return String(rawFormula).replaceAll('Q("', "").replaceAll('")', "");
  }

  return `${target} ~ ${variables.join(" + ")}`;
}

function emptyIfMissing(value) {
  return value === Number.NEGATIVE_INFINITY ? "" : value;
}

const input = await FileBlob.load(sourcePath);
const sourceWorkbook = await SpreadsheetFile.importXlsx(input);
const selectedRows = [];
const detailRows = [];

for (let variableCount = 1; variableCount <= 6; variableCount += 1) {
  const sheetName = `summary_${variableCount}`;
  const sourceSheet = sourceWorkbook.worksheets.getItem(sheetName);
  const values = sourceSheet.getUsedRange(true).values;
  const headers = values[0];
  const rows = values
    .slice(1)
    .filter((row) => row.some((value) => value !== null && value !== undefined && value !== ""))
    .map((row) => byHeader(headers, row));

  const eligibleRows = rows.filter(isOkModel);
  eligibleRows.sort((a, b) => numeric(b.Gini_penalized) - numeric(a.Gini_penalized));

  if (eligibleRows.length === 0) {
    selectedRows.push([
      variableCount,
      "Aucun modele conforme aux criteres",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "KO",
    ]);
    detailRows.push([
      variableCount,
      "Aucun modele conforme",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
    ]);
    continue;
  }

  const best = eligibleRows[0];
  selectedRows.push([
    variableCount,
    simplifyFormula(best.formula),
    best.Gini,
    best.Gini_test,
    best.Gini_OOT,
    best.Gini_folds_mean,
    best.Gini_penalized,
    best.Recall_penalized,
    best.F1_penalized,
    best.PR_AUC_penalized,
    "OK",
  ]);

  detailRows.push([
    variableCount,
    best.formula,
    best.global_signif_flag,
    best.variables_signif_flag,
    best.modalities_signif_flag,
    best.vif_flag,
    best.Gini_penalized,
    best.Gini_folds_mean,
    best.Recall_penalized,
    best.F1_penalized,
    best.PR_AUC_penalized,
    best.all_checks_OK,
  ]);
}

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Best models");
sheet.showGridLines = false;

sheet.getRange("A1:K1").merge();
sheet.getRange("A1").values = [["BEST LOGISTIC MODELS - SYNTHESIS"]];
sheet.getRange("A1").format = {
  fill: "#FFFFFF",
  font: { bold: true, color: RED, size: 18 },
  horizontalAlignment: "left",
  verticalAlignment: "middle",
};

sheet.getRange("A2:K2").merge();
sheet.getRange("A2").values = [["Selection du meilleur modele par nombre de variables: checks OK puis Gini penalized maximal."]];
sheet.getRange("A2").format = {
  font: { color: "#555555", size: 10 },
  horizontalAlignment: "left",
};

sheet.getRange("A3:D3").merge();
sheet.getRange("A3").values = [[""]];
sheet.getRange("A3:D3").format = { fill: RED };
sheet.getRange("A3:D3").format.rowHeightPx = 6;

const headers = [[
  "Nombre de variables",
  "Model formula",
  "Gini Train",
  "Gini Test",
  "Gini OOT",
  "Gini mean fold",
  "Gini penalized",
  "Recall penalized",
  "F1 penalized",
  "PR AUC penalized",
  "Checks",
]];

sheet.getRange("A5:K5").values = headers;
sheet.getRangeByIndexes(5, 0, selectedRows.length, headers[0].length).values = selectedRows;

sheet.getRange("A5:K5").format = {
  fill: DARK_RED,
  font: { bold: true, color: "#FFFFFF", size: 11 },
  horizontalAlignment: "center",
  verticalAlignment: "middle",
  wrapText: true,
  borders: { preset: "all", style: "thin", color: "#303030" },
};
sheet.getRange("G5:J5").format = {
  fill: YELLOW,
  font: { bold: true, color: TEXT, size: 11 },
  horizontalAlignment: "center",
  verticalAlignment: "middle",
  wrapText: true,
  borders: { preset: "all", style: "thin", color: "#303030" },
};
sheet.getRange("A5:K11").format.borders = { preset: "all", style: "thin", color: "#303030" };
sheet.getRange("A6:K11").format = {
  font: { color: TEXT, size: 10 },
  verticalAlignment: "middle",
  wrapText: true,
};
sheet.getRange("A6:A11").format = {
  horizontalAlignment: "center",
  font: { color: TEXT, size: 11 },
};
sheet.getRange("B6:B11").format = {
  horizontalAlignment: "left",
  font: { color: TEXT, size: 10 },
  wrapText: true,
};
sheet.getRange("C6:J11").format = {
  numberFormat: "0.00%",
  horizontalAlignment: "center",
  verticalAlignment: "middle",
};
sheet.getRange("K6:K11").format = {
  horizontalAlignment: "center",
  font: { bold: true, color: GREEN },
};

for (let r = 6; r <= 11; r += 1) {
  const fill = r % 2 === 0 ? "#FFFFFF" : LIGHT_GRAY;
  sheet.getRange(`A${r}:K${r}`).format.fill = fill;
  sheet.getRange(`G${r}:J${r}`).format.fill = r % 2 === 0 ? LIGHT_YELLOW : "#F7E8B8";

  if (sheet.getRange(`K${r}`).values[0][0] === "KO") {
    sheet.getRange(`A${r}:K${r}`).format.fill = LIGHT_RED;
    sheet.getRange(`K${r}`).format.font = { bold: true, color: RED };
  }
}

sheet.getRange("A:A").format.columnWidthPx = 115;
sheet.getRange("B:B").format.columnWidthPx = 600;
sheet.getRange("C:F").format.columnWidthPx = 105;
sheet.getRange("G:J").format.columnWidthPx = 125;
sheet.getRange("K:K").format.columnWidthPx = 75;
sheet.getRange("5:5").format.rowHeightPx = 42;
sheet.getRange("6:11").format.rowHeightPx = 52;
sheet.freezePanes.freezeRows(5);

const details = workbook.worksheets.add("Selection details");
details.showGridLines = false;
details.getRange("A1:L1").merge();
details.getRange("A1").values = [["Selection audit trail"]];
details.getRange("A1").format = {
  fill: "#FFFFFF",
  font: { bold: true, color: RED, size: 16 },
};

details.getRange("A3:L3").values = [[
  "Nombre de variables",
  "Raw formula",
  "Global signif",
  "Variables signif",
  "Modalites signif",
  "VIF",
  "Gini penalized",
  "Gini mean fold",
  "Recall penalized",
  "F1 penalized",
  "PR AUC penalized",
  "All checks OK",
]];
details.getRangeByIndexes(3, 0, detailRows.length, 12).values = detailRows;
details.getRange("A3:L3").format = {
  fill: DARK_RED,
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "middle",
  wrapText: true,
};
details.getRange("A3:L9").format.borders = { preset: "all", style: "thin", color: "#CFCFCF" };
details.getRange("A4:L9").format = {
  font: { color: TEXT, size: 10 },
  verticalAlignment: "middle",
  wrapText: true,
};
details.getRange("G4:K9").format.numberFormat = "0.00%";
details.getRange("A:A").format.columnWidthPx = 115;
details.getRange("B:B").format.columnWidthPx = 620;
details.getRange("C:F").format.columnWidthPx = 105;
details.getRange("G:K").format.columnWidthPx = 120;
details.getRange("L:L").format.columnWidthPx = 100;
details.freezePanes.freezeRows(3);

const check = await workbook.inspect({
  kind: "table",
  range: "Best models!A5:K11",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 11,
});
console.log(check.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(outputDir, { recursive: true });
const preview = await workbook.render({
  sheetName: "Best models",
  autoCrop: "all",
  scale: 1,
  format: "png",
});
await fs.writeFile(path.join(outputDir, "preview_best_models.png"), new Uint8Array(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`Saved ${outputPath}`);
