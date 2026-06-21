import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = "/Users/juniorjumbong/Desktop/personal-website/00_tds/01_PD_credit_scoring";
const inputPath = path.join(projectRoot, ".codex_work", "best_4_variables_scorecard.json");
const outputDir = path.join(projectRoot, "outputs", "model_selection");
const outputPath = path.join(outputDir, "best_4_variables_scorecard.xlsx");

const RED = "#8E1712";
const DARK_RED = "#6F110E";
const LIGHT_RED = "#F3DFDD";
const DARK_GRAY = "#9B9B9B";
const MID_GRAY = "#C8C8C8";
const LIGHT_GRAY = "#EDEDED";
const WHITE = "#FFFFFF";
const TEXT = "#222222";

const raw = await fs.readFile(inputPath, "utf8");
const payload = JSON.parse(raw);
const rows = payload.scorecard_table;
const sortedRows = [
  ...rows
    .filter((row) => row.variable !== "Intercept")
    .sort((a, b) => {
      const variableOrder = Number(a.variable_number) - Number(b.variable_number);

      if (variableOrder !== 0) {
        return variableOrder;
      }

      return Number(a.coefficient) - Number(b.coefficient);
    }),
  ...rows.filter((row) => row.variable === "Intercept"),
];
const maxCoefficientByVariable = {};

for (const row of rows) {
  if (row.variable === "Intercept") {
    continue;
  }

  const coefficient = Number(row.coefficient);
  maxCoefficientByVariable[row.variable] = Math.max(
    maxCoefficientByVariable[row.variable] ?? Number.NEGATIVE_INFINITY,
    coefficient,
  );
}

const scoreDenominator = Object.values(maxCoefficientByVariable)
  .reduce((sum, value) => sum + value, 0);

const variableLabels = {
  loan_int_rate_dis: "loan_int_rate",
  loan_percent_income_dis: "loan_percent_income",
  cb_person_default_on_file: "cb_person_default_on_file",
  home_ownership_3: "home_ownership_3",
};

const modalityLabels = {
  loan_int_rate_dis: {
    "1": "(12.21, 21.825]",
    "2": "(9.91, 12.21]",
    "3": "(5.419, 9.91]",
  },
  loan_percent_income_dis: {
    "1": "(0.2, 0.44]",
    "2": "(0.11, 0.2]",
    "3": "(-0.001, 0.11]",
  },
  cb_person_default_on_file: {
    N: "N",
    Y: "Y",
  },
  home_ownership_3: {
    OWN: "OWN",
    MORTGAGE: "MORTGAGE",
    OTHER_RENT: "OTHER_RENT",
  },
};

function variableDisplay(variable) {
  return variableLabels[variable] ?? variable;
}

function modalityDisplay(row) {
  return modalityLabels[row.variable]?.[String(row.modality)] ?? row.modality;
}

function coefficientName(row) {
  const variable = variableDisplay(row.variable);
  const modality = String(row.modality).replaceAll(/[^A-Za-z0-9]+/g, "_");

  return `Coeff_${variable}_${modality}`;
}

function coefDisplay(row) {
  const coefficient = Number(row.coefficient);

  if (Number.isNaN(coefficient)) {
    return "";
  }

  if (row.coef_name === "REFERENCE") {
    return "0";
  }

  if (row.coef_name === "Intercept") {
    return coefficient;
  }

  return `${coefficient.toFixed(12)} (${coefficientName(row)})`;
}

function scoreDisplay(row) {
  if (row.variable === "Intercept") {
    return "";
  }

  const coefficient = Number(row.coefficient);
  const maxCoefficient = maxCoefficientByVariable[row.variable];
  const score = 1000 * Math.abs(coefficient - maxCoefficient) / scoreDenominator;

  return Number(score.toFixed(2));
}

function rowHeightForVariable(variable) {
  if (variable.length > 28) {
    return 54;
  }

  return 42;
}

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Score computation");
sheet.showGridLines = false;

sheet.getRange("A1:E1").merge();
sheet.getRange("A1").values = [["The score is computed as follows based on 4 variables:"]];
sheet.getRange("A1").format = {
  fill: WHITE,
  font: { bold: true, color: TEXT, size: 16 },
  horizontalAlignment: "left",
  verticalAlignment: "middle",
};

sheet.getRange("A3:E3").values = [["#", "Variable", "Modality", "Coeff", "Score"]];
sheet.getRange("A3:E3").format = {
  fill: RED,
  font: { bold: true, color: WHITE, size: 12 },
  horizontalAlignment: "center",
  verticalAlignment: "middle",
  borders: { preset: "all", style: "thin", color: WHITE },
};

const tableRows = sortedRows.map((row) => [
  row.variable_number,
  variableDisplay(row.variable),
  modalityDisplay(row),
  coefDisplay(row),
  scoreDisplay(row),
]);

const firstDataRow = 4;
const lastDataRow = firstDataRow + tableRows.length - 1;
sheet.getRangeByIndexes(firstDataRow - 1, 0, tableRows.length, 5).values = tableRows;

sheet.getRange(`A${firstDataRow}:E${lastDataRow}`).format = {
  font: { color: TEXT, size: 11 },
  verticalAlignment: "middle",
  wrapText: true,
  borders: { preset: "all", style: "thin", color: WHITE },
};

for (let rowIndex = firstDataRow; rowIndex <= lastDataRow; rowIndex += 1) {
  const values = sheet.getRange(`A${rowIndex}:E${rowIndex}`).values[0];
  const variable = String(values[1] ?? "");
  const isIntercept = variable === "Intercept";
  const fill = rowIndex % 2 === 0 ? LIGHT_GRAY : MID_GRAY;

  sheet.getRange(`A${rowIndex}:E${rowIndex}`).format.fill = isIntercept ? RED : fill;
  sheet.getRange(`A${rowIndex}:E${rowIndex}`).format.rowHeightPx = isIntercept
    ? 34
    : rowHeightForVariable(variable);

  sheet.getRange(`A${rowIndex}:A${rowIndex}`).format = {
    fill: isIntercept ? RED : DARK_RED,
    font: { bold: true, color: WHITE, size: 12 },
    horizontalAlignment: "center",
    verticalAlignment: "middle",
    borders: { preset: "all", style: "thin", color: WHITE },
  };

  sheet.getRange(`B${rowIndex}:B${rowIndex}`).format = {
    fill: isIntercept ? RED : fill,
    font: { bold: true, color: isIntercept ? WHITE : TEXT, size: 11 },
    horizontalAlignment: "center",
    verticalAlignment: "middle",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: WHITE },
  };

  sheet.getRange(`C${rowIndex}:E${rowIndex}`).format = {
    fill: isIntercept ? MID_GRAY : fill,
    font: { bold: false, color: TEXT, size: 11 },
    horizontalAlignment: "center",
    verticalAlignment: "middle",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: WHITE },
  };

  if (isIntercept) {
    sheet.getRange(`A${rowIndex}:C${rowIndex}`).merge();
    sheet.getRange(`A${rowIndex}`).values = [["Intercept"]];
    sheet.getRange(`A${rowIndex}`).format = {
      fill: RED,
      font: { bold: true, color: WHITE, size: 12 },
      horizontalAlignment: "center",
      verticalAlignment: "middle",
      borders: { preset: "all", style: "thin", color: WHITE },
    };
    sheet.getRange(`D${rowIndex}`).format = {
      fill: MID_GRAY,
      font: { bold: false, color: TEXT, size: 11 },
      horizontalAlignment: "center",
      verticalAlignment: "middle",
      borders: { preset: "all", style: "thin", color: WHITE },
    };
    sheet.getRange(`E${rowIndex}`).format = {
      fill: MID_GRAY,
      font: { bold: false, color: TEXT, size: 11 },
      horizontalAlignment: "center",
      verticalAlignment: "middle",
      borders: { preset: "all", style: "thin", color: WHITE },
    };
  }
}

let groupStart = firstDataRow;

while (groupStart <= lastDataRow) {
  const variable = sheet.getRange(`B${groupStart}`).values[0][0];

  if (variable === "Intercept") {
    groupStart += 1;
    continue;
  }

  let groupEnd = groupStart;

  while (
    groupEnd + 1 <= lastDataRow
    && sheet.getRange(`B${groupEnd + 1}`).values[0][0] === variable
  ) {
    groupEnd += 1;
  }

  if (groupEnd > groupStart) {
    sheet.getRange(`A${groupStart}:A${groupEnd}`).merge();
    sheet.getRange(`B${groupStart}:B${groupEnd}`).merge();
  }

  const variableNumber = sheet.getRange(`A${groupStart}`).values[0][0];
  const variableFill = groupStart % 2 === 0 ? LIGHT_GRAY : MID_GRAY;
  sheet.getRange(`A${groupStart}`).values = [[variableNumber]];
  sheet.getRange(`B${groupStart}`).values = [[variable]];
  sheet.getRange(`A${groupStart}`).format = {
    fill: DARK_RED,
    font: { bold: true, color: WHITE, size: 12 },
    horizontalAlignment: "center",
    verticalAlignment: "middle",
    borders: { preset: "all", style: "thin", color: WHITE },
  };
  sheet.getRange(`B${groupStart}`).format = {
    fill: variableFill,
    font: { bold: true, color: TEXT, size: 11 },
    horizontalAlignment: "center",
    verticalAlignment: "middle",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: WHITE },
  };

  groupStart = groupEnd + 1;
}

sheet.getRange("A:A").format.columnWidthPx = 78;
sheet.getRange("B:B").format.columnWidthPx = 270;
sheet.getRange("C:C").format.columnWidthPx = 370;
sheet.getRange("D:D").format.columnWidthPx = 360;
sheet.getRange("E:E").format.columnWidthPx = 130;
sheet.getRange(`E${firstDataRow}:E${lastDataRow}`).format.numberFormat = "0.00";
sheet.getRange("3:3").format.rowHeightPx = 34;
sheet.freezePanes.freezeRows(3);

const meta = workbook.worksheets.add("Model metrics");
meta.showGridLines = false;
meta.getRange("A1:D1").merge();
meta.getRange("A1").values = [["Selected 4-variable model"]];
meta.getRange("A1").format = {
  fill: WHITE,
  font: { bold: true, color: RED, size: 16 },
};

meta.getRange("A3:B3").values = [["Field", "Value"]];
meta.getRange("A3:B3").format = {
  fill: RED,
  font: { bold: true, color: WHITE },
  horizontalAlignment: "center",
  borders: { preset: "all", style: "thin", color: WHITE },
};

const metrics = payload.metrics ?? {};
const metaRows = [
  ["Formula", payload.formula],
  ["Target", payload.target],
  ["Variables", payload.variables.join(", ")],
  ["Gini train", metrics.Gini_train],
  ["Gini test", metrics.Gini_test],
  ["Gini OOT", metrics.Gini_OOT],
  ["Gini penalized", metrics.Gini_penalized],
  ["Recall penalized", metrics.Recall_penalized],
  ["F1 penalized", metrics.F1_penalized],
  ["PR AUC penalized", metrics.PR_AUC_penalized],
];

meta.getRangeByIndexes(3, 0, metaRows.length, 2).values = metaRows;
meta.getRange(`A4:B${3 + metaRows.length}`).format = {
  font: { color: TEXT, size: 10 },
  verticalAlignment: "middle",
  wrapText: true,
  borders: { preset: "all", style: "thin", color: "#D0D0D0" },
};
meta.getRange("A4:A13").format = {
  fill: LIGHT_RED,
  font: { bold: true, color: TEXT, size: 10 },
};
meta.getRange("B7:B13").format.numberFormat = "0.00%";
meta.getRange("A:A").format.columnWidthPx = 150;
meta.getRange("B:B").format.columnWidthPx = 720;
meta.freezePanes.freezeRows(3);

const inspect = await workbook.inspect({
  kind: "table",
  range: `Score computation!A1:E${lastDataRow}`,
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 5,
});
console.log(inspect.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(outputDir, { recursive: true });
const preview = await workbook.render({
  sheetName: "Score computation",
  autoCrop: "all",
  scale: 1,
  format: "png",
});
await fs.writeFile(path.join(outputDir, "preview_best_4_scorecard.png"), new Uint8Array(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`Saved ${outputPath}`);
